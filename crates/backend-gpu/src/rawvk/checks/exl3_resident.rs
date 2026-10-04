//! EXL3 레이어 스트리밍 디코드 (plans/118 §3-2) — 트렐리스 13GB 상주 +
//! 선형층 vk GEMV + 비선형 CPU. 53.8GB F16 전개 없이 전 모델 디코드.
//!
//! v2 속도: 입력버퍼 재사용(alloc 제거) + 배치 3커널(run_rw 배리어).
//! v3(plans/120 A1): 배치 내 선형별 had_in — suh는 텐서별 입력채널 스케일
//! (W = diag(suh)·H·diag(svh), trellis.rs)이라 공유 had_in은 수치 오염이다.
//! ah/yb 슬롯 3종, sb는 run_rw 배리어(WAR/WAW 커버, plans/104)로 공유.

use crate::rawvk::context::{Pipes, VkBuf, VkCtx};
use half::f16;

/// 레이어별 선형 가중치 — vk 버퍼 3종(trellis, suh, svh).
pub struct VkLinear {
    pub k: usize,
    pub n: usize,
    pub krate: u32,
    pub tre: VkBuf,
    pub suh: VkBuf,
    pub svh: VkBuf,
}

/// 지연 판독 핸들(plans/120 A1, LLM170_VK_DBUF=1) — 제출 후 즉시 반환된
/// 배치의 결과 슬롯. TrellisResident::fetch()가 wait_pending으로 완료를 보장.
pub struct PendingLin {
    /// 결과 yb 슬롯 (0..2).
    pub yb: usize,
    pub n: usize,
}

/// 트렐리스 + 무양자화 가중치 전체를 vk에 상주.
pub struct TrellisResident {
    pub linears: Vec<(String, VkLinear)>,
    /// GPU GDN 상태 유효 플래그(plans/121 F2 스케줄) — 프리필 중 불요한
    /// 층별 상태 업로드 스킵. sync(CPU 다운로드) 후 false 복귀.
    pub gdn_st_valid: Vec<bool>,
    /// 무양자화 노름·스케일러 (CPU f32).
    pub norms: Vec<(String, Vec<f32>)>,
    /// 임베딩 (CPU f32, vocab×hidden).
    pub embed: Vec<f32>,
    pub vocab: usize,
    pub hidden: usize,
    pub n_layers: usize,
    pub ctx: VkCtx,
    p1: Pipes,
    p2: Pipes,
    p3: Pipes,
    p4: Pipes,
    // 스크래치 (재사용 — alloc 폭탄 제거)
    ahb1: VkBuf,
    ahb2: VkBuf,
    ahb3: VkBuf,
    sb: VkBuf,
    yb1: VkBuf,
    yb2: VkBuf,
    yb3: VkBuf,
    xb: VkBuf,
    /// FFN ew 출력(= down의 had_in 입력) — max_n 폭 f16.
    x2b: VkBuf,
    /// T-배치 프리필 스크래치 (plans/121 A1-pp) — 지연 초기화(디코드 전용
    /// 사용 시 미할당). 영속 아레나라 재할당 없음 — TMAX 고정.
    pub(crate) batch: Option<BatchScratch>,
}

impl TrellisResident {
    /// EXL3 디렉터리에서 전 선형을 vk에 업로드.
    pub fn load(dir: &str) -> Result<Self, String> {
        let ar =
            llm170_exl3::StArchive::open(std::path::Path::new(dir)).map_err(|e| e.to_string())?;
        let mut ctx = VkCtx::new()?;
        let p1 = ctx.pipeline_pipes(include_bytes!("../spv/exl3_had_in.spv"), 3, 4)?;
        let p2 = ctx.pipeline_pipes(include_bytes!("../spv/exl3_gemv.spv"), 3, 12)?;
        let p3 = ctx.pipeline_pipes(include_bytes!("../spv/exl3_had_out.spv"), 3, 8)?;
        let p4 = ctx.pipeline_pipes(include_bytes!("../spv/exl3_ffn_ew.spv"), 3, 4)?;

        let mut linears = Vec::new();
        let mut norms = Vec::new();
        let mut embed = Vec::new();
        let mut vocab = 0;
        let mut hidden = 0;
        let mut n_layers = 0;

        for (name, e) in ar.entries() {
            if name.contains("model.visual.") {
                continue;
            }
            // mtp.*는 plans/121 A2부터 상주(MTP 드래프트용 — fc/1층/norm).
            if name.ends_with(".trellis") {
                let base = &name[..name.len() - 8];
                let (kt, nt, tw) = (e.shape[0], e.shape[1], e.shape[2]);
                if tw % 16 != 0 {
                    continue;
                }
                let (k, n) = ((kt * 16) as usize, (nt * 16) as usize);
                let tre_bytes = e.nbytes() as usize;
                let treb = ctx.alloc(tre_bytes)?;
                let suhb = ctx.alloc(k * 2)?;
                let svhb = ctx.alloc(n * 2)?;
                unsafe {
                    ar.read_into(name, treb.ptr, tre_bytes)
                        .map_err(|e| e.to_string())?;
                    ar.read_into(&format!("{base}.suh"), suhb.ptr, k * 2)
                        .map_err(|e| e.to_string())?;
                    ar.read_into(&format!("{base}.svh"), svhb.ptr, n * 2)
                        .map_err(|e| e.to_string())?;
                }
                linears.push((
                    base.to_string(),
                    VkLinear {
                        k,
                        n,
                        krate: (tw / 16) as u32,
                        tre: treb,
                        suh: suhb,
                        svh: svhb,
                    },
                ));
            } else if name.ends_with(".weight")
                || name.ends_with("A_log")
                || name.ends_with("dt_bias")
            {
                let bytes = ar.read(name).map_err(|e| e.to_string())?;
                let shape = &e.shape;
                let numel: usize = shape.iter().product::<u64>() as usize;
                match e.dtype {
                    llm170_exl3::StDtype::Bf16 => {
                        let mut v = Vec::with_capacity(numel);
                        let (chunks, _) = bytes.as_chunks::<2>();
                        for c in chunks {
                            let f = f32::from_bits(((c[1] as u32) << 24) | ((c[0] as u32) << 16));
                            v.push(f);
                        }
                        if name.ends_with("embed_tokens.weight") {
                            embed = v;
                            vocab = shape[0] as usize;
                            hidden = shape[1] as usize;
                        } else if name.ends_with("A_log") {
                            norms.push((name.clone(), v));
                        } else {
                            let is_residual = name.ends_with("layernorm.weight")
                                || name.ends_with("q_norm.weight")
                                || name.ends_with("k_norm.weight")
                                || name.ends_with("language_model.norm.weight")
                                // mtp.* norm도 동일 규약(w-1 저장) — 접미 불일치
                                // 3종(plans/121 A2: 누락 시 드래프트 corr 0.17全멸).
                                || name == "mtp.norm.weight"
                                || name.starts_with("mtp.pre_fc_norm_");
                            if is_residual {
                                for f in v.iter_mut() {
                                    *f += 1.0;
                                }
                            }
                            norms.push((name.clone(), v));
                        }
                    }
                    llm170_exl3::StDtype::F16 => {
                        let mut v = Vec::with_capacity(numel);
                        let (chunks, _) = bytes.as_chunks::<2>();
                        for c in chunks {
                            v.push(f16::from_le_bytes([c[0], c[1]]).to_f32());
                        }
                        if name.ends_with("embed_tokens.weight") {
                            embed = v;
                            vocab = shape[0] as usize;
                            hidden = shape[1] as usize;
                        } else {
                            norms.push((name.clone(), v));
                        }
                    }
                    _ => {}
                }
                if let Some(idx) = name.find(".layers.") {
                    let rest = &name[idx + 8..];
                    if let Some(dot) = rest.find('.')
                        && let Ok(l) = rest[..dot].parse::<usize>()
                    {
                        n_layers = n_layers.max(l + 1);
                    }
                }
            } else if name.ends_with(".bias") {
                let bytes = ar.read(name).map_err(|e| e.to_string())?;
                let shape = &e.shape;
                let numel: usize = shape.iter().product::<u64>() as usize;
                let mut v = Vec::with_capacity(numel);
                let (chunks, _) = bytes.as_chunks::<2>();
                for c in chunks {
                    v.push(f16::from_le_bytes([c[0], c[1]]).to_f32());
                }
                norms.push((name.clone(), v));
            }
        }

        // 스크래치 버퍼 — 재사용 (linear 호출당 alloc 폭탄 제거)
        let max_k = linears.iter().map(|(_, l)| l.k).max().unwrap_or(5120);
        let max_n = linears.iter().map(|(_, l)| l.n).max().unwrap_or(17408);
        // plans/120 A1: k-분할 4→16 — 병렬성 증가(벤치 72→87GB/s).
        // had_out 환원 분해 변화 = 규칙 10a(환원 순서, 동일 정밀도).
        let nseg = 16u32;
        let ahb1 = ctx.alloc(max_k * 2)?;
        let ahb2 = ctx.alloc(max_k * 2)?;
        let ahb3 = ctx.alloc(max_k * 2)?;
        let sb = ctx.alloc(max_n * 4 * nseg as usize)?;
        let yb1 = ctx.alloc(max_n * 4)?;
        let yb2 = ctx.alloc(max_n * 4)?;
        let yb3 = ctx.alloc(max_n * 4)?;
        let xb = ctx.alloc(max_k * 2)?;
        let x2b = ctx.alloc(max_n * 2)?;

        Ok(Self {
            linears,
            gdn_st_valid: vec![false; n_layers - n_layers / 4],
            norms,
            embed,
            vocab,
            hidden,
            n_layers,
            ctx,
            p1,
            p2,
            p3,
            ahb1,
            ahb2,
            ahb3,
            sb,
            yb1,
            yb2,
            yb3,
            xb,
            x2b,
            p4,
            batch: None,
        })
    }

    pub(crate) fn find_linear(&self, key: &str) -> Result<usize, String> {
        self.linears
            .iter()
            .position(|(k, _)| k == key)
            .ok_or_else(|| format!("linear not found: {key}"))
    }

    /// x를 f16으로 변환해 재사용 xb에 직접 기록.
    /// plans/120 A1: 임시 Vec+이중 복사 제거, 8청크 기록으로 벡터화 유도.
    /// from_f32 요소별 호출 유지 — 비트동일.
    fn upload_x(&mut self, x: &[f32]) -> Result<(), String> {
        let k = x.len();
        // SAFETY: xb는 max_k*2 바이트(k ≤ max_k, 호출자가 linear의 k와
        // x.len()을 일치시킴). u16 기록은 LE 호스트에서 to_le_bytes와 동일.
        let dst = unsafe { std::slice::from_raw_parts_mut(self.xb.ptr as *mut u16, k) };
        let (chunks, rem) = x.as_chunks::<8>();
        for (i, c) in chunks.iter().enumerate() {
            let b: [u16; 8] = std::array::from_fn(|j| f16::from_f32(c[j]).to_bits());
            dst[i * 8..i * 8 + 8].copy_from_slice(&b);
        }
        let base = chunks.len() * 8;
        for (j, &v) in rem.iter().enumerate() {
            dst[base + j] = f16::from_f32(v).to_bits();
        }
        Ok(())
    }

    /// 배치: 선형별 had_in→gemv→had_out 체인, 단일 제출·단일 동기.
    /// suh는 텐서별 스케일이라 had_in을 공유하지 않고 슬롯별 ah에서 변환.
    /// sb 공유는 run_rw 배리어가 WAR/WAW를 커버(context plans/104 판정식).
    /// 선형 1개의 3커널 체인 발행 (배치 내부 — begin_batch 후에만 호출).
    /// x_src: had_in 입력 버퍼(xb 또는 FFN ew 출력 x2b).
    fn chain_one(
        &mut self,
        x_src: ash::vk::Buffer,
        li: usize,
        ah: ash::vk::Buffer,
        yb: ash::vk::Buffer,
    ) -> Result<(), String> {
        let nseg = 16u32; // load의 sb 할당과 일치(k-분할)
        let l = &self.linears[li].1;
        let (k, n, krate) = (l.k, l.n, l.krate);
        let (suh_b, tre_b, svh_b) = (l.suh.buf, l.tre.buf, l.svh.buf);
        let sb_b = self.sb.buf;

        // had_in: x_src × suh_i → ah_i (텐서별 suh — 공유 금지)
        let ds = self.ctx.fresh_ds_for(&self.p1, 3)?;
        self.ctx.bind_bufs(ds, &[x_src, suh_b, ah]);
        let push = (k as u32 / 128).to_le_bytes().to_vec();
        // ts 표 라벨은 site::tag()(TAG thread-local) — scope은 CUR만 바꾼다.
        crate::rawvk::context::site::set_tag("e3_had_in");
        self.ctx.run_rw(
            self.p1.pl,
            ds,
            self.p1.pipe,
            &push,
            (k / 128) as u32,
            1,
            1,
            &[x_src, suh_b],
            &[ah],
        )?;

        // gemv: ah_i × tre → sb
        let ds2 = self.ctx.fresh_ds_for(&self.p2, 3)?;
        self.ctx.bind_bufs(ds2, &[ah, tre_b, sb_b]);
        let push2: Vec<u8> = [(k / 16) as u32, (n / 16) as u32, krate]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        crate::rawvk::context::site::set_tag("e3_gemv");
        self.ctx.run_rw(
            self.p2.pl,
            ds2,
            self.p2.pipe,
            &push2,
            ((n / 16) as u32).div_ceil(8),
            nseg,
            1,
            &[ah, tre_b],
            &[sb_b],
        )?;

        // had_out: sb × svh → yb_slot
        let ds3 = self.ctx.fresh_ds_for(&self.p3, 3)?;
        self.ctx.bind_bufs(ds3, &[sb_b, svh_b, yb]);
        let push3: Vec<u8> = [(n as u32 / 128), nseg]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        crate::rawvk::context::site::set_tag("e3_had_out");
        self.ctx.run_rw(
            self.p3.pl,
            ds3,
            self.p3.pipe,
            &push3,
            (n / 128) as u32,
            1,
            1,
            &[sb_b, svh_b],
            &[yb],
        )?;
        Ok(())
    }

    /// 배치: 선형별 had_in→gemv→had_out 체인, 단일 제출·단일 동기.
    /// suh는 텐서별 스케일이라 had_in을 공유하지 않고 슬롯별 ah에서 변환.
    /// sb 공유는 run_rw 배리어가 WAR/WAW를 커버(context plans/104 판정식).
    fn batch_chain(
        &mut self,
        inputs: &[usize],  // linear 인덱스 목록 (≤3)
        outputs: &[usize], // 호환 — 결과는 슬롯 순서대로 yb1..yb3
    ) -> Result<(), String> {
        let _ = outputs;

        self.ctx.begin_batch()?;
        for (slot, &li) in inputs.iter().enumerate() {
            let ah = [self.ahb1.buf, self.ahb2.buf, self.ahb3.buf][slot];
            let yb = [self.yb1.buf, self.yb2.buf, self.yb3.buf][slot];
            self.chain_one(self.xb.buf, li, ah, yb)?;
        }
        self.ctx.end_batch_wait()?;
        Ok(())
    }

    /// FFN 3선형 + GPU ew 단일 배치 (plans/120 A1):
    /// gate/up GEMV → ew(silu(g)·u → f16 x2b) → down GEMV — 동기 1회.
    /// 게이트/업 판독·CPU 활성화·업로드가 사라진다. ew의 GPU exp는
    /// CPU libm exp와 근사차(10a, 동일 f32 클래스) — vk-check ew 미러가
    /// 허용치를 검증한다.
    pub fn ffn_triple(
        &mut self,
        key_g: &str,
        key_u: &str,
        key_d: &str,
        x: &[f32],
    ) -> Result<Vec<f32>, String> {
        let ig = self.find_linear(key_g)?;
        let iu = self.find_linear(key_u)?;
        let id = self.find_linear(key_d)?;
        let kg = self.linears[ig].1.k;
        let ng = self.linears[ig].1.n;
        let nd = self.linears[id].1.n;
        if x.len() != kg || self.linears[iu].1.k != kg || self.linears[id].1.k != ng {
            return Err(format!("{key_g}/{key_u}/{key_d}: FFN 차원 불일치"));
        }
        self.upload_x(x)?;

        self.ctx.begin_batch()?;
        self.chain_one(self.xb.buf, ig, self.ahb1.buf, self.yb1.buf)?;
        self.chain_one(self.xb.buf, iu, self.ahb2.buf, self.yb2.buf)?;
        // ew: yb1(g) × yb2(u) → x2b (f16 쌍팩)
        {
            let ds4 = self.ctx.fresh_ds_for(&self.p4, 3)?;
            self.ctx
                .bind_bufs(ds4, &[self.yb1.buf, self.yb2.buf, self.x2b.buf]);
            let push4 = (ng as u32).to_le_bytes().to_vec();
            crate::rawvk::context::site::set_tag("e3_ffn_ew");
            self.ctx.run_rw(
                self.p4.pl,
                ds4,
                self.p4.pipe,
                &push4,
                ng.div_ceil(512) as u32,
                1,
                1,
                &[self.yb1.buf, self.yb2.buf],
                &[self.x2b.buf],
            )?;
        }
        self.chain_one(self.x2b.buf, id, self.ahb3.buf, self.yb3.buf)?;
        self.ctx.end_batch_wait()?;
        self.ctx.wait_pending()?; // DBUF 비동기 잔여 배출(비동기 아님 시 무연산)

        let mut y = vec![0f32; nd];
        // SAFETY: end_batch_wait 후 yb3 판독 — nd ≤ max_n(할당 상한).
        unsafe {
            std::ptr::copy_nonoverlapping(self.yb3.ptr as *const f32, y.as_mut_ptr(), nd);
        }
        Ok(y)
    }

    /// 비동기 선형 쌍: 제출만 하고 즉시 반환. 호출자는 qkvz 출력과 무독립한
    /// CPU 일(예: alpha/beta dot)을 수행한 뒤 fetch로 합류한다. DBUF 부재
    /// 시 end_batch_wait가 이미 대기 — fetch는 판독만 한다(의미 동일).
    pub fn linear_pair_deferred(
        &mut self,
        key1: &str,
        key2: &str,
        x: &[f32],
    ) -> Result<(PendingLin, PendingLin), String> {
        let i1 = self.find_linear(key1)?;
        let i2 = self.find_linear(key2)?;
        let (n1, n2) = (self.linears[i1].1.n, self.linears[i2].1.n);
        let k = self.linears[i1].1.k;
        if x.len() != k || self.linears[i2].1.k != k {
            return Err(format!("{key1}/{key2}: input dim mismatch"));
        }
        self.upload_x(x)?;
        self.batch_chain(&[i1, i2], &[0, 1])?;
        Ok((PendingLin { yb: 0, n: n1 }, PendingLin { yb: 1, n: n2 }))
    }

    /// 지연 결과 판독 — wait_pending 후 슬롯 버퍼에서 복사.
    pub fn fetch(&mut self, p: PendingLin) -> Result<Vec<f32>, String> {
        self.ctx.wait_pending()?;
        let src = [self.yb1.ptr, self.yb2.ptr, self.yb3.ptr][p.yb];
        let mut y = vec![0f32; p.n];
        // SAFETY: wait_pending 후 매핑 판독 — n ≤ max_n.
        unsafe {
            std::ptr::copy_nonoverlapping(src as *const f32, y.as_mut_ptr(), p.n);
        }
        Ok(y)
    }

    /// 선형 투영: y = x @ W^T. 단일 선형.
    pub fn linear(&mut self, key: &str, x: &[f32]) -> Result<Vec<f32>, String> {
        let idx = self.find_linear(key)?;
        let n = self.linears[idx].1.n;
        if x.len() != self.linears[idx].1.k {
            return Err(format!("{key}: input len mismatch"));
        }

        self.upload_x(x)?;
        self.batch_chain(&[idx], &[0])?;
        self.ctx.wait_pending()?; // DBUF 잔여 배출

        let mut y = vec![0f32; n];
        // SAFETY: yb1 매핑 — batch_chain의 end_batch_wait 후.
        unsafe {
            std::ptr::copy_nonoverlapping(self.yb1.ptr as *const f32, y.as_mut_ptr(), n);
        }
        Ok(y)
    }

    /// 공유 입력 선형 쌍: (y1, y2) = (x @ W1^T, x @ W2^T).
    /// had_in 1회 + gemv/had_out 2회 — 동기 1회, 업로드 1회.
    pub fn linear_pair(
        &mut self,
        key1: &str,
        key2: &str,
        x: &[f32],
    ) -> Result<(Vec<f32>, Vec<f32>), String> {
        let i1 = self.find_linear(key1)?;
        let i2 = self.find_linear(key2)?;
        let (n1, n2) = (self.linears[i1].1.n, self.linears[i2].1.n);
        let k = self.linears[i1].1.k;
        if x.len() != k || self.linears[i2].1.k != k {
            return Err(format!("{key1}/{key2}: input dim mismatch"));
        }

        self.upload_x(x)?;
        self.batch_chain(&[i1, i2], &[0, 1])?;
        self.ctx.wait_pending()?; // DBUF 잔여 배출

        let mut y1 = vec![0f32; n1];
        let mut y2 = vec![0f32; n2];
        // SAFETY: 배치 완료 후 판독.
        unsafe {
            std::ptr::copy_nonoverlapping(self.yb1.ptr as *const f32, y1.as_mut_ptr(), n1);
            std::ptr::copy_nonoverlapping(self.yb2.ptr as *const f32, y2.as_mut_ptr(), n2);
        }
        Ok((y1, y2))
    }

    /// 공유 입력 선형 3중: attention q+k+v용.
    pub fn linear_triple(
        &mut self,
        key1: &str,
        key2: &str,
        key3: &str,
        x: &[f32],
    ) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>), String> {
        let i1 = self.find_linear(key1)?;
        let i2 = self.find_linear(key2)?;
        let i3 = self.find_linear(key3)?;
        let (n1, n2, n3) = (
            self.linears[i1].1.n,
            self.linears[i2].1.n,
            self.linears[i3].1.n,
        );
        let k = self.linears[i1].1.k;
        if x.len() != k || self.linears[i2].1.k != k || self.linears[i3].1.k != k {
            return Err(format!("{key1}/{key2}/{key3}: input dim mismatch"));
        }
        self.upload_x(x)?;
        self.batch_chain(&[i1, i2, i3], &[0, 1, 2])?;
        self.ctx.wait_pending()?; // DBUF 잔여 배출
        let mut y1 = vec![0f32; n1];
        let mut y2 = vec![0f32; n2];
        let mut y3 = vec![0f32; n3];
        // SAFETY: 배치 완료 후 판독.
        unsafe {
            std::ptr::copy_nonoverlapping(self.yb1.ptr as *const f32, y1.as_mut_ptr(), n1);
            std::ptr::copy_nonoverlapping(self.yb2.ptr as *const f32, y2.as_mut_ptr(), n2);
            std::ptr::copy_nonoverlapping(self.yb3.ptr as *const f32, y3.as_mut_ptr(), n3);
        }
        Ok((y1, y2, y3))
    }

    /// 무양자화 노름/스케일러 획득.
    pub fn norm(&self, key: &str) -> Option<&[f32]> {
        self.norms
            .iter()
            .find(|(k, _)| k.ends_with(key))
            .map(|(_, v)| v.as_slice())
    }

    /// 임베딩 행 조회.
    pub fn embed_row(&self, token: u32) -> &[f32] {
        &self.embed[token as usize * self.hidden..(token as usize + 1) * self.hidden]
    }

    /// 선형 키 존재 확인.
    pub fn has_linear(&self, key: &str) -> bool {
        self.linears.iter().any(|(k, _)| k == key)
    }
}

// ── T-배치 프리필 경로 (plans/121 A1-pp) ─────────────────────────────
//
// GEMV(토큰 1개) 대비: 디코드된 가중치 1쌍이 Tt=32행에 FMA 재사용 —
// 연산강도 32배로 87GB/s 메모리 벽을 탈출해 pp를 GEMM 연산량 바닥으로.
// 스크래치는 지연 초기화·영속(ADR-0014) — 디코드 전용 프로세스는 미할당.

/// 배치 행 상한 — 초과는 드라이버(프리필)가 청크 분할.
pub const BATCH_TMAX: usize = 512;

fn n_gdn_bytes() -> usize {
    // gstate 151MB + gring(conv 링, 롤백 정합 — 재생 NaN 원인 2026-10-03)
    48 * 48 * 16384 * 4 + 48 * 3 * 10240 * 4
}

/// T-배치 스크래치 — 슬롯 3종(GEMV ahb/yb 관례와 동일 구조, run_rw
/// 배리어가 WAR/WAW 커버 — plans/104 판정식, sb는 단일 공유).
pub struct BatchScratch {
    /// [TMAX*max_k] f32 캐시 — 스테이징 입력(호스트가 병렬 직접 기록,
    /// had_in_tf32가 직독 — CPU f16 변환 제거, 병목 원장 #3).
    pub(crate) xtb: VkBuf,
    pub(crate) ah: [VkBuf; 3],
    pub(crate) yb: [VkBuf; 3], // [TMAX*max_n] f32 ×3
    pub(crate) sb: VkBuf,      // [TMAX*NSEG*max_n] f32
    /// [TMAX*max_n] f16 캐브아웃 — ew_t 출력(= down의 had_in_t f16 입력).
    x2t: VkBuf,
    p1: Pipes,  // had_in_t (f16 입력 — ew_t 다운 레그)
    p1f: Pipes, // had_in_tf32 (f32 직독)
    p2: Pipes,  // gemm
    p2d: Pipes, // gemm2d(메가융합 3호 — 듀얼입력)
    p3: Pipes,  // had_out_t
    p3d: Pipes, // had_out_td(슬래브 오프셋 환원)
    p4t: Pipes, // ffn_ew_t
    /// GDN 프레임(plans/121 F1) — GPU 상주 비선형 체인(지연 초기화).
    pub(crate) gframe: Option<GdnFrame>,
    /// 어텐션 프레임(plans/121 F2b) — GPU KV 캐시·prep/fwd3(지연 초기화).
    pub(crate) aframe: Option<AttnFrame>,
    /// 원-서브밋 프레임(plans/121) — 잔차 스트림 상주.
    pub(crate) fframe: Option<FFrame>,
}

/// 어텐션 프레임(plans/121 F2b) — KV 캐시 16층 GPU 상주 + 파이프라인.
// ═══ 어텐션 모듈(plans/121 F2b, fwd3) ═══════════════════════════════
// 측정(2026-10-03): fwd3 = warp-shuffle 스코어 + t블록4 k/v 공유(트래픽 ÷4)
// + LDS 트리 소프트맥스(배리어 10회) + dim-열 AV. 산술 5.3e-7,
// T512 8.2ms/층(CPU 38.5 → 4.7배, fwd2 1033ms → 126배).
// 부착 성과 pp512 96.82→109.33 t/s(+13%), corr 0.999999·8/8.
// KV 16층 134MB GPU 상주 + 프리필 말미 kvc 벌크 동기(디코드 정합).
// 폐기: fwd(3패스 비결합), fwd2(FA블록, 배리어 64회 폭주).
pub struct AttnFrame {
    pbuf: VkBuf, // [4] u32 — pos0(재생: 푸시 불변으로 만들기 위한 매개변수 버퍼)
    kkc: VkBuf,  // [16*1024][1024] f32
    vkc: VkBuf,
    qh: VkBuf,   // [TMAX*6144] f32 norm+rope q
    qnws: VkBuf, // [16*256]
    knws: VkBuf, // [16*256]
    pa: Pipes,   // attn_prep
    pf3: Pipes,  // attn_fwd3
}

/// 프레임 버퍼(plans/121 원-서브밋) — 잔차 스트림 GPU 상주.
// ═══ 원-서브밋 프레임(plans/121, 옵트인 LLM170_EXL3_FRAME=1) ══════
// 측정(2026-10-03): T60 68.7 t/s(구경로 54.5 → +26%). **계류: corr 0.944631
// (결정적 drift — L0-55 모듈출력 일치, L56 FFN부터 이탈, 배리어·얼라이어싱
// ·행오프셋 3버그 수정 후에도 잔류)**. VkCtx 배치 깊이 인식(begin_outer/
// end_outer) + norm_resid(잔차+노름 융합, ab 합산 후 norm).
// 주의: "LLM170_FRAME"은 qwen4exp 게이트와 이름 충돌 — EXL3 접두 필수.
pub struct FFrame {
    pub(crate) xbuf: VkBuf,          // [TMAX*5120] f32 잔차
    pub(crate) gsnap: Option<VkBuf>, // GDN 상태 스냅샷(스펙 롤백용 — 지연 할당, 비스펙 151MB 절약)
    pub(crate) zeros: VkBuf,         // [TMAX*5120] 첫 노름용 ab=0
    pub(crate) nw128: VkBuf, // [128*5120] 노름 행(2il=input_ln, 2il+1=post_ln, 127행=output_norm)
    pub(crate) pnr: Pipes,   // e3_norm_resid
    pub(crate) pnrh: Pipes,  // e3_norm_resid_had(융합 — 메가융합 1호)
}

/// GDN 프레임 버퍼+파이프라인(plans/121 F1).
// ═══ GDN 모듈(plans/121 F1/F2) ═══════════════════════════════════════
// 측정(2026-10-03): 4커널(conv/l2perm/scan/gate) 전층 GPU 상주.
// scan v3(FLA 전LDS): T512 9.3ms/디스패치(v1 53.4), corr 0.999998·8/8.
// 부착 성과 pp512 72.55→96.82 t/s(+33.5%). 컴파일러 결함 2종 확증:
// ① private 동적 배열[32]은 유입 상태≠0일 때 오염(9.4e-2) — LDS 쓸 것
// ② 상태 flush는 offset 기반(flush_range_at). 버그 3종: 순열 방향·
// conv 레이어 오프셋·P6 tid/4 매핑(행당 32열만 기록).
pub struct GdnFrame {
    pub(crate) gq: VkBuf,     // [TMAX*2048] f32 L2 norm q
    pub(crate) gk: VkBuf,     // [TMAX*2048] f32 L2 norm k
    pub(crate) gv: VkBuf,     // [TMAX*6144] f32 v lc
    pub(crate) gbg: VkBuf,    // [TMAX*96] f32 beta|g lc
    pub(crate) go: VkBuf,     // [TMAX*6144] f32 o lc
    pub(crate) gqr: VkBuf,    // [TMAX*2048] f32 conv q raw(HF)
    pub(crate) gkr: VkBuf,    // [TMAX*2048] f32 conv k raw(HF)
    pub(crate) gvr: VkBuf,    // [TMAX*6144] f32 conv v raw(HF)
    pub(crate) ab: VkBuf,     // [n_gdn][2*48*5120] f32
    pub(crate) cw: VkBuf,     // [n_gdn][10240*4] f32
    pub(crate) alog: VkBuf,   // [n_gdn*48] f32
    pub(crate) dtb: VkBuf,    // [n_gdn*48] f32
    pub(crate) nw: VkBuf,     // [n_gdn*128] f32
    pub(crate) gring: VkBuf,  // [n_gdn*3*10240] f32
    pub(crate) gstate: VkBuf, // [n_gdn*48*16384] f32
    pub(crate) pgc: Pipes,
    pub(crate) pgl: Pipes,
    pub(crate) pgs: Pipes,
    pub(crate) pgg: Pipes,
}

impl TrellisResident {
    /// GDN 체인 격리 프로브 지원(plans/121 워크플로): 합성 입력으로
    /// layer0 체인(conv→l2perm→scan→gate) 실행 후 gated 반환.
    pub fn gdn_chain_run(
        &mut self,
        t_rows: usize,
        xn: &[f32],
        qkv: &[f32],
        z: &[f32],
    ) -> Result<Vec<f32>, String> {
        self.gdn_frame_init()?;
        let (yb0_b, yb1_b) = {
            let b = self.batch.as_ref().ok_or("batch")?;
            (b.yb[0].buf, b.yb[1].buf)
        };
        {
            let (xptr, qptr, zptr) = {
                let b = self.batch.as_ref().ok_or("batch")?;
                (
                    b.xtb.ptr as *mut f32,
                    b.yb[0].ptr as *mut f32,
                    b.yb[1].ptr as *mut f32,
                )
            };
            unsafe {
                std::ptr::copy_nonoverlapping(xn.as_ptr(), xptr, t_rows * 5120);
                std::ptr::copy_nonoverlapping(qkv.as_ptr(), qptr, t_rows * 10240);
                std::ptr::copy_nonoverlapping(z.as_ptr(), zptr, t_rows * 6144);
            }
        }
        // 상태/ring 제로(layer0 슬라이스)
        {
            let gf = self
                .batch
                .as_ref()
                .and_then(|b| b.gframe.as_ref())
                .ok_or("gframe")?;
            unsafe {
                std::ptr::write_bytes(gf.gstate.ptr, 0, 48 * 16384 * 4);
                std::ptr::write_bytes(gf.gring.ptr, 0, 3 * 10240 * 4);
            }
            self.ctx.flush_range(&gf.gstate, 48 * 16384 * 4);
            self.ctx.flush_range(&gf.gring, 3 * 10240 * 4);
        }
        // 입력 flush
        {
            let b = self.batch.as_ref().ok_or("batch")?;
            self.ctx.flush_range(&b.xtb, t_rows * 5120 * 4);
            self.ctx.flush_range(&b.yb[0], t_rows * 10240 * 4);
            self.ctx.flush_range(&b.yb[1], t_rows * 6144 * 4);
        }
        self.gdn_layer_gpu(0, t_rows, std::ptr::null_mut(), yb0_b, yb1_b)?;
        // gated 판독
        let b = self.batch.as_ref().ok_or("batch")?;
        self.ctx.invalidate_range(&b.xtb, t_rows * 6144 * 4);
        let out =
            unsafe { std::slice::from_raw_parts(b.xtb.ptr as *const f32, t_rows * 6144).to_vec() };
        // 중간 산출(l2perm gbg/conv gqr 선두) 판독 — 단계 격리 진단.
        {
            let gf = self
                .batch
                .as_ref()
                .and_then(|b| b.gframe.as_ref())
                .ok_or("gframe")?;
            self.ctx.invalidate_range(&gf.gbg, t_rows * 96 * 4);
            self.ctx.invalidate_range(&gf.gqr, t_rows * 2048 * 4);
        }
        Ok(out)
    }

    /// 체인 중간 산출 판독(gbg·gqr·gq[L2 q]·go[scan]) — 프로브 진단.
    pub fn gdn_chain_mids(
        &mut self,
        t_rows: usize,
    ) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>), String> {
        let gf = self
            .batch
            .as_ref()
            .and_then(|b| b.gframe.as_ref())
            .ok_or("gframe")?;
        let bg =
            unsafe { std::slice::from_raw_parts(gf.gbg.ptr as *const f32, t_rows * 96).to_vec() };
        let gqr =
            unsafe { std::slice::from_raw_parts(gf.gqr.ptr as *const f32, t_rows * 2048).to_vec() };
        let _ = &gqr;
        self.ctx.invalidate_range(&gf.gq, t_rows * 2048 * 4);
        self.ctx.invalidate_range(&gf.gk, t_rows * 2048 * 4);
        self.ctx.invalidate_range(&gf.gv, t_rows * 6144 * 4);
        let gq =
            unsafe { std::slice::from_raw_parts(gf.gq.ptr as *const f32, t_rows * 2048).to_vec() };
        let gk =
            unsafe { std::slice::from_raw_parts(gf.gk.ptr as *const f32, t_rows * 2048).to_vec() };
        let gv =
            unsafe { std::slice::from_raw_parts(gf.gv.ptr as *const f32, t_rows * 6144).to_vec() };
        self.ctx.invalidate_range(&gf.go, t_rows * 6144 * 4);
        let go =
            unsafe { std::slice::from_raw_parts(gf.go.ptr as *const f32, t_rows * 6144).to_vec() };
        Ok((bg, gq, gk, gv, go))
    }

    /// layer0 체인 상수 판독(cw/ab/alog/dtb/nw) — 프로브 미러용.
    pub fn gdn_chain_consts(
        &mut self,
    ) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>), String> {
        self.gdn_frame_init()?;
        let gf = self
            .batch
            .as_ref()
            .and_then(|b| b.gframe.as_ref())
            .ok_or("gframe")?;
        // 호스트 기록 상수는 그대로 판독(배치 후 GPU 미기입).
        unsafe {
            let cw = std::slice::from_raw_parts(gf.cw.ptr as *const f32, 48 * 10240 * 4).to_vec();
            let ab =
                std::slice::from_raw_parts(gf.ab.ptr as *const f32, 48 * 2 * 48 * 5120).to_vec();
            let alog = std::slice::from_raw_parts(gf.alog.ptr as *const f32, 48 * 48).to_vec();
            let dtb = std::slice::from_raw_parts(gf.dtb.ptr as *const f32, 48 * 48).to_vec();
            let nw = std::slice::from_raw_parts(gf.nw.ptr as *const f32, 48 * 128).to_vec();
            Ok((cw, ab, alog, dtb, nw))
        }
    }

    /// GPU GDN 상태 유효 플래그 전체 무효화 — fresh 시퀀스(슬롯 교체·기준
    /// 재생) 프리필 전 호출. GPU 상태가 다른 시퀀스의 잔류일 수 있어 강제 재업로드.
    pub fn gdn_st_invalidate_all(&mut self) {
        for v in self.gdn_st_valid.iter_mut() {
            *v = false;
        }
    }

    /// GPU 프레임(GDN/어텐션) 활성 여부 — 벌크 sync 가드(T≤8 CPU 경로 보호).
    pub fn gpu_frames_active(&self) -> bool {
        self.batch
            .as_ref()
            .is_some_and(|b| b.gframe.is_some() || b.aframe.is_some())
    }

    /// 배치 스크래치 지연 초기화.
    fn ensure_batch(&mut self) -> Result<(), String> {
        if self.batch.is_some() {
            return Ok(());
        }
        if llm170_diag::dump::opts().key("exl3_lindbg") {
            self.ctx.dump_mem_types();
        }
        let p1 = self
            .ctx
            .pipeline_pipes(include_bytes!("../spv/exl3_had_in_t.spv"), 3, 8)?;
        let p1f = self
            .ctx
            .pipeline_pipes(include_bytes!("../spv/exl3_had_in_tf32.spv"), 3, 8)?;
        let p2 = self
            .ctx
            .pipeline_pipes(include_bytes!("../spv/exl3_gemm2.spv"), 3, 16)?;
        let p2d = self
            .ctx
            .pipeline_pipes(include_bytes!("../spv/exl3_gemm2d.spv"), 5, 20)?;
        let p3d = self
            .ctx
            .pipeline_pipes(include_bytes!("../spv/exl3_had_out_td.spv"), 3, 20)?;
        let p3 = self
            .ctx
            .pipeline_pipes(include_bytes!("../spv/exl3_had_out_t.spv"), 3, 12)?;
        let p4t = self
            .ctx
            .pipeline_pipes(include_bytes!("../spv/exl3_ffn_ew_t.spv"), 3, 4)?;
        let max_k = self.linears.iter().map(|(_, l)| l.k).max().unwrap_or(5120);
        let max_n = self.linears.iter().map(|(_, l)| l.n).max().unwrap_or(17408);
        // CPU가 직접 읽/쓰는 버퍼(xtb 스테이징·yb 판독)는 호스트 RAM(캐시됨) —
        // APU 커브아웃 매핑 판독은 무캐시로 T×n MB급 판독이 ~300MB/s에
        // 걸려 프리필 병목이었다(2026-10-03 계측: lin_gu 34ms/층 중 대부분).
        // ah/sb/x2t는 GPU 전용 — 커브아웃 유지.
        let xtb = self.ctx.alloc_host_cached(BATCH_TMAX * max_k * 4)?;
        let ah1 = self.ctx.alloc(BATCH_TMAX * max_k * 2)?;
        let ah2 = self.ctx.alloc(BATCH_TMAX * max_k * 2)?;
        let ah3 = self.ctx.alloc(BATCH_TMAX * max_k * 2)?;
        let y1 = self.ctx.alloc_host_cached(BATCH_TMAX * max_n * 4)?;
        let y2 = self.ctx.alloc_host_cached(BATCH_TMAX * max_n * 4)?;
        let y3 = self.ctx.alloc_host_cached(BATCH_TMAX * max_n * 4)?;
        let sb = self.ctx.alloc(BATCH_TMAX * max_n * 4)?;
        let x2t = self.ctx.alloc(BATCH_TMAX * max_n * 2)?;
        self.batch = Some(BatchScratch {
            aframe: None,
            fframe: None,
            xtb,
            ah: [ah1, ah2, ah3],
            yb: [y1, y2, y3],
            sb,
            x2t,
            p1,
            p1f,
            p2,
            p2d,
            p3d,
            p3,
            p4t,
            gframe: None,
        });
        Ok(())
    }

    /// 배치 선형 1개 체인(had_in→gemm→had_out_t) — begin_batch 내부 전용.
    /// f32_in=true: x_src를 f32 [T][k]로 직독(had_in_tf32) — 스테이징 경로.
    /// false: f16 쌍팩(ew_t 출력 → down 레그).
    /// had_in 생략 체인(gemm2+had_out만) — norm_resid_had가 ah를 이미 기록한
    /// 소비용(plans/121 메가융합 1호). ah는 호출자가 지정(ah[slot]).
    fn chain_gemmonly(
        &mut self,
        li: usize,
        t_rows: u32,
        ah: ash::vk::Buffer,
        yb: ash::vk::Buffer,
    ) -> Result<(), String> {
        let l = &self.linears[li].1;
        let (k, n, krate) = (l.k, l.n, l.krate);
        let (tre_b, svh_b) = (l.tre.buf, l.svh.buf);
        let sb_b = self.batch.as_ref().ok_or("batch scratch 미초기화")?.sb.buf;
        {
            let b = self.batch.as_ref().ok_or("batch scratch")?;
            let ktiles = (k / 16) as u32;
            let ntiles = (n / 16) as u32;
            let ds = self.ctx.fresh_ds_for(&b.p2, 3)?;
            self.ctx.bind_bufs(ds, &[ah, tre_b, sb_b]);
            let push: Vec<u8> = [ktiles, ntiles, krate, t_rows]
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect();
            crate::rawvk::context::site::set_tag("e3_gemm2");
            self.ctx.run_rw(
                b.p2.pl,
                ds,
                b.p2.pipe,
                &push,
                ((n / 64) as u32).max(1),
                t_rows.div_ceil(128),
                1,
                &[ah, tre_b],
                &[sb_b],
            )?;
        }
        {
            let b = self.batch.as_ref().ok_or("batch scratch")?;
            let ds = self.ctx.fresh_ds_for(&b.p3, 3)?;
            self.ctx.bind_bufs(ds, &[sb_b, svh_b, yb]);
            let push: Vec<u8> = [(n / 128) as u32, 1u32, n as u32]
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect();
            crate::rawvk::context::site::set_tag("e3_had_out_t");
            self.ctx.run_rw(
                b.p3.pl,
                ds,
                b.p3.pipe,
                &push,
                (n / 128) as u32,
                t_rows,
                1,
                &[sb_b, svh_b],
                &[yb],
            )?;
        }
        Ok(())
    }

    fn chain_batch_one(
        &mut self,
        x_src: ash::vk::Buffer,
        li: usize,
        t_rows: u32,
        ah: ash::vk::Buffer,
        yb: ash::vk::Buffer,
        f32_in: bool,
    ) -> Result<(), String> {
        let l = &self.linears[li].1;
        let (k, n, krate) = (l.k, l.n, l.krate);
        let (suh_b, tre_b, svh_b) = (l.suh.buf, l.tre.buf, l.svh.buf);
        let b = self.batch.as_ref().ok_or("batch scratch 미초기화")?;
        let sb_b = b.sb.buf;

        // had_in: x_src × suh → ah (행별 WHT, grid=(k/128, T))
        {
            let (pl, pipe, tag) = if f32_in {
                (&b.p1f.pl, b.p1f.pipe, "e3_had_in_tf32")
            } else {
                (&b.p1.pl, b.p1.pipe, "e3_had_in_t")
            };
            let ds = self
                .ctx
                .fresh_ds_for(if f32_in { &b.p1f } else { &b.p1 }, 3)?;

            self.ctx.bind_bufs(ds, &[x_src, suh_b, ah]);
            let push: Vec<u8> = [(k / 128) as u32, k as u32]
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect();
            crate::rawvk::context::site::set_tag(tag);
            self.ctx.run_rw(
                *pl,
                ds,
                pipe,
                &push,
                (k / 128) as u32,
                t_rows,
                1,
                &[x_src, suh_b],
                &[ah],
            )?;
        }

        // gemm2(coopmat): ah × tre → sb ([T][n], nseg=1 — k-분할 없음,
        // grid=(n/64, ceil(T/128))). II-3(reconstruct+hgemm)은 측정 부정
        // (2026-10-03: 56.2 vs 71.2 t/s — 재구성 트래픽+디스패치가 이득
        // 상쇄, BN=128 gemm2는 이미 디코드를 128토큰에 상각) — 원장 참조。
        {
            let ktiles = (k / 16) as u32;
            let ntiles = (n / 16) as u32;
            let ds = self.ctx.fresh_ds_for(&b.p2, 3)?;
            self.ctx.bind_bufs(ds, &[ah, tre_b, sb_b]);
            let push: Vec<u8> = [ktiles, ntiles, krate, t_rows]
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect();
            crate::rawvk::context::site::set_tag("e3_gemm2");
            self.ctx.run_rw(
                b.p2.pl,
                ds,
                b.p2.pipe,
                &push,
                ((n / 64) as u32).max(1),
                t_rows.div_ceil(128),
                1,
                &[ah, tre_b],
                &[sb_b],
            )?;
        }

        // had_out_t: sb 환원 × svh → yb (grid=(n/128, T), nseg=1 — gemm2)
        {
            let ds = self.ctx.fresh_ds_for(&b.p3, 3)?;
            self.ctx.bind_bufs(ds, &[sb_b, svh_b, yb]);
            let push: Vec<u8> = [(n / 128) as u32, 1u32, n as u32]
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect();
            crate::rawvk::context::site::set_tag("e3_had_out_t");
            self.ctx.run_rw(
                b.p3.pl,
                ds,
                b.p3.pipe,
                &push,
                (n / 128) as u32,
                t_rows,
                1,
                &[sb_b, svh_b],
                &[yb],
            )?;
        }
        Ok(())
    }

    /// T-배치 공유 입력 다중 선형(스테이징): y_i[t] = x[t] @ W_i^T (i ≤ 3).
    /// 입력은 호출자가 stage_f32() 버퍼에 [T][k]로 미리 기록했어야 한다
    /// (생산 par_rows가 직접 기록 — CPU f16 변환·업로드 복사 제거, 원장 #3).
    /// 체인 직렬·슬롯 분리·동기 1회 — 디코드 linear_triple의 배치판.
    pub fn linear_batch_multi_staged(
        &mut self,
        keys: &[&str],
        t_rows: usize,
    ) -> Result<Vec<Vec<f32>>, String> {
        if keys.is_empty() || keys.len() > 3 {
            return Err(format!(
                "linear_batch_multi_staged: keys {}개 (1..=3)",
                keys.len()
            ));
        }
        if t_rows == 0 || t_rows > BATCH_TMAX {
            return Err(format!("linear_batch: T={t_rows} 상한 {BATCH_TMAX} 위반"));
        }
        let mut idxs = Vec::with_capacity(keys.len());
        for key in keys {
            idxs.push(self.find_linear(key)?);
        }
        let k0 = self.linears[idxs[0]].1.k;
        for &li in &idxs[1..] {
            if self.linears[li].1.k != k0 {
                return Err("linear_batch_multi_staged: 입력 차원 불일치".to_string());
            }
        }
        self.ensure_batch()?;

        // 진단 분해(원장 89 키: exl3_lindbg) — flush/디스패치/대기/판독.
        static LINDBG_N: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let lindbg = llm170_diag::dump::opts().key("exl3_lindbg")
            && LINDBG_N.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 24;
        let lt0 = std::time::Instant::now();

        // 캐시(비결합) xtb 쓰기 → GPU 가시화 flush — 실사용 구간만(T×k).
        {
            let k0 = self.linears[idxs[0]].1.k;
            let b = self.batch.as_ref().ok_or("batch scratch")?;
            self.ctx.flush_range(&b.xtb, t_rows * k0 * 4);
        }
        let lt1 = std::time::Instant::now();

        let xtb = self.batch.as_ref().ok_or("batch scratch")?.xtb.buf;
        self.ctx.begin_batch()?;
        for (slot, &li) in idxs.iter().enumerate() {
            let b = self.batch.as_ref().ok_or("batch scratch")?;
            self.chain_batch_one(xtb, li, t_rows as u32, b.ah[slot].buf, b.yb[slot].buf, true)?;
        }
        let lt2 = std::time::Instant::now();
        self.ctx.end_batch_wait()?;
        self.ctx.wait_pending()?; // DBUF 비동기 잔여 배출
        let lt3 = std::time::Instant::now();

        // 캐시(비결합) yb — GPU 쓰기 판독 전 인밸리데이트(실사용 구간 T×n).
        let b = self.batch.as_ref().ok_or("batch scratch")?;
        for (slot, &li) in idxs.iter().enumerate() {
            let n = self.linears[li].1.n;
            self.ctx.invalidate_range(&b.yb[slot], t_rows * n * 4);
        }

        let mut outs = Vec::with_capacity(idxs.len());
        for (slot, &li) in idxs.iter().enumerate() {
            let n = self.linears[li].1.n;
            let b = self.batch.as_ref().ok_or("batch scratch")?;
            let mut y = vec![0f32; t_rows * n];
            // SAFETY: end_batch_wait 후 매핑 판독 — t_rows*n ≤ TMAX*max_n.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    b.yb[slot].ptr as *const f32,
                    y.as_mut_ptr(),
                    t_rows * n,
                );
            }
            outs.push(y);
        }
        if lindbg {
            let lt4 = std::time::Instant::now();
            eprintln!(
                "[lindbg] {:52} fl {:6.2} disp {:6.2} wait {:6.2} rd {:6.2} ms",
                keys[0],
                (lt1 - lt0).as_secs_f64() * 1e3,
                (lt2 - lt1).as_secs_f64() * 1e3,
                (lt3 - lt2).as_secs_f64() * 1e3,
                (lt4 - lt3).as_secs_f64() * 1e3,
            );
        }
        Ok(outs)
    }

    /// T-배치 단일 선형(스테이징) — stage_f32 버퍼의 [T][k]를 소비.
    pub fn linear_batch_staged(&mut self, key: &str, t_rows: usize) -> Result<Vec<f32>, String> {
        self.linear_batch_multi_staged(&[key], t_rows)?
            .into_iter()
            .next()
            .ok_or_else(|| "linear_batch_staged: 결과 없음".to_string())
    }

    /// 스테이징 버퍼 포인터 — 호출자의 병렬 생산자가 [T][k] f32 행을 직접
    /// 기록한다(행 스트라이드 = 해당 선형의 k). 기록 후 staged 호출로 소비.
    pub fn stage_f32(&mut self) -> Result<*mut f32, String> {
        self.ensure_batch()?;
        Ok(self.batch.as_ref().ok_or("batch scratch")?.xtb.ptr as *mut f32)
    }

    pub fn ffn_trio_batch(
        &mut self,
        key_g: &str,
        key_u: &str,
        key_d: &str,
        t_rows: usize,
    ) -> Result<Vec<f32>, String> {
        self.ffn_trio_impl(key_g, key_u, key_d, t_rows, false, false)
    }

    #[allow(clippy::too_many_arguments)]
    fn ffn_trio_impl(
        &mut self,
        key_g: &str,
        key_u: &str,
        key_d: &str,
        t_rows: usize,
        gpu_input: bool,
        preah: bool,
    ) -> Result<Vec<f32>, String> {
        if t_rows == 0 || t_rows > BATCH_TMAX {
            return Err(format!("ffn_trio_batch: T={t_rows} 상한 {BATCH_TMAX} 위반"));
        }
        let ig = self.find_linear(key_g)?;
        let iu = self.find_linear(key_u)?;
        let id = self.find_linear(key_d)?;
        let (kg, ng, nd) = (
            self.linears[ig].1.k,
            self.linears[ig].1.n,
            self.linears[id].1.n,
        );
        if self.linears[iu].1.k != kg || self.linears[id].1.k != ng {
            return Err(format!("{key_g}/{key_u}/{key_d}: FFN 차원 불일치"));
        }
        self.ensure_batch()?;
        if !gpu_input {
            let b = self.batch.as_ref().ok_or("batch scratch")?;
            self.ctx.flush_range(&b.xtb, t_rows * kg * 4);
        }
        let b0 = self.batch.as_ref().ok_or("batch scratch")?;
        let (xtb, ah0, yb0, ah1, yb1, ah2, yb2, x2t_b, p4t_pl, p4t_pipe) = (
            b0.xtb.buf,
            b0.ah[0].buf,
            b0.yb[0].buf,
            b0.ah[1].buf,
            b0.yb[1].buf,
            b0.ah[2].buf,
            b0.yb[2].buf,
            b0.x2t.buf,
            b0.p4t.pl,
            b0.p4t.pipe,
        );
        // ew_t 디스크립터는 체인 전에 선점 확보(borrow 분리 — 복사본 사용).
        let ds4 = self.ctx.fresh_ds_for(&b0.p4t, 3)?;
        self.ctx.begin_batch()?;
        // preah: 선행 norm_resid_had가 ah0(gate)/ah1(up) 기록 — had_in 생략.
        if preah {
            self.chain_gemmonly(ig, t_rows as u32, ah0, yb0)?;
            self.chain_gemmonly(iu, t_rows as u32, ah1, yb1)?;
        } else {
            self.chain_batch_one(xtb, ig, t_rows as u32, ah0, yb0, true)?;
            self.chain_batch_one(xtb, iu, t_rows as u32, ah1, yb1, true)?;
        }
        // ew_t: yb1(g) × yb2(u) → x2t(f16 쌍팩 [T][ng/2])
        self.ctx.bind_bufs(ds4, &[yb0, yb1, x2t_b]);
        let push4 = (ng as u32).to_le_bytes().to_vec();
        crate::rawvk::context::site::set_tag("e3_ffn_ew_t");
        self.ctx.run_rw(
            p4t_pl,
            ds4,
            p4t_pipe,
            &push4,
            ng.div_ceil(512) as u32,
            t_rows as u32,
            1,
            &[yb0, yb1],
            &[x2t_b],
        )?;
        // down: x2t(f16) 직독 — had_in_t(f16) 레그.
        self.chain_batch_one(x2t_b, id, t_rows as u32, ah2, yb2, false)?;
        self.ctx.end_batch_wait()?;
        self.ctx.wait_pending()?;
        if gpu_input {
            return Ok(Vec::new());
        }
        {
            let b = self.batch.as_ref().ok_or("batch scratch")?;
            self.ctx.invalidate_range(&b.yb[2], t_rows * nd * 4);
        }
        let mut y = vec![0f32; t_rows * nd];
        {
            let b = self.batch.as_ref().ok_or("batch scratch")?;
            // SAFETY: end_batch_wait 후 매핑 판독 — t_rows*nd ≤ TMAX*max_n.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    b.yb[2].ptr as *const f32,
                    y.as_mut_ptr(),
                    t_rows * nd,
                );
            }
        }
        Ok(y)
    }
    /// T-배치 다중 선형(GPU 상주 변형) — 결과를 yb 슬롯에 남긴다(판독 없음).
    /// plans/121 F1: GDN 프레임이 yb0/yb1을 직접 소비.
    /// 반환: 슬롯별 (buffer handle, n) — 소비자 커널이 바인딩에 사용.
    pub fn linear_batch_multi_gpu(
        &mut self,
        keys: &[&str],
        t_rows: usize,
    ) -> Result<Vec<(ash::vk::Buffer, usize)>, String> {
        if keys.is_empty() || keys.len() > 3 {
            return Err(format!(
                "linear_batch_multi_gpu: keys {}개 (1..=3)",
                keys.len()
            ));
        }
        if t_rows == 0 || t_rows > BATCH_TMAX {
            return Err(format!("linear_batch_gpu: T={t_rows} 상한 위반"));
        }
        let mut idxs = Vec::with_capacity(keys.len());
        for key in keys {
            idxs.push(self.find_linear(key)?);
        }
        self.ensure_batch()?;
        {
            let k0 = self.linears[idxs[0]].1.k;
            let b = self.batch.as_ref().ok_or("batch scratch")?;
            self.ctx.flush_range(&b.xtb, t_rows * k0 * 4);
        }
        let xtb = self.batch.as_ref().ok_or("batch scratch")?.xtb.buf;
        self.ctx.begin_batch()?;
        for (slot, &li) in idxs.iter().enumerate() {
            let b = self.batch.as_ref().ok_or("batch scratch")?;
            self.chain_batch_one(xtb, li, t_rows as u32, b.ah[slot].buf, b.yb[slot].buf, true)?;
        }
        self.ctx.end_batch_wait()?;
        self.ctx.wait_pending()?;
        // 판독 없음 — yb에 잔류. 소비자가 invalidate 후 판독하거나 GPU 직독.
        let b = self.batch.as_ref().ok_or("batch scratch")?;
        Ok(idxs
            .iter()
            .enumerate()
            .map(|(slot, &li)| (b.yb[slot].buf, self.linears[li].1.n))
            .collect())
    }

    /// GDN 프레임 초기화(plans/121 F1) — 상수 업로드 1회 + 스크래치/상태 할당.
    /// GDN 층 수는 64층 중 il%4!=3 → 48층.
    pub fn gdn_frame_init(&mut self) -> Result<(), String> {
        if self.batch.as_ref().is_some_and(|b| b.gframe.is_some()) {
            return Ok(());
        }
        self.ensure_batch()?;
        let n_gdn = self.n_layers - self.n_layers / 4; // 48

        // 파이프라인 4종
        let pgc = self
            .ctx
            .pipeline_pipes(include_bytes!("../spv/exl3_gdn_conv.spv"), 6, 8)?;
        let pgl = self
            .ctx
            .pipeline_pipes(include_bytes!("../spv/exl3_gdn_l2perm.spv"), 11, 20)?;
        let pgs = self
            .ctx
            .pipeline_pipes(include_bytes!("../spv/exl3_gdn_scan.spv"), 6, 20)?;
        let pgg = self
            .ctx
            .pipeline_pipes(include_bytes!("../spv/exl3_gdn_gate.spv"), 4, 20)?;

        // 스크래치(TMAX 기준)
        let gq = self.ctx.alloc_host_cached(BATCH_TMAX * 2048 * 4)?;
        let gk = self.ctx.alloc_host_cached(BATCH_TMAX * 2048 * 4)?;
        let gv = self.ctx.alloc_host_cached(BATCH_TMAX * 6144 * 4)?;
        let gbg = self.ctx.alloc_host_cached(BATCH_TMAX * 96 * 4)?;
        let go = self.ctx.alloc_host_cached(BATCH_TMAX * 6144 * 4)?;
        let gqr = self.ctx.alloc_host_cached(BATCH_TMAX * 2048 * 4)?;
        let gkr = self.ctx.alloc_host_cached(BATCH_TMAX * 2048 * 4)?;
        let gvr = self.ctx.alloc_host_cached(BATCH_TMAX * 6144 * 4)?;

        // 상수(48층 분) — 트레이리던트 norms에서 직접 복사.
        let ab = self.ctx.alloc_host_cached(n_gdn * 2 * 48 * 5120 * 4)?;
        let cw = self.ctx.alloc_host_cached(n_gdn * 10240 * 4 * 4)?;
        let alog = self.ctx.alloc_host_cached(n_gdn * 48 * 4)?;
        let dtb = self.ctx.alloc_host_cached(n_gdn * 48 * 4)?;
        let nw = self.ctx.alloc_host_cached(n_gdn * 128 * 4)?;

        // 상태(48층 분, GPU)
        let gring = self.ctx.alloc_host_cached(n_gdn * 3 * 10240 * 4)?;
        let gstate = self.ctx.alloc_host_cached(n_gdn * 48 * 16384 * 4)?;

        // 상수 업로드 — 각 GDN 층(il%4!=3)의 norms를 GPU 버퍼에.
        unsafe {
            let abp = ab.ptr as *mut f32;
            let cwp = cw.ptr as *mut f32;
            let alp = alog.ptr as *mut f32;
            let dtp = dtb.ptr as *mut f32;
            let nwp = nw.ptr as *mut f32;
            let mut g = 0usize;
            for il in 0..self.n_layers {
                if il % 4 == 3 {
                    continue;
                } // full attention
                let lp = format!("model.language_model.layers.{il}.linear_attn");
                let a_p = self
                    .norm(&format!("{lp}.in_proj_a.weight"))
                    .ok_or("a_proj")?;
                let b_p = self
                    .norm(&format!("{lp}.in_proj_b.weight"))
                    .ok_or("b_proj")?;
                let c_w = self.norm(&format!("{lp}.conv1d.weight")).ok_or("conv_w")?;
                let a_l = self.norm(&format!("{lp}.A_log")).ok_or("A_log")?;
                let d_b = self.norm(&format!("{lp}.dt_bias")).ok_or("dt_bias")?;
                let n_w = self.norm(&format!("{lp}.norm.weight")).ok_or("norm_w")?;
                std::ptr::copy_nonoverlapping(a_p.as_ptr(), abp.add(g * 2 * 48 * 5120), 48 * 5120);
                std::ptr::copy_nonoverlapping(
                    b_p.as_ptr(),
                    abp.add(g * 2 * 48 * 5120 + 48 * 5120),
                    48 * 5120,
                );
                std::ptr::copy_nonoverlapping(c_w.as_ptr(), cwp.add(g * 10240 * 4), 10240 * 4);
                std::ptr::copy_nonoverlapping(a_l.as_ptr(), alp.add(g * 48), 48);
                std::ptr::copy_nonoverlapping(d_b.as_ptr(), dtp.add(g * 48), 48);
                std::ptr::copy_nonoverlapping(n_w.as_ptr(), nwp.add(g * 128), 128);
                g += 1;
            }
        }
        self.ctx.flush_buf(&ab);
        self.ctx.flush_buf(&cw);
        self.ctx.flush_buf(&alog);
        self.ctx.flush_buf(&dtb);
        self.ctx.flush_buf(&nw);

        if let Some(b) = self.batch.as_mut() {
            b.gframe = Some(GdnFrame {
                gq,
                gk,
                gv,
                gbg,
                go,
                gqr,
                gkr,
                gvr,
                ab,
                cw,
                alog,
                dtb,
                nw,
                gring,
                gstate,
                pgc,
                pgl,
                pgs,
                pgg,
            });
        }
        Ok(())
    }

    /// GDN 층 전체 GPU 상주 처리(plans/121 F1) — qkv/yb0, z/yb1에서 xtb(gated)까지.
    /// 호출 전 linear_batch_multi_gpu(qkv+z)가 yb 슬롯에 결과를 남겼어야 한다.
    /// gdn_il은 GDN 층 인덱스(0..48, il%4!=3 순서).
    pub fn gdn_layer_gpu(
        &mut self,
        gdn_il: usize,
        t_rows: usize,
        _xn_ptr: *mut f32,
        yb0: ash::vk::Buffer,
        yb1: ash::vk::Buffer,
    ) -> Result<(), String> {
        let gf = match self.batch.as_ref().and_then(|b| b.gframe.as_ref()) {
            Some(g) => g,
            None => return Err("gdn_frame 미초기화".into()),
        };

        self.ctx.begin_batch()?;

        // ① conv: yb0(qkv) → gqr/gkr/gvr + ring 갱신
        {
            let ds = self.ctx.fresh_ds_for(&gf.pgc, 6)?;
            self.ctx.bind_bufs(
                ds,
                &[
                    yb0,
                    gf.cw.buf,
                    gf.gring.buf,
                    gf.gqr.buf,
                    gf.gkr.buf,
                    gf.gvr.buf,
                ],
            );
            let push: Vec<u8> = [t_rows as u32, gdn_il as u32]
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect();
            crate::rawvk::context::site::set_tag("e3_gdn_conv");
            self.ctx.run_rw(
                gf.pgc.pl,
                ds,
                gf.pgc.pipe,
                &push,
                80,
                1,
                1,
                &[yb0],
                &[gf.gqr.buf, gf.gkr.buf, gf.gvr.buf, gf.gring.buf],
            )?;
        }

        // ② l2perm: gqr/gkr/gvr + xn(xtb) + 상수 → gq/gk/gv/gbg
        {
            let ds = self.ctx.fresh_ds_for(&gf.pgl, 11)?;
            // xn은 xtb 버퍼 — 호출자가 stage_f32()에 기록했음.
            let xtb = self.batch.as_ref().ok_or("batch")?.xtb.buf;
            // 상수는 gdn_il 슬라이스… 마찬가지로 전체 버퍼 바인딩 + 커널에 층 오프셋 필요.
            // TODO: l2perm 커널에 gdn_il push 추가(상수 오프셋).
            self.ctx.bind_bufs(
                ds,
                &[
                    gf.gqr.buf,
                    gf.gkr.buf,
                    gf.gvr.buf,
                    xtb,
                    gf.ab.buf,
                    gf.alog.buf,
                    gf.dtb.buf,
                    gf.gq.buf,
                    gf.gk.buf,
                    gf.gv.buf,
                    gf.gbg.buf,
                ],
            );
            let push: Vec<u8> = [t_rows as u32, 16u32, 48u32, 128u32, gdn_il as u32]
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect();
            crate::rawvk::context::site::set_tag("e3_gdn_l2perm");
            self.ctx.run_rw(
                gf.pgl.pl,
                ds,
                gf.pgl.pipe,
                &push,
                48,
                t_rows as u32,
                1,
                &[
                    gf.gqr.buf,
                    gf.gkr.buf,
                    gf.gvr.buf,
                    xtb,
                    gf.ab.buf,
                    gf.alog.buf,
                    gf.dtb.buf,
                ],
                &[gf.gq.buf, gf.gk.buf, gf.gv.buf, gf.gbg.buf],
            )?;
        }

        // 실입력 캡처(모듈 격리 디버그 — plans/121 F2): l2perm 직후 gq/gk/gv/gbg.
        if gdn_il == 0
            && let Some(path) =
                llm170_diag::flag::val("LLM170_EXL3_SCAN_CAP").map(|s| s.to_string())
        {
            let g2 = self.batch.as_ref().and_then(|b| b.gframe.as_ref());
            if let Some(g) = g2 {
                let n_q = t_rows * 2048;
                let n_bg = t_rows * 96;
                self.ctx.invalidate_range(&g.gq, n_q * 4);
                self.ctx.invalidate_range(&g.gk, n_q * 4);
                self.ctx.invalidate_range(&g.gv, t_rows * 6144 * 4);
                self.ctx.invalidate_range(&g.gbg, n_bg * 4);
                // SAFETY: 호스트 매핑 버퍼 직독 — invalidate 직후 유효.
                let qs = unsafe { std::slice::from_raw_parts(g.gq.ptr as *const f32, n_q) };
                let ks = unsafe { std::slice::from_raw_parts(g.gk.ptr as *const f32, n_q) };
                let vs =
                    unsafe { std::slice::from_raw_parts(g.gv.ptr as *const f32, t_rows * 6144) };
                let bgs = unsafe { std::slice::from_raw_parts(g.gbg.ptr as *const f32, n_bg) };
                let mut buf = Vec::with_capacity((n_q * 2 + t_rows * 6144 + n_bg) * 4);
                // SAFETY: 위 직독 슬라이스의 원시 바이트 재해석 — 같은 라이프타임.
                unsafe {
                    for s in [qs, ks, vs, bgs] {
                        buf.extend_from_slice(std::slice::from_raw_parts(
                            s.as_ptr() as *const u8,
                            s.len() * 4,
                        ));
                    }
                }
                std::fs::write(&path, &buf).map_err(|e| format!("scan cap: {e}"))?;
                eprintln!("  [scancap] L0 t={t_rows} → {path}");
            }
        }
        // ③ scan: gq/gk/gv/gbg + gstate → go
        {
            let ds = self.ctx.fresh_ds_for(&gf.pgs, 6)?;
            self.ctx.bind_bufs(
                ds,
                &[
                    gf.gq.buf,
                    gf.gk.buf,
                    gf.gv.buf,
                    gf.gbg.buf,
                    gf.gstate.buf,
                    gf.go.buf,
                ],
            );
            let push: Vec<u8> = [t_rows as u32, 16u32, 48u32, 128u32, gdn_il as u32]
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect();
            crate::rawvk::context::site::set_tag("e3_gdn_scan");
            self.ctx.run_rw(
                gf.pgs.pl,
                ds,
                gf.pgs.pipe,
                &push,
                48,
                1,
                1,
                &[gf.gq.buf, gf.gk.buf, gf.gv.buf, gf.gbg.buf, gf.gstate.buf],
                &[gf.gstate.buf, gf.go.buf],
            )?;
        }

        // ④ gate: go + yb1(z) + nw → xtb(gated)
        {
            let ds = self.ctx.fresh_ds_for(&gf.pgg, 4)?;
            let xtb = self.batch.as_ref().ok_or("batch")?.xtb.buf;
            self.ctx.bind_bufs(ds, &[gf.go.buf, yb1, gf.nw.buf, xtb]);
            let push: Vec<u8> = [t_rows as u32, 16u32, 48u32, 128u32, gdn_il as u32]
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect();
            crate::rawvk::context::site::set_tag("e3_gdn_gate");
            self.ctx.run_rw(
                gf.pgg.pl,
                ds,
                gf.pgg.pipe,
                &push,
                48,
                t_rows as u32,
                1,
                &[gf.go.buf, yb1, gf.nw.buf],
                &[xtb],
            )?;
        }

        self.ctx.end_batch_wait()?;
        self.ctx.wait_pending()?;
        // 체인 입출력 캡처(plans/121 워크플로 — GDN 비선형 전체 프로브용):
        #[allow(clippy::collapsible_if, clippy::needless_borrows_for_generic_args)]
        // 입력 yb0(qkv)/yb1(z) + 게이트 출력 xtb를 gdn_il==0에서 파일로.
        if gdn_il == 0 {
            if let Some(path) = llm170_diag::flag::val("LLM170_EXL3_CHAINCAP") {
                let gf2 = self.batch.as_ref().and_then(|b| b.gframe.as_ref());
                if gf2.is_some() {
                    let b2 = self.batch.as_ref().ok_or("batch")?;
                    let nq = t_rows * 10240;
                    let nz = t_rows * 6144;
                    let nx = t_rows * 6144;
                    self.ctx.invalidate_range(&b2.yb[0], nq * 4);
                    self.ctx.invalidate_range(&b2.yb[1], nz * 4);
                    self.ctx.invalidate_range(&b2.xtb, nx * 4);
                    // SAFETY: end_batch_wait 후 매핑 판독.
                    let (q, z, x) = unsafe {
                        (
                            std::slice::from_raw_parts(b2.yb[0].ptr as *const f32, nq),
                            std::slice::from_raw_parts(b2.yb[1].ptr as *const f32, nz),
                            std::slice::from_raw_parts(b2.xtb.ptr as *const f32, nx),
                        )
                    };
                    let mut buf = Vec::with_capacity((nq + nz + nx) * 4);
                    // SAFETY: 위 직독 슬라이스의 바이트 재해석 — 동일 수명.
                    unsafe {
                        for sl in [q, z, x] {
                            buf.extend_from_slice(std::slice::from_raw_parts(
                                sl.as_ptr() as *const u8,
                                sl.len() * 4,
                            ));
                        }
                    }
                    std::fs::write(&path, buf).map_err(|e| format!("chain cap: {e}"))?;
                    eprintln!("  [chaincap] L0 t={t_rows} → {path}");
                }
            }
        }
        Ok(())
    }

    /// GDN 상태 GPU→CPU 동기화(plans/121 F1) — GPU 배치 후 차기 디코드 정합.
    /// states는 [48*16384] f32, conv는 [3*10240] f32 다운로드.
    /// 다운로드 후 CPU 사본이 권위 — 이후 CPU 디코드가 상태를 진화시킬 수
    /// 있으므로 GPU 유효 플래그는 해제한다(plans/121 F2 스케줄).
    pub fn gdn_state_sync(
        &mut self,
        gdn_il: usize,
        states: &mut [f32],
        conv: &mut [f32],
    ) -> Result<(), String> {
        let gf = match self.batch.as_ref().and_then(|b| b.gframe.as_ref()) {
            Some(g) => g,
            None => return Err("gdn_frame ì±ì ìí ëê¸°í ì¤í¨".into()),
        };
        let st_bytes = 48 * 16384 * 4;
        let ring_bytes = 3 * 10240 * 4;
        self.ctx
            .invalidate_range_at(&gf.gstate, gdn_il * st_bytes, st_bytes);
        self.ctx
            .invalidate_range_at(&gf.gring, gdn_il * ring_bytes, ring_bytes);
        let st_off = gdn_il * 48 * 16384;
        let ring_off = gdn_il * 3 * 10240;
        unsafe {
            std::ptr::copy_nonoverlapping(
                gf.gstate.ptr.add(st_off * 4) as *const f32,
                states.as_mut_ptr(),
                48 * 16384,
            );
            std::ptr::copy_nonoverlapping(
                gf.gring.ptr.add(ring_off * 4) as *const f32,
                conv.as_mut_ptr(),
                3 * 10240,
            );
        }
        if gdn_il < self.gdn_st_valid.len() {
            self.gdn_st_valid[gdn_il] = false;
        }
        Ok(())
    }

    /// xtb 범위 인밸리데이트(GPU gate 출력 → 호스트 가시화, plans/121 F1).
    pub fn invalidate_xtb(&mut self, bytes: usize) {
        if let Some(b) = self.batch.as_ref() {
            self.ctx.invalidate_range(&b.xtb, bytes);
        }
    }

    /// GDN 상태 CPU→GPU 업로드(plans/121 F1) — 초기화/리셋용.
    pub fn gdn_state_upload(
        &mut self,
        gdn_il: usize,
        states: &[f32],
        conv: &[f32],
    ) -> Result<(), String> {
        // 프리필 스케줄(plans/121 F2): GPU 상태가 이미 유효하면 업로드 스킵.
        if *self.gdn_st_valid.get(gdn_il).unwrap_or(&false) {
            return Ok(());
        }
        let gf = match self.batch.as_ref().and_then(|b| b.gframe.as_ref()) {
            Some(g) => g,
            None => return Err("gdn_frame 미초기화".into()),
        };
        let st_off = gdn_il * 48 * 16384;
        let ring_off = gdn_il * 3 * 10240;
        unsafe {
            std::ptr::copy_nonoverlapping(
                states.as_ptr(),
                gf.gstate.ptr.add(st_off * 4) as *mut f32,
                48 * 16384,
            );
            std::ptr::copy_nonoverlapping(
                conv.as_ptr(),
                gf.gring.ptr.add(ring_off * 4) as *mut f32,
                3 * 10240,
            );
        }
        let st_bytes = 48 * 16384 * 4;
        let ring_bytes = 3 * 10240 * 4;
        self.ctx
            .flush_range_at(&gf.gstate, gdn_il * st_bytes, st_bytes);
        self.ctx
            .flush_range_at(&gf.gring, gdn_il * ring_bytes, ring_bytes);
        if gdn_il < self.gdn_st_valid.len() {
            self.gdn_st_valid[gdn_il] = true;
        }
        Ok(())
    }

    /// out_proj GPU 직결(plans/121 F2 스케줄) — gate가 xtb에 GPU 기록한 출력을
    /// flush/인밸리데이트 왕복 없이 GEMM 입력으로 소비하고 결과만 판독한다.
    pub fn linear_out_gpu(&mut self, key: &str, t_rows: usize) -> Result<Vec<f32>, String> {
        let li = self.find_linear(key)?;
        let n = self.linears[li].1.n;
        self.ensure_batch()?;
        let (xtb, ah0, yb0) = {
            let b = self.batch.as_ref().ok_or("batch scratch")?;
            (b.xtb.buf, b.ah[0].buf, b.yb[0].buf)
        };
        self.ctx.begin_batch()?;
        self.chain_batch_one(xtb, li, t_rows as u32, ah0, yb0, true)?;
        self.ctx.end_batch_wait()?;
        self.ctx.wait_pending()?;
        {
            let b = self.batch.as_ref().ok_or("batch scratch")?;
            self.ctx.invalidate_range(&b.yb[0], t_rows * n * 4);
        }
        let mut y = vec![0f32; t_rows * n];
        {
            let b = self.batch.as_ref().ok_or("batch scratch")?;
            // SAFETY: end_batch_wait 후 매핑 판독 — t_rows*n ≤ TMAX*max_n.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    b.yb[0].ptr as *const f32,
                    y.as_mut_ptr(),
                    t_rows * n,
                );
            }
        }
        Ok(y)
    }

    /// yb 슬롯 선두 값 판독(디버그) — invalidate 후 읽음.
    pub fn read_yb_head(&mut self, slot: usize, count: usize) -> Vec<f32> {
        let b = match self.batch.as_ref() {
            Some(b) => b,
            None => return vec![],
        };
        if slot >= 3 {
            return vec![];
        }
        self.ctx.invalidate_range(&b.yb[slot], count * 4);
        let mut out = vec![0f32; count];
        unsafe {
            std::ptr::copy_nonoverlapping(b.yb[slot].ptr as *const f32, out.as_mut_ptr(), count);
        }
        out
    }

    /// GDN 프레임 gqr(conv q 출력) 선두 판독(디버그).
    pub fn read_gqr_head(&mut self, count: usize) -> Vec<f32> {
        let b = match self.batch.as_ref() {
            Some(b) => b,
            None => return vec![],
        };
        let gf = match b.gframe.as_ref() {
            Some(g) => g,
            None => return vec![],
        };
        self.ctx.invalidate_range(&gf.gqr, count * 4);
        let mut out = vec![0f32; count];
        unsafe {
            std::ptr::copy_nonoverlapping(gf.gqr.ptr as *const f32, out.as_mut_ptr(), count);
        }
        out
    }

    /// GDN 프레임 gq(L2 norm q) 선두 판독(디버그).
    pub fn read_gq_head(&mut self, count: usize) -> Vec<f32> {
        let b = match self.batch.as_ref() {
            Some(b) => b,
            None => return vec![],
        };
        let gf = match b.gframe.as_ref() {
            Some(g) => g,
            None => return vec![],
        };
        self.ctx.invalidate_range(&gf.gq, count * 4);
        let mut out = vec![0f32; count];
        unsafe {
            std::ptr::copy_nonoverlapping(gf.gq.ptr as *const f32, out.as_mut_ptr(), count);
        }
        out
    }
    /// GDN 프레임 gbg(beta|g) 선두 판독(디버그).
    pub fn read_gbg_head(&mut self, count: usize) -> Vec<f32> {
        let b = match self.batch.as_ref() {
            Some(b) => b,
            None => return vec![],
        };
        let gf = match b.gframe.as_ref() {
            Some(g) => g,
            None => return vec![],
        };
        self.ctx.invalidate_range(&gf.gbg, count * 4);
        let mut out = vec![0f32; count];
        unsafe {
            std::ptr::copy_nonoverlapping(gf.gbg.ptr as *const f32, out.as_mut_ptr(), count);
        }
        out
    }
}

impl TrellisResident {
    /// 어텐션 프레임 초기화(plans/121 F2b) — KV 캐시·q/k 노름 상주.
    pub fn attn_frame_init(&mut self) -> Result<(), String> {
        if self.batch.as_ref().is_some_and(|b| b.aframe.is_some()) {
            return Ok(());
        }
        self.ensure_batch()?;
        let pa = self
            .ctx
            .pipeline_pipes(include_bytes!("../spv/exl3_attn_prep.spv"), 9, 8)?;
        let pf3 = self
            .ctx
            .pipeline_pipes(include_bytes!("../spv/exl3_attn_fwd3.spv"), 6, 8)?;
        let pbuf = self.ctx.alloc_host_cached(16)?;
        unsafe {
            std::ptr::write_bytes(pbuf.ptr, 0, 16);
        }
        self.ctx.flush_buf(&pbuf);
        let kkc = self.ctx.alloc_host_cached(16 * 1024 * 1024 * 4)?;
        let vkc = self.ctx.alloc_host_cached(16 * 1024 * 1024 * 4)?;
        let qh = self.ctx.alloc_host_cached(BATCH_TMAX * 6144 * 4)?;
        let qnws = self.ctx.alloc_host_cached(16 * 256 * 4)?;
        let knws = self.ctx.alloc_host_cached(16 * 256 * 4)?;
        // 노름 업로드 — 어텐션 층(il%4==3)의 q_norm/k_norm.
        unsafe {
            std::ptr::write_bytes(kkc.ptr, 0, 16 * 1024 * 1024 * 4);
            std::ptr::write_bytes(vkc.ptr, 0, 16 * 1024 * 1024 * 4);
            let mut ai = 0usize;
            for il in 0..self.n_layers {
                if il % 4 != 3 {
                    continue;
                }
                let lp = format!("model.language_model.layers.{il}.self_attn");
                let qw = self.norm(&format!("{lp}.q_norm.weight")).ok_or("q_norm")?;
                let kw = self.norm(&format!("{lp}.k_norm.weight")).ok_or("k_norm")?;
                std::ptr::copy_nonoverlapping(
                    qw.as_ptr(),
                    qnws.ptr.add(ai * 256 * 4) as *mut f32,
                    256,
                );
                std::ptr::copy_nonoverlapping(
                    kw.as_ptr(),
                    knws.ptr.add(ai * 256 * 4) as *mut f32,
                    256,
                );
                ai += 1;
            }
        }
        self.ctx.flush_buf(&qnws);
        self.ctx.flush_buf(&knws);
        self.ctx.flush_buf(&kkc);
        self.ctx.flush_buf(&vkc);
        if let Some(b) = self.batch.as_mut() {
            b.aframe = Some(AttnFrame {
                pbuf,
                kkc,
                vkc,
                qh,
                qnws,
                knws,
                pa,
                pf3,
            });
        }
        Ok(())
    }

    /// 디버그: gstate 선두 8값 판독(재생 NaN 국소화 — plans/121 tg).
    pub fn debug_gstate_head(&mut self) -> Result<[f32; 8], String> {
        let g = self
            .batch
            .as_ref()
            .and_then(|b| b.gframe.as_ref())
            .ok_or("gframe")?;
        self.ctx.invalidate_range(&g.gstate, 32);
        let mut out = [0f32; 8];
        unsafe {
            std::ptr::copy_nonoverlapping(g.gstate.ptr as *const f32, out.as_mut_ptr(), 8);
        }
        Ok(out)
    }

    /// pos0 매개변수 버퍼 기록(재생 경로 — 호스트가 라운드마다 갱신).
    pub fn attn_set_pos(&mut self, pos0: u32) -> Result<(), String> {
        let af = self
            .batch
            .as_ref()
            .and_then(|b| b.aframe.as_ref())
            .ok_or("aframe")?;
        unsafe {
            *(af.pbuf.ptr as *mut u32) = pos0;
        }
        self.ctx.flush_range(&af.pbuf, 4);
        Ok(())
    }

    /// 어텐션 층 GPU 경로(plans/121 F2b): prep(q/k norm+rope+KV 적립) →
    /// fwd3(인과 어텐션+게이트) → xtb 직접 기록. yb0/1/2 = q‖gate/k/v GEMM 출력.
    #[allow(clippy::too_many_arguments)]
    pub fn attn_layer_gpu(
        &mut self,
        attn_il: usize,
        t_rows: usize,
        pos0: u32,
        yb0: ash::vk::Buffer,
        yb1: ash::vk::Buffer,
        yb2: ash::vk::Buffer,
    ) -> Result<(), String> {
        self.attn_frame_init()?;
        let af = self
            .batch
            .as_ref()
            .and_then(|b| b.aframe.as_ref())
            .ok_or("aframe")?;
        let xtb = self.batch.as_ref().ok_or("batch")?.xtb.buf;
        let (kkc, vkc, qh, qnws, knws, pbuf) = (
            af.kkc.buf,
            af.vkc.buf,
            af.qh.buf,
            af.qnws.buf,
            af.knws.buf,
            af.pbuf.buf,
        );
        // pos0 → 매개변수 버퍼(재생 지원 — 커널은 pp[0] 판독, 푸시는 불변).
        unsafe {
            *(af.pbuf.ptr as *mut u32) = pos0;
        }
        self.ctx.flush_range(&af.pbuf, 4);
        self.ctx.begin_batch()?;
        {
            let ds = self.ctx.fresh_ds_for(&af.pa, 9)?;
            self.ctx
                .bind_bufs(ds, &[yb0, yb1, yb2, qnws, knws, qh, kkc, vkc, pbuf]);
            let push: Vec<u8> = [t_rows as u32, pos0, attn_il as u32]
                .iter()
                .flat_map(|x| x.to_le_bytes())
                .collect();
            crate::rawvk::context::site::set_tag("e3_attn_prep");
            self.ctx.run_rw(
                af.pa.pl,
                ds,
                af.pa.pipe,
                &push,
                t_rows as u32,
                28,
                1,
                &[yb0, yb1, yb2, qnws, knws],
                &[qh, kkc, vkc],
            )?;
        }
        {
            let ds = self.ctx.fresh_ds_for(&af.pf3, 6)?;
            self.ctx.bind_bufs(ds, &[qh, kkc, vkc, yb0, xtb, pbuf]);
            let push: Vec<u8> = [t_rows as u32, pos0, attn_il as u32]
                .iter()
                .flat_map(|x| x.to_le_bytes())
                .collect();
            let nq = t_rows.div_ceil(4);
            crate::rawvk::context::site::set_tag("e3_attn_fwd3");
            self.ctx.run_rw(
                af.pf3.pl,
                ds,
                af.pf3.pipe,
                &push,
                nq as u32,
                24,
                1,
                &[qh, kkc, vkc, yb0],
                &[xtb],
            )?;
        }
        self.ctx.end_batch_wait()?;
        self.ctx.wait_pending()?;
        Ok(())
    }

    /// 프리필 종료 시 KV 캐시 GPU→CPU 벌크 동기(차기 디코드 정합).
    pub fn attn_kv_sync(&mut self, attn_il: usize, kv_len: usize) -> Result<[Vec<f32>; 2], String> {
        let b = self.batch.as_ref().ok_or("batch")?;
        let a = b.aframe.as_ref().ok_or("aframe")?;
        let n = kv_len * 1024;
        self.ctx
            .invalidate_range_at(&a.kkc, attn_il * 1024 * 1024 * 4, n * 4);
        self.ctx
            .invalidate_range_at(&a.vkc, attn_il * 1024 * 1024 * 4, n * 4);
        let k = unsafe {
            std::slice::from_raw_parts(a.kkc.ptr.add(attn_il * 1024 * 1024 * 4) as *const f32, n)
                .to_vec()
        };
        let v = unsafe {
            std::slice::from_raw_parts(a.vkc.ptr.add(attn_il * 1024 * 1024 * 4) as *const f32, n)
                .to_vec()
        };
        Ok([k, v])
    }
}

impl TrellisResident {
    /// 프레임 초기화(plans/121 원-서브밋) — 잔차/노름 상주.
    pub fn fframe_init(&mut self) -> Result<(), String> {
        if self.batch.as_ref().is_some_and(|b| b.fframe.is_some()) {
            return Ok(());
        }
        self.ensure_batch()?;
        let pnr = self
            .ctx
            .pipeline_pipes(include_bytes!("../spv/e3_norm_resid.spv"), 4, 8)?;
        let pnrh = self
            .ctx
            .pipeline_pipes(include_bytes!("../spv/e3_norm_resid_had.spv"), 8, 8)?;
        let xbuf = self.ctx.alloc_host_cached(BATCH_TMAX * 5120 * 4)?;
        let zeros = self.ctx.alloc_host_cached(BATCH_TMAX * 5120 * 4)?;
        let nw128 = self.ctx.alloc_host_cached(129 * 5120 * 4)?;
        unsafe {
            std::ptr::write_bytes(zeros.ptr, 0, BATCH_TMAX * 5120 * 4);
            std::ptr::write_bytes(xbuf.ptr, 0, BATCH_TMAX * 5120 * 4);
            for il in 0..self.n_layers {
                let lp = format!("model.language_model.layers.{il}");
                let wi = self
                    .norm(&format!("{lp}.input_layernorm.weight"))
                    .ok_or("input_ln")?;
                let wp = self
                    .norm(&format!("{lp}.post_attention_layernorm.weight"))
                    .ok_or("post_ln")?;
                std::ptr::copy_nonoverlapping(
                    wi.as_ptr(),
                    nw128.ptr.add((2 * il) * 5120 * 4) as *mut f32,
                    5120,
                );
                std::ptr::copy_nonoverlapping(
                    wp.as_ptr(),
                    nw128.ptr.add((2 * il + 1) * 5120 * 4) as *mut f32,
                    5120,
                );
            }
            let wo = self
                .norm("model.language_model.norm.weight")
                .ok_or("output_norm")?;
            // 행 128 = output_norm(행 127은 L63의 post_ln — 과거 덮어씀 버그)
            std::ptr::copy_nonoverlapping(
                wo.as_ptr(),
                nw128.ptr.add(128 * 5120 * 4) as *mut f32,
                5120,
            );
        }
        self.ctx.flush_buf(&nw128);
        self.ctx.flush_buf(&zeros);
        if let Some(b) = self.batch.as_mut() {
            b.fframe = Some(FFrame {
                xbuf,
                gsnap: None,
                zeros,
                nw128,
                pnr,

                pnrh,
            });
        }
        Ok(())
    }

    /// GDN 상태 전체 스냅샷(스펙 검증 전 — 롤백 보험, plans/121).
    pub fn gdn_state_snapshot(&mut self) -> Result<(), String> {
        // 지연 할당(첫 스냅샷 시 151MB — 일반 프리필은 미할당).
        let need_alloc = self
            .batch
            .as_ref()
            .and_then(|b| b.fframe.as_ref())
            .is_some_and(|f| f.gsnap.is_none());
        if need_alloc
            && let Some(gsnap) = self.ctx.alloc_host_cached(n_gdn_bytes()).ok()
            && let Some(b) = self.batch.as_mut()
            && let Some(f) = b.fframe.as_mut()
        {
            f.gsnap = Some(gsnap);
        }
        let dst = {
            let b = self.batch.as_ref().ok_or("batch")?;
            let f = b.fframe.as_ref().ok_or("fframe")?;
            f.gsnap.as_ref().ok_or("gsnap")?.buf
        };
        let (gs, gr, gn) = {
            let b = self.batch.as_ref().ok_or("batch")?;
            let g = b.gframe.as_ref().ok_or("gframe")?;
            (g.gstate.buf, g.gring.buf, g.gstate.bytes)
        };
        let rn = 48 * 3 * 10240 * 4;
        self.ctx.copy_dev(&[
            (gs, 0, dst, 0, gn as u64),
            (gr, 0, dst, gn as u64, rn as u64),
        ])
    }

    /// 스냅샷 복원(발산 라운드 — kvc는 재실행이 정확히 덮으므로 미복원).
    pub fn gdn_state_restore(&mut self) -> Result<(), String> {
        let dst = {
            let b = self.batch.as_ref().ok_or("batch")?;
            let f = b.fframe.as_ref().ok_or("fframe")?;
            f.gsnap.as_ref().ok_or("gsnap")?.buf
        };
        let (gs, gr, gn) = {
            let b = self.batch.as_ref().ok_or("batch")?;
            let g = b.gframe.as_ref().ok_or("gframe")?;
            (g.gstate.buf, g.gring.buf, g.gstate.bytes)
        };
        let rn = 48 * 3 * 10240 * 4;
        // 영역 소스: gstate←gsnap+0, gring←gsnap+gn(4703aa3 정리가 첫 소스를
        // gn 오프셋으로 잘못 바꿔 gstate에 gring 복사+OOB → DEVICE_LOST였음).
        self.ctx.copy_dev(&[
            (dst, 0, gs, 0, gn as u64),
            (dst, gn as u64, gr, 0, rn as u64),
        ])
    }

    /// 잔차 버퍼 포인터 — 호출자가 임베딩 행을 직접 기록한다.
    pub fn frame_x_ptr(&mut self) -> Result<*mut f32, String> {
        self.fframe_init()?;
        Ok(self
            .batch
            .as_ref()
            .and_then(|b| b.fframe.as_ref())
            .ok_or("fframe")?
            .xbuf
            .ptr as *mut f32)
    }

    pub fn frame_x_flush(&mut self, t_rows: usize) -> Result<(), String> {
        let ff = self
            .batch
            .as_ref()
            .and_then(|b| b.fframe.as_ref())
            .ok_or("fframe")?;
        self.ctx.flush_range(&ff.xbuf, t_rows * 5120 * 4);
        Ok(())
    }

    /// norm_resid 디스패치: xn=norm(x+ab)·w[row] → xtb, x+=ab → xbuf 제자리.
    /// ab에 zeros를 주면 사전 전용(잔차 0).
    pub fn frame_norm_resid(
        &mut self,
        w_row: usize,
        t_rows: usize,
        ab: ash::vk::Buffer,
    ) -> Result<(), String> {
        let ff = self
            .batch
            .as_ref()
            .and_then(|b| b.fframe.as_ref())
            .ok_or("fframe")?;
        let xtb = self.batch.as_ref().ok_or("batch")?.xtb.buf;
        let ds = self.ctx.fresh_ds_for(&ff.pnr, 4)?;
        self.ctx
            .bind_bufs(ds, &[ff.xbuf.buf, ff.nw128.buf, ab, xtb]);
        let push: Vec<u8> = [t_rows as u32, (w_row * 5120) as u32]
            .iter()
            .flat_map(|x| x.to_le_bytes())
            .collect();
        crate::rawvk::context::site::set_tag("e3_norm_resid");
        self.ctx.run_rw(
            ff.pnr.pl,
            ds,
            ff.pnr.pipe,
            &push,
            t_rows as u32,
            1,
            1,
            &[ff.nw128.buf, ab],
            &[xtb, ff.xbuf.buf],
        )?;
        Ok(())
    }

    /// 노름 전체 [129][5120](hip 프로브용).
    pub fn norms_full_dump(&mut self) -> Result<Vec<f32>, String> {
        self.fframe_init()?;
        let ff = self
            .batch
            .as_ref()
            .and_then(|b| b.fframe.as_ref())
            .ok_or("fframe")?;
        Ok(unsafe { std::slice::from_raw_parts(ff.nw128.ptr as *const f32, 129 * 5120).to_vec() })
    }

    /// 선형 키 전체(hip 프로브용).
    pub fn linear_keys(&self) -> Vec<String> {
        self.linears.iter().map(|(k, _)| k.clone()).collect()
    }

    /// 어텐션 q/k_norm 가중치(hip 프로브용) — 12층분 각 [256].
    pub fn attn_norms_dump(&mut self) -> Result<(Vec<f32>, Vec<f32>), String> {
        let mut q = Vec::with_capacity(16 * 256);
        let mut k = Vec::with_capacity(16 * 256);
        for ai in 0..16usize {
            let il = ai * 4 + 3;
            let lp = format!("model.language_model.layers.{il}");
            q.extend(
                self.norm(&format!("{lp}.self_attn.q_norm.weight"))
                    .ok_or("qnw")?,
            );
            k.extend(
                self.norm(&format!("{lp}.self_attn.k_norm.weight"))
                    .ok_or("knw")?,
            );
        }
        Ok((q, k))
    }

    /// fframe nw128 판독(hip 프로브용).
    pub fn nw128_dump(&mut self) -> Result<Vec<f32>, String> {
        self.fframe_init()?;
        let ff = self
            .batch
            .as_ref()
            .and_then(|b| b.fframe.as_ref())
            .ok_or("fframe")?;
        Ok(unsafe { std::slice::from_raw_parts(ff.nw128.ptr as *const f32, 5120).to_vec() })
    }

    /// 선형 원시 트레이리던트 덤프(k, n, krate, suh·tre·svh 바이트) — hip 프로브.
    pub fn linear_raw(
        &mut self,
        key: &str,
    ) -> Result<(usize, usize, u32, Vec<u8>, Vec<u8>, Vec<u8>), String> {
        let li = self.find_linear(key)?;
        let l = &self.linears[li].1;
        unsafe {
            let rd = |b: &crate::rawvk::context::VkBuf, n: usize| {
                std::slice::from_raw_parts(b.ptr as *const u8, n).to_vec()
            };
            Ok((
                l.k,
                l.n,
                l.krate,
                rd(&l.suh, l.k * 2), // f16쌍팩 [k/2]u32 = k half = 2k 바이트
                rd(&l.tre, l.tre.bytes),
                rd(&l.svh, l.n * 2),
            ))
        }
    }

    /// 선형 suh 버퍼 조회 — norm_resid_had 부착부에서 소비 suh 지정용.
    pub fn suh_of(&mut self, key: &str) -> Result<ash::vk::Buffer, String> {
        let li = self.find_linear(key)?;
        Ok(self.linears[li].1.suh.buf)
    }

    /// 융합 norm_resid + had 2종(plans/121 메가융합 1호): xtb·xbuf 갱신에 더해
    /// ah[0]=WHT(xn⊙suh1), ah[1]=WHT(xn⊙suh2)까지 1디스패치로 — 소비 2선형의
    /// had_in 흡수(-2디스패치/층, 배리어 드레인 절감).
    pub fn frame_norm_resid_had(
        &mut self,
        w_row: usize,
        t_rows: usize,
        ab: ash::vk::Buffer,
        suh1: ash::vk::Buffer,
        suh2: ash::vk::Buffer,
    ) -> Result<(), String> {
        let ff = self
            .batch
            .as_ref()
            .and_then(|b| b.fframe.as_ref())
            .ok_or("fframe")?;
        let b = self.batch.as_ref().ok_or("batch")?;
        let (xtb, ah0, ah1) = (b.xtb.buf, b.ah[0].buf, b.ah[1].buf);
        let ds = self.ctx.fresh_ds_for(&ff.pnrh, 8)?;
        self.ctx.bind_bufs(
            ds,
            &[ff.xbuf.buf, ff.nw128.buf, ab, xtb, suh1, suh2, ah0, ah1],
        );
        let push: Vec<u8> = [t_rows as u32, (w_row * 5120) as u32]
            .iter()
            .flat_map(|x| x.to_le_bytes())
            .collect();
        crate::rawvk::context::site::set_tag("e3_norm_resid_had");
        self.ctx.run_rw(
            ff.pnrh.pl,
            ds,
            ff.pnrh.pipe,
            &push,
            t_rows as u32,
            1,
            1,
            &[ff.nw128.buf, ab, suh1, suh2],
            &[xtb, ff.xbuf.buf, ah0, ah1],
        )?;
        Ok(())
    }

    /// 듀얼 GEMM(메가융합 3호): ah0/ah1(선행 norm_had)을 단일 gemm2d로
    /// 병합 환원 — GEMM 커널 수 절반. 결과 yb[0]/yb[1].
    pub fn linear_pair_dual(
        &mut self,
        keys: [&str; 2],
        t_rows: usize,
    ) -> Result<[(ash::vk::Buffer, usize); 2], String> {
        let i1 = self.find_linear(keys[0])?;
        let i2 = self.find_linear(keys[1])?;
        let (k1, n1, kr1, kt1) = {
            let l = &self.linears[i1].1;
            (l.k, l.n, l.krate, l.n / 16)
        };
        let (k2, n2, kt2) = {
            let l = &self.linears[i2].1;
            (l.k, l.n, l.n / 16)
        };
        if k1 != k2 || kt1 % 4 != 0 || kt2 % 4 != 0 {
            return Err(format!(
                "linear_pair_dual: {}/{} k 불일치 또는 n 비64배수",
                keys[0], keys[1]
            ));
        }
        let (kr2v, n2b) = {
            let l = &self.linears[i2].1;
            (l.krate, l.n)
        };
        let _ = n2b;
        // 혼합정밀 아카이브(8/48층 qkv r=4 vs z r=5/3 — gemmd L5 재현 8.6e0):
        // krate 불일치 쌍은 듀얈 단일-K 디코드 불가 → preah 2체인 폴백.
        if kr1 != kr2v {
            self.ensure_batch()?;
            self.ctx.begin_batch()?;
            for (slot, li) in [(0usize, i1), (1usize, i2)] {
                let (ah, yb) = {
                    let b = self.batch.as_ref().ok_or("batch")?;
                    (b.ah[slot].buf, b.yb[slot].buf)
                };
                self.chain_gemmonly(li, t_rows as u32, ah, yb)?;
            }
            self.ctx.end_batch_wait()?;
            self.ctx.wait_pending()?;
            let b = self.batch.as_ref().ok_or("batch")?;
            return Ok([(b.yb[0].buf, n1), (b.yb[1].buf, n2)]);
        }
        self.ensure_batch()?;
        self.ctx.begin_batch()?;
        {
            let b = self.batch.as_ref().ok_or("batch")?;
            let (ah0, ah1, tre1, tre2, sb) = (
                b.ah[0].buf,
                b.ah[1].buf,
                self.linears[i1].1.tre.buf,
                self.linears[i2].1.tre.buf,
                b.sb.buf,
            );
            let kk = self.linears[i1].1.krate;
            let ds = self.ctx.fresh_ds_for(&b.p2d, 5)?;
            self.ctx.bind_bufs(ds, &[ah0, tre1, ah1, tre2, sb]);
            let push: Vec<u8> = [(k1 / 16) as u32, kt1 as u32, kt2 as u32, kk, t_rows as u32]
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect();
            crate::rawvk::context::site::set_tag("e3_gemm2d");
            self.ctx.run_rw(
                b.p2d.pl,
                ds,
                b.p2d.pipe,
                &push,
                ((n1 + n2) / 64) as u32,
                t_rows.div_ceil(64) as u32,
                1,
                &[ah0, tre1, ah1, tre2],
                &[sb],
            )?;
        }
        // had_out_td 2회 — n_off로 슬래브 분리.
        for (slot, li, n_off, ntiles) in [(0usize, i1, 0usize, kt1), (1usize, i2, n1, kt2)] {
            let b = self.batch.as_ref().ok_or("batch")?;
            let (sb, svh, yb) = (b.sb.buf, self.linears[li].1.svh.buf, b.yb[slot].buf);
            let ds = self.ctx.fresh_ds_for(&b.p3d, 3)?;
            self.ctx.bind_bufs(ds, &[sb, svh, yb]);
            let n_this = ntiles * 16;
            let push: Vec<u8> = [
                (n_this / 128) as u32,
                1u32,
                ((kt1 + kt2) * 16) as u32,
                n_off as u32,
                n_this as u32,
            ]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
            crate::rawvk::context::site::set_tag("e3_had_out_td");
            self.ctx.run_rw(
                b.p3d.pl,
                ds,
                b.p3d.pipe,
                &push,
                (n_this / 128) as u32,
                t_rows as u32,
                1,
                &[sb, svh],
                &[yb],
            )?;
        }
        self.ctx.end_batch_wait()?;
        self.ctx.wait_pending()?;
        let b = self.batch.as_ref().ok_or("batch")?;
        Ok([(b.yb[0].buf, n1), (b.yb[1].buf, n2)])
    }

    /// ah 사전 기록 전제 2선형 배치(gemm2+had_out만) — norm_resid_had 소비용.
    pub fn linear_pair_preah(
        &mut self,
        keys: &[&str],
        t_rows: usize,
    ) -> Result<Vec<(ash::vk::Buffer, usize)>, String> {
        if keys.len() != 2 {
            return Err(format!("linear_pair_preah: keys {}개 (2 고정)", keys.len()));
        }
        let mut idxs = Vec::with_capacity(2);
        for key in keys {
            idxs.push(self.find_linear(key)?);
        }
        self.ensure_batch()?;
        self.ctx.begin_batch()?;
        for (slot, &li) in idxs.iter().enumerate() {
            let (ah, yb) = {
                let b = self.batch.as_ref().ok_or("batch")?;
                (b.ah[slot].buf, b.yb[slot].buf)
            };
            self.chain_gemmonly(li, t_rows as u32, ah, yb)?;
        }
        self.ctx.end_batch_wait()?;
        self.ctx.wait_pending()?;
        let b = self.batch.as_ref().ok_or("batch")?;
        Ok(idxs
            .iter()
            .enumerate()
            .map(|(slot, &li)| (b.yb[slot].buf, self.linears[li].1.n))
            .collect())
    }

    pub fn frame_zeros_buf(&mut self) -> Result<ash::vk::Buffer, String> {
        Ok(self
            .batch
            .as_ref()
            .and_then(|b| b.fframe.as_ref())
            .ok_or("fframe")?
            .zeros
            .buf)
    }

    /// 마지막 행 판독(로그릿용) — end_outer 후 호출.
    pub fn frame_read_xtb_row(&mut self, t_rows: usize) -> Result<Vec<f32>, String> {
        let b = self.batch.as_ref().ok_or("batch")?;
        let base = (t_rows - 1) * 5120;
        self.ctx.invalidate_range_at(&b.xtb, base * 4, 5120 * 4);
        let p = unsafe {
            std::slice::from_raw_parts(b.xtb.ptr.add(base * 4) as *const f32, 5120).to_vec()
        };
        Ok(p)
    }

    /// 디버그: xbuf 선두 행 판독(잔차 검증).
    pub fn debug_xbuf_row(&mut self) -> Result<Vec<f32>, String> {
        let ff = self
            .batch
            .as_ref()
            .and_then(|b| b.fframe.as_ref())
            .ok_or("fframe")?;
        self.ctx.invalidate_range_at(&ff.xbuf, 0, 5120 * 4);
        Ok(unsafe { std::slice::from_raw_parts(ff.xbuf.ptr as *const f32, 5120).to_vec() })
    }

    /// 디버그: xtb 선두 행 판독(FFN 입력 xn 검증).
    pub fn debug_xtb_row(&mut self) -> Result<Vec<f32>, String> {
        let b = self.batch.as_ref().ok_or("batch")?;
        self.ctx.invalidate_range_at(&b.xtb, 0, 5120 * 4);
        Ok(unsafe { std::slice::from_raw_parts(b.xtb.ptr as *const f32, 5120).to_vec() })
    }

    /// 디버그: yb[2] 선두 행 판독(FFN out row0 검증).
    pub fn debug_yb2_row(&mut self) -> Result<Vec<f32>, String> {
        let b = self.batch.as_ref().ok_or("batch")?;
        self.ctx.invalidate_range_at(&b.yb[2], 0, 5120 * 4);
        Ok(unsafe { std::slice::from_raw_parts(b.yb[2].ptr as *const f32, 5120).to_vec() })
    }

    /// 디버그: yb[0] 선두 행 판독(L0 out row0 검증).
    pub fn debug_yb0_row(&mut self, t_rows: usize) -> Result<Vec<f32>, String> {
        let b = self.batch.as_ref().ok_or("batch")?;
        let base = (t_rows - 1) * 5120;
        self.ctx.invalidate_range_at(&b.yb[0], base * 4, 5120 * 4);
        Ok(unsafe {
            std::slice::from_raw_parts(b.yb[0].ptr.add(base * 4) as *const f32, 5120).to_vec()
        })
    }

    /// 열린 외부 배치 내부용 선형 체인(begin/end 없음 — 프레임 내 lm_head 등).
    pub fn linear_chain_inside(
        &mut self,
        key: &str,
        t_rows: usize,
        slot: usize,
    ) -> Result<(ash::vk::Buffer, usize), String> {
        let li = self.find_linear(key)?;
        let n = self.linears[li].1.n;
        self.ensure_batch()?;
        let (xtb, ah0, yb0) = {
            let b = self.batch.as_ref().ok_or("batch scratch")?;
            (b.xtb.buf, b.ah[slot].buf, b.yb[slot].buf)
        };
        self.chain_batch_one(xtb, li, t_rows as u32, ah0, yb0, true)?;
        Ok((yb0, n))
    }

    /// yb 슬롯에서 t_rows×n 판독(스펙 행별 로짓).
    pub fn read_yb_rows(
        &mut self,
        slot: usize,
        t_rows: usize,
        n: usize,
    ) -> Result<Vec<f32>, String> {
        let b = self.batch.as_ref().ok_or("batch")?;
        self.ctx.invalidate_range(&b.yb[slot], t_rows * n * 4);
        let mut y = vec![0f32; t_rows * n];
        // SAFETY: end_outer 후 매핑 판독.
        unsafe {
            std::ptr::copy_nonoverlapping(b.yb[slot].ptr as *const f32, y.as_mut_ptr(), t_rows * n);
        }
        Ok(y)
    }

    /// xtb 선두 t_rows행 판독(스펙 행별 노름 입력).
    pub fn read_xtb_rows(&mut self, t_rows: usize) -> Result<Vec<f32>, String> {
        let b = self.batch.as_ref().ok_or("batch")?;
        self.ctx.invalidate_range(&b.xtb, t_rows * 5120 * 4);
        let mut y = vec![0f32; t_rows * 5120];
        // SAFETY: end_outer 후 매핑 판독.
        unsafe {
            std::ptr::copy_nonoverlapping(b.xtb.ptr as *const f32, y.as_mut_ptr(), t_rows * 5120);
        }
        Ok(y)
    }

    /// xbuf 마지막 행 판독(MTP h 스냅샷) — GPU 갱신 반영(invalidate).
    pub fn frame_read_x_last(&mut self, t_rows: usize) -> Result<Vec<f32>, String> {
        let ff = self
            .batch
            .as_ref()
            .and_then(|b| b.fframe.as_ref())
            .ok_or("fframe")?;
        let base = (t_rows - 1) * 5120;
        self.ctx.invalidate_range_at(&ff.xbuf, base * 4, 5120 * 4);
        Ok(unsafe {
            std::slice::from_raw_parts(ff.xbuf.ptr.add(base * 4) as *const f32, 5120).to_vec()
        })
    }

    /// 단일 선형 체인(판독 없음·flush 없음 — xtb가 GPU 기록 전제, plans/121 프레임).
    /// 결과는 yb[0]에 잔류: (버퍼, n) 반환.
    pub fn linear_chain(
        &mut self,
        key: &str,
        t_rows: usize,
        slot: usize,
    ) -> Result<(ash::vk::Buffer, usize), String> {
        let li = self.find_linear(key)?;
        let n = self.linears[li].1.n;
        self.ensure_batch()?;
        let (xtb, ah0, yb0) = {
            let b = self.batch.as_ref().ok_or("batch scratch")?;
            (b.xtb.buf, b.ah[slot].buf, b.yb[slot].buf)
        };
        self.ctx.begin_batch()?;
        self.chain_batch_one(xtb, li, t_rows as u32, ah0, yb0, true)?;
        self.ctx.end_batch_wait()?;
        self.ctx.wait_pending()?;
        Ok((yb0, n))
    }

    /// FFN 트리오 체인(판독 없음) — 결과 yb[0] 잔류(주의: gate/up이 ah/yb 슬롯
    /// 을 재사용하므로 down은 yb[2]가 아니라 ffn_trio의 슬롯 배정을 그대로
    /// 둔다 — 여기선 down 결과를 yb[0]으로 재배치한다).
    pub fn ffn_trio_chain(
        &mut self,
        key_g: &str,
        key_u: &str,
        key_d: &str,
        t_rows: usize,
    ) -> Result<(ash::vk::Buffer, usize), String> {
        let id = self.find_linear(key_d)?;
        let nd = self.linears[id].1.n;
        self.ffn_trio_impl(key_g, key_u, key_d, t_rows, true, false)?;
        let b = self.batch.as_ref().ok_or("batch")?;
        Ok((b.yb[2].buf, nd))
    }

    /// FFN 트리오 preah 변형(메가융합 2호) — ah0/ah1은 선행 norm_resid_had가
    /// gate/up suh로 기록. down 체인(f16 leg)은 불변.
    pub fn ffn_trio_preah(
        &mut self,
        key_g: &str,
        key_u: &str,
        key_d: &str,
        t_rows: usize,
    ) -> Result<(ash::vk::Buffer, usize), String> {
        let id = self.find_linear(key_d)?;
        let nd = self.linears[id].1.n;
        self.ffn_trio_impl(key_g, key_u, key_d, t_rows, true, true)?;
        let b = self.batch.as_ref().ok_or("batch")?;
        Ok((b.yb[2].buf, nd))
    }
}
// 마커 pbuf
// 마커 28a
// 마커 restfx
// 마커 ch3
// 마커 mid1
// 마커 go1
// 마커 gqgv
// 마커 gq3
// 마커 gk1
// 마커 nrh1
// 마커 nrh2
// 마커 nrhdbg
// 마커 suhf
// 마커 xtbm
// 마커 nrht
// 마커 dual1
// 마커 dual2
// 마커 dual3
// 마커 gemmd
// 마커 t60
// 마커 t60b
// 마커 t60c
// 마커 kaud
// 마커 kr2
// 마커 l5
// 마커 l5b
// 마커 fd
// 마커 sep1
