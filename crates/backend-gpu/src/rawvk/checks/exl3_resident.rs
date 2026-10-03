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
    batch: Option<BatchScratch>,
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

    fn find_linear(&self, key: &str) -> Result<usize, String> {
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

/// T-배치 스크래치 — 슬롯 3종(GEMV ahb/yb 관례와 동일 구조, run_rw
/// 배리어가 WAR/WAW 커버 — plans/104 판정식, sb는 단일 공유).
pub struct BatchScratch {
    /// [TMAX*max_k] f32 캐시 — 스테이징 입력(호스트가 병렬 직접 기록,
    /// had_in_tf32가 직독 — CPU f16 변환 제거, 병목 원장 #3).
    xtb: VkBuf,
    ah: [VkBuf; 3],
    yb: [VkBuf; 3], // [TMAX*max_n] f32 ×3
    sb: VkBuf,      // [TMAX*NSEG*max_n] f32
    /// [TMAX*max_n] f16 캐브아웃 — ew_t 출력(= down의 had_in_t f16 입력).
    x2t: VkBuf,
    p1: Pipes,  // had_in_t (f16 입력 — ew_t 다운 레그)
    p1f: Pipes, // had_in_tf32 (f32 직독)
    p2: Pipes,  // gemm
    p3: Pipes,  // had_out_t
    p4t: Pipes, // ffn_ew_t
    /// GDN 프레임(plans/121 F1) — GPU 상주 비선형 체인(지연 초기화).
    gframe: Option<GdnFrame>,
}

/// GDN 프레임 버퍼+파이프라인(plans/121 F1).
pub struct GdnFrame {
    gq: VkBuf,     // [TMAX*2048] f32 L2 norm q
    gk: VkBuf,     // [TMAX*2048] f32 L2 norm k
    gv: VkBuf,     // [TMAX*6144] f32 v lc
    gbg: VkBuf,    // [TMAX*96] f32 beta|g lc
    go: VkBuf,     // [TMAX*6144] f32 o lc
    gqr: VkBuf,    // [TMAX*2048] f32 conv q raw(HF)
    gkr: VkBuf,    // [TMAX*2048] f32 conv k raw(HF)
    gvr: VkBuf,    // [TMAX*6144] f32 conv v raw(HF)
    ab: VkBuf,     // [n_gdn][2*48*5120] f32
    cw: VkBuf,     // [n_gdn][10240*4] f32
    alog: VkBuf,   // [n_gdn*48] f32
    dtb: VkBuf,    // [n_gdn*48] f32
    nw: VkBuf,     // [n_gdn*128] f32
    gring: VkBuf,  // [n_gdn*3*10240] f32
    gstate: VkBuf, // [n_gdn*48*16384] f32
    pgc: Pipes,
    pgl: Pipes,
    pgs: Pipes,
    pgg: Pipes,
}

impl TrellisResident {
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
            xtb,
            ah: [ah1, ah2, ah3],
            yb: [y1, y2, y3],
            sb,
            x2t,
            p1,
            p1f,
            p2,
            p3,
            p4t,
            gframe: None,
        });
        Ok(())
    }

    /// 배치 선형 1개 체인(had_in→gemm→had_out_t) — begin_batch 내부 전용.
    /// f32_in=true: x_src를 f32 [T][k]로 직독(had_in_tf32) — 스테이징 경로.
    /// false: f16 쌍팩(ew_t 출력 → down 레그).
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
        {
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
        self.chain_batch_one(xtb, ig, t_rows as u32, ah0, yb0, true)?;
        self.chain_batch_one(xtb, iu, t_rows as u32, ah1, yb1, true)?;
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
            && let Some(path) = llm170_diag::flag::val("LLM170_SCAN_CAP").map(|s| s.to_string())
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

// ── scan 모듈 독립 프로브(plans/121 F2) ──
// 모델 적재 없이 합성 입력으로 scan 커널만 검증: 속도·산술 격리 작업장.
// Rust f32 기준(커널 수식 미러)과 행별 출력·최종 상태를 직접 비교한다.
pub fn scan_check(t_len: usize, cap_path: &str) -> Result<String, String> {
    const HK: usize = 16;
    const HV: usize = 48;
    const D: usize = 128;
    let use_cap = !cap_path.is_empty();
    if use_cap {
        let raw = std::fs::read(cap_path).map_err(|e| format!("cap read: {e}"))?;
        let nf = raw.len() / 4;
        let fl: Vec<f32> =
            unsafe { std::slice::from_raw_parts(raw.as_ptr() as *const f32, nf) }.to_vec();
        let n_q = t_len * 2048;
        let mut o = 0usize;
        let take = |o: &mut usize, n: usize| -> Vec<f32> {
            let v = fl[*o..*o + n].to_vec();
            *o += n;
            v
        };
        let q = take(&mut o, n_q);
        let k = take(&mut o, n_q);
        let v = take(&mut o, t_len * 6144);
        let bg = take(&mut o, t_len * 96);
        return scan_check_run(q, k, v, bg, t_len, true);
    }
    // 합성 입력(LCG) — q/k는 L2 정규화 후 스케일(≈1/√128), beta∈(0,1), g=음수.
    let mut seed: u32 = 0x1234_5678;
    let mut rnd = || {
        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        (seed >> 8) as f32 / 16_777_216.0
    };
    let mut q = vec![0f32; t_len * HK * D];
    let mut k = vec![0f32; t_len * HK * D];
    let mut v = vec![0f32; t_len * HV * D];
    let mut bg = vec![0f32; t_len * 2 * HV];
    for e in q.iter_mut() {
        *e = (rnd() * 2.0 - 1.0) * 0.09;
    }
    for e in k.iter_mut() {
        *e = (rnd() * 2.0 - 1.0) * 0.09;
    }
    for e in v.iter_mut() {
        *e = (rnd() * 2.0 - 1.0) * 0.5;
    }
    for t in 0..t_len {
        for h in 0..HV {
            bg[t * 2 * HV + h] = rnd();
            bg[t * 2 * HV + HV + h] = -rnd() * 2.0;
        }
    }

    scan_check_run(q, k, v, bg, t_len, false)
}

fn scan_check_run(
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    bg: Vec<f32>,
    t_len: usize,
    from_cap: bool,
) -> Result<String, String> {
    let mut ctx = crate::rawvk::context::VkCtx::new()?;
    const HK: usize = 16;
    const HV: usize = 48;
    const D: usize = 128;
    let gq = ctx.alloc_host_cached(t_len.max(64) * HK * D * 4)?;
    let gk = ctx.alloc_host_cached(t_len.max(64) * HK * D * 4)?;
    let gv = ctx.alloc_host_cached(t_len.max(64) * HV * D * 4)?;
    let gbg = ctx.alloc_host_cached(t_len.max(64) * 2 * HV * 4)?;
    let go = ctx.alloc_host_cached(t_len.max(64) * HV * D * 4)?;
    let gstate = ctx.alloc_host_cached(HV * D * D * 4)?; // 1층분
    let st0: Vec<f32> = if llm170_diag::flag::on("LLM170_SCAN_ST0") {
        let mut sd: u32 = 0xC0FF_EE01;
        (0..HV * D * D)
            .map(|_| {
                sd = sd.wrapping_mul(1664525).wrapping_add(1013904223);
                ((sd >> 8) as f32 / 16_777_216.0 - 0.5) * 2.0
            })
            .collect()
    } else {
        vec![0f32; HV * D * D]
    };
    unsafe {
        std::ptr::copy_nonoverlapping(q.as_ptr(), gq.ptr as *mut f32, q.len());
        std::ptr::copy_nonoverlapping(k.as_ptr(), gk.ptr as *mut f32, k.len());
        std::ptr::copy_nonoverlapping(v.as_ptr(), gv.ptr as *mut f32, v.len());
        std::ptr::copy_nonoverlapping(bg.as_ptr(), gbg.ptr as *mut f32, bg.len());
        std::ptr::copy_nonoverlapping(st0.as_ptr(), gstate.ptr as *mut f32, st0.len());
        std::ptr::write_bytes(go.ptr, 0, t_len.max(64) * HV * D * 4);
    }
    ctx.flush_buf(&gq);
    ctx.flush_buf(&gk);
    ctx.flush_buf(&gv);
    ctx.flush_buf(&gbg);
    ctx.flush_buf(&gstate);
    let pgs = ctx.pipeline_pipes(include_bytes!("../spv/exl3_gdn_scan.spv"), 6, 20)?;
    let dispatch = |ctx: &mut crate::rawvk::context::VkCtx| -> Result<(), String> {
        ctx.begin_batch()?;
        let ds = ctx.fresh_ds_for(&pgs, 6)?;
        ctx.bind_bufs(ds, &[gq.buf, gk.buf, gv.buf, gbg.buf, gstate.buf, go.buf]);
        let push: Vec<u8> = [t_len as u32, HK as u32, HV as u32, D as u32, 0u32]
            .iter()
            .flat_map(|x| x.to_le_bytes())
            .collect();
        crate::rawvk::context::site::set_tag("e3_scan_probe");
        ctx.run_rw(
            pgs.pl,
            ds,
            pgs.pipe,
            &push,
            HV as u32,
            1,
            1,
            &[gq.buf, gk.buf, gv.buf, gbg.buf, gstate.buf],
            &[go.buf, gstate.buf],
        )?;
        ctx.end_batch_wait()?;
        ctx.wait_pending()?;
        Ok(())
    };
    dispatch(&mut ctx)?;
    // 시간 측정(5회 중앙값)
    let mut times: Vec<f64> = Vec::new();
    for _ in 0..5 {
        unsafe {
            std::ptr::copy_nonoverlapping(st0.as_ptr(), gstate.ptr as *mut f32, st0.len());
        }
        ctx.flush_buf(&gstate);
        let t0 = std::time::Instant::now();
        dispatch(&mut ctx)?;
        times.push(t0.elapsed().as_secs_f64() * 1000.0);
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());

    // 판독
    ctx.invalidate_buf(&go);
    ctx.invalidate_buf(&gstate);
    let out_gpu: Vec<f32> =
        unsafe { std::slice::from_raw_parts(go.ptr as *const f32, t_len * HV * D).to_vec() };
    let st_gpu: Vec<f32> =
        unsafe { std::slice::from_raw_parts(gstate.ptr as *const f32, HV * D * D).to_vec() };

    // core 기준(gdn_chunk_seq — CPU f32, v1과 동일 경로)
    let mut beta_v = vec![0f32; t_len * HV];
    let mut g_v = vec![0f32; t_len * HV];
    for t in 0..t_len {
        for h in 0..HV {
            beta_v[t * HV + h] = bg[t * 2 * HV + h];
            g_v[t * HV + h] = bg[t * 2 * HV + HV + h];
        }
    }
    let mut st_core = st0.clone();
    let mut out_core = vec![0f32; t_len * HV * D];
    llm170_core::gdn::gdn_chunk_seq(
        &q,
        &k,
        &v,
        &beta_v,
        &g_v,
        &mut st_core,
        &mut out_core,
        t_len,
        HK,
        HV,
    );
    let mut kern_vs_core = 0f32;
    for i in 0..out_gpu.len() {
        kern_vs_core = kern_vs_core.max((out_gpu[i] - out_core[i]).abs());
    }
    let mut stc_max = 0f32;
    for i in 0..st_gpu.len() {
        stc_max = stc_max.max((st_gpu[i] - st_core[i]).abs());
    }

    // Rust f32 기준 — 커널 수식 미러(CS=32)
    let (out_ref, st_ref) = scan_ref(&q, &k, &v, &bg, t_len, false, &st0);
    let (out_ref16, _) = scan_ref(&q, &k, &v, &bg, t_len, true, &st0);
    let mut kern_vs_f16ref = 0f32;
    for i in 0..out_gpu.len() {
        kern_vs_f16ref = kern_vs_f16ref.max((out_gpu[i] - out_ref16[i]).abs());
    }
    let mut mirror_vs_core = 0f32;
    for i in 0..out_ref.len() {
        mirror_vs_core = mirror_vs_core.max((out_ref[i] - out_core[i]).abs());
    }

    let _ = from_cap;
    let mut out_max = 0f32;
    let mut out_rel = 0f64;

    for i in 0..out_gpu.len() {
        let d = (out_gpu[i] - out_ref[i]).abs();
        out_max = out_max.max(d);
        let denom = out_ref[i].abs().max(1e-3);
        out_rel = out_rel.max(d as f64 / denom as f64);
    }
    let mut st_max = 0f32;
    for i in 0..st_gpu.len() {
        st_max = st_max.max((st_gpu[i] - st_ref[i]).abs());
    }
    Ok(format!(
        "scan-check T={t_len}: kern-vs-mirror={out_max:.3e} · kern-vs-f16mirror={kern_vs_f16ref:.3e} · mirror(CS32)-vs-core(CS64)={mirror_vs_core:.3e} · st(kern-vs-core)={stc_max:.3e} · kernel {:.2}ms (5회 중앙값)",
        times[2]
    ))
}

/// scan 커널의 f32 기준 미러 — A/KQ/sk/sv를 f32로 계산(커널의 f16과의 차이가
/// 판정 대상). CS 고정 32.
fn h16(x: f32) -> f32 {
    half::f16::from_f32(x).to_f32()
}

fn scan_ref(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    bg: &[f32],
    t_len: usize,
    f16_emul: bool,
    st0: &[f32],
) -> (Vec<f32>, Vec<f32>) {
    const CS: usize = 32;
    const HK: usize = 16;
    const HV: usize = 48;
    const D: usize = 128;
    let qscale = 1.0f32 / (D as f32).sqrt();
    let mut st = st0.to_vec();
    let mut out = vec![0f32; t_len * HV * D];
    let n_chunks = t_len.div_ceil(CS);
    for c in 0..n_chunks {
        let t0 = c * CS;
        let n = (t_len - t0).min(CS);
        for h in 0..HV {
            let kh = h % HK;
            let mut sk = [[0f32; D]; CS];
            let mut sv = [[0f32; D]; CS];
            let mut bp = [0f32; CS];
            let mut gcs = [0f32; CS + 1];
            for i in 0..CS {
                let live = i < n;
                if live {
                    for s2 in 0..D {
                        let kv2 = k[(t0 + i) * HK * D + kh * D + s2];
                        let vv2 = v[(t0 + i) * HV * D + h * D + s2];
                        sk[i][s2] = if f16_emul { h16(kv2) } else { kv2 };
                        sv[i][s2] = if f16_emul { h16(vv2) } else { vv2 };
                    }
                    bp[i] = bg[(t0 + i) * 2 * HV + h];
                }
            }
            let mut acc = 0f32;
            for t in 0..CS {
                acc += if t < n {
                    bg[(t0 + t) * 2 * HV + HV + h]
                } else {
                    0.0
                };
                gcs[t] = acc;
            }
            gcs[CS] = acc;
            let mut a = [[0f32; CS]; CS];
            let mut kq = [[0f32; CS]; CS];
            for i in 0..n {
                for j in 0..=i {
                    let mut dk = 0f32;
                    let mut dq = 0f32;
                    for s2 in 0..D {
                        dk += sk[i][s2] * sk[j][s2];
                        dq += q[(t0 + i) * HK * D + kh * D + s2] * sk[j][s2];
                    }
                    if j < i {
                        let a2 = dk * bp[i] * (gcs[i] - gcs[j]).exp();
                        a[i][j] = if f16_emul { h16(a2) } else { a2 };
                    }
                    let kq2 = dq * qscale * (gcs[i] - gcs[j]).exp();
                    kq[i][j] = if f16_emul { h16(kq2) } else { kq2 };
                }
            }
            // ks/qs: [CS][D]
            let mut ks = [[0f32; D]; CS];
            let mut qs = [[0f32; D]; CS];
            for i in 0..n {
                for col in 0..D {
                    let mut ak = 0f32;
                    let mut aq = 0f32;
                    for s2 in 0..D {
                        let s_el = st[h * D * D + s2 * D + col];
                        ak += sk[i][s2] * s_el;
                        aq += q[(t0 + i) * HK * D + kh * D + s2] * s_el;
                    }
                    ks[i][col] = ak;
                    qs[i][col] = aq * qscale;
                }
            }
            let mut dc = [[0f32; D]; CS];
            for i in 0..n {
                for col in 0..D {
                    let mut rhs = bp[i] * (sv[i][col] - gcs[i].exp() * ks[i][col]);
                    for j in 0..i {
                        rhs -= a[i][j] * dc[j][col];
                    }
                    dc[i][col] = rhs;
                    let mut oi = gcs[i].exp() * qs[i][col];
                    for p in 0..=i {
                        oi += kq[i][p] * dc[p][col];
                    }
                    out[(t0 + i) * HV * D + h * D + col] = oi;
                }
            }
            let gt_exp = gcs[CS].exp();
            let mut wsm = [0f32; CS];
            for j in 0..CS {
                wsm[j] = if j < n { (gcs[CS] - gcs[j]).exp() } else { 0.0 };
            }
            for s2 in 0..D {
                for col in 0..D {
                    let base = h * D * D + s2 * D + col;
                    let mut a2 = st[base] * gt_exp;
                    for j in 0..n {
                        a2 += sk[j][s2] * wsm[j] * dc[j][col];
                    }
                    st[base] = a2;
                }
            }
        }
    }
    (out, st)
}

// ── 어텐션 모듈 독립 프로브(plans/121 F2b) ──
// 합성 q‖gate/k/v + 규격 노름으로 prep+fwd 2커널만 검증: 속도·산술 격리 작업장.
// Rust 미러는 core::ops::rope_head를 직접 재사용(수학 단일 진실 공급원).
pub fn attn_check(t_len: usize, pos0: usize) -> Result<String, String> {
    use crate::rawvk::context::VkCtx;
    const NH: usize = 24;
    const NKV: usize = 4;
    const D: usize = 256;
    let mut ctx = VkCtx::new()?;

    let mut seed: u32 = 0xBEEF_5A17;
    let mut rnd = || {
        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        (seed >> 8) as f32 / 16_777_216.0
    };
    let mut qg = vec![0f32; t_len * NH * D * 2];
    let mut kin = vec![0f32; t_len * NKV * D];
    let mut vin = vec![0f32; t_len * NKV * D];
    let mut qnw = vec![0f32; D];
    let mut knw = vec![0f32; D];
    for e in qg.iter_mut() {
        *e = (rnd() * 2.0 - 1.0) * 0.5;
    }
    for e in kin.iter_mut() {
        *e = (rnd() * 2.0 - 1.0) * 0.3;
    }
    for e in vin.iter_mut() {
        *e = (rnd() * 2.0 - 1.0) * 0.8;
    }
    for e in qnw.iter_mut() {
        *e = 0.9 + rnd() * 0.2;
    }
    for e in knw.iter_mut() {
        *e = 0.9 + rnd() * 0.2;
    }

    let cap = pos0 + t_len + 8;
    let b_qg = ctx.alloc_host_cached(t_len.max(64) * NH * D * 2 * 4)?;
    let b_k = ctx.alloc_host_cached(t_len.max(64) * NKV * D * 4)?;
    let b_v = ctx.alloc_host_cached(t_len.max(64) * NKV * D * 4)?;
    let b_qnw = ctx.alloc_host_cached(D * 4)?;
    let b_knw = ctx.alloc_host_cached(D * 4)?;
    let b_qh = ctx.alloc_host_cached(t_len.max(64) * NH * D * 4)?;
    let b_kc = ctx.alloc_host_cached(cap * NKV * D * 4)?;
    let b_vc = ctx.alloc_host_cached(cap * NKV * D * 4)?;
    let b_out = ctx.alloc_host_cached(t_len.max(64) * NH * D * 4)?;
    unsafe {
        std::ptr::copy_nonoverlapping(qg.as_ptr(), b_qg.ptr as *mut f32, qg.len());
        std::ptr::copy_nonoverlapping(kin.as_ptr(), b_k.ptr as *mut f32, kin.len());
        std::ptr::copy_nonoverlapping(vin.as_ptr(), b_v.ptr as *mut f32, vin.len());
        std::ptr::copy_nonoverlapping(qnw.as_ptr(), b_qnw.ptr as *mut f32, D);
        std::ptr::copy_nonoverlapping(knw.as_ptr(), b_knw.ptr as *mut f32, D);
        std::ptr::write_bytes(b_kc.ptr, 0, cap * NKV * D * 4);
        std::ptr::write_bytes(b_vc.ptr, 0, cap * NKV * D * 4);
        std::ptr::write_bytes(b_out.ptr, 0, t_len.max(64) * NH * D * 4);
    }
    for b in [&b_qg, &b_k, &b_v, &b_qnw, &b_knw] {
        ctx.flush_buf(b);
    }
    let pp = ctx.pipeline_pipes(include_bytes!("../spv/exl3_attn_prep.spv"), 8, 8)?;
    let pf = ctx.pipeline_pipes(include_bytes!("../spv/exl3_attn_fwd2.spv"), 5, 8)?;

    let run = |ctx: &mut VkCtx| -> Result<(), String> {
        ctx.begin_batch()?;
        let d1 = ctx.fresh_ds_for(&pp, 8)?;
        ctx.bind_bufs(
            d1,
            &[
                b_qg.buf, b_k.buf, b_v.buf, b_qnw.buf, b_knw.buf, b_qh.buf, b_kc.buf, b_vc.buf,
            ],
        );
        let push1: Vec<u8> = [t_len as u32, pos0 as u32]
            .iter()
            .flat_map(|x| x.to_le_bytes())
            .collect();
        crate::rawvk::context::site::set_tag("e3_attn_prep");
        ctx.run_rw(
            pp.pl,
            d1,
            pp.pipe,
            &push1,
            t_len as u32,
            28,
            1,
            &[b_qg.buf, b_k.buf, b_v.buf],
            &[b_qh.buf, b_kc.buf, b_vc.buf],
        )?;
        let d2 = ctx.fresh_ds_for(&pf, 5)?;
        ctx.bind_bufs(d2, &[b_qh.buf, b_kc.buf, b_vc.buf, b_qg.buf, b_out.buf]);
        let push2: Vec<u8> = [t_len as u32, pos0 as u32]
            .iter()
            .flat_map(|x| x.to_le_bytes())
            .collect();
        crate::rawvk::context::site::set_tag("e3_attn_fwd");
        ctx.run_rw(
            pf.pl,
            d2,
            pf.pipe,
            &push2,
            t_len as u32,
            24,
            1,
            &[b_qh.buf, b_kc.buf, b_vc.buf, b_qg.buf],
            &[b_out.buf],
        )?;
        ctx.end_batch_wait()?;
        ctx.wait_pending()?;
        Ok(())
    };
    run(&mut ctx)?;
    let mut times: Vec<f64> = Vec::new();
    for _ in 0..5 {
        unsafe {
            std::ptr::write_bytes(b_kc.ptr, 0, cap * NKV * D * 4);
            std::ptr::write_bytes(b_vc.ptr, 0, cap * NKV * D * 4);
        }
        ctx.flush_buf(&b_kc);
        ctx.flush_buf(&b_vc);
        let t0 = std::time::Instant::now();
        run(&mut ctx)?;
        times.push(t0.elapsed().as_secs_f64() * 1000.0);
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    ctx.invalidate_buf(&b_out);
    let out_gpu: Vec<f32> =
        unsafe { std::slice::from_raw_parts(b_out.ptr as *const f32, t_len * NH * D).to_vec() };

    ctx.invalidate_buf(&b_qh);
    ctx.invalidate_buf(&b_kc);
    let qh_gpu: Vec<f32> =
        unsafe { std::slice::from_raw_parts(b_qh.ptr as *const f32, t_len * NH * D).to_vec() };
    let kc_gpu: Vec<f32> = unsafe {
        std::slice::from_raw_parts(b_kc.ptr as *const f32, (pos0 + t_len) * NKV * D).to_vec()
    };
    let (qh_ref, kc_ref, out_ref) = attn_ref2(&qg, &kin, &vin, &qnw, &knw, t_len, pos0);
    let mut qh_max = 0f32;
    for i in 0..qh_gpu.len() {
        qh_max = qh_max.max((qh_gpu[i] - qh_ref[i]).abs());
    }
    let mut kc_max = 0f32;
    for i in 0..kc_gpu.len() {
        kc_max = kc_max.max((kc_gpu[i] - kc_ref[i]).abs());
    }
    eprintln!("  [attndbg] qh maxdiff={qh_max:.3e} kc maxdiff={kc_max:.3e}");
    let mut out_max = 0f32;
    let mut out_rel = 0f64;
    for i in 0..out_gpu.len() {
        let d = (out_gpu[i] - out_ref[i]).abs();
        out_max = out_max.max(d);
        let denom = out_ref[i].abs().max(1e-3);
        out_rel = out_rel.max(d as f64 / denom as f64);
    }
    Ok(format!(
        "attn-check T={t_len} pos0={pos0}: maxdiff={out_max:.3e} rel={out_rel:.3e} · {times:.2?}ms"
    ))
}

fn attn_ref2(
    qg: &[f32],
    kin: &[f32],
    vin: &[f32],
    qnw: &[f32],
    knw: &[f32],
    t_len: usize,
    pos0: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    const NH: usize = 24;
    const NKV: usize = 4;
    const D: usize = 256;
    let mut out = vec![0f32; t_len * NH * D];
    let mut qh_all = vec![0f32; t_len * NH * D];
    let mut kcache = vec![0f32; (pos0 + t_len) * NKV * D];
    let mut vcache = vec![0f32; (pos0 + t_len) * NKV * D];
    for t in 0..t_len {
        let pos = pos0 + t;
        for hh in 0..NKV {
            let src = t * NKV * D + hh * D;
            let mut head: Vec<f32> = kin[src..src + D].to_vec();
            let ss: f32 = head.iter().map(|x| x * x).sum();
            let inv = 1.0 / ((ss / D as f32 + 1e-6).sqrt());
            for d in 0..D {
                head[d] *= inv * knw[d];
            }
            llm170_core::ops::rope_head(&mut head, pos as u32, 64, 1e7);
            let kb = pos * NKV * D + hh * D;
            kcache[kb..kb + D].copy_from_slice(&head);
            vcache[kb..kb + D].copy_from_slice(&vin[src..src + D]);
        }
    }
    for t in 0..t_len {
        let kv_len = pos0 + t + 1;
        for hh in 0..NH {
            let kh = hh / 6;
            let src = t * NH * D * 2 + hh * D * 2;
            let mut q: Vec<f32> = qg[src..src + D].to_vec();
            let ss: f32 = q.iter().map(|x| x * x).sum();
            let inv = 1.0 / ((ss / D as f32 + 1e-6).sqrt());
            for d in 0..D {
                q[d] *= inv * qnw[d];
            }
            llm170_core::ops::rope_head(&mut q, (pos0 + t) as u32, 64, 1e7);
            qh_all[t * NH * D + hh * D..t * NH * D + hh * D + D].copy_from_slice(&q);
            let scale = 1.0f32 / (D as f32).sqrt();
            let mut scores = vec![0f32; kv_len];
            for (i, s) in scores.iter_mut().enumerate() {
                let kb = i * NKV * D + kh * D;
                *s = q
                    .iter()
                    .zip(&kcache[kb..kb + D])
                    .map(|(a, b)| a * b)
                    .sum::<f32>()
                    * scale;
            }
            let mx = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut acc = vec![0f32; D];
            let mut wsum = 0f32;
            for i in 0..kv_len {
                let wgt = (scores[i] - mx).exp();
                wsum += wgt;
                let vb = i * NKV * D + kh * D;
                for d in 0..D {
                    acc[d] += wgt * vcache[vb + d];
                }
            }
            for d in 0..D {
                let g = qg[t * NH * D * 2 + hh * D * 2 + D + d];
                let sg = 1.0 / (1.0 + (-g).exp());
                out[t * NH * D + hh * D + d] = acc[d] / wsum * sg;
            }
        }
    }
    (qh_all, kcache, out)
}

// ── 전 모듈 격리 프로브(plans/121 F2c) ──
// 실모델의 모든 선형 형상에 대해 GEMM만 단독 측정 — 형상별 유효 TFLOPS로
// 숨은 타일 비효율을 노출한다(사용자 지시: 전 모듈 격리 점검).
pub fn gemm_check(dir: &str) -> Result<String, String> {
    let mut tr = TrellisResident::load(dir)?;
    let t_rows = 512usize;
    let stage = tr.stage_f32()?;
    // 입력: 균일 값(수치 무의미 — 속도 프로브)
    unsafe {
        std::ptr::write_bytes(stage, 0, t_rows * 6144 * 4);
        for t in 0..t_rows {
            let p = stage.add(t * 6144);
            for i in 0..6144usize {
                *p.add(i) = ((i % 17) as f32 - 8.0) * 0.01;
            }
        }
    }
    let mut report = Vec::new();
    let shapes: &[(&str, &str)] = &[
        (
            "GDN qkv",
            "model.language_model.layers.0.linear_attn.in_proj_qkv",
        ),
        (
            "GDN z",
            "model.language_model.layers.0.linear_attn.in_proj_z",
        ),
        (
            "GDN out",
            "model.language_model.layers.0.linear_attn.out_proj",
        ),
        ("ATTN q", "model.language_model.layers.3.self_attn.q_proj"),
        ("FFN gate", "model.language_model.layers.0.mlp.gate_proj"),
        ("FFN down", "model.language_model.layers.0.mlp.down_proj"),
        ("lm_head", "lm_head"),
    ];
    // 산술 검증 추가: T=8 배치 1행 vs 순차 GEMV 기준(BK=64 변형 판정용)
    {
        let t8 = 8usize;
        let st8 = tr.stage_f32()?;
        unsafe {
            for t in 0..t8 {
                let p8 = st8.add(t * 5120);
                for i in 0..5120usize {
                    *p8.add(i) = ((i % 31) as f32 - 15.0) * 0.013 + (t as f32) * 0.001;
                }
            }
        }
        let key = "model.language_model.layers.0.mlp.gate_proj";
        let _slots = tr.linear_batch_multi_gpu(&[key], t8)?;
        let li0 = tr.find_linear(key)?;
        let n0 = tr.linears[li0].1.n;
        let got = tr.read_yb_head(0, 8 * 4096);
        let xrow: Vec<f32> =
            unsafe { std::slice::from_raw_parts(st8 as *const f32, 5120).to_vec() };
        let want = tr.linear(key, &xrow)?;
        let mut md = 0f32;
        let mut nan_at: Vec<usize> = Vec::new();
        let mut nan_cnt = 0usize;
        for i in 0..n0.min(4096) {
            let g = got[i];
            if !g.is_finite() {
                nan_cnt += 1;
                if nan_at.len() < 6 {
                    nan_at.push(i);
                }
            }
            let d = (g - want[i]).abs();
            if d.is_finite() {
                md = md.max(d);
            }
        }
        let mut row_nan = vec![0usize; 8];
        for t in 0..8usize {
            let seg = &got[t * 4096..(t + 1) * 4096];
            row_nan[t] = seg.iter().filter(|v| !v.is_finite()).count();
        }
        eprintln!(
            "  [gemmdbg] gate T=8 row0 maxdiff={md:.3e} nan={nan_cnt} row_nan={row_nan:?} n0={n0}"
        );
    }
    for (name, key) in shapes {
        let li = tr.find_linear(key)?;
        let (k, n) = (tr.linears[li].1.k, tr.linears[li].1.n);
        // 5회 중앙값
        let mut ts: Vec<f64> = Vec::new();
        for _ in 0..5 {
            let t0 = std::time::Instant::now();
            tr.linear_batch_multi_gpu(&[key], t_rows)?;
            ts.push(t0.elapsed().as_secs_f64() * 1e3);
        }
        ts.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let ms = ts[2];
        let tf = 2.0 * t_rows as f64 * k as f64 * n as f64 / (ms * 1e-3) / 1e12;
        report.push(format!(
            "{name:10} K={k:6} N={n:6}  {ms:7.2}ms  {tf:5.2} TF"
        ));
    }
    Ok(report.join("\n"))
}
