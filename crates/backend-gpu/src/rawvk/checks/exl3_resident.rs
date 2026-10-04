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
    pub(crate) p1: Pipes,
    pub(crate) p2: Pipes,
    pub(crate) p3: Pipes,
    p4: Pipes,
    // 스크래치 (재사용 — alloc 폭탄 제거)
    ahb1: VkBuf,
    ahb2: VkBuf,
    ahb3: VkBuf,
    pub(crate) sb: VkBuf,
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

pub(crate) fn n_gdn_bytes() -> usize {
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
    pub(crate) x2t: VkBuf,
    pub(crate) p1: Pipes,  // had_in_t (f16 입력 — ew_t 다운 레그)
    pub(crate) p1f: Pipes, // had_in_tf32 (f32 직독)
    pub(crate) p2: Pipes,  // gemm
    pub(crate) p2d: Pipes, // gemm2d(메가융합 3호 — 듀얼입력)
    pub(crate) p3: Pipes,  // had_out_t
    pub(crate) p3d: Pipes, // had_out_td(슬래브 오프셋 환원)
    pub(crate) p4t: Pipes, // ffn_ew_t
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
    pub(crate) pbuf: VkBuf, // [4] u32 — pos0(재생: 푸시 불변으로 만들기 위한 매개변수 버퍼)
    pub(crate) kkc: VkBuf,  // [16*1024][1024] f32
    pub(crate) vkc: VkBuf,
    pub(crate) qh: VkBuf,   // [TMAX*6144] f32 norm+rope q
    pub(crate) qnws: VkBuf, // [16*256]
    pub(crate) knws: VkBuf, // [16*256]
    pub(crate) pa: Pipes,   // attn_prep
    pub(crate) pf3: Pipes,  // attn_fwd3
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
}
