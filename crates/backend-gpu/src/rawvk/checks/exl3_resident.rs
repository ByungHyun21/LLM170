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
            if name.contains("model.visual.") || name.starts_with("mtp.") {
                continue;
            }
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
                                || name.ends_with("language_model.norm.weight");
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

/// gemm k-분할 수 — GEMV(16)보다 작다: ntg(T/32) 병렬이 있어 스플릿 수요가
/// 적고, 작을수록 had_out_t 환원 부담이 줄어든다. 청크 분해 → 규칙 10a.
const GEMM_NSEG: u32 = 4;
/// 배치 행 상한 — 초과는 드라이버(프리필)가 청크 분할.
pub const BATCH_TMAX: usize = 512;

/// T-배치 스크래치 — 슬롯 3종(GEMV ahb/yb 관례와 동일 구조, run_rw
/// 배리어가 WAR/WAW 커버 — plans/104 판정식, sb는 단일 공유).
pub struct BatchScratch {
    xtb: VkBuf, // [TMAX*max_k] f16 — 업로드(xt)·had_in_t 입력
    ah: [VkBuf; 3],
    yb: [VkBuf; 3], // [TMAX*max_n] f32 ×3
    sb: VkBuf,      // [TMAX*NSEG*max_n] f32
    p1: Pipes,      // had_in_t
    p2: Pipes,      // gemm
    p3: Pipes,      // had_out_t
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
        let p2 = self
            .ctx
            .pipeline_pipes(include_bytes!("../spv/exl3_gemm.spv"), 3, 20)?;
        let p3 = self
            .ctx
            .pipeline_pipes(include_bytes!("../spv/exl3_had_out_t.spv"), 3, 12)?;
        let max_k = self.linears.iter().map(|(_, l)| l.k).max().unwrap_or(5120);
        let max_n = self.linears.iter().map(|(_, l)| l.n).max().unwrap_or(17408);
        // CPU가 직접 읽/쓰는 버퍼(xtb 업로드·yb 판독)는 호스트 RAM(캐시됨) —
        // APU 커브아웃 매핑 판독은 무캐시로 T×n MB급 판독이 ~300MB/s에
        // 걸려 프리필 병목이었다(2026-10-03 계측: lin_gu 34ms/층 중 대부분).
        // ah/sb는 GPU 전용 — 커브아웃 유지.
        let xtb = self.ctx.alloc_host_cached(BATCH_TMAX * max_k * 2)?;
        let ah1 = self.ctx.alloc(BATCH_TMAX * max_k * 2)?;
        let ah2 = self.ctx.alloc(BATCH_TMAX * max_k * 2)?;
        let ah3 = self.ctx.alloc(BATCH_TMAX * max_k * 2)?;
        let y1 = self.ctx.alloc_host_cached(BATCH_TMAX * max_n * 4)?;
        let y2 = self.ctx.alloc_host_cached(BATCH_TMAX * max_n * 4)?;
        let y3 = self.ctx.alloc_host_cached(BATCH_TMAX * max_n * 4)?;
        let sb = self
            .ctx
            .alloc(BATCH_TMAX * GEMM_NSEG as usize * max_n * 4)?;
        self.batch = Some(BatchScratch {
            xtb,
            ah: [ah1, ah2, ah3],
            yb: [y1, y2, y3],
            sb,
            p1,
            p2,
            p3,
        });
        Ok(())
    }

    /// 배치 선형 1개 체인(had_in_t→gemm→had_out_t) — begin_batch 내부 전용.
    fn chain_batch_one(
        &mut self,
        x_src: ash::vk::Buffer,
        li: usize,
        t_rows: u32,
        ah: ash::vk::Buffer,
        yb: ash::vk::Buffer,
    ) -> Result<(), String> {
        let l = &self.linears[li].1;
        let (k, n, krate) = (l.k, l.n, l.krate);
        let (suh_b, tre_b, svh_b) = (l.suh.buf, l.tre.buf, l.svh.buf);
        let b = self.batch.as_ref().ok_or("batch scratch 미초기화")?;
        let sb_b = b.sb.buf;

        // had_in_t: x_src × suh → ah (행별 WHT, grid=(k/128, T))
        {
            let ds = self.ctx.fresh_ds_for(&b.p1, 3)?;
            self.ctx.bind_bufs(ds, &[x_src, suh_b, ah]);
            let push: Vec<u8> = [(k / 128) as u32, k as u32]
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect();
            crate::rawvk::context::site::set_tag("e3_had_in_t");
            self.ctx.run_rw(
                b.p1.pl,
                ds,
                b.p1.pipe,
                &push,
                (k / 128) as u32,
                t_rows,
                1,
                &[x_src, suh_b],
                &[ah],
            )?;
        }

        // gemm: ah × tre → sb (부분합 [T][NSEG][n])
        {
            let ktiles = (k / 16) as u32;
            let ntiles = (n / 16) as u32;
            let n_wgs = ntiles.div_ceil(8);
            let ntg = t_rows.div_ceil(32);
            let ds = self.ctx.fresh_ds_for(&b.p2, 3)?;
            self.ctx.bind_bufs(ds, &[ah, tre_b, sb_b]);
            let push: Vec<u8> = [ktiles, ntiles, krate, t_rows, ntg]
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect();
            crate::rawvk::context::site::set_tag("e3_gemm");
            self.ctx.run_rw(
                b.p2.pl,
                ds,
                b.p2.pipe,
                &push,
                n_wgs * ntg,
                GEMM_NSEG,
                1,
                &[ah, tre_b],
                &[sb_b],
            )?;
        }

        // had_out_t: sb 환원 × svh → yb (grid=(n/128, T))
        {
            let ds = self.ctx.fresh_ds_for(&b.p3, 3)?;
            self.ctx.bind_bufs(ds, &[sb_b, svh_b, yb]);
            let push: Vec<u8> = [(n / 128) as u32, GEMM_NSEG, n as u32]
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

    /// T-배치 공유 입력 다중 선형: y_i[t] = x[t] @ W_i^T (i ≤ 3).
    /// 업로드 1회 + 배치 1회(체인 직렬, 슬롯 분리) — 디코드 linear_triple
    /// 구조의 배치판. x는 [T][k] 행 우선 f32.
    pub fn linear_batch_multi(
        &mut self,
        keys: &[&str],
        x: &[f32],
        t_rows: usize,
    ) -> Result<Vec<Vec<f32>>, String> {
        if keys.is_empty() || keys.len() > 3 {
            return Err(format!("linear_batch_multi: keys {}개 (1..=3)", keys.len()));
        }
        if t_rows == 0 || t_rows > BATCH_TMAX {
            return Err(format!("linear_batch: T={t_rows} 상한 {BATCH_TMAX} 위반"));
        }
        let mut idxs = Vec::with_capacity(keys.len());
        for key in keys {
            idxs.push(self.find_linear(key)?);
        }
        let k = self.linears[idxs[0]].1.k;
        if x.len() != t_rows * k {
            return Err(format!(
                "{}: 배치 형상 {t_rows}x{k} != x.len {}",
                keys[0],
                x.len()
            ));
        }
        for &li in &idxs[1..] {
            if self.linears[li].1.k != k {
                return Err("linear_batch_multi: 입력 차원 불일치".to_string());
            }
        }
        self.ensure_batch()?;

        // 진단 분해(원장 89 키: exl3_lindbg) — 업로드/디스패치/대기/판독.
        static LINDBG_N: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let lindbg = llm170_diag::dump::opts().key("exl3_lindbg")
            && LINDBG_N.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 24;
        let lt0 = std::time::Instant::now();

        // f32→f16 업로드 — upload_x의 8청크 관례(비트동일 RTNE).
        {
            let b = self.batch.as_ref().ok_or("batch scratch")?;
            // SAFETY: xtb는 TMAX*max_k*2 바이트(t_rows*k ≤ 상한, 호출자가
            // linear의 k와 x 형상을 일치시킴 — 위에서 검증).
            let dst = unsafe { std::slice::from_raw_parts_mut(b.xtb.ptr as *mut u16, t_rows * k) };
            for (r, row) in x.chunks(k).enumerate() {
                let off = r * k;
                let (chunks, rem) = row.as_chunks::<8>();
                for (i, c) in chunks.iter().enumerate() {
                    let w: [u16; 8] = std::array::from_fn(|j| f16::from_f32(c[j]).to_bits());
                    dst[off + i * 8..off + i * 8 + 8].copy_from_slice(&w);
                }
                let base = off + chunks.len() * 8;
                for (j, &v) in rem.iter().enumerate() {
                    dst[base + j] = f16::from_f32(v).to_bits();
                }
            }
        }
        let lt1 = std::time::Instant::now();

        // 캐시(비결합) xtb 쓰기 → GPU 가시화 flush.
        {
            let b = self.batch.as_ref().ok_or("batch scratch")?;
            self.ctx.flush_buf(&b.xtb);
        }

        let xtb = self.batch.as_ref().ok_or("batch scratch")?.xtb.buf;
        self.ctx.begin_batch()?;
        for (slot, &li) in idxs.iter().enumerate() {
            let b = self.batch.as_ref().ok_or("batch scratch")?;
            self.chain_batch_one(xtb, li, t_rows as u32, b.ah[slot].buf, b.yb[slot].buf)?;
        }
        let lt2 = std::time::Instant::now();
        self.ctx.end_batch_wait()?;
        self.ctx.wait_pending()?; // DBUF 비동기 잔여 배출
        let lt3 = std::time::Instant::now();

        // 캐시(비결합) yb — GPU 쓰기 판독 전 인밸리데이트.
        {
            let b = self.batch.as_ref().ok_or("batch scratch")?;
            for slot in 0..idxs.len() {
                self.ctx.invalidate_buf(&b.yb[slot]);
            }
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
                "[lindbg] {:52} up {:6.2} disp {:6.2} wait {:6.2} rd {:6.2} ms",
                keys[0],
                (lt1 - lt0).as_secs_f64() * 1e3,
                (lt2 - lt1).as_secs_f64() * 1e3,
                (lt3 - lt2).as_secs_f64() * 1e3,
                (lt4 - lt3).as_secs_f64() * 1e3,
            );
        }
        Ok(outs)
    }

    /// T-배치 단일 선형: y[t] = x[t] @ W^T. x는 [T][k] 행 우선 f32.
    pub fn linear_batch(
        &mut self,
        key: &str,
        x: &[f32],
        t_rows: usize,
    ) -> Result<Vec<f32>, String> {
        self.linear_batch_multi(&[key], x, t_rows)?
            .into_iter()
            .next()
            .ok_or_else(|| "linear_batch: 결과 없음".to_string())
    }
}
