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
    // 스크래치 (재사용 — alloc 폭탄 제거)
    ahb1: VkBuf,
    ahb2: VkBuf,
    ahb3: VkBuf,
    sb: VkBuf,
    yb1: VkBuf,
    yb2: VkBuf,
    yb3: VkBuf,
    xb: VkBuf,
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
        let nseg = 4u32;
        let ahb1 = ctx.alloc(max_k * 2)?;
        let ahb2 = ctx.alloc(max_k * 2)?;
        let ahb3 = ctx.alloc(max_k * 2)?;
        let sb = ctx.alloc(max_n * 4 * nseg as usize)?;
        let yb1 = ctx.alloc(max_n * 4)?;
        let yb2 = ctx.alloc(max_n * 4)?;
        let yb3 = ctx.alloc(max_n * 4)?;
        let xb = ctx.alloc(max_k * 2)?;

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
    fn batch_chain(
        &mut self,
        inputs: &[usize],  // linear 인덱스 목록 (≤3)
        outputs: &[usize], // 호환 — 결과는 슬롯 순서대로 yb1..yb3
    ) -> Result<(), String> {
        let _ = outputs;
        let nseg = 4u32;

        self.ctx.begin_batch()?;
        for (slot, &li) in inputs.iter().enumerate() {
            let l = &self.linears[li].1;
            let (k, n, krate) = (l.k, l.n, l.krate);
            let (suh_b, tre_b, svh_b) = (l.suh.buf, l.tre.buf, l.svh.buf);
            let ah = [self.ahb1.buf, self.ahb2.buf, self.ahb3.buf][slot];
            let yb = [self.yb1.buf, self.yb2.buf, self.yb3.buf][slot];
            let (xb_b, sb_b) = (self.xb.buf, self.sb.buf);

            // had_in: xb × suh_i → ah_i (텐서별 suh — 공유 금지)
            let ds = self.ctx.fresh_ds_for(&self.p1, 3)?;
            self.ctx.bind_bufs(ds, &[xb_b, suh_b, ah]);
            let push = (k as u32 / 128).to_le_bytes().to_vec();
            // ts 표 라벨은 site::tag()(TAG thread-local) — scope은 CUR만 바꾼다.
            crate::rawvk::context::site::set_tag("e3_had_in");
            self.ctx
                .run_rw(
                    self.p1.pl,
                    ds,
                    self.p1.pipe,
                    &push,
                    (k / 128) as u32,
                    1,
                    1,
                    &[xb_b, suh_b],
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
        }

        self.ctx.end_batch_wait()?;
        Ok(())
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
