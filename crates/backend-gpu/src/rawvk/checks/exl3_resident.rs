//! EXL3 레이어 스트리밍 디코드 (plans/118 §3-2) — 트렐리스 13GB 상주 +
//! 선형층 vk GEMV + 비선형 CPU. 53.8GB F16 전개 없이 전 모델 디코드.
//!
//! 구조: 각 층의 선형 투영(qkv, z, out, gate, up, down 등)은 이미 검증된
//! 3-커널 체인(had_in→gemv→had_out)으로, norm·conv·GDN·attention은 CPU로.
//! 활성화는 f32 벡터(5120 float = 20KB)라 매 선형 호출 시 업/다운로드.

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
    // 파이프라인
    p1: Pipes,
    p2: Pipes,
    pub p3: Pipes,
    // 스크래치
    ahb: VkBuf,
    sb: VkBuf,
    yb: VkBuf,
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

        let nseg: u32 = 4;

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
                // 무양자화 — CPU f32로
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
                            // A = -exp(A_log), V헤드 순열 적용
                            let nv = numel;
                            let mut a = Vec::with_capacity(nv);
                            for i in 0..nv {
                                let j = 3 * (i % 16) + i / 16;
                                a.push(-v[j.min(nv - 1)].exp());
                            }
                            norms.push((name.clone(), a));
                        } else {
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
                // 층 수 추정
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

        // 스크래치 버퍼
        let max_k = linears.iter().map(|(_, l)| l.k).max().unwrap_or(5120);
        let max_n = linears.iter().map(|(_, l)| l.n).max().unwrap_or(17408);
        let ahb = ctx.alloc(max_k * 2)?;
        let sb = ctx.alloc(max_n * 4 * nseg as usize)?;
        let yb = ctx.alloc(max_n * 4)?;

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
            ahb,
            sb,
            yb,
        })
    }

    /// 선형 투영: y = x @ W^T (트렐리스 vk GEMV).
    pub fn linear(&mut self, key: &str, x: &[f32]) -> Result<Vec<f32>, String> {
        let idx = self
            .linears
            .iter()
            .position(|(k, _)| k == key)
            .ok_or_else(|| format!("linear not found: {key}"))?;
        let (k, n) = (self.linears[idx].1.k, self.linears[idx].1.n);
        if x.len() != k {
            return Err(format!("{key}: input len {} != k {k}", x.len()));
        }

        // x를 f16으로 업로드
        let mut x16 = vec![0u8; k * 2];
        for (i, &v) in x.iter().enumerate() {
            let h = f16::from_f32(v);
            x16[i * 2..i * 2 + 2].copy_from_slice(&h.to_le_bytes());
        }
        let xb = self.ctx.alloc(k * 2)?;
        unsafe {
            std::ptr::copy_nonoverlapping(x16.as_ptr(), xb.ptr, k * 2);
        }

        let nseg: u32 = 4;
        let l = &self.linears[idx].1;

        // had_in
        let ds1 = self.ctx.fresh_ds_for(&self.p1, 3)?;
        self.ctx.bind_bufs(ds1, &[xb.buf, l.suh.buf, self.ahb.buf]);
        let push1 = (k as u32 / 128).to_le_bytes().to_vec();
        self.ctx.run_rw(
            self.p1.pl,
            ds1,
            self.p1.pipe,
            &push1,
            (k / 128) as u32,
            1,
            1,
            &[xb.buf, l.suh.buf],
            &[self.ahb.buf],
        )?;

        // gemv
        let ds2 = self.ctx.fresh_ds_for(&self.p2, 3)?;
        self.ctx
            .bind_bufs(ds2, &[self.ahb.buf, l.tre.buf, self.sb.buf]);
        let push2: Vec<u8> = [(k / 16) as u32, (n / 16) as u32, l.krate]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        self.ctx.run_rw(
            self.p2.pl,
            ds2,
            self.p2.pipe,
            &push2,
            ((n / 16) as u32).div_ceil(8),
            nseg,
            1,
            &[self.ahb.buf, l.tre.buf],
            &[self.sb.buf],
        )?;

        // had_out
        let ds3 = self.ctx.fresh_ds_for(&self.p3, 3)?;
        self.ctx
            .bind_bufs(ds3, &[self.sb.buf, l.svh.buf, self.yb.buf]);
        let push3: Vec<u8> = [(n as u32 / 128), nseg]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        self.ctx.run_rw(
            self.p3.pl,
            ds3,
            self.p3.pipe,
            &push3,
            (n / 128) as u32,
            1,
            1,
            &[self.sb.buf, l.svh.buf],
            &[self.yb.buf],
        )?;

        // run_rw(비배치)는 디스패치마다 자체 wait_for_fences — GPU 완료 보장됨.
        // (end_batch_wait은 배치 모드 전용 — 비배치에서 호출하면 미개시
        // cmdbuf2 종료로 세그폴트, exl3-bench 교훈 2026-10-04.)

        // 결과 다운로드
        let mut y = vec![0f32; n];
        // SAFETY: yb 매핑 — end_batch_wait 후 판독.
        unsafe {
            // 세그먼트 부분합 합산
            let mut parts = vec![0f32; n * nseg as usize];
            std::ptr::copy_nonoverlapping(
                self.sb.ptr as *const f32,
                parts.as_mut_ptr(),
                n * nseg as usize,
            );
            for g in 0..nseg as usize {
                for i in 0..n {
                    y[i] += parts[g * n + i];
                }
            }
        }
        Ok(y)
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
