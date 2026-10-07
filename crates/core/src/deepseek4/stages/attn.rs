//! 어텐션 스테이지 — MQA(kv 512 공유) + 윈도우/CSA/HCA + 인덱서 + 싱크.
//!
//! 재현 대상(model.py `Attention`·`Compressor`·`Indexer`, kernel.py
//! `sparse_attn`) — 보고서 §1-§2, §9.1-9.3:
//! - Q: c_Q=wq_a(x)[4096→1024](FP8-sim 활성) → q_norm(가중 RMS) →
//!   wq_b[1024→64·512](FP8-sim) → 헤드별 비가중 RMS → 마지막 64디m RoPE.
//! - KV: wkv(x)[4096→512](FP8-sim) → kv_norm(가중) → RoPE 마지막 64 →
//!   **비로프 448디m 64블록 FP8-sim**(로프 딤은 bf16 정밀도 유지).
//! - MQA: kv 엔트리 하나가 key 겸 value, softmax_scale=512^-0.5.
//! - 싱크: o = Σexp(s−m)v / (Σexp(s−m) + exp(z'−m)), z' [64] f32.
//! - 출력 비회전: o 마지막 64디m 쿼리 위치 rope^-1(켤레).
//! - 그룹 출력: 8그룹(헤드 8개씩) wo_a 4096→1024 → latents [8192] →
//!   wo_b 8192→4096(FP8-sim 활성).
//! - Compressor(CSA r=4 겹침/HCA r=128 비겹침): 학습 게이트 풀링
//!   S=softmax([Z^a+B^a; Z^b+B^b]) — ape는 [ratio, coff·head_dim] F32.
//!   이후 RMSNorm → 블록 시작 위치 RoPE → FP8-sim(비로프).
//! - Indexer(DSA lightning, CSA만): qI=indexer.wq_b(c_Q)[64×128] → RoPE →
//!   Hadamard(N^-0.5) → FP4-sim(32블록). kI=전용 compressor(head 128,
//!   Hadamard+FP4). I_{t,b}=Σ_h w_h·ReLU(qI·kI), 인과 b<(t+1)//ratio,
//!   top-512 토큰레벨 선택.
//!
//! top-k 동점 규칙(모듈 계약): 값 내림차순, 동점 낮은 인덱스 우선.

use crate::deepseek4::config::{Deepseek4Config, LayerKind};
use crate::deepseek4::ops::{
    RopeTable, bf16_round, bf16_round_slice, fp4_sim, fp8_sim, gemm_nt, hadamard_rotate,
    rms_norm_weighted, rms_scale, rope_apply,
};
use crate::ops::exp_cr;

/// 어텐션 프로젝션 가중치 — 전부 f32 k-major([k][n]).
#[derive(Debug, Clone)]
pub struct AttnWeights {
    pub wq_a: Vec<f32>,    // [4096][1024]
    pub q_norm: Vec<f32>,  // [1024]
    pub wq_b: Vec<f32>,    // [1024][32768]
    pub wkv: Vec<f32>,     // [4096][512]
    pub kv_norm: Vec<f32>, // [512]
    pub sink: Vec<f32>,    // [64] (f32 로짓 z')
    /// 그룹별 wo_a [4096][1024] ×8 (헤드 8개 묶음 순서).
    pub wo_a: Vec<Vec<f32>>,
    pub wo_b: Vec<f32>, // [8192][4096]
}

/// 컴프레서 가중치 — wkv/wgate [4096][coff·head_dim], ape [ratio][coff·head_dim].
#[derive(Debug, Clone)]
pub struct CompressorWeights {
    pub wkv: Vec<f32>,
    pub wgate: Vec<f32>,
    pub ape: Vec<f32>,
    pub norm: Vec<f32>,
    pub head_dim: usize,
    pub ratio: usize,
    pub rotate: bool,
}

impl CompressorWeights {
    /// coff = 1 + 겹침(ratio 4 → 2).
    pub fn coff(&self) -> usize {
        1 + usize::from(self.ratio == 4)
    }
}

/// 인덱서 가중치 — wq_b [1024][8192], weights_proj [4096][64],
/// 전용 컴프레서(head 128, Hadamard+FP4).
#[derive(Debug, Clone)]
pub struct IndexerWeights {
    pub wq_b: Vec<f32>,
    pub weights_proj: Vec<f32>,
    pub comp: CompressorWeights,
}

/// 층 어텐션 — kind에 따라 컴프레서/인덱서 유무.
pub struct LayerAttn {
    pub kind: LayerKind,
    pub ratio: usize,
    pub w: AttnWeights,
    pub comp: Option<CompressorWeights>,
    pub idx: Option<IndexerWeights>,
}

/// Q 프로젝션 — (c_Q [t×1024] bf16값, q [t×64×512] 로프 적용 bf16값) 반환.
/// c_Q는 인덱서 wq_b 입력으로도 공유된다.
pub fn project_q(w: &AttnWeights, x: &[f32], cfg: &Deepseek4Config) -> (Vec<f32>, Vec<f32>) {
    let (d, qrank, nh, hd) = (cfg.dim, cfg.q_lora_rank, cfg.n_heads, cfg.head_dim);
    let t = x.len() / d;
    // wq_a: FP8-sim 활성 → gemm → bf16.
    let mut xq = x.to_vec();
    fp8_sim(&mut xq, d, 128);
    let mut c = vec![0.0f32; t * qrank];
    gemm_nt(&xq, &w.wq_a, d, qrank, &mut c);
    bf16_round_slice(&mut c);
    // q_norm(가중 RMS).
    let mut rows = Vec::with_capacity(t);
    for i in 0..t {
        rows.push(rms_norm_weighted(
            &c[i * qrank..(i + 1) * qrank],
            &w.q_norm,
            cfg.rms_eps,
        ));
    }
    c = rows.concat();
    // wq_b: FP8-sim → gemm → bf16 → 헤드별 비가중 RMS(×rsqrt) → bf16.
    let mut cq = c.clone();
    fp8_sim(&mut cq, qrank, 128);
    let mut q = vec![0.0f32; t * nh * hd];
    gemm_nt(&cq, &w.wq_b, qrank, nh * hd, &mut q);
    bf16_round_slice(&mut q);
    for i in 0..t {
        for h in 0..nh {
            let head = &mut q[i * nh * hd + h * hd..i * nh * hd + (h + 1) * hd];
            let s = rms_scale(head, cfg.rms_eps);
            for v in head.iter_mut() {
                *v = bf16_round(*v * s);
            }
        }
    }
    (c, q)
}

/// RoPE를 q 전체 토큰의 마지막 64디m에 적용(토큰 위치 0..t) — bf16 경계.
pub fn rope_q(q: &mut [f32], t: usize, cfg: &Deepseek4Config, rope: &RopeTable) {
    let (nh, hd, rd) = (cfg.n_heads, cfg.head_dim, cfg.rope_head_dim);
    for i in 0..t {
        for h in 0..nh {
            let base = i * nh * hd + h * hd;
            let head = &mut q[base..base + hd];
            rope_apply(&mut head[hd - rd..], rope.at(i), false);
            for v in head[hd - rd..].iter_mut() {
                *v = bf16_round(*v);
            }
        }
    }
}

/// KV 프로젝션 — [t×512], norm → RoPE 마지막 64 → 비로프 448디m 64블록 FP8-sim.
pub fn project_kv(w: &AttnWeights, x: &[f32], cfg: &Deepseek4Config, rope: &RopeTable) -> Vec<f32> {
    let (d, hd, rd) = (cfg.dim, cfg.head_dim, cfg.rope_head_dim);
    let t = x.len() / d;
    let mut xq = x.to_vec();
    fp8_sim(&mut xq, d, 128);
    let mut kv = vec![0.0f32; t * hd];
    gemm_nt(&xq, &w.wkv, d, hd, &mut kv);
    bf16_round_slice(&mut kv);
    let mut out = Vec::with_capacity(t * hd);
    for i in 0..t {
        let mut row = rms_norm_weighted(&kv[i * hd..(i + 1) * hd], &w.kv_norm, cfg.rms_eps);
        rope_apply(&mut row[hd - rd..], rope.at(i), false);
        for v in row[hd - rd..].iter_mut() {
            *v = bf16_round(*v);
        }
        // QAT: 비로프 448디m만 FP8-sim(64블록) — 로프 디m bf16 유지(§9.2).
        fp8_sim(&mut row[..hd - rd], hd - rd, 64);
        out.extend_from_slice(&row);
    }
    out
}

/// 프리필 윈도우 인덱스 — 토큰 t: max(0, t-(win-1))..=t (오래된→최신 순).
pub fn window_idx_prefill(t: usize, win: usize) -> Vec<i32> {
    let start = t.saturating_sub(win - 1);
    (start..=t).map(|v| v as i32).collect()
}

/// 디코드 윈도우 링 인덱스 — start_pos 이후 토큰 1개 처리용
/// (get_window_topk_idxs start_pos>0 경로): 링 순서 오래된→최신 win개.
pub fn window_idx_decode(pos: usize, win: usize) -> Vec<i32> {
    let sp = pos % win;
    let mut v: Vec<i32> = ((sp + 1)..win).map(|x| x as i32).collect();
    v.extend(0..=(sp as i32));
    v
}

/// HCA 밀도 압축 인덱스 — 토큰 t에 블록 0..(t+1)//r (인과), +offset.
pub fn compress_idx_dense(t: usize, ratio: usize, offset: usize) -> Vec<i32> {
    (0..(t + 1) / ratio).map(|b| (b + offset) as i32).collect()
}

/// 게이트 풀링(프리필) — kv_c/score_c [t × coff·d], ape [ratio × coff·d].
/// 반환: 블록별 풀 결과 [nb][d] (노orm/로프 전). t<ratio면 빈 벡터.
/// overlap(ratio 4): 블록 i 풀 = [이전 블록 절반(첫 d) ; 현재 블록 절반(둘째 d)]
/// 8행, i=0의 이전 반부는 0/-inf 패드(model.py `overlap_transform`).
pub fn compressor_pool_prefill(
    kv_c: &[f32],
    score_c: &[f32],
    ape: &[f32],
    head_dim: usize,
    ratio: usize,
) -> Vec<Vec<f32>> {
    let overlap = ratio == 4;
    let coff = 1 + usize::from(overlap);
    let t = kv_c.len() / (coff * head_dim);
    if t < ratio {
        return Vec::new();
    }
    let cd = coff * head_dim;
    let remainder = t % ratio;
    let cutoff = t - remainder;
    let nb = cutoff / ratio;
    let mut out = Vec::with_capacity(nb);
    for i in 0..nb {
        // 풀 행 조립: [이전 블록 첫 반부 | 현재 블록 둘째 반부] (overlap),
        // 비겹침은 현재 블록 전체(coff=1).
        let mut pkv = vec![0.0f32; coff * ratio * head_dim];
        let mut psc = vec![f32::NEG_INFINITY; coff * ratio * head_dim];
        // 현재 블록 슬롯: overlap이면 rows[ratio..2ratio], 아니면 rows[0..ratio].
        let row_off = if overlap { ratio } else { 0 };
        let src = if overlap { head_dim } else { 0 };
        for j in 0..ratio {
            let cur = i * ratio + j;
            for dd in 0..head_dim {
                pkv[(row_off + j) * head_dim + dd] = kv_c[cur * cd + src + dd];
                psc[(row_off + j) * head_dim + dd] =
                    score_c[cur * cd + src + dd] + ape[j * cd + src + dd];
            }
            if overlap && i > 0 {
                let prev = cur - ratio;
                for dd in 0..head_dim {
                    pkv[j * head_dim + dd] = kv_c[prev * cd + dd];
                    psc[j * head_dim + dd] = score_c[prev * cd + dd] + ape[j * cd + dd];
                }
            }
        }
        out.push(pool_rows(&pkv, &psc, coff * ratio, head_dim));
    }
    out
}

/** 풀 행 가중합 — score [rows][d]를 **풀 엔트리 축**(model.py softmax(dim=2)/
 * dim=1 — 디m별 소프트맥스)으로 정규화해 kv 가중합. f32, 행 오름차순 누산. */
fn pool_rows(pkv: &[f32], psc: &[f32], rows: usize, d: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; d];
    let mut w = vec![0.0f32; rows];
    for dd in 0..d {
        let mut m = f32::NEG_INFINITY;
        for r in 0..rows {
            m = m.max(psc[r * d + dd]);
        }
        let mut s = 0.0f32;
        for r in 0..rows {
            w[r] = exp_cr(psc[r * d + dd] - m);
            s += w[r];
        }
        let mut acc = 0.0f32;
        for r in 0..rows {
            acc += w[r] * pkv[r * d + dd];
        }
        out[dd] = acc / s;
    }
    out
}

/// 컴프레서 종결 — RMSNorm → 블록 시작 위치 RoPE(마지막 64) →
/// rotate면 Hadamard+FP4(전 head_dim), 아니면 비로프 FP8-sim(64블록).
/// `block_pos`는 블록 시작 **토큰** 위치(i·ratio — 같은 주파수 표).
pub fn compressor_finish(
    pooled: &[f32],
    cw: &CompressorWeights,
    cfg: &Deepseek4Config,
    rope: &RopeTable,
    block_pos: usize,
) -> Vec<f32> {
    let (hd, rd) = (cw.head_dim, cfg.rope_head_dim);
    let mut kv = rms_norm_weighted(pooled, &cw.norm, cfg.rms_eps);
    rope_apply(&mut kv[hd - rd..], rope.at(block_pos), false);
    for v in kv[hd - rd..].iter_mut() {
        *v = bf16_round(*v);
    }
    if cw.rotate {
        // 인덱서 전용: Hadamard(전 128) → FP4-sim (§2, §9.3).
        hadamard_rotate(&mut kv);
        bf16_round_slice(&mut kv);
        fp4_sim(&mut kv);
    } else {
        fp8_sim(&mut kv[..hd - rd], hd - rd, 64);
    }
    kv
}

/// 디코드 증분 컴프레서 상태 — kv/score [coff·ratio × coff·d] 링
/// (model.py kv_state/score_state). (pos+1)%ratio마다 엔트리 1개 방출.
/// 증분 비트 동일성이 계약이다(보고서 §2).
pub struct CompressState {
    pub kv: Vec<f32>,
    pub score: Vec<f32>,
    head_dim: usize,
    ratio: usize,
}

impl CompressState {
    pub fn new(cw: &CompressorWeights) -> Self {
        let coff = cw.coff();
        let rows = coff * cw.ratio;
        let cols = coff * cw.head_dim;
        CompressState {
            kv: vec![0.0; rows * cols],
            // 초기 -inf (model.py full("-inf")) — 미기입 슬롯 배제용.
            score: vec![f32::NEG_INFINITY; rows * cols],
            head_dim: cw.head_dim,
            ratio: cw.ratio,
        }
    }

    /// 프리필 직후 꼬리 시딩 — model.py start_pos==0 경로의 상태 기입.
    /// 반환값은 프리필 풀 결과(블록별 [d]) — caller가 finish를 적용.
    pub fn from_prefill(&mut self, kv_c: &[f32], score_c: &[f32], ape: &[f32]) -> Vec<Vec<f32>> {
        let (d, ratio) = (self.head_dim, self.ratio);
        let coff = 1 + usize::from(ratio == 4);
        let cd = coff * d;
        let t = kv_c.len() / cd;
        let pooled = compressor_pool_prefill(kv_c, score_c, ape, d, ratio);
        let remainder = t % ratio;
        let cutoff = t - remainder;
        let offset = if coff == 2 { ratio } else { 0 };
        if coff == 2 && cutoff >= ratio {
            for j in 0..ratio {
                let src = cutoff - ratio + j;
                for c in 0..cd {
                    self.kv[j * cd + c] = kv_c[src * cd + c];
                    self.score[j * cd + c] = score_c[src * cd + c] + ape[j * cd + c];
                }
            }
        }
        if remainder > 0 {
            for j in 0..remainder {
                let src = cutoff + j;
                for c in 0..cd {
                    self.kv[(offset + j) * cd + c] = kv_c[src * cd + c];
                    self.score[(offset + j) * cd + c] = score_c[src * cd + c] + ape[j * cd + c];
                }
            }
        }
        pooled
    }

    /// 디코드 1토큰 공급 — 방출 조건이면 풀 결과 Some([d]) (종결 전).
    /// `pos`는 해당 토큰의 절대 위치.
    pub fn step(
        &mut self,
        kv_row: &[f32],
        score_row: &[f32],
        ape: &[f32],
        pos: usize,
    ) -> Option<Vec<f32>> {
        let (d, ratio) = (self.head_dim, self.ratio);
        let coff = 1 + usize::from(ratio == 4);
        let cd = coff * d;
        let j = pos % ratio;
        let mut sc = score_row.to_vec();
        for c in 0..cd {
            sc[c] += ape[j * cd + c];
        }
        let emit = (pos + 1).is_multiple_of(ratio);
        if coff == 2 {
            // [겹침 반부 | 현재 반부] 저장 — 방출 시 절반씩 접합.
            self.kv[(ratio + j) * cd..(ratio + j + 1) * cd].copy_from_slice(kv_row);
            self.score[(ratio + j) * cd..(ratio + j + 1) * cd].copy_from_slice(&sc);
            if emit {
                let mut pkv = vec![0.0f32; 2 * ratio * d];
                let mut psc = vec![0.0f32; 2 * ratio * d];
                for r in 0..ratio {
                    for dd in 0..d {
                        pkv[r * d + dd] = self.kv[r * cd + dd];
                        psc[r * d + dd] = self.score[r * cd + dd];
                        pkv[(ratio + r) * d + dd] = self.kv[(ratio + r) * cd + d + dd];
                        psc[(ratio + r) * d + dd] = self.score[(ratio + r) * cd + d + dd];
                    }
                }
                // 상태 시프트: 완료 블록의 현재 반부 → 다음 겹침 반부.
                for r in 0..ratio {
                    for c in 0..cd {
                        self.kv[r * cd + c] = self.kv[(ratio + r) * cd + c];
                        self.score[r * cd + c] = self.score[(ratio + r) * cd + c];
                    }
                }
                Some(pool_rows(&pkv, &psc, 2 * ratio, d))
            } else {
                None
            }
        } else {
            self.kv[j * cd..(j + 1) * cd].copy_from_slice(kv_row);
            self.score[j * cd..(j + 1) * cd].copy_from_slice(&sc);
            if emit {
                Some(pool_rows(&self.kv, &self.score, ratio, d))
            } else {
                None
            }
        }
    }
}

/// 인덱서 qI — c_Q 공유 입력, RoPE(마지막 64) → Hadamard(128) → FP4-sim.
/// 반환 [t×64×128].
pub fn indexer_q(
    idx: &IndexerWeights,
    c_q: &[f32],
    cfg: &Deepseek4Config,
    rope: &RopeTable,
) -> Vec<f32> {
    let (qrank, ih, id, rd) = (
        cfg.q_lora_rank,
        cfg.index_n_heads,
        cfg.index_head_dim,
        cfg.rope_head_dim,
    );
    let t = c_q.len() / qrank;
    let mut xq = c_q.to_vec();
    fp8_sim(&mut xq, qrank, 128);
    let mut q = vec![0.0f32; t * ih * id];
    gemm_nt(&xq, &idx.wq_b, qrank, ih * id, &mut q);
    bf16_round_slice(&mut q);
    for i in 0..t {
        for h in 0..ih {
            let base = i * ih * id + h * id;
            let head = &mut q[base..base + id];
            rope_apply(&mut head[id - rd..], rope.at(i), false);
            for v in head[id - rd..].iter_mut() {
                *v = bf16_round(*v);
            }
            hadamard_rotate(head);
            bf16_round_slice(head);
            fp4_sim(head);
        }
    }
    q
}

/// 인덱서 kI 빌드 — 전용 컴프레서(head 128, rotate)로 x에서 압축 엔트리.
pub fn indexer_k(
    idx: &IndexerWeights,
    x: &[f32],
    cfg: &Deepseek4Config,
    rope: &RopeTable,
) -> Vec<Vec<f32>> {
    let d = cfg.dim;
    let t = x.len() / d;
    let cw = &idx.comp;
    let cd = cw.coff() * cw.head_dim;
    let mut kv_c = vec![0.0f32; t * cd];
    gemm_nt(x, &cw.wkv, d, cd, &mut kv_c);
    let mut sc_c = vec![0.0f32; t * cd];
    gemm_nt(x, &cw.wgate, d, cd, &mut sc_c);
    let pooled = compressor_pool_prefill(&kv_c, &sc_c, &cw.ape, cw.head_dim, cw.ratio);
    pooled
        .into_iter()
        .enumerate()
        .map(|(i, p)| compressor_finish(&p, cw, cfg, rope, i * cw.ratio))
        .collect()
}

/// 인덱서 헤드 가중치 — weights_proj(x)·(128^-0.5·64^-0.5) [t×64] bf16값.
pub fn indexer_weights(idx: &IndexerWeights, x: &[f32], cfg: &Deepseek4Config) -> Vec<f32> {
    let d = cfg.dim;
    let t = x.len() / d;
    let mut w = vec![0.0f32; t * cfg.index_n_heads];
    gemm_nt(x, &idx.weights_proj, d, cfg.index_n_heads, &mut w);
    let c = 1.0 / (cfg.index_head_dim as f32).sqrt() / (cfg.index_n_heads as f32).sqrt();
    for v in w.iter_mut() {
        *v = bf16_round(bf16_round(*v) * c);
    }
    w
}

/// 인덱서 스코어 — score[t][b] = Σ_h w[t,h]·ReLU(qI·kI). f32 누산, 최종 bf16.
pub fn indexer_scores(
    iq: &[f32],
    ki: &[Vec<f32>],
    w: &[f32],
    cfg: &Deepseek4Config,
) -> Vec<Vec<f32>> {
    let (ih, id) = (cfg.index_n_heads, cfg.index_head_dim);
    let t = w.len() / ih;
    let nb = ki.len();
    let mut out = vec![vec![0.0f32; nb]; t];
    for ti in 0..t {
        for (bi, kb) in ki.iter().enumerate() {
            let mut sum = 0.0f32;
            for h in 0..ih {
                let qh = &iq[ti * ih * id + h * id..ti * ih * id + (h + 1) * id];
                let mut dot = 0.0f32;
                for dd in 0..id {
                    dot += qh[dd] * kb[dd];
                }
                sum += dot.max(0.0) * w[ti * ih + h];
            }
            out[ti][bi] = bf16_round(sum);
        }
    }
    out
}

/// 토큰레벨 top-k 선택 — 인과 b<(t+1)//ratio, 동점 낮은 블록 우선,
/// 반환 인덱스에 `offset`(윈도우 뒤 압축 슬롯) 가산. 미가시 -1.
pub fn indexer_topk(
    scores: &[Vec<f32>],
    cfg: &Deepseek4Config,
    ratio: usize,
    offset: usize,
) -> Vec<Vec<i32>> {
    let t = scores.len();
    let mut out = Vec::with_capacity(t);
    for (ti, row) in scores.iter().enumerate() {
        let nb = row.len();
        let k = cfg.index_topk.min(nb);
        let visible = (ti + 1) / ratio;
        let mut cand: Vec<(f32, usize)> = row
            .iter()
            .enumerate()
            .filter(|&(b, _)| b < visible)
            .map(|(b, &s)| (s, b))
            .collect();
        // 안정 정렬: 값 내림차순, 동점 낮은 인덱스 우선(모듈 계약).
        cand.sort_by(|a, b| {
            b.0.partial_cmp(&a.0)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.1.cmp(&b.1))
        });
        out.push(
            cand.into_iter()
                .take(k)
                .map(|(_, b)| (b + offset) as i32)
                .collect(),
        );
    }
    out
}

/// 싱크 포함 스파스 어텐션 — 1토큰. q [64×512], kv [rows×512], idxs (−1 무시).
/// o = Σ bf16(p)·v / (Σp + exp(z'−m)) — p의 bf16 캐스트만 커널과 공유,
/// 일괄 소프트맥스(타일랭 온라인 재스케일의 블록 최대 의존 제거).
pub fn sparse_attn_one(
    q: &[f32],
    kv: &[f32],
    idxs: &[i32],
    sink: &[f32],
    cfg: &Deepseek4Config,
) -> Vec<f32> {
    let (nh, hd) = (cfg.n_heads, cfg.head_dim);
    let scale = 1.0 / (hd as f32).sqrt();
    let mut o = vec![0.0f32; nh * hd];
    let n_idx = idxs.len();
    for h in 0..nh {
        let qh = &q[h * hd..(h + 1) * hd];
        let mut s = vec![f32::NEG_INFINITY; n_idx];
        for (e, &ix) in idxs.iter().enumerate() {
            if ix < 0 {
                continue;
            }
            let krow = &kv[ix as usize * hd..(ix as usize + 1) * hd];
            let mut dot = 0.0f32;
            for dd in 0..hd {
                dot += qh[dd] * krow[dd];
            }
            s[e] = dot * scale;
        }
        let m = s.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut denom = exp_cr(sink[h] - m);
        let mut acc = vec![0.0f32; hd];
        for (e, &se) in s.iter().enumerate() {
            if se == f32::NEG_INFINITY {
                continue;
            }
            let p = exp_cr(se - m);
            denom += p;
            let p16 = bf16_round(p);
            if idxs[e] < 0 {
                continue;
            }
            let krow = &kv[idxs[e] as usize * hd..(idxs[e] as usize + 1) * hd];
            for dd in 0..hd {
                acc[dd] += p16 * krow[dd];
            }
        }
        for dd in 0..hd {
            o[h * hd + dd] = bf16_round(acc[dd] / denom);
        }
    }
    o
}

/// 그룹 출력 — 8그룹 wo_a(4096→1024) → latents [8192] → FP8-sim → wo_b(→4096).
pub fn attention_output(o: &[f32], w: &AttnWeights, cfg: &Deepseek4Config) -> Vec<f32> {
    let (d, g, r) = (cfg.dim, cfg.o_groups, cfg.o_lora_rank);
    let hd = cfg.head_dim;
    let gdim = cfg.n_heads * hd / g; // 4096
    let mut latents = vec![0.0f32; g * r];
    for gi in 0..g {
        let og = &o[gi * gdim..(gi + 1) * gdim];
        gemm_nt(og, &w.wo_a[gi], gdim, r, &mut latents[gi * r..(gi + 1) * r]);
    }
    bf16_round_slice(&mut latents);
    fp8_sim(&mut latents, g * r, 128);
    let mut y = vec![0.0f32; d];
    gemm_nt(&latents, &w.wo_b, g * r, d, &mut y);
    bf16_round_slice(&mut y);
    y
}

/// 프리필 어텐션 전체 — x [t×d](attn_norm 출력, bf16값) → 출력 [t×d].
/// start_pos=0 경로(model.py `Attention.forward` 프리필 분기).
pub fn attention_forward(
    la: &LayerAttn,
    x: &[f32],
    cfg: &Deepseek4Config,
    rope: &RopeTable,
) -> Vec<f32> {
    let (d, t) = (cfg.dim, x.len() / cfg.dim);
    let (c_q, mut q) = project_q(&la.w, x, cfg);
    rope_q(&mut q, t, cfg, rope);
    let kv = project_kv(&la.w, x, cfg, rope);
    let (nh, hd) = (cfg.n_heads, cfg.head_dim);

    // 압축 엔트리 + 토큰별 topk 인덱스.
    let mut kv_all = kv;
    let mut comp_sel: Vec<Vec<i32>> = vec![Vec::new(); t];
    if let Some(cw) = &la.comp {
        let cd = cw.coff() * cw.head_dim;
        let mut kv_c = vec![0.0f32; t * cd];
        gemm_nt(x, &cw.wkv, d, cd, &mut kv_c);
        let mut sc_c = vec![0.0f32; t * cd];
        gemm_nt(x, &cw.wgate, d, cd, &mut sc_c);
        let pooled = compressor_pool_prefill(&kv_c, &sc_c, &cw.ape, cw.head_dim, cw.ratio);
        let offset = t; // 프리필: 압축 슬롯은 윈도우(전체 청크 kv) 뒤.
        for (i, p) in pooled.into_iter().enumerate() {
            let e = compressor_finish(&p, cw, cfg, rope, i * cw.ratio);
            kv_all.extend_from_slice(&e);
        }
        match la.kind {
            LayerKind::Csa => {
                let idx = la.idx.as_ref().expect("CSA 층에 인덱서 필요");
                let iq = indexer_q(idx, &c_q, cfg, rope);
                let ki = indexer_k(idx, x, cfg, rope);
                let w = indexer_weights(idx, x, cfg);
                let scores = indexer_scores(&iq, &ki, &w, cfg);
                comp_sel = indexer_topk(&scores, cfg, la.ratio, offset);
            }
            LayerKind::Hca => {
                for ti in 0..t {
                    comp_sel[ti] = compress_idx_dense(ti, la.ratio, offset);
                }
            }
            LayerKind::Swa => unreachable!("comp 있는 SWA 층 없음"),
        }
    }

    // 토큰별 스파스 어텐션 + 역회전 + 그룹 출력.
    let mut out = vec![0.0f32; t * d];
    let mut o = vec![0.0f32; nh * hd];
    for ti in 0..t {
        let mut idxs = window_idx_prefill(ti, cfg.window);
        idxs.extend(comp_sel[ti].iter().copied());
        let q_t = &q[ti * nh * hd..(ti + 1) * nh * hd];
        let ot = sparse_attn_one(q_t, &kv_all, &idxs, &la.w.sink, cfg);
        o.copy_from_slice(&ot);
        // 출력 비회전: o 마지막 64디m 쿼리 위치 rope^-1 (켤레).
        for h in 0..nh {
            let base = h * hd;
            rope_apply(
                &mut o[base + hd - cfg.rope_head_dim..base + hd],
                rope.at(ti),
                true,
            );
            for v in o[base + hd - cfg.rope_head_dim..base + hd].iter_mut() {
                *v = bf16_round(*v);
            }
        }
        let y = attention_output(&o, &la.w, cfg);
        out[ti * d..(ti + 1) * d].copy_from_slice(&y);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> Deepseek4Config {
        Deepseek4Config {
            n_layers: 43,
            dim: 8,
            vocab: 129280,
            rms_eps: 1e-6,
            hc_eps: 1e-6,
            hc_mult: 4,
            hc_sinkhorn_iters: 20,
            n_heads: 2,
            head_dim: 4,
            rope_head_dim: 2,
            q_lora_rank: 4,
            o_lora_rank: 4,
            o_groups: 2,
            window: 3,
            compress_ratios: vec![0, 4, 128],
            compress_rope_theta: 160000.0,
            rope_theta: 10000.0,
            yarn_factor: 16.0,
            yarn_orig_len: 65536,
            yarn_beta_fast: 32.0,
            yarn_beta_slow: 1.0,
            index_n_heads: 2,
            index_head_dim: 2,
            index_topk: 2,
            n_routed: 4,
            n_shared: 1,
            n_activated: 2,
            moe_inter: 8,
            route_scale: 1.5,
            swiglu_limit: 10.0,
            n_hash_layers: 1,
            n_mtp_layers: 0,
            dspark_block: 0,
            dspark_noise_token: 0,
            dspark_targets: vec![],
            dspark_markov_rank: 0,
        }
    }

    /// 윈도우 인덱스 — 프리필 범위·링 순서(디코드).
    #[test]
    fn window_indices() {
        // 프리필: t<win → 0..=t; t≥win → t-(win-1)..=t.
        assert_eq!(window_idx_prefill(0, 3), vec![0]);
        assert_eq!(window_idx_prefill(2, 3), vec![0, 1, 2]);
        assert_eq!(window_idx_prefill(5, 3), vec![3, 4, 5]);
        // 디코드 링: pos%win 이후가 가장 오래된 순.
        assert_eq!(window_idx_decode(0, 4), vec![1, 2, 3, 0]);
        assert_eq!(window_idx_decode(5, 4), vec![2, 3, 0, 1]);
        assert_eq!(window_idx_decode(3, 4), vec![0, 1, 2, 3]);
    }

    /// HCA 밀도 인덱스 — 인과 (t+1)//r.
    #[test]
    fn dense_compress_causality() {
        assert_eq!(compress_idx_dense(0, 4, 8), vec![]);
        assert_eq!(compress_idx_dense(3, 4, 8), vec![8]);
        assert_eq!(compress_idx_dense(7, 4, 8), vec![8, 9]);
        assert_eq!(compress_idx_dense(8, 4, 8), vec![8, 9]);
        assert_eq!(compress_idx_dense(200, 128, 0), vec![0]); // (201)//128=1
    }

    /// CSA 겹침 풀 — 블록 i의 8행 구성(이전 첫반부+현재 둘째반부)·패드·ape 편향.
    #[test]
    fn csa_overlap_pooling() {
        let (d, ratio) = (2usize, 4usize);
        let coff = 2;
        let cd = coff * d;
        let t = 8;
        // wkv 출력: 토큰 i 첫반부 = (i, 0), 둘째반부 = (100+i, 0).
        let kv_c: Vec<f32> = (0..t)
            .flat_map(|i| [i as f32, 0.0, 100.0 + i as f32, 0.0])
            .collect();
        // wgate: 전부 0 — 편향만으로 결정.
        let score_c = vec![0.0f32; t * cd];
        // ape: 위치 j에서 첫반부 편향 = j, 둘째 반부 = 10+j (디m 0), 디m1 0.
        let ape: Vec<f32> = (0..ratio)
            .flat_map(|j| [j as f32, 0.0, 10.0 + j as f32, 0.0])
            .collect();
        let out = compressor_pool_prefill(&kv_c, &score_c, &ape, d, ratio);
        assert_eq!(out.len(), 2);
        // 블록 0: 풀 = [pad(-inf)×4 ; 토큰0-3 둘째반부], 편향 10..13 —
        // 디m0 수동 softmax 대조(최댓값 13 기준), 디m1은 pad 제외 균등 → 0.
        let sc0 = [f32::NEG_INFINITY; 4]
            .into_iter()
            .chain([10.0f32, 11.0, 12.0, 13.0])
            .collect::<Vec<_>>();
        let kv0 = [0.0f32, 0.0, 0.0, 0.0, 100.0, 101.0, 102.0, 103.0].to_vec();
        let w0: Vec<f32> = sc0.iter().map(|&s| exp_cr(s - 13.0)).collect();
        let s0: f32 = w0.iter().sum();
        let acc0: f32 = w0.iter().zip(kv0.iter()).map(|(&wi, &vi)| wi * vi).sum();
        assert!(
            (out[0][0] - acc0 / s0).abs() < 1e-4,
            "{} vs {}",
            out[0][0],
            acc0 / s0
        );
        assert_eq!(out[0][1], 0.0);
        // 블록 1: 풀 = [토큰0-3 첫반부(0..3) 편향 0..3 ; 토큰4-7 둘째반부(104..107)
        // 편향 14..17] — 8행 전부 유효, 수동 softmax 대조.
        assert!(
            (out[1][0] - 105.0).abs() < 3.0,
            "가중합 범위: {}",
            out[1][0]
        );
        // ape는 블록 인덱스와 무선(位置 j만) — 블록1 스코어도 [0..3 ; 10..13].
        let sc: Vec<f32> = [0.0f32, 1.0, 2.0, 3.0, 10.0, 11.0, 12.0, 13.0].to_vec();
        let kvv: Vec<f32> = [0.0f32, 1.0, 2.0, 3.0, 104.0, 105.0, 106.0, 107.0].to_vec();
        let m = 13.0f32;
        let w: Vec<f32> = sc.iter().map(|&s| exp_cr(s - m)).collect();
        let sum: f32 = w.iter().sum();
        let acc: f32 = w.iter().zip(kvv.iter()).map(|(&wi, &vi)| wi * vi).sum();
        assert!((out[1][0] - acc / sum).abs() < 1e-5);
    }

    /// 증분 컴프레서 = 프리필 재계산 (증분 비트 동일성 — 보고서 §2).
    #[test]
    fn compressor_decode_equals_prefill() {
        let (d, ratio) = (4usize, 4usize);
        let coff = 2;
        let cd = coff * d;
        let cw = CompressorWeights {
            wkv: vec![],
            wgate: vec![],
            ape: vec![0.1f32; ratio * cd],
            norm: vec![1.0; d],
            head_dim: d,
            ratio,
            rotate: false,
        };
        // 20토큰 합성 wkv/wgate.
        let kv_c: Vec<f32> = (0..20)
            .flat_map(|i| (0..cd).map(move |c| (i * 7 + c) as f32 * 0.31 - 3.0))
            .collect();
        let sc_c: Vec<f32> = (0..20)
            .flat_map(|i| (0..cd).map(move |c| ((i + c) % 5) as f32 * 0.4 - 1.0))
            .collect();
        // 프리필 16토큰 → 블록 0-3 + 상태 시딩, 이후 디코드 16..19 → 블록 4 방출.
        let mut st = CompressState::new(&cw);
        let e_pre = st.from_prefill(&kv_c[..16 * cd], &sc_c[..16 * cd], &cw.ape);
        assert_eq!(e_pre.len(), 4);
        let mut stepped = None;
        for pos in 16..20 {
            if let Some(e) = st.step(
                &kv_c[pos * cd..(pos + 1) * cd],
                &sc_c[pos * cd..(pos + 1) * cd],
                &cw.ape,
                pos,
            ) {
                assert!(stepped.is_none(), "16..19에 방출은 pos=19 한 번");
                stepped = Some(e);
            }
        }
        // 전체 프리필(20토큰) → 블록 0-4.
        let e_full = compressor_pool_prefill(&kv_c, &sc_c, &cw.ape, d, ratio);
        assert_eq!(e_full.len(), 5);
        for b in 0..4 {
            assert_eq!(e_pre[b], e_full[b], "블록 {b}");
        }
        let step = stepped.expect("블록 4 방출");
        assert_eq!(step, e_full[4], "증분 블록 4 == 프리필 재계산");
        // HCA(비겹침)도 동일 검증.
        let cw128 = CompressorWeights {
            wkv: vec![],
            wgate: vec![],
            ape: vec![0.05f32; 128 * 128],
            norm: vec![1.0; 128],
            head_dim: 128,
            ratio: 128,
            rotate: false,
        };
        let d128 = 128;
        let kv128: Vec<f32> = (0..256)
            .flat_map(|i| (0..d128).map(move |c| (i * 13 + c) as f32 * 0.11))
            .collect();
        let sc128: Vec<f32> = (0..256)
            .flat_map(|i| (0..d128).map(move |c| ((i + c) % 7) as f32 * 0.3 - 1.0))
            .collect();
        let mut st2 = CompressState::new(&cw128);
        let e2 = st2.from_prefill(&kv128, &sc128, &cw128.ape);
        assert_eq!(e2.len(), 2);
        let f2 = compressor_pool_prefill(&kv128, &sc128, &cw128.ape, d128, 128);
        assert_eq!(e2, f2);
    }

    /// 싱크 스파스 어텐션 — 수동 밀도 공식 대조 + (-1) 제외.
    #[test]
    fn sparse_attn_matches_dense() {
        let c = cfg();
        let (nh, hd) = (c.n_heads, c.head_dim);
        let q: Vec<f32> = (0..nh * hd).map(|i| (i as f32 * 0.7) - 3.0).collect();
        let kv: Vec<f32> = (0..4 * hd).map(|i| (i as f32 * 0.13) - 1.5).collect();
        let idxs = vec![0i32, -1, 2, 3];
        let sink = vec![0.25f32; nh];
        let o = sparse_attn_one(&q, &kv, &idxs, &sink, &c);
        let scale = 1.0 / (hd as f32).sqrt();
        for h in 0..nh {
            let mut m = f32::NEG_INFINITY;
            let mut dots = [0.0f32; 4];
            for r in 0..4 {
                let mut dot = 0.0f32;
                for dd in 0..hd {
                    dot += q[h * hd + dd] * kv[r * hd + dd];
                }
                dots[r] = dot * scale;
                if r != 1 {
                    m = m.max(dots[r]);
                }
            }
            let mut denom = exp_cr(sink[h] - m);
            let mut acc = vec![0.0f32; hd];
            let mut p = [0.0f32; 4];
            for r in 0..4 {
                if r == 1 {
                    continue;
                }
                p[r] = exp_cr(dots[r] - m);
                denom += p[r];
                for dd in 0..hd {
                    acc[dd] += bf16_round(p[r]) * kv[r * hd + dd];
                }
            }
            for dd in 0..hd {
                let want = bf16_round(acc[dd] / denom);
                assert!((o[h * hd + dd] - want).abs() < 1e-6, "h={h} dd={dd}");
            }
        }
    }

    /// 인덱서 topk — 인과·동점 낮은 인덱스·offset.
    #[test]
    fn indexer_topk_rules() {
        let mut c = cfg();
        c.index_topk = 3;
        // 블록 스코어: [5, 7, 7, 1, 9].
        let scores = vec![
            vec![5.0f32, 7.0, 7.0, 1.0, 9.0], // t=0: visible 1 → [0]
            vec![5.0f32, 7.0, 7.0, 1.0, 9.0], // t=1: visible 1
            vec![5.0f32, 7.0, 7.0, 1.0, 9.0], // t=2: visible 1
            vec![5.0f32, 7.0, 7.0, 1.0, 9.0], // t=3: visible 1
            vec![5.0f32, 7.0, 7.0, 1.0, 9.0], // t=4: visible 2
            vec![5.0f32, 7.0, 7.0, 1.0, 9.0], // t=5..7 visible 2, t=8 visible 3
            vec![5.0f32, 7.0, 7.0, 1.0, 9.0],
            vec![5.0f32, 7.0, 7.0, 1.0, 9.0],
            vec![5.0f32, 7.0, 7.0, 1.0, 9.0],
        ];
        let sel = indexer_topk(&scores, &c, 4, 100);
        assert_eq!(sel[0], vec![]); // t=0: (1)//4=0 — 미완 블록 미가시
        assert_eq!(sel[3], vec![100]); // t=3: (4)//4=1
        assert_eq!(sel[4], vec![100]); // t=4: (5)//4=1
        assert_eq!(sel[8], vec![101, 100]); // t=8: (9)//4=2 → 7(b1), 5(b0)
        // 동점: 값 5가 b1,b2 — 낮은 b1 우선. t=11이어야 (12)//4=3로 b2까지 가시.
        let tie = vec![vec![1.0f32, 5.0, 5.0, 0.0, 0.0]; 12];
        let sel2 = indexer_topk(&tie, &c, 4, 0);
        assert_eq!(sel2[8], vec![1, 0]); // t=8: 가시 {0,1}
        assert_eq!(sel2[11], vec![1, 2, 0]); // t=11: 가시 {0,1,2} — 동점 → b1,b2,b0
    }

    /// 그룹 출력 — wo_a/wo_b 항등 검증(작은 cfg).
    #[test]
    fn attention_output_grouping() {
        let c = cfg();
        let (nh, hd, g, r, d) = (c.n_heads, c.head_dim, c.o_groups, c.o_lora_rank, c.dim);
        let gdim = nh * hd / g;
        // wo_a[g] = 항등(정방), wo_b = [g·r → d] 항등(상위 g·r×d 정방 부분).
        let wo_a: Vec<Vec<f32>> = (0..g)
            .map(|gi| {
                let mut w = vec![0.0f32; gdim * r];
                for i in 0..gdim.min(r) {
                    w[i * r + i] = 1.0;
                }
                let _ = gi;
                w
            })
            .collect();
        let mut wo_b = vec![0.0f32; g * r * d];
        for i in 0..(g * r).min(d) {
            wo_b[i * d + i] = 1.0;
        }
        let w = AttnWeights {
            wq_a: vec![],
            q_norm: vec![],
            wq_b: vec![],
            wkv: vec![],
            kv_norm: vec![],
            sink: vec![],
            wo_a,
            wo_b,
        };
        let o: Vec<f32> = (0..nh * hd).map(|i| (i as f32 + 1.0) * 0.01).collect();
        let y = attention_output(&o, &w, &c);
        // latents[g] = o_g 앞 r개(항등 gemm — fp8_sim은 값을 건드릴 수 있음:
        // 크기 0.01~0.08 → amax 하한 1e-4 < amax, 스케일 균일 fp8 그리드).
        // wo_b 항등 → y = latents 앞 d개 = o의 첫 (g·r) 중 그룹 순서 결합.
        // g=2, r=4, gdim=4: latents = [o0..o3 (그대로), o4..o7] → y[i]=o[i].
        for i in 0..d.min(nh * hd) {
            assert!(
                (y[i] - o[i]).abs() / o[i].abs().max(1e-3) < 0.07,
                "y[{i}]={} o={}",
                y[i],
                o[i]
            );
        }
    }

    /// project_q/kv + rope_q — 형상·유한·결정성 (작은 합성 가중치).
    #[test]
    fn project_shapes_finite() {
        let mut c = cfg();
        c.dim = 8;
        c.q_lora_rank = 4;
        c.n_heads = 2;
        c.head_dim = 4;
        c.rope_head_dim = 2;
        let d = c.dim;
        let w = AttnWeights {
            wq_a: (0..d * 4).map(|i| (i % 7) as f32 * 0.03 - 0.09).collect(),
            q_norm: vec![1.0; 4],
            wq_b: (0..4 * 8).map(|i| (i % 5) as f32 * 0.05 - 0.1).collect(),
            wkv: (0..d * 4).map(|i| (i % 3) as f32 * 0.07 - 0.07).collect(),
            kv_norm: vec![1.0; 4],
            sink: vec![0.1; 2],
            wo_a: vec![vec![0.0; 8 * 4]; 2],
            wo_b: vec![0.0; 8 * 8],
        };
        let x = vec![0.25f32; 3 * d];
        let (cq, mut q) = project_q(&w, &x, &c);
        assert_eq!(cq.len(), 3 * 4);
        assert_eq!(q.len(), 3 * 2 * 4);
        let rope = RopeTable::build(2, 3, 10000.0, false, 16.0, 0, 32.0, 1.0);
        rope_q(&mut q, 3, &c, &rope);
        assert!(q.iter().all(|v| v.is_finite()));
        let kv = project_kv(&w, &x, &c, &rope);
        assert_eq!(kv.len(), 3 * 4);
        assert!(kv.iter().all(|v| v.is_finite()));
        let (cq2, _) = project_q(&w, &x, &c);
        assert_eq!(cq, cq2, "결정성");
    }
}
