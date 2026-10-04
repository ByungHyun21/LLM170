//! EXL3 CPU 수학·상태 (R4, plans/129 — decode.rs에서 순수 이동).
//! rms/rope/활성·GdnState/KvCache/SeqState·스펙 스냅샷·par_rows 풀·위상
//! 프로파일러. corr/속도 원장은 decode.rs 헤더 참조(B-3).
use std::cell::RefCell;
use std::collections::HashMap;

pub(super) struct PhaseAgg {
    count: u64,
    ns: u64,
}
thread_local! {
    static PHASES: RefCell<HashMap<&'static str, PhaseAgg>> = RefCell::new(HashMap::new());
}
pub(super) fn phase_on() -> bool {
    llm170_diag::dump::opts().key("exl3_phase")
}

/// usize 포인터 래퍼 — core::gdn::ar_pool 잡 캡처용(Send).
/// SAFETY: 주소의 생명은 run_par 완료 대기로 증명(호출 스코프 내).
#[derive(Clone, Copy)]
pub(super) struct SendP(pub(super) usize);
pub(super) struct PhaseGuard(std::time::Instant, &'static str);
impl Drop for PhaseGuard {
    fn drop(&mut self) {
        let ns = self.0.elapsed().as_nanos() as u64;
        PHASES.with(|m| {
            if let Ok(mut m) = m.try_borrow_mut() {
                let e = m.entry(self.1).or_insert(PhaseAgg { count: 0, ns: 0 });
                e.count += 1;
                e.ns += ns;
            }
        });
    }
}
pub(super) fn ph(name: &'static str) -> Option<PhaseGuard> {
    phase_on().then(|| PhaseGuard(std::time::Instant::now(), name))
}
pub(super) fn phase_report() {
    if !phase_on() {
        return;
    }
    PHASES.with(|m| {
        if let Ok(m) = m.try_borrow() {
            let mut v: Vec<_> = m.iter().collect();
            v.sort_by_key(|(_, a)| std::cmp::Reverse(a.ns));
            eprintln!("=== exl3 phase (wall — 중첩 포함: *_fwd 값은 하위 위상 합 포함) ===");
            for (k, a) in v {
                eprintln!(
                    "  {k:20} {:>7}회 {:>10.1}ms  평균 {:>8.3}ms",
                    a.count,
                    a.ns as f64 / 1e6,
                    a.ns as f64 / a.count as f64 / 1e6
                );
            }
        }
    });
}

/// GDN 시퀀스 상태 (per-layer per-head 128×128).
pub struct GdnState {
    /// [48 heads][128*128] — core::gdn 형식, **llama.cpp 헤드 순서**로 저장
    /// (hf_to_lc 순열을 초기화 시 고정 — plans/120 A1, 매 토큰 순열복사 제거).
    pub states: Vec<f32>,
    /// conv1d 링 [conv_k-1][conv_ch].
    pub conv: Vec<f32>,
}

/// KV 캐시 (full-attn 층용).
pub struct KvCache {
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    pub len: usize,
}

/// 전체 시퀀스 상태.
pub struct SeqState {
    pub gdn: Vec<GdnState>,
    pub kv: Vec<KvCache>,
    /// MTP 드래프트 층 자체 KV (plans/121 A2) — 1층.
    pub mtp_kv: Vec<KvCache>,
    /// 본체 잔차 hidden(output_norm 전) — MTP h_in 스냅샷(qwen35 mtp_h 관례).
    pub last_h: Vec<f32>,
    /// 마지막 타깃 로짓(스펙 검증 기준 — 불필요 시 비움).
    pub last_logits: Vec<f32>,
    pub last_tok: u32,
    pub pos: u32,
}

// ── 유틸리티 ── // PhaseAgg..phase_report + state structs

pub(super) fn rms_norm(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len();
    let ss: f32 = x.iter().map(|&v| v * v).sum();
    let inv = 1.0 / ((ss / n as f32 + eps).sqrt());
    x.iter().zip(w.iter()).map(|(&v, &g)| v * inv * g).collect()
}

pub(super) fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

pub(super) fn softplus(x: f32) -> f32 {
    if x > 20.0 { x } else { (1.0 + x.exp()).ln() }
}

pub(super) fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

pub(super) fn l2_norm(x: &[f32], eps: f32) -> Vec<f32> {
    let ss: f32 = x.iter().map(|&v| v * v).sum();
    let inv = 1.0 / ((ss + eps).sqrt());
    x.iter().map(|&v| v * inv).collect()
}

/// RoPE — core::ops::rope_head 위임.
pub(super) fn rope(head: &mut [f32], pos: u32, n_rot: usize, base: f32) {
    // core의 rope_head는 half-split 방식 — 직접 호출.
    // SAFETY: head가 n_rot보다 크거나 같음을 보장.
    llm170_core::ops::rope_head(head, pos, n_rot, base);
}

// ── GDN 단일 토큰 (t=1) ──

/// 시퀀스 상태 초기화.
pub fn new_seq_state(n_layers: usize, ctx_len: usize) -> SeqState {
    let n_v = 48;
    let d_state = 128;
    let conv_ch = 10240;
    let conv_k = 4;
    let n_full_attn = 16; // 64/4
    let _n_head = 24;

    SeqState {
        gdn: (0..n_layers)
            .map(|_| GdnState {
                states: vec![0f32; n_v * d_state * d_state],
                conv: vec![0f32; (conv_k - 1) * conv_ch],
            })
            .collect(),
        kv: (0..n_full_attn)
            .map(|_| KvCache {
                k: vec![0f32; ctx_len * 4 * 256],
                v: vec![0f32; ctx_len * 4 * 256],
                len: 0,
            })
            .collect(),
        mtp_kv: (0..1)
            .map(|_| KvCache {
                k: vec![0f32; ctx_len * 4 * 256],
                v: vec![0f32; ctx_len * 4 * 256],
                len: 0,
            })
            .collect(),
        last_h: Vec::new(),
        last_logits: Vec::new(),
        last_tok: 0,
        pos: 0,
    }
}

/// 원시 포인터 usize 래퍼 — par_rows 잡 캡처용(gdn.rs SendPtr와 동일 계약).
#[derive(Clone, Copy)]
pub(super) struct PP(pub(super) usize);

/// 행 병렬 — thread::scope 청크. body는 Sync(공유 읽기)이며 행별 분리
/// 쓰기는 원시 포인터(PP)로 수행한다(호출 스코프 내 유효 — scope join 증명).
pub(super) fn par_rows(t_rows: usize, body: impl Fn(usize) + Sync + Send) {
    if t_rows == 0 {
        return;
    }
    // 스레드 수: 전 코어(SMT 포함) 최적 — 2026-10-03 A/B 측정(pp512 3회
    // 중앙값): 32=60.53 > 24=59.66 > 16=59.31 t/s. 물리코어 제한 가설 기각.
    let nt = std::thread::available_parallelism()
        .map(|v| v.get())
        .unwrap_or(8)
        .min(t_rows);
    let per = t_rows.div_ceil(nt);
    std::thread::scope(|s| {
        for lo in (0..t_rows).step_by(per) {
            let hi = (lo + per).min(t_rows);
            let body = &body;
            s.spawn(move || (lo..hi).for_each(body));
        }
    });
}

pub(super) fn argmax32(v: &[f32]) -> u32 {
    v.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
}

pub(super) fn greedy_ref(v: &[f32]) -> u32 {
    argmax32(v)
}

// ── 스펙 라운드 (plans/121 A2) ─────────────────────────────────────────
// k 드래프트(mtp 체인) → 상태 스냅샷 → [d0..d_{k-1}] T=k 배치 타깃 forward
// → 행별 argmax 검증 → 발산 시 롤백+수용 접두 재실행. GDN 201MB 클론이
// 라운드당 ~25ms(7% 수준) — KV는 len 절단만(멱등 재기입).

/// 스펙 상태 스냅샷 — GDN/conv 클론 + KV len + pos + hidden/로짓.
pub struct SpecSnap {
    gdn_states: Vec<Vec<f32>>,
    conv: Vec<f32>,
    kv_lens: Vec<usize>,
    /// 발산 롤백용 — 드래프트가 늘린 mtp KV len도 되돌린다(잔여
    /// 슬롯 내용은 헤드 재기입이 덮는다).
    mtp_kv_lens: Vec<usize>,
    pos: u32,
    last_h: Vec<f32>,
    last_logits: Vec<f32>,
}

#[allow(dead_code)] // 프레임 롤백 대체 전 CPU 스냅샷 — 참조 보존(plans/121 tg)
pub(super) fn spec_snap(seq: &SeqState) -> SpecSnap {
    SpecSnap {
        gdn_states: seq.gdn.iter().map(|g| g.states.clone()).collect(),
        conv: seq
            .gdn
            .iter()
            .flat_map(|g| g.conv.iter().copied())
            .collect(),
        kv_lens: seq.kv.iter().map(|k| k.len).collect(),
        mtp_kv_lens: seq.mtp_kv.iter().map(|k| k.len).collect(),
        pos: seq.pos,
        last_h: seq.last_h.clone(),
        last_logits: seq.last_logits.clone(),
    }
}

#[allow(dead_code)]
pub(super) fn spec_restore(seq: &mut SeqState, snap: &SpecSnap) {
    for (g, st) in seq.gdn.iter_mut().zip(snap.gdn_states.iter()) {
        g.states.copy_from_slice(st);
    }
    let mut off = 0usize;
    for g in seq.gdn.iter_mut() {
        let n = g.conv.len();
        g.conv.copy_from_slice(&snap.conv[off..off + n]);
        off += n;
    }
    for (k, &l) in seq.kv.iter_mut().zip(snap.kv_lens.iter()) {
        k.len = l;
    }
    for (k, &l) in seq.mtp_kv.iter_mut().zip(snap.mtp_kv_lens.iter()) {
        k.len = l;
    }
    seq.pos = snap.pos;
    seq.last_h.copy_from_slice(&snap.last_h);
    seq.last_logits.copy_from_slice(&snap.last_logits);
}
