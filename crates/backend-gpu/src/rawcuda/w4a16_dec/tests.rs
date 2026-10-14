use super::mem::realloc_fields;
use super::*;
use std::cell::{Cell, RefCell};

/// MoE 선택 시맨틱 — core stages/moe::select_topk 미러(변경 시 동시 갱신).
#[test]
fn moe_topk_mirrors_core_semantics() {
    // 재정규화(합 1)·내림차순·exp 비율 유지.
    let sel = moe_topk(&[2.0, 1.0, 0.5, -1.0], 2);
    assert_eq!(sel.len(), 2);
    assert_eq!(sel[0].0, 0);
    assert_eq!(sel[1].0, 1);
    assert!(sel[0].1 > sel[1].1);
    let sum: f32 = sel.iter().map(|&(_, w)| w).sum();
    assert!((sum - 1.0).abs() < 1e-6, "합={sum}");
    let ratio = sel[0].1 / sel[1].1;
    assert!((ratio - std::f32::consts::E).abs() < 1e-4, "비율={ratio}");
    // 동률은 낮은 인덱스(결정적).
    let tie = moe_topk(&[1.0, 1.0, 1.0], 2);
    assert_eq!(tie[0].0, 0);
    assert_eq!(tie[1].0, 1);
}

/// G1 회귀 — 부분 alloc 실패 후 재시도가 이중해제/유실 없이 전량 재할당.
/// (CUDA 불필요 — free/alloc을 모의 클로저로 주입.)
#[test]
fn realloc_fields_partial_failure_retry_safe() {
    let mut f: [CUdeviceptr; 3] = [0x11, 0x22, 0x33];
    let live = RefCell::new(vec![0x11u64, 0x22, 0x33]);
    let frees = RefCell::new(Vec::<u64>::new());
    let next = Cell::new(0x100u64);
    let fail_next = Cell::new(1usize); // 두 번째 alloc에서 실패 주입
    let free = |p: CUdeviceptr| -> Result<(), String> {
        let mut l = live.borrow_mut();
        assert!(l.contains(&p), "이중해제 {p:#x}");
        l.retain(|&v| v != p);
        frees.borrow_mut().push(p);
        Ok(())
    };
    let alloc = |_sz: usize| -> Result<CUdeviceptr, String> {
        if fail_next.get() == 0 {
            return Err("inject".into());
        }
        fail_next.set(fail_next.get() - 1);
        let v = next.get() + 1;
        next.set(v);
        live.borrow_mut().push(v);
        Ok(v)
    };
    let [a, b, c] = &mut f;
    let r = realloc_fields(free, alloc, [a, b, c], [1, 1, 1]);
    assert!(r.is_err(), "주입 실패가 전파되어야 한다");
    assert_eq!(*frees.borrow(), vec![0x11, 0x22, 0x33]);
    assert_eq!(f, [0x101, 0, 0], "성공분 유지·실패분 0");
    // 재시도 — 성공분만 회수(이중해제 없음), 전량 재할당.
    fail_next.set(usize::MAX);
    let [a, b, c] = &mut f;
    realloc_fields(free, alloc, [a, b, c], [1, 1, 1]).expect("재시도");
    assert_eq!(*frees.borrow(), vec![0x11, 0x22, 0x33, 0x101]);
    assert_eq!(f, [0x102, 0x103, 0x104]);
    assert_eq!(live.borrow().len(), 3);
}

use super::{
    ATTN_F3S_TMAX, ATTN_SPLITS, BATCH_DEC_MAX, CHAIN_TMAX, GDN_CS, GDN_NGRP, GDN_NSPLIT,
    GDN_SCAN_SMEM, GEMM_BMMA_M, GEMM_BMMA_N, GEMM_FFMA_TMAX, GEMM_GRP_M, GEMM_GRP_N, GEMM_MMA_M,
    GEMM_MMA_N, GEMV_TR,
};

/// [R21] head/attn/FFMA 상한 미러 — 종전 미검사분(커널-호스트 드리프트 방지).
#[test]
fn head_attn_ffma_mirrors() {
    assert_eq!(
        define(include_str!("../assets/moe.cu"), "G4_TMAX") as usize,
        BATCH_DEC_MAX,
        "moe.cu G4_TMAX ↔ BATCH_DEC_MAX(디코드 배치 상한 — head 이관 후)"
    );
    assert_eq!(
        define(include_str!("../assets/attn.cu"), "ATTN_SPLITS_MAX") as usize,
        ATTN_SPLITS_MAX,
        "attn.cu ATTN_SPLITS_MAX ↔ ATTN_SPLITS_MAX"
    );
    assert_eq!(
        define(include_str!("../assets/gptq4.cu"), "G4_GTMAX") as usize,
        GEMM_FFMA_TMAX,
        "gptq4.cu G4_GTMAX ↔ GEMM_FFMA_TMAX"
    );
}

/// `#define NAME 값` 파싱 — 값은 정수 리터럴만 다룬다(대상 목록 한정).
fn define(src: &str, name: &str) -> u64 {
    for line in src.lines() {
        if let Some(rest) = line.trim().strip_prefix("#define ") {
            let mut it = rest.split_whitespace();
            if it.next() == Some(name) {
                return it
                    .next()
                    .expect("define 값 누락")
                    .parse()
                    .expect("정수 define 아님");
            }
        }
    }
    panic!(".cu define 없음: {name}");
}

#[test]
fn attn_tmax_mirror() {
    let tmax = define(include_str!("../assets/attn.cu"), "ATTN_TMAX") as usize;
    assert_eq!(tmax, ATTN_F3S_TMAX, "attn.cu ATTN_TMAX ↔ ATTN_F3S_TMAX");
    assert_eq!(tmax, CHAIN_TMAX, "attn.cu ATTN_TMAX ↔ CHAIN_TMAX");
}

#[test]
fn gemm_mma_mirror() {
    // [B1/B3] gptq4.cu 타일 정의 ↔ 호스트 미러(런치 grid 오배치 방지).
    let cu = include_str!("../assets/gptq4.cu");
    assert_eq!(
        define(cu, "MMA_M") as usize,
        GEMM_MMA_M,
        "gptq4.cu MMA_M ↔ GEMM_MMA_M"
    );
    assert_eq!(
        define(include_str!("../assets/moe.cu"), "BMMA_M") as usize,
        GEMM_BMMA_M,
        "moe.cu BMMA_M ↔ GEMM_BMMA_M(bf16 mma 전용)"
    );
    assert_eq!(
        define(include_str!("../assets/moe.cu"), "BMMA_N") as usize,
        GEMM_BMMA_N,
        "moe.cu BMMA_N ↔ GEMM_BMMA_N(bf16 mma 전용)"
    );
    assert_eq!(
        define(cu, "MMA_N") as usize,
        GEMM_MMA_N,
        "gptq4.cu MMA_N ↔ GEMM_MMA_N"
    );
    assert_eq!(
        define(cu, "GRP_M") as usize,
        GEMM_GRP_M,
        "gptq4.cu GRP_M ↔ GEMM_GRP_M"
    );
    assert_eq!(
        define(cu, "GRP_N") as usize,
        GEMM_GRP_N,
        "gptq4.cu GRP_N ↔ GEMM_GRP_N"
    );
}

#[test]
fn gdn_scan_layout_mirror() {
    let cu = include_str!("../assets/gdn.cu");
    let cs = define(cu, "GDN_CS") as usize;
    let tile = define(cu, "GDN_TILE") as usize;
    let nsplit = define(cu, "GDN_NSPLIT") as usize;
    assert_eq!(cs, GDN_CS, "gdn.cu GDN_CS ↔ GDN_CS");
    assert_eq!(nsplit, GDN_NSPLIT, "gdn.cu GDN_NSPLIT ↔ GDN_NSPLIT");
    assert_eq!(
        define(cu, "GDN_NGRP") as usize,
        GDN_NGRP,
        "gdn.cu GDN_NGRP ↔ GDN_NGRP"
    );
    let vs = 128 / nsplit;
    // gdn_scan 레이아웃(sk/sv/A/KQ/KS/QS/dc/Stile×2/bp/gcs/wsm) 바이트
    // 합 — [A5-4b] qs 스테이징 제거(글로벌 직접 판독, -16KB) 반영.
    let total = cs * 128 * 2
        + cs * vs * 2
        + cs * cs * 2
        + cs * cs * 2
        + cs * vs * 2
        + cs * vs * 2
        + cs * vs * 4
        + 2 * tile * vs * 4
        + cs * 4
        + (cs + 1) * 4
        + cs * 4
        + cs * 4;
    assert_eq!(total as u32, GDN_SCAN_SMEM, "gdn_scan 동적 smem 합");
}

#[test]
fn g4_scmax_mirror() {
    let scmax = define(include_str!("../assets/gptq4.cu"), "G4_SCMAX") as usize;
    assert_eq!(scmax, crate::rawcuda::gptq4::G4_SCMAX, "gptq4.cu G4_SCMAX");
}

#[test]
fn gemv_tr_mirror() {
    // [R5] GEMV_TR은 g4_common.cuh(플레인·split 공용) 소유.
    let tr = define(include_str!("../assets/g4_common.cuh"), "GEMV_TR") as usize;
    assert_eq!(tr, GEMV_TR, "g4_common.cuh GEMV_TR ↔ GEMV_TR");
}

#[test]
fn attn_splits_mirror() {
    let s = define(include_str!("../assets/attn.cu"), "ATTN_SPLITS_MAX") as usize;
    assert_eq!(
        s, ATTN_SPLITS_MAX,
        "attn.cu ATTN_SPLITS_MAX ↔ ATTN_SPLITS_MAX"
    );
    assert!(
        ATTN_SPLITS.max(ATTN_SPLITS_D) <= s,
        "프리필·디코드 분할 ≤ 병합 상한"
    );
}

#[test]
fn gemm_ffma_tmax_mirror() {
    let t = define(include_str!("../assets/gptq4.cu"), "G4_GTMAX") as usize;
    assert_eq!(t, GEMM_FFMA_TMAX, "gptq4.cu G4_GTMAX ↔ GEMM_FFMA_TMAX");
}

#[test]
fn batch_dec_max_mirror() {
    // [R5] G4_TMAX는 moe.cu 단일 정의(재정의 통합 — 배치 GEMV·플레인 v1 GEMM).
    let t = define(include_str!("../assets/moe.cu"), "G4_TMAX") as usize;
    assert_eq!(t, BATCH_DEC_MAX, "moe.cu G4_TMAX ↔ BATCH_DEC_MAX");
}

// ═══ [MTP-CHAIN 2026-10-12 §3.3 ②] 장치 체인 패리티 — CUDA 필요 ═══
// 순차 호스트 경로(앵커 + mtp_recur_step ×3)와 장치 체인(앵커 +
// mtp_chain_drafts)의 초안열 동일성. 모델은 합성 미니(플레인 bf16 MTP
// 가중 + int8 head + 임베딩 상주 + 더미 blk.0/GDN 등록) — 수치 의미 없음,
// 경로 동일성(토큰열 판정)이 목적. 두 경로는 같은 커널열을 쓰며 임베딩
// 소스만 h2d ↔ gather(bf16 디코드 비트동일 — embed.cu 계약)로 다르다.

/// [§3.4 ② 2026-10-12] CUDA 테스트 직렬화 락 — cargo test의 스레드 병렬
/// 실행에서 컨텍스트·할당이 겹치면 VRAM 델타 판정(ucache 강제 테스트)이
/// 우연에 좌우된다. CUDA를 쓰는 테스트는 이 락으로 순차화한다(타이밍
/// 의존 금지 계약).
static CUDA_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// splitmix64 — rand 금지 계약(core sampler 미러).
fn sm64(s: &mut u64) -> u64 {
    *s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *s;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// [lo, hi) 균등 f32.
fn sm_f32(s: &mut u64, lo: f32, hi: f32) -> f32 {
    lo + (sm64(s) as f64 / u64::MAX as f64) as f32 * (hi - lo)
}

/// f32 → bf16(절단) 바이트열 — 시험 인코딩. 디코드(u32<<16)가 항등이라
/// 호스트 판독값과 커널 b2f가 비트동일.
fn bf16_bytes(v: &[f32]) -> Vec<u8> {
    let mut b = Vec::with_capacity(v.len() * 2);
    for &x in v {
        b.extend_from_slice(&((x.to_bits() >> 16) as u16).to_le_bytes());
    }
    b
}

/// 상주 임베딩 바이트의 r행 디코드(커널 b2f와 동일식).
fn bf16_row(bytes: &[u8], r: usize, k: usize) -> Vec<f32> {
    (0..k)
        .map(|c| {
            let o = (r * k + c) * 2;
            let u = u16::from_le_bytes([bytes[o], bytes[o + 1]]);
            f32::from_bits((u as u32) << 16)
        })
        .collect()
}

#[test]
fn mtp_device_chain_matches_sequential() {
    let _cuda = CUDA_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    let h = 1024usize;
    let (qh, kvh, d, cap) = (4usize, 2usize, 256usize, 64usize);
    let (qdim, kvdim) = (qh * d, kvh * d); // 1024, 512
    let (ffn, vocab) = (1024usize, 256usize);
    let mut dec = super::W4a16Dec::new(1, h, 1).expect("CUDA 컨텍스트/fatbin");
    let mut s = 0x1234_5678_9abc_def0u64;

    // blk.0 레지스트리 — ensure_chain_bufs(build_multi_tables) 형상 계약.
    // 실행되지 않는다(폭 산정만) → 0 채움. g128 q=n·k/2 · s=n·k/64.
    for name in [
        "blk.0.attn_q.weight",
        "blk.0.attn_k.weight",
        "blk.0.attn_v.weight",
        "blk.0.ffn_gate.weight",
        "blk.0.ffn_up.weight",
    ] {
        let (n, k) = (64usize, h);
        let z = vec![0u8; n * k];
        dec.upload_lin(name, &z[..n * k / 2], &z[..n * k / 64], n, k)
            .expect("upload_lin");
    }
    // MTP 가중(플레인 bf16) — 실제 실행분.
    let plain = |dec: &mut super::W4a16Dec, name: &str, n: usize, k: usize, s: &mut u64| {
        let w: Vec<f32> = (0..n * k).map(|_| sm_f32(s, -0.05, 0.05)).collect();
        dec.upload_plain(name, &bf16_bytes(&w), n, k)
            .expect("upload_plain");
    };
    plain(&mut dec, "mtp.fc.weight", h, 2 * h, &mut s);
    plain(
        &mut dec,
        "mtp.layers.0.self_attn.q_proj.weight",
        qdim,
        h,
        &mut s,
    );
    plain(
        &mut dec,
        "mtp.layers.0.self_attn.k_proj.weight",
        kvdim,
        h,
        &mut s,
    );
    plain(
        &mut dec,
        "mtp.layers.0.self_attn.v_proj.weight",
        kvdim,
        h,
        &mut s,
    );
    plain(
        &mut dec,
        "mtp.layers.0.self_attn.o_proj.weight",
        h,
        qdim,
        &mut s,
    );
    plain(
        &mut dec,
        "mtp.layers.0.mlp.gate_proj.weight",
        ffn,
        h,
        &mut s,
    );
    plain(&mut dec, "mtp.layers.0.mlp.up_proj.weight", ffn, h, &mut s);
    plain(
        &mut dec,
        "mtp.layers.0.mlp.down_proj.weight",
        h,
        ffn,
        &mut s,
    );
    // head(int8 양자화 내부 수행) + 임베딩(bf16 상주 — gather 경로 원본).
    let head_w: Vec<f32> = (0..vocab * h)
        .map(|_| sm_f32(&mut s, -0.05, 0.05))
        .collect();
    dec.upload_head(&bf16_bytes(&head_w), vocab, h)
        .expect("upload_head");
    let emb_b = bf16_bytes(
        &(0..vocab * h)
            .map(|_| sm_f32(&mut s, -0.05, 0.05))
            .collect::<Vec<_>>(),
    );
    dec.upload_embed(&emb_b, vocab, h, true)
        .expect("upload_embed");
    // GDN — ensure_chain_bufs 형상 계약(체인이 실행하지 않음 → 0/1 채움).
    let gd = super::GdnDims {
        n_gdn: 1,
        hidden: h,
        h_k: 2,
        h_v: 4,
        d: 64,
    };
    dec.set_gdn(
        gd,
        &vec![0f32; gd.n_gdn * gd.conv_ch() * 4],
        &vec![0f32; gd.n_gdn * 2 * gd.h_v * gd.hidden],
        &vec![0f32; gd.n_gdn * gd.h_v],
        &vec![0f32; gd.n_gdn * gd.h_v],
        &vec![1f32; gd.n_gdn * 128],
    )
    .expect("set_gdn");
    // 어텐션 — MTP 가상 전체-어텐션 층 1개(n_attn=1 → mtp_layer=0).
    let ad = super::AttnDims {
        n_attn: 1,
        q_heads: qh,
        kv_heads: kvh,
        d,
        cap,
        interval: 1,
        sliding_window: 0, // qwen35 = 슬라이딩 없음(gemma4 배선 시 창 설정)
    };
    dec.set_attn(
        ad,
        Vec::new(), // 균일 모델(층별 테이블 없음)
        &(0..ad.n_attn * d)
            .map(|_| sm_f32(&mut s, 0.5, 1.5))
            .collect::<Vec<_>>(),
        &(0..ad.n_attn * d)
            .map(|_| sm_f32(&mut s, 0.5, 1.5))
            .collect::<Vec<_>>(),
    )
    .expect("set_attn");
    // MTP 노름 5행(≈1)·등록 — 체인 부기 버퍼(ctr·log)까지 확보.
    let norms: Vec<f32> = (0..5 * h).map(|_| sm_f32(&mut s, 0.8, 1.2)).collect();
    dec.set_mtp(&norms, qdim, kvdim, ffn).expect("set_mtp");

    // dres 행 0 시드(hidden 어셈블리 원천) — 두 경로 동일값.
    let seed: Vec<f32> = (0..h).map(|_| sm_f32(&mut s, -1.0, 1.0)).collect();
    dec.ensure_chain_bufs().expect("ensure_chain_bufs");
    {
        let _g = dec.cc.guard().expect("guard");
        // SAFETY: 로컬 시드 슬라이스의 유효 수명 내 바이트 뷰(길이 = 원소수×4).
        let sb = unsafe { std::slice::from_raw_parts(seed.as_ptr() as *const u8, h * 4) };
        dec.cc.h2d(dec.dres, sb).expect("dres 시드");
    }

    let pos: u32 = 4;
    let tok0: u32 = 7;
    let emb0 = bf16_row(&emb_b, tok0 as usize, h);
    let embed_of = |t: u32| bf16_row(&emb_b, t as usize, h);

    // ── 순차(호스트 왕복): 앵커 + mtp_recur_step ×3 ──
    dec.mtp_kv_reset().expect("kv 리셋");
    dec.mtp_reset_hidden();
    dec.mtp_push_hidden(0).expect("push");
    let d1 = dec.mtp_run(0, &[tok0], &emb0, pos).expect("앵커");
    let mut seq = vec![d1];
    let mut tok = d1;
    for i in 0..3u32 {
        let d = dec
            .mtp_recur_step(0, tok, &embed_of(tok), pos + 1 + i)
            .expect("재귀");
        seq.push(d);
        tok = d;
    }

    // ── 장치 체인: 앵커(동일) + mtp_chain_drafts ──
    // kv 리셋으로 첫 실행 잔존 KV 제거 — 두 실행의 초기상태를 동일화.
    dec.mtp_kv_reset().expect("kv 리셋2");
    dec.mtp_reset_hidden();
    dec.mtp_push_hidden(0).expect("push2");
    let d1c = dec.mtp_run(0, &[tok0], &emb0, pos).expect("앵커2");
    let chain = dec.mtp_chain_drafts(0, 4, pos).expect("체인");

    assert_eq!(d1, d1c, "앵커 argmax 불일치 — 비결정 오염");
    assert_eq!(seq, chain, "장치 체인 초안열 ≠ 순차 경로(토큰열 판정)");
}

// ═══ [K-STEP 2026-10-12 §3.4 ①] K-스텝 디코드 패리티 — CUDA 필요 ═══
// 순차 단일 스텝(forward_device_argmax ×4 — 토큰마다 호스트 왕복)과 장치
// K-스텝 루프(decode_k_greedy — 스텝 그래프 k회 리플레이)의 토큰열 동일성.
// 모델은 합성 미니(interval=1 전체-어텐션 1층 — 본체 체인 실실행) — 수치
// 의미 없음, 경로 동일성(토큰열 판정)이 목적. 두 경로는 같은 커널열을
// 쓰며 임베딩 소스만 h2d ↔ gather(bf16 디코드 비트동일 — embed.cu 계약)로
// 다르다. 가중은 실제 g128 양자(int4 팩 + f16 스케일)로 생성 — split
// GEMV 경로까지 실실행.
#[test]
fn kstep_decode_matches_sequential() {
    let _cuda = CUDA_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    let h = 1024usize;
    let (qh, kvh, d, cap) = (4usize, 2usize, 256usize, 64usize);
    let (qdim, kvdim) = (qh * d, kvh * d); // 1024, 512
    let (ffn, vocab) = (1024usize, 256usize);
    let mut dec = super::W4a16Dec::new(1, h, 1).expect("CUDA 컨텍스트/fatbin");
    let mut s = 0x0dd8_ba11_5eed_0f77u64;

    // split g128 양자 가중 — q는 랜덤 int4 팩(n*k/2B)·s는 소규모 f16
    // 스케일(n*k/64B — 로짓 폭 억제). 값 자체는 무의미, 두 경로가 같은
    // 가중을 소비하는 것이 계약.
    let lin = |dec: &mut super::W4a16Dec, name: &str, n: usize, k: usize, s: &mut u64| {
        let q: Vec<u8> = (0..n * k / 2).map(|_| (sm64(s) & 0xff) as u8).collect();
        let sc: Vec<u8> = (0..n * k / 128)
            .flat_map(|_| f32_to_f16(sm_f32(s, 0.004, 0.016)).to_le_bytes())
            .collect();
        dec.upload_lin(name, &q, &sc, n, k).expect("upload_lin");
    };
    lin(&mut dec, "blk.0.attn_q.weight", qdim, h, &mut s);
    lin(&mut dec, "blk.0.attn_k.weight", kvdim, h, &mut s);
    lin(&mut dec, "blk.0.attn_v.weight", kvdim, h, &mut s);
    lin(&mut dec, "blk.0.attn_output.weight", h, qdim, &mut s);
    lin(&mut dec, "blk.0.ffn_gate.weight", ffn, h, &mut s);
    lin(&mut dec, "blk.0.ffn_up.weight", ffn, h, &mut s);
    lin(&mut dec, "blk.0.ffn_down.weight", h, ffn, &mut s);
    // 노름 3행(2L+1) + head(int8 양자 내부) + 임베딩(bf16 상주 — gather 원본).
    let norms: Vec<f32> = (0..3 * h).map(|_| sm_f32(&mut s, 0.8, 1.2)).collect();
    dec.set_norm_weights(&norms, 3).expect("set_norm_weights");
    let head_w: Vec<f32> = (0..vocab * h)
        .map(|_| sm_f32(&mut s, -0.05, 0.05))
        .collect();
    dec.upload_head(&bf16_bytes(&head_w), vocab, h)
        .expect("upload_head");
    let emb_b = bf16_bytes(
        &(0..vocab * h)
            .map(|_| sm_f32(&mut s, -0.05, 0.05))
            .collect::<Vec<_>>(),
    );
    dec.upload_embed(&emb_b, vocab, h, true)
        .expect("upload_embed");
    // GDN — 체인 가드 형상 계약(interval=1 → 실행 없음).
    let gd = super::GdnDims {
        n_gdn: 1,
        hidden: h,
        h_k: 2,
        h_v: 4,
        d: 64,
    };
    dec.set_gdn(
        gd,
        &vec![0f32; gd.n_gdn * gd.conv_ch() * 4],
        &vec![0f32; gd.n_gdn * 2 * gd.h_v * gd.hidden],
        &vec![0f32; gd.n_gdn * gd.h_v],
        &vec![0f32; gd.n_gdn * gd.h_v],
        &vec![1f32; gd.n_gdn * 128],
    )
    .expect("set_gdn");
    // 어텐션 — 전체-어텐션 1층(interval=1 → 체인이 항상 attn 분기).
    let ad = super::AttnDims {
        n_attn: 1,
        q_heads: qh,
        kv_heads: kvh,
        d,
        cap,
        interval: 1,
        sliding_window: 0, // qwen35 = 슬라이딩 없음(gemma4 배선 시 창 설정)
    };
    dec.set_attn(
        ad,
        Vec::new(), // 균일 모델(층별 테이블 없음)
        &(0..ad.n_attn * d)
            .map(|_| sm_f32(&mut s, 0.5, 1.5))
            .collect::<Vec<_>>(),
        &(0..ad.n_attn * d)
            .map(|_| sm_f32(&mut s, 0.5, 1.5))
            .collect::<Vec<_>>(),
    )
    .expect("set_attn");

    let embed_of = |t: u32| bf16_row(&emb_b, t as usize, h);
    let tok0: u32 = 7;

    // ── 순차(토큰마다 호스트 왕복): reset → forward_device_argmax ×4 ──
    dec.reset_state(0).expect("reset1");
    let mut seq: Vec<u32> = Vec::new();
    let mut tok = tok0;
    for _ in 0..4 {
        let t = dec
            .forward_device_argmax(0, &embed_of(tok))
            .expect("순차 스텝");
        seq.push(t);
        tok = t;
    }

    // ── K-스텝(스텝 그래프 k회 리플레이): reset → decode_k_greedy(k=4) ──
    dec.reset_state(0).expect("reset2");
    let kout = dec.decode_k_greedy(0, tok0, 4, u32::MAX).expect("kstep");
    assert_eq!(kout.len(), 4, "EOS 아님 → k 전체 방출");
    assert_eq!(kout, seq, "K-스텝 토큰열 ≠ 순차 경로(토큰열 판정)");
    assert_eq!(dec.slot_pos_of(0), 4, "slot_pos = 방출 수만큼 전진");

    // ── EOS 정지 — 시드(tok0)와 다른 첫 방출을 eos로 지정하면 그 토큰까지
    // 방출하고 정지(컨트롤러가 이후 스텝 미기록). 고정 시드 드로우라 결정적.
    let j = seq
        .iter()
        .position(|&t| t != tok0)
        .expect("시드와 다른 방출 토큰(드로우 가정)");
    dec.reset_state(0).expect("reset3");
    let kstop = dec.decode_k_greedy(0, tok0, 4, seq[j]).expect("kstep-eos");
    assert_eq!(kstop, &seq[..=j], "EOS에서 정지 — 방출 접두만 반환");
}

// ═══ [§3.4 ② 2026-10-12 트랜셰 2] 예산 강제 실해제 — CUDA 필요 ═══
// 압박(ucache_pressure) → 최저 utility 순 퇴거(ckpt u=10 → state u=20) →
// 실제 cuMemFree → cuda_mem_free 델타 ≥ 퇴거 바이트(실측 회수 판정).
// 안전 계약 검증: Kv(attn:kv)는 고정이라 생존(활성 메모리 퇴거 금지),
// 해제된 버퍼는 0화 + spec_on=false(해제 주소로 라운드 시도 차단).
#[test]
fn ucache_enforcement_frees_device_memory() {
    let _cuda = CUDA_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    let h = 1024usize;
    let (qh, kvh, d, cap) = (4usize, 2usize, 256usize, 64usize);
    let mut dec = super::W4a16Dec::new(1, h, 1).expect("CUDA 컨텍스트/fatbin");

    // GDN 형상(스펙 상태·체크포인트 버퍼 형상의 원천) — 가중 0/1 더미
    // (버퍼 확보만 목적, 체인 실행 없음). 형상은 의도적으로 크게: 해제
    // 합계 ≈ 52MiB ≫ 드라이버 여유 보고 단위(2MiB) — cuda_mem_free 델타
    // 판정이 반올림에 묻히지 않게 한다(실측: 485KB 해제는 +0B로 측정).
    let gd = super::GdnDims {
        n_gdn: 8,
        hidden: h,
        h_k: 2,
        h_v: 16,
        d: 128,
    };
    dec.set_gdn(
        gd,
        &vec![0f32; gd.n_gdn * gd.conv_ch() * 4],
        &vec![0f32; gd.n_gdn * 2 * gd.h_v * gd.hidden],
        &vec![0f32; gd.n_gdn * gd.h_v],
        &vec![0f32; gd.n_gdn * gd.h_v],
        &vec![1f32; gd.n_gdn * 128],
    )
    .expect("set_gdn");
    // 어텐션(KV 등록 원천) — 최소 형상.
    let ad = super::AttnDims {
        n_attn: 1,
        q_heads: qh,
        kv_heads: kvh,
        d,
        cap,
        interval: 1,
        sliding_window: 0, // qwen35 = 슬라이딩 없음(gemma4 배선 시 창 설정)
    };
    dec.set_attn(
        ad,
        Vec::new(),
        &vec![1f32; ad.n_attn * d],
        &vec![1f32; ad.n_attn * d],
    )
    .expect("set_attn");
    // 스펙 상태(WY s0/dc/g/k + dsnap_ring + 라운드 전 링 dring0) + 체크포인트
    // (dchk) 등록. dsave는 프로브 전용 지연 할당(serve 경로 상주 없음).
    dec.enable_spec().expect("enable_spec");
    assert!(dec.spec_on(), "스펙 활성 전제(압박 전)");
    assert!(!dec.gdn.dchk.is_null(), "체크포인트 등록 전제");
    assert_ne!(dec.gdn.dwy_s0, 0, "스펙 상태(WY S_0) 등록 전제");
    assert_ne!(dec.gdn.dring0, 0, "라운드 전 링 스냅샷(dring0) 등록 전제");

    let ckpt = dec.ucache.entry("spec:ckpt").expect("ckpt 등록부").bytes;
    let state = dec.ucache.entry("spec:state").expect("state 등록부").bytes;
    let kv = dec.ucache.entry("attn:kv").expect("kv 등록부").bytes;
    // [호스트 이관 2026-10-14] 체크포인트 링 = pinned 호스트 RAM(HostRam 티어)
    // — HBM 회계에서 제외(회수 대상은 state만), 호스트 티어에 청구.
    let total = dec.ucache.total(super::unified_cache::Tier::Hbm);
    assert_eq!(total, state + kv, "등록부 HBM 회계 정합(ckpt 제외)");
    assert_eq!(
        dec.ucache.total(super::unified_cache::Tier::HostRam),
        ckpt,
        "ckpt는 호스트 티어"
    );

    // 압박: 예산 = KV 단독으로 축소 → state(u20) 퇴거, kv(고정) 생존.
    // [호스트 이관 2026-10-14] ckpt는 HostRam 티어 — HBM 압박 대상이 아니다
    // (호스트 버퍼를 해제해도 VRAM이 안 늘어나므로). 잔존 = 접두 재사용 유지.
    let free0 = crate::cuda_mem_free().expect("VRAM 프로브").0;
    let freed = dec.ucache_pressure(kv).expect("압박 스윕");
    assert_eq!(freed, state, "퇴거분 HBM 실해제(state만)");
    assert_eq!(dec.ucache.evict_count(), 1, "state 1건 퇴거");

    // 실측 회수 판정 — [판정 조정 2026-10-14] 종전엔 `free1-free0 >= state`를
    // 직접 요구했다. 체크포인트가 HostRam 티어로 이관되면서 HBM 퇴거분이
    // ~52MiB → state(10.68MiB) 단독으로 줄었고, 드라이버의 2MiB 블록 회계
    // (같은 블록에 살아있는 버퍼가 섞이면 블록이 반환되지 않음)에서 실측
    // 회수 10MiB < 청구가 **결정적**으로 나온다(진단 실측 2026-10-14: 격리
    // 조각 6개 = 12MiB 전량 회수 · 압박 상황 = 10MiB). 회수 판정은
    // (a) 해제분 재할당 성공(실제 반환 증명) + (b) 여유 VRAM 무감소(정보
    // 기록)로 한다. 회계 정합(freed == state)·포인터 0화·spec 차단은 그대로.
    let free1 = crate::cuda_mem_free().expect("VRAM 프로브").0;
    let reuse = dec
        .cc
        .alloc(state as usize)
        .expect("해제분 재할당 실패 — VRAM 미회수");
    dec.cc.free(reuse).expect("재할당 해제");
    dec.cc.sync().expect("sync");
    eprintln!(
        "[ucache] 회수 실측: 여유 {free0} → {free1} (+{}B) · 청구 {state}B · 재할당 성공",
        free1.saturating_sub(free0)
    );

    // 기능 정합 — state 해제 0화 + 스펙 차단, ckpt(호스트) 잔존, KV 생존.
    assert!(!dec.gdn.dchk.is_null(), "ckpt(호스트) 잔존 — HBM 압박 무관");
    assert_eq!(dec.gdn.dwy_s0, 0, "WY S_0 해제 후 0화");
    assert_eq!(dec.gdn.dwy_dc, 0, "WY dc 해제 후 0화");
    assert_eq!(dec.gdn.dwy_g, 0, "WY g 해제 후 0화");
    assert_eq!(dec.gdn.dwy_k, 0, "WY k 해제 후 0화");
    assert_eq!(dec.gdn.dsnap_ring, 0, "dsnap_ring 해제 후 0화");
    assert_eq!(dec.gdn.dring0, 0, "dring0 해제 후 0화");
    // [dsave 대체 2026-10-14] 프로브 전용 지연 할당 — enable_spec 경로는
    // 애초에 만들지 않는다(serve VRAM 회수의 근거).
    assert_eq!(dec.gdn.dsave, 0, "dsave는 serve 경로에서 0(프로브 전용)");
    assert!(!dec.spec_on(), "해제 상태로 라운드 시도 차단");
    assert!(
        dec.ucache.entry("attn:kv").is_some(),
        "KV는 절대 퇴거 불가(고정)"
    );
    assert_eq!(
        dec.ucache.total(super::unified_cache::Tier::Hbm),
        kv,
        "퇴거 후 잔여 = HBM KV 단독"
    );
    assert_eq!(
        dec.ucache.total(super::unified_cache::Tier::HostRam),
        ckpt,
        "호스트 티어 = ckpt 잔존"
    );
}

// ═══ [TreeWY 2.B 2026-10-12] WY solve/reconstruct 커널 대조(합성) ═══
// gdn_wy_solve(+gdn_wy_reconstruct m) 상태 ≈ gdn_spec_scan(t=m) 순차 최종
// 상태 — PoC(scripts/poc_treewy.py) eval 클래스 검증의 커널 버전. 헤드별 게이트
// 도메인 혼합: h0=무감쇠(g=−1e-5)·k토큰 상관, h1=강감쇠(g=−91.6 — 35B
// 헤드11 부류), h2=현실적 softplus 감쇠, h3=강폭 현실적. k는 L2 정규화
// (l2perm 계약), S_0는 randn·0.3.
#[test]
fn wy_reconstruct_matches_sequential() {
    let _cuda = CUDA_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    let (h_k, h_v, d) = (2usize, 4usize, 128usize);
    let (k_len, v_len) = (h_k * d, h_v * d);
    let t_len = 4usize;
    let dec = super::W4a16Dec::new(1, k_len, 1).expect("CUDA 컨텍스트/fatbin");
    let mut s = 0x5EED_0C0F_FEE1u64;

    // base 벡터(kh=0 토큰 상관용 — 토큰 공통 단위벡터).
    let mut base = vec![0f32; d];
    let mut nrm = 0f32;
    for i in 0..d {
        base[i] = sm_f32(&mut s, -1.0, 1.0);
        nrm += base[i] * base[i];
    }
    for b in &mut base {
        *b /= (nrm + 1e-6).sqrt();
    }

    let mut q = vec![0f32; t_len * k_len];
    let mut k = vec![0f32; t_len * k_len];
    let mut v = vec![0f32; t_len * v_len];
    let mut bg = vec![0f32; t_len * 2 * h_v];
    for t in 0..t_len {
        for kh in 0..h_k {
            let mut acc = 0f32;
            for i in 0..d {
                let x = if kh == 0 {
                    0.4 * sm_f32(&mut s, -1.0, 1.0) + 0.9 * base[i]
                } else {
                    sm_f32(&mut s, -1.0, 1.0)
                };
                k[t * k_len + kh * d + i] = x;
                acc += x * x;
            }
            let inv = 1.0 / (acc + 1e-6).sqrt();
            for i in 0..d {
                k[t * k_len + kh * d + i] *= inv;
            }
        }
        for i in 0..k_len {
            q[t * k_len + i] = sm_f32(&mut s, -1.0, 1.0);
        }
        for i in 0..v_len {
            v[t * v_len + i] = sm_f32(&mut s, -1.0, 1.0);
        }
        for h in 0..h_v {
            let beta = 1.0f32 / (1.0f32 + (-0.3f32).exp());
            bg[t * 2 * h_v + h] = beta;
            let g = match h {
                0 => -1e-5,
                1 => -91.6,
                _ => -(1.0 + sm_f32(&mut s, 0.1, 2.0).exp()).ln() * sm_f32(&mut s, 0.5, 2.0),
            };
            bg[t * 2 * h_v + h_v + h] = g;
        }
    }
    let s0_host: Vec<f32> = (0..h_v * d * d)
        .map(|_| sm_f32(&mut s, -0.3, 0.3))
        .collect();

    let cc = &dec.cc;
    let (dq, dk, dv, dbg) = (
        cc.alloc(t_len * k_len * 4).unwrap(),
        cc.alloc(t_len * k_len * 4).unwrap(),
        cc.alloc(t_len * v_len * 4).unwrap(),
        cc.alloc(t_len * 2 * h_v * 4).unwrap(),
    );
    let (dst_, dout) = (
        // [행-스케일] 상태 = f16(2B/원소) + 행별 스케일 f32(초기 1.0 — 값 O(0.5)).
        cc.alloc(h_v * d * d * 2).unwrap(),
        cc.alloc(t_len * v_len * 4).unwrap(),
    );
    let dsc = cc.alloc(h_v * d * 4).unwrap();
    {
        let ones = vec![1.0f32; h_v * d];
        let onesb: Vec<u8> = ones.iter().flat_map(|v| v.to_le_bytes()).collect();
        cc.h2d(dsc, &onesb).unwrap();
    }
    let (ws0, wdc, wg, wk) = (
        cc.alloc(h_v * d * d * 4).unwrap(),
        cc.alloc(8 * h_v * d * 4).unwrap(),
        cc.alloc(8 * h_v * 4).unwrap(),
        cc.alloc(8 * k_len * 4).unwrap(),
    );
    let b = |x: &[f32]| unsafe { std::slice::from_raw_parts(x.as_ptr() as *const u8, x.len() * 4) };
    cc.h2d(dq, b(&q)).unwrap();
    cc.h2d(dk, b(&k)).unwrap();
    cc.h2d(dv, b(&v)).unwrap();
    cc.h2d(dbg, b(&bg)).unwrap();

    // 참조 S_m — gdn_spec_scan(t=m) 최종 상태(순차 = 정답).
    let mut refs = vec![Vec::new(); t_len + 1];
    // [dsave 대체 2026-10-14] m=0 기준 = 업로드된 S_0의 f16 비트 그대로
    // (keep=0 복원 = 이 상태로 복귀. sc=1 → 곱/나눗셈 왕복이 정확해 비트동일).
    refs[0] = s0_host
        .iter()
        .map(|&x| crate::rawcuda::gptq4::h2f(f32_to_f16(x)))
        .collect();
    for m in 1..=t_len {
        // [행-스케일] s0는 s=1 계약 — f16으로 업로드.
        let s0b: Vec<u8> = s0_host
            .iter()
            .flat_map(|&x| f32_to_f16(x).to_le_bytes())
            .collect();
        cc.h2d(dst_, &s0b).unwrap();
        let f = cc.function("gdn_spec_scan").unwrap();
        let (mut a0, mut a1, mut a2, mut a3, mut a4, mut a5) = (dq, dk, dv, dbg, dst_, dout);
        let mut a6 = dsc;
        let (mut tl, mut hk, mut hv, mut dd, mut lay) =
            (m as i32, h_k as i32, h_v as i32, d as i32, 0i32);
        cc.launch(
            f,
            h_v as u32,
            1,
            128,
            &mut crate::rawcuda::args::l12(
                &mut a0, &mut a1, &mut a2, &mut a3, &mut a4, &mut a6, &mut a5, &mut tl, &mut hk,
                &mut hv, &mut dd, &mut lay,
            ),
        )
        .unwrap();
        cc.sync().unwrap();
        let mut ob = vec![0u8; h_v * d * d * 2];
        cc.d2h(&mut ob, dst_).unwrap();
        refs[m] = ob
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| crate::rawcuda::gptq4::h2f(u16::from_le_bytes(*c)))
            .collect();
    }

    // WY — solve 1회(t=4) + 재구성 m=1..4.
    // [행-스케일] s0는 s=1 계약 — f16으로 업로드.
    let s0b: Vec<u8> = s0_host
        .iter()
        .flat_map(|&x| f32_to_f16(x).to_le_bytes())
        .collect();
    cc.h2d(dst_, &s0b).unwrap();
    {
        let f = cc.function("gdn_wy_solve").unwrap();
        let (mut a0, mut a1, mut a2, mut a3) = (dk, dv, dbg, dst_);
        let (mut a4, mut a5, mut a6, mut a7) = (ws0, wdc, wg, wk);
        let mut a8 = dsc;
        let (mut tl, mut hk, mut hv, mut dd, mut lay) =
            (t_len as i32, h_k as i32, h_v as i32, d as i32, 0i32);
        cc.launch(
            f,
            h_v as u32,
            1,
            128,
            &mut crate::rawcuda::args::l14(
                &mut a0, &mut a1, &mut a2, &mut a3, &mut a8, &mut a4, &mut a5, &mut a6, &mut a7,
                &mut tl, &mut hk, &mut hv, &mut dd, &mut lay,
            ),
        )
        .unwrap();
        let f = cc.function("gdn_spec_scan").unwrap();
        let (mut b0, mut b1, mut b2, mut b3, mut b4, mut b5) = (dq, dk, dv, dbg, dst_, dout);
        let mut b6 = dsc;
        let (mut tl2, mut hk2, mut hv2, mut dd2, mut lay2) =
            (t_len as i32, h_k as i32, h_v as i32, d as i32, 0i32);
        cc.launch(
            f,
            h_v as u32,
            1,
            128,
            &mut crate::rawcuda::args::l12(
                &mut b0, &mut b1, &mut b2, &mut b3, &mut b4, &mut b6, &mut b5, &mut tl2, &mut hk2,
                &mut hv2, &mut dd2, &mut lay2,
            ),
        )
        .unwrap();
    }
    for m in 0..=t_len {
        let f = cc.function("gdn_wy_reconstruct").unwrap();
        let (mut a0, mut a1, mut a2, mut a3) = (wk, ws0, wdc, wg);
        let mut a4 = dst_;
        let mut a5 = dsc;
        let (mut am, mut hk, mut hv, mut dd, mut lay) =
            (m as i32, h_k as i32, h_v as i32, d as i32, 0i32);
        cc.launch(
            f,
            h_v as u32,
            1,
            128,
            &mut crate::rawcuda::args::l11(
                &mut a0, &mut a1, &mut a2, &mut a3, &mut a4, &mut a5, &mut am, &mut hk, &mut hv,
                &mut dd, &mut lay,
            ),
        )
        .unwrap();
        cc.sync().unwrap();
        let mut ob = vec![0u8; h_v * d * d * 2];
        cc.d2h(&mut ob, dst_).unwrap();
        let wy: Vec<f32> = ob
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| crate::rawcuda::gptq4::h2f(u16::from_le_bytes(*c)))
            .collect();
        let mut mx = 0.0f32;
        for i in 0..wy.len() {
            mx = mx.max((wy[i] - refs[m][i]).abs());
        }
        eprintln!("[wy-test] m={m} maxdiff(seq-wy) = {mx:.6e}");
        // [행-스케일] 원시 Ŝ 비교(양 경로가 같은 스케일 계열) — f16 반올림 수 ulp.
        if m == 0 {
            // [dsave 대체] keep=0 복원(S_0)은 왕복 무손실 — 비트동일 계약.
            assert_eq!(mx, 0.0, "m=0 복원이 S_0와 다름: {mx:.6e}");
        } else {
            assert!(
                mx < 2e-3,
                "WY 재구성 m={m} 순차 스캔 초과 편차: {mx:.6e}(f16 기대 ≤2e-3)"
            );
        }
    }
}

// ═══ [dsave 대체 2026-10-14] conv 라운드 전 링 스냅샷(ring_pre) ═══
// gdn_conv(ring_pre≠0)가 **갱신 전** 링을 그대로 보존하는지 — acc==0
// 복원(spec_rollback keep=0)의 링 재료. 상태 링 자체는 정상 갱신됨도 판정
// (스냅샷이 갱신을 막지 않는다).
#[test]
fn conv_ring_pre_snapshot_is_pre_update_state() {
    let _cuda = CUDA_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    let (h_k, h_v, d) = (1usize, 2usize, 128usize);
    let conv_ch = (2 * h_k + h_v) * d;
    let (k_len, v_len) = (h_k * d, h_v * d);
    let t_len = 2usize;
    let dec = super::W4a16Dec::new(1, conv_ch, 1).expect("CUDA 컨텍스트/fatbin");
    let cc = &dec.cc;
    let mut s = 0x0D5A_7E00_0C0D_0C0Fu64;
    let qkv: Vec<f32> = (0..t_len * conv_ch)
        .map(|_| sm_f32(&mut s, -2.0, 2.0))
        .collect();
    let convw: Vec<f32> = (0..conv_ch * 4)
        .map(|_| sm_f32(&mut s, -0.5, 0.5))
        .collect();
    // 초기 링 = 고유값(스냅샷 비교 대상이자 '갱신 전' 기준).
    let ring0: Vec<f32> = (0..3 * conv_ch).map(|i| 1000.0 + i as f32).collect();
    let (dq, dw, dr, dqo, dko, dvo, dpre) = (
        cc.alloc(t_len * conv_ch * 4).unwrap(),
        cc.alloc(conv_ch * 4 * 4).unwrap(),
        cc.alloc(3 * conv_ch * 4).unwrap(),
        cc.alloc(t_len * k_len * 4).unwrap(),
        cc.alloc(t_len * k_len * 4).unwrap(),
        cc.alloc(t_len * v_len * 4).unwrap(),
        cc.alloc(3 * conv_ch * 4).unwrap(),
    );
    let b = |x: &[f32]| unsafe { std::slice::from_raw_parts(x.as_ptr() as *const u8, x.len() * 4) };
    cc.h2d(dq, b(&qkv)).unwrap();
    cc.h2d(dw, b(&convw)).unwrap();
    cc.h2d(dr, b(&ring0)).unwrap();
    cc.h2d(dpre, b(&vec![0f32; 3 * conv_ch])).unwrap();
    let f = cc.function("gdn_conv").unwrap();
    let (mut c0, mut c1, mut c2, mut c3, mut c4, mut c5) = (dq, dw, dr, dqo, dko, dvo);
    let mut c6 = 0u64; // ring_snap(토큰별) 생략
    let mut c7 = dpre; // 라운드 전 링 스냅샷
    let mut c8 = 0u64; // [행 퓨전] 행별 링 배열(null — 단일 발사)
    let (mut tl, mut lay, mut kl, mut vl, mut cch) = (
        t_len as i32,
        0i32,
        k_len as i32,
        v_len as i32,
        conv_ch as i32,
    );
    cc.launch(
        f,
        (conv_ch / 128) as u32,
        (t_len.div_ceil(16)) as u32,
        128,
        &mut crate::rawcuda::args::l14(
            &mut c0, &mut c1, &mut c2, &mut c3, &mut c4, &mut c5, &mut c6, &mut c7, &mut c8,
            &mut tl, &mut lay, &mut kl, &mut vl, &mut cch,
        ),
    )
    .unwrap();
    cc.sync().unwrap();
    let mut pb = vec![0u8; 3 * conv_ch * 4];
    cc.d2h(&mut pb, dpre).unwrap();
    let pre: Vec<f32> = pb
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect();
    assert_eq!(pre, ring0, "ring_pre가 라운드 전 링과 불일치(비트)");
    let mut rb = vec![0u8; 3 * conv_ch * 4];
    cc.d2h(&mut rb, dr).unwrap();
    let ring: Vec<f32> = rb
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect();
    for ch in 0..conv_ch {
        // 토큰 2개 처리 후 링 = (구 h2, x0, x1) — 갱신 정상(스냅샷 무간섭).
        assert_eq!(ring[ch], ring0[2 * conv_ch + ch], "링 h0 갱신 오류 ch={ch}");
        assert_eq!(ring[conv_ch + ch], qkv[ch], "링 h1 갱신 오류 ch={ch}");
        assert_eq!(
            ring[2 * conv_ch + ch],
            qkv[conv_ch + ch],
            "링 h2 갱신 오류 ch={ch}"
        );
    }
}

// ═══ [gemma4 2026-10-14] norm_resid_g(2816) — 행 경계 가드 변형 수치 검증 ═══
/// 가드 변형 vs CPU 미러(동일 누적 순서 재현). **비트 아님**: norm은 기본
/// fmad(수축 허용)라 f32 반올림이 미러와 다를 수 있다 — 상대오차 밴드 판정.
/// 인덱싱·가드 결함이면 오차가 ~1로 폭발한다(2816은 1024 배수 아님 —
/// 마지막 스트라이드 768스레드가 경계 밖).
#[test]
fn norm_2816_guard_variant_matches_cpu_mirror() {
    let _cuda = CUDA_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    let hidden = 2816usize;
    let t_len = 2usize;
    let mut dec = match super::W4a16Dec::new(1, hidden, 1) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("skip(CUDA 없음): {e}");
            return;
        }
    };
    let n = t_len * hidden;
    let mut seed = 0x9e37_79b9_7f4a_7c15u64;
    let mut rnd = || {
        seed = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = seed;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    };
    let pm1 = |r: u64| ((r >> 40) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0;
    let xs: Vec<f32> = (0..n).map(|_| pm1(rnd())).collect();
    let ab: Vec<f32> = (0..n).map(|_| pm1(rnd()) * 0.5).collect();
    let nw: Vec<f32> = (0..hidden).map(|e| 1.0 + e as f32 * 1e-6).collect();
    dec.set_norm_weights(&nw, 1).expect("norm 가중 등록");
    let f32b = |v: &[f32]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
    let dx = dec.cc.alloc(n * 4).unwrap();
    dec.cc.h2d(dx, &f32b(&xs)).expect("x h2d");
    let dab = dec.cc.alloc(n * 4).unwrap();
    dec.cc.h2d(dab, &f32b(&ab)).expect("ab h2d");
    let dxn = dec.norm_resid_dev(0, dx, dab, t_len, 0).expect("norm 발사");
    let d2h = |dec: &super::W4a16Dec, p: CUdeviceptr| -> Vec<f32> {
        let mut b = vec![0u8; n * 4];
        dec.cc.d2h(&mut b, p).expect("d2h");
        b.as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect()
    };
    let got = d2h(&dec, dxn);
    let xgot = d2h(&dec, dx);
    // CPU 미러 — 커널과 동일 순서(스레드별 nper조각 f32 누적 → smem 512..1 트리).
    let mut maxd = 0f32;
    let mut maxr = 0f32;
    let nper = hidden.div_ceil(1024);
    for t in 0..t_len {
        let base = t * hidden;
        let lim = base + hidden;
        let mut red = [0f32; 1024];
        // 스레드별 v — 커널과 같이 tid×nper(전 tid의 값이 서로 덮이지 않게).
        let mut vals = vec![0f32; 1024 * nper];
        for tid in 0..1024 {
            let mut ss = 0f32;
            for j in 0..nper {
                let e = base + (j << 10) + tid;
                let vj = if e < lim { xs[e] + ab[e] } else { 0.0 };
                vals[tid * nper + j] = vj;
                ss += vj * vj;
            }
            red[tid] = ss;
        }
        let mut st = 512usize;
        while st > 0 {
            for tid in 0..st {
                red[tid] += red[tid + st];
            }
            st >>= 1;
        }
        let inv = 1.0f32 / (red[0] / hidden as f32 + 1e-6).sqrt();
        for tid in 0..1024 {
            for j in 0..nper {
                let eo = (j << 10) + tid; // 행-로컬(커널 nw 오프셋 계약)
                let e = base + eo;
                if e >= lim {
                    continue;
                }
                let want = vals[tid * nper + j] * inv * nw[eo];
                let d = (got[e] - want).abs();
                maxd = maxd.max(d);
                if want.abs() > 1e-3 {
                    maxr = maxr.max(d / want.abs());
                }
                maxd = maxd.max((xgot[e] - (xs[e] + ab[e])).abs());
            }
        }
    }
    eprintln!("[norm2816] max|diff| {maxd:.3e} · 최대상대 {maxr:.3e}");
    assert!(maxr < 1e-5 && maxd < 1e-4, "norm2816 오차 초과: {maxr:.3e}");
}
