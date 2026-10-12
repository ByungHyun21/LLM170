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
    };
    dec.set_attn(
        ad,
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
    };
    dec.set_attn(
        ad,
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
