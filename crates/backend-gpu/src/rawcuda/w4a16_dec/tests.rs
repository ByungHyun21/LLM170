use super::mem::realloc_fields;
use super::*;
use std::cell::{Cell, RefCell};

#[test]
fn moe_topk_mirrors_core_semantics() {
    let sel = moe_topk(&[2.0, 1.0, 0.5, -1.0], 2);
    assert_eq!(sel.len(), 2);
    assert_eq!(sel[0].0, 0);
    assert_eq!(sel[1].0, 1);
    assert!(sel[0].1 > sel[1].1);
    let sum: f32 = sel.iter().map(|&(_, w)| w).sum();
    assert!((sum - 1.0).abs() < 1e-6, "합={sum}");
    let ratio = sel[0].1 / sel[1].1;
    assert!((ratio - std::f32::consts::E).abs() < 1e-4, "비율={ratio}");
    let tie = moe_topk(&[1.0, 1.0, 1.0], 2);
    assert_eq!(tie[0].0, 0);
    assert_eq!(tie[1].0, 1);
}

#[test]
fn realloc_fields_partial_failure_retry_safe() {
    let mut f: [CUdeviceptr; 3] = [0x11, 0x22, 0x33];
    let live = RefCell::new(vec![0x11u64, 0x22, 0x33]);
    let frees = RefCell::new(Vec::<u64>::new());
    let next = Cell::new(0x100u64);
    let fail_next = Cell::new(1usize);
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
    fail_next.set(usize::MAX);
    let [a, b, c] = &mut f;
    realloc_fields(free, alloc, [a, b, c], [1, 1, 1]).expect("재시도");
    assert_eq!(*frees.borrow(), vec![0x11, 0x22, 0x33, 0x101]);
    assert_eq!(f, [0x102, 0x103, 0x104]);
    assert_eq!(live.borrow().len(), 3);
}

use super::{
    ATTN_F3S_TMAX, ATTN_SPLITS, CHAIN_TMAX, G4_CHUNK_MAX, G4_TROW_MAX, GEMM_BMMA_M, GEMM_BMMA_N,
    GEMM_FFMA_TMAX, GEMM_GRP_M, GEMM_GRP_N, GEMM_MMA_M, GEMM_MMA_N, GEMV_TR,
};

const _: () = {
    assert!(G4_CHUNK_MAX <= CHAIN_TMAX);
    assert!(G4_CHUNK_MAX.is_multiple_of(G4_TROW_MAX));
};

#[test]
fn head_attn_ffma_mirrors() {
    assert_eq!(
        define(include_str!("../assets/moe.cu"), "G4_TMAX") as usize,
        G4_TROW_MAX,
        "moe.cu G4_TMAX ↔ G4_TROW_MAX(t행 GEMV 스위치 상한)"
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
    assert_eq!(
        define(include_str!("../assets/moepp.cu"), "PP_MZ") as usize,
        MOE_PP_MZ,
        "moepp.cu PP_MZ ↔ MOE_PP_MZ(MoE 프리필 m타일 z분할)"
    );
}

#[test]
fn gemv_tr_mirror() {
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
    let t = define(include_str!("../assets/moe.cu"), "G4_TMAX") as usize;
    assert_eq!(t, G4_TROW_MAX, "moe.cu G4_TMAX ↔ G4_TROW_MAX");
}

#[test]
fn trow_switch_coverage() {
    let gptq4 = include_str!("../assets/gptq4.cu");
    let moe = include_str!("../assets/moe.cu");
    let head = include_str!("../assets/head.cu");
    let groups: &[(&str, &str, usize)] = &[
        (gptq4, "GEMV_T_CASE(", 2),
        (gptq4, "GEMV_T_CASE_G32(", 2),
        (moe, "GEMV_T_CASE_BF(", 2),
        (head, "GEMV_I8_TROW_CASE(", 1),
    ];
    for (src, mac, nsyms) in groups {
        for n in 1..=G4_TROW_MAX {
            let pat = format!("{mac}{n})");
            let hits = src.matches(&pat).count();
            assert_eq!(
                hits, *nsyms,
                "{mac}{n}) = {hits}곳(기대 {nsyms} — t-row 스위치 누락?)"
            );
        }
    }
}


static CUDA_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn sm64(s: &mut u64) -> u64 {
    *s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *s;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn sm_f32(s: &mut u64, lo: f32, hi: f32) -> f32 {
    lo + (sm64(s) as f64 / u64::MAX as f64) as f32 * (hi - lo)
}

fn bf16_bytes(v: &[f32]) -> Vec<u8> {
    let mut b = Vec::with_capacity(v.len() * 2);
    for &x in v {
        b.extend_from_slice(&((x.to_bits() >> 16) as u16).to_le_bytes());
    }
    b
}

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
    let (qdim, kvdim) = (qh * d, kvh * d);
    let (ffn, vocab) = (1024usize, 256usize);
    let mut dec = super::W4a16Dec::new(1, h, 1).expect("CUDA 컨텍스트/fatbin");
    let mut s = 0x1234_5678_9abc_def0u64;

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
    let ad = super::AttnDims {
        n_attn: 1,
        q_heads: qh,
        kv_heads: kvh,
        d,
        cap,
        interval: 1,
        sliding_window: 0,
    };
    dec.set_attn(
        ad,
        Vec::new(),
        &(0..ad.n_attn * d)
            .map(|_| sm_f32(&mut s, 0.5, 1.5))
            .collect::<Vec<_>>(),
        &(0..ad.n_attn * d)
            .map(|_| sm_f32(&mut s, 0.5, 1.5))
            .collect::<Vec<_>>(),
    )
    .expect("set_attn");
    let norms: Vec<f32> = (0..5 * h).map(|_| sm_f32(&mut s, 0.8, 1.2)).collect();
    dec.set_mtp(&norms, qdim, kvdim, ffn).expect("set_mtp");

    let seed: Vec<f32> = (0..h).map(|_| sm_f32(&mut s, -1.0, 1.0)).collect();
    dec.ensure_chain_bufs().expect("ensure_chain_bufs");
    {
        let _g = dec.cc.guard().expect("guard");
        let sb = unsafe { std::slice::from_raw_parts(seed.as_ptr() as *const u8, h * 4) };
        dec.cc.h2d(dec.dres, sb).expect("dres 시드");
    }

    let pos: u32 = 4;
    let tok0: u32 = 7;
    let emb0 = bf16_row(&emb_b, tok0 as usize, h);
    let embed_of = |t: u32| bf16_row(&emb_b, t as usize, h);

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

    dec.mtp_kv_reset().expect("kv 리셋2");
    dec.mtp_reset_hidden();
    dec.mtp_push_hidden(0).expect("push2");
    let d1c = dec.mtp_run(0, &[tok0], &emb0, pos).expect("앵커2");
    let chain = dec.mtp_chain_drafts(0, 4, pos).expect("체인");

    assert_eq!(d1, d1c, "앵커 argmax 불일치 — 비결정 오염");
    assert_eq!(seq, chain, "장치 체인 초안열 ≠ 순차 경로(토큰열 판정)");
}

#[test]
fn kstep_decode_matches_sequential() {
    let _cuda = CUDA_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    let h = 1024usize;
    let (qh, kvh, d, cap) = (4usize, 2usize, 256usize, 64usize);
    let (qdim, kvdim) = (qh * d, kvh * d);
    let (ffn, vocab) = (1024usize, 256usize);
    let mut dec = super::W4a16Dec::new(1, h, 1).expect("CUDA 컨텍스트/fatbin");
    let mut s = 0x0dd8_ba11_5eed_0f77u64;

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
    let ad = super::AttnDims {
        n_attn: 1,
        q_heads: qh,
        kv_heads: kvh,
        d,
        cap,
        interval: 1,
        sliding_window: 0,
    };
    dec.set_attn(
        ad,
        Vec::new(),
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

    dec.reset_state(0).expect("reset2");
    let kout = dec.decode_k_greedy(0, tok0, 4, u32::MAX).expect("kstep");
    assert_eq!(kout.len(), 4, "EOS 아님 → k 전체 방출");
    assert_eq!(kout, seq, "K-스텝 토큰열 ≠ 순차 경로(토큰열 판정)");
    assert_eq!(dec.slot_pos_of(0), 4, "slot_pos = 방출 수만큼 전진");

    let j = seq
        .iter()
        .position(|&t| t != tok0)
        .expect("시드와 다른 방출 토큰(드로우 가정)");
    dec.reset_state(0).expect("reset3");
    let kstop = dec.decode_k_greedy(0, tok0, 4, seq[j]).expect("kstep-eos");
    assert_eq!(kstop, &seq[..=j], "EOS에서 정지 — 방출 접두만 반환");
}

#[test]
fn ucache_enforcement_frees_device_memory() {
    let _cuda = CUDA_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    let h = 1024usize;
    let (qh, kvh, d, cap) = (4usize, 2usize, 256usize, 64usize);
    let mut dec = super::W4a16Dec::new(1, h, 1).expect("CUDA 컨텍스트/fatbin");

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
    let ad = super::AttnDims {
        n_attn: 1,
        q_heads: qh,
        kv_heads: kvh,
        d,
        cap,
        interval: 1,
        sliding_window: 0,
    };
    dec.set_attn(
        ad,
        Vec::new(),
        &vec![1f32; ad.n_attn * d],
        &vec![1f32; ad.n_attn * d],
    )
    .expect("set_attn");
    dec.enable_spec().expect("enable_spec");
    assert!(dec.spec_on(), "스펙 활성 전제(압박 전)");
    assert!(!dec.gdn.dchk.is_null(), "체크포인트 등록 전제");
    assert_ne!(dec.gdn.dwy_s0, 0, "스펙 상태(WY S_0) 등록 전제");
    assert_ne!(dec.gdn.dring0, 0, "라운드 전 링 스냅샷(dring0) 등록 전제");

    let ckpt = dec.ucache.entry("spec:ckpt").expect("ckpt 등록부").bytes;
    let state = dec.ucache.entry("spec:state").expect("state 등록부").bytes;
    let kv = dec.ucache.entry("attn:kv").expect("kv 등록부").bytes;
    let total = dec.ucache.total(super::unified_cache::Tier::Hbm);
    assert_eq!(total, state + kv, "등록부 HBM 회계 정합(ckpt 제외)");
    assert_eq!(
        dec.ucache.total(super::unified_cache::Tier::HostRam),
        ckpt,
        "ckpt는 호스트 티어"
    );

    let free0 = crate::cuda_mem_free().expect("VRAM 프로브").0;
    let freed = dec.ucache_pressure(kv).expect("압박 스윕");
    assert_eq!(freed, state, "퇴거분 HBM 실해제(state만)");
    assert_eq!(dec.ucache.evict_count(), 1, "state 1건 퇴거");

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

    assert!(!dec.gdn.dchk.is_null(), "ckpt(호스트) 잔존 — HBM 압박 무관");
    assert_eq!(dec.gdn.dwy_s0, 0, "WY S_0 해제 후 0화");
    assert_eq!(dec.gdn.dwy_dc, 0, "WY dc 해제 후 0화");
    assert_eq!(dec.gdn.dwy_g, 0, "WY g 해제 후 0화");
    assert_eq!(dec.gdn.dwy_k, 0, "WY k 해제 후 0화");
    assert_eq!(dec.gdn.dsnap_ring, 0, "dsnap_ring 해제 후 0화");
    assert_eq!(dec.gdn.dring0, 0, "dring0 해제 후 0화");
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

#[test]
fn wy_reconstruct_matches_sequential() {
    let _cuda = CUDA_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    let (h_k, h_v, d) = (2usize, 4usize, 128usize);
    let (k_len, v_len) = (h_k * d, h_v * d);
    let t_len = 4usize;
    let dec = super::W4a16Dec::new(1, k_len, 1).expect("CUDA 컨텍스트/fatbin");
    let mut s = 0x5EED_0C0F_FEE1u64;

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

    let mut refs = vec![Vec::new(); t_len + 1];
    refs[0] = s0_host
        .iter()
        .map(|&x| crate::rawcuda::gptq4::h2f(f32_to_f16(x)))
        .collect();
    for m in 1..=t_len {
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
        if m == 0 {
            assert_eq!(mx, 0.0, "m=0 복원이 S_0와 다름: {mx:.6e}");
        } else {
            assert!(
                mx < 2e-3,
                "WY 재구성 m={m} 순차 스캔 초과 편차: {mx:.6e}(f16 기대 ≤2e-3)"
            );
        }
    }
}

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
    let mut c6 = 0u64;
    let mut c7 = dpre;
    let mut c8 = 0u64;
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
        assert_eq!(ring[ch], ring0[2 * conv_ch + ch], "링 h0 갱신 오류 ch={ch}");
        assert_eq!(ring[conv_ch + ch], qkv[ch], "링 h1 갱신 오류 ch={ch}");
        assert_eq!(
            ring[2 * conv_ch + ch],
            qkv[conv_ch + ch],
            "링 h2 갱신 오류 ch={ch}"
        );
    }
}

#[test]
fn rms_rows_matches_host_norm_bit_exact() {
    let _cuda = CUDA_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    let hidden = 2816usize;
    let t_len = 3usize;
    let eps = 1e-6f32;
    let mut dec = match super::W4a16Dec::new(1, hidden, 1) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("skip(CUDA 없음): {e}");
            return;
        }
    };
    let n = t_len * hidden;
    let mut seed = 0x51ed_2701_9e37_79b9u64;
    let mut rnd = || {
        seed = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = seed;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    };
    let pm1 = |r: u64| ((r >> 40) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0;
    let xs: Vec<f32> = (0..n)
        .map(|_| pm1(rnd()) * (1.0 + (rnd() % 8) as f32))
        .collect();
    let nw: Vec<f32> = (0..hidden).map(|e| 1.0 + e as f32 * 1e-6).collect();
    dec.set_norm_weights(&nw, 1).expect("norm 가중 등록");
    let f32b = |v: &[f32]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
    let dx = dec.cc.alloc(n * 4).unwrap();
    dec.cc.h2d(dx, &f32b(&xs)).expect("x h2d");
    let dy = dec.cc.alloc(n * 4).unwrap();
    dec.rms_rows_dev(0, dx, dy, t_len, eps)
        .expect("rms_rows 발사");
    dec.cc.sync().unwrap();
    let mut ob = vec![0u8; n * 4];
    dec.cc.d2h(&mut ob, dy).expect("d2h");
    let got: Vec<f32> = ob
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect();
    let host_norm = |x: &[f32]| -> Vec<f32> {
        const SEG: usize = 32;
        let n = x.len();
        let chunk = n.div_ceil(SEG);
        let mut sum = 0.0f64;
        for u in 0..SEG {
            let lo = u * chunk;
            if lo >= n {
                break;
            }
            let hi = (lo + chunk).min(n);
            let mut part = 0.0f32;
            for &v in &x[lo..hi] {
                part += v * v;
            }
            sum += part as f64;
        }
        let scale = 1.0 / ((sum / n as f64 + eps as f64).sqrt() as f32);
        x.iter().zip(&nw).map(|(&v, &g)| v * scale * g).collect()
    };
    let mut bad = 0usize;
    let mut first = None;
    for t in 0..t_len {
        let row = &xs[t * hidden..(t + 1) * hidden];
        let want = host_norm(row);
        for e in 0..hidden {
            if got[t * hidden + e].to_bits() != want[e].to_bits() {
                bad += 1;
                if first.is_none() {
                    first = Some((t, e, got[t * hidden + e], want[e]));
                }
            }
        }
    }
    assert_eq!(
        bad, 0,
        "rms_rows 비트 불일치 {bad}건 (첫 사례 {:?}) — 수축/누적순서/스케일 점검",
        first
    );
    let _ = dec.cc.free(dx);
    let _ = dec.cc.free(dy);
}

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
    let mut maxd = 0f32;
    let mut maxr = 0f32;
    let nper = hidden.div_ceil(1024);
    for t in 0..t_len {
        let base = t * hidden;
        let lim = base + hidden;
        let mut red = [0f32; 1024];
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
                let eo = (j << 10) + tid;
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

#[test]
fn ckpt_victim_flop_aware() {
    let z = [0u32; 4];
    assert_eq!(W4a16Dec::ckpt_victim(&z, &[0; 4]), 0);
    assert_eq!(W4a16Dec::ckpt_victim(&z, &[1, 2, 3, 4]), 0);
    assert_eq!(W4a16Dec::ckpt_victim(&z, &[9, 6, 7, 8]), 1);
    assert_eq!(W4a16Dec::ckpt_victim(&[5, 0, 0, 0], &[1, 2, 3, 4]), 1);
    assert_eq!(W4a16Dec::ckpt_victim(&[9, 9, 9, 0], &[9, 8, 7, 6]), 3);
    assert_eq!(W4a16Dec::ckpt_victim(&[7, 7, 7, 7], &[5, 2, 9, 3]), 1);
}

#[test]
fn layer_window_27b_split_math() {
    let n = 64usize;
    let (wa, wb) = (
        LayerWindow { start: 0, end: 32 },
        LayerWindow { start: 32, end: n },
    );
    assert_eq!(wa.gdn_start(4), 0);
    assert_eq!(wb.gdn_start(4), 24);
    assert_eq!(wb.abs(0), 32);
    assert!(wa.owns_attn(7, 4, n));
    assert!(!wa.owns_attn(8, 4, n));
    assert!(wb.owns_attn(8, 4, n));
    assert!(!wb.owns_attn(7, 4, n));
    let full = LayerWindow::full(n);
    assert!(full.owns_attn(16, 4, n));
    assert!(wb.owns_attn(16, 4, n));
    assert!(!wa.owns_attn(16, 4, n));
    assert!(wb.owns_final(n));
    assert!(!wa.owns_final(n));
    for off in 0..wa.n() {
        assert_eq!(wa.abs(off), off);
    }
    for off in 0..wb.n() {
        assert_eq!(wb.abs(off), 32 + off);
    }
}

#[test]
fn pp2shard_matches_full_window() {
    let _cuda = CUDA_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    let h = 1024usize;
    let (qh, kvh, dch) = (4usize, 2usize, 256usize);
    let (qdim, kvdim) = (qh * dch, kvh * dch);
    let ffn = 1024usize;
    let vocab = 256usize;
    let n_layers = 4usize;
    let gd = GdnDims {
        n_gdn: 3,
        hidden: h,
        h_k: 2,
        h_v: 4,
        d: 128,
    };
    let (conv_ch, gate_w) = (gd.conv_ch(), gd.h_v * 128);
    let (cap, interval) = (64usize, 4usize);

    let seed_of = |name: &str| -> u64 {
        let mut x = 0xcbf2_9ce4_8422_2325u64;
        for b in name.bytes() {
            x ^= b as u64;
            x = x.wrapping_mul(0x1000_0000_01b3);
        }
        x ^ 0x0dd8_ba11_5eed_0f77
    };
    let lin = |dec: &mut super::W4a16Dec, name: &str, n: usize, k: usize| {
        let mut s = seed_of(name);
        let q: Vec<u8> = (0..n * k / 2)
            .map(|_| (sm64(&mut s) & 0xff) as u8)
            .collect();
        let sc: Vec<u8> = (0..n * k / 128)
            .flat_map(|_| f32_to_f16(sm_f32(&mut s, 0.004, 0.016)).to_le_bytes())
            .collect();
        dec.upload_lin(name, &q, &sc, n, k).expect("upload_lin");
    };
    let upload_layers = |dec: &mut super::W4a16Dec, layers: (usize, usize)| {
        for il in layers.0..layers.1 {
            if (il + 1) % interval == 0 {
                lin(dec, &format!("blk.{il}.attn_q.weight"), qdim, h);
                lin(dec, &format!("blk.{il}.attn_k.weight"), kvdim, h);
                lin(dec, &format!("blk.{il}.attn_v.weight"), kvdim, h);
                lin(dec, &format!("blk.{il}.attn_output.weight"), h, qdim);
            } else {
                lin(dec, &format!("blk.{il}.attn_qkv.weight"), conv_ch, h);
                lin(dec, &format!("blk.{il}.attn_gate.weight"), gate_w, h);
                lin(dec, &format!("blk.{il}.ssm_out.weight"), h, gate_w);
            }
            lin(dec, &format!("blk.{il}.ffn_gate.weight"), ffn, h);
            lin(dec, &format!("blk.{il}.ffn_up.weight"), ffn, h);
            lin(dec, &format!("blk.{il}.ffn_down.weight"), h, ffn);
        }
    };
    let consts = |dec: &mut super::W4a16Dec| {
        let mut s = 0x5eed_5eed_5eed_5eedu64;
        let norms: Vec<f32> = (0..(2 * n_layers + 1) * h)
            .map(|_| sm_f32(&mut s, 0.8, 1.2))
            .collect();
        dec.set_norm_weights(&norms, 2 * n_layers + 1)
            .expect("set_norm_weights");
        dec.set_gdn(
            gd,
            &(0..gd.n_gdn * gd.conv_ch() * 4)
                .map(|_| sm_f32(&mut s, -0.05, 0.05))
                .collect::<Vec<_>>(),
            &(0..gd.n_gdn * 2 * gd.h_v * h)
                .map(|_| sm_f32(&mut s, -0.05, 0.05))
                .collect::<Vec<_>>(),
            &(0..gd.n_gdn * gd.h_v)
                .map(|_| sm_f32(&mut s, 0.0, 1.0))
                .collect::<Vec<_>>(),
            &(0..gd.n_gdn * gd.h_v)
                .map(|_| sm_f32(&mut s, -0.05, 0.05))
                .collect::<Vec<_>>(),
            &(0..gd.n_gdn * 128)
                .map(|_| sm_f32(&mut s, 0.5, 1.5))
                .collect::<Vec<_>>(),
        )
        .expect("set_gdn");
        let ad = AttnDims {
            n_attn: n_layers / interval,
            q_heads: qh,
            kv_heads: kvh,
            d: dch,
            cap,
            interval,
            sliding_window: 0,
        };
        dec.set_attn(
            ad,
            Vec::new(),
            &(0..ad.n_attn * dch)
                .map(|_| sm_f32(&mut s, 0.5, 1.5))
                .collect::<Vec<_>>(),
            &(0..ad.n_attn * dch)
                .map(|_| sm_f32(&mut s, 0.5, 1.5))
                .collect::<Vec<_>>(),
        )
        .expect("set_attn");
    };
    let head_w: Vec<f32> = {
        let mut s = seed_of("head");
        (0..vocab * h)
            .map(|_| sm_f32(&mut s, -0.05, 0.05))
            .collect()
    };
    let embed_of = |t: u32| -> Vec<f32> {
        let mut s = seed_of(&format!("emb.{t}"));
        (0..h).map(|_| sm_f32(&mut s, -1.0, 1.0)).collect()
    };
    let argmax_min = |lg: &[f32]| -> u32 {
        let mut bi = 0usize;
        for (i, v) in lg.iter().enumerate() {
            if *v > lg[bi] {
                bi = i;
            }
        }
        bi as u32
    };

    let mut gold = super::W4a16Dec::new_dev(1, h, n_layers, 0).expect("골든 CUDA");
    upload_layers(&mut gold, (0, n_layers));
    consts(&mut gold);
    gold.upload_head(&bf16_bytes(&head_w), vocab, h)
        .expect("upload_head");

    let mut sa = super::W4a16Dec::new_dev(1, h, n_layers, 0).expect("샤드A CUDA");
    sa.set_layer_window(0, 2).expect("set_layer_window A");
    upload_layers(&mut sa, (0, 2));
    consts(&mut sa);
    let mut sb = super::W4a16Dec::new_dev(1, h, n_layers, 1).expect("샤드B CUDA(2번째 카드)");
    sb.set_layer_window(2, 4).expect("set_layer_window B");
    upload_layers(&mut sb, (2, 4));
    consts(&mut sb);
    sb.upload_head(&bf16_bytes(&head_w), vocab, h)
        .expect("upload_head B");

    let prompt = [7u32, 123, 42];
    let mut rows: Vec<f32> = Vec::new();
    for &t in &prompt {
        rows.extend_from_slice(&embed_of(t));
    }

    gold.reset_state(0).expect("reset 골든");
    let lg = gold
        .forward_prefill(0, &rows, 3, true)
        .expect("골든 프리필");
    let mut gold_toks = vec![argmax_min(&lg)];
    let mut cur = gold_toks[0];
    for _ in 0..4 {
        cur = gold
            .forward_device_argmax(0, &embed_of(cur))
            .expect("골든 디코드");
        gold_toks.push(cur);
    }

    sa.reset_state(0).expect("reset A");
    sb.reset_state(0).expect("reset B");
    let lg2 = forward_2shard(&mut sa, &mut sb, 0, &rows, 3, true).expect("2샤드 프리필");
    assert_eq!(lg2.len(), lg.len(), "2샤드 로짓 폭 = 골든 head_n");
    let mut pp_toks = vec![argmax_min(&lg2)];
    let mut cur = pp_toks[0];
    for _ in 0..4 {
        cur = forward_2shard_argmax(&mut sa, &mut sb, 0, &embed_of(cur)).expect("2샤드 디코드");
        pp_toks.push(cur);
    }
    assert_eq!(pp_toks, gold_toks, "2샤드 토큰열 ≠ 전체창 골든(비트 판정)");
    assert_eq!(sa.slot_pos_of(0), gold.slot_pos_of(0), "slot_pos 정합");
    assert_eq!(sb.slot_pos_of(0), gold.slot_pos_of(0), "slot_pos 정합 B");
}
