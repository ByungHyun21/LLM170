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
        define(include_str!("../assets/gptq4.cu"), "G4_TMAX") as usize,
        BATCH_DEC_MAX,
        "gptq4.cu G4_TMAX ↔ BATCH_DEC_MAX(디코드 배치 상한 — head 이관 후)"
    );
    assert_eq!(
        define(include_str!("../assets/attn.cu"), "ATTN_SPLITS_C") as usize,
        ATTN_SPLITS,
        "attn.cu ATTN_SPLITS_C ↔ ATTN_SPLITS"
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
        define(include_str!("../assets/gptq4.cu"), "BMMA_M") as usize,
        GEMM_BMMA_M,
        "gptq4.cu BMMA_M ↔ GEMM_BMMA_M(bf16 mma 전용)"
    );
    assert_eq!(
        define(include_str!("../assets/gptq4.cu"), "BMMA_N") as usize,
        GEMM_BMMA_N,
        "gptq4.cu BMMA_N ↔ GEMM_BMMA_N(bf16 mma 전용)"
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
    let tr = define(include_str!("../assets/gptq4.cu"), "GEMV_TR") as usize;
    assert_eq!(tr, GEMV_TR, "gptq4.cu GEMV_TR ↔ GEMV_TR");
}
