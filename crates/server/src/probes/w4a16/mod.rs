//! W4A16 프로브 — `w4a16-load`(로더 완전성) · `w4a16-ref`(참조 러너).
//!
//! `w4a16-ref`: CPU 참조 greedy 토큰열 — 커널/서빙 판정의 오라클.
//! [R2 2026-10-10] mod.rs 분해 — ref_cmd(load/ref) · gates(모듈 게이트) ·
//! gpu(w4a16-gpu 디코드 프로브 + 캐스트 계약 테스트).

use super::finish;
use std::process::ExitCode;

mod gates;
mod gpu;
mod ref_cmd;

pub fn try_run(cmd: &str, args: &[String]) -> Option<ExitCode> {
    match cmd {
        "w4a16-load" => Some(finish(ref_cmd::load(args))),
        "w4a16-ref" => Some(finish(ref_cmd::reference(args))),
        "w4a16-gemv" => Some(finish(gates::gemm_gate(args, 1))),
        "w4a16-gemm" => Some(finish(gates::gemm_gate(args, 8))),
        "w4a16-gpu" => Some(finish(gpu::gpu_run(args))),
        _ => None,
    }
}
