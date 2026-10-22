
use super::finish;
use std::process::ExitCode;

#[cfg(test)]
mod tiny_equiv_tests;

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
