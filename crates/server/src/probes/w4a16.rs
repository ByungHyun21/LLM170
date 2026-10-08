//! W4A16 로더 프로브 — `w4a16-load` (plans/w4a16-cuda.md §1).
//!
//! 로더 완전성 검증 — 트리플 구조·weight_shape 값·커버리지(기대 텐서 전수)
//! 판정. 무게는 헤더 + 행 단위 pread만. (구 w4a16-xcheck/to-gguf는 GGUF 탈락
//! 2026-10-08로 제거 — 비트순서 확정·변환 검증은 이력에 기록.)

use super::{arg_str, finish};
use std::process::ExitCode;

pub fn try_run(cmd: &str, args: &[String]) -> Option<ExitCode> {
    match cmd {
        "w4a16-load" => Some(finish(load(args))),
        _ => None,
    }
}

fn load(args: &[String]) -> Result<String, String> {
    let dir = arg_str(args, 0, "");
    if dir.is_empty() {
        return Err(
            "w4a16-load <dir> — 사용법: llm170 w4a16-load ../models/Qwen3.8-27B-W4A16-AutoRound"
                .into(),
        );
    }
    let m = llm170_core::w4a16::W4a16Model::open(std::path::Path::new(&dir))
        .map_err(|e| e.to_string())?;
    let rep = m.validate().map_err(|e| e.to_string())?;
    let c = &m.cfg;
    let head = format!(
        "w4a16-load {dir}\n  hidden={} layers={} heads={}/{} head_dim={} ffn={} vocab={} interval={}\n  양자화=compressed-tensors pack-quantized int4 sym g{} (zp=8 상수) · 선형 {}개",
        c.hidden,
        c.layers,
        c.heads,
        c.kv_heads,
        c.head_dim,
        c.ffn,
        c.vocab,
        c.full_interval,
        c.group_size,
        m.n_lins()
    );
    let body = rep.summary();
    if rep.ok() {
        Ok(format!("{head}\n{body}  판정: 완전성 검증 통과"))
    } else {
        Err(format!("{head}\n{body}  판정: 검증 실패"))
    }
}
