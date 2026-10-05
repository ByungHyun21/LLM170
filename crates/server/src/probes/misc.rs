//! vk 스모크·frame-check·gguf 검사(dims·tty-probe) (plans/129 R2③ — probes/ 분리, arm 본문 무변경 이동).
use std::process::ExitCode;

use super::arg_str;

pub(super) fn try_run(cmd: &str, args: &[String]) -> Option<ExitCode> {
    let d_fn = crate::resource::DEFAULT_FN_MODEL;
    let r: Result<String, String> = match cmd {
        "vk-check" => llm170_backend_gpu::rawvk::smoke_test(),
        // 109 P15-1a(7e06e90)에서 우발 삭제된 진입점 복원(110 P12c 검증용).
        "vk-frame-check" => {
            let path = arg_str(args, 0, d_fn);
            let tn = arg_str(args, 1, "blk.0.ffn_down_shexp.weight");
            llm170_backend_gpu::rawvk::checks::frame_check(&path, &tn)
        }
        "dims" => {
            let a: Vec<&str> = args.iter().map(|s| s.as_str()).collect();
            print!("{}", llm170_backend_gpu::rawhip::dims_of(a[0], &a[1..]));
            return Some(ExitCode::SUCCESS);
        }
        "tty-probe" => {
            let path = args
                .first()
                .cloned()
                .unwrap_or_else(|| "/tmp/model_link.gguf".into());
            return Some(
                match llm170_gguf::GgufFile::open(std::path::Path::new(&path)) {
                    Ok(g) => {
                        use std::collections::BTreeMap;
                        let mut cnt: BTreeMap<u32, usize> = BTreeMap::new();
                        let mut bytes: BTreeMap<u32, u64> = BTreeMap::new();
                        for t in &g.tensors {
                            *cnt.entry(t.ty as u32).or_insert(0) += 1;
                            *bytes.entry(t.ty as u32).or_insert(0) += t.nbytes().unwrap_or(0);
                        }
                        for (k, c) in cnt {
                            println!("ty{k}: {c} tensors {:.1}MB", bytes[&k] as f64 / 1e6);
                        }
                        ExitCode::SUCCESS
                    }
                    Err(e) => {
                        eprintln!("error: {e}");
                        ExitCode::FAILURE
                    }
                },
            );
        }
        _ => return None,
    };
    Some(super::finish(r))
}
