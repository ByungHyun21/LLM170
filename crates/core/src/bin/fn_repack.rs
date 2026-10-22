
use llm170_core::qwen4exp::repack;
use std::path::Path;
use std::process::ExitCode;

fn main() -> ExitCode {
    if llm170_diag::flag::val("LLM170_NO_OOM_ADJ").is_none() {
        let _ = std::fs::write("/proc/self/oom_score_adj", b"1000");
    }
    let args: Vec<String> = std::env::args().skip(1).collect();
    let r: Result<String, String> = match args.first().map(String::as_str) {
        Some("verify") => run_verify(&args[1..]),
        Some(dir) if !dir.starts_with('-') => {
            if args.len() != 2 {
                Err("사용법: fn_repack <gptq_dir> <dst_ct_dir> | fn_repack verify <gptq_dir> <ct_dir> [--layers ..] [--experts ..]".into())
            } else {
                repack::repack(Path::new(&args[0]), Path::new(&args[1])).map(|s| s.to_string())
            }
        }
        _ => Err("사용법: fn_repack <gptq_dir> <dst_ct_dir> | fn_repack verify <gptq_dir> <ct_dir> [--layers ..] [--experts ..]".into()),
    };
    match r {
        Ok(msg) => {
            println!("{msg}");
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

fn run_verify(args: &[String]) -> Result<String, String> {
    let mut layers = vec![0u64, 23, 47];
    let mut experts = vec![0u64, 7, 511];
    let mut dirs: Vec<&String> = Vec::new();
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--layers" => layers = parse_csv(next_val(&mut it, "--layers")?)?,
            "--experts" => experts = parse_csv(next_val(&mut it, "--experts")?)?,
            other if other.starts_with('-') => return Err(format!("미지원 플래그 {other}")),
            _ => dirs.push(a),
        }
    }
    if dirs.len() != 2 {
        return Err(
            "사용법: fn_repack verify <gptq_dir> <ct_dir> [--layers 0,23,47] [--experts 0,7,511]"
                .into(),
        );
    }
    if layers.is_empty() || experts.is_empty() {
        return Err("--layers/--experts — 최소 1개 값 필요".into());
    }
    repack::verify_sample(Path::new(dirs[0]), Path::new(dirs[1]), &layers, &experts)
}

fn next_val<'a, I: Iterator<Item = &'a String>>(it: &mut I, flag: &str) -> Result<&'a str, String> {
    it.next()
        .map(String::as_str)
        .ok_or_else(|| format!("{flag} 값 필요"))
}

fn parse_csv(v: &str) -> Result<Vec<u64>, String> {
    v.split(',')
        .map(|t| {
            t.trim()
                .parse::<u64>()
                .map_err(|e| format!("값 {t:?}: {e}"))
        })
        .collect()
}
