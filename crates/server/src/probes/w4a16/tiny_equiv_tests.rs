
use super::ref_cmd::RefEngine;
use std::path::PathBuf;

fn fixture(name: &str) -> PathBuf {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../target/tiny-fixtures")
        .join(name);
    assert!(
        p.exists(),
        "tiny fixture 없음: {} — `python3 scripts/gen_tiny_w4a16.py` 먼저 실행",
        p.display()
    );
    p
}

fn cpu_tokens(dir: &std::path::Path, prompt: &[u32], n: usize) -> Vec<u32> {
    let mut e = RefEngine::load(dir.to_str().unwrap(), 1, 1024).expect("CPU 오라클 로드");
    let l = e.prefill(0, prompt).expect("CPU prefill");
    let mut t = vec![llm170_core::qwen35::greedy(&l)];
    for _ in 0..n {
        let nx = e.decode_greedy(0, *t.last().unwrap()).expect("CPU decode");
        t.push(nx);
    }
    t
}

fn gpu_tokens(dir: &std::path::Path, prompt: &[u32], n: usize, n_slots: usize) -> Vec<u32> {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let _ = std::env::set_current_dir(&root);
    let mut e = crate::gpu_engine::QwenEngine::load(dir, n_slots, 1024, None, None, None)
        .expect("GPU 엔진 로드");
    let t0 = e.prefill_greedy(0, prompt).expect("GPU prefill_greedy");
    let mut t = vec![t0];
    for _ in 0..n {
        let nx = e
            .decode_greedy(0, *t.last().unwrap())
            .expect("GPU decode_greedy");
        t.push(nx);
    }
    t
}

fn lcg_prompt(seed: u64, n: usize) -> Vec<u32> {
    let mut x = seed;
    (0..n)
        .map(|_| {
            x = x
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (x >> 33) as u32 % 500
        })
        .collect()
}

fn equiv_case(dir: &std::path::Path, prompt: &[u32], n: usize, label: &str) {
    let cpu = cpu_tokens(dir, prompt, n);
    let gpu = gpu_tokens(dir, prompt, n, 1);
    assert_eq!(
        cpu,
        gpu,
        "{label}: CPU 오라클 ≠ GPU (prompt={}tok) — 배선 회귀",
        prompt.len()
    );
}

#[test]
#[ignore = "gpu"]
fn tiny_dense_short_equiv() {
    let d = fixture("tiny-dense");
    equiv_case(&d, &[7, 13, 42, 100, 255], 12, "dense 단문");
}

#[test]
#[ignore = "gpu"]
fn tiny_dense_multichunk_equiv() {
    let d = fixture("tiny-dense");
    equiv_case(&d, &lcg_prompt(0xDEAD_BEEF, 700), 12, "dense 다중 청크");
}

#[test]
#[ignore = "gpu"]
fn tiny_moe_equiv() {
    let d = fixture("tiny-moe");
    equiv_case(&d, &lcg_prompt(0xC0FF_EE01, 300), 12, "MoE");
    equiv_case(&d, &lcg_prompt(0xC0FF_EE02, 700), 12, "MoE 다중 청크");
}

#[test]
#[ignore = "gpu"]
fn tiny_dense_batch_invariance() {
    let d = fixture("tiny-dense");
    let prompt = lcg_prompt(0x1234_5678, 40);
    let n = 8;
    let single = gpu_tokens(&d, &prompt, n, 1);
    let mut e = crate::gpu_engine::QwenEngine::load(&d, 2, 1024, None, None, None)
        .expect("GPU 엔진 로드(2슬롯)");
    let t0a = e.prefill_greedy(0, &prompt).expect("slot0 prefill");
    let t0b = e.prefill_greedy(1, &prompt).expect("slot1 prefill");
    assert_eq!(t0a, t0b, "슬롯 간 프리필 토큰 불일치");
    let mut a = vec![t0a];
    let mut b = vec![t0b];
    for _ in 0..n {
        let toks = e
            .decode_np_greedy(&[0, 1], &[*a.last().unwrap(), *b.last().unwrap()])
            .expect("배치 디코드");
        a.push(toks[0]);
        b.push(toks[1]);
    }
    assert_eq!(a, b, "배치 디코드 슬롯 간 불일치");
    assert_eq!(a, single, "배치(2슬롯) ≠ 단일 경로 토큰열");
}

#[test]
#[ignore = "gpu"]
fn tiny_dense_slot_isolation() {
    let d = fixture("tiny-dense");
    let pa = lcg_prompt(0xA11CE000, 33);
    let pb = lcg_prompt(0xB0B00000, 47);
    let n = 6;
    let sa = gpu_tokens(&d, &pa, n, 1);
    let sb = gpu_tokens(&d, &pb, n, 1);
    let mut e = crate::gpu_engine::QwenEngine::load(&d, 2, 1024, None, None, None)
        .expect("GPU 엔진 로드(2슬롯)");
    let mut a = vec![e.prefill_greedy(0, &pa).expect("pa prefill")];
    let mut b = vec![e.prefill_greedy(1, &pb).expect("pb prefill")];
    for _ in 0..n {
        let toks = e
            .decode_np_greedy(&[0, 1], &[*a.last().unwrap(), *b.last().unwrap()])
            .expect("배치 디코드");
        a.push(toks[0]);
        b.push(toks[1]);
    }
    assert_eq!(a, sa, "슬롯0 병행 ≠ 단독");
    assert_eq!(b, sb, "슬롯1 병행 ≠ 단독");
}
