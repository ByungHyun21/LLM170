//! [테스트 재설계 2026-10-22] tiny 고정재료 CPU↔GPU 같은 실행 등가 테스트.
//!
//! 타 엔진(vLLM/SGLang)의 "런타임 참조 비교" 패턴을 우리 CPU 오라클로 내재화:
//! 골든 문자열 없이 **같은 실행에서** 참조와 GPU 토큰열을 비교한다. 재료 =
//! `scripts/gen_tiny_w4a16.py` 생성 `target/tiny-fixtures/{tiny-dense,tiny-moe}`
//! (결정론 시드 — 재현 보장, 로드 ~1s). 실모델 골든 캐나리는 이걸 대체하지
//! 않고 병행한다(regress t2 = 실모델 슬라이스).
//!
//! GPU 필요 — `cargo test --release -p llm170-server -- --ignored`
//! (또는 `scripts/regress.sh t1`). 무GPU 환경의 기본 유닛트스트 침입을 막기
//! 전 테스트에 `#[ignore]`.

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

/// CPU 오라클 greedy 토큰열 — w4a16-ref 계약(프리필 첫 토큰 + n_predict 디코드).
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

/// GPU 엔진 greedy 토큰열 — serve와 동일 경로(prefill_greedy=디바이스 argmax,
fn gpu_tokens(dir: &std::path::Path, prompt: &[u32], n: usize, n_slots: usize) -> Vec<u32> {
    // cargo test의 CWD = 크레이트 루트 → fatbin 상대 탐색 경로가 깨진다
    // (nsys/serve와 동일 계약: 저장소 루트에서 실행). 테스트 시작점에서 1회
    // 워크스페이스 루트로 — 전 테스트 --test-threads=1 러너 계약(regress t1).
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

/// LCG 프롬프트(하네스와 동일 계열 — vocab 안전역).
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

/// dense(27B형 g128, GDN 3층+어텐션 1층) — 단문 프리필+디코드 등가.
#[test]
#[ignore = "gpu"]
fn tiny_dense_short_equiv() {
    let d = fixture("tiny-dense");
    equiv_case(&d, &[7, 13, 42, 100, 255], 12, "dense 단문");
}

/// dense — 700토큰 프롬프트(엔진 청크 상한 512 경계 = 512+188 다중 청크).
#[test]
#[ignore = "gpu"]
fn tiny_dense_multichunk_equiv() {
    let d = fixture("tiny-dense");
    equiv_case(&d, &lcg_prompt(0xDEAD_BEEF, 700), 12, "dense 다중 청크");
}

/// MoE(35B형 g32 전문가 8·top2, 어텐션/GDN 플레인 bf16) — 라우터·전문가
/// 디스패치·shared 경로 포함.
#[test]
#[ignore = "gpu"]
fn tiny_moe_equiv() {
    let d = fixture("tiny-moe");
    equiv_case(&d, &lcg_prompt(0xC0FF_EE01, 300), 12, "MoE");
    equiv_case(&d, &lcg_prompt(0xC0FF_EE02, 700), 12, "MoE 다중 청크");
}

/// 배치 불변성 — 2슬롯 동일 프리필 후 배치 디코드(np) = 단일 디코드 토큰.
/// 스모크 c2a/c2b(serve HTTP)의 엔진층 등가(2026-10-13 회귀 가드와 동일 축).
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

/// 슬롯 독립성 — 서로 다른 프롬프트 2슬롯 병행 시 각자 단일 경로와 동일.
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
