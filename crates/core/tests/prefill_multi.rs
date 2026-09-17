//! 다중 시퀀스 청크 프리필 검증 (plans/76, 2026-09-17).
//!
//! 프레임 경로는 GPU 가속기(rawhip Q4Acc)가 있어야 돌므로 단위 테스트가
//! 아니라 **통합 테스트**로 둔다 — dev-의존 순환(backend-gpu → core)에서
//! 단위 테스트는 core를 두 번 컴파일해 트레잇 신원이 갈라진다(실측 E0308).
//! 통합 테스트 타깃은 lib과 backend-gpu가 같은 core 사본을 공유한다.
//!
//! 단언하는 것(둘 다 배치 경로의 슬라이스 배관·시퀀스 격리를 직접 친다):
//!  1) n_seq=1이면 배치 경로가 단일 시퀀스 프리필과 **logits 비트 동일**하다.
//!     행 대역 핸들·per-seq 상태 op·헤드 마지막 행 GEMM의 정확성 증명.
//!  2) 같은 입력을 다른 슬롯 집합에서 배치로 돌리면 **토큰이 같다**(결정성).
//!     per-seq 사슬(conv 링·AR)에 경합이나 미초기화 읽기가 있으면 여기서 깨진다.
//!  3) 4시퀀스에서 각 행의 logits가 자기 프롬프트 참조에 더 가깝다(오염이면
//!     행이 남의 상태를 읽어 자기 참조와 멀어진다). 실측 드리프트도 함께 찍는다.
//!
//! 단언하지 않는 것과 그 이유(2026-09-17 실측): "배치 토큰 == 독립 프리필
//! 토큰"의 **비트 일치는 이 백엔드에서 불가능**하다. dense op 행 수가 n_seq배가
//! 되면 커널 선택·환원 순서가 갈리고(백엔드의 행 수 게이트: 타일 mm/wm/j128,
//! MoE 전문가 경로 등), 48층을 지나며 커진다 — 4×128에서 행별 logit 최대차
//! 0.56-4.0, 4행 중 1행 토큰 뒤집힘. 이는 기존 단일 경로가 청크 크기만 바꿔도
//! 겪는 것과 같은 계열이며 그쪽이 더 크다: 같은 256토큰 프롬프트를 1×256 vs
//! 4×64로 넣으면 tok 271 vs 1375(최대차 9.1), 2×128이면 tok 82(6.6). 즉 배치
//! 자체의 결함이 아니라 백엔드 수치의 행 수 의존성이고, 등가성을 비트로
//! 요구하면 어떤 구현도 통과할 수 없다. 그래서 함수는 기본 off 게이트를 갖고
//! (LLM170_PREFILL_MULTI=1로 켠다) 채택 여부는 드리프트 정책 결정에 맡긴다.
//!
//! 모델 파일이 없으면 skip (crates/gguf/tests/real_models.rs 관례).
//! 주의: 모델이 VRAM(≈80GiB)에 상주하므로 다른 GPU 프로세스와 동시에 돌리면
//! hipMalloc OOM으로 실패한다.

use llm170_core::qwen4exp::frame;
use llm170_core::qwen4exp::layers::SeqState4;
use llm170_core::qwen4exp::stages::Ctx;
use llm170_core::qwen4exp::Model4;
use std::path::Path;

const MODEL: &str =
    "/home/yoon/models/qwen3.8-Flash-Next/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf";

const CTX: usize = 512;
const PER_SEQ: usize = 128;
const N_SEQ: usize = 4;

/// 시퀀스 si의 청크 — 시퀀스마다 다른 프롬프트.
fn chunk(si: usize, per_seq: usize) -> Vec<u32> {
    (0..per_seq).map(|j| 760 + (si * 37 + j) as u32 * 17).collect()
}

fn maxdiff(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max)
}

#[test]
fn prefill_multi_matches_sequential() {
    if !Path::new(MODEL).exists() {
        eprintln!("skip: {MODEL} 없음");
        return;
    }
    let m = Model4::load(Path::new(MODEL)).expect("load");
    let acc = match llm170_backend_gpu::new_q4_acc_with_sources(m.part_sources()) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("skip: GPU 가속기 없음 ({e})");
            return;
        }
    };
    let hp = m.hp.clone();
    // 배치 forward 게이트(기본 off) — 등가성 정책 확정 전 옵트인.
    unsafe { std::env::set_var("LLM170_PREFILL_MULTI", "1") };
    acc.set_ctx_len(CTX);
    let ctx = Ctx { model: &m, acc: Some(&*acc) };
    // 슬롯 배분: 0/1 = n=1 검증, 2..6/6..10 = 배치 A/B, 10..14 = 순차 참조.
    // QSA 디바이스 풀과 GDN 상태가 (층, 슬롯) 키라서 같은 슬롯을 다른 프롬프트로
    // 되감으면 갈라진다 — 모든 실행에 서로 다른 슬롯을 준다.
    let mut sts: Vec<SeqState4> = (0..14).map(|_| SeqState4::new(&hp, CTX)).collect();
    let mut fb = frame::Frame4::new(&*acc, &m, &sts, N_SEQ * PER_SEQ).expect("frame(배치)");
    let mut fq = frame::Frame4::new(&*acc, &m, &sts, PER_SEQ).expect("frame(순차)");
    let vocab = hp.vocab;

    // 1) n_seq=1 — 배치 경로 == 단일 프리필 (logits 비트 동일).
    {
        let ch = chunk(0, PER_SEQ);
        let got = frame::frame_forward_prefill_multi(
            &*acc, &m, &ctx, &[0], &mut sts, &mut fb, &ch, PER_SEQ,
        )
        .expect("배치 프리필(n=1)");
        let want =
            frame::frame_forward(&*acc, &m, &ctx, 1, &mut sts[1], &mut fq, &ch).expect("단일 프리필");
        let mut lg = vec![0.0f32; vocab];
        acc.frame_read(fb.logits_t, &mut lg).expect("logits 판독");
        assert_eq!(got[0], llm170_core::qwen35::greedy(&want), "n=1 토큰 불일치");
        assert_eq!(maxdiff(&lg, &want), 0.0, "n=1 로짓 비트 불일치");
    }

    // 2) 시퀀스 격리 — 서로 다른 프롬프트 4개, 다른 슬롯에서 같은 결과.
    {
        let chunks: Vec<Vec<u32>> = (0..N_SEQ).map(|si| chunk(si, PER_SEQ)).collect();
        let toks: Vec<u32> = chunks.concat();
        let a = frame::frame_forward_prefill_multi(
            &*acc, &m, &ctx, &[2, 3, 4, 5], &mut sts, &mut fb, &toks, PER_SEQ,
        )
        .expect("배치 A");
        let mut lg = vec![0.0f32; N_SEQ * vocab];
        acc.frame_read(fb.logits_t, &mut lg).expect("logits_t 판독");
        let b = frame::frame_forward_prefill_multi(
            &*acc, &m, &ctx, &[6, 7, 8, 9], &mut sts, &mut fb, &toks, PER_SEQ,
        )
        .expect("배치 B");
        assert_eq!(a, b, "배치 결정성 위반 — 같은 입력·다른 슬롯인데 토큰이 다르다");
        // 참조(단일 프리필) — 슬롯당 **1회**만 돌린다(같은 슬롯 재사용은 프레임
        // 상태(pos·GDN)를 이어가 참조가 오염된다).
        let refs: Vec<Vec<f32>> = (0..N_SEQ)
            .map(|sj| {
                let rs = 10 + sj;
                frame::frame_forward(&*acc, &m, &ctx, rs, &mut sts[rs], &mut fq, &chunks[sj])
                    .expect("단일 프리필")
            })
            .collect();
        for si in 0..N_SEQ {
            let row = &lg[si * vocab..(si + 1) * vocab];
            let own = maxdiff(row, &refs[si]);
            let other = (0..N_SEQ)
                .filter(|&sj| sj != si)
                .map(|sj| maxdiff(row, &refs[sj]))
                .fold(f32::INFINITY, f32::min);
            eprintln!(
                "# prefill-multi seq{si}: batch_tok={} ref_tok={} | 자기 참조 차 {own:.3e} / 타 참조 최소차 {other:.3e}",
                a[si],
                llm170_core::qwen35::greedy(&refs[si])
            );
            assert!(
                own < other,
                "seq{si}: 자기 참조 차 {own:.3e} >= 타 참조 차 {other:.3e} (슬라이스 오염 의심)"
            );
        }
    }
}
