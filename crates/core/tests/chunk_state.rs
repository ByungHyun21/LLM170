//! 청크 크기 불변성 진단 (2026-09-17) — 원인 특정용.
//!
//! 편성: 같은 64토큰을 여러 청크 크기로 프리필하고 **최종 로짓**과 GDN 상태를
//! 참조(1×64)와 맞댄다. 토큰 일치보다 로짓 max|Δ|가 훨씬 민감한 연속 신호라
//! 어느 크기에서 얼마나 갈리는지 한 번에 보인다.
//!
//! 발단: `LLM170_Q4_CHUNK=64`에서 Flash-Next 출력이 붕괴(내용 무관, t=64의
//! 순수 함수). 프레임 체크섬(LLM170_NP_CHECKSUM)으로 L0/L1 상태 발산까지
//! 좁혔고, 이 테스트가 청크 크기 축을 쓸어 원인 구조를 드러낸다.
//! PLE 스킵 실험 결과: PLE가 청크 의존성의 주원인 하나(스킵 시 16≡192 완전
//! 일치)이고, t≈64 붕괴는 PLE와 무관한 **별개** 원인이다 — 이 스윕이 그 둘을
//! 크기 축에서 분리한다.
//!
//! 프레임 경로는 GPU 가속기(rawhip Q4Acc)가 필요해 통합 테스트로 둔다
//! (prefill_multi.rs와 같은 이유). 모델 파일이 없으면 skip. 단독 실행할 것.
//!
//! 한계(2026-09-17): 이 하네스는 아직 *자기 결과가 안정적이지 않다* — 검증
//! 블록을 추가하면 뒤 실행의 수치가 바뀐다(프레임의 호출 간 캐시 의심: np 뷰
//! 테이블·크기 키 스크래치). 따라서 여기서 나온 "t 의존" 수치는 *방향*으로만
//! 읽고, 커널 판정은 rawhip probes(`gdn_ar_invariance`/`gdn_conv_invariance`,
//! 모델 불필요·결정적)를 신뢰할 것. mm_group 블록(무상태)은 신뢰 가능하다.

use llm170_core::qwen4exp::frame;
use llm170_core::qwen4exp::layers::SeqState4;
use llm170_core::qwen4exp::stages::Ctx;
use llm170_core::qwen4exp::Model4;
use std::path::Path;

const MODEL: &str =
    "/home/yoon/models/qwen3.8-Flash-Next/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf";

const CTX: usize = 256;
const NTOK: usize = 64;

/// 청크 크기 후보 — 16 = 엔진 최소, 64 = 붕괴 지점, 그 주변 ±1, 그리고 큰 값들.
const SIZES: [usize; 4] = [16, 24, 32, 64];

fn maxdiff(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max)
}

#[test]
fn chunk_size_state_divergence() {
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
    acc.set_ctx_len(CTX);
    let ctx = Ctx { model: &m, acc: Some(&*acc) };
    eprintln!(
        "# hp n_group={} d_state={} dt_rank={} conv_k={} n_layer={} vocab={}",
        hp.n_group, hp.d_state, hp.dt_rank, hp.conv_k, hp.n_layer, hp.vocab
    );
    // 게이트와 같은 실프롬프트 — 합성 토큰은 모델을 혼돈 영역에 넣어 어떤
    // 수치 차이도 증폭되므로 불변성 판정에 부적합하다(2026-09-17 실측).
    let toks: Vec<u32> = vec![386, 18, 15, 15, 643, 20, 20, 19586, 5876, 8058, 4144, 67, 21, 7307, 22, 20, 23, 24902, 17, 16, 23, 386, 18, 66, 19, 386, 17, 24, 19, 24902, 16, 16, 19586, 66, 21, 65, 23, 1692, 22, 22, 19, 4144, 341, 15, 11, 17374, 67, 23, 15, 1692, 15, 65, 15, 1692, 22, 19, 15, 17374, 66, 16, 19, 386, 17, 69];

    let n_slot = SIZES.len() + 1;
    let mut sts: Vec<SeqState4> = (0..n_slot).map(|_| SeqState4::new(&hp, CTX)).collect();
    // 프레임은 하나 — 활성 버퍼는 공유하고 GDN 상태만 슬롯별로 소유한다(엔진과 동일).
    let mut f = frame::Frame4::new(&*acc, &m, &sts, NTOK).expect("frame");

    // 엔진 규칙: 프레임 상태는 호스트(SeqState4)가 정본 — 첫 사용 전 동기화.
    for slot in 0..n_slot {
        let st = std::mem::replace(&mut sts[slot], SeqState4::new(&hp, CTX));
        f.sync_states(&*acc, slot, &st, hp.d_state).expect("sync_states");
        sts[slot] = st;
    }

    // ── 슬롯 비결정성: 같은 1×64를 여러 슬롯에서 돌려 비교 ──
    {
        let mut logs: Vec<Vec<f32>> = Vec::new();
        for k in 0..4 {
            let slot = n_slot - 1 - k;
            let mut st = SeqState4::new(&hp, CTX);
            f.sync_states(&*acc, slot, &st, hp.d_state).expect("sync");
            let lg = frame::frame_forward(&*acc, &m, &ctx, slot, &mut st, &mut f, &toks)
                .expect("1×64");
            logs.push(lg);
        }
        for k in 1..4 {
            let d = maxdiff(&logs[0], &logs[k]);
            eprintln!("# 슬롯 비결정성: 슬롯{} vs 슬롯{} maxΔ={:.3e} {}", n_slot - 1, n_slot - 1 - k, d,
                if d == 0.0 { "동일 ✓" } else { "다름 ✗" });
        }
    }

    // ── 청크별 은닉 비교: 같은 행을 t=64 한 번 vs t=16 첫 청크로 ──
    // 행 [0,16)은 두 실행에서 토큰 0..15 — 완전히 같은 입력이다. mix(hc 투영
    // 출력)와 res_hc(잔차)를 읽어 어느 단계에서 갈리는지 본다.
    {
        let n = hp.n_embd;
        let hc = hp.hc;
        let mut m64 = vec![0.0f32; 16 * n];
        let mut r64 = vec![0.0f32; 16 * hc * n];
        frame::frame_forward(&*acc, &m, &ctx, n_slot - 1, &mut sts[n_slot - 1], &mut f, &toks)
            .expect("1×64(비교용)");
        acc.frame_read(f.mix, &mut m64).expect("mix 1×64");
        acc.frame_read(f.res_hc, &mut r64).expect("res 1×64");
        // 슬롯 n_slot-2: 첫 16토큰만 별도로
        let mut st16 = SeqState4::new(&hp, CTX);
        f.sync_states(&*acc, n_slot - 2, &st16, hp.d_state).expect("sync");
        frame::frame_forward(&*acc, &m, &ctx, n_slot - 2, &mut st16, &mut f, &toks[..16])
            .expect("1×16");
        let mut m16 = vec![0.0f32; 16 * n];
        let mut r16 = vec![0.0f32; 16 * hc * n];
        acc.frame_read(f.mix, &mut m16).expect("mix 1×16");
        acc.frame_read(f.res_hc, &mut r16).expect("res 1×16");
        eprintln!(
            "# 은닉 행0..15: res_hc maxΔ={:.3e} · mix maxΔ={:.3e} → {}",
            maxdiff(&r64, &r16),
            maxdiff(&m64, &m16),
            if maxdiff(&r64, &r16) == 0.0 && maxdiff(&m64, &m16) == 0.0 { "t 불변 ✓" } else { "t 의존 ✗" }
        );
    }

    // 참조: 슬롯 0 = 단일 64토큰 청크.
    let ref_logits =
        frame::frame_forward(&*acc, &m, &ctx, 0, &mut sts[0], &mut f, &toks).expect("1×64 프리필");
    let ref_tok = llm170_core::qwen35::greedy(&ref_logits);
    eprintln!("# 참조(1×{NTOK}) 토큰={ref_tok}");

    eprintln!("# {:>5} {:>6} {:>12} {:>12} {:>12}", "청크", "청크수", "logits maxΔ", "상대", "토큰");
    for (i, &c) in SIZES.iter().enumerate() {
        let slot = i + 1;
        let mut f_last = Vec::new();
        for ch in toks.chunks(c) {
            f_last = frame::frame_forward(&*acc, &m, &ctx, slot, &mut sts[slot], &mut f, ch)
                .expect("프리필");
        }
        let lg = f_last;
        let md = maxdiff(&lg, &ref_logits);
        // 참조 스케일(최대 |logit|) 대비 상대 오차
        let scale = ref_logits.iter().fold(0.0f32, |m, &x| m.max(x.abs())).max(1e-3);
        let tok = llm170_core::qwen35::greedy(&lg);
        eprintln!(
            "# {:>5} {:>6} {:>12.3e} {:>12.3e} {:>12}",
            c,
            NTOK.div_ceil(c),
            md,
            md / scale,
            if tok == ref_tok { format!("{tok}") } else { format!("{tok} ✗") }
        );
    }

    // ── 밀집 GEMM(frame_mm_group)의 t 불변성 — 실제 무게로 직접 판정 ──
    // conv 상태 = 이 GEMM 출력의 마지막 행 복사이므로, 상태가 갈리면 여기가 원인이다.
    {
        let n = hp.n_embd;
        let wk = m.w4("blk.0.attn_qkv.weight").expect("wqkv");
        let wz = m.w4("blk.0.attn_gate.weight").expect("wz");
        let wb = m.w4("blk.0.ssm_beta.weight").expect("wb");
        let wa = m.w4("blk.0.ssm_alpha.weight").expect("wa");
        let conv_ch = 2 * hp.n_group * hp.d_state + hp.dt_rank * hp.d_state;
        // 투영별 출력 버퍼(크기가 다름) — qkv/z/β/α.
        let (a_in, b_in) = (
            acc.frame_alloc(NTOK * n).expect("in"),
            acc.frame_alloc(NTOK * n).expect("in2"),
        );
        let (az, bz) = (
            acc.frame_alloc(NTOK * wz.n_out as usize).expect("z"),
            acc.frame_alloc(NTOK * wz.n_out as usize).expect("z2"),
        );
        let (aba, bba) = (
            acc.frame_alloc(NTOK * hp.dt_rank).expect("b"),
            acc.frame_alloc(NTOK * hp.dt_rank).expect("b2"),
        );
        let (a_out, b_out) = (
            acc.frame_alloc(NTOK * conv_ch).expect("out"),
            acc.frame_alloc(NTOK * conv_ch).expect("out2"),
        );
        let mut seed = 0x1234_5678u64;
        let mut lcg = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((seed >> 33) as f32 / (1u32 << 31) as f32) - 0.5
        };
        let xin: Vec<f32> = (0..NTOK * n).map(|_| lcg()).collect();
        acc.frame_write(a_in, &xin).expect("write A");
        acc.frame_write(b_in, &xin).expect("write B");
        acc.frame_begin(NTOK);
        acc.frame_mm_group(a_in, &[wk, wz, wb, wa],
            &[a_out, az, aba, aba], NTOK).expect("mm_group 1×t");
        let mut oa = vec![0.0f32; NTOK * conv_ch];
        acc.frame_read(a_out, &mut oa).expect("read A");
        // 밴드 4×16 — 입력·출력 모두 행 밴드 슬라이스
        for c in 0..(NTOK / 16) {
            let off = c * 16;
            let (i2, o2) = (
                acc.frame_slice(b_in, off * n, 16 * n).expect("in band"),
                acc.frame_slice(b_out, off * conv_ch, 16 * conv_ch).expect("out band"),
            );
            let z2 = acc.frame_slice(bz, off * n, 16 * n).expect("z band");
            let b2 = acc.frame_slice(bba, off * hp.dt_rank, 16 * hp.dt_rank).expect("b band");
            acc.frame_begin(16);
            acc.frame_mm_group(i2, &[wk, wz, wb, wa],
                &[o2, z2, b2, b2], 16).expect("mm_group band");
        }
        let mut ob = vec![0.0f32; NTOK * conv_ch];
        acc.frame_read(b_out, &mut ob).expect("read B");
        let mx = maxdiff(&oa, &ob);
        let first = oa.iter().zip(&ob).enumerate()
            .find(|(_, (x, y))| x.to_bits() != y.to_bits())
            .map(|(i, _)| format!("out[{i}] (행 {} 열 {})", i / conv_ch, i % conv_ch));
        let nbad = oa.iter().zip(&ob).filter(|(x, y)| x.to_bits() != y.to_bits()).count();
        eprintln!("# mm_group(실무게) 1×{NTOK} vs {}×16: maxΔ={mx:.3e} ({nbad}/{}) {} → {}",
            NTOK / 16,
            oa.len(),
            first.unwrap_or_else(|| "-".into()),
            if mx == 0.0 { "t 불변 ✓" } else { "t 의존 ✗" });
    }

    // GDN 상태 비교 — 참조(1×64) vs 각 청킹, 첫 발산 층 보고.
    let conv_ch = 2 * hp.n_group * hp.d_state + hp.dt_rank * hp.d_state;
    let cs = (hp.conv_k - 1) * conv_ch;
    let gs = hp.dt_rank * hp.d_state * hp.d_state;
    for (i, &c) in SIZES.iter().enumerate() {
        let slot = i + 1;
        let mut ri = 0usize;
        let (mut fc, mut fa) = (None::<usize>, None::<usize>);
        let (mut mc, mut ma) = (0.0f32, 0.0f32);
        for il in 0..hp.n_layer {
            if !hp.is_recr(il) {
                continue;
            }
            let mut ca = vec![0.0f32; cs];
            let mut cb = vec![0.0f32; cs];
            let mut ga = vec![0.0f32; gs];
            let mut gb = vec![0.0f32; gs];
            acc.frame_read(f.st_conv[0][ri], &mut ca).expect("conv A");
            acc.frame_read(f.st_conv[slot][ri], &mut cb).expect("conv B");
            acc.frame_read(f.st_gdn[0][ri], &mut ga).expect("ar A");
            acc.frame_read(f.st_gdn[slot][ri], &mut gb).expect("ar B");
            let (dc, dg) = (maxdiff(&ca, &cb), maxdiff(&ga, &gb));
            if fc.is_none() && dc > 0.0 {
                fc = Some(il);
                mc = dc;
            }
            if fa.is_none() && dg > 0.0 {
                fa = Some(il);
                ma = dg;
            }
            ri += 1;
        }
        eprintln!(
            "# 청크 {c:>3}: 첫 발산 conv=L{fc:?}({mc:.2e}) ar=L{fa:?}({ma:.2e})"
        );
    }
}
