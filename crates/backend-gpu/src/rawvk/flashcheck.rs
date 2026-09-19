//! vk-flash-check — qsa_flash(단일쿼리) vs qsa_flash_gq(다중쿼리) A/B 프로브
//! (plans/83 D). 합성 결정적 데이터로 두 커널의 출력을 맞대어 다중쿼리 판의
//! 정확성을 모델 적재 없이 검증한다. 산술 순서 차이를 감안해 상대 비교.

use crate::rawvk::context::{Pipes, VkCtx};
use crate::rawvk::decoder::{
    QSA_FLASH_GQ_SPV, QSA_FLASH_SPV,
};

const T: usize = 208;    // 게이트 프롬프트 형상
const NH: usize = 24;
const NK: usize = 4;
const HD: usize = 256;  // 27B head_dim — 엔진 hparams 일치
const POS0: usize = 0;
const CTX: usize = 256;

fn synth(i: usize) -> f32 {
    (((i * 2654435761) % 100003) as i32 - 50000) as f32 * 1e-4
}

/// vk-flash-check — 두 flash 커널 A/B. 성공시 요약, 실패시 첫 불일치.
pub fn flash_check() -> Result<String, String> {
    let mut ctx = VkCtx::new()?;

    // Q [t][nh*2hd] — q|gate 인터리브
    let file_mode = std::path::Path::new("/tmp/gq_q.f32").exists();
    let mut q = vec![0f32; T * NH * 2 * HD];
    let mut k = vec![0f32; CTX * NK * HD];
    let mut v = vec![0f32; CTX * NK * HD];
    if file_mode {
        let load = |p: &str, n: usize| -> Vec<f32> {
            let b = std::fs::read(p).unwrap();
            bytemuck::cast_slice(&b)[..n].to_vec()
        };
        q = load("/tmp/gq_q.f32", T * NH * 2 * HD);
        k = load("/tmp/gq_k.f32", (POS0 + T) * NK * HD);
        v = load("/tmp/gq_v.f32", (POS0 + T) * NK * HD);
        eprintln!("[flash-check] 파일 입력 모드 — 엔진 활성 덤프 사용");
    } else {
        for i in 0..q.len() { q[i] = synth(i); }
        for i in 0..k.len() { k[i] = synth(i + 77); v[i] = synth(i + 1_000_003); }
    }
    let out_n = T * NH * HD;
    let mut o_old = vec![0f32; out_n];
    let mut o_new = vec![0f32; out_n];

    let bq = ctx.alloc_host(q.len() * 4)?;
    let bk = ctx.alloc_host(k.len() * 4)?;
    let bv = ctx.alloc_host(v.len() * 4)?;
    let bo = ctx.alloc_host(out_n * 4)?;
    unsafe {
        std::ptr::copy_nonoverlapping(q.as_ptr() as *const u8, bq.ptr, q.len() * 4);
        std::ptr::copy_nonoverlapping(k.as_ptr() as *const u8, bk.ptr, k.len() * 4);
        std::ptr::copy_nonoverlapping(v.as_ptr() as *const u8, bv.ptr, v.len() * 4);
    }

    let push_old: Vec<u8> = [POS0 as u32, NH as u32, NK as u32, HD as u32]
        .iter().flat_map(|x| x.to_le_bytes()).collect();
    let push_new: Vec<u8> = [POS0 as u32, NH as u32, NK as u32, HD as u32, T as u32]
        .iter().flat_map(|x| x.to_le_bytes()).collect();
    let binds = [bq.buf, bk.buf, bv.buf, bo.buf];

    // OLD — grid (t, nh)
    let p_old: Pipes = ctx.pipeline_pipes(QSA_FLASH_SPV, 4, 16)?;
    ctx.begin_batch()?;
    let ds = ctx.bind_ds(&p_old, &binds)?;
    ctx.run(p_old.pl, ds, p_old.pipe, &push_old, T as u32, NH as u32, 1)?;
    ctx.end_batch_wait()?;
    unsafe { std::ptr::copy_nonoverlapping(bo.ptr as *const f32, o_old.as_mut_ptr(), out_n) };

    // NEW — grid (t, nk) [R=1 판]. o 버퍼를 패턴으로 덮어 쓴 뒤 실행해
    // 미기록 행이 OLD 결과로 남는 참사를 차단한다.
    let p_new: Pipes = ctx.pipeline_pipes(QSA_FLASH_GQ_SPV, 4, 20)?;
    for x in o_new.iter_mut() { *x = 1e30; }
    unsafe { std::ptr::copy_nonoverlapping(o_new.as_ptr() as *const u8, bo.ptr, out_n * 4) };
    ctx.begin_batch()?;
    let ds = ctx.bind_ds(&p_new, &binds)?;
    ctx.run(p_new.pl, ds, p_new.pipe, &push_new, T as u32, NK as u32, 1)?;
    ctx.end_batch_wait()?;
    unsafe { std::ptr::copy_nonoverlapping(bo.ptr as *const f32, o_new.as_mut_ptr(), out_n) };

    // CPU 참조 — 표준 게이티드 어텐션 (순서: 키 오름차순 f32 합)
    let gq = NH / NK;
    let mut o_ref = vec![0f32; out_n];
    for row in 0..T {
        let pos = POS0 + row;
        for h in 0..NH {
            let kvh = h / gq;
            let qb = row * NH * 2 * HD + h * 2 * HD;
            let mut scores = vec![0f32; pos + 1];
            for p in 0..=pos {
                let mut s = 0f32;
                let kb = p * NK * HD + kvh * HD;
                for d in 0..HD {
                    s += q[qb + d] * k[kb + d];
                }
                scores[p] = s;
            }
            let mx = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0f32;
            let mut probs = vec![0f32; pos + 1];
            for p in 0..=pos {
                probs[p] = (scores[p] - mx).exp();
                sum += probs[p];
            }
            for d in 0..HD {
                let mut acc = 0f32;
                for p in 0..=pos {
                    acc += probs[p] * v[p * NK * HD + kvh * HD + d];
                }
                let g = 1.0 / (1.0 + (-q[qb + HD + d]).exp());
                o_ref[(row * NH + h) * HD + d] = acc / sum * g;
            }
        }
    }

    let maxdiff = |a: &[f32], b: &[f32]| -> (f32, usize) {
        let mut mx = 0f32;
        let mut at = 0usize;
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            let d = (x - y).abs();
            if d > mx {
                mx = d;
                at = i;
            }
        }
        (mx, at)
    };
    let (md_old, at_old) = maxdiff(&o_old, &o_ref);
    let (md_new, at_new) = maxdiff(&o_new, &o_ref);

    let at = |i: usize| {
        let d = i % HD;
        let h = (i / HD) % NH;
        let r = i / (HD * NH);
        (r, h, d)
    };
    let (r1, h1, d1) = at(at_old);
    let (r2, h2, d2) = at(at_new);
    let mut out = format!(
        "vk-flash-check t={T} pos0={POS0} nh={NH} nk={NK} hd={HD}\n  OLD vs ref: max|D|={md_old:.3e} @ (row={r1},h={h1},d={d1}) old={:.5} ref={:.5}\n  NEW vs ref: max|D|={md_new:.3e} @ (row={r2},h={h2},d={d2}) new={:.5} ref={:.5}\n",
        o_old[at_old], o_ref[at_old], o_new[at_new], o_ref[at_new]
    );
    let tol = 5e-3;
    if md_old > tol {
        out.push_str(&format!("FAIL: OLD 커널 자체가 참조와 불일치 ({md_old:.3e} > {tol})\n"));
        return Ok(out);
    }
    if md_new > tol {
        // NEW 불일치 — 행별 최대 위치로 다중행 경계 버그 여부 판단 근거 제공
        out.push_str(&format!("FAIL: NEW 커널 불일치 ({md_new:.3e} > {tol})\n"));
        return Ok(out);
    }
    out.push_str("PASS: 두 커널 모두 참조 일치\n");
    Ok(out)
}
