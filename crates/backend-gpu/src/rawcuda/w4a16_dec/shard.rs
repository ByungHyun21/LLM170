//! [k2 (c) 1단 2026-10-17] PP 2샤드 직렬 오케스트레이터 — 판정 게이트 ①
//! (직렬 PP 골든 = 단일 GPU 대비
//! 비트동일). 샤드A 체인 → 경계 잔차 d2h(호스트 Vec 완충 — 성능 비목표) →
//! 샤드B h2d → 샤드B 체인 → head. 오버랩(더블버퍼)·SCHED/spec 팬아웃·
//! serve 통합·KV 분리 정책은 2단(비목표).
//!
//! 계약: 두 샤드의 `LayerWindow`는 **연속 분할**([0,k)·[k,n_layers))이고
//! hidden·n_layers·slot_pos가 정합(각 forward가 양쪽 slot_pos를 같은 양만
//! 전진시킨다). 가중치 업로드는 호출부 소관 — 각 샤드는 자기 창의 선형만
//! 올리면 된다(체인·멀티 테이블은 창 층만 참조). 임베딩 행·head 로짓의
//! 원천도 호출부(임베딩은 호스트가 공급, head·최종 노름은 샤드B).

use super::W4a16Dec;

/// 샤드 쌍 정합 검사 — 연속 창·동일 형상·B가 마지막 층 소유.
fn check_pair(a: &W4a16Dec, b: &W4a16Dec) -> Result<(), String> {
    if a.hidden != b.hidden || a.n_layers != b.n_layers {
        return Err(format!(
            "2샤드: 형상 불일치 hidden {}/{} n_layers {}/{}",
            a.hidden, b.hidden, a.n_layers, b.n_layers
        ));
    }
    let (wa, wb) = (a.layer_window(), b.layer_window());
    if wa.start != 0 || wb.end != a.n_layers || wa.end != wb.start {
        return Err(format!(
            "2샤드: 창 비연속 A=[{},{}), B=[{},{}) — n_layers={}",
            wa.start, wa.end, wb.start, wb.end, a.n_layers
        ));
    }
    if !b.owns_final_layer() {
        return Err("2샤드: 샤드B가 마지막 층 미소유".into());
    }
    Ok(())
}

/// [k2 (c) 1단] 2샤드 순전파 — 프리필 청크(t≥2 배치)·t=1 공용.
/// `rows` = t개 임베딩 행(t×hidden). `head`=true면 샤드B가 로짓
/// (head_n)을, 아니면 최종 xn(hidden, 마지막 행)을 반환.
/// 비최종 샤드A의 경계 반환은 t×hidden 잔차(전 행 — chain_t_sel 계약).
pub fn forward_2shard(
    a: &mut W4a16Dec,
    b: &mut W4a16Dec,
    slot: usize,
    rows: &[f32],
    t: usize,
    head: bool,
) -> Result<Vec<f32>, String> {
    check_pair(a, b)?;
    let h = a.hidden;
    if rows.len() != t * h {
        return Err(format!("forward_2shard: rows={} != t{t}×h{h}", rows.len()));
    }
    let mid = if t == 1 {
        a.forward_device(slot, &rows[..h])?
    } else {
        a.forward_prefill(slot, rows, t, false)?
    };
    if mid.len() != t * h {
        return Err(format!(
            "forward_2shard: 경계 잔차 {} != t{t}×h{h} — 샤드A 비최종 창 계약 위반",
            mid.len()
        ));
    }
    if t == 1 {
        if head {
            b.forward_device_head(slot, &mid)
        } else {
            b.forward_device(slot, &mid)
        }
    } else {
        b.forward_prefill(slot, &mid, t, head)
    }
}

/// [k2 (c) 1단] 2샤드 디코드 스텝 — 임베딩 행 1개 → 샤드B head+argmax →
/// 토큰(단일 GPU forward_device_argmax와 동일 계약).
pub fn forward_2shard_argmax(
    a: &mut W4a16Dec,
    b: &mut W4a16Dec,
    slot: usize,
    embed_row: &[f32],
) -> Result<u32, String> {
    check_pair(a, b)?;
    if embed_row.len() != a.hidden {
        return Err(format!(
            "forward_2shard_argmax: 임베딩 폭 {} != {}",
            embed_row.len(),
            a.hidden
        ));
    }
    let mid = a.forward_device(slot, embed_row)?;
    b.forward_device_argmax(slot, &mid)
}
