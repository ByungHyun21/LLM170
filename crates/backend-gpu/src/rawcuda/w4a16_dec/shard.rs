
use super::W4a16Dec;

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

pub fn forward_2shard_toks(
    a: &mut W4a16Dec,
    b: &mut W4a16Dec,
    slot: usize,
    toks: &[u32],
    t: usize,
    head: bool,
) -> Result<Vec<f32>, String> {
    check_pair(a, b)?;
    if !(2..=super::CHAIN_TMAX).contains(&t) || toks.len() != t {
        return Err(format!(
            "forward_2shard_toks: t={t} toks={} 계약 위반",
            toks.len()
        ));
    }
    let mid = a.forward_prefill_toks(slot, toks, t, false)?;
    if mid.len() != t * a.hidden {
        return Err(format!(
            "forward_2shard_toks: 경계 잔차 {} != t{t}×h{} — 샤드A 비최종 창 계약 위반",
            mid.len(),
            a.hidden
        ));
    }
    b.forward_prefill(slot, &mid, t, head)
}

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
