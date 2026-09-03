//! mtmd 전처리 포트 — smart_resize + Pillow 호환 bicubic (a=-0.5, 22비트 고정소수점,
//! 분리형 2패스) + pos embd bilinear(align_corners). mtmd-image.cpp 산술 미러.

/// calc_size_preserved_ratio — qwen3vl 규칙 (align=patch·merge, min/max 픽셀).
pub fn smart_resize(w: i64, h: i64, patch: i64, merge: i64, min_tok: i64, max_tok: i64) -> (i64, i64) {
    let align = patch * merge;
    let patch_area = patch * patch * merge * merge;
    let min_pixels = min_tok * patch_area;
    let max_pixels = max_tok * patch_area;
    let round_by = |x: i64| ((x as f64 / align as f64).round() as i64) * align;
    let ceil_by = |x: i64| ((x as f64 / align as f64).ceil() as i64) * align;
    let floor_by = |x: i64| ((x as f64 / align as f64).floor() as i64) * align;
    let mut wb = round_by(w).max(align);
    let mut hb = round_by(h).max(align);
    if wb * hb > max_pixels {
        let beta = ((h as f64 * w as f64) / max_pixels as f64).sqrt();
        hb = floor_by((h as f64 / beta) as i64).max(align);
        wb = floor_by((w as f64 / beta) as i64).max(align);
    } else if wb * hb < min_pixels {
        let beta = (min_pixels as f64 / (h as f64 * w as f64)).sqrt();
        hb = ceil_by((h as f64 * beta) as i64).max(align);
        wb = ceil_by((w as f64 * beta) as i64).max(align);
    }
    (wb, hb)
}

/// Pillow 호환 분리형 리샘플 (RGB u8). algo: true=bicubic, false=bilinear.
pub fn resize_pillow(
    src: &[u8],
    sw: usize,
    sh: usize,
    dw: usize,
    dh: usize,
    bicubic: bool,
) -> Vec<u8> {
    const PRECISION: i64 = 1 << 22;
    let support = if bicubic { 2.0f64 } else { 1.0 };
    let filter = |x: f64| -> f64 {
        let x = x.abs();
        if !bicubic {
            return if x < 1.0 { 1.0 - x } else { 0.0 };
        }
        const A: f64 = -0.5;
        if x < 1.0 {
            return ((A + 2.0) * x - (A + 3.0)) * x * x + 1.0;
        }
        if x < 2.0 {
            return (((x - 5.0) * x + 8.0) * x - 4.0) * A;
        }
        0.0
    };
    // 1차원 가중치 사전계산 — f64 정규화 후 22비트 고정소수점 양자화
    let precompute = |insz: usize, outsz: usize| -> (Vec<(usize, usize)>, Vec<i64>, usize) {
        let scale = insz as f64 / outsz as f64;
        let filterscale = scale.max(1.0);
        let supp = support * filterscale;
        let ksize = (supp.ceil() as usize) * 2 + 1;
        let mut bounds = vec![(0usize, 0usize); outsz];
        let mut weights = vec![0i64; outsz * ksize];
        for xx in 0..outsz {
            let center = (xx as f64 + 0.5) * scale;
            let ww = 1.0 / filterscale;
            let mut xmin = (center - supp + 0.5) as isize;
            if xmin < 0 {
                xmin = 0;
            }
            let mut xmax = (center + supp + 0.5) as isize;
            if xmax > insz as isize {
                xmax = insz as isize;
            }
            let (xmin, xmax) = (xmin as usize, xmax as usize);
            bounds[xx] = (xmin, xmax.saturating_sub(xmin));
            let mut fs = Vec::with_capacity(ksize);
            let mut wsum = 0.0f64;
            for ii in xmin..xmax {
                let d = (ii as f64 + 0.5 - center) * ww;
                let wv = filter(d);
                wsum += wv;
                fs.push(wv);
            }
            if wsum > 0.0 {
                for (k, wv) in fs.iter().enumerate() {
                    weights[xx * ksize + k] = (wv / wsum * PRECISION as f64) as i64;
                }
            }
        }
        (bounds, weights, ksize)
    };
    let (hb, hw, hk) = precompute(sw, dw);
    let (vb, vw, vk) = precompute(sh, dh);
    // 수평 패스: src[sh][sw] → tmp[dh][dw]
    let mut tmp = vec![0u8; sh * dw * 3];
    for y in 0..sh {
        for xx in 0..dw {
            let (xmin, cnt) = hb[xx];
            for c in 0..3 {
                let mut acc = 0i64;
                for k in 0..cnt {
                    let w = hw[xx * hk + k];
                    acc += w * src[(y * sw + xmin + k) * 3 + c] as i64;
                }
                let v = ((acc + (1 << 21)) >> 22).clamp(0, 255);
                tmp[y * dw * 3 + xx * 3 + c] = v as u8;
            }
        }
    }
    // 수직 패스: tmp[sh?]... 수평 결과는 sh행 — dh로 압축
    let mut out = vec![0u8; dh * dw * 3];
    for yy in 0..dh {
        let (ymin, cnt) = vb[yy];
        for xx in 0..dw {
            for c in 0..3 {
                let mut acc = 0i64;
                for k in 0..cnt {
                    let w = vw[yy * vk + k];
                    acc += w * tmp[(ymin + k) * dw * 3 + xx * 3 + c] as i64;
                }
                let v = ((acc + (1 << 21)) >> 22).clamp(0, 255);
                out[(yy * dw + xx) * 3 + c] = v as u8;
            }
        }
    }
    out
}

/// pos embd 2D bilinear (align_corners) — [src_n][ch] (raster) → [dst_h][dst_w][ch].
pub fn pos_resize_bilinear(
    src: &[f32],
    n_side_src: usize,
    dst_w: usize,
    dst_h: usize,
    ch: usize,
) -> Vec<f32> {
    let mut out = vec![0f32; dst_w * dst_h * ch];
    for y in 0..dst_h {
        let sy = if dst_h > 1 {
            y as f64 * (n_side_src - 1) as f64 / (dst_h - 1) as f64
        } else {
            0.0
        };
        let y0 = sy.floor() as usize;
        let y1 = (y0 + 1).min(n_side_src - 1);
        let fy = (sy - y0 as f64) as f32;
        for x in 0..dst_w {
            let sx = if dst_w > 1 {
                x as f64 * (n_side_src - 1) as f64 / (dst_w - 1) as f64
            } else {
                0.0
            };
            let x0 = sx.floor() as usize;
            let x1 = (x0 + 1).min(n_side_src - 1);
            let fx = (sx - x0 as f64) as f32;
            for c in 0..ch {
                let v00 = src[(y0 * n_side_src + x0) * ch + c];
                let v10 = src[(y0 * n_side_src + x1) * ch + c];
                let v01 = src[(y1 * n_side_src + x0) * ch + c];
                let v11 = src[(y1 * n_side_src + x1) * ch + c];
                out[(y * dst_w + x) * ch + c] = v00 * (1.0 - fx) * (1.0 - fy)
                    + v10 * fx * (1.0 - fy)
                    + v01 * (1.0 - fx) * fy
                    + v11 * fx * fy;
            }
        }
    }
    out
}
