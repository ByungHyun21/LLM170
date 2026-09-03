//! mmproj (CLIP ViT) 비전 인코더 — qwen3vl 계열 (plans/16).
//! llama.cpp mtmd/models/qwen3vl.cpp 산술 미러:
//!   conv 패치 → merge-major 재배열 → +patch_bias → +pos_embd → 27×(LN→qkv→비전 M-RoPE→
//!   MHA→잔차→LN→GELU FFN→잔차) → post_ln → 2×2 결합 → mm.0 GELU → mm.2 → [576][5120].

use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

pub struct Clip {
    file: std::fs::File,
    data_offset: u64,
    tensors: std::collections::HashMap<String, llm170_gguf::TensorInfo>,
    pub image_size: usize,
    pub patch_size: usize,
    pub n_blk: usize,
    n_embd: usize,
    n_head: usize,
    d_head: usize,
    n_ff: usize,
    eps: f32,
    /// 캐시된 f32 가중치 (행major) — 이름 → (rows, data)
    cache: std::collections::HashMap<String, Vec<f32>>,
}

#[inline]
fn cin_base(ky: usize, kx: usize, ps: usize) -> usize {
    kx + ky * ps
}

fn gelu(x: f32) -> f32 {
    const A: f32 = 0.044715;
    const SQ2OPI: f32 = 0.797_884_56;
    0.5 * x * (1.0 + (SQ2OPI * x * (1.0 + A * x * x)).tanh())
}

fn layer_norm(x: &mut [f32], w: &[f32], b: &[f32], eps: f32) {
    let n = x.len();
    let mean: f64 = x.iter().map(|&v| v as f64).sum::<f64>() / n as f64;
    let var: f64 = x.iter().map(|&v| ((v as f64) - mean).powi(2)).sum::<f64>() / n as f64;
    let inv = 1.0 / (var + eps as f64).sqrt();
    for i in 0..n {
        x[i] = (((x[i] as f64) - mean) * inv) as f32 * w[i] + b[i];
    }
}

impl Clip {
    pub fn load(path: &Path) -> Result<Self, String> {
        let g = llm170_gguf::GgufFile::open(path).map_err(|e| e.to_string())?;
        let image_size = g.arch_kv_u64("vision.image_size").ok_or("image_size")? as usize;
        let patch_size = g.arch_kv_u64("vision.patch_size").ok_or("patch_size")? as usize;
        let n_blk = g.arch_kv_u64("vision.block_count").ok_or("block_count")? as usize;
        let n_embd = g.arch_kv_u64("vision.embedding_length").ok_or("embd")? as usize;
        let n_head = g.arch_kv_u64("vision.attention.head_count").ok_or("heads")? as usize;
        let n_ff = g.arch_kv_u64("vision.feed_forward_length").ok_or("ffn")? as usize;
        let eps = g
            .arch_kv_u64("vision.attention.layer_norm_epsilon")
            .unwrap_or(1e-6 as u64) as f32;
        let mut tensors = std::collections::HashMap::new();
        let mut data_offset = 0u64;
        // 텐서 테이블 재수집 — GgufFile이 iterator 노출하지 않으면 헤더 재파싱
        // (find_tensor는 개별 조회 가능하지만 전목록이 필요)
        let mut f = std::fs::File::open(path).map_err(|e| e.to_string())?;
        Self::collect_tensors(&mut f, &mut tensors, &mut data_offset)?;
        let _ = &g;
        Ok(Self {
            file: f,
            data_offset,
            tensors,
            image_size,
            patch_size,
            n_blk,
            n_embd,
            n_head,
            d_head: n_embd / n_head,
            n_ff,
            eps,
            cache: std::collections::HashMap::new(),
        })
    }

    /// 헤더 직접 파싱 — 텐서 전목록 (GgufFile API 한계 우회).
    fn collect_tensors(
        f: &mut std::fs::File,
        out: &mut std::collections::HashMap<String, llm170_gguf::TensorInfo>,
        data_offset: &mut u64,
    ) -> Result<(), String> {
        use std::io::Read;
        let mut hdr = Vec::new();
        f.seek(SeekFrom::Start(0)).map_err(|e| e.to_string())?;
        f.read_to_end(&mut hdr).map_err(|e| e.to_string())?;
        let mut p = 0usize;
        let rd_u32 = |b: &[u8], p: &mut usize| -> u32 {
            let v = u32::from_le_bytes([b[*p], b[*p + 1], b[*p + 2], b[*p + 3]]);
            *p += 4;
            v
        };
        let rd_u64 = |b: &[u8], p: &mut usize| -> u64 {
            let mut v = [0u8; 8];
            v.copy_from_slice(&b[*p..*p + 8]);
            *p += 8;
            u64::from_le_bytes(v)
        };
        let rd_str = |b: &[u8], p: &mut usize| -> String {
            let l = rd_u64(b, p) as usize;
            let s = String::from_utf8_lossy(&b[*p..*p + l]).to_string();
            *p += l;
            s
        };
        let _ = rd_u32(&hdr, &mut p); // magic
        let _ = rd_u32(&hdr, &mut p); // version
        let nd = rd_u64(&hdr, &mut p);
        let nl = rd_u64(&hdr, &mut p);
        let rd_value = |b: &[u8], p: &mut usize| {
            let t = rd_u32(b, p);
            match t {
                8 => {
                    let _ = rd_str(b, p);
                }
                9 => {
                    let it = rd_u32(b, p);
                    let n = rd_u64(b, p);
                    for _ in 0..n {
                        match it {
                            8 => {
                                rd_str(b, p);
                            }
                            _ => {
                                let sz = [1, 1, 2, 2, 4, 4, 4, 1, 0, 0, 8, 8, 8][it as usize];
                                *p += sz as usize;
                            }
                        }
                    }
                }
                _ => {
                    let sz = [1, 1, 2, 2, 4, 4, 4, 1, 0, 0, 8, 8, 8][t as usize];
                    *p += sz as usize;
                }
            }
        };
        for _ in 0..nl {
            let _k = rd_str(&hdr, &mut p);
            rd_value(&hdr, &mut p);
        }
        for _ in 0..nd {
            let name = rd_str(&hdr, &mut p);
            let ndim = rd_u32(&hdr, &mut p) as usize;
            let mut ne = [1u64; 4];
            for d in ne.iter_mut().take(ndim) {
                *d = rd_u64(&hdr, &mut p);
            }
            let ty = rd_u32(&hdr, &mut p);
            let off = rd_u64(&hdr, &mut p);
            let info = llm170_gguf::TensorInfo {
                name: name.clone(),
                n_dims: ndim as u32,
                ne,
                ty: unsafe { std::mem::transmute::<u32, llm170_gguf::GgmlType>(ty) },
                offset: off,
            };
            out.insert(name, info);
        }
        // 데이터 시작 = 텐서 테이블 종료 후 32바이트 정렬 (GGUF 표준)
        *data_offset = ((p as u64) + 31) & !31u64;
        Ok(())
    }

    fn tensor_f32(&mut self, name: &str) -> Result<Vec<f32>, String> {
        if let Some(v) = self.cache.get(name) {
            return Ok(v.clone());
        }
        let info = self
            .tensors
            .get(name)
            .cloned()
            .ok_or_else(|| format!("clip tensor 없음: {name}"))?;
        let n = info.nbytes().ok_or("nbytes")? as usize;
        let mut buf = vec![0u8; n];
        self.file
            .seek(SeekFrom::Start(self.data_offset + info.offset))
            .map_err(|e| e.to_string())?;
        self.file.read_exact(&mut buf).map_err(|e| e.to_string())?;
        let total: u64 = info.ne.iter().product();
        let mut out = vec![0f32; total as usize];
        crate::quant::dequant_row(info.ty, &buf, 0, total, &mut out);
        self.cache.insert(name.to_string(), out.clone());
        Ok(out)
    }

    /// GEMM: out[o] = Σ_i x[i]·W[o][i] + b — W f32 [rows][ni].
    fn mm_bias(x: &[f32], w: &[f32], b: Option<&[f32]>, ni: usize, out: &mut [f32]) {
        let n_out = out.len();
        let nth = std::thread::available_parallelism().map(|v| v.get()).unwrap_or(8).min(32);
        if n_out >= 256 && nth > 1 {
            let csize = n_out.div_ceil(nth);
            std::thread::scope(|sc| {
                let mut off = 0usize;
                let mut hs = Vec::new();
                for ch in out.chunks_mut(csize) {
                    let coff = off;
                    off += ch.len();
                    hs.push(sc.spawn(move || {
                        for (j, ov) in ch.iter_mut().enumerate() {
                            let o = coff + j;
                            *ov = Self::dot_bias(x, &w[o * ni..(o + 1) * ni], b.map(|bb| bb[o]));
                        }
                    }));
                }
                for h in hs { let _ = h.join(); }
            });
            return;
        }
        for (o, ov) in out.iter_mut().enumerate() {
            *ov = Self::dot_bias(x, &w[o * ni..(o + 1) * ni], b.map(|bb| bb[o]));
        }
    }

    /// 배치 GEMM: out[t][o] = Σ x[t][i]·W[o][i] + b — (토큰×행) 청크 병렬.
    fn mm_bias_batch(x: &[Vec<f32>], w: &[f32], b: &[f32], ni: usize, out: &mut [Vec<f32>]) {
        let n_out = out[0].len();
        let nth = std::thread::available_parallelism().map(|v| v.get()).unwrap_or(8).min(32);
        // 토큰 청크 분할 병렬 — 각 스레드가 서로소 토큰 슬라이스 담당.
        let nth = nth.max(1).min(x.len());
        std::thread::scope(|sc| {
            let mut hs = Vec::new();
            let tchunk = x.len().div_ceil(nth);
            let (xo, oo) = (x, out);
            let mut rest = oo;
            let mut t0 = 0usize;
            while t0 < xo.len() {
                let t1 = (t0 + tchunk).min(xo.len());
                let xs = &xo[t0..t1];
                let (head, tail) = rest.split_at_mut(t1 - t0);
                rest = tail;
                hs.push(sc.spawn(move || {
                    for (t, orow) in head.iter_mut().enumerate() {
                        let xr = &xs[t];
                        for o in 0..n_out {
                            let row = &w[o * ni..(o + 1) * ni];
                            let mut acc = b[o];
                            let mut i = 0;
                            while i + 8 <= ni {
                                acc += xr[i] * row[i] + xr[i + 1] * row[i + 1]
                                    + xr[i + 2] * row[i + 2] + xr[i + 3] * row[i + 3]
                                    + xr[i + 4] * row[i + 4] + xr[i + 5] * row[i + 5]
                                    + xr[i + 6] * row[i + 6] + xr[i + 7] * row[i + 7];
                                i += 8;
                            }
                            while i < ni {
                                acc += xr[i] * row[i];
                                i += 1;
                            }
                            orow[o] = acc;
                        }
                    }
                }));
                t0 = t1;
            }
            for h in hs {
                let _ = h.join();
            }
        });
    }

    fn dot_bias(x: &[f32], row: &[f32], bias: Option<f32>) -> f32 {
        let ni = row.len();
        let mut s = bias.unwrap_or(0.0) as f64;
        let mut i = 0;
        while i + 8 <= ni {
            s += f64::from(x[i]) * f64::from(row[i]) + f64::from(x[i + 1]) * f64::from(row[i + 1]);
            s += f64::from(x[i + 2]) * f64::from(row[i + 2]) + f64::from(x[i + 3]) * f64::from(row[i + 3]);
            s += f64::from(x[i + 4]) * f64::from(row[i + 4]) + f64::from(x[i + 5]) * f64::from(row[i + 5]);
            s += f64::from(x[i + 6]) * f64::from(row[i + 6]) + f64::from(x[i + 7]) * f64::from(row[i + 7]);
            i += 8;
        }
        while i < ni {
            s += f64::from(x[i]) * f64::from(row[i]);
            i += 1;
        }
        s as f32
    }

    /// 토큰 범위 attention (스레드 단위) — toks: 처리할 토큰 인덱스 목록.
    fn attn_range(
        qkv: &[Vec<f32>],
        toks: &[usize],
        n_pos: usize,
        n_embd: usize,
        d_head: usize,
        kq_scale: f32,
    ) -> Vec<Vec<f32>> {
        let n_head = n_embd / d_head;
        let mut out = vec![vec![0f32; n_embd]; toks.len()];
        let mut scores = vec![0f32; n_pos];
        for h in 0..n_head {
            for (ti, &t) in toks.iter().enumerate() {
                let qb = h * d_head;
                let kb = n_embd + h * d_head;
                let mut mx = f32::NEG_INFINITY;
                for s2 in 0..n_pos {
                    let mut d = 0.0f32;
                    for i in 0..d_head {
                        d += qkv[t][qb + i] * qkv[s2][kb + i];
                    }
                    scores[s2] = d * kq_scale;
                    mx = mx.max(scores[s2]);
                }
                let mut sum = 0.0f32;
                for v in scores.iter_mut() {
                    *v = (*v - mx).exp();
                    sum += *v;
                }
                for s2 in 0..n_pos {
                    let w2 = scores[s2] / sum;
                    if w2 == 0.0 {
                        continue;
                    }
                    let vb = 2 * n_embd + h * d_head;
                    for i in 0..d_head {
                        out[ti][h * d_head + i] += w2 * qkv[s2][vb + i];
                    }
                }
            }
        }
        out
    }

    /// 이미지 → 비전 토큰 임베딩 [n_tok][5120].
    /// img: RGB f32 [h][w][3] 정규화 완료 (mean/std 적용된 것).
    pub fn encode(&mut self, img: &[f32], w: usize, h: usize) -> Result<Vec<Vec<f32>>, String> {
        let (ps, n_embd, n_head, d_head) =
            (self.patch_size, self.n_embd, self.n_head, self.d_head);
        assert_eq!(w % (ps * 2), 0, "이미지 폭은 patch·merge 배수 필요");
        assert_eq!(h % (ps * 2), 0);
        let (pw, ph) = (w / ps, h / ps);
        let n_pos = pw * ph;

        // ── patch conv → merge-major 토큰 순서 (llama.cpp positions fill 차용)
        let conv_w = self.tensor_f32("v.patch_embd.weight")?;
        let patch_bias = self.tensor_f32("v.patch_embd.bias")?;
        // conv_w 레이아웃 ne=(kx, ky, cin, cout) — ne[0]=kx 최우선 (GGUF 행major)
        let mut toks = vec![vec![0f32; n_embd]; n_pos];
        {
            let kstride = ps * ps; // cin 오프셋
            let cstride = ps * ps * 3; // cout 오프셋
            let mut p = 0usize;
            for y0 in (0..ph).step_by(2) {
                for x0 in (0..pw).step_by(2) {
                    for dx in 0..2 {
                        for dy in 0..2 {
                            let (py, px) = ((y0 + dy) * ps, (x0 + dx) * ps);
                            let t = &mut toks[p];
                            for cout in 0..n_embd {
                                let mut s = patch_bias[cout] as f64;
                                for ky in 0..ps {
                                    for kx in 0..ps {
                                        let wbase = cout * cstride + cin_base(ky, kx, ps);
                                        let ibase = ((py + ky) * w + (px + kx)) * 3;
                                        for cin in 0..3 {
                                            s += f64::from(img[ibase + cin])
                                                * f64::from(conv_w[wbase + cin * kstride]);
                                        }
                                    }
                                }
                                t[cout] = s as f32;
                            }
                            p += 1;
                        }
                    }
                }
            }
        }

        // ── pos embd (raster → merge-major 매핑)
        let pos = self.tensor_f32("v.position_embd.weight")?;
        assert_eq!(pos.len(), n_embd * n_pos, "pos embd 크기 불일치 — 리사이즈 미구현 (768 고정)");
        {
            let mut p = 0usize;
            for y0 in (0..ph).step_by(2) {
                for x0 in (0..pw).step_by(2) {
                    for dx in 0..2 {
                        for dy in 0..2 {
                            let raster = (y0 + dy) * pw + (x0 + dx);
                            let row = &pos[raster * n_embd..(raster + 1) * n_embd];
                            for i in 0..n_embd {
                                toks[p][i] += row[i];
                            }
                            p += 1;
                        }
                    }
                }
            }
        }

        // ── 비전 M-RoPE 사전 계산: 쌍 (i, i+d_head/2), 섹션 [d_head/4 ×4] 중 앞 2개 사용
        // theta_scale = 10000^(-2/(d_head/2)) — n_dims = d_head/2.
        let n_dims = d_head / 2;
        let theta_scale = 10000f32.powf(-2.0 / n_dims as f32);
        let half_rot = d_head / 4; // 섹션 크기 (쌍 단위) = 18
        // 위치: 쌍 p<18 → y, p≥18 → x (재배열된 토큰의 좌표)

        // ── 블록
        let qkv_ni = n_embd;
        let mut cur = toks;
        for il in 0..self.n_blk {
            let ln1w = self.tensor_f32(&format!("v.blk.{il}.ln1.weight"))?;
            let ln1b = self.tensor_f32(&format!("v.blk.{il}.ln1.bias"))?;
            let qkvw = self.tensor_f32(&format!("v.blk.{il}.attn_qkv.weight"))?;
            let qkvb = self.tensor_f32(&format!("v.blk.{il}.attn_qkv.bias"))?;
            let ow = self.tensor_f32(&format!("v.blk.{il}.attn_out.weight"))?;
            let ob = self.tensor_f32(&format!("v.blk.{il}.attn_out.bias"))?;
            let ln2w = self.tensor_f32(&format!("v.blk.{il}.ln2.weight"))?;
            let ln2b = self.tensor_f32(&format!("v.blk.{il}.ln2.bias"))?;
            let upw = self.tensor_f32(&format!("v.blk.{il}.ffn_up.weight"))?;
            let upb = self.tensor_f32(&format!("v.blk.{il}.ffn_up.bias"))?;
            let dnw = self.tensor_f32(&format!("v.blk.{il}.ffn_down.weight"))?;
            let dnb = self.tensor_f32(&format!("v.blk.{il}.ffn_down.bias"))?;

            // LN1 → qkv (배치)
            let mut xn: Vec<Vec<f32>> = cur.clone();
            for x in xn.iter_mut() {
                layer_norm(x, &ln1w, &ln1b, self.eps);
            }
            let mut qkv = vec![vec![0f32; 3 * n_embd]; n_pos];
            Self::mm_bias_batch(&xn, &qkvw, &qkvb, qkv_ni, &mut qkv);

            // 비전 rope (q, k) — 토큰별 (y, x)
            let mut coords = vec![(0u32, 0u32); n_pos];
            {
                let mut p = 0usize;
                for y0 in (0..ph).step_by(2) {
                    for x0 in (0..pw).step_by(2) {
                        for dy in 0..2 {
                            for dx in 0..2 {
                                coords[p] = ((y0 + dy) as u32, (x0 + dx) as u32);
                                p += 1;
                            }
                        }
                    }
                }
            }
            for t in 0..n_pos {
                let (py, px) = coords[t];
                for part in 0..2 {
                    // 0=q, 1=k — 헤드 순회
                    for h in 0..n_head {
                        let hb = part * n_embd + h * d_head;
                        // 쌍 (i, i+n_dims) 회전 — llama.cpp VISION rope 미러
                        for i in 0..n_dims {
                            let sec_pos = if i < half_rot { py } else { px };
                            let local = if i < half_rot { i } else { i - half_rot };
                            let theta = sec_pos as f32 * theta_scale.powi(local as i32);
                            let (c, s) = (theta.cos(), theta.sin());
                            let x0 = qkv[t][hb + i];
                            let x1 = qkv[t][hb + i + n_dims];
                            qkv[t][hb + i] = x0 * c - x1 * s;
                            qkv[t][hb + i + n_dims] = x0 * s + x1 * c;
                        }
                    }
                }
            }

            // MHA (전체 attention, 무마스크)
            let kq_scale = 1.0f32 / (d_head as f32).sqrt();
            let nth = std::thread::available_parallelism().map(|v| v.get()).unwrap_or(8).min(32);
            let mut attn_out = vec![vec![0f32; n_embd]; n_pos];
            {
                let csize = n_pos.div_ceil(nth);
                let idx: Vec<usize> = (0..n_pos).collect();
                std::thread::scope(|sc| {
                    let mut hs = Vec::new();
                    for ch in idx.chunks(csize) {
                        let qkv_ref: &Vec<Vec<f32>> = &qkv;
                        hs.push(sc.spawn(move || {
                            Self::attn_range(qkv_ref, ch, n_pos, n_embd, d_head, kq_scale)
                        }));
                    }
                    let mut off = 0usize;
                    for h in hs {
                        let r = h.join().unwrap();
                        let n = r.len();
                        for (j, row) in r.into_iter().enumerate() {
                            attn_out[off + j] = row;
                        }
                        off += n;
                    }
                });
            }
            // attn_out proj + 잔차 (배치)
            let mut aproj = vec![vec![0f32; n_embd]; n_pos];
            Self::mm_bias_batch(&attn_out, &ow, &ob, n_embd, &mut aproj);
            for t in 0..n_pos {
                for i in 0..n_embd {
                    cur[t][i] += aproj[t][i];
                }
            }

            // LN2 → FFN(GELU) → 잔차 (배치)
            let mut xn2: Vec<Vec<f32>> = cur.clone();
            for x in xn2.iter_mut() {
                layer_norm(x, &ln2w, &ln2b, self.eps);
            }
            let mut mid = vec![vec![0f32; self.n_ff]; n_pos];
            Self::mm_bias_batch(&xn2, &upw, &upb, n_embd, &mut mid);
            for row in mid.iter_mut() {
                for v in row.iter_mut() {
                    *v = gelu(*v);
                }
            }
            let mut dn = vec![vec![0f32; n_embd]; n_pos];
            Self::mm_bias_batch(&mid, &dnw, &dnb, self.n_ff, &mut dn);
            for t in 0..n_pos {
                for i in 0..n_embd {
                    cur[t][i] += dn[t][i];
                }
            }
        }

        // post_ln
        let plw = self.tensor_f32("v.post_ln.weight")?;
        let plb = self.tensor_f32("v.post_ln.bias")?;
        for t in cur.iter_mut() {
            layer_norm(t, &plw, &plb, self.eps);
        }

        // 2×2 결합 → mm.0 GELU → mm.2
        let mm0w = self.tensor_f32("mm.0.weight")?;
        let mm0b = self.tensor_f32("mm.0.bias")?;
        let mm2w = self.tensor_f32("mm.2.weight")?;
        let mm2b = self.tensor_f32("mm.2.bias")?;
        let n_out_tok = n_pos / 4;
        let mut cat_rows = Vec::with_capacity(n_out_tok);
        for m in 0..n_out_tok {
            let mut cat = vec![0f32; n_embd * 4];
            for j in 0..4 {
                cat[j * n_embd..(j + 1) * n_embd].copy_from_slice(&cur[m * 4 + j]);
            }
            cat_rows.push(cat);
        }
        let mut mid = vec![vec![0f32; n_embd * 4]; n_out_tok];
        Self::mm_bias_batch(&cat_rows, &mm0w, &mm0b, n_embd * 4, &mut mid);
        for row in mid.iter_mut() {
            for v in row.iter_mut() {
                *v = gelu(*v);
            }
        }
        let mut out = vec![vec![0f32; 5120]; n_out_tok];
        Self::mm_bias_batch(&mid, &mm2w, &mm2b, n_embd * 4, &mut out);
        Ok(out)
    }
}
