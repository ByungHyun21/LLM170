//! CPU 참조 엔진 체크포인트 코덱 — 순수 바이트 직렬화(의존 없음).
//!
//! 용도: 장문 오라클(`w4a16-ref`)의 중단·재개. 엔진 상태(pos + 전층 상태
//! f32 벡터 전부)를 그대로 덤프한다. 형식(전부 LE):
//! ```text
//! magic 8B "LLM170CK" · ver u32=1 · arch u32 · n_seqs u32 · ctx u32
//! per seq: pos u32 · n_vecs u32 · per vec: len u32 + len×f32(4B)
//! ```
//! 벡터 순서는 아치 계약(호출부 문서): gemma4 = kv_k[0..L)·kv_v[0..L),
//! qwen35 = kv_k·kv_v·gdn_s·conv. 로드는 매직·버전·아치·시퀀스 수·벡터 수·
//! 길이를 전부 대조한다(불일치는 거부 — 조용한 오염 금지).

pub const MAGIC: &[u8; 8] = b"LLM170CK";
pub const VER: u32 = 1;
pub const ARCH_QWEN: u32 = 1;
pub const ARCH_GEMMA: u32 = 2;

fn put_u32(v: &mut Vec<u8>, x: u32) {
    v.extend_from_slice(&x.to_le_bytes());
}

fn get_u32(b: &[u8], p: &mut usize) -> Result<u32, String> {
    if *p + 4 > b.len() {
        return Err(format!("ckpt: 헤더 잘림(offset {p})"));
    }
    let x = u32::from_le_bytes([b[*p], b[*p + 1], b[*p + 2], b[*p + 3]]);
    *p += 4;
    Ok(x)
}

/// 인코딩 — seqs[i] = (pos, 상태 벡터 슬라이스열).
pub fn encode(arch: u32, ctx: u32, seqs: Vec<(u32, Vec<&[f32]>)>) -> Vec<u8> {
    let mut v = Vec::new();
    v.extend_from_slice(MAGIC);
    put_u32(&mut v, VER);
    put_u32(&mut v, arch);
    put_u32(&mut v, seqs.len() as u32);
    put_u32(&mut v, ctx);
    for (pos, vecs) in &seqs {
        put_u32(&mut v, *pos);
        put_u32(&mut v, vecs.len() as u32);
        for w in vecs {
            put_u32(&mut v, w.len() as u32);
            v.reserve(w.len() * 4);
            for x in w.iter() {
                v.extend_from_slice(&x.to_le_bytes());
            }
        }
    }
    v
}

/// 디코딩 — (arch, ctx, per-seq (pos, 벡터열)). 호출부가 아치·형상을 대조한다.
pub fn decode(blob: &[u8]) -> Result<(u32, u32, Vec<(u32, Vec<Vec<f32>>)>), String> {
    let mut p = 8usize;
    if blob.len() < 8 || &blob[..8] != MAGIC {
        return Err("ckpt: 매직 불일치".into());
    }
    let ver = get_u32(blob, &mut p)?;
    if ver != VER {
        return Err(format!("ckpt: 버전 {ver} ≠ {VER}"));
    }
    let arch = get_u32(blob, &mut p)?;
    let n_seqs = get_u32(blob, &mut p)? as usize;
    if n_seqs > 1024 {
        return Err(format!("ckpt: 시퀀스 수 {n_seqs} 비정상"));
    }
    let ctx = get_u32(blob, &mut p)?;
    let mut seqs = Vec::with_capacity(n_seqs);
    for _ in 0..n_seqs {
        let pos = get_u32(blob, &mut p)?;
        let n_vecs = get_u32(blob, &mut p)? as usize;
        if n_vecs > 4096 {
            return Err(format!("ckpt: 벡터 수 {n_vecs} 비정상"));
        }
        let mut vecs = Vec::with_capacity(n_vecs);
        for _ in 0..n_vecs {
            let len = get_u32(blob, &mut p)? as usize;
            if len > (1usize << 32) || p + len * 4 > blob.len() {
                return Err(format!("ckpt: 벡터 길이 {len} 비정상(잘림)"));
            }
            let mut w = Vec::with_capacity(len);
            for i in 0..len {
                let o = p + i * 4;
                w.push(f32::from_le_bytes([
                    blob[o],
                    blob[o + 1],
                    blob[o + 2],
                    blob[o + 3],
                ]));
            }
            p += len * 4;
            vecs.push(w);
        }
        seqs.push((pos, vecs));
    }
    Ok((arch, ctx, seqs))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 라운드트립 + 비트 보존.
    #[test]
    fn roundtrip_bit_exact() {
        let a = vec![1.5f32, -2.25, f32::from_bits(0x3f800001)];
        let b: Vec<f32> = (0..300).map(|i| i as f32 * 0.5 - 75.0).collect();
        let blob = encode(
            ARCH_GEMMA,
            4096,
            vec![(7, vec![&a[..], &b[..]]), (0, vec![])],
        );
        let (arch, ctx, seqs) = decode(&blob).expect("복원");
        assert_eq!(arch, ARCH_GEMMA);
        assert_eq!(ctx, 4096);
        assert_eq!(seqs.len(), 2);
        assert_eq!(seqs[0].0, 7);
        assert_eq!(seqs[0].1[0], a, "비트 정확");
        assert_eq!(seqs[0].1[1], b);
        assert!(seqs[1].1.is_empty());
    }

    /// 손상·오타입 거부.
    #[test]
    fn rejects_corrupt() {
        let a = vec![1.0f32];
        let mut blob = encode(ARCH_QWEN, 128, vec![(0, vec![&a[..]])]);
        assert!(decode(&blob[..blob.len() - 2]).is_err(), "잘림 거부");
        blob[0] = b'X';
        assert!(decode(&blob).is_err(), "매직 거부");
        let mut blob2 = encode(ARCH_QWEN, 128, vec![(0, vec![&a[..]])]);
        // 버전 필드(8..12) 변조.
        blob2[8] = 99;
        assert!(decode(&blob2).is_err(), "버전 거부");
    }
}
