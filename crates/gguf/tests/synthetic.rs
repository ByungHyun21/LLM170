//! 합성 GGUF v3 파일을 직접 생성해 파서를 검증한다 (무게 포함 왕복).

use llm170_gguf::{GgmlType, GgufFile, Value};
use std::io::Write;
use std::path::PathBuf;

fn push_u32(v: &mut Vec<u8>, x: u32) {
    v.extend_from_slice(&x.to_le_bytes());
}
fn push_u64(v: &mut Vec<u8>, x: u64) {
    v.extend_from_slice(&x.to_le_bytes());
}
fn push_str(v: &mut Vec<u8>, s: &str) {
    push_u64(v, s.len() as u64);
    v.extend_from_slice(s.as_bytes());
}

/// v3 GGUF: kv 3개 + Q4_K 텐서 1개(ne=[512,4]) + 패딩 + 더미 데이터
fn write_sample(path: &std::path::Path, data_len: usize) {
    let mut b = Vec::new();
    b.extend_from_slice(b"GGUF");
    push_u32(&mut b, 3);
    push_u64(&mut b, 1); // n_tensors
    push_u64(&mut b, 3); // n_kv

    // kv: general.architecture(str), general.alignment(u32), test.ctx(u32)
    push_str(&mut b, "general.architecture");
    push_u32(&mut b, 8); // STRING
    push_str(&mut b, "test");
    push_str(&mut b, "general.alignment");
    push_u32(&mut b, 4); // U32
    push_u32(&mut b, 32);
    push_str(&mut b, "test.arr");
    push_u32(&mut b, 9); // ARRAY
    push_u32(&mut b, 6); // F32
    push_u64(&mut b, 3);
    for x in [1.0f32, 2.5, -0.5] {
        push_u32(&mut b, x.to_bits());
    }

    // tensor: Q4_K, ne=[512,4] → 512/256=2블록 ×144B ×4 = 1152B
    push_str(&mut b, "token_embd.weight");
    push_u32(&mut b, 2); // n_dims
    push_u64(&mut b, 512);
    push_u64(&mut b, 4);
    push_u32(&mut b, 12); // GGML_TYPE_Q4_K
    push_u64(&mut b, 0); // offset

    while b.len() % 32 != 0 {
        b.push(0);
    }
    b.resize(b.len() + data_len, 0xAB);

    let mut f = std::fs::File::create(path).unwrap();
    f.write_all(&b).unwrap();
}

fn tmp(name: &str) -> PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!(
        "llm170-gguf-test-{name}-{}.gguf",
        std::process::id()
    ));
    p
}

#[test]
fn roundtrip_v3() {
    let path = tmp("roundtrip");
    write_sample(&path, 1152);

    let f = GgufFile::open(&path).unwrap();
    assert_eq!(f.version, 3);
    assert_eq!(f.alignment, 32);
    assert_eq!(f.arch(), Some("test"));
    assert_eq!(f.kv_u64("test.ctx"), None); // 넣지 않은 키
    assert_eq!(f.kv("general.alignment").and_then(Value::as_u64), Some(32));
    let (et, arr) = f.kv("test.arr").and_then(Value::as_array).unwrap();
    assert_eq!(*et, llm170_gguf::ValueType::F32);
    assert_eq!(arr.len(), 3);

    assert_eq!(f.tensors.len(), 1);
    let t = &f.tensors[0];
    assert_eq!(t.name, "token_embd.weight");
    assert_eq!(t.ne, [512, 4, 1, 1]);
    assert_eq!(t.nbytes(), Some(1152));
    assert_eq!(
        t.file_range(f.data_offset),
        Some((f.data_offset, f.data_offset + 1152))
    );
    assert_eq!(f.tensor_bytes_total(), Some(1152));
    assert_eq!(
        f.find_tensor("token_embd.weight").unwrap().ty,
        GgmlType::Q4K
    );

    let _ = std::fs::remove_file(&path);
}

#[test]
fn bad_magic_rejected() {
    let path = tmp("badmagic");
    let mut f = std::fs::File::create(&path).unwrap();
    f.write_all(b"JUNK").unwrap();
    let err = GgufFile::open(&path).unwrap_err();
    assert!(matches!(err, llm170_gguf::GgufError::NotGguf(_)), "{err}");
    let _ = std::fs::remove_file(&path);
}

#[test]
fn truncated_data_rejected_at_open() {
    let path = tmp("trunc");
    write_sample(&path, 1152);
    let full = std::fs::read(&path).unwrap();
    // 데이터 섹션 일부만 남기고 자르기 — 헤더는 유효. plans/90 A3부터 open 이
    // 텐서 범위 검증(TensorOutOfBounds)으로 절단을 거부한다(구계약: read 시점 감지).
    std::fs::write(&path, &full[..full.len() / 4]).unwrap();
    let err = GgufFile::open(&path).unwrap_err();
    assert!(
        matches!(err, llm170_gguf::GgufError::TensorOutOfBounds { .. }),
        "{err}"
    );
    let _ = std::fs::remove_file(&path);
}

#[test]
fn type_table_matches_upstream() {
    // ggml-common.h static_assert 산식에서 유도한 값 — 대표 타입 재확인
    assert_eq!(GgmlType::Q4K.block_info(), (256, 144));
    assert_eq!(GgmlType::Q6K.block_info(), (256, 210));
    assert_eq!(GgmlType::Q8K.block_info(), (256, 292));
    assert_eq!(GgmlType::Q8_0.block_info(), (32, 34));
    assert_eq!(GgmlType::Iq4Xs.block_info(), (256, 136));
    assert_eq!(GgmlType::Iq4Nl.block_info(), (32, 18));
    assert_eq!(GgmlType::Bf16.block_info(), (1, 2));
    assert_eq!(GgmlType::Tq1_0.block_info(), (256, 54));
    assert_eq!(GgmlType::Nvfp4.block_info(), (64, 36));
    assert_eq!(GgmlType::Q1_0.block_info(), (128, 18));
    assert!((GgmlType::Q4K.bits_per_weight() - 4.5).abs() < 1e-9);
}

/// QA-25(plans/114): n_kv 거대값 — 종전엔 Vec/HashSet with_capacity가
/// capacity overflow 패닉(어보트). Result 경계에서 거부해야 한다.
#[test]
fn huge_nkv_rejected_without_panic() {
    let p = tmp("huge_nkv.gguf");
    let mut b = Vec::new();
    b.extend_from_slice(b"GGUF");
    push_u32(&mut b, 3);
    push_u64(&mut b, 0); // n_tensors
    push_u64(&mut b, u64::MAX); // n_kv — 24바이트 크래프트
    let mut f = std::fs::File::create(&p).unwrap();
    f.write_all(&b).unwrap();
    let r = GgufFile::open(&p);
    assert!(
        matches!(
            r,
            Err(llm170_gguf::GgufError::LengthOverflow { what: "kv", .. })
        ),
        "{r:?}"
    );
}

/// QA-26(plans/114): offset 랩어라운드 — end가 0으로 감겨 경계검증 우회.
#[test]
fn offset_wraparound_rejected() {
    let p = tmp("offset_wrap.gguf");
    let mut b = Vec::new();
    b.extend_from_slice(b"GGUF");
    push_u32(&mut b, 3);
    push_u64(&mut b, 1); // n_tensors
    push_u64(&mut b, 0); // n_kv
    push_str(&mut b, "t");
    push_u32(&mut b, 1); // n_dims
    push_u64(&mut b, 32); // ne[0] — F32 128B
    push_u32(&mut b, 0); // GGML_TYPE_F32
    // data_offset+offset+nbytes == 2^64 → end가 0으로 랩되는 offset.
    let data_offset = (b.len() as u64 + 8).div_ceil(32) * 32; // offset 필드 8B 포함
    let offset = 0u64.wrapping_sub(data_offset + 128);
    push_u64(&mut b, offset);
    let mut f = std::fs::File::create(&p).unwrap();
    f.write_all(&b).unwrap();
    let r = GgufFile::open(&p);
    assert!(
        matches!(r, Err(llm170_gguf::GgufError::OffsetOverflow { .. })),
        "{r:?}"
    );
}

/// QA-31(plans/114): general.alignment u64→u32 절단 — 2^32+32가 32로 잘려
/// 조용한 미정렬로 수용되던 결함. 범위 밖은 거부.
#[test]
fn alignment_over_u32_rejected() {
    let p = tmp("align_over_u32.gguf");
    let mut b = Vec::new();
    b.extend_from_slice(b"GGUF");
    push_u32(&mut b, 3);
    push_u64(&mut b, 0); // n_tensors
    push_u64(&mut b, 1); // n_kv
    push_str(&mut b, "general.alignment");
    push_u32(&mut b, 4); // U32가 아닌 U64로 넣는다 (as_u64 경로)
    push_u32(&mut b, 0);
    push_u32(&mut b, 0);
    push_u64(&mut b, (1u64 << 32) + 32);
    let mut f = std::fs::File::create(&p).unwrap();
    f.write_all(&b).unwrap();
    let r = GgufFile::open(&p);
    assert!(
        matches!(r, Err(llm170_gguf::GgufError::BadAlignment(_))),
        "{r:?}"
    );
}
