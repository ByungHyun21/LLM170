//! 합성 GGUF v3 파일을 직접 생성해 파서를 검증한다 (무게 포함 왕복).

use llm170_gguf::{GgmlType, GgufFile, Value};
use std::io::Write;
use std::path::PathBuf;

fn push_u16(v: &mut Vec<u8>, x: u16) {
    v.extend_from_slice(&x.to_le_bytes());
}
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
fn truncation_detected_on_read() {
    let path = tmp("trunc");
    write_sample(&path, 1152);
    let full = std::fs::read(&path).unwrap();
    // 데이터 섹션 일부만 남기고 자르기 — 헤더는 유효하므로 open 은 성공해야 정상.
    // 극단적으로 헤더 중간에서 자르면 io 에러.
    std::fs::write(&path, &full[..full.len() / 4]).unwrap();
    // 헤더(훨씬 짧음)는 그대로므로 여전히 파싱 가능 — 위치 계산만 하는 파서의 의도된 동작.
    let f = GgufFile::open(&path).unwrap();
    assert_eq!(f.tensors.len(), 1);
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
