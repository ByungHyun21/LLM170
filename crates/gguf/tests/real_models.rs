//! 실제 기준 모델 GGUF 검증 (이 개발기에 파일이 있을 때만).
//!
//! 기대값 근거: 2026-08-30 실측 덤프 (llm170 gguf-dump) + ISSUES.md.
//! 유의: qwen35.block_count=65 — 64본체층 + MTP층(blk.64) 포함.
//! qwen4exp split: part1(no=0)은 메타 전용(텐서 0), split.tensors.count=1224는 전체 모델 텐서 수.

use llm170_gguf::GgufFile;

const P27B: &str = "/home/yoon/local_llm/models/qwen3.8-27b/Qwen3.8-27B-UD-Q4_K_XL.gguf";
const PFLASH: &str = "/home/yoon/local_llm/models/qwen3.8-Flash-Next/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf";

fn open_if_exists(path: &str) -> Option<GgufFile> {
    let p = std::path::Path::new(path);
    if !p.exists() {
        eprintln!("skip (파일 없음): {path}");
        return None;
    }
    Some(GgufFile::open(p).expect("실제 GGUF 파싱 실패"))
}

#[test]
fn qwen35_27b_ud_q4_k_xl() {
    let Some(f) = open_if_exists(P27B) else { return };
    assert_eq!(f.version, 3);
    assert_eq!(f.arch(), Some("qwen35"));

    // 하이퍼파라미터 (config.json + 실측 덤프)
    assert_eq!(f.arch_kv_u64("block_count"), Some(65)); // 64층 + MTP(blk.64)
    assert_eq!(f.arch_kv_u64("embedding_length"), Some(5120));
    assert_eq!(f.arch_kv_u64("feed_forward_length"), Some(17408));
    assert_eq!(f.arch_kv_u64("context_length"), Some(262144));
    assert_eq!(f.arch_kv_u64("full_attention_interval"), Some(4));
    assert_eq!(f.arch_kv_u64("nextn_predict_layers"), Some(1));
    assert_eq!(f.arch_kv_u64("attention.head_count"), Some(24));
    assert_eq!(f.arch_kv_u64("attention.head_count_kv"), Some(4));
    assert_eq!(f.arch_kv_u64("attention.key_length"), Some(256));
    assert_eq!(f.arch_kv("rope.freq_base").and_then(llm170_gguf::Value::as_f64), Some(10_000_000.0));

    // 토크나이저
    let tokens = f.kv("tokenizer.ggml.tokens").and_then(llm170_gguf::Value::as_array);
    assert_eq!(tokens.map(|(_, v)| v.len()), Some(248320), "vocab 크기");
    assert_eq!(f.kv_str("tokenizer.ggml.pre"), Some("qwen35"));

    // 텐서 구조 (실측 866개)
    assert_eq!(f.tensors.len(), 866);
    let n_blk = f.tensors.iter().filter(|t| t.name.starts_with("blk.")).count();
    let max_blk = f.tensors.iter()
        .filter_map(|t| t.name.strip_prefix("blk."))
        .filter_map(|r| r.split('.').next())
        .filter_map(|n| n.parse::<u32>().ok())
        .max();
    assert_eq!(max_blk, Some(64), "블록 인덱스 0..=64 (64 = MTP)");
    assert!(f.tensors.iter().any(|t| t.name.starts_with("blk.64.nextn.")), "MTP 텐서");
    assert!(f.find_tensor("token_embd.weight").is_some());
    assert!(f.find_tensor("output.weight").is_some());
    assert!(f.find_tensor("per_layer_token_embd.weight").is_none(), "27B에는 PLE 없음");

    // 무게가 파일에 정확히 들어맞는다 (실측: slack 0)
    let total = f.tensor_bytes_total().expect("모든 텐서 nbytes 계산 가능");
    assert_eq!(f.data_offset + total, f.file_size);
    assert_eq!(f.tensors.first().unwrap().offset, 0);

    // 모델카드 샘플링 기본값 (run.sh 플래그와 동일 출처)
    assert_eq!(f.kv_u64("general.sampling.top_k"), Some(20));
}

#[test]
fn qwen4exp_flash_next_split1() {
    let Some(f) = open_if_exists(PFLASH) else { return };
    assert_eq!(f.arch(), Some("qwen4exp"));

    let s = f.split().expect("split 메타데이터");
    assert_eq!((s.no, s.count), (0, 4));
    assert_eq!(s.tensors_count, 1224, "전체 모델 텐서 수 (ISSUES.md 기록)");
    assert_eq!(f.tensors.len(), 0, "part1(no=0)은 메타 전용 — 텐서 없음");

    // 하이퍼파라미터 (실측 덤프)
    assert_eq!(f.arch_kv_u64("block_count"), Some(48));
    assert_eq!(f.arch_kv_u64("embedding_length"), Some(2560));
    assert_eq!(f.arch_kv_u64("context_length"), Some(262144));
    assert_eq!(f.arch_kv_u64("expert_count"), Some(512));
    assert_eq!(f.arch_kv_u64("expert_used_count"), Some(10));
    assert_eq!(f.arch_kv_u64("expert_feed_forward_length"), Some(640));
    assert_eq!(f.arch_kv_u64("expert_shared_feed_forward_length"), Some(640), "shared expert FFN(구 추정) 확정");
    assert_eq!(f.arch_kv_u64("hyper_connection.count"), Some(4));
    assert_eq!(f.arch_kv_u64("hyper_connection.low_rank"), Some(320));
    assert_eq!(f.arch_kv_u64("attention.indexer.head_count"), Some(4));
    assert_eq!(f.arch_kv_u64("attention.indexer.key_length"), Some(128));
    assert_eq!(f.arch_kv_u64("attention.indexer.top_k"), Some(2048), "research 문서의 512가 아님 — 실측 2048");
    assert_eq!(f.arch_kv("rope.freq_base").and_then(llm170_gguf::Value::as_f64), Some(10_000_000.0), "구 미확정 — 실측 확정");

    // compress_ratios: QSA층 4, GDN층 0 — 48개 배열
    let cr = f.kv("qwen4exp.attention.compress_ratios")
        .and_then(llm170_gguf::Value::as_array).unwrap().1;
    assert_eq!(cr.len(), 48);
    let qsa4 = cr.iter().filter(|v| v.as_u64() == Some(4)).count();
    assert_eq!(qsa4, 12, "QSA층(compress=4)은 12개");
    assert_eq!(cr[3].as_u64(), Some(4), "첫 QSA층은 il=3");

    // PLE (실측: ngram 3, heads/ngram 8, per-layer-input 160)
    let (et, ple_layers) = f.kv("qwen4exp.ple.layers").and_then(llm170_gguf::Value::as_array).unwrap();
    assert_eq!(*et, llm170_gguf::ValueType::I32);
    assert_eq!(ple_layers.len(), 1);
    assert_eq!(f.arch_kv_u64("ple.ngram_size"), Some(3));
    assert_eq!(f.arch_kv_u64("ple.heads_per_ngram"), Some(8));
    assert_eq!(f.arch_kv_u64("ple.conv_kernel"), Some(4));
    assert_eq!(f.arch_kv_u64("ple.eos_token_id"), Some(248044));
    assert_eq!(f.arch_kv_u64("embedding_length_per_layer_input"), Some(160));
    let (et, offs) = f.kv("qwen4exp.ple.head_offsets").and_then(llm170_gguf::Value::as_array).unwrap();
    assert_eq!(*et, llm170_gguf::ValueType::U64);
    assert_eq!(offs.len(), 16, "PLE 헤드 수 = (ngram-1)×heads_per_ngram = 16");

    // MTP는 UD 변환에서 drop (ISSUES.md)
    assert_eq!(f.arch_kv_u64("nextn_predict_layers"), None);
}
