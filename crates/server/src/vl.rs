//! vl — mmproj 비전 인코딩 + LLM 스플라이스 추론 (plans/16).
//! main.rs에서 이관(plans/35 P4) — 루트 헬퍼(parse_ids_ref·usage_err) 공유.

use std::path::PathBuf;
use std::process::ExitCode;

use crate::{parse_ids_ref, usage_err};

/// vl — mmproj 비전 인코딩 + LLM 스플라이스 추론 (plans/16).
pub fn cmd_vl(args: &[String], ma: &crate::ModelArgs) -> ExitCode {
    let mut mmproj: Option<PathBuf> = None;
    let mut images: Vec<PathBuf> = Vec::new();
    let mut n_predict = 48usize;
    let mut ctx = 4096usize;
    let backend = ma.backend.clone().unwrap_or_else(|| "gpu".into());
    let gpu_runtime = ma
        .gpu_runtime
        .clone()
        .or_else(|| std::env::var("LLM170_GPU_RUNTIME").ok())
        .unwrap_or_else(|| "hip".into());
    let mut spec_k = 0usize;
    // 장문·임의 질문 지원 (plans/28): prefix는 vision_start 앞, question은
    // vision_end 뒤 — 기본(미지정)은 기존 하드코딩 프롬프트와 동일.
    let mut prefix_ids: Vec<u32> = Vec::new();
    let mut question_ids: Option<Vec<u32>> = None;
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--mmproj" => mmproj = it.next().map(PathBuf::from),
            "--image" => {
                if let Some(p) = it.next() {
                    images.push(PathBuf::from(p));
                }
            }
            "--n-predict" => n_predict = it.next().and_then(|v| v.parse().ok()).unwrap_or(48),
            "--ctx" => ctx = it.next().and_then(|v| v.parse().ok()).unwrap_or(4096),
            "--spec" => spec_k = it.next().and_then(|v| v.parse().ok()).unwrap_or(0),
            "--prefix-tokens" => match it.next().map(String::as_str).and_then(parse_ids_ref) {
                Some(v) => prefix_ids = v,
                None => return usage_err("--prefix-tokens requires comma-separated ids"),
            },
            "--question-tokens" => match it.next().map(String::as_str).and_then(parse_ids_ref) {
                Some(v) => question_ids = Some(v),
                None => return usage_err("--question-tokens requires comma-separated ids"),
            },
            _ => {}
        }
    }
    let model = ma.model.clone().map(PathBuf::from);
    let (model, mmproj) = match (model, mmproj) {
        (Some(m), Some(p)) => (m, p),
        _ => {
            eprintln!(
                "usage: llm170 vl --model <llm.gguf> --mmproj <mmproj.gguf> --image <img> [--image <img>...] [--spec k] [--n-predict N]"
            );
            return ExitCode::from(2);
        }
    };
    if images.is_empty() {
        eprintln!("usage: at least one --image required");
        return ExitCode::from(2);
    }
    // qwen3.8 VL 템플릿 (서버 /tokenize 확정):
    // <|im_start|>user\n [prefix] <|vision_start|><|image_pad|><|vision_end|> [question]
    // <|im_end|>\n<|im_start|>assistant\n — prefix/question 미지정 시 기존 프롬프트 동일.
    let question: Vec<u32> =
        question_ids.unwrap_or_else(|| vec![72240, 411, 2099, 303, 799, 2716, 11316, 13]);
    let mut prompt: Vec<u32> = Vec::with_capacity(prefix_ids.len() + question.len() + 12);
    prompt.extend_from_slice(&[248045, 846, 198]);
    prompt.extend_from_slice(&prefix_ids);
    prompt.extend_from_slice(&[248053, 248056, 248054]);
    prompt.extend_from_slice(&question);
    prompt.extend_from_slice(&[248046, 198, 248045, 74455, 198]);
    // 1) 이미지 → 스마트리사이즈·정규화 → CLIP 인코딩 (이미지별 = 시퀀스별)
    let t0 = std::time::Instant::now();
    let mut clip = match llm170_core::clip::Clip::load(&mmproj) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("clip load: {e}");
            return ExitCode::FAILURE;
        }
    };
    let n_img = images.len();
    // QA-13: Vit tmax는 전체 이미지 최댓값으로 — 종전 첫 이미지 기준 고정으로
    // 이후 큰 이미지가 "해상도 초과" Err → CPU 폴백(수 배 느림)이었다. 이미지
    // 목록은 루프 전에 전부 알 수 있다(차원 프로브는 메타 판독뿐 — 저렴).
    let tmax_max = images
        .iter()
        .filter_map(|p| image::image_dimensions(p).ok())
        .map(|(w, h)| {
            let (tw, th) =
                llm170_core::clip_preproc::smart_resize(w as i64, h as i64, 16, 2, 8, 4096);
            (tw as usize / 16) * (th as usize / 16)
        })
        .max()
        .unwrap_or(0);
    let mut all_vis: Vec<Vec<Vec<f32>>> = Vec::with_capacity(n_img);
    let mut vit_cache: Option<(
        std::sync::Arc<llm170_backend_gpu::rawhip::RawCtx>,
        std::sync::Arc<llm170_backend_gpu::rawhip::vit::Vit>,
        usize,
    )> = None;
    for (si, ipath) in images.iter().enumerate() {
        let img = match image::open(ipath) {
            Ok(i) => i.to_rgb8(),
            Err(e) => {
                eprintln!("image: {e}");
                return ExitCode::FAILURE;
            }
        };
        let (iw, ih) = (img.width() as i64, img.height() as i64);
        // qwen3vl smart_resize (align 32, 토큰 8..4096) + Pillow bicubic
        let (tw, th) = llm170_core::clip_preproc::smart_resize(iw, ih, 16, 2, 8, 4096);
        let raw = img.as_raw();
        let rgb8 = llm170_core::clip_preproc::resize_pillow(
            raw,
            iw as usize,
            ih as usize,
            tw as usize,
            th as usize,
            true,
        );
        eprintln!("# resize[{si}] {iw}x{ih} -> {tw}x{th}");
        let (tw, th) = (tw as usize, th as usize);
        let mut px = vec![0f32; tw * th * 3];
        for (i, v) in rgb8.iter().enumerate() {
            px[i] = ((*v as f32) / 255.0 - 0.5) / 0.5;
        }
        let vis = if backend != "cpu" {
            // GPU 경로 (plans/17): CPU conv+pos → ViT 27블록·merger GPU
            let (n_embd, n_head, n_blk, eps) = (1152usize, 16usize, 27usize, 1e-6f32);
            let tmax = (tw / 16) * (th / 16);
            let v = (|| -> Result<Vec<Vec<f32>>, String> {
                if vit_cache.is_none() {
                    let weights = clip.vit_weights()?;
                    let n_ff = clip.n_ff();
                    let tw0 = std::time::Instant::now();
                    #[allow(clippy::arc_with_non_send_sync)]
                    // vl 단일스레드 경로 — HIP 런타임 직렬화(Q4Acc unsafe Send/Sync와 동일 계약)
                    let ctx = std::sync::Arc::new(llm170_backend_gpu::rawhip::RawCtx::new()?);
                    let vit = llm170_backend_gpu::rawhip::vit::Vit::new(
                        ctx.clone(),
                        weights,
                        n_embd,
                        n_head,
                        n_ff,
                        n_blk,
                        eps,
                        tmax_max,
                    )?;
                    eprintln!(
                        "# vit weights+upload {:.1}s (tmax={tmax_max})",
                        tw0.elapsed().as_secs_f64()
                    );
                    vit_cache = Some((ctx, std::sync::Arc::new(vit), tmax_max));
                }
                let (_, vit, tmax0) = vit_cache.as_ref().unwrap();
                if tmax > *tmax0 {
                    return Err(format!("tmax {tmax} > 초기화 {tmax0} — 이미지 해상도 초과"));
                }
                let tp0 = std::time::Instant::now();
                let (toks, yx, pw, ph) = clip.prep_tokens(&px, tw, th)?;
                eprintln!("# vit prep(conv) {:.1}s", tp0.elapsed().as_secs_f64());
                let tf0 = std::time::Instant::now();
                let flat = vit.forward(&toks, &yx, pw, ph)?;
                eprintln!("# vit forward {:.1}s", tf0.elapsed().as_secs_f64());
                let n_out = flat.len() / 5120;
                Ok((0..n_out)
                    .map(|i| flat[i * 5120..(i + 1) * 5120].to_vec())
                    .collect())
            })();
            match v {
                Ok(rows) => rows,
                Err(e) => {
                    eprintln!("vit gpu: {e} — CPU 폴백");
                    match clip.encode(&px, tw, th) {
                        Ok(r) => r,
                        Err(e) => {
                            eprintln!("clip encode: {e}");
                            return ExitCode::FAILURE;
                        }
                    }
                }
            }
        } else {
            match clip.encode(&px, tw, th) {
                Ok(v) => v,
                Err(e) => {
                    eprintln!("clip encode: {e}");
                    return ExitCode::FAILURE;
                }
            }
        };
        eprintln!(
            "# clip[{si}]: {} tokens (총 {:.1}s)",
            vis.len(),
            t0.elapsed().as_secs_f64()
        );
        if std::env::var_os("LLM170_VIS_HASH").is_some() {
            let mut x: u64 = 0x9E3779B97F4A7C15;
            for row in &vis[..vis.len().min(2)] {
                for &v in row[..row.len().min(256)].iter() {
                    x ^= (v.to_bits() as u64).wrapping_mul(0xC2B2AE3D27D4EB4F);
                    x = x.rotate_left(17);
                }
            }
            eprintln!("# vis_hash[{si}] {x:016x} v0={:.6}", vis[0][0]);
        }
        all_vis.push(vis);
    }
    // 3) LLM
    let m = match llm170_core::qwen35::Model::load(&model) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("model: {e}");
            return ExitCode::FAILURE;
        }
    };
    let mut eng = llm170_core::qwen35::Engine::new(m, n_img, ctx);
    if spec_k > 0 {
        eng.mtp_wanted = true; // 스펙 의도 — prefill 훅 활성
    }
    if backend != "cpu" {
        // GPU 부착 — 단일 경로(attach_q4/35 정책과 동일 구조. vl은 경고 후
        // CPU 지속 — 종전과 동일하나 vk-q35 게이트·VkAcc 분기가 serve와 동일해짐).
        eng = crate::engine::attach_q35(
            eng,
            gpu_runtime == "vulkan",
            crate::engine::AttachPolicy::Warn,
        )
        .unwrap_or_else(|_| unreachable!("Warn policy cannot fail"));
    }
    let eos = 248044u32;
    let t1 = std::time::Instant::now();
    let mut last_logits = Vec::with_capacity(n_img);
    for s in 0..n_img {
        let l = match eng.prefill_vision(s, &prompt, 248056, &all_vis[s]) {
            Ok(l) => l,
            Err(e) => {
                eprintln!("prefill_vision: {e}");
                return ExitCode::FAILURE;
            }
        };
        last_logits.push(l);
    }
    eprintln!("# prefill({n_img} seqs) {:.1}s", t1.elapsed().as_secs_f64());
    let mut finished = vec![false; n_img];
    let mut gen_toks: Vec<Vec<u32>> = vec![Vec::new(); n_img];
    let mut texts: Vec<String> = vec![String::new(); n_img];
    let next: Vec<u32> = last_logits
        .iter()
        .map(|l| llm170_core::qwen35::greedy(l))
        .collect();
    // 시퀀스별 유효 프롬프트 길이 (마커 1 → vis 행수 치환) — JSONL pos 기준.
    let base_len: Vec<usize> = (0..n_img)
        .map(|s| prompt.len() - 1 + all_vis[s].len())
        .collect();
    // 토큰 스트림 JSONL(infer와 동일 형식) — CPU/GPU·spec 동일성 판정용.
    for s in 0..n_img {
        println!(
            "{{\"seq\":{s},\"pos\":{},\"token\":{}}}",
            base_len[s], next[s]
        );
        texts[s].push_str(&eng.piece(next[s]));
        if next[s] == eos {
            finished[s] = true;
        }
        gen_toks[s].push(next[s]);
    }
    let gen_res = (|| -> Result<(), String> {
        // 생성 — 단일 루프(generate_q35): spec-multi/spec-single/batch.
        // vl 싱크: 텍스트 누적 + JSONL(token만, infer 형식에서 text 생략).
        // pos는 infer 규칙(base_len + 생성 수, prefill 토큰 pos=base_len과
        // 충돌하지 않게 +1)로 통일 — 구 vl은 prefill pos와 첫 생성 pos가
        // 같은 값이었다(충돌).
        let mut st = crate::engine::GenState {
            finished,
            gen_toks,
            next,
            pos: base_len.iter().map(|&b| b as u32).collect(),
        };
        let mut sink = VlSink {
            texts: &mut texts,
            eos,
        };
        crate::engine::generate_q35(&mut eng, &mut st, n_predict, spec_k, eos, &mut sink)?;
        Ok(())
    })();
    if let Err(e) = gen_res {
        eprintln!("decode: {e}");
    }
    for s in 0..n_img {
        if n_img > 1 {
            println!("seq{s}: {}", texts[s]);
        } else {
            println!("{}", texts[s]);
        }
    }
    use std::io::Write;
    let _ = std::io::stdout().flush();
    eprintln!("# done");
    ExitCode::SUCCESS
}

/// vl 싱크 — 최종 텍스트 누적(eos 조각 제외) + JSONL(token만).
struct VlSink<'a> {
    texts: &'a mut Vec<String>,
    eos: u32,
}
impl crate::engine::TokenSink for VlSink<'_> {
    fn on_token(&mut self, s: usize, pos: u32, t: u32, eng: &llm170_core::qwen35::Engine) {
        if t != self.eos {
            self.texts[s].push_str(&eng.piece(t));
        }
        println!("{{\"seq\":{s},\"pos\":{pos},\"token\":{t}}}");
    }
}
