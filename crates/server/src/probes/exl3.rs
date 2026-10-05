//! EXL3 프로브(vk checks·hip 프로브·to-gguf 변환기) (plans/129 R2③ — probes/ 분리, arm 본문 무변경 이동).
use std::process::ExitCode;

use super::{arg_num, arg_str};

pub(super) fn try_run(cmd: &str, args: &[String]) -> Option<ExitCode> {
    let r: Result<String, String> = match cmd {
        "exl3-check" => {
            let d_exl3 = "/home/yoon/models/Qwen3.8-27B-exl3-4.00bpw";
            let d_q8 = "/home/yoon/models/qwen3.8-27b/Qwen3.8-27B-UD-Q8_K_XL.gguf";
            cmd_exl3_check(&arg_str(args, 0, d_exl3), &arg_str(args, 1, d_q8))
        }
        "exl3-to-gguf" => {
            let d_exl3 = "/home/yoon/models/Qwen3.8-27B-exl3-4.00bpw";
            let out = arg_str(
                args,
                1,
                "/home/yoon/models/Qwen3.8-27B-exl3-4.00bpw/exl3-f16.gguf",
            );
            // args[2] = 층 상한(스모크용 절단) — 0/생략 = 전체.
            let maxl = arg_num::<usize>(args, 2, 0);
            let maxl = if maxl == 0 { None } else { Some(maxl) };
            match llm170_exl3::convert::exl3_to_gguf(
                std::path::Path::new(&arg_str(args, 0, d_exl3)),
                std::path::Path::new(&out),
                maxl,
            ) {
                Ok(s) => Ok(format!(
                    "exl3-to-gguf: {}개 텐서 {:.1} GB — {:.0}초 → {out}",
                    s.tensors,
                    s.bytes as f64 / 1e9,
                    s.elapsed_s
                )),
                Err(e) => Err(format!("{e}")),
            }
        }
        "exl3-load" => {
            let d_exl3 = "/home/yoon/models/Qwen3.8-27B-exl3-4.00bpw";
            let path = arg_str(args, 0, d_exl3);
            match llm170_exl3::Exl3Model::open(std::path::Path::new(&path)) {
                Ok(m) => {
                    let tre_mb: f64 = m
                        .linears
                        .iter()
                        .map(|l| l.svh.len + l.suh.len + l.tre.len)
                        .sum::<u64>() as f64
                        / 1e6;
                    let plain_mb: f64 =
                        m.plains.iter().map(|p| p.slab.len).sum::<u64>() as f64 / 1e6;
                    Ok(format!(
                        "exl3-load {path}\n  layers={} hidden={} ffn={} vocab={} interval={}\n  선형 {}개 ({tre_mb:.0} MB) · 무양자화 {}개 ({plain_mb:.0} MB) — 완전성 검증 통과",
                        m.cfg.num_hidden_layers,
                        m.cfg.hidden_size,
                        m.cfg.intermediate_size,
                        m.cfg.vocab_size,
                        m.cfg.full_attention_interval,
                        m.linears.len(),
                        m.plains.len(),
                    ))
                }
                Err(e) => Err(format!("{e}")),
            }
        }
        "exl3-decode" => {
            let d_exl3 = "/home/yoon/models/Qwen3.8-27B-exl3-4.00bpw";
            llm170_backend_gpu::rawvk::checks::exl3_decode(
                &arg_str(args, 0, d_exl3),
                &arg_str(args, 1, "9707"),
                arg_num(args, 2, 8usize),
            )
        }
        "exl3-pp" => {
            let d_exl3 = "/home/yoon/models/Qwen3.8-27B-exl3-4.00bpw";
            llm170_backend_gpu::rawvk::checks::exl3_pp(
                &arg_str(args, 0, d_exl3),
                &arg_str(args, 1, "9707"),
                arg_num(args, 2, 8usize),
            )
        }
        "exl3-scan-check" => llm170_backend_gpu::rawvk::checks::scan_check(
            arg_num(args, 0, 512usize),
            &arg_str(args, 1, ""),
        ),
        "exl3-attn-check" => llm170_backend_gpu::rawvk::checks::attn_check(
            arg_num(args, 0, 512usize),
            arg_num(args, 1, 0usize),
        ),
        "exl3-nr-check" => llm170_backend_gpu::rawvk::checks::nr_check(),
        "exl3-hip-gmini2" => llm170_backend_gpu::rawhip::exl3_hip_probe::hip_graph_mini2(),
        "exl3-hip-gmini" => llm170_backend_gpu::rawhip::exl3_hip_probe::hip_graph_mini(),
        "exl3-hip-graph" => {
            let dir = arg_str(args, 0, "");
            let tok = arg_num(args, 1, 1000u32);
            let tl = arg_num(args, 2, 4usize);
            llm170_backend_gpu::rawhip::exl3_hip_probe::hip_graph_check(&dir, tok, tl)
        }
        "exl3-hip-tbench" => {
            let dir = arg_str(args, 0, "");
            let tok = arg_num(args, 1, 1000u32);
            let tmax = arg_num(args, 2, 16usize);
            llm170_backend_gpu::rawhip::exl3_hip_probe::hip_tbench(&dir, tok, tmax)
        }
        "exl3-hip-hcmp" => {
            let dir = arg_str(args, 0, "");
            let tok = arg_num(args, 1, 1000u32);
            let steps = arg_num(args, 2, 6usize);
            llm170_backend_gpu::rawhip::exl3_hip_probe::hip_h_pair(&dir, tok, steps)
        }
        "exl3-hip-a1" => {
            let dir = arg_str(args, 0, "");
            let tok = arg_num(args, 1, 1000u32);
            let steps = arg_num(args, 2, 24usize);
            llm170_backend_gpu::rawhip::exl3_hip_probe::hip_mtp_a1(&dir, tok, steps)
        }
        "exl3-hip-mtp-round" => {
            let dir = arg_str(args, 0, "");
            let tok = arg_num(args, 1, 1000u32);
            let rounds = arg_num(args, 2, 8usize);
            llm170_backend_gpu::rawhip::exl3_hip_probe::hip_mtp_round(&dir, tok, rounds)
        }
        "exl3-hip-batch" => llm170_backend_gpu::rawhip::exl3_hip_probe::hip_batch_check(
            &arg_str(args, 0, ""),
            arg_num(args, 1, 1000u32),
            arg_num(args, 2, 4usize),
        ),
        "exl3-hip-mtp" => llm170_backend_gpu::rawhip::exl3_hip_probe::hip_mtp_check(
            &arg_str(args, 0, ""),
            arg_num(args, 1, 1000u32),
        ),
        "exl3-hip-decode" => {
            let dir = arg_str(args, 0, "/home/yoon/models/Qwen3.8-27B-exl3-4.00bpw");
            let tok = arg_str(args, 1, "1000").parse::<u32>().unwrap_or(1000);
            let lim = arg_str(args, 2, "64").parse::<usize>().unwrap_or(64);
            llm170_backend_gpu::rawhip::exl3_hip_probe::hip_decode_check(&dir, tok, lim)
        }
        "exl3-hip-attn" => {
            let dir = arg_str(args, 0, "/home/yoon/models/Qwen3.8-27B-exl3-4.00bpw");
            // t/pos0/layer 선택 인자(plans/128 P0·P1) — pos0+t>256이면
            // fwd3s 온라인 소프트맥스의 다중 청크·층 스트라이드를 검증한다.
            let t = arg_str(args, 1, "8").parse::<usize>().unwrap_or(8);
            let pos0 = arg_str(args, 2, "0").parse::<usize>().unwrap_or(0);
            let lay = arg_str(args, 3, "0").parse::<usize>().unwrap_or(0);
            llm170_backend_gpu::rawhip::exl3_hip_probe::hip_attn_check(&dir, t, pos0, lay)
        }
        "exl3-hip-gdn" => {
            let dir = arg_str(args, 0, "/home/yoon/models/Qwen3.8-27B-exl3-4.00bpw");
            let lay = arg_str(args, 1, "0").parse::<usize>().unwrap_or(0);
            llm170_backend_gpu::rawhip::exl3_hip_probe::hip_gdn_check(&dir, lay)
        }
        "exl3-hip-gemm" => {
            let dir = arg_str(args, 0, "/home/yoon/models/Qwen3.8-27B-exl3-4.00bpw");
            // key 단축명(g,u,d,qkv,z,gop,q,k,v,o,lh,g5 — plans/128 P2 형상 스윕) · t
            let key = arg_str(args, 1, "g");
            let t = arg_str(args, 2, "64").parse::<usize>().unwrap_or(64);
            llm170_backend_gpu::rawhip::exl3_hip_probe::hip_gemm_check(&dir, &key, t)
        }
        "exl3-hip-linear" => {
            let dir = arg_str(args, 0, "/home/yoon/models/Qwen3.8-27B-exl3-4.00bpw");
            let key = arg_str(args, 1, "model.language_model.layers.0.mlp.gate_proj");
            llm170_backend_gpu::rawhip::exl3_hip_probe::hip_linear_check(&dir, &key)
        }
        "exl3-hip-nr" => {
            let dir = arg_str(args, 0, "/home/yoon/models/Qwen3.8-27B-exl3-4.00bpw");
            llm170_backend_gpu::rawhip::exl3_hip_probe::hip_nr_check(&dir)
        }
        "exl3-hip-gemv" => {
            let dir = arg_str(args, 0, "/home/yoon/models/Qwen3.8-27B-exl3-4.00bpw");
            llm170_backend_gpu::rawhip::exl3_hip_probe::hip_gemv_check(&dir)
        }
        "exl3-gemmd-check" => {
            let dir = arg_str(args, 0, "/home/yoon/models/Qwen3.8-27B-exl3-4.00bpw");
            let t = arg_str(args, 1, "8").parse::<usize>().unwrap_or(8);
            let il = arg_str(args, 2, "0").parse::<usize>().unwrap_or(0);
            llm170_backend_gpu::rawvk::checks::TrellisResident::gemmd_check(&dir, t, il)
        }
        "exl3-nrh-check" => llm170_backend_gpu::rawvk::checks::TrellisResident::nrh_check(
            &arg_str(args, 0, "/home/yoon/models/Qwen3.8-27B-exl3-4.00bpw"),
        ),
        "exl3-chain-check" => llm170_backend_gpu::rawvk::checks::chain_check(&arg_str(
            args,
            0,
            "/home/yoon/models/Qwen3.8-27B-exl3-4.00bpw",
        )),
        "exl3-ffn-check" => llm170_backend_gpu::rawvk::checks::ffn_check(&arg_str(
            args,
            0,
            "/home/yoon/models/Qwen3.8-27B-exl3-4.00bpw",
        )),
        "exl3-gemm-check" => llm170_backend_gpu::rawvk::checks::gemm_check(&arg_str(
            args,
            0,
            "/home/yoon/models/Qwen3.8-27B-exl3-4.00bpw",
        )),
        "exl3-mtp" => {
            let d_exl3 = "/home/yoon/models/Qwen3.8-27B-exl3-4.00bpw";
            llm170_backend_gpu::rawvk::checks::exl3_mtp(
                &arg_str(args, 0, d_exl3),
                &arg_str(args, 1, "9707"),
                arg_num(args, 2, 32usize),
            )
        }
        "exl3-mtp2" => {
            let d_exl3 = "/home/yoon/models/Qwen3.8-27B-exl3-4.00bpw";
            llm170_backend_gpu::rawvk::checks::exl3_mtp2(
                &arg_str(args, 0, d_exl3),
                &arg_str(args, 1, "9707"),
                arg_num(args, 2, 48usize),
                arg_num(args, 3, 2usize),
            )
        }
        "exl3-bench" => {
            let d_exl3 = "/home/yoon/models/Qwen3.8-27B-exl3-4.00bpw";
            llm170_backend_gpu::rawvk::checks::exl3_bench(
                &arg_str(args, 0, d_exl3),
                arg_num(args, 1, 5usize),
            )
        }
        "exl3-vk-check" => {
            let d_exl3 = "/home/yoon/models/Qwen3.8-27B-exl3-4.00bpw";
            llm170_backend_gpu::rawvk::checks::exl3_vk_check(
                &arg_str(args, 0, d_exl3),
                &arg_str(args, 1, "model.language_model.layers.0.mlp.gate_proj"),
            )
        }
        _ => return None,
    };
    Some(super::finish(r))
}


/// llm170 exl3-check <exl3_dir> <q8.gguf> — EXL3 참조 디코드 ↔ GGUF Q8 대조
/// (plans/118 §3-1). Python 검증기(scripts/exl3_validate.py)의 내부화:
/// K=3/4/5 혼재 텐서의 128×128 블록 상관계수. 기준: corr ≥ 0.97(K=3 양자화
/// 오차 수준) — 그 미만이면 디코드 회귀.
fn cmd_exl3_check(exl3_dir: &str, gguf_path: &str) -> Result<String, String> {
    let ar =
        llm170_exl3::StArchive::open(std::path::Path::new(exl3_dir)).map_err(|e| e.to_string())?;
    let g =
        llm170_gguf::GgufFile::open(std::path::Path::new(gguf_path)).map_err(|e| e.to_string())?;
    let cases: &[(&str, &str)] = &[
        (
            "model.language_model.layers.0.mlp.gate_proj",
            "blk.0.ffn_gate.weight",
        ),
        (
            "model.language_model.layers.0.mlp.down_proj",
            "blk.0.ffn_down.weight",
        ),
        (
            "model.language_model.layers.3.self_attn.o_proj",
            "blk.3.attn_output.weight",
        ),
        (
            "model.language_model.layers.0.linear_attn.in_proj_qkv",
            "blk.0.attn_qkv.weight",
        ),
    ];
    use std::io::{Read, Seek, SeekFrom};
    let mut report = String::new();
    let mut all_ok = true;
    for (key, gname) in cases {
        let (corr, krate) = (|| -> Result<(f64, u32), String> {
            let w = llm170_exl3::Exl3Linear::load(&ar, key).map_err(|e| e.to_string())?;
            let ti = g
                .find_tensor(gname)
                .ok_or_else(|| format!("gguf tensor not found: {gname}"))?;
            if ti.ty != llm170_gguf::GgmlType::Q8_0 {
                return Err(format!("{gname}: Q8_0 아님({:?})", ti.ty));
            }
            let (start, end) = ti
                .file_range(g.data_offset)
                .ok_or_else(|| format!("{gname}: 범위 계산 불가"))?;
            // 참조는 처음 128행(출력)만 — 텐서 전체 미로딩.
            let row_bytes = (ti.ne[0] as usize / 32) * 34;
            let nread = ((end - start) as usize).min(128 * row_bytes);
            let mut raw = vec![0u8; nread];
            let mut f = std::fs::File::open(&g.path).map_err(|e| e.to_string())?;
            f.seek(SeekFrom::Start(start)).map_err(|e| e.to_string())?;
            f.read_exact(&mut raw).map_err(|e| e.to_string())?;
            let k = ti.ne[0] as usize;
            let mut wref = vec![0f64; 128 * 128]; // [k][n]
            let mut row = vec![0f32; k];
            for n in 0..128usize {
                llm170_core::quant::dequant_row(ti.ty, &raw, n as u64, k as u64, &mut row);
                for (i, v) in row.iter().take(128).enumerate() {
                    wref[i * 128 + n] = *v as f64;
                }
            }
            let wex = w.dequant_block_f64(0, 0, 128, 128);
            let (ma, mb) = (
                wex.iter().sum::<f64>() / 16384.0,
                wref.iter().sum::<f64>() / 16384.0,
            );
            let (mut sab, mut saa, mut sbb) = (0f64, 0f64, 0f64);
            for i in 0..16384 {
                let (a, b) = (wex[i] - ma, wref[i] - mb);
                sab += a * b;
                saa += a * a;
                sbb += b * b;
            }
            Ok((sab / (saa * sbb).sqrt(), w.krate))
        })()
        .inspect_err(|_| all_ok = false)?;
        let ok = corr >= 0.97;
        all_ok &= ok;
        let short: String = key.split('.').rev().take(2).collect::<Vec<_>>().join(".");
        report.push_str(&format!(
            "{short:44} K={krate} corr={corr:.5} {}\n",
            if ok { "ok" } else { "FAIL" }
        ));
    }
    if all_ok {
        report.push_str("exl3-check: 전 텐서 통과 (기준 corr ≥ 0.97)");
        Ok(report)
    } else {
        Err(report)
    }
}
// 마커 mtpg

