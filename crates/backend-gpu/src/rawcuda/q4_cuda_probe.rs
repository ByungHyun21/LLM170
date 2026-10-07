//! [검증층 원장 2026-10-04] q4-cuda-dequant/gemv/gemm/neg 프로브 — plans/124 G8.
//! 3층 분리 원칙(§5): 프로브 함수 삽입 금지 — 모듈층(q4_cuda.rs) 오염 없이
//! 이 파일에서만 검증 자산(파서·오라클)을 다룬다.
//!
//! Q4 모듈의 정합 기준은 **core ggml 미러**(EXL3 트렐리스 아님 — 계약 §1.1):
//! 전체 워크스페이스가 Windows에서 빌드 불가(llm170-core mmap 결함 — G1
//! 원장)이므로 오라클은 core를 링크하지 않고 참조 산식을 줄 단위로 베낀다
//! (G2-G7 노선):
//! - crates/core/src/quant/deq.rs — half_to_f32 L13-37 · scale_min_k4 L40-50
//!   · deq_q4_k L53-70(디양자화 비트동일 기준)
//! - crates/core/src/quant/q8.rs — Q8Block·quantize_row_q8_ref L7-34
//!   · dot_q4k_q8 L76-102(블록 내적 순차 f32 기준)
//! - crates/core/src/quant/lane.rs — dot_row_w4a8_q4k_lane_parts L166-191
//!   · tree64 L373-385(64레인 분할 f32 + f64 트리 = GEMV 커널 환원 미러)
//! - crates/core/src/sampler.rs — splitmix64(Rng — exl3_cuda_probe와 동일
//!   미러 방식, 활성 생성)
//!
//! [GGUF 판독 계약] 픽스처는 **실 GGUF**(D:/models/qwen3.8-27b/
//! Qwen3.8-27B-UD-Q4_K_XL.gguf)에서 최소 헤더 파싱(v3 규격 — 매직·버전·
//! KV 스킵·텐서 정보·alignment)으로 텐서 데이터 오프셋을 구해 필요한
//! 행만 파일에서 읽는다 — 파일 전체를 VRAM에 올리지 않는다(12GB 개발
//! VRAM 계약). UD-Q4_K_XL은 GGUF에 표준 Q4_K(ty=12, 144B 블록)로 저장됨을
//! 이 판독으로 재확인(헤더 실측: ty12 텐서 다수, 27B hidden=5120 행폭
//! 2880B = 20블록×144B).
//!
//! [형상 계약] 27B hidden은 config.json(D:/models/Qwen3.8-27B-exl3-4.00bpw,
//! text_config 없는 평면형 — 실측 5120), 35B-A3B hidden·moe_intermediate는
//! D:/models/Qwen3.6-35B-A3B-exl3-4.00bpw/config.json(2048·512) — 계약:
//! 형상은 명시 판독, 추정 금지(결함 1호 정신).
//!
//! [판정 계약 — plans/124 §5: 값 maxdiff, argmax 아님]
//! - (i)   디양자화: **비트동일**(to_bits 전원 일치) — deq_q4_k 미러와.
//! - (ii)  GEMV(t=1): maxdiff ≤1e-5 + bitdiff 인쇄(커널 = 레인 미러 그
//!         자체라 비트동일 기대 — 환원 순서 선택: 64레인 분할 f32 누산
//!         → f64 트리64 = lane.rs dot_row_w4a8_q4k_lane 미러. bit-exact
//!         GEMV 누산이 가능한 구조라 별도 maxdiff 방어 계약 불요).
//! - (iii) GEMM(T=32): maxdiff ≤1e-5 + bitdiff(블록 순차 f32 = dot_q4k_q8
//!         좌폴드 미리 — 역시 비트동일 기대, 환원 순서 선택 사유 동일).
//! - (iv)  음성대조: 슈퍼블록 d 지수 비트 반전(0x0100 xor)이 maxdiff
//!         >1e-4 로 탐지되어야(NEG-DETECTED + 비영 exit — 원장 17호:
//!         검증 계기도 스스로 검증).
//!
//! [실측 원장 2026-10-04, RTX 4070 SUPER(sm_89) — 검증 호스트]
//!   (i)   디양자화 blk.1.attn_gate.weight 8행×5120(160블록): bitdiff=0
//!         maxdiff=0.000e0 — deq_q4_k 미러와 비트동일.
//!   (ii)  GEMV t=1 blk.1.attn_gate.weight 5120×6144(122,880블록):
//!         bitdiff=0 maxdiff=0.000e0 — 레인 미러(lane.rs)와 비트동일
//!         (디바이스 quant 포함 — 정수 1개 어긋나면 maxdiff ≫1e-5).
//!   (iii) GEMM MoE 2048×512 T=32(4,096블록, 실 GGUF 블록 재배치):
//!         bitdiff=0 maxdiff=0.000e0 — dot_q4k_q8 그룹핑 미러와 비트동일.
//!         hip 이격 기록(§6): 원판 평폴드(hip q4_gemm_q4k_m)와 core
//!         그룹핑의 같은 픽스처 maxdiff=5.960e-7 — G8 커널은 core 채택.
//!   (iv)  음성대조 d 지수 반전(0x0100 xor): maxdiff=2.692e-2 > 1e-4 →
//!         NEG-DETECTED(비영 exit).
//!   G8 디버그 원장: gemv_host 인자 순서 결함(out을 ni/no 뒤에 배치 →
//!   ILLEGAL ADDRESS 719) — C 드라이버 API 단독 재현으로 커널·팻빈을
//!   결백 증명하고 Rust 인자열을 수정. 원장화 교훈: 커널 시그니처 순서를
//!   모듈 코드에 주석 계약으로 못박는다(q4_cuda.rs gemv_host).
//!   CMP 170HX(sm_80) 실측은 장비 도착 후(plans/124 §0) — 자원 증거는
//!   커밋 본문 cuobjdump --dump-resource-usage.

use crate::rawcuda::q4_cuda::Q4Cuda;
use std::io::{BufReader, Read, Seek, SeekFrom};

/// 값 maxdiff 판정 임계(GEMV/GEMM — 비트동일 기대, 여유 계약).
const Q4_GEMV_THRESH: f32 = 1e-5;
const Q4_GEMM_THRESH: f32 = 1e-5;
/// 음성대조 탐지 임계(d 지수 반전 → 블록 값 계통 변화).
const Q4_NEG_THRESH: f32 = 1e-4;

const GGUF27: &str = "D:/models/qwen3.8-27b/Qwen3.8-27B-UD-Q4_K_XL.gguf";
const CFG27: &str = "D:/models/Qwen3.8-27B-exl3-4.00bpw/config.json";
const CFG35: &str = "D:/models/Qwen3.6-35B-A3B-exl3-4.00bpw/config.json";

// ── 결정론 RNG(splitmix64 변환 — 결정성이 계약; core sampler.rs 참조) ──
struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed)
    }
    fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// 균일 활성 [-amp, amp).
fn gen_unif(n: usize, seed: u64, amp: f64) -> Vec<f32> {
    let mut r = Rng::new(seed);
    (0..n)
        .map(|_| ((r.next_f64() - 0.5) * 2.0 * amp) as f32)
        .collect()
}

// ── core 미러 오라클 1: 디양자화(deq.rs L13-70 그대로 베낌) ──

/// IEEE 754 binary16 → f32 (deq.rs half_to_f32 L13-37).
fn half_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as u32;
    let exp = ((h >> 10) & 0x1f) as u32;
    let frac = (h & 0x3ff) as u32;
    let bits = match exp {
        0 => {
            if frac == 0 {
                sign << 31
            } else {
                // subnormal
                let mut e = 127 - 15 + 1;
                let mut f = frac;
                while f & 0x400 == 0 {
                    f <<= 1;
                    e -= 1;
                }
                f &= 0x3ff;
                (sign << 31) | (e << 23) | (f << 13)
            }
        }
        0x1f => (sign << 31) | (0xff << 23) | (frac << 13),
        _ => (sign << 31) | ((exp + 112) << 23) | (frac << 13),
    };
    f32::from_bits(bits)
}

#[inline]
fn f16(b: &[u8], off: usize) -> f32 {
    half_to_f32(u16::from_le_bytes([b[off], b[off + 1]]))
}

/// get_scale_min_k4 (deq.rs L40-50 — ggml-quants.c:880).
fn scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        (
            (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4),
            (q[j + 4] >> 4) | ((q[j] >> 6) << 4),
        )
    }
}

/// q4_K 블록: d(2) dmin(2) scales(12) qs(128) (deq.rs deq_q4_k L53-70).
fn deq_q4_k(blk: &[u8], y: &mut [f32]) {
    let d = f16(blk, 0);
    let min = f16(blk, 2);
    let scales = &blk[4..16];
    let qs = &blk[16..144];
    let mut is = 0;
    let mut qi = 0;
    let mut yi = 0;
    for _ in 0..4 {
        let (sc1, m1) = scale_min_k4(is, scales);
        let (sc2, m2) = scale_min_k4(is + 1, scales);
        let (d1, mm1) = (d * sc1 as f32, min * m1 as f32);
        let (d2, mm2) = (d * sc2 as f32, min * m2 as f32);
        for l in 0..32 {
            y[yi + l] = d1 * (qs[qi + l] & 0xF) as f32 - mm1;
            y[yi + 32 + l] = d2 * (qs[qi + l] >> 4) as f32 - mm2;
        }
        qi += 32;
        yi += 64;
        is += 2;
    }
}

// ── core 미러 오라클 2: 활성 q8 양자화 + 정수 내적(q8.rs 그대로 베낌) ──

/// q8_0 블록 (변형): d f32 + qs i8×32 (q8.rs L7-10).
#[derive(Clone, Copy)]
struct Q8Block {
    d: f32,
    qs: [i8; 32],
}

/// 활성 행을 q8 블록으로 양자화 — ggml quantize_row_q8_ref 산술
/// (q8.rs quantize_row_q8_ref L13-34).
fn quantize_row_q8_ref(x: &[f32]) -> Vec<Q8Block> {
    let blocks = x.len().div_ceil(32);
    let mut out = vec![
        Q8Block {
            d: 0.0,
            qs: [0; 32],
        };
        blocks
    ];
    for (b, o) in out.iter_mut().enumerate() {
        let s = &x[b * 32..(b * 32 + 32).min(x.len())];
        let mut amax = 0.0f32;
        for &v in s {
            amax = amax.max(v.abs());
        }
        let d = amax / 127.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        o.d = d;
        for (j, &v) in s.iter().enumerate() {
            o.qs[j] = ((v * id).round()).clamp(-127.0, 127.0) as i8;
        }
    }
    out
}

/// y의 평탄 요소 정수값 (q8.rs y_el L40-43).
#[inline]
fn y_el(y: &[Q8Block], p: usize) -> i64 {
    y[p / 32].qs[p % 32] as i64
}

/// q4_K(256) × q8 — deq_q4_k 순서 (q8.rs dot_q4k_q8 L76-102).
fn dot_q4k_q8(w: &[u8], y: &[Q8Block]) -> f32 {
    let d = f16(w, 0);
    let min = f16(w, 2);
    let sc = &w[4..16];
    let qs = &w[16..144];
    let mut sum = 0.0f32;
    for it in 0..4 {
        let (sc1, m1) = scale_min_k4(2 * it, sc);
        let (sc2, m2) = scale_min_k4(2 * it + 1, sc);
        let (d1, mm1) = (d * sc1 as f32, min * m1 as f32);
        let (d2, mm2) = (d * sc2 as f32, min * m2 as f32);
        let mut isum1 = 0i64;
        let mut isum2 = 0i64;
        for l in 0..32 {
            let q = qs[it * 32 + l];
            isum1 += (q & 0xF) as i64 * y_el(y, it * 64 + l);
            isum2 += (q >> 4) as i64 * y_el(y, it * 64 + 32 + l);
        }
        let qsum1: i64 = (0..32).map(|l| y_el(y, it * 64 + l)).sum();
        let qsum2: i64 = (0..32).map(|l| y_el(y, it * 64 + 32 + l)).sum();
        let (yd1, yd2) = (y[2 * it].d, y[2 * it + 1].d);
        sum += yd1 * (d1 * isum1 as f32 - mm1 * qsum1 as f32);
        sum += yd2 * (d2 * isum2 as f32 - mm2 * qsum2 as f32);
    }
    sum
}

/// 행 단위 순차 f32(lane.rs dot_row_w4a8 L11-27 구조 — GEMM 커널 순서).
fn dot_row_q4k_serial(data: &[u8], k: usize, y: &[Q8Block]) -> f32 {
    let blocks = k / 256;
    let mut acc = 0.0f32;
    for b in 0..blocks {
        acc += dot_q4k_q8(&data[b * 144..b * 144 + 144], &y[b * 8..b * 8 + 8]);
    }
    acc
}

/// hip 판 순서 미러(q4_gemm_q4k_m 원본 — 서브블록 항의 행 acc 평폴드,
/// 항 내부는 동일한 괄호 yd·((d·sc)·isum − (dm·m)·qsum)). CUDA G8 커널은
/// core 그룹핑(블록 국소합)으로 수정했으므로 이 변형은 **hip 이격 측정
/// 전용**(plans/124 §6: 편차는 수치로 기록).
fn dot_row_q4k_hiporder(data: &[u8], k: usize, y: &[Q8Block]) -> f32 {
    let n_sub = k / 32;
    let mut acc = 0.0f32;
    for sb in 0..n_sub {
        let js = sb % 8;
        let (it, half) = (js / 2, js % 2);
        let wb = &data[(sb / 8) * 144..(sb / 8) * 144 + 144];
        let d = f16(wb, 0);
        let dm = f16(wb, 2);
        let (sc, m_) = scale_min_k4(js, &wb[4..16]);
        let (mut isum, mut qsum) = (0i64, 0i64);
        for j in 0..32 {
            let nib = if half == 0 {
                wb[16 + it * 32 + j] & 0xF
            } else {
                wb[16 + it * 32 + j] >> 4
            };
            let yv = y_el(y, sb * 32 + j);
            isum += nib as i64 * yv;
            qsum += yv;
        }
        let yd = y[sb].d;
        acc += yd * ((d * sc as f32) * isum as f32 - (dm * m_ as f32) * qsum as f32);
    }
    acc
}

/// 64레인 환원 — warp 트리 순서 (lane.rs tree64 L373-385 — GEMV 커널 환원).
fn tree64(v: &[f64; 64]) -> f64 {
    let mut a = *v;
    for i in 0..32 {
        a[i] += a[i + 32];
    }
    for &off in &[16usize, 8, 4, 2, 1] {
        for i in 0..off {
            a[i] += a[i + off];
        }
    }
    a[0]
}

/// q4_K 레인 미러(q4k) — q5_K과 동일 분할 형태, qh 없음
/// (lane.rs dot_row_w4a8_q4k_lane_parts L166-191 — GEMV 커널 산술 미러).
fn dot_row_q4k_lane(data: &[u8], k: usize, y: &[Q8Block]) -> f32 {
    let n_sub = k / 32;
    let mut lane = [0.0f64; 64];
    for l in 0..64usize {
        let cnt = (n_sub + 63 - l) / 64;
        let mut acc = 0.0f32;
        for m in 0..cnt {
            let sb = l + m * 64;
            let js = sb % 8;
            let (it, half) = (js / 2, js % 2);
            let wb = &data[(sb / 8) * 144..(sb / 8) * 144 + 144];
            let d = f16(wb, 0);
            let dm = f16(wb, 2);
            let (sc, m_) = scale_min_k4(js, &wb[4..16]);
            let (mut isum, mut qsum) = (0i64, 0i64);
            for j in 0..32 {
                let nib = if half == 0 {
                    wb[16 + it * 32 + j] & 0xF
                } else {
                    wb[16 + it * 32 + j] >> 4
                };
                let yv = y_el(y, sb * 32 + j);
                isum += nib as i64 * yv;
                qsum += yv;
            }
            let yd = y[sb].d;
            acc += yd * (d * sc as f32) * isum as f32;
            acc -= yd * (dm * m_ as f32) * qsum as f32;
        }
        lane[l] = acc as f64;
    }
    tree64(&lane) as f32
}

// ── GGUF v3 최소 헤더 리더(파일 전체 적재 금지 — 오프셋 판독 계약) ──

struct GgufTensor {
    name: String,
    dims: Vec<u64>,
    ty: u32,
    off: u64,
}

/// 스트리밍 판독기 — 위치 추적 BufReader.
struct GgufReader<R: Read> {
    r: R,
    pos: u64,
}

impl<R: Read> GgufReader<R> {
    fn new(r: R) -> Self {
        GgufReader { r, pos: 0 }
    }
    fn take(&mut self, n: usize) -> Result<Vec<u8>, String> {
        let mut b = vec![0u8; n];
        self.r
            .read_exact(&mut b)
            .map_err(|e| format!("gguf 판독(pos={} n={n}): {e}", self.pos))?;
        self.pos += n as u64;
        Ok(b)
    }
    fn u8v(&mut self) -> Result<u8, String> {
        Ok(self.take(1)?[0])
    }
    fn u32v(&mut self) -> Result<u32, String> {
        let b = self.take(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }
    fn u64v(&mut self) -> Result<u64, String> {
        let b = self.take(8)?;
        Ok(u64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }
    fn string(&mut self) -> Result<String, String> {
        let n = self.u64v()? as usize;
        if n > (1 << 20) {
            return Err(format!("gguf 문자열 길이 {n} — 손상 파일 가드"));
        }
        let b = self.take(n)?;
        String::from_utf8(b).map_err(|e| format!("gguf 문자열 utf8: {e}"))
    }
    /// 값 1개 스킵(타입별 크기 — 배열은 원소 재귀).
    fn skip_value(&mut self, ty: u32) -> Result<(), String> {
        const S: [u64; 13] = [1, 1, 2, 2, 4, 4, 4, 1, 0, 0, 8, 8, 8];
        match ty {
            8 => {
                let n = self.u64v()? as usize;
                if n > (1 << 20) {
                    return Err(format!("gguf 문자열 길이 {n} — 손상 파일 가드"));
                }
                self.take(n)?;
            }
            9 => {
                let et = self.u32v()?;
                let cnt = self.u64v()?;
                if S.get(et as usize).copied().unwrap_or(0) > 0 {
                    self.take((S[et as usize] * cnt) as usize)?;
                } else {
                    for _ in 0..cnt {
                        self.skip_value(et)?;
                    }
                }
            }
            t => {
                let s = S.get(t as usize).copied().unwrap_or(0);
                if s == 0 {
                    return Err(format!("gguf 값 타입 {t} — v3 규격 위반"));
                }
                self.take(s as usize)?;
            }
        }
        Ok(())
    }
}

/// GGUF v3 헤더 스캔 → (텐서 정보들, 데이터 베이스 바이트 오프셋).
fn gguf_scan(path: &str) -> Result<(Vec<GgufTensor>, u64), String> {
    let f = std::fs::File::open(path).map_err(|e| format!("{path} 열기: {e}"))?;
    let mut g = GgufReader::new(BufReader::with_capacity(1 << 23, f));
    let magic = g.take(4)?;
    if magic != b"GGUF" {
        return Err("gguf: 매직 불일치 — GGUF 아님".into());
    }
    let ver = g.u32v()?;
    if ver != 3 {
        return Err(format!("gguf: 버전 {ver} — v3 전용 계약"));
    }
    let nten = g.u64v()?;
    let nkv = g.u64v()?;
    if nten > 1_000_000 || nkv > 100_000 {
        return Err(format!("gguf: 카운트 비정상({nten}/{nkv}) — 손상 가드"));
    }
    let mut align: u32 = 0;
    for _ in 0..nkv {
        let key = g.string()?;
        let ty = g.u32v()?;
        if key == "general.alignment" {
            if ty != 4 {
                return Err("gguf: general.alignment 타입 위반".into());
            }
            align = g.u32v()?;
        } else {
            g.skip_value(ty)?;
        }
    }
    let mut tensors = Vec::with_capacity(nten as usize);
    for _ in 0..nten {
        let name = g.string()?;
        let nd = g.u32v()? as usize;
        if nd == 0 || nd > 4 {
            return Err(format!("gguf: 텐서 {name} n_dims={nd} — 가드"));
        }
        let mut dims = Vec::with_capacity(nd);
        for _ in 0..nd {
            dims.push(g.u64v()?);
        }
        let ty = g.u32v()?;
        let off = g.u64v()?;
        tensors.push(GgufTensor {
            name,
            dims,
            ty,
            off,
        });
    }
    let align = if align == 0 { 32 } else { align as u64 };
    let data_base = g.pos.div_ceil(align) * align;
    Ok((tensors, data_base))
}

/// 텐서의 지정 행들만 파일에서 판독(행 우선, Q4_K 행 = ne0/256×144B).
fn gguf_read_q4k_rows(
    path: &str,
    info: &GgufTensor,
    data_base: u64,
    row0: usize,
    nrows: usize,
) -> Result<Vec<u8>, String> {
    let ne0 = info.dims[0] as usize;
    let row_b = ne0 / 256 * 144;
    let rows_total = info.dims.get(1).copied().unwrap_or(1) as usize;
    if row0 + nrows > rows_total {
        return Err(format!(
            "gguf 판독: row0={row0}+{nrows} > 행수 {rows_total} ({})",
            info.name
        ));
    }
    let mut f = std::fs::File::open(path).map_err(|e| format!("{path} 열기: {e}"))?;
    f.seek(SeekFrom::Start(
        data_base + info.off + (row0 * row_b) as u64,
    ))
    .map_err(|e| format!("gguf seek: {e}"))?;
    let mut buf = vec![0u8; nrows * row_b];
    f.read_exact(&mut buf)
        .map_err(|e| format!("gguf 행 판독({nrows}×{row_b}B): {e}"))?;
    Ok(buf)
}

/// GGUF에서 Q4_K(ty=12)·ne[0]=hidden 텐서 선택 — blk.* 선형 우선(파일
/// 순서 첫 번째; token_embd 같은 초대형 행렬은 grid-y 상한·오라클 비용
/// 회피). 없으면 임의 매치로 폴백.
fn gguf_pick_q4k(tensors: &[GgufTensor], hidden: usize) -> Result<GgufTensor, String> {
    let hit = |t: &GgufTensor, blk_only: bool| {
        t.ty == 12
            && t.dims.len() == 2
            && t.dims[0] as usize == hidden
            && (!blk_only || t.name.starts_with("blk."))
            && t.dims[1] <= 65535
    };
    for blk_only in [true, false] {
        for t in tensors {
            if hit(t, blk_only) {
                return Ok(GgufTensor {
                    name: t.name.clone(),
                    dims: t.dims.clone(),
                    ty: t.ty,
                    off: t.off,
                });
            }
        }
    }
    Err(format!(
        "gguf: Q4_K ty=12 ne[0]={hidden} 2차원 텐서 없음 — UD-Q4_K_XL 배치 가정 위반"
    ))
}

/// config.json 평면 숫자 필드 판독(최소 파서 — 첫 등장 값; 이 파일군은
/// hidden_size·moe_intermediate_size가 평면/단일 등장 — 실측 2026-10-04).
fn cfg_num(cfg: &str, key: &str) -> Result<usize, String> {
    let pat = format!("\"{key}\"");
    let Some(p) = cfg.find(&pat) else {
        return Err(format!("config.json: {key} 없음"));
    };
    let rest = &cfg[p + pat.len()..];
    let Some(c) = rest.find(':') else {
        return Err(format!("config.json: {key} 콜론 없음"));
    };
    let s = rest[c + 1..].trim_start();
    let end = s.find(|ch: char| !ch.is_ascii_digit()).unwrap_or(s.len());
    if end == 0 {
        return Err(format!("config.json: {key} 숫자 아님"));
    }
    s[..end]
        .parse::<usize>()
        .map_err(|e| format!("config.json: {key} 파싱: {e}"))
}

/// 27B hidden(config.json 판독 — 형상 명시 계약).
fn hidden27() -> Result<usize, String> {
    let cfg = std::fs::read_to_string(CFG27).map_err(|e| format!("{CFG27}: {e}"))?;
    cfg_num(&cfg, "hidden_size")
}

/// 35B-A3B MoE 형상(hidden·moe_intermediate — config.json 판독).
fn moe35() -> Result<(usize, usize), String> {
    let cfg = std::fs::read_to_string(CFG35).map_err(|e| format!("{CFG35}: {e}"))?;
    let h = cfg_num(&cfg, "hidden_size")?;
    let m = cfg_num(&cfg, "moe_intermediate_size")?;
    Ok((h, m))
}

/// (got, want) → (maxdiff, 비트 상이 수). NaN은 maxdiff에 반영(불일치 처리).
fn cmp_f32(got: &[f32], want: &[f32]) -> (f32, usize) {
    let mut md = 0.0f32;
    let mut bd = 0usize;
    for (a, b) in got.iter().zip(want.iter()) {
        let d = (a - b).abs();
        let d = if d.is_nan() { f32::INFINITY } else { d };
        if d > md {
            md = d;
        }
        if a.to_bits() != b.to_bits() {
            bd += 1;
        }
    }
    (md, bd)
}

/// q4-cuda-dequant — (i) 실 GGUF 블록 디양자화 **비트동일** 판정.
/// 8행 × hidden(=Q4_K 열폭) — 27B hidden=5120 → 160 슈퍼블록.
pub fn cuda_q4_dequant_check() -> Result<String, String> {
    let hidden = hidden27()?;
    let (tensors, data_base) = gguf_scan(GGUF27)?;
    let t = gguf_pick_q4k(&tensors, hidden)?;
    let rows = 8usize;
    let bytes = gguf_read_q4k_rows(GGUF27, &t, data_base, 0, rows)?;
    let mut m = Q4Cuda::new()?;
    let dev = m.device_name().to_string();
    m.add_q4k_bytes("t", &bytes, hidden, rows)?;
    let got = m.dequant_host("t", rows)?;
    // 오라클: deq_q4_k 미러(deq.rs L53-70) — 블록 순회 순서 동일.
    let mut want = vec![0.0f32; rows * hidden];
    let blocks = hidden / 256;
    for r in 0..rows {
        for b in 0..blocks {
            deq_q4_k(
                &bytes[(r * blocks + b) * 144..][..144],
                &mut want[r * hidden + b * 256..][..256],
            );
        }
    }
    let (md, bd) = cmp_f32(&got, &want);
    let pass = bd == 0;
    println!(
        "device: {dev} | q4-cuda-dequant (i) {} rows={rows} n={hidden} blocks={} total={}: bitdiff={bd} maxdiff={md:.3e} | {}",
        t.name,
        rows * blocks,
        rows * hidden,
        if pass { "PASS" } else { "FAIL" }
    );
    if pass {
        Ok(format!(
            "device: {dev} | q4-cuda-dequant {}/{} bitdiff=0 maxdiff={md:.3e} | ALL PASS",
            t.name,
            rows * blocks
        ))
    } else {
        Err(format!(
            "q4-cuda-dequant 실패 — bitdiff={bd} maxdiff={md:.3e} (device: {dev})"
        ))
    }
}

/// q4-cuda-gemv — (ii) MMQ GEMV(t=1) 27B 형상(hidden=config.json 판독).
/// 실 텐서 전 행(n_out) vs 레인 미러(lane.rs dot_row_w4a8_q4k_lane).
pub fn cuda_q4_gemv_check() -> Result<String, String> {
    let hidden = hidden27()?;
    let (tensors, data_base) = gguf_scan(GGUF27)?;
    let t = gguf_pick_q4k(&tensors, hidden)?;
    let n_out = t.dims[1] as usize;
    let bytes = gguf_read_q4k_rows(GGUF27, &t, data_base, 0, n_out)?;
    let mut m = Q4Cuda::new()?;
    let dev = m.device_name().to_string();
    m.add_q4k_bytes("t", &bytes, hidden, n_out)?;
    let x = gen_unif(hidden, 0x5EED_0000_0000_8001, 1.0);
    let got = m.gemv_host("t", &x)?;
    // 오라클: quantize_row_q8_ref + 64레인 분할 f32 + f64 tree64(lane.rs 미러).
    let y = quantize_row_q8_ref(&x);
    let blocks = hidden / 256;
    let mut want = Vec::with_capacity(n_out);
    for o in 0..n_out {
        want.push(dot_row_q4k_lane(
            &bytes[o * blocks * 144..][..blocks * 144],
            hidden,
            &y,
        ));
    }
    let (md, bd) = cmp_f32(&got, &want);
    let pass = md <= Q4_GEMV_THRESH && got.iter().all(|v| v.is_finite());
    println!(
        "device: {dev} | q4-cuda-gemv (ii) {} n_in={hidden} n_out={n_out} blocks={} T=1: bitdiff={bd} maxdiff={md:.3e} | {}",
        t.name,
        n_out * blocks,
        if pass { "PASS" } else { "FAIL" }
    );
    if pass {
        Ok(format!(
            "device: {dev} | q4-cuda-gemv n_in={hidden} n_out={n_out} maxdiff={md:.3e} bitdiff={bd} | ALL PASS"
        ))
    } else {
        Err(format!(
            "q4-cuda-gemv 실패 — maxdiff={md:.3e} > {Q4_GEMV_THRESH:.0e} bitdiff={bd} (device: {dev})"
        ))
    }
}

/// q4-cuda-gemm — (iii) MMQ GEMM 35B-A3B MoE 형상(hidden×moe_intermediate,
/// config.json 판독) T=32. 가중 블록은 실 GGUF Q4_K 블록 열(형상만 MoE
/// 전문가 폭으로 재배치 — 블록은 자립적이라 값 무관·실데이터 유지).
pub fn cuda_q4_gemm_check() -> Result<String, String> {
    let (n_in, n_out) = moe35()?;
    if n_in % 256 != 0 {
        return Err(format!("q4-cuda-gemm: hidden={n_in} — 256배수 계약 위반"));
    }
    let (tensors, data_base) = gguf_scan(GGUF27)?;
    let hidden27b = hidden27()?;
    let ts = gguf_pick_q4k(&tensors, hidden27b)?;
    let tt = 32usize;
    let gemm_blocks = n_out * (n_in / 256);
    // 필요 블록 수만큼 실 텐서 행을 읽어 잘라낸다(행폭 2880B=20블록).
    let src_row_blocks = hidden27b / 256;
    let need_rows = gemm_blocks.div_ceil(src_row_blocks);
    let raw = gguf_read_q4k_rows(GGUF27, &ts, data_base, 0, need_rows)?;
    let bytes: Vec<u8> = raw[..gemm_blocks * 144].to_vec();
    let mut m = Q4Cuda::new()?;
    let dev = m.device_name().to_string();
    m.add_q4k_bytes("t", &bytes, n_in, n_out)?;
    let xs = gen_unif(tt * n_in, 0x5EED_0000_0000_8003, 1.0);
    let got = m.gemm_host("t", &xs)?;
    // 오라클: 행별 quantize_row_q8_ref → 블록 순차 f32(dot_q4k_q8 좌폴드).
    let mut want = vec![0.0f32; tt * n_out];
    for r in 0..tt {
        let y = quantize_row_q8_ref(&xs[r * n_in..(r + 1) * n_in]);
        for o in 0..n_out {
            want[r * n_out + o] = dot_row_q4k_serial(
                &bytes[o * (n_in / 256) * 144..][..(n_in / 256) * 144],
                n_in,
                &y,
            );
        }
    }
    let (md, bd) = cmp_f32(&got, &want);
    // hip 이격 기록(plans/124 §6 — 수치 원장): 원판 hip 평폴드 순서와 core
    // 그룹핑 미러의 같은 픽스처에서의 편차.
    let mut hip_gap = 0.0f32;
    for r in 0..tt {
        let y = quantize_row_q8_ref(&xs[r * n_in..(r + 1) * n_in]);
        for o in 0..n_out {
            let a = dot_row_q4k_hiporder(
                &bytes[o * (n_in / 256) * 144..][..(n_in / 256) * 144],
                n_in,
                &y,
            );
            let b = dot_row_q4k_serial(
                &bytes[o * (n_in / 256) * 144..][..(n_in / 256) * 144],
                n_in,
                &y,
            );
            let d = (a - b).abs();
            if d > hip_gap {
                hip_gap = d;
            }
        }
    }
    let pass = md <= Q4_GEMM_THRESH && got.iter().all(|v| v.is_finite());
    println!(
        "device: {dev} | q4-cuda-gemm (iii) MoE n_in={n_in} n_out={n_out} blocks={} T={tt}: bitdiff={bd} maxdiff={md:.3e} | {}",
        n_out * (n_in / 256),
        if pass { "PASS" } else { "FAIL" }
    );
    println!(
        "device: {dev} | q4-cuda-gemm hip-parity: 원판 평폴드(hip q4_gemm_q4k_m) vs core 그룹핑 maxdiff={hip_gap:.3e} (그룹핑 편차 — G8 커널은 core 채택, §6 원장 기록)"
    );
    if pass {
        Ok(format!(
            "device: {dev} | q4-cuda-gemm n_in={n_in} n_out={n_out} T={tt} maxdiff={md:.3e} bitdiff={bd} | ALL PASS"
        ))
    } else {
        Err(format!(
            "q4-cuda-gemm 실패 — maxdiff={md:.3e} > {Q4_GEMM_THRESH:.0e} bitdiff={bd} (device: {dev})"
        ))
    }
}

/// q4-cuda-neg — (iv) 음성대조(원장 17호): 슈퍼블록 d 지수 비트 반전
/// (f16 상위 바이트 xor 0x01)이 디양자화에서 탐지되어야 한다.
/// 정상 동작 시 NEG-DETECTED 마커와 함께 Err(→ CLI 비영 exit).
pub fn cuda_q4_negative_check() -> Result<String, String> {
    let hidden = hidden27()?;
    let (tensors, data_base) = gguf_scan(GGUF27)?;
    let t = gguf_pick_q4k(&tensors, hidden)?;
    let rows = 8usize;
    let mut bytes = gguf_read_q4k_rows(GGUF27, &t, data_base, 0, rows)?;
    // 오라클(원본 기준) — 오염 전에 계산.
    let blocks = hidden / 256;
    let mut want = vec![0.0f32; rows * hidden];
    for r in 0..rows {
        for b in 0..blocks {
            deq_q4_k(
                &bytes[(r * blocks + b) * 144..][..144],
                &mut want[r * hidden + b * 256..][..256],
            );
        }
    }
    // 오염: 행 0·슈퍼블록 0의 d f16 지수 최하위 비트 반전(0x0100 xor —
    // 유한값 유지·값은 계통 변화. d 비트가 극단(0/subnormal)이면 탐지
    // 민감도가 낮아지는 경계는 원장에 기록).
    bytes[1] ^= 0x01;
    let mut m = Q4Cuda::new()?;
    let dev = m.device_name().to_string();
    m.add_q4k_bytes("t", &bytes, hidden, rows)?;
    let got = m.dequant_host("t", rows)?;
    let (md, _) = cmp_f32(&got, &want);
    println!(
        "device: {dev} | q4-cuda-neg (iv) corrupted superblock d (bit flip 0x0100): maxdiff={md:.3e} | FAIL(expected)"
    );
    if md > Q4_NEG_THRESH {
        Err(format!(
            "NEG-DETECTED maxdiff={md:.3e} > {Q4_NEG_THRESH:.0e} — 검증계기 정상(스케일 오염 감지)"
        ))
    } else {
        Err(format!(
            "NEG-MISSED maxdiff={md:.3e} <= {Q4_NEG_THRESH:.0e} — 검증계기 결함: d 비트 반전이 탐지되지 않음"
        ))
    }
}
