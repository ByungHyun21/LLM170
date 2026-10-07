//! [검증층 공유 지원 파일 — plans/124 G10 파일 분할 2026-10-04]
//! 스모크 체인(G1) + 전 모듈 프루브 공용 자산(결정론 RNG·f16 비트
//! 변환·트렐리스 참조·WHT·maxdiff 판정·safetensors 판독·균일 생성기).
//! 모듈별 프로브는 분리 파일: gemv_cuda_probe·norm_cuda_probe·
//! gemm2_cuda_probe·gdn_cuda_probe·attn_cuda_probe·ew_argmax_cuda_probe
//! (q4_cuda_probe·mtp_cuda_probe는 G8·G9 독립 파일). 3층 분리 원칙
//! (plans/124 §5): 프로브 함수 삽입 금지 — 모듈층 오염 없이 검증층
//! 파일에서만 검증 자산을 다룬다. 아래 [검증층 원장]은 G2-G4 시대의
//! 역사 원장으로 그대로 보존한다(모듈별 요약은 각 프루브 파일 머리).
//!
//! [커널 파일→프루브 매핑 표 — plans/129-cuda-only C5(129 A22 유비) 기록]
//! | 커널 자산           | 프루브 파일          | 서브커맨드                                  |
//! |--------------------|----------------------|---------------------------------------------|
//! | assets/smoke.cu    | exl3_cuda_probe(본파일)| smoke                                      |
//! | assets/exl3_gemv.cu| gemv_cuda_probe      | gemv·gemv-neg·gemv-debug·gemv-real          |
//! | assets/exl3_norm.cu| norm_cuda_probe      | norm·norm-neg                               |
//! | assets/exl3_gemm2.cu| gemm2_cuda_probe    | gemm2·gemm2-neg·gemm2-debug                 |
//! | assets/exl3_gdn.cu | gdn_cuda_probe       | gdn·gdn-neg                                 |
//! | assets/exl3_attn.cu| attn_cuda_probe      | attn·attn-neg                               |
//! | assets/exl3_ew.cu  | ew_argmax_cuda_probe | ew·argmax·argmax-neg                        |
//! | assets/exl3_q4.cu  | q4_cuda_probe        | q4-dequant·q4-gemv·q4-gemm·q4-neg           |
//! | assets/exl3_mtp.cu | mtp_cuda_probe       | mtp·mtp-neg                                 |
//! (모듈 파일 구조상 1:1 대응 — 검증기 없는 커널 없음. Q4는 비트동일
//! 직접 검증기가 존재 — 원본 A22가 지적한 vk Q4 무직접검증과 대비.)
//!
//!
//! 스모크 체인(G1): fatbin 로드 → llm170_smoke_add 발사 → 비트동일 값 검증.
//! 실패(자산 부재 포함)는 전부 Err → CLI exit 비영.
//!
//! GEMV 체인 프로브(G2): 합성 트레일리스 선형(결정론 시드 — splitmix64
//! 미러)으로 had_in→gemv→had_out 체인을 실행, 이 파일에 내장한 f32 오라클과
//! 값 maxdiff 판정(임계 3e-4 — argmax 판정 금지, plans/124 §5). 전체
//! 워크스페이스가 Windows에서 빌드 불가(llm170-core mmap 결함 — G1 원장)이므로
//! 오라클은 core를 링크하지 않고 참조 산식을 그대로 베낀다(아래 인용).
//!
//! 노름 프로브(G3): norm_resid 커널(x+ab 제자리 가산 + rms_norm·nw[w행])을
//! 결정론 합성 입력으로 실행, crates/core/src/ops.rs(sq_sum L11-31 ·
//! rms_norm L33-37)을 줄 단위로 베낀 f32 오라클과 값 maxdiff 판정(임계
//! 3e-6 — plans/124 §1·§3.2). 결함 2호(w 행 오프셋 누락 → 전 노름 L0
//! 판독)은 행 강분리 nw + w=0 음성대조로, eps 계약은 1e-5 오라클 주입
//! 대조로 각각 감지 가능함을 증명한다(원장 17호 — 계기 자체 검증).
//!
//! 배치 GEMM 프로브(G4): gemm2/kseg 체인(had_in T행 → mma m16n8k16
//! gemm2 → had_out 1회)을 결정론 시드 합성으로 실행, 아래 내장 f64/f32
//! 미러 오라클과 값 maxdiff 판정(임계 4e-4 — plans/124 §1). 경로 양측
//! 커버: T=32 plain · T≤8 kseg(결함 18호 — grid (n/64, kseg=8)).
//! gemm2 오라클 산술(인용 — 커널 산술 계약의 f32 미러):
//! - had_in 행별 f16 비트(비트일치 — 아래 원장·결함 20호 수정 후 전 행 0).
//! - gemm2: f16×f16 곱은 f32에서 정확(가수 11+11<24) — f64 누산(코어
//!   참조 계급) → s f32. 커널 mma f32 누산과의 차이는 누산 순서 차이의
//!   예상 계급(~1e-6 실측 2.2e-6 — 임계의 1/180).
//! - had_out: nseg 순서 합산 f32 → WHT-128 f32 → ×R·svh(원본 산술).
//!
//! 오라클 근거 소스(전부 워크트리 기준 줄번호, 2026-10-04):
//! - crates/core/src/ops.rs — sq_sum L11-31 · rms_norm L33-37(G3 노름 기준).
//! - crates/exl3/src/trellis.rs — PERM_INV L24-44 · mul1_decode L47-52 ·
//!   tile_word L60-70 · decode_tile L85-91(트렐리스 비트 산식 원본).
//! - crates/backend-gpu/src/rawvk/checks/exl3.rs — had128_f32 L11-25 ·
//!   CPU 참조 체인(had_in/gemv/had_out 미러) L54-105.
//! - crates/core/src/sampler.rs — Rng::next_u64 L21-28(splitmix64).
//! - crates/backend-gpu/src/rawhip/kernels/src_exl3.hip — exl3_norm_resid
//!   (노름 커널 산술 원본 — assets/exl3_norm.cu가 1:1 직이식).
//!
//! 오라클 산술 계약(G2 정합 기준 = 코어 f32 미러, 값 maxdiff ≤3e-4):
//! - gemv 누산은 코어 미러 순서 그대로: prod=f16(a·w) → acc=f16(acc+prod)
//!   (f32 곱/합 각 1회). 커널 __hfma2의 단일 반올림과의 반올림 지점 차이는
//!   연산당 ≤1 ulp의 예상 계급 — hip도 같은 계급으로 2.664e-4 통과(원장).
//! - mul1은 trellis.rs 원식(f32 fma) — 이 호스트에 +fma 특성이 없어
//!   f32::mul_add가 소프트웨어 이중 반올림으로 떨어지므로, 참조가 의도한
//!   correctly-rounded fma를 f64 정확합→f16 1회 반올림으로 대체
//!   (실측 2026-10-04: 타이 예 mul1(0xd52e)=-1.2875977에서 2연산은
//!   절반-ULP 경계를 어겨 GPU FFMA 결과와 1 ulp 어긋남).
//! - nseg=16 분할·FOLD=4 케이던스·had_out nseg 합산 순서는 커널과 동일.
//!
//! [실측 원장 2026-10-04, RTX 4070 SUPER(sm_89) — 검증 호스트]
//!   합성 (i) 27B gate_proj 형상: 2.720e-4 · (ii) 35B qkv 형상: 1.490e-4
//!   (iiia/iiib) 동일 k 이중 선형: 2.064e-4 / 2.184e-4 · 음성대조: 2.844e-1
//!   (G2 값 — 결함 20호 수정 전 오라클. 수정 후 동일 시드 재측:
//!   1.623e-4 · 1.021e-4 · 1.414e-4 · 1.511e-4 — 전 항목 개선, 원인 참조)
//!   실가중 27B gate_proj(모듈 load_keys): 2.216e-4
//!   노름(G3) (i) 27B hidden=5120 w=1/129: 4.768e-7 · (ii) 35B hidden=2048:
//!   0.000e0 · (iii) 강분리 행 전 w + T=4: 0.000e0(코어 오라클과 비트일치)
//!   노름 음성대조 (a) L0 판독: 6.613e0 · (b) eps 1e-5: 6.719e-4
//!   sm_80 자원 증거(cuobjdump --dump-resource-usage, 커밋 fatbin):
//!   exl3_norm_resid REG:30 STACK:32 SHARED:4096(블록 1024스레드) →
//!   GA100 2블록/SM(레지스터 61440/65536 · smem 8KB) = 2048스레드 풀점유
//!   (sm_80 CMP 170HX 실측은 도착 후 — plans/124 §0)
//!   배치 GEMM(G4): (i) 27B k=5120 n=17408 K=3 T=32[plain]: 2.190e-6 ·
//!   (ii) T=1[kseg]: 2.235e-7 · (iii) T=4[kseg]: 2.831e-7 ·
//!   (iv) 35B k=2048 n=8192 K=4 T=32[plain]: 5.662e-7 — 전 항목 임계
//!   4e-4 대비 180~1780배 여유(mma f32 누산 순서 계급 실측).
//!   had_in 비트일치: kseg T=4 0/5120 · plain T=32 0/40960(결함 20호
//!   수정 후 — 이전에는 1 ulp 오라클 결함이 11/5120 위양).
//!   음성대조 had_out 2회: 3.118e-1 > 4e-4 → NEG-DETECTED(결함 15호 감지).
//!   sm_80 자원 증거: exl3_gemm2 REG:72 SHARED:6144 → 7블록/SM(28와프),
//!   exl3_gemm2_kseg REG:80 SHARED:6144 → 6블록/SM(24와프) — GA100
//!   레지스터 65536 기준, 디코드 INT 2 IPC 포화(와프 8 필요) 대비 여유.
//!
//! GDN 체인 프로브(G5): conv→l2perm→scan→gate 4커널 종단을 실가중 상수
//! (아카이브 conv1d/A_log/dt_bias/in_proj_a·b/norm — vk gdn_frame_init과
//! 동일 무변환 경로) + 결정론 시드 입력(S0≠0·링≠0 의무)으로 실행, 아래
//! 내장 f16-정밀 미라 오라클과 값 판정. 트랜센던트는 자작 f64 미러
//! 트윈(assets/exl3_gdn.cu gdn_exp_d/gdn_log_d와 비트동일 — 빌드
//! -fmad=false)로 재현한다: libdevice expf는 참값 ±1ulp(실측 3.1M점
//! 중 30%가 정확히 1ulp 편차)·__expf/__logf는 호스트 비트재현 불가,
//! 그 잔차가 f16 저장 경계 교차를 일으켜 게이트 rms 증폭(실측
//! 1/rms(o_lc)≈289) 후 종단 3.3e-4로 임계 2e-4를 초과함을 실측
//! (2026-10-04) — 미러 계약(plans/124 §6)으로 해소, 아래 원장 참조.
//!
//! [실측 원장 2026-10-04, RTX 4070 SUPER(sm_89) — 검증 호스트]
//!   GDN (i) 27B lay=47 T=32 S0≠0: 종단·전 단계 maxdiff 0.000e0(오라클과
//!   비트동일) · rel>5% 0/196608 · (ii) 35B-A3B lay=29 hidden=2048 hv=32
//!   conv_ch=8192: 0.000e0 · rel 0/131072 — 미러 계약 실증(f16 저장
//!   지점·환원 순서·소거 순서까지 전 단계 일치).
//!   음성대조 (a) l2perm gather: 2.706e-1 · (b) S0=0: 1.150e0 > 2e-4 →
//!   NEG-DETECTED(방향 결함·상태 경로 무시 모두 탐지됨).
//!   [G5 정밀화 원장 — 계기 발견 사실] libdevice expf 오라클 시대 실측:
//!   스캔 단계 7.6e-6(f16 경계 교차) → 게이트 증폭 → 종단 3.3e-4(FAIL).
//!   발산원 국소화: l2perm/conv 산출을 디바이스 판독값으로 대체해도
//!   동일 7.612e-6 → 발산원은 scan 내부(expf 편차 + FMA 수축)로 특정.
//!   처방: 자작 f64 트랜센던트 + -fmad=false + IEEE sqrt qscale(커널측
//!   4항 차이, assets/exl3_gdn.cu 헤더) → 전 단계 비트동일.
//!   sm_80 자원 증거(cuobjdump --dump-resource-usage, 커밋 fatbin):
//!   exl3_gdn_scan REG:48 SHARED(정적):0(+동적 61,828B opt-in) → GA100
//!   164KB/SM 기준 2블록/SM 상한(그리드 h_v=48 블록 — 청크 순차 상태
//!   의존으로 헤드 이상 병렬화 불가, 48<70SM) · exl3_gdn_l2perm(+gather)
//!   REG:32 SHARED:512 → 12블록/SM(와프 48 상한, T=32 그리드 1536블록
//!   포화) · exl3_gdn_conv REG:32 SHARED:0(스트리밍) · gate REG:20
//!   SHARED:512. 개발기 4070(sm_89)은 정합 검증 전용 — CMP 170HX 실측은
//!   도착 후(plans/124 §0).
//!
//! 진단 계기 발견 사실(2026-10-04, 커밋 본문에도 기록):
//! - [결함 20호 — G4 발견, 수정] f32_to_f16 노멀 경로 반올림 경계가
//!   rem==0x0fff(절반 직전)로 오타 — RTNE 경계는 0x1000(정확한 절반).
//!   h 홀수+rem 0x0fff에서 호스트만 올림 → 커널 __float2half_rn과 1 ulp
//!   어긋남. 실측: pre-scale 곱 비트동일(0x39292FFF)인데 호스트만
//!   0x094A/커널 0x0949 — G2의 "bit-diff=0"은 데이터 행운(적중 기대
//!   ~2^-14/값). 수정 후 had_in 전 행 비트일치 + gemv 4종 개선.
//! - HFMA2(plain)는 단일 반올림 IEEE fma로 실측 확인(3-입수 프로브,
//!   1048576표본 0 불일치 — 서브노멀 포함).
//! - 미러 f64→f16 서브노멀 경로 시프트 결함(43-e를 14-e로 오기 — f32판
//!   공식 복사 사고): 결함 시 sb maxdiff 4.2e-3 → 수정 후 잔여 오차는
//!   코어 미러와의 예상 계급으로 수렴.

use crate::rawcuda::ctx::CudaCtx;
use crate::rawcuda::exl3_cuda::Exl3CudaDecoder;
use std::ffi::c_void;

/// 1/√128(커널 R 상수와 동일 비트).
pub(crate) const R_SCALE: f32 = 0.08838834764831845;

/// 스모크 fatbin 자산 해석 — LLM170_CUDA_FATBIN_PATH 오버라이드 우선
/// (rawhip LLM170_CO*_PATH 미러 — 자산 경로 오버라이드일 뿐 계산 경로
/// 분기 아님). 후보는 실행 기준 상대 경로 2종(저장소 루트·크레이트 루트).
fn smoke_fatbin_bytes() -> Result<Vec<u8>, String> {
    const ENV: &str = "LLM170_CUDA_FATBIN_PATH";
    const REL: &[&str] = &[
        "crates/backend-gpu/src/rawcuda/assets/smoke.fatbin", // 저장소 루트 실행(서버·게이트 규약)
        "src/rawcuda/assets/smoke.fatbin", // 크레이트 루트 실행(cargo run --example)
    ];
    if let Some(p) = std::env::var_os(ENV) {
        return std::fs::read(&p).map_err(|e| format!("{ENV}({p:?}) 읽기 실패: {e}"));
    }
    for r in REL {
        if let Ok(b) = std::fs::read(r) {
            return Ok(b);
        }
    }
    Err(format!(
        "smoke.fatbin 없음 — scripts/build_cuda.bat 실행 또는 {ENV} 지정 (탐색: {REL:?})"
    ))
}

/// exl3-cuda-smoke — 디바이스·드라이버 API·fatbin 로드·런치·복사 전 경로 검증.
pub fn cuda_smoke_check() -> Result<String, String> {
    let image = smoke_fatbin_bytes()?;
    let mut cc = CudaCtx::new()?;
    let _g = cc.guard()?;
    cc.load_fatbin("smoke", &image, &["llm170_smoke_add"])?;
    let f = cc.function("llm170_smoke_add")?;

    // 검증 산술: in[i]=i·0.5, scale=4.0 → out[i]=2i+i=3i — 전 단계 f32 정확
    // 표현(i<2^24)이라 FMA 수축 여부와 무관하게 비트동일 가능(스모크는 배관
    // 검증 — 근사 허용 없음, plans/124 §5 값 maxdiff 원칙의 스켈레톤 판).
    const N: usize = 4096;
    const BLOCK: u32 = 256;
    let scale = 4.0f32;
    let input: Vec<f32> = (0..N).map(|i| i as f32 * 0.5).collect();
    let din = cc.alloc(N * 4)?;
    let dout = cc.alloc(N * 4)?;
    // SAFETY: input은 길이 N*4 바이트의 f32 슬라이스 — 바이트 뷰 변환.
    let inb = unsafe { std::slice::from_raw_parts(input.as_ptr() as *const u8, N * 4) };
    cc.h2d(din, inb)?;

    let (mut a0, mut a1, mut a2, mut a3) = (din, dout, N as i32, scale);
    let mut args: [*mut c_void; 4] = [
        (&mut a0) as *mut _ as *mut c_void,
        (&mut a1) as *mut _ as *mut c_void,
        (&mut a2) as *mut _ as *mut c_void,
        (&mut a3) as *mut _ as *mut c_void,
    ];
    cc.launch(f, (N as u32).div_ceil(BLOCK), 1, BLOCK, &mut args)?;
    cc.sync()?;

    let mut outb = vec![0u8; N * 4];
    cc.d2h(&mut outb, dout)?;
    let verify = {
        // SAFETY: outb는 d2h가 채운 N*4 바이트 — f32 배열로 재해석(정렬·길이 일치).
        let out = unsafe { std::slice::from_raw_parts(outb.as_ptr() as *const f32, N) };
        let mut bad: Vec<usize> = Vec::new();
        for (i, &v) in out.iter().enumerate() {
            let want = (3 * i) as f32;
            if v.to_bits() != want.to_bits() && bad.len() < 4 {
                bad.push(i);
            }
        }
        if bad.is_empty() {
            Ok(())
        } else {
            Err(format!(
                "스모크 값 불일치 — 첫 {}건: {} (기대 3i 비트동일)",
                bad.len(),
                bad.iter()
                    .map(|&i| format!("out[{i}]={:.6e} want={:.6e}", out[i], (3 * i) as f32))
                    .collect::<Vec<_>>()
                    .join(", ")
            ))
        }
    };
    cc.free(din)?;
    cc.free(dout)?;
    verify?;
    Ok(format!("device: {} | PASS", cc.device_name))
}

// ── f16 비트 변환(half 크레이트 금지 — std 전용 RTNE 미러) ──

/// f32 → f16 RTNE 비트(__float2half_rn 규약 — 서브노멀·포화 지원).
pub(crate) fn f32_to_f16(v: f32) -> u16 {
    let x = v.to_bits();
    let sign = ((x >> 16) & 0x8000) as u16;
    let exp = ((x >> 23) & 0xff) as i32;
    let mant = x & 0x007f_ffff;
    if exp == 0xff {
        // inf/NaN — quiet 비트 유지(half 규약).
        return sign | 0x7c00 | if mant != 0 { 0x0200 } else { 0 };
    }
    let e = exp - 127 + 15;
    if e >= 0x1f {
        // 포화: f16 최대 초과 → inf(half::from_f32 규약).
        return sign | 0x7c00;
    }
    if e <= 0 {
        if e < -10 {
            // |v| < 최소 서브노멀의 절반 → ±0.
            return sign;
        }
        // 서브노멀: M = round(1.mant × 2^(e-14)) RTNE.
        let m = mant | 0x0080_0000;
        let shift = (14 - e) as u32;
        let half = 1u32 << (shift - 1);
        let rem = m & ((1u32 << shift) - 1);
        let mut mm = m >> shift;
        if rem > half || (rem == half && (mm & 1) == 1) {
            mm += 1;
        }
        // mm == 1024 → 최소 노멀(0x0400) 승격 — 비트 표현이 그대로 맞음.
        return sign | mm as u16;
    }
    let mut h = ((e as u32) << 10) | (mant >> 13);
    let rem = mant & 0x1fff;
    // RTNE: 반올림 경계는 rem==0x1000(버림 13비트의 정확한 절반)이다.
    // 0x0fff(절반 직전) 기준이면 h 홀수 때 절반 미만을 올려버린다 —
    // G4 발견 결함 20호(원장 17호: 검증 계기 자체 결함): 0x0fff 조건에서
    // 커널 __float2half_rn(참 RTNE)과 1 ulp 어긋남(실측: k[1186] pre-scale
    // 곱 비트동일 0x39292FFF에서 호스트만 올림 — G2 bit-diff=0은 데이터
    // 행운, 기대 적중률 ~2^-14/값). f64_to_f16·서브노멀 경로는 원래 정상.
    if rem > 0x1000 || (rem == 0x1000 && (h & 1) == 1) {
        h += 1; // 올림이 지수 경계를 넘기면 비트 표현이 자연 처리(0x7BFF→0x7C00).
    }
    sign | h as u16
}

/// f64(정확합) → f16 RTNE — hfma2 단일 반올림 재현(f64 경유 이중 반올림
/// 경계(≈2^-13 확률/연산)를 제거하기 위해 f64에서 직접 반올림).
pub(crate) fn f64_to_f16(v: f64) -> u16 {
    let x = v.to_bits();
    let sign = ((x >> 48) & 0x8000) as u16;
    let exp = ((x >> 52) & 0x7ff) as i32;
    let mant = x & 0x000f_ffff_ffff_ffff;
    if exp == 0x7ff {
        return sign | 0x7c00 | if mant != 0 { 0x0200 } else { 0 };
    }
    let e = exp - 1023 + 15;
    if e >= 0x1f {
        return sign | 0x7c00;
    }
    if e <= 0 {
        if e < -10 {
            return sign;
        }
        // 서브노멀: M = round(1.mant52 × 2^(e-15+24)) = m53 >> (43-e).
        // (f32 판과 시프트 폭이 다르다 — 53비트 가수: 43-e. 14-e로 쓰면
        // mm가 39비트로 넘쳐 u16 절단 쓰레기가 된다 — 실측 결함 원인.)
        let m = mant | 0x0010_0000_0000_0000; // 53비트 1.m
        let shift = (43 - e) as u32;
        let half = 1u64 << (shift - 1);
        let rem = m & ((1u64 << shift) - 1);
        let mut mm = m >> shift;
        if rem > half || (rem == half && (mm & 1) == 1) {
            mm += 1;
        }
        return sign | mm as u16;
    }
    let mut h = ((e as u64) << 10) | (mant >> 42);
    let rem = mant & ((1u64 << 42) - 1);
    let half = 1u64 << 41;
    if rem > half || (rem == half && (h & 1) == 1) {
        h += 1;
    }
    sign | h as u16
}

/// f16 비트 → f32(비트동일 확장).
pub(crate) fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h & 0x8000) as u32) << 16;
    let exp = ((h >> 10) & 0x1f) as u32;
    let mant = (h & 0x03ff) as u32;
    let bits = if exp == 0 {
        if mant == 0 {
            sign
        } else {
            // 서브노멀 정규화: 값 = mant×2^-24 = (1.f)×2^(-14-s) —
            // f32 지수 필드 = 113-s(h=0x0001 → s=10 → 103 → 2^-24).
            let mut m = mant;
            let mut e = 0u32;
            while m & 0x0400 == 0 {
                m <<= 1;
                e += 1;
            }
            sign | ((113 - e) << 23) | ((m & 0x03ff) << 13)
        }
    } else if exp == 0x1f {
        sign | 0x7f80_0000 | (mant << 13)
    } else {
        sign | ((exp + 112) << 23) | (mant << 13)
    };
    f32::from_bits(bits)
}

/// 바이트 버퍼에서 i번째 f16 LE 비트.
pub(crate) fn f16le(buf: &[u8], i: usize) -> u16 {
    u16::from_le_bytes([buf[2 * i], buf[2 * i + 1]])
}

// ── 결정론 RNG — crates/core/src/sampler.rs Rng::next_u64(L21-28) 미러 ──
/// 외부 rand 크레이트 금지 규약 — 자작 splitmix64(원본 상수 그대로).
pub(crate) struct Rng(u64);

impl Rng {
    pub(crate) fn new(seed: u64) -> Self {
        Rng(seed)
    }
    pub(crate) fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    /// [0,1) 균일 — 상위 53비트(sampler.rs next_f64 L30-33 미러).
    pub(crate) fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }
}

// ── 트렐리스 참조 디코드 — crates/exl3/src/trellis.rs 미러 ──

/// tensor_core_perm 역표 — 위치 (r*16+c) → 트렐리스 워드 인덱스 t.
/// trellis.rs PERM_INV L24-44 const 블록 직이식(원본 그대로).
#[rustfmt::skip]
pub(crate) const PERM_INV: [u16; 256] = {
    let mut inv = [0u16; 256];
    let mut t = 0;
    while t < 32 {
        let r0 = (t % 4) * 2;
        let c0 = t / 4;
        let mut s = 0;
        while s < 8 {
            let r = r0 + [0, 1, 8, 9, 0, 1, 8, 9][s];
            let c = c0 + if s < 4 { 0 } else { 8 };
            inv[r * 16 + c] = (t * 8 + s) as u16;
            s += 1;
        }
        t += 1;
    }
    inv
};

/// mul1 코드북: 16비트 워드 → f16 비트 — trellis.rs mul1_decode L47-52 원식.
/// 커널 경로는 FFMA(단일 반올림) → F2FP(RTNE) — 미러는 f64 정확합(base·k−b
/// 는 53비트 내 정확) → f16 1회 RTNE로 동일하게 재현한다. f32 mul_add는
/// 대상 특성에 따라 소프트웨어 경로(이중 반올림)로 떨어져 f16 타이 경계에서
//  1 ulp 어긋난다 — 실측 원인(2026-10-04, 타이 예: mul1(0xd52e)=-1.2875977).
pub(crate) fn mul1_f16(word: u16) -> u16 {
    let x = (word as u32).wrapping_mul(0x83DCD12D);
    let sum = (x & 0xFF) + ((x >> 8) & 0xFF) + ((x >> 16) & 0xFF) + (x >> 24);
    let v = (1024.0f64 + sum as f64) * 0.00676727294921875 - 10.3828125;
    f64_to_f16(v)
}

/// 타일 비트링에서 워드 t 추출 — trellis.rs tile_word L60-70 직이식.
/// `u32s`: 타일의 8K u32 워드. 워드 t = 링 비트 [(t+1)K-16, (t+1)K)
/// (테일바이팅 환형 — MSB 우선).
pub(crate) fn tile_word(u32s: &[u32], krate: u32, t: u32) -> u16 {
    let words32 = 8 * krate as usize;
    let b0 = (t * krate + (krate + 256 * krate - 16)) as usize;
    let b1 = b0 + 16;
    let i0 = (b0 / 32) % words32;
    let i1 = ((b1 - 1) / 32) % words32;
    let s = ((b1 - 1) / 32 + 1) * 32 - b1;
    let merged = ((u32s[i0] as u64) << 32) | u32s[i1] as u64;
    ((merged >> s) & 0xFFFF) as u16
}

/// 타일 1개 디코드 — trellis.rs decode_tile L85-91 미러 + 커널의 f16
/// 팩(__floats2half2_rn ≡ FFMA+F2FP) 적용. out은 위치 순서 (r*16+c) 256값(f32).
pub(crate) fn decode_tile_f16(
    tre_u32: &[u32],
    krate: u32,
    kt: usize,
    nt: usize,
    ntiles: usize,
    out: &mut [f32; 256],
) {
    let words32 = 8 * krate as usize;
    let tile = &tre_u32[(kt * ntiles + nt) * words32..][..words32];
    for pos in 0..256usize {
        let t = PERM_INV[pos] as u32;
        out[pos] = f16_to_f32(mul1_f16(tile_word(tile, krate, t)));
    }
}

/// f32 자연 순서 WHT-128 — rawvk/checks/exl3.rs had128_f32 L11-25 직이식
/// (커널 버터플라이와 동일 순서).
pub(crate) fn had128_f32(v: &mut [f32]) {
    let mut w = 1usize;
    while w < 128 {
        let mut blk = 0;
        while blk < 128 {
            for i in 0..w {
                let a = v[blk + i];
                let b = v[blk + w + i];
                v[blk + i] = a + b;
                v[blk + w + i] = a - b;
            }
            blk += 2 * w;
        }
        w *= 2;
    }
}

// ── 오라클 체인 참조 — rawvk/checks/exl3.rs L54-105 미러 + GEMV 커널
// (assets/exl3_gemv.cu)의 nseg/FOLD/hfma2 의미론 반영 ──

/// 선형 3중(참조 소유 — 모듈 등록과 동일 바이트를 가리킨다).
pub(crate) struct RefLin<'a> {
    pub(crate) k: usize,
    pub(crate) n: usize,
    pub(crate) krate: u32,
    pub(crate) suh: &'a [u8],
    pub(crate) svh: &'a [u8],
    pub(crate) tre_u32: &'a [u32],
}
/// 값 maxdiff·nan 집계(정합은 값으로 — argmax 금지).
pub(crate) fn maxdiff_nan(got: &[f32], want: &[f32]) -> (f32, usize) {
    let mut md = 0f32;
    let mut nan = 0usize;
    for (g, w) in got.iter().zip(want) {
        if !g.is_finite() {
            nan += 1;
            continue;
        }
        md = md.max((g - w).abs());
    }
    (md, nan)
}


// ── 공유 생성기(norm·ew/argmax 프루브 공용 — G10 분할 이동) ──
/// 결정론 균일 ±amp(잔류 스트림 계급 ±0.2 — 초기층 잔차 스케일급).
pub(crate) fn gen_unif(n: usize, seed: u64, amp: f64) -> Vec<f32> {
    let mut rng = Rng::new(seed);
    (0..n)
        .map(|_| ((rng.next_f64() * 2.0 - 1.0) * amp) as f32)
        .collect()
}

// ── 공유 safetensors 판독(gdn·attn 픽스처 공용 — G10 분할 이동) ──
/// safetensors 원시 바이트 → f32(F16/BF16/F32 — dtype_of 코드별).
pub(crate) fn st_to_f32(bytes: &[u8], dt: u8) -> Result<Vec<f32>, String> {
    match dt {
        1 => Ok(bytes
            .as_chunks::<2>().0.iter()
            .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect()),
        2 => Ok(bytes
            .as_chunks::<2>().0.iter()
            .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
            .collect()),
        0 => Ok(bytes
            .as_chunks::<4>().0.iter()
            .map(|c| f32::from_bits(u32::from_le_bytes([c[0], c[1], c[2], c[3]])))
            .collect()),
        other => Err(format!("GDN 상수 dtype 코드 {other} 미지원")),
    }
}
// ── 합성 선형·등록·입력 생성(gemv·gemm2 프루브 공용 — G10 분할 이동) ──
// ── 합성 선형(결정론 시드) ──
pub(crate) struct SynthLin {
    pub(crate) k: usize,
    pub(crate) n: usize,
    pub(crate) krate: u32,
    suh: Vec<u8>,
    svh: Vec<u8>,
    tre_u32: Vec<u32>,
}

impl SynthLin {
    /// suh/svh: 실측 보정 스케일 분포 미러(2026-10-04 실모델 계측 —
    /// 27B gate_proj suh p50≈1.1e-2·svh p50≈9.9e-1 등) · trellis: 균일 u32
    /// (코드북 전 도메인 커버). 입력 스케일은 hip 프로브와 동일 ±0.1.
    pub(crate) fn generate(k: usize, n: usize, krate: u32, seed: u64) -> Self {
        let mut rng = Rng::new(seed);
        let suh16 = |rng: &mut Rng| f32_to_f16((0.010 + rng.next_f64() * 0.003) as f32);
        let svh16 = |rng: &mut Rng| f32_to_f16((0.8 + rng.next_f64() * 0.4) as f32);
        let suh: Vec<u8> = (0..k).flat_map(|_| suh16(&mut rng).to_le_bytes()).collect();
        let svh: Vec<u8> = (0..n).flat_map(|_| svh16(&mut rng).to_le_bytes()).collect();
        let tre_u32: Vec<u32> = (0..(k / 16) * (n / 16) * 8 * krate as usize)
            .map(|_| rng.next_u64() as u32)
            .collect();
        Self {
            k,
            n,
            krate,
            suh,
            svh,
            tre_u32,
        }
    }

    /// trellis u32 → LE 바이트(모듈 등록용 — 단일 소스).
    pub(crate) fn tre_bytes(&self) -> Vec<u8> {
        let mut b = Vec::with_capacity(self.tre_u32.len() * 4);
        for w in &self.tre_u32 {
            b.extend_from_slice(&w.to_le_bytes());
        }
        b
    }

    pub(crate) fn ref_lin(&self) -> RefLin<'_> {
        RefLin {
            k: self.k,
            n: self.n,
            krate: self.krate,
            suh: &self.suh,
            svh: &self.svh,
            tre_u32: &self.tre_u32,
        }
    }
}

/// 결정론 입력 x: 균일 ±0.1(hip gemv 프로브와 동일 스케일 — 실측 정합
/// 기준의 데이터 계급을 맞춘다).
pub(crate) fn gen_x(k: usize, seed: u64) -> Vec<f32> {
    let mut rng = Rng::new(seed);
    (0..k)
        .map(|_| ((rng.next_f64() * 2.0 - 1.0) * 0.1) as f32)
        .collect()
}


/// 합성 선형 등록(모듈 add_linear_bytes 경유 — 상주는 모듈층 소유).
pub(crate) fn register(dec: &mut Exl3CudaDecoder, key: &str, lin: &SynthLin) -> Result<(), String> {
    dec.add_linear_bytes(
        key,
        lin.k,
        lin.n,
        lin.krate,
        &lin.suh,
        &lin.tre_bytes(),
        &lin.svh,
    )
}

// ── GDN 트랜센던트 트윈·트리 환원(gdn·attn·ew 프루브 공용 — G10 분할 이동,
// assets/exl3_gdn.cu gdn_exp_d/gdn_log_d와 비트동일 미러 — 빌드 -fmad=false) ──
/// f16 RTNE 왕복(커널 __float2half_rn → __half2float 저장 지점 미러).
pub(crate) fn h16f(v: f32) -> f32 {
    f16_to_f32(f32_to_f16(v))
}

/// 미러 트랜센던트 트윈 — assets/exl3_gdn.cu의 gdn_exp_d/gdn_log_d와
/// 동일 f64 연산 DAG(리터럴까지 동일 — 비트동일 계약, -fmad=false 빌드).
/// Rust f64 mul/add/div/floor는 IEEE 정확 반올림, 수축 없음 → 장치와
/// 비트동일. 절대 재작성 금지(원장 17호: 계기 자체가 계약).
pub(crate) fn gdn_exp_d(x: f64) -> f64 {
    let invln2 = 1.4426950408889634f64;
    let ln2_hi = 6.9314718036912382e-01f64;
    let ln2_lo = 1.9082149292705877e-10f64;
    let k = (x * invln2 + 0.5).floor() as i32;
    let mut r = x - k as f64 * ln2_hi;
    r -= k as f64 * ln2_lo;
    let p = 1.0 + r * (1.0 + r * (0.5 + r * (0.16666666666666666f64
        + r * (0.041666666666666664f64 + r * (0.008333333333333333f64
        + r * (0.001388888888888889f64 + r * 0.0001984126984126984f64))))));
    let scale = f64::from_bits(((1023 + k) as u64) << 52);
    p * scale
}

/// gdn_expf 트윈(장치 (float)gdn_exp_d((double)x) 캐스트와 동일 RTNE).
pub(crate) fn gdn_expf(x: f32) -> f32 {
    gdn_exp_d(x as f64) as f32
}

/// gdn_log_d 트윈(m·2^e 규약 → atanh 급수 z^11차 → +e·ln2).
pub(crate) fn gdn_log_d(y: f64) -> f64 {
    let ln2 = 6.9314718036912382e-01f64;
    let bits = y.to_bits();
    let mut e = (((bits >> 52) & 0x7ff) as i32) - 1023;
    let mut m = f64::from_bits((bits & 0x800f_ffff_ffff_ffff) | 0x3ff0_0000_0000_0000);
    if m > 1.4142135623730951f64 {
        m *= 0.5;
        e += 1;
    }
    let s = (m - 1.0) / (m + 1.0);
    let z = s * s;
    let q = 1.0 + z * (0.3333333333333333f64 + z * (0.2 + z * (0.14285714285714285f64
        + z * (0.1111111111111111f64 + z * (0.09090909090909091f64 + z * (0.07692307692307693f64
        + z * (0.06666666666666667f64 + z * (0.058823529411764705f64 + z * (0.05263157894736842f64
        + z * (0.047619047619047616f64 + z * 0.043478260869565216f64))))))))));
    2.0 * s * q + e as f64 * ln2
}

/// gdn_logf 트윈.
pub(crate) fn gdn_logf(y: f32) -> f32 {
    gdn_log_d(y as f64) as f32
}

/// red[128] 트리 환원(l2perm/gate 커널 red[tid] += red[tid+st],
/// st=64..1 — 산술 순서 미러).
pub(crate) fn red128_tree(red: &mut [f32; 128]) {
    let mut st = 64usize;
    while st > 0 {
        for tid in 0..st {
            red[tid] += red[tid + st];
        }
        st >>= 1;
    }
}
