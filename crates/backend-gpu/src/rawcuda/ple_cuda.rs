//! Flash-Next PLE(n-gram 해시 임베딩) CUDA 모듈층 — plans/124 FNB, 2026-10-05.
//!
//! [용도] core qwen4exp PLE 스테이지의 rawcuda 모듈 구현(G8 Q4Cuda·G9
//! Exl3CudaMtp의 독립 모듈 파일 계열 — fn_*_cuda.rs 스캐폴드와의 병합 배선은
//! 리드 소관, 병렬 작업 계약):
//! - `ple_hash_rows`: 호스트 u64 n-gram 해시 — core stages/ple.rs
//!   ple_hash_rows L276-335 미러(순수 함수 — 값 경로와 동일 계약.
//!   GPU u64 나머지 경제성 없음, 스캐폴드 계약 지도 그대로).
//! - `ple_gather_stage`: GGUF `per_layer_token_embd.weight`(IQ4_NL
//!   [160, 320001536] — 2026-10-05 실측) 행 pread 스테이징 → 장치 디양자화.
//!   오프셋 산식은 core mod.rs ple_gather_parts L471-505(block_info 기반,
//!   행당 90B)·다중 샤드 행 소스(PartMap 상당)는 FnGguf(fn_support)가 담당.
//!   37GiB급 테이블 전량 상주/적재 금지 — 행 단위 pread만(plans/86 §6).
//! - `ple_block`: 게이트·value 방송 grouped norm·dilated conv(상태 링)·
//!   잔차 2경로 — core stages/ple.rs L25-235 산술 순서 미러(커널 산술 계약은
//!   assets/exl3_fn_ple.cu 헤더). key/value 투영은 호출자 입력(REUSE —
//!   G2/G8 gemv/gemm 영역, 리드가 병합 시 배선).
//!
//! [정합 원장 — 프로브 ple_cuda_probe.rs `ple`/`ple-neg`(plans/129-cuda C2)]
//! - ple_hash: 결정론·실측 파라미터 입력 rows/hist 코어 미러와 u32 전량
//!   일치(2026-10-05 실측, RTX 4070 SUPER 검증 호스트: 4세트 64/17/1/24
//!   (EOS 절단 포함) rows 1696개 불일치 0 — 청크 분할 16+48=64 단일 호출과
//!   rows/hist 동일).
//! - ple_gather: 실해시 행 128개(8토큰×16헤드) IQ4_NL pread emb 비트동일
//!   (bitdiff=0, maxdiff 0.000e0) — 재스테이징 결정성 포함.
//! - ple_block: 실가중(norms F32·conv1d F32 [4,10240]·key/value Q8_0
//!   [2560,10240]/[2560,2560] 디양자 공유 입력) t=13 — emb/gates/gated/
//!   conv_out/res_hc/상태 전 6단계 bitdiff=0(maxdiff 0.000e0)·청크 8+5
//!   분할 res·상태 비트동일(2청크째 상태 비영 — S0≠0).
//! - 음성대조: 해시 계수 vs[3]+7 오염 maxdiff 3.277e-2·게더 인덱스 오염
//!   3.178e-2 → NEG-DETECTED(비영 exit, 임계 1e-3 초과).
//! [속도] 측정 대기 sm_80 — CMP 170HX 도착 후(개발기 4070 SUPER는 정합
//! 검증 전용, plans/124 §0).
//!
//! [독립 컴파일 계약] scripts/cuda_probe_shim.rs가 rustc로 단독 컴파일 —
//! std 외 크레이트 금지(G1 원장). 청크 h2d는 G8 패턴 미러(임포트 아닌
//! 자작 사본 — 모듈별 독립 파일 구조 계약).
//!
//! 단일 상주 원칙(2026-10-04 동결 사고): 본 모듈이 디바이스 컨텍스트를
//! 소유한다 — 한 프로세스에 모델 1개(plans/124 §5).

use crate::rawcuda::ctx::CudaCtx;
use crate::rawcuda::ffi::CUdeviceptr;
use crate::rawcuda::fn_support::FnDims;

/// GGUF PLE 테이블 텐서명 — core Model4::ple_gather(mod.rs L437-444)와 동일 키.
pub const PLE_TABLE_TENSOR: &str = "per_layer_token_embd.weight";
/// IQ4_NL 블록 형상(ggml block_info: 32원소 18B) — deq.rs deq_iq4_nl 단위.
pub const IQ4NL_BLCK: usize = 32;
pub const IQ4NL_BYTES: usize = 18;
/// 스테이징 상한 토큰 수(PLE 청크 폭 — 프리필 밴드 상한, 프로브 t≤16).
const T_CAP: usize = 1024;
/// 게이트 커널 블록 폭(스레드=토큰×스트림 — 스레드가 순차 환원을 소유하는
/// 비트 계약상 그리드가 작다. sm_80 점유 실측은 도착 후).
const GATE_BLOCK: u32 = 64;
/// 원소별 커널 블록 폭(gather/conv/state/resid — HBM2e 스트리밍 지향).
const ELT_BLOCK: u32 = 256;
/// 대형 h2d 청크 상한(4MB — 페이지 미매핑 가드, G2/G8 원장 패턴 미러).
const H2D_CHUNK: usize = 4 << 20;

/// PLE n-gram 해시 — core stages/ple.rs ple_hash_rows L276-335 직미러.
/// 원본 산술(래핑 곱/xor/%vs+offs, EOS 절단 cut 전파, 청크 경계 lookback은
/// 호출 시작 스냅샷 hist0 판독, hist 진화 drain)을 그대로 — 리터럴·순서
/// "정리" 금지(quant AGENTS 비트 계약과 동일 등급).
/// 반환: (rows[t·heads], 진화된 hist[ngram-1]).
#[allow(clippy::too_many_arguments)]
pub fn ple_hash_rows(
    hist0: &[u32],
    hist_valid: bool,
    tokens: &[u32],
    ngram: usize,
    hpng: usize,
    mult: &[u64],
    offs: &[u64],
    vs: &[u64],
    eos: u32,
) -> (Vec<u32>, Vec<u32>) {
    let heads = hpng * 2;
    let mut hist: Vec<u32> = if hist_valid {
        hist0.to_vec()
    } else {
        vec![eos; ngram - 1]
    };
    let mut rows = Vec::with_capacity(tokens.len() * heads);
    for (i, &tok) in tokens.iter().enumerate() {
        let mut ctx = vec![tok as u64; ngram];
        let mut cut = false;
        for s in 1..ngram {
            let j = i as i64 - s as i64;
            let prev: u64 = if j >= 0 {
                tokens[j as usize] as u64
            } else {
                // 청크 경계 lookback은 호출 시작 스냅샷(hist0)에서.
                let back = s as i64 - i as i64;
                let k = hist0.len() as i64 - back;
                if k >= 0 && (k as usize) < hist0.len() {
                    hist0[k as usize] as u64
                } else {
                    eos as u64
                }
            };
            ctx[s] = if cut { eos as u64 } else { prev };
            if ctx[s] == eos as u64 {
                cut = true;
            }
        }
        for n in 2..=ngram {
            let mut mixed = ctx[0].wrapping_mul(mult[0]);
            for j in 1..n {
                mixed ^= ctx[j].wrapping_mul(mult[j]);
            }
            let base = (n - 2) * hpng;
            for g in 0..hpng {
                let h = base + g;
                rows.push((mixed % vs[h] + offs[h]) as u32);
            }
        }
        hist.push(tok);
        if hist.len() > ngram - 1 {
            let cutn = hist.len() - (ngram - 1);
            hist.drain(..cutn);
        }
    }
    (rows, hist)
}

/// ple_block 검증 중간 산출(G5 단계별 값 판정 노선 — 호출은 검증층).
#[derive(Default)]
pub struct PleMids {
    /// 게이트 [t][hc].
    pub gates: Vec<f32>,
    /// 방송→grouped norm 출력(conv 입력) [t][hc·n_embd].
    pub gated: Vec<f32>,
    /// dilated conv+silu 출력 [t][hc·n_embd].
    pub conv_out: Vec<f32>,
}

/// PLE CUDA 모듈 — 가중치(norm류·conv1d)·conv 상태 링 상주 + 스테이지 API.
pub struct PleCuda {
    cc: CudaCtx,
    /// 형상(FnGguf/EXL3 config에서 확정해 전달 — 형상 추정 금지).
    pub dims: FnDims,
    /// 유도 형상 — 게더·판정 가드 공용.
    heads: usize,
    emb_w: usize,
    hc_dim: usize,
    kern: usize,
    hist: usize,
    row_bytes: usize,
    // ── 디바이스 버퍼 ──
    d_nk: CUdeviceptr,
    d_nq: CUdeviceptr,
    d_nc: CUdeviceptr,
    d_cw: CUdeviceptr,
    /// conv 상태 링 더블버퍼 [hist][hc_dim] — 제자리 갱신은 t<hist에서
    /// 판독·기록 열이 겹쳐 경합(assets/exl3_fn_ple.cu 헤더) — 스왑으로 회피.
    d_st: [CUdeviceptr; 2],
    st_cur: usize,
    d_raw: CUdeviceptr,
    d_emb: CUdeviceptr,
    d_key: CUdeviceptr,
    d_value: CUdeviceptr,
    d_res: CUdeviceptr,
    d_gates: CUdeviceptr,
    d_gated: CUdeviceptr,
    d_conv: CUdeviceptr,
    /// 마지막 게더 토큰 수(ple_block 게더-블록 t 일치 가드).
    gather_t: usize,
}

/// f32 슬라이스 → 바이트 뷰(h2d 공용 — 길이·정렬 일치).
fn f32_bytes(v: &[f32]) -> &[u8] {
    // SAFETY: f32 슬라이스의 원시 표현 — 읽기 전용 바이트 뷰(수명은 v에 종속).
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) }
}

/// 대형 h2d 청크 업로드(4MB 상한 — 페이지 미매핑 가드).
fn h2d_chunked(cc: &CudaCtx, dst: CUdeviceptr, src: &[u8]) -> Result<(), String> {
    let mut off = 0usize;
    while off < src.len() {
        let end = (off + H2D_CHUNK).min(src.len());
        // dst는 alloc이 돌려준 유효 할당 내 오프셋 슬라이스(길이 계약).
        cc.h2d(dst + off as u64, &src[off..end])?;
        off = end;
    }
    Ok(())
}

/// f32 바이트 → Vec<f32>(d2h 출력 재해석 — LE 바이트 조립).
fn bytes_f32(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

impl PleCuda {
    /// exl3_fn_ple.fatbin 자산 해석 — LLM170_CUDA_FN_PLE_FATBIN_PATH 오버라이드
    /// 우선(자산 경로 오버라이드일 뿐 계산 경로 분기 아님).
    fn ple_fatbin_bytes() -> Result<Vec<u8>, String> {
        const ENV: &str = "LLM170_CUDA_FN_PLE_FATBIN_PATH";
        const REL: &[&str] = &[
            "crates/backend-gpu/src/rawcuda/assets/exl3_fn_ple.fatbin",
            "src/rawcuda/assets/exl3_fn_ple.fatbin",
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
            "exl3_fn_ple.fatbin 없음 — scripts/build_cuda.bat 실행 또는 {ENV} 지정 (탐색: {REL:?})"
        ))
    }

    /// 모듈 생성 — 커널 등록 + 상태·작업 버퍼 할당(상태 0 초기화).
    /// 형상 가드: ple_head_dim=160·IQ4_NL 행 90B는 게더 커널 형식 계약
    /// (실측 [160, 320001536] — 2026-10-05). hc·n_embd·conv_k·ngram은
    /// FnDims 원값 그대로.
    pub fn new(dims: FnDims) -> Result<Self, String> {
        let heads = dims.ple_heads_per_ngram * 2;
        let emb_w = heads * dims.ple_head_dim;
        let hc_dim = dims.hc * dims.n_embd;
        let kern = dims.ple_conv_k;
        let hist = (kern - 1) * dims.ple_ngram;
        let row_bytes = dims.ple_head_dim.div_ceil(IQ4NL_BLCK) * IQ4NL_BYTES;
        if dims.ple_head_dim != 160 || row_bytes != 90 {
            return Err(format!(
                "ple: ple_head_dim {} (행 {}B) — IQ4_NL 게더 커널 160/90 형식 계약 위반",
                dims.ple_head_dim, row_bytes
            ));
        }
        if dims.ple_layers.is_empty() {
            return Err("ple: ple_layers 비었음 — 형상 등록 계약 위반".into());
        }
        let image = Self::ple_fatbin_bytes()?;
        let mut cc = CudaCtx::new()?;
        {
            let _g = cc.guard()?;
            cc.load_fatbin(
                "exl3fnple",
                &image,
                &[
                    "llm170_ple_iq4nl_gather",
                    "llm170_ple_gate",
                    "llm170_ple_conv",
                    "llm170_ple_conv_state",
                    "llm170_ple_resid",
                ],
            )?;
        }
        let zeros = |n: usize| vec![0u8; n * 4];
        let d_nk = cc.alloc(hc_dim * 4)?;
        let d_nq = cc.alloc(hc_dim * 4)?;
        let d_nc = cc.alloc(hc_dim * 4)?;
        let d_cw = cc.alloc(hc_dim * kern * 4)?;
        let d_st = [cc.alloc(hist * hc_dim * 4)?, cc.alloc(hist * hc_dim * 4)?];
        h2d_chunked(&cc, d_st[0], &zeros(hist * hc_dim))?;
        h2d_chunked(&cc, d_st[1], &zeros(hist * hc_dim))?;
        let d_raw = cc.alloc(T_CAP * heads * row_bytes)?;
        let d_emb = cc.alloc(T_CAP * emb_w * 4)?;
        let d_key = cc.alloc(T_CAP * hc_dim * 4)?;
        let d_value = cc.alloc(T_CAP * dims.n_embd * 4)?;
        let d_res = cc.alloc(T_CAP * hc_dim * 4)?;
        let d_gates = cc.alloc(T_CAP * dims.hc * 4)?;
        let d_gated = cc.alloc(T_CAP * hc_dim * 4)?;
        let d_conv = cc.alloc(T_CAP * hc_dim * 4)?;
        Ok(PleCuda {
            cc,
            dims,
            heads,
            emb_w,
            hc_dim,
            kern,
            hist,
            row_bytes,
            d_nk,
            d_nq,
            d_nc,
            d_cw,
            d_st,
            st_cur: 0,
            d_raw,
            d_emb,
            d_key,
            d_value,
            d_res,
            d_gates,
            d_gated,
            d_conv,
            gather_t: 0,
        })
    }

    pub fn device_name(&self) -> &str {
        &self.cc.device_name
    }

    /// 유도 형상 접근자(검증층 가드·판정용): (heads, emb_w, hc_dim, kern,
    /// hist, row_bytes).
    pub fn shapes(&self) -> (usize, usize, usize, usize, usize, usize) {
        (
            self.heads,
            self.emb_w,
            self.hc_dim,
            self.kern,
            self.hist,
            self.row_bytes,
        )
    }

    /// 블록 가중치 업로드 — norms 3종[hc_dim]·conv1d[hc_dim·kern] f32.
    /// (core는 f32_vec4로 원본 파일 순서 그대로 — 호출자가 GGUF F32/F16
    /// 원시열을 f32로 변환해 전달. conv1d는 ne=[kern, hc_dim] 행우선
    /// → flat c·kern+k — 실측 [4, 10240].)
    pub fn ple_load_block_weights(
        &mut self,
        n_key: &[f32],
        n_query: &[f32],
        n_conv: &[f32],
        conv_w: &[f32],
    ) -> Result<(), String> {
        if n_key.len() != self.hc_dim
            || n_query.len() != self.hc_dim
            || n_conv.len() != self.hc_dim
            || conv_w.len() != self.hc_dim * self.kern
        {
            return Err(format!(
                "ple: 가중치 길이 {}/{}/{}/{} != {}/{}/{}/{}",
                n_key.len(),
                n_query.len(),
                n_conv.len(),
                conv_w.len(),
                self.hc_dim,
                self.hc_dim,
                self.hc_dim,
                self.hc_dim * self.kern
            ));
        }
        let _g = self.cc.guard()?;
        h2d_chunked(&self.cc, self.d_nk, f32_bytes(n_key))?;
        h2d_chunked(&self.cc, self.d_nq, f32_bytes(n_query))?;
        h2d_chunked(&self.cc, self.d_nc, f32_bytes(n_conv))?;
        h2d_chunked(&self.cc, self.d_cw, f32_bytes(conv_w))
    }

    /// conv 상태 링 주입(호스트 → 현재 버퍼) — 길이 hist·hc_dim.
    pub fn ple_set_conv_state(&mut self, st: &[f32]) -> Result<(), String> {
        if st.len() != self.hist * self.hc_dim {
            return Err(format!(
                "ple: 상태 {} != {}×{}",
                st.len(),
                self.hist,
                self.hc_dim
            ));
        }
        let _g = self.cc.guard()?;
        h2d_chunked(&self.cc, self.d_st[self.st_cur], f32_bytes(st))
    }

    /// conv 상태 링 판독(현재 버퍼 → 호스트) — 청크 판정·종단 불변량용.
    pub fn ple_conv_state(&self) -> Result<Vec<f32>, String> {
        let n = self.hist * self.hc_dim;
        let mut b = vec![0u8; n * 4];
        self.cc.d2h(&mut b, self.d_st[self.st_cur])?;
        Ok(bytes_f32(&b))
    }

    /// 게더 스테이징 — 호출자가 pread한 원시 행 바이트(t·heads행 × row_bytes,
    /// 행 순서: rows[ti·heads + h])를 업로드해 IQ4_NL 디양자화 → d_emb.
    /// (스테이징 소유를 호스트에 둔 분할 결정 — .cu 헤더 [호스트/디바이스
    /// 분할 결정] 참조.)
    // [rustfmt skip — G6 병리 계승(2026-10-04, 시간제한 계약)] 발사 인자 배열.
    #[rustfmt::skip]
    pub fn ple_gather_stage(&mut self, raw_rows: &[u8], t: usize) -> Result<(), String> {
        if t == 0 || t > T_CAP {
            return Err(format!("ple gather: t={t} — (0, {T_CAP}] 도메인 위반"));
        }
        let want = t * self.heads * self.row_bytes;
        if raw_rows.len() != want {
            return Err(format!(
                "ple gather: 원시 행 {}B != t{t}·{}h·{}B = {want}B",
                raw_rows.len(), self.heads, self.row_bytes
            ));
        }
        let n_elems = t * self.emb_w;
        let _g = self.cc.guard()?;
        h2d_chunked(&self.cc, self.d_raw, raw_rows)?;
        let f = self.cc.function("llm170_ple_iq4nl_gather")?;
        let (mut a0, mut a1) = (self.d_raw, self.d_emb);
        let mut a2 = n_elems as i32;
        let mut args: [*mut std::ffi::c_void; 3] = [
            (&mut a0) as *mut _ as *mut _,
            (&mut a1) as *mut _ as *mut _,
            (&mut a2) as *mut _ as *mut _,
        ];
        self.cc.launch(f, n_elems.div_ceil(ELT_BLOCK as usize) as u32, 1, ELT_BLOCK, &mut args)?;
        self.gather_t = t;
        Ok(())
    }

    /// 게더 산출 판독[t·emb_w] — 검증층 단계별 값 판정용.
    pub fn ple_emb_download(&self, t: usize) -> Result<Vec<f32>, String> {
        if t != self.gather_t {
            return Err(format!(
                "ple: emb 판독 t={t} != 마지막 게더 t={}",
                self.gather_t
            ));
        }
        let mut b = vec![0u8; t * self.emb_w * 4];
        self.cc.d2h(&mut b, self.d_emb)?;
        Ok(bytes_f32(&b))
    }

    /// 게이트 커널 발사 — 그리드 ⌈t·hc/64⌉(스레드=토큰×스트림).
    // [rustfmt skip — G6 병리 계승(2026-10-04, 시간제한 계약)] 12인자 발사 배열.
    #[rustfmt::skip]
    fn gate_dev(&mut self, t: usize) -> Result<(), String> {
        let (hc, n) = (self.dims.hc, self.dims.n_embd);
        let f = self.cc.function("llm170_ple_gate")?;
        let (mut g0, mut g1) = (self.d_key, self.d_res);
        let (mut g2, mut g3) = (self.d_nk, self.d_nq);
        let (mut g4, mut g5) = (self.d_value, self.d_nc);
        let (mut g6, mut g7) = (self.d_gates, self.d_gated);
        let (mut gt, mut ghc) = (t as i32, hc as i32);
        let (mut gn, mut geps) = (n as i32, self.dims.eps);
        let mut args: [*mut std::ffi::c_void; 12] = [
            (&mut g0) as *mut _ as *mut _, (&mut g1) as *mut _ as *mut _,
            (&mut g2) as *mut _ as *mut _, (&mut g3) as *mut _ as *mut _,
            (&mut g4) as *mut _ as *mut _, (&mut g5) as *mut _ as *mut _,
            (&mut g6) as *mut _ as *mut _, (&mut g7) as *mut _ as *mut _,
            (&mut gt) as *mut _ as *mut _, (&mut ghc) as *mut _ as *mut _,
            (&mut gn) as *mut _ as *mut _, (&mut geps) as *mut _ as *mut _,
        ];
        self.cc.launch(f, (t * hc).div_ceil(GATE_BLOCK as usize) as u32, 1, GATE_BLOCK, &mut args)
    }

    /// dilated conv 발사 — 상태는 현재 버퍼에서 판독(그리드 ⌈t·hc_dim/256⌉).
    // [rustfmt skip — G6 병리 계승(2026-10-04, 시간제한 계약)] 9인자 발사 배열.
    #[rustfmt::skip]
    fn conv_dev(&mut self, t: usize) -> Result<(), String> {
        let hcd = self.hc_dim;
        let f = self.cc.function("llm170_ple_conv")?;
        let (mut c0, mut c1) = (self.d_gated, self.d_st[self.st_cur]);
        let (mut c2, mut c3) = (self.d_cw, self.d_conv);
        let (mut ct, mut chcd) = (t as i32, hcd as i32);
        let (mut ck, mut cd, mut ch) = (
            self.kern as i32,
            self.dims.ple_ngram as i32,
            self.hist as i32,
        );
        let mut args: [*mut std::ffi::c_void; 9] = [
            (&mut c0) as *mut _ as *mut _, (&mut c1) as *mut _ as *mut _,
            (&mut c2) as *mut _ as *mut _, (&mut c3) as *mut _ as *mut _,
            (&mut ct) as *mut _ as *mut _, (&mut chcd) as *mut _ as *mut _,
            (&mut ck) as *mut _ as *mut _, (&mut cd) as *mut _ as *mut _,
            (&mut ch) as *mut _ as *mut _,
        ];
        self.cc.launch(f, (t * hcd).div_ceil(ELT_BLOCK as usize) as u32, 1, ELT_BLOCK, &mut args)
    }

    /// 상태 tail 갱신 발사(더블버퍼 src→dst) — conv 판독 완료 후 순서 계약.
    // [rustfmt skip — G6 병리 계승(2026-10-04, 시간제한 계약)] 6인자 발사 배열.
    #[rustfmt::skip]
    fn conv_state_dev(&mut self, t: usize) -> Result<(), String> {
        let hcd = self.hc_dim;
        let hist = self.hist;
        let f = self.cc.function("llm170_ple_conv_state")?;
        let (mut s0, mut s1) = (self.d_st[self.st_cur], self.d_gated);
        let mut s2 = self.d_st[1 - self.st_cur];
        let (mut st_t, mut shcd) = (t as i32, hcd as i32);
        let mut shist = hist as i32;
        let mut args: [*mut std::ffi::c_void; 6] = [
            (&mut s0) as *mut _ as *mut _,
            (&mut s1) as *mut _ as *mut _,
            (&mut s2) as *mut _ as *mut _,
            (&mut st_t) as *mut _ as *mut _,
            (&mut shcd) as *mut _ as *mut _,
            (&mut shist) as *mut _ as *mut _,
        ];
        self.cc.launch(f, (hist * hcd).div_ceil(ELT_BLOCK as usize) as u32, 1, ELT_BLOCK, &mut args)
    }

    /// 잔차 2경로 발사 — res += value·g + conv_out(그리드 ⌈t·hc_dim/256⌉).
    // [rustfmt skip — G6 병리 계승(2026-10-04, 시간제한 계약)] 8인자 발사 배열.
    #[rustfmt::skip]
    fn resid_dev(&mut self, t: usize) -> Result<(), String> {
        let (hc, n, hcd) = (self.dims.hc, self.dims.n_embd, self.hc_dim);
        let f = self.cc.function("llm170_ple_resid")?;
        let (mut r0, mut r1) = (self.d_res, self.d_value);
        let (mut r2, mut r3) = (self.d_gates, self.d_conv);
        let (mut rt, mut rhcd) = ((t * hcd) as i32, hcd as i32);
        let (mut rn, mut rhc) = (n as i32, hc as i32);
        let mut args: [*mut std::ffi::c_void; 8] = [
            (&mut r0) as *mut _ as *mut _,
            (&mut r1) as *mut _ as *mut _,
            (&mut r2) as *mut _ as *mut _,
            (&mut r3) as *mut _ as *mut _,
            (&mut rt) as *mut _ as *mut _,
            (&mut rhcd) as *mut _ as *mut _,
            (&mut rn) as *mut _ as *mut _,
            (&mut rhc) as *mut _ as *mut _,
        ];
        self.cc.launch(f, (t * hcd).div_ceil(ELT_BLOCK as usize) as u32, 1, ELT_BLOCK, &mut args)
    }

    /// PLE 블록 체인(게이트→conv→상태 갱신→잔차) — key/value는 호출자
    /// 입력(공유 판정 — 투영은 REUSE 영역). res_hc는 제자리 잔차 가산
    /// (core와 동일 계약). mids에 단계 산출을 채운다(검증층 판정용).
    pub fn ple_block(
        &mut self,
        t: usize,
        key: &[f32],
        value: &[f32],
        res_hc: &mut [f32],
        mids: &mut PleMids,
    ) -> Result<(), String> {
        if t == 0 || t > T_CAP {
            return Err(format!("ple block: t={t} — (0, {T_CAP}] 도메인 위반"));
        }
        if t != self.gather_t {
            return Err(format!(
                "ple block: 게더 t={} != 블록 t={t} — 스테이징 순서 계약 위반",
                self.gather_t
            ));
        }
        if key.len() != t * self.hc_dim
            || value.len() != t * self.dims.n_embd
            || res_hc.len() != t * self.hc_dim
        {
            return Err(format!(
                "ple block: key/value/res {}/{}/{} != {t}×{}/{t}×{}/{t}×{}",
                key.len(),
                value.len(),
                res_hc.len(),
                self.hc_dim,
                self.dims.n_embd,
                self.hc_dim
            ));
        }
        let (hc, hcd) = (self.dims.hc, self.hc_dim);
        let _g = self.cc.guard()?;
        h2d_chunked(&self.cc, self.d_key, f32_bytes(key))?;
        h2d_chunked(&self.cc, self.d_value, f32_bytes(value))?;
        // res_hc는 잔차 가산의 입·출력 — 제자리 갱신(core ple_block 계약).
        // SAFETY: res_hc 표현 재해석(길이 t·hc_dim·4B — h2d 직전 읽기 전용).
        let res_ro: &[f32] = res_hc;
        h2d_chunked(&self.cc, self.d_res, f32_bytes(res_ro))?;
        self.gate_dev(t)?;
        self.conv_dev(t)?;
        self.conv_state_dev(t)?;
        self.st_cur = 1 - self.st_cur;
        self.resid_dev(t)?;
        self.cc.sync()?;
        // 판독 — res 제자리 반환 + mids 단계 산출.
        let mut rb = vec![0u8; t * hcd * 4];
        self.cc.d2h(&mut rb, self.d_res)?;
        let res_out = bytes_f32(&rb);
        res_hc.copy_from_slice(&res_out);
        let mut gb = vec![0u8; t * hc * 4];
        self.cc.d2h(&mut gb, self.d_gates)?;
        mids.gates = bytes_f32(&gb);
        let mut db = vec![0u8; t * hcd * 4];
        self.cc.d2h(&mut db, self.d_gated)?;
        mids.gated = bytes_f32(&db);
        let mut cb = vec![0u8; t * hcd * 4];
        self.cc.d2h(&mut cb, self.d_conv)?;
        mids.conv_out = bytes_f32(&cb);
        Ok(())
    }
}
