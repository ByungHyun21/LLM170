//! 원시 HIP 실행기 — cubecl을 거치지 않는 직접 경로 (2026-09-03 재작성).
//! hipRTC로 임베디드 HIP C++ 소스를 컴파일하고 hipModuleLaunchKernel로
//! 실행. 버퍼는 영속 아레나(해제 없음, ADR-0014 동일 규칙). 커널 산술은
//! core 미러(dot_row_w4a8_*_lane)와 동일 연산열 — to_bits 검증 게이트.

use cubecl_hip_sys as hip;
use std::collections::HashMap;
use std::ffi::CString;

pub mod decode;
pub mod kernels;
pub mod vit;

/// 로드된 오프라인 타일 코드오브젝트 패밀리 (임베딩 or LLM170_CO*_PATH
/// 오버라이드). RawCtx::new 완료 후 불변. 타일 발사 게이트는 env가 아니라
/// 이 비트를 본다 — 무환경 기본 성능 = 튜닝 성능.
pub const CO_J128: u8 = 1; // w32b.co: *_j128 계열 (t≤128)
pub const CO_V4: u8 = 2; // v4all.co: *_v4 + *_wm 4종
pub const CO_ODD: u8 = 4; // odd_all.co: nl/q3k/iq3s v4 (plans/04)
static CO_FAM: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

pub fn co_loaded(bit: u8) -> bool {
    CO_FAM.load(std::sync::atomic::Ordering::Relaxed) & bit != 0
}

fn ck(status: hip::hipError_t, what: &str) -> Result<(), String> {
    if status == hip::hipError_t_hipSuccess {
        Ok(())
    } else {
        Err(format!("rawhip: {what}: {status:?}"))
    }
}

/// 컴파일된 커널 실행기.
pub struct RawCtx {
    module: hip::hipModule_t,
    fns: HashMap<&'static str, hip::hipFunction_t>,
    stream: hip::hipStream_t,
    stream2: hip::hipStream_t,
    /// 크기별 스크래치 풀 — 해제 없는 재사용 (호출마다 신규 할당이
    /// 메모리 고갈→illegal address 유발, 2026-09-03 RCA).
    scratch: std::sync::Mutex<HashMap<usize, Vec<*mut u8>>>,
    cursors: std::sync::Mutex<HashMap<usize, usize>>,
}

/// 타일 발사 파라미터 (스택 로컬 소유 — args 포인터 유효성 보장).
struct TileLaunch {
    kern: &'static str,
    xp: *mut std::ffi::c_void,
    wp: *mut std::ffi::c_void,
    op: *mut std::ffi::c_void,
    ktp: *mut std::ffi::c_void,
    ni: i32,
    no: i32,
    xw: i32,
    tt: i32,
    gx: u32,
    gz: u32,
    block: u32,
    ktab: bool,
}


impl RawCtx {
    pub fn new() -> Result<Self, String> {
        unsafe {
            ck(hip::hipSetDevice(0), "hipSetDevice")?;
            let _ = hip::hipSetDeviceFlags(hip::hipDeviceScheduleSpin);

            let src = CString::new(kernels::SRC).unwrap();
            let mut prog: hip::hiprtcProgram = std::ptr::null_mut();
            let rs = hip::hiprtcCreateProgram(&mut prog, src.as_ptr(), std::ptr::null(), 0, std::ptr::null_mut(), std::ptr::null_mut());
            if rs != hip::hiprtcResult_HIPRTC_SUCCESS {
                return Err(format!("hiprtcCreateProgram: {rs:?}"));
            }
            let inc = cubecl_hip_sys::get_hip_include_path().map_err(|e| e.to_string())?;
            let o1 = CString::new(format!("-I{inc}")).unwrap();
            let o2 = CString::new("--std=c++17").unwrap();
            let o3 = CString::new("-O3").unwrap();
            // FMA 수축 차단 — CPU 비트계약 (a+=b*c 축약이 비트 불일치,
            // 2026-09-03 AR xor RCA)
            let o4 = CString::new("-ffp-contract=off").unwrap();
            let o5 = CString::new("-I/opt/rocm/include").unwrap();
            let mut opts = vec![o1.as_ptr(), o2.as_ptr(), o3.as_ptr(), o4.as_ptr(), o5.as_ptr()];
            let rs = hip::hiprtcCompileProgram(prog, opts.len() as i32, opts.as_mut_ptr());
            if rs != hip::hiprtcResult_HIPRTC_SUCCESS {
                let mut sz = 0usize;
                let _ = hip::hiprtcGetProgramLogSize(prog, &mut sz);
                let mut buf = vec![0i8; sz.max(1)];
                let _ = hip::hiprtcGetProgramLog(prog, buf.as_mut_ptr());
                let log = String::from_utf8_lossy(std::slice::from_raw_parts(buf.as_ptr() as *const u8, sz));
                return Err(format!("rawhip 컴파일 실패: {log}"));
            }
            let mut code_sz = 0usize;
            if hip::hiprtcGetCodeSize(prog, &mut code_sz) != hip::hiprtcResult_HIPRTC_SUCCESS {
                return Err("GetCodeSize".into());
            }
            let mut code = vec![0i8; code_sz];
            if hip::hiprtcGetCode(prog, code.as_mut_ptr()) != hip::hiprtcResult_HIPRTC_SUCCESS {
                return Err("GetCode".into());
            }
            if let Some(path) = std::env::var_os("LLM170_DUMP_MODULE") {
                let _ = std::fs::write(&path, std::slice::from_raw_parts(code.as_ptr() as *const u8, code_sz));
            }
            let mut module: hip::hipModule_t = std::ptr::null_mut();
            ck(hip::hipModuleLoadData(&mut module, code.as_ptr() as *const _), "ModuleLoadData")?;
            let mut fns = HashMap::new();
            for name in kernels::NAMES {
                let cname = CString::new(*name).unwrap();
                let mut f: hip::hipFunction_t = std::ptr::null_mut();
                ck(hip::hipModuleGetFunction(&mut f, module, cname.as_ptr()), "GetFunction")?;
                fns.insert(*name, f);
            }
            // 오프라인 코드오브젝트 병행 로드 (wave32 커널 등).
            // 기본: 바이너리 임베딩(crates/.../co/*.co, gfx1151 빌드).
            // LLM170_CO*_PATH가 있으면 그 파일이 우선 (커널 실험 오버라이드).
            // LLM170_NO_CO: 전부 생략 (hipRTC wm/mm + GEMV 폴백 측정용).
            if std::env::var_os("LLM170_NO_CO").is_none() {
                let slots: &[(u8, &str, &[u8], &[&str])] = &[
                    (
                        CO_V4,
                        "LLM170_CO2_PATH",
                        include_bytes!("co/v4all.co"),
                        &["gemm_q5k_v4", "gemm_q4k_v4", "gemm_xs_v4",
                          "gemm_q5k_wm", "gemm_q4k_wm", "gemm_q6k_wm", "gemm_xs_wm"],
                    ),
                    (
                        CO_ODD,
                        "LLM170_CO3_PATH",
                        include_bytes!("co/odd_all.co"),
                        &["gemm_nl_v4", "gemm_q3k_v4", "gemm_iq3s_v4"],
                    ),
                    (
                        CO_J128,
                        "LLM170_CO_PATH",
                        include_bytes!("co/w32b.co"),
                        &["gemm_q5k_j128", "gemm_q4k_j128", "gemm_q6k_j128",
                          "gemm_xs_j128", "gemm_q8_j128"],
                    ),
                ];
                for (bit, env_key, embedded, names) in slots {
                    let bytes: Vec<u8> = match std::env::var_os(env_key) {
                        Some(p) => std::fs::read(&p)
                            .map_err(|e| format!("{env_key} 읽기({p:?}): {e}"))?,
                        None => embedded.to_vec(),
                    };
                    let mut m: hip::hipModule_t = std::ptr::null_mut();
                    ck(hip::hipModuleLoadData(&mut m, bytes.as_ptr() as *const _),
                       &format!("{env_key} ModuleLoadData"))?;
                    let mut loaded = 0u8;
                    for name in *names {
                        let cname = CString::new(*name).unwrap();
                        let mut f: hip::hipFunction_t = std::ptr::null_mut();
                        if hip::hipModuleGetFunction(&mut f, m, cname.as_ptr())
                            == hip::hipError_t_hipSuccess
                        {
                            fns.insert(name, f);
                            loaded |= bit;
                        }
                    }
                    CO_FAM.fetch_or(loaded, std::sync::atomic::Ordering::Relaxed);
                }
            }

            let mut stream: hip::hipStream_t = std::ptr::null_mut();
            ck(hip::hipStreamCreate(&mut stream), "StreamCreate")?;
            let mut stream2: hip::hipStream_t = std::ptr::null_mut();
            ck(hip::hipStreamCreate(&mut stream2), "StreamCreate2")?;
            Ok(RawCtx { module, fns, stream, stream2, scratch: std::sync::Mutex::new(HashMap::new()), cursors: std::sync::Mutex::new(HashMap::new()) })
        }
    }

    /// 스크래시 획득 — 같은 크기는 항상 슬롯 0 재사용 (스트림 순서가
    /// 이전 사용 완료를 보장 — 단일 스트림). 호출마다 신규 할당은
    /// 메모리 고갈→illegal address (2026-09-03 RCA).
    pub fn scratch(&self, bytes: usize) -> Result<*mut u8, String> {
        let mut sc = self.scratch.lock().map_err(|e| e.to_string())?;
        let v = sc.entry(bytes).or_default();
        if v.is_empty() {
            let p = self.alloc(bytes)?;
            v.push(p);
        }
        Ok(v[0])
    }

    pub fn scratch_rewind(&self) {}

    /// 영속 디바이스 할당 (해제 없음).
    pub fn alloc(&self, bytes: usize) -> Result<*mut u8, String> {
        let mut p: *mut std::ffi::c_void = std::ptr::null_mut();
        unsafe { ck(hip::hipMalloc(&mut p, bytes), "hipMalloc")?; }
        Ok(p as *mut u8)
    }

    pub fn h2d(&self, dst: *mut u8, src: &[u8]) -> Result<(), String> {
        unsafe {
            ck(hip::hipMemcpyAsync(dst as *mut _, src.as_ptr() as *const _, src.len(), hip::hipMemcpyKind_hipMemcpyHostToDevice, self.stream), "h2d")?;
            self.sync()
        }
    }

    pub fn d2h(&self, dst: &mut [u8], src: *const u8) -> Result<(), String> {
        unsafe {
            ck(hip::hipMemcpyAsync(dst.as_mut_ptr() as *mut _, src as *const _, dst.len(), hip::hipMemcpyKind_hipMemcpyDeviceToHost, self.stream), "d2h")?;
            self.sync()
        }
    }

    pub fn sync(&self) -> Result<(), String> {
        unsafe { ck(hip::hipStreamSynchronize(self.stream), "sync") }
    }

    /// 커널 런치 — args는 각 인자 값에 대한 포인터 배열 (호출자 슬롯 유지).
    #[allow(clippy::too_many_arguments)]
    pub fn launch(
        &self,
        name: &str,
        gx: u32,
        gy: u32,
        block: u32,
        args: &mut [*mut std::ffi::c_void],
    ) -> Result<(), String> {
        let f = *self.fns.get(name).ok_or_else(|| format!("커널 없음: {name}"))?;
        unsafe {
            ck(hip::hipModuleLaunchKernel(f, gx, gy, 1, block, 1, 1, 0, self.stream, args.as_mut_ptr(), std::ptr::null_mut()), "launch").map_err(|e| format!("{e} kern={name} gx={gx} blk={block}"))?;
        }
        Ok(())
    }

    pub fn gemv_iq3s_sub4(&self, xq: *const u8, w: *const u8, n_in: usize, sub: usize) -> Result<[f64; 8], String> {
        let part = self.scratch(64)?;
        let mut xq_p = xq as *mut std::ffi::c_void;
        let mut w_p = w as *mut std::ffi::c_void;
        let mut part_p = part as *mut std::ffi::c_void;
        let mut ni = n_in as i32;
        let mut sb = sub as i64;
        let mut args = vec![
            &mut xq_p as *mut _ as *mut std::ffi::c_void,
            &mut w_p as *mut _ as *mut std::ffi::c_void,
            &mut part_p as *mut _ as *mut std::ffi::c_void,
            &mut ni as *mut _ as *mut std::ffi::c_void,
            &mut sb as *mut _ as *mut std::ffi::c_void,
        ];
        self.launch("gemm_iq3s_sub", 1, 1, 1, &mut args)?;
        self.sync()?;
        let mut r = [0f64; 8];
        self.d2h(bytemuck::cast_slice_mut(&mut r).as_mut(), part)?;
        Ok(r)
    }

    /// 3차원 그리드 런치 (qsa용 — gy 추가).
    #[allow(clippy::too_many_arguments)]
    pub fn launch3(
        &self,
        name: &str,
        gx: u32,
        gy: u32,
        gz: u32,
        block: u32,
        args: &mut [*mut std::ffi::c_void],
    ) -> Result<(), String> {
        let f = *self.fns.get(name).ok_or_else(|| format!("커널 없음: {name}"))?;
        unsafe {
            ck(hip::hipModuleLaunchKernel(f, gx, gy, gz, block, 1, 1, 0, self.stream, args.as_mut_ptr(), std::ptr::null_mut()), "launch3").map_err(|e| format!("{e} kern={name} gx={gx} gy={gy} gz={gz} blk={block}"))?;
        }
        Ok(())
    }

    /// 사이드 스트림 발사 (비동기 — join2로 합류)
    pub fn launch3s(
        &self,
        name: &str,
        gx: u32,
        gy: u32,
        gz: u32,
        block: u32,
        args: &mut [*mut std::ffi::c_void],
    ) -> Result<(), String> {
        let f = *self.fns.get(name).ok_or_else(|| format!("커널 없음: {name}"))?;
        unsafe {
            ck(hip::hipModuleLaunchKernel(f, gx, gy, gz, block, 1, 1, 0, self.stream2, args.as_mut_ptr(), std::ptr::null_mut()), "launch3s")?;
        }
        Ok(())
    }
    /// 사이드 스트림 → 주 스트림 합류: 이벤트 경유
    pub fn join2(&self) -> Result<(), String> {
        unsafe {
            let mut ev: hip::hipEvent_t = std::ptr::null_mut();
            ck(hip::hipEventCreateWithFlags(&mut ev, 0), "evCreate")?;
            ck(hip::hipEventRecord(ev, self.stream2), "evRecord")?;
            ck(hip::hipStreamWaitEvent(self.stream, ev, 0), "evWait")?;
            ck(hip::hipEventDestroy(ev), "evDestroy")?;
        }
        Ok(())
    }
    /// 주 스트림 현재 시점 → 사이드 대기 (사이드 입력 준비 경합 방지)
    pub fn side_wait_main(&self) -> Result<(), String> {
        unsafe {
            let mut ev: hip::hipEvent_t = std::ptr::null_mut();
            ck(hip::hipEventCreateWithFlags(&mut ev, 0), "evCreate2")?;
            ck(hip::hipEventRecord(ev, self.stream), "evRecord2")?;
            ck(hip::hipStreamWaitEvent(self.stream2, ev, 0), "evWait2")?;
            ck(hip::hipEventDestroy(ev), "evDestroy2")?;
        }
        Ok(())
    }

    /// W4A8 t=1 GEMV — 타입별 커널 선택, 부분합 reduce까지 수행.
    /// 반환 [n_out] f32. 수치: dot_row_w4a8_*_lane 미러와 동일열.
    pub fn gemv_q8(
        &self,
        xq: *const u8,
        w: *const u8,
        ktab2: *const u8,
        ty: u32,
        n_in: usize,
        n_out: usize,
    ) -> Result<Vec<f32>, String> {
        let part = self.scratch(n_out * 64 * 8)?;
        let out = self.scratch(n_out * 4)?;
        let gy = n_out.min(65535) as u32;
        let gz = n_out.div_ceil(65535) as u32;
        let kern = match ty {
            23 => "gemm_xs",   // iq4_xs
            13 => "gemm_q5k",  // q5_K
            8 => "gemm_q8_0",  // q8_0
            12 => "gemm_q4k",  // q4_K
            14 => "gemm_q6k",  // q6_K
            20 => "gemm_nl",   // iq4_nl
            11 => "gemm_q3k",  // q3_K
            21 => "gemm_iq3s", // iq3_s
            _ => return Err(format!("미지원 타입 {ty}")),
        };
        let mut xq_p = xq as *mut std::ffi::c_void;
        let mut w_p = w as *mut std::ffi::c_void;
        let mut part_p = part as *mut std::ffi::c_void;
        let mut kt_p = ktab2 as *mut std::ffi::c_void;
        let mut n_in_a = n_in as i32;
        let mut n_out_a = n_out as i32;
        let mut gx_a = 1i32;
        let mut args_v: Vec<*mut std::ffi::c_void> = match ty {
            23 | 20 => vec![
                &mut xq_p as *mut _ as *mut std::ffi::c_void,
                &mut w_p as *mut _ as *mut std::ffi::c_void,
                &mut part_p as *mut _ as *mut std::ffi::c_void,
                &mut kt_p as *mut _ as *mut std::ffi::c_void,
                &mut n_in_a as *mut _ as *mut std::ffi::c_void,
                &mut n_out_a as *mut _ as *mut std::ffi::c_void,
            ],
            _ => vec![
                &mut xq_p as *mut _ as *mut std::ffi::c_void,
                &mut w_p as *mut _ as *mut std::ffi::c_void,
                &mut part_p as *mut _ as *mut std::ffi::c_void,
                &mut n_in_a as *mut _ as *mut std::ffi::c_void,
                &mut n_out_a as *mut _ as *mut std::ffi::c_void,
            ],
        };
        let _ = &mut gx_a;
        let mut out_p0 = out as *mut std::ffi::c_void;
        match ty {
            23 | 20 => args_v.insert(4, &mut out_p0 as *mut _ as *mut std::ffi::c_void),
            _ => args_v.insert(3, &mut out_p0 as *mut _ as *mut std::ffi::c_void),
        }
        let mut xw_a = (n_in / 4 + n_in / 32 + n_in / 16) as i32;
        args_v.push(&mut xw_a as *mut _ as *mut std::ffi::c_void);
        self.launch3(kern, 1, gy, gz, 64, &mut args_v)?;
        let mut res = vec![0f32; n_out];
        self.sync()?;
        self.d2h(bytemuck::cast_slice_mut(&mut res).as_mut(), out as *const u8)?;
        Ok(res)
    }

    /// W4A8 GEMV — reduce 결과를 상주 out에 직접 기록 (왕복 제거).
    /// 수치는 gemv_q8과 동일열 (동일 커널·reduce).
    #[allow(clippy::too_many_arguments)]
    pub fn gemv_q8_out(
        &self,
        xq: *const u8,
        w: *const u8,
        ktab2: *const u8,
        ty: u32,
        n_in: usize,
        n_out: usize,
        out: *mut u8,
        xq_w: usize,
        t: usize,
    ) -> Result<(), String> {
        let part = self.scratch(n_out * 64 * 8)?;
        let gy = n_out.min(65535) as u32;
        let gz = n_out.div_ceil(65535) as u32;
        let kern = match ty {
            23 => "gemm_xs",
            13 => "gemm_q5k",
            8 => "gemm_q8_0",
            12 => "gemm_q4k",
            14 => "gemm_q6k",
            20 => "gemm_nl",
            11 => "gemm_q3k",
            21 => "gemm_iq3s",
            _ => return Err(format!("미지원 타입 {ty}")),
        };
        let mut xq_p = xq as *mut std::ffi::c_void;
        let mut w_p = w as *mut std::ffi::c_void;
        let mut part_p = part as *mut std::ffi::c_void;
        let mut kt_p = ktab2 as *mut std::ffi::c_void;
        let mut n_in_a = n_in as i32;
        let mut n_out_a = n_out as i32;
        let mut args_v: Vec<*mut std::ffi::c_void> = match ty {
            23 | 20 => vec![
                &mut xq_p as *mut _ as *mut std::ffi::c_void,
                &mut w_p as *mut _ as *mut std::ffi::c_void,
                &mut part_p as *mut _ as *mut std::ffi::c_void,
                &mut kt_p as *mut _ as *mut std::ffi::c_void,
                &mut n_in_a as *mut _ as *mut std::ffi::c_void,
                &mut n_out_a as *mut _ as *mut std::ffi::c_void,
            ],
            _ => vec![
                &mut xq_p as *mut _ as *mut std::ffi::c_void,
                &mut w_p as *mut _ as *mut std::ffi::c_void,
                &mut part_p as *mut _ as *mut std::ffi::c_void,
                &mut n_in_a as *mut _ as *mut std::ffi::c_void,
                &mut n_out_a as *mut _ as *mut std::ffi::c_void,
            ],
        };
        let gz = n_out.div_ceil(65535) as u32;
        let mut out_p0 = out as *mut std::ffi::c_void;
        match ty {
            23 | 20 => args_v.insert(4, &mut out_p0 as *mut _ as *mut std::ffi::c_void),
            _ => args_v.insert(3, &mut out_p0 as *mut _ as *mut std::ffi::c_void),
        }
        let mut xw_a = xq_w as i32;
        let xw_ptr = &mut xw_a as *mut _ as *mut std::ffi::c_void;
        args_v.push(xw_ptr);
        self.launch3(kern, t as u32, gy, gz, 64, &mut args_v)?;
        Ok(())
    }

    fn tile_core(&self, xq: *const u8, w: *const u8, ktab2: *const u8, ty: u32, n_in: usize, n_out: usize, xq_w: usize, t: usize, out: *mut u8) -> Result<TileLaunch, String> {
        let j128 = std::env::var_os("LLM170_EXACT").is_none()
            && co_loaded(CO_J128) && t > 64;
        self.tile_core_inner(xq, w, ktab2, ty, n_in, n_out, xq_w, t, out, j128)
    }

    /// head 강제판 — j128 타일을 t≤64에서도 (n_out 초대형일 때 이득).
    fn tile_core_head(&self, xq: *const u8, w: *const u8, ktab2: *const u8, ty: u32, n_in: usize, n_out: usize, xq_w: usize, t: usize, out: *mut u8) -> Result<TileLaunch, String> {
        let j128 = std::env::var_os("LLM170_EXACT").is_none()
            && co_loaded(CO_J128);
        self.tile_core_inner(xq, w, ktab2, ty, n_in, n_out, xq_w, t, out, j128)
    }

    fn tile_core_inner(&self, xq: *const u8, w: *const u8, ktab2: *const u8, ty: u32, n_in: usize, n_out: usize, xq_w: usize, t: usize, out: *mut u8, j128: bool) -> Result<TileLaunch, String> {
        // wm·mm 상한 64: t>64 무CO는 유효 커널 없음 — 침묵 오답 대신 에러
        if t > 64 && !j128 {
            return Err(format!("타일 미지원: t={t}는 CO 사전컴파일(j128/v4) 필요"));
        }
        let (v4, odd) = (co_loaded(CO_V4), co_loaded(CO_ODD));
        let kern: &'static str = match ty {
            13 => if j128 && v4 { "gemm_q5k_v4" } else if j128 { "gemm_q5k_j128" } else if std::env::var_os("LLM170_EXACT").is_none() && t >= 32 { "gemm_q5k_wm" } else { "gemm_q5k_mm" },
            12 => if j128 && v4 { "gemm_q4k_v4" } else if j128 { "gemm_q4k_j128" } else if std::env::var_os("LLM170_EXACT").is_none() && t >= 32 { "gemm_q4k_wm" } else { "gemm_q4k_mm" },
            14 => if j128 { "gemm_q6k_j128" } else if std::env::var_os("LLM170_EXACT").is_none() && t >= 32 { "gemm_q6k_wm" } else { "gemm_q6k_mm" },
            23 => if j128 { "gemm_xs_j128" } else if v4 && std::env::var_os("LLM170_EXACT").is_none() && t >= 32 { "gemm_xs_v4" } else if std::env::var_os("LLM170_EXACT").is_none() && t >= 32 { "gemm_xs_wm" } else { "gemm_xs_mm" },
            20 => if odd && std::env::var_os("LLM170_EXACT").is_none() && t >= 32 { "gemm_nl_v4" } else { return Err("타일 미지원 타입 20 (GEMV 경로 사용)".into()) },
            11 => if odd && std::env::var_os("LLM170_EXACT").is_none() && t >= 32 { "gemm_q3k_v4" } else { return Err("타일 미지원 타입 11 (GEMV 경로 사용)".into()) },
            21 => if odd && std::env::var_os("LLM170_EXACT").is_none() && t >= 32 { "gemm_iq3s_v4" } else { return Err("타일 미지원 타입 21 (GEMV 경로 사용)".into()) },
            8 => if j128 { "gemm_q8_j128" } else { return Err("타일 미지원 타입 8 (GEMV 경로 사용)".into()) },
            _ => return Err(format!("타일 미지원 타입 {ty}")),
        };
        let mm = kern.ends_with("_mm") || kern.ends_with("_wm") || kern.ends_with("_j128") || kern.ends_with("_v4");
        let rows_per_block: usize = if kern.ends_with("_j128") || kern.ends_with("_v4") { 128 } else if mm { 64 } else { 1 };
        let nblocks = n_out.div_ceil(rows_per_block);
        Ok(TileLaunch {
            kern,
            xp: xq as *mut std::ffi::c_void,
            wp: w as *mut std::ffi::c_void,
            op: out as *mut std::ffi::c_void,
            ktp: ktab2 as *mut std::ffi::c_void,
            ni: n_in as i32,
            no: n_out as i32,
            xw: xq_w as i32,
            tt: t as i32,
            gx: nblocks.min(65535) as u32,
            gz: nblocks.div_ceil(65535) as u32,
            block: if mm { 256 } else { 64 },
            ktab: ty == 23 || kern == "gemm_nl_v4",
        })
    }

    fn tile_args(l: &mut TileLaunch) -> Vec<*mut std::ffi::c_void> {
        let mut args: Vec<*mut std::ffi::c_void> = vec![
            (&mut l.xp) as *mut _ as *mut std::ffi::c_void,
            (&mut l.wp) as *mut _ as *mut std::ffi::c_void,
            (&mut l.op) as *mut _ as *mut std::ffi::c_void,
        ];
        if l.ktab {
            args.push((&mut l.ktp) as *mut _ as *mut std::ffi::c_void);
        }
        args.push((&mut l.ni) as *mut _ as *mut std::ffi::c_void);
        args.push((&mut l.no) as *mut _ as *mut std::ffi::c_void);
        args.push((&mut l.xw) as *mut _ as *mut std::ffi::c_void);
        args.push((&mut l.tt) as *mut _ as *mut std::ffi::c_void);
        args
    }

    /// spec verify head 전용 — j128/v4 타일 강제 (가중 1회 독서).
    /// 산술은 동일 W4A8이나 환원 순서가 mm 계열와 달라 스트림 비트계약 대상 아님
    /// (spec 내부 draft↔verify 일관성만 요구).
    pub fn gemm_tile_head(&self, xq: *const u8, w: *const u8, ktab2: *const u8, ty: u32, n_in: usize, n_out: usize, xq_w: usize, t: usize, out: *mut u8) -> Result<(), String> {
        let mut l = self.tile_core_head(xq, w, ktab2, ty, n_in, n_out, xq_w, t, out)?;
        let mut args = Self::tile_args(&mut l);
        self.launch3(l.kern, l.gx, 1, l.gz, l.block, &mut args)
    }

    pub fn gemm_tile(&self, xq: *const u8, w: *const u8, ktab2: *const u8, ty: u32, n_in: usize, n_out: usize, xq_w: usize, t: usize, out: *mut u8) -> Result<(), String> {
        let mut l = self.tile_core(xq, w, ktab2, ty, n_in, n_out, xq_w, t, out)?;
        let mut args = Self::tile_args(&mut l);
        self.launch3(l.kern, l.gx, 1, l.gz, l.block, &mut args)
    }

    /// gemm_tile의 사이드 스트림판 — 커널 선택·인자 구성은 공용 코어에 위임.
    pub fn gemm_tile_s(&self, xq: *const u8, w: *const u8, ktab2: *const u8, ty: u32, n_in: usize, n_out: usize, xq_w: usize, t: usize, out: *mut u8) -> Result<(), String> {
        let mut l = self.tile_core(xq, w, ktab2, ty, n_in, n_out, xq_w, t, out)?;
        let mut args = Self::tile_args(&mut l);
        self.launch3s(l.kern, l.gx, 1, l.gz, l.block, &mut args)
    }

    /// 버퍼 센티널 기입 (디버그) — 미기록 판별.
    pub fn fill_f32(&self, dst: *mut u8, n: usize, v: f32) -> Result<(), String> {
        let vs = vec![v; n];
        self.h2d(dst, bytemuck::cast_slice(&vs))
    }

    /// 활성 양자화 — quantize_row_q8_ref 비트 미러. 출력은 xq 하나:
    /// [0..n/4) 워드 + [n/4..n/4+n/32) d 비트(u32 편승 — 저장 경로 단일화).
    /// 버퍼 크기 (n/4 + n/32)·4 바이트 필요.
    pub fn quant_q8(&self, x: *const u8, xq: *mut u8, n: usize) -> Result<(), String> {
        self.quant_q8_b(x, xq, n, n / 4 + n / 32 + n / 16, 1)
    }

    /// 배치 양자화 — t토큰 [t][n] → [t][xq_w 워드].
    pub fn quant_q8_b(&self, x: *const u8, xq: *mut u8, n: usize, xq_w: usize, t: usize) -> Result<(), String> {
        let nblk = n / 32;
        let mut x_p = x as *mut std::ffi::c_void;
        let mut xq_p = xq as *mut std::ffi::c_void;
        let mut n_a = n as i32;
        let mut xw = xq_w as i32;
        let mut args = vec![
            &mut x_p as *mut _ as *mut std::ffi::c_void,
            &mut xq_p as *mut _ as *mut std::ffi::c_void,
            &mut n_a as *mut _ as *mut std::ffi::c_void,
            &mut xw as *mut _ as *mut std::ffi::c_void,
        ];
        self.launch3("quant_q8", nblk.div_ceil(64) as u32, t as u32, 1, 64, &mut args)
    }
}

/// 스파이크 벤치 — 개별 원시 런치 오버헤드 측정 (quant_q8 커널 재사용).
pub fn raw_probe(iters: usize) -> Result<String, String> {
    let ctx = RawCtx::new()?;
    let n = 1usize << 16;
    let x = ctx.alloc(n * 4)?;
    let xh: Vec<f32> = vec![0.5; n];
    ctx.h2d(x, bytemuck::cast_slice(&xh))?;
    let nblk = n / 32;
    let gx = nblk.div_ceil(64) as u32;
    let mut xq = ctx.alloc((n / 4 + n / 32 + n / 16) * 4)?;
    let mut xp = x as *mut std::ffi::c_void;
    let mut xqp = xq as *mut std::ffi::c_void;
    let mut na = n as i32;
    let mut args = vec![
        &mut xp as *mut _ as *mut std::ffi::c_void,
        &mut xqp as *mut _ as *mut std::ffi::c_void,
        &mut na as *mut _ as *mut std::ffi::c_void,
    ];
    // 워밍
    ctx.launch("quant_q8", gx, 1, 64, &mut args)?;
    ctx.sync()?;
    let t0 = std::time::Instant::now();
    for _ in 0..iters {
        ctx.launch("quant_q8", gx, 1, 64, &mut args)?;
    }
    ctx.sync()?;
    let dt = t0.elapsed();
    Ok(format!(
        "원시 런치 {}회 = {:.2}µs/런치 (cubecl 경로 ~10.7µs — 실행기 정상)",
        iters,
        dt.as_secs_f64() * 1e6 / iters as f64
    ))
}

/// qk_norm_rope 단독 검증 — 디코드와 동일 파라미터.
pub fn qk_check() -> Result<String, String> {
    let ctx = RawCtx::new()?;
    let n_head = 24usize;
    let n_kv = 4usize;
    let hd = 256usize;
    let n_rot = 64usize;
    let half = n_rot / 2;
    let pos = 2usize;
    let rows = n_head + n_kv;
    let aq = ctx.alloc(n_head * 2 * hd * 4)?;
    let ak = ctx.alloc(n_kv * hd * 4)?;
    let qw: Vec<f32> = (0..n_head * hd).map(|i| 1.0 + (i % 7) as f32 * 0.01).collect();
    let kw: Vec<f32> = (0..n_kv * hd).map(|i| 1.0 + (i % 5) as f32 * 0.01).collect();
    let qw_d = ctx.alloc(qw.len() * 4)?;
    let kw_d = ctx.alloc(kw.len() * 4)?;
    ctx.h2d(qw_d, bytemuck::cast_slice(&qw))?;
    ctx.h2d(kw_d, bytemuck::cast_slice(&kw))?;
    let hq: Vec<f32> = (0..n_head * 2 * hd).map(|i| ((i as i32 % 11) as f32 - 5.0) * 0.1).collect();
    let hk: Vec<f32> = (0..n_kv * hd).map(|i| ((i as i32 % 13) as f32 - 6.0) * 0.1).collect();
    ctx.h2d(aq, bytemuck::cast_slice(&hq))?;
    ctx.h2d(ak, bytemuck::cast_slice(&hk))?;
    let cs: Vec<f32> = (0..2048 * half * 2).map(|i| ((i % 9) as f32 - 4.0) * 0.1).collect();
    let cs_d = ctx.alloc(cs.len() * 4)?;
    ctx.h2d(cs_d, bytemuck::cast_slice(&cs))?;
    let eps = 1e-5f32;
    let kqs = 0.05f32;
    let mut qp = aq as *mut std::ffi::c_void;
    let mut kp = ak as *mut std::ffi::c_void;
    let mut qwp = qw_d as *mut std::ffi::c_void;
    let mut kwp = kw_d as *mut std::ffi::c_void;
    let mut csp = cs_d as *mut std::ffi::c_void;
    let mut e = eps;
    let mut k = kqs;
    let mut posv = pos as i32;
    let mut nh = n_head as i32;
    let mut nk = n_kv as i32;
    let mut h = hd as i32;
    let mut nr = n_rot as i32;
    fn pp<T>(v: &mut T) -> *mut std::ffi::c_void {
        v as *mut T as *mut std::ffi::c_void
    }
    let mut args = vec![
        pp(&mut qp), pp(&mut kp), pp(&mut qwp), pp(&mut kwp), pp(&mut csp), pp(&mut e),
        pp(&mut k), pp(&mut posv), pp(&mut nh), pp(&mut nk), pp(&mut h), pp(&mut nr),
    ];
    ctx.launch("qk_norm_rope", rows as u32, 1, 32, &mut args)?;
    ctx.sync()?;
    let mut oq = vec![0f32; n_head * 2 * hd];
    ctx.d2h(bytemuck::cast_slice_mut(&mut oq).as_mut(), aq)?;
    Ok(format!("qk_check ok: q[0]={:.4} q[511]={:.4} q[512]={:.4} k[0]={:.4}", oq[0], oq[511], oq[512], {
        let mut ok = vec![0f32; n_kv * hd];
        ctx.d2h(bytemuck::cast_slice_mut(&mut ok).as_mut(), ak)?;
        ok[0]
    }))
}

/// iq3s 1블록 프로브 — 커널 part[64] vs 미러 lane[64] 레인별 비교.
pub fn iq3s_probe() -> Result<String, String> {
    let ctx = RawCtx::new()?;
    let k = 256usize * 68;
    let nblk = k / 256;
    // 결정적 데이터 — 블록별 변화 + 음수 d 포함
    let mut wbytes = vec![0u8; 110 * nblk];
    for b in 0..nblk {
        for i in 0..110 {
            wbytes[b * 110 + i] = ((i * 37 + 11 + b * 13) % 251) as u8;
        }
        if b % 3 == 1 { wbytes[b * 110 + 1] |= 0x80; } // 음수 d
    }
    wbytes[0] = 0x38; wbytes[1] = 0x53;
    let w_d = ctx.alloc(110 * nblk)?;
    ctx.h2d(w_d, &wbytes)?;
    // x 양자화
    let x: Vec<f32> = (0..k).map(|i| ((i as i32 % 17) as f32 - 8.0) * 0.25).collect();
    let xq = ctx.alloc((k / 4 + k / 32) * 4)?;
    let blocks = llm170_core::quant::quantize_row_q8_ref(&x);
    let mut xq_host: Vec<u32> = Vec::with_capacity(k / 4 + k / 32);
    for b in &blocks {
        for c in 0..8 {
            let w = (b.qs[c * 4] as u32 & 0xFF)
                | ((b.qs[c * 4 + 1] as u32 & 0xFF) << 8)
                | ((b.qs[c * 4 + 2] as u32 & 0xFF) << 16)
                | ((b.qs[c * 4 + 3] as u32 & 0xFF) << 24);
            xq_host.push(w);
        }
    }
    for b in &blocks {
        xq_host.push(b.d.to_bits());
    }
    ctx.h2d(xq, bytemuck::cast_slice(&xq_host))?;
    let part = ctx.alloc(64 * 8)?;
    let mut w_p = w_d as *mut std::ffi::c_void;
    let mut xq_p = xq as *mut std::ffi::c_void;
    let mut part_p = part as *mut std::ffi::c_void;
    let mut ni = k as i32;
    let mut no = 1i32;
    let _ = &no;
    let mut args = vec![
        (&mut xq_p) as *mut _ as *mut std::ffi::c_void,
        (&mut w_p) as *mut _ as *mut std::ffi::c_void,
        (&mut part_p) as *mut _ as *mut std::ffi::c_void,
        (&mut ni) as *mut _ as *mut std::ffi::c_void,
        (&mut no) as *mut _ as *mut std::ffi::c_void,
    ];
    ctx.launch("gemm_iq3s", 1, 1, 64, &mut args)?;
    ctx.sync()?;
    let mut p64 = vec![0f64; 64];
    ctx.d2h(bytemuck::cast_slice_mut(&mut p64).as_mut(), part)?;
    // 미러 — y를 Q8Block으로 재구성
    let y = blocks;
    let lane = llm170_core::quant::dot_row_w4a8_iq3s_lane_parts(&wbytes, k as u64, &y);
    let mut bad = 0;
    let mut msg = String::new();
    for l in 0..64 {
        if p64[l].to_bits() != lane[l].to_bits() {
            bad += 1;
            if bad <= 4 {
                msg += &format!("lane {l}: gpu={:.6e} cpu={:.6e}\n", p64[l], lane[l]);
            }
        }
    }
    Ok(format!("iq3s_probe: {bad}/64 lanes differ\n{msg}"))
}


/// 디버그: expf vs Rust exp 비트 비교.
pub fn exp_ab() -> Result<String, String> {
    let ctx = RawCtx::new()?;
    let n = 4096usize;
    let x: Vec<f32> = (0..n).map(|i| (i as f32 - 2048.0) / 97.0).collect();
    let xd = ctx.alloc(n * 4)?;
    let bits = ctx.alloc(n * 4)?;
    ctx.h2d(xd, bytemuck::cast_slice(&x))?;
    let mut xp = xd as *mut std::ffi::c_void;
    let mut bp = bits as *mut std::ffi::c_void;
    let mut na = n as i32;
    let mut args = vec![
        (&mut xp) as *mut _ as *mut std::ffi::c_void,
        (&mut bp) as *mut _ as *mut std::ffi::c_void,
        (&mut na) as *mut _ as *mut std::ffi::c_void,
    ];
    ctx.launch("exp_probe", n.div_ceil(64) as u32, 1, 64, &mut args)?;
    ctx.sync()?;
    let mut gbits = vec![0u32; n];
    ctx.d2h(bytemuck::cast_slice_mut(&mut gbits).as_mut(), bits)?;
    let mut bad = 0;
    let mut msg = String::new();
    for i in 0..n {
        let host = x[i].exp().to_bits();
        if host != gbits[i] {
            bad += 1;
            if bad <= 3 {
                msg += &format!("x={:.6e} dev={:#010x} host={:#010x}\n", x[i], gbits[i], host);
            }
        }
    }
    Ok(format!("exp_ab: {bad}/{n} differ\n{msg}"))
}

/// dp4a 가용성 테스트.
pub fn dp4a_test() -> Result<String, String> {
    let ctx = RawCtx::new()?;
    let x: Vec<u32> = vec![0x12, 0x11, 0x00010000, 0x00050000];
    let xd = ctx.alloc(16)?;
    let od = ctx.alloc(32)?;
    ctx.h2d(xd, bytemuck::cast_slice(&x))?;
    let mut xp = xd as *mut std::ffi::c_void;
    let mut op = od as *mut std::ffi::c_void;
    let mut args = vec![
        (&mut xp) as *mut _ as *mut std::ffi::c_void,
        (&mut op) as *mut _ as *mut std::ffi::c_void,
    ];
    ctx.launch("dp4a_probe", 1, 1, 1, &mut args)?;
    ctx.sync()?;
    let mut r = [0i32; 8];
    ctx.d2h(bytemuck::cast_slice_mut(&mut r).as_mut(), od)?;
    Ok(format!("sdot8={} (3) udot8={} (5) sdot2={} (204) sdot4={} (204) neg={} (-4) lit_mix={} (-538) bc_mix={} (-538) bc_acc={} (462)", r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7]))
}

/// 대역폭 상한 프로브 — q5_K ffn_gate 형상 [5120→17408, 176B] 재현.
pub fn bw_test() -> Result<String, String> {
    let ctx = RawCtx::new()?;
    let (n_in, n_out, bsize) = (5120usize, 17408usize, 176usize);
    let bytes = n_out * (n_in / 256) * bsize;
    let w = ctx.alloc(bytes)?;
    let part = ctx.scratch(n_out * 64 * 8)?;
    let mut wp = w as *mut std::ffi::c_void;
    let mut pp = part as *mut std::ffi::c_void;
    let mut ni = n_in as i32;
    let mut no = n_out as i32;
    let mut bs = bsize as i32;
    let mut args = vec![
        (&mut wp) as *mut _ as *mut std::ffi::c_void,
        (&mut pp) as *mut _ as *mut std::ffi::c_void,
        (&mut ni) as *mut _ as *mut std::ffi::c_void,
        (&mut no) as *mut _ as *mut std::ffi::c_void,
        (&mut bs) as *mut _ as *mut std::ffi::c_void,
    ];
    // 워밍
    ctx.launch("bw_probe", 17408, 1, 64, &mut args)?;
    ctx.sync()?;
    let reps = 30;
    let t0 = std::time::Instant::now();
    for _ in 0..reps {
        ctx.launch("bw_probe", 17408, 1, 64, &mut args)?;
    }
    ctx.sync()?;
    let dt = t0.elapsed().as_secs_f64() / reps as f64;
    let mut r = vec![0f64; 64];
    ctx.d2h(bytemuck::cast_slice_mut(&mut r).as_mut(), part)?;
    let _ = r[0];
    Ok(format!("bw_probe: {:.1}us -> {:.0} GB/s (checksum={})", dt * 1e6, bytes as f64 / dt / 1e9, r[63] as u32))
}

/// q6_K old/new isum A/B — blk.11.attn_k (q6_K) 실데이터.
pub fn q6k_ab_test() -> Result<String, String> {
    use std::io::Read;
    // gguf 직접 파싱 대신 llm170_core로 로드
    let args: Vec<String> = std::env::args().collect();
    let path = args.get(2).cloned().unwrap_or_else(|| "/home/yoon/models/qwen3.8-27b/q35work.gguf".into());
    let tname = args.get(3).cloned().unwrap_or_else(|| "blk.11.attn_k.weight".into());
    let model = llm170_core::model::Model::load(std::path::Path::new(&path)).map_err(|e| e.to_string())?;
    let w = model.w(&tname).ok_or("tensor 없음")?;
    let ctx = RawCtx::new()?;
    let wd = ctx.alloc(w.data.len())?;
    ctx.h2d(wd, w.data)?;
    let n_in = w.n_in as usize;
    // 임의 x: q8 양자화
    let mut seed = 0x9e3779b9u64;
    let mut lcg = || { seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); (seed >> 33) as f32 / 2147483648.0 - 0.5 };
    let x: Vec<f32> = (0..n_in).map(|_| lcg()).collect();
    let blocks_q = llm170_core::quant::quantize_row_q8_ref(&x);
    let mut xq_h: Vec<u32> = Vec::new();
    for b in &blocks_q {
        for c in 0..8 {
            xq_h.push((b.qs[c*4] as u32 & 0xFF) | ((b.qs[c*4+1] as u32 & 0xFF) << 8) | ((b.qs[c*4+2] as u32 & 0xFF) << 16) | ((b.qs[c*4+3] as u32 & 0xFF) << 24));
        }
    }
    for b in &blocks_q { xq_h.push(b.d.to_bits()); }
    let xqd = ctx.alloc(xq_h.len() * 4)?;
    ctx.h2d(xqd, bytemuck::cast_slice(&xq_h))?;
    let part = ctx.alloc(32)?;
    let mut msg = String::new();
    // row 0, 여러 그룹 시험
    for g in [0usize, 1, 5, 8, 17, 40, 100, 200] {
        let mut wp = wd as *mut std::ffi::c_void;
        let mut xp = xqd as *mut std::ffi::c_void;
        let mut pp = part as *mut std::ffi::c_void;
        let mut r0 = 0i32;
        let mut gi = g as i32;
        // 커널 wb 계산이 n_in=256 가정이므로 row_base를 0으로 — g>>4 블록 인덱스는 행 내 오프셋으로 유효
        let mut args = vec![
            (&mut wp) as *mut _ as *mut std::ffi::c_void,
            (&mut xp) as *mut _ as *mut std::ffi::c_void,
            (&mut pp) as *mut _ as *mut std::ffi::c_void,
            (&mut r0) as *mut _ as *mut std::ffi::c_void,
            (&mut gi) as *mut _ as *mut std::ffi::c_void,
        ];
        ctx.launch("q6k_ab", 1, 1, 1, &mut args)?;
        ctx.sync()?;
        let mut r = [0f64; 4];
        ctx.d2h(bytemuck::cast_slice_mut(&mut r).as_mut(), part)?;
        msg += &format!("g={}: old={} dot4={} packed={} src={} al={} old==dot4:{} old==packed:{}\n", g, r[0] as i64, r[1] as i64, r[2] as i64, r[2] as i64, r[3] as i64, r[0]==r[1], r[0]==r[2]);
    }
    Ok(msg)
}

/// 트리 환원 순서 A/B — GPU 셔플 vs Rust tree64.
pub fn tree_test() -> Result<String, String> {
    let ctx = RawCtx::new()?;
    let od = ctx.alloc(64)?;
    let mut op = od as *mut std::ffi::c_void;
    let mut args = vec![(&mut op) as *mut _ as *mut std::ffi::c_void];
    ctx.launch("tree_probe", 1, 1, 64, &mut args)?;
    ctx.sync()?;
    let mut r = [0f64; 5];
    ctx.d2h(bytemuck::cast_slice_mut(&mut r).as_mut(), od)?;
    Ok(format!("off32={} (32) off1={} (1) tree={} (2016) w64_off32={} (32) w64_tree={} (2016)", r[0], r[1], r[2], r[3], r[4]))
}

/// 배치 A/B — t=2 quant+gemv가 행별 단일 결과와 동일한지.
pub fn batch_ab_test() -> Result<String, String> {
    let args: Vec<String> = std::env::args().collect();
    let path = args.get(2).cloned().unwrap_or_else(|| "/home/yoon/models/qwen3.8-27b/q35work.gguf".into());
    let tname = args.get(3).cloned().unwrap_or_else(|| "blk.0.attn_gate.weight".into());
    let model = llm170_core::model::Model::load(std::path::Path::new(&path)).map_err(|e| e.to_string())?;
    let w = model.w(&tname).ok_or("tensor 없음")?;
    let ctx = RawCtx::new()?;
    let n_in = w.n_in as usize;
    let n_out = w.n_out as usize;
    let wd = ctx.alloc(w.data.len())?;
    ctx.h2d(wd, w.data)?;
    let ktab2: Vec<u32> = (0..256u32)
        .map(|b| {
            let lo = llm170_core::KVALUES_IQ4NL[(b & 0xF) as usize] as u8 as u32;
            let hi = llm170_core::KVALUES_IQ4NL[(b >> 4) as usize] as u8 as u32;
            lo | (hi << 8)
        })
        .collect();
    let kt_d = ctx.alloc(1024)?;
    ctx.h2d(kt_d, bytemuck::cast_slice(&ktab2))?;
    // 서로 다른 x 2행
    let mut seed = 0x9e3779b9u64;
    let mut lcg = || { seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); (seed >> 33) as f32 / 2147483648.0 - 0.5 };
    let x0: Vec<f32> = (0..n_in).map(|_| lcg()).collect();
    let x1: Vec<f32> = (0..n_in).map(|_| lcg()).collect();
    // 단일 경로 결과 (기준)
    let mut base = Vec::new();
    for xr in [&x0, &x1] {
        let xd = ctx.alloc(n_in * 4)?;
        ctx.h2d(xd, bytemuck::cast_slice(xr))?;
        let xq = ctx.alloc((n_in / 4 + n_in / 32 + n_in / 16) * 4)?;
        ctx.quant_q8(xd, xq, n_in)?;
        let out = ctx.alloc(n_out * 4)?;
        ctx.gemv_q8_out(xq, wd, kt_d, w.ty as u32, n_in, n_out, out, n_in / 4 + n_in / 32 + n_in / 16, 1)?;
        let mut o = vec![0f32; n_out];
        ctx.sync()?;
        ctx.d2h(bytemuck::cast_slice_mut(&mut o).as_mut(), out)?;
        base.push(o);
    }
    // 배치 경로
    let mut xall: Vec<f32> = x0.clone();
    xall.extend_from_slice(&x1);
    let xd = ctx.alloc(n_in * 2 * 4)?;
    ctx.h2d(xd, bytemuck::cast_slice(&xall))?;
    let xq_w = n_in / 4 + n_in / 32 + n_in / 16;
    let xq = ctx.alloc(xq_w * 4 * 2)?;
    ctx.quant_q8_b(xd, xq, n_in, xq_w, 2)?;
    let out = ctx.alloc(n_out * 4 * 2)?;
    ctx.gemv_q8_out(xq, wd, kt_d, w.ty as u32, n_in, n_out, out, xq_w, 2)?;
    ctx.sync()?;
    let mut o2 = vec![0f32; n_out * 2];
    ctx.d2h(bytemuck::cast_slice_mut(&mut o2).as_mut(), out)?;
    // quant y=1 영역 검사
    let mut xqh = vec![0u32; xq_w * 2];
    ctx.d2h(bytemuck::cast_slice_mut(&mut xqh).as_mut(), xq)?;
    let nzq1 = xqh[xq_w..].iter().filter(|v| **v != 0).count();
    let mut xq1 = vec![0u32; xq_w];
    ctx.d2h(bytemuck::cast_slice_mut(&mut xq1).as_mut(), unsafe { xq.add(xq_w * 4) } as *const u8)?;
    eprintln!("diag: quant y1 nonzero {nzq1}/{xq_w}");
    // 진단: xq_w=0 — y=1이 row0 값을 복사하면 블록 실행·아웃오프셋 정상
    let out3 = ctx.alloc(n_out * 4)?;
    ctx.gemv_q8_out(xq, wd, kt_d, w.ty as u32, n_in, n_out, out3, 0, 2)?;
    ctx.sync()?;
    let mut o3 = vec![0f32; n_out];
    ctx.d2h(bytemuck::cast_slice_mut(&mut o3).as_mut(), out3)?;
    let nz1 = o2[n_out..].iter().filter(|v| **v != 0.0).count();
    let cp = o3.iter().zip(&o2[..n_out]).filter(|(a, b)| a.to_bits() == b.to_bits()).count();
    Ok(format!("batch t=2: row0 {} row1 {} 일치 | xqw0=단일 out≠0: {} / o2row1 비영: {}", base[0].iter().zip(&o2[..n_out]).filter(|(a, b)| a.to_bits() == b.to_bits()).count(), base[1].iter().zip(&o2[n_out..]).filter(|(a, b)| a.to_bits() == b.to_bits()).count(), n_out - cp, nz1))
}

/// 배치 mm 타이밍 — gy=1 대비 gy=t 배율.
pub fn mm_batch_bench() -> Result<String, String> {
    let args: Vec<String> = std::env::args().collect();
    let path = args.get(2).cloned().unwrap_or_else(|| "/home/yoon/models/qwen3.8-27b/q35work.gguf".into());
    let tname = args.get(3).cloned().unwrap_or_else(|| "blk.0.attn_gate.weight".into());
    let model = llm170_core::model::Model::load(std::path::Path::new(&path)).map_err(|e| e.to_string())?;
    let w = model.w(&tname).ok_or("tensor 없음")?;
    let ctx = RawCtx::new()?;
    let n_in = w.n_in as usize;
    let n_out = w.n_out as usize;
    let wd = ctx.alloc(w.data.len())?;
    ctx.h2d(wd, w.data)?;
    let ktab2: Vec<u32> = (0..256u32).map(|b| {
        let lo = llm170_core::KVALUES_IQ4NL[(b & 0xF) as usize] as u8 as u32;
        let hi = llm170_core::KVALUES_IQ4NL[(b >> 4) as usize] as u8 as u32;
        lo | (hi << 8)
    }).collect();
    let kt_d = ctx.alloc(1024)?;
    ctx.h2d(kt_d, bytemuck::cast_slice(&ktab2))?;
    let xq_w = n_in / 4 + n_in / 32 + n_in / 16;
    let mut seed = 0x9e3779b9u64;
    let mut lcg = || { seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); (seed >> 33) as f32 / 2147483648.0 - 0.5 };
    let x: Vec<f32> = (0..n_in * 64).map(|_| lcg()).collect();
    let xd = ctx.alloc(n_in * 64 * 4)?;
    ctx.h2d(xd, bytemuck::cast_slice(&x))?;
    let xq = ctx.alloc(xq_w * 4 * 64)?;
    ctx.quant_q8_b(xd, xq, n_in, xq_w, 64)?;
    let out = ctx.alloc(n_out * 4 * 64)?;
    let mut msg = String::new();
    for &t in &[1usize, 8, 64] {
        ctx.gemv_q8_out(xq, wd, kt_d, w.ty as u32, n_in, n_out, out, xq_w, t)?;
        ctx.sync()?;
        let reps = 5;
        let t0 = std::time::Instant::now();
        for _ in 0..reps {
            ctx.gemv_q8_out(xq, wd, kt_d, w.ty as u32, n_in, n_out, out, xq_w, t)?;
        }
        ctx.sync()?;
        let dt = t0.elapsed().as_secs_f64() / reps as f64;
        msg += &format!("t={}: {:.3}ms ({:.0} GB/s-equiv)\n", t, dt * 1e3, w.data.len() as f64 / dt / 1e9);
    }
    Ok(msg)
}

/// 타일 커널 검증+타이밍 — gemm_q5k_bt vs 미러.
pub fn mm_tile_bench() -> Result<String, String> {
    let args: Vec<String> = std::env::args().collect();
    let path = args.get(2).cloned().unwrap_or_else(|| "/home/yoon/models/qwen3.8-27b/q35work.gguf".into());
    let tname = args.get(3).cloned().unwrap_or_else(|| "blk.0.attn_gate.weight".into());
    let model = llm170_core::model::Model::load(std::path::Path::new(&path)).map_err(|e| e.to_string())?;
    let w = model.w(&tname).ok_or("tensor 없음")?;
    let ctx = RawCtx::new()?;
    let n_in = w.n_in as usize;
    let n_out = w.n_out as usize;
    let wd = ctx.alloc(w.data.len())?;
    ctx.h2d(wd, w.data)?;
    let ktab2: Vec<u32> = (0..256u32).map(|b| {
        let lo = llm170_core::KVALUES_IQ4NL[(b & 0xF) as usize] as u8 as u32;
        let hi = llm170_core::KVALUES_IQ4NL[(b >> 4) as usize] as u8 as u32;
        lo | (hi << 8)
    }).collect();
    let kt_d = ctx.alloc(1024)?;
    ctx.h2d(kt_d, bytemuck::cast_slice(&ktab2))?;
    let xq_w = n_in / 4 + n_in / 32 + n_in / 16;
    let mut seed = 0x9e3779b9u64;
    let mut lcg = || { seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); (seed >> 33) as f32 / 2147483648.0 - 0.5 };
    let t = std::env::var("LLM170_TILE_T").ok().and_then(|v| v.parse().ok()).unwrap_or(16usize);
    let mut xs = Vec::new();
    let mut q8s = Vec::new();
    for _ in 0..t {
        let x: Vec<f32> = (0..n_in).map(|_| lcg()).collect();
        q8s.push(llm170_core::quant::quantize_row_q8_ref(&x));
        xs.push(x);
    }
    let mut xq_h: Vec<u32> = Vec::new();
    for tok in &q8s {
        for blk in tok {
            for c in 0..8 {
                let base = c * 4;
                xq_h.push((blk.qs[base] as u32 & 0xFF) | ((blk.qs[base+1] as u32 & 0xFF) << 8) | ((blk.qs[base+2] as u32 & 0xFF) << 16) | ((blk.qs[base+3] as u32 & 0xFF) << 24));
            }
        }
        for blk in tok { xq_h.push(blk.d.to_bits()); }
        for blk in tok {
            let s0: i32 = blk.qs[..16].iter().map(|&v| v as i32).sum();
            let s1: i32 = blk.qs[16..].iter().map(|&v| v as i32).sum();
            xq_h.push(s0 as u32);
            xq_h.push(s1 as u32);
        }
    }
    let xq = ctx.alloc(xq_h.len() * 4)?;
    if let Some(f) = std::env::var_os("LLM170_XQN_FILE") {
        let bytes = std::fs::read(&f).unwrap();
        assert_eq!(bytes.len(), xq_h.len() * 4, "덤프 크기 불일치: {} vs {}", bytes.len(), xq_h.len() * 4);
        ctx.h2d(xq, &bytes)?;
        eprintln!("xq 리플레이: {}", f.to_string_lossy());
    } else {
        ctx.h2d(xq, bytemuck::cast_slice(&xq_h))?;
    }
    let out = ctx.alloc(n_out * 4 * t)?;
    let mut wp = wd as *mut std::ffi::c_void;
    let mut op = out as *mut std::ffi::c_void;
    let mut xp = xq as *mut std::ffi::c_void;
    let mut ni = n_in as i32;
    let mut no = n_out as i32;
    let mut xw = xq_w as i32;
    let mut tt = t as i32;
    ctx.gemm_tile(xq, wd, kt_d, w.ty as u32, n_in, n_out, xq_w, t, out)?;
    ctx.sync()?;
    let mut o = vec![0f32; n_out * t];
    ctx.d2h(bytemuck::cast_slice_mut(&mut o).as_mut(), out)?;
    // 미러 검증
    let blck = w.ty.blck_size() as usize;
    let bsize = w.ty.type_size() as usize;
    let rb = (n_in / blck) * bsize;
    let mut mism = 0;
    let mut first_dbg = String::new();
    for ti in 0..t {
        for oo in 0..n_out.min(256) {
            let row = &w.data[oo * rb..];
            let c = match w.ty {
                llm170_gguf::GgmlType::Q5K => llm170_core::quant::dot_row_w4a8_q5k_lane(row, n_in as u64, &q8s[ti]),
                llm170_gguf::GgmlType::Q4K => llm170_core::quant::dot_row_w4a8_q4k_lane(row, n_in as u64, &q8s[ti]),
                llm170_gguf::GgmlType::Q6K => llm170_core::quant::dot_row_w4a8_q6k_lane(row, n_in as u64, &q8s[ti]),
                _ => llm170_core::quant::dot_row_w4a8_iq4xs_lane(row, n_in as u64, &q8s[ti]),
            };
            if c.to_bits() != o[ti * n_out + oo].to_bits() {
                mism += 1;
                if mism == 1 {
                    first_dbg = format!("ti={ti} o={oo}: cpu={c:.7e} gpu={:.7e}", o[ti * n_out + oo]);
                }
            }
        }
    }
    eprintln!("dbg: {first_dbg}");
    // 타이밍
    let reps = 10;
    let t0 = std::time::Instant::now();
    for _ in 0..reps {
        ctx.gemm_tile(xq, wd, kt_d, w.ty as u32, n_in, n_out, xq_w, t, out)?;
    }
    ctx.sync()?;
    let dt = t0.elapsed().as_secs_f64() / reps as f64;
    Ok(format!("tile t={t}: 불일치 {mism}/{} (첫 256행×t) — {:.3}ms → {:.0} GB/s-equiv, 토큰당 {:.1}µs", n_out.min(256) * t, dt * 1e3, w.data.len() as f64 / dt / 1e9, dt * 1e6 / t as f64))
}

/// dot4 루프-오버헤드 루프 프로브 — 모드별 유효 TIOPS.
pub fn roof_test() -> Result<String, String> {
    let ctx = RawCtx::new()?;
    let n_in = 5120usize;
    let xq = ctx.alloc(n_in * 4)?;
    let w = ctx.alloc(n_in * 4)?;
    let out = ctx.alloc(16)?;
    let data: Vec<u32> = (0..n_in).map(|i| (i as u32).wrapping_mul(2654435761)).collect();
    ctx.h2d(xq, bytemuck::cast_slice(&data))?;
    ctx.h2d(w, bytemuck::cast_slice(&data))?;
    let mut msg = String::new();
    for &mode in &[0usize, 1, 2] {
        let iters = 20000usize;
        let mut xp = xq as *mut std::ffi::c_void;
        let mut wp = w as *mut std::ffi::c_void;
        let mut op = out as *mut std::ffi::c_void;
        let mut m = mode as i32;
        let mut it = iters as i32;
        let mut ni = n_in as i32;
        let mut args = vec![
            (&mut xp) as *mut _ as *mut std::ffi::c_void,
            (&mut wp) as *mut _ as *mut std::ffi::c_void,
            (&mut op) as *mut _ as *mut std::ffi::c_void,
            (&mut m) as *mut _ as *mut std::ffi::c_void,
            (&mut it) as *mut _ as *mut std::ffi::c_void,
            (&mut ni) as *mut _ as *mut std::ffi::c_void,
        ];
        // grid: 40CU 채우도록 640블록×64스레드
        ctx.launch3("dot_roof", 640, 1, 1, 64, &mut args)?;
        ctx.sync()?;
        let reps = 20;
        let t0 = std::time::Instant::now();
        for _ in 0..reps {
            ctx.launch3("dot_roof", 640, 1, 1, 64, &mut args)?;
        }
        ctx.sync()?;
        let dt = t0.elapsed().as_secs_f64() / reps as f64;
        let total_dots = iters as f64 * 640.0 * 64.0;
        let tips = total_dots * 4.0 / dt / 1e12; // MAC 4개/dot
        msg += &format!("mode{mode} ({}): {:.2}ms → {:.2} TIOPS MAC\n",
            ["reg-chain", "same-addr load", "stride load"][mode], dt * 1e3, tips);
    }
    // mfma 발행률 — rocwmma 16x16x16f32 (8192 FLOP/wave/mma)
    {
        let ntiles = 64usize;
        let ah = vec![0x3c00u16; ntiles * 256];
        let ad = ctx.alloc(ah.len() * 2)?;
        let bd = ctx.alloc(ah.len() * 2)?;
        ctx.h2d(ad, bytemuck::cast_slice(&ah))?;
        ctx.h2d(bd, bytemuck::cast_slice(&ah))?;
        let om = ctx.alloc(24)?;
        for &mode in &[0usize, 1] {
            let iters = 20000usize;
            let mut ap = ad as *mut std::ffi::c_void;
            let mut bp = bd as *mut std::ffi::c_void;
            let mut op = om as *mut std::ffi::c_void;
            let mut m = mode as i32;
            let mut it = iters as i32;
            let mut nn = ntiles as i32;
            let mut args = vec![
                (&mut ap) as *mut _ as *mut std::ffi::c_void,
                (&mut bp) as *mut _ as *mut std::ffi::c_void,
                (&mut op) as *mut _ as *mut std::ffi::c_void,
                (&mut m) as *mut _ as *mut std::ffi::c_void,
                (&mut it) as *mut _ as *mut std::ffi::c_void,
                (&mut nn) as *mut _ as *mut std::ffi::c_void,
            ];
            ctx.launch3("mfma_roof", 640, 1, 1, 64, &mut args)?;
            ctx.sync()?;
            let reps = 20;
            let t0 = std::time::Instant::now();
            for _ in 0..reps {
                ctx.launch3("mfma_roof", 640, 1, 1, 64, &mut args)?;
            }
            ctx.sync()?;
            let dt = t0.elapsed().as_secs_f64() / reps as f64;
            let mut o3 = [0f64; 3];
            ctx.d2h(bytemuck::cast_slice_mut(&mut o3).as_mut(), om)?;
            let wavesize = o3[2];
            let waves = 640.0 * 64.0 / wavesize;
            let tflops = waves * iters as f64 * 8192.0 / dt / 1e12;
            msg += &format!("mfma{mode} ({} wave{}): {:.2}ms → {:.2} TFLOPS f32\n",
                ["reg-resident", "L1-fed"][mode], wavesize, dt * 1e3, tflops);
        }
    }
    Ok(msg)
}



/// MMQ 포트 A/B — bt vs mm (각 미러).
pub fn mm_bench() -> Result<String, String> {
    let args: Vec<String> = std::env::args().collect();
    let path = args.get(2).cloned().unwrap_or_else(|| "/home/yoon/models/qwen3.8-27b/q35work.gguf".into());
    let tname = args.get(3).cloned().unwrap_or_else(|| "blk.0.attn_gate.weight".into());
    let model = llm170_core::model::Model::load(std::path::Path::new(&path)).map_err(|e| e.to_string())?;
    let w = model.w(&tname).ok_or("tensor 없음")?;
    let ctx = RawCtx::new()?;
    let n_in = w.n_in as usize;
    let n_out = w.n_out as usize;
    let wd = ctx.alloc(w.data.len())?;
    ctx.h2d(wd, w.data)?;
    let ktab2: Vec<u32> = (0..256u32).map(|b| {
        let lo = llm170_core::KVALUES_IQ4NL[(b & 0xF) as usize] as u8 as u32;
        let hi = llm170_core::KVALUES_IQ4NL[(b >> 4) as usize] as u8 as u32;
        lo | (hi << 8)
    }).collect();
    let kt_d = ctx.alloc(1024)?;
    ctx.h2d(kt_d, bytemuck::cast_slice(&ktab2))?;
    let mut seed = 0x9e3779b9u64;
    let mut lcg = || { seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); (seed >> 33) as f32 / 2147483648.0 - 0.5 };
    let t = std::env::var("LLM170_MM_T").ok().and_then(|v| v.parse().ok()).unwrap_or(16usize);
    let mut q8s = Vec::new();
    let mut xq_h: Vec<u32> = Vec::new();
    for _ in 0..t {
        let x: Vec<f32> = (0..n_in).map(|_| lcg()).collect();
        let blocks = llm170_core::quant::quantize_row_q8_ref(&x);
        for blk in &blocks {
            for c in 0..8 {
                let b = c * 4;
                xq_h.push((blk.qs[b] as u32 & 0xFF) | ((blk.qs[b+1] as u32 & 0xFF) << 8) | ((blk.qs[b+2] as u32 & 0xFF) << 16) | ((blk.qs[b+3] as u32 & 0xFF) << 24));
            }
        }
        for blk in &blocks { xq_h.push(blk.d.to_bits()); }
        for blk in &blocks {
            let s0: i32 = blk.qs[..16].iter().map(|&v| v as i32).sum();
            let s1: i32 = blk.qs[16..].iter().map(|&v| v as i32).sum();
            xq_h.push(s0 as u32);
            xq_h.push(s1 as u32);
        }
        q8s.push(blocks);
    }
    let xq_w = n_in / 4 + n_in / 32 + n_in / 16;
    let xq = ctx.alloc(xq_h.len() * 4)?;
    ctx.h2d(xq, bytemuck::cast_slice(&xq_h))?;
    let out = ctx.alloc(n_out * 4 * t)?;
    // wm 상한 64 + bench GEMV 그리드 부적합: 128-커널 계열 미로드면 에러
    let (v4, j128f, odd) = (co_loaded(CO_V4), co_loaded(CO_J128), co_loaded(CO_ODD));
    let big_ok = match w.ty {
        llm170_gguf::GgmlType::Q5K | llm170_gguf::GgmlType::Q4K | llm170_gguf::GgmlType::Iq4Xs
            => v4 || j128f,
        llm170_gguf::GgmlType::Q6K | llm170_gguf::GgmlType::Q8_0 => j128f,
        llm170_gguf::GgmlType::Iq4Nl | llm170_gguf::GgmlType::Q3K | llm170_gguf::GgmlType::Iq3S => odd,
        _ => true,
    };
    if t > 64 && !big_ok {
        return Err(format!("mm-bench 미지원: t={t}는 타입별 128-커널 필요"));
    }
    let kern_name = match w.ty {
        llm170_gguf::GgmlType::Q5K => if v4 { "gemm_q5k_v4" }
            else if j128f { "gemm_q5k_j128" }
            else if std::env::var_os("LLM170_EXACT").is_none() { "gemm_q5k_wm" } else { "gemm_q5k_mm" },
        llm170_gguf::GgmlType::Q4K => if v4 { "gemm_q4k_v4" }
            else if j128f { "gemm_q4k_j128" } else if std::env::var_os("LLM170_EXACT").is_none() { "gemm_q4k_wm" } else { "gemm_q4k_mm" },
        llm170_gguf::GgmlType::Q6K => if j128f { "gemm_q6k_j128" } else if std::env::var_os("LLM170_EXACT").is_none() { "gemm_q6k_wm" } else { "gemm_q6k_mm" },
        llm170_gguf::GgmlType::Q8_0 => if j128f { "gemm_q8_j128" } else { return Err("mm-bench 미지원: q8_0은 j128 커널 필요".into()) },
        llm170_gguf::GgmlType::Iq4Xs => if v4 { "gemm_xs_v4" }
            else if j128f { "gemm_xs_j128" }
            else if std::env::var_os("LLM170_EXACT").is_none() { "gemm_xs_wm" } else { "gemm_xs_mm" },
        llm170_gguf::GgmlType::Iq4Nl => if odd { "gemm_nl_v4" }
            else { return Err("mm-bench 미지원: iq4_nl 타일은 odd CO 필요".into()) },
        llm170_gguf::GgmlType::Q3K => if odd { "gemm_q3k_v4" }
            else { return Err("mm-bench 미지원: q3_K 타일은 odd CO 필요".into()) },
        llm170_gguf::GgmlType::Iq3S => if odd { "gemm_iq3s_v4" }
            else { return Err("mm-bench 미지원: iq3_s 타일은 odd CO 필요".into()) },
        _ => "gemm_xs_mm",
    };

    let launch = |ctx: &RawCtx| -> Result<(), String> {
        let mut xp = xq as *mut std::ffi::c_void;
        let mut wp = wd as *mut std::ffi::c_void;
        let mut op = out as *mut std::ffi::c_void;
        let mut ktp = kt_d as *mut std::ffi::c_void;
        let mut ni = n_in as i32;
        let mut no = n_out as i32;
        let mut xw = xq_w as i32;
        let mut tt = t as i32;
        let mut args: Vec<*mut std::ffi::c_void> = vec![
            (&mut xp) as *mut _ as *mut std::ffi::c_void,
            (&mut wp) as *mut _ as *mut std::ffi::c_void,
            (&mut op) as *mut _ as *mut std::ffi::c_void,
        ];
        if kern_name == "gemm_xs_mm" || kern_name == "gemm_xs_wm" || kern_name == "gemm_xs_j128" || kern_name == "gemm_xs_v4" || kern_name == "gemm_nl_v4" {
            args.push((&mut ktp) as *mut _ as *mut std::ffi::c_void);
        }
        args.push((&mut ni) as *mut _ as *mut std::ffi::c_void);
        args.push((&mut no) as *mut _ as *mut std::ffi::c_void);
        args.push((&mut xw) as *mut _ as *mut std::ffi::c_void);
        args.push((&mut tt) as *mut _ as *mut std::ffi::c_void);
        let rpb = if kern_name.ends_with("_j128") || kern_name.ends_with("_v4") { 128 } else { 64 };
        let gx = n_out.div_ceil(rpb).min(65535) as u32;
        let gz = n_out.div_ceil(rpb).div_ceil(65535) as u32;
        let gz = n_out.div_ceil(64).div_ceil(65535) as u32;
        ctx.launch3(kern_name, gx, 1, gz, 256, &mut args)
    };
    launch(&ctx)?;
    ctx.sync()?;
    let mut o2 = vec![0f32; n_out * t];
    ctx.d2h(bytemuck::cast_slice_mut(&mut o2).as_mut(), out)?;
    let reps = 20;
    let t0 = std::time::Instant::now();
    for _ in 0..reps { launch(&ctx)?; }
    ctx.sync()?;
    let dt2 = t0.elapsed().as_secs_f64() / reps as f64;
    let blck = w.ty.blck_size() as usize;
    let bsize = w.ty.type_size() as usize;
    let rb = (n_in / blck) * bsize;
    let mut m2 = 0usize;
    let mut maxrel = 0f32;
    for ti in 0..t {
        for oo in 0..n_out.min(256) {
            let row = &w.data[oo * rb..];
            let c2 = match w.ty {
                llm170_gguf::GgmlType::Q5K => llm170_core::quant::dot_row_w4a8_q5k_mm(row, n_in as u64, &q8s[ti]),
                llm170_gguf::GgmlType::Q4K => llm170_core::quant::dot_row_w4a8_q4k_mm(row, n_in as u64, &q8s[ti]),
                llm170_gguf::GgmlType::Q6K => llm170_core::quant::dot_row_w4a8_q6k_mm(row, n_in as u64, &q8s[ti]),
                llm170_gguf::GgmlType::Iq4Nl => llm170_core::quant::dot_row_w4a8_iq4nl_lane(row, n_in as u64, &q8s[ti]),
                llm170_gguf::GgmlType::Q3K => llm170_core::quant::dot_row_w4a8_q3k_lane(row, n_in as u64, &q8s[ti]),
                llm170_gguf::GgmlType::Iq3S => llm170_core::quant::dot_row_w4a8_iq3s_lane(row, n_in as u64, &q8s[ti]),
                llm170_gguf::GgmlType::Q8_0 => {
                    let nblk = n_in as usize / 32;
                    let mut acc = 0.0f32;
                    for b in 0..nblk {
                        let wb = &row[b * 34..b * 34 + 34];
                        let h = ((wb[1] as u16) << 8) | wb[0] as u16;
                        let sign = if h & 0x8000 != 0 { -1.0f32 } else { 1.0 };
                        let exp = ((h >> 10) & 0x1F) as i32;
                        let man = (h & 0x3FF) as f32;
                        let d = if exp == 0 { sign * man * 2f32.powi(-24) } else { sign * (man / 1024.0 + 1.0) * 2f32.powi(exp - 15) };
                        let mut isum = 0i64;
                        for j in 0..32 {
                            let wv = wb[2 + j] as i8 as i64;
                            let yv = q8s[ti][b].qs[j] as i64;
                            isum += wv * yv;
                        }
                        let yd = q8s[ti][b].d;
                        acc += yd * d * isum as f32;
                    }
                    acc
                }
                _ => llm170_core::quant::dot_row_w4a8_iq4xs_mm(row, n_in as u64, &q8s[ti]),
            };
            if kern_name.ends_with("_wm") || kern_name.ends_with("_w32") || kern_name.ends_with("_j128") || kern_name.ends_with("_v4") {
                let g = o2[ti * n_out + oo];
                let denom = c2.abs().max(1.0);
                let rel = (g - c2).abs() / denom;
                if rel > maxrel { maxrel = rel; }
                if rel > 5e-3 { m2 += 1; }
            } else if c2.to_bits() != o2[ti * n_out + oo].to_bits() { m2 += 1; }
        }
    }
    Ok(format!("mm({kern_name}): {:.3}ms ({:.1}us/tok) mism {m2} maxrel {maxrel:.2e}", dt2 * 1e3, dt2 * 1e6 / t as f64))
}

/// 텐서 차원 출력 (디버그 보조)
pub fn dims_of(path: &str, names: &[&str]) -> String {
    let g = match llm170_gguf::GgufFile::open(std::path::Path::new(path)) {
        Ok(g) => g, Err(e) => return e.to_string(),
    };
    let mut s = String::new();
    for n in names {
        if let Some(t) = g.tensors.iter().find(|t| t.name == *n) {
            s += &format!("{n} ne={:?} ty={:?}\n", t.ne, t.ty);
        }
    }
    s
}
