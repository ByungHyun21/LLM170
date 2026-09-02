//! 원시 HIP 실행기 — cubecl을 거치지 않는 직접 경로 (2026-09-03 재작성).
//! hipRTC로 임베디드 HIP C++ 소스를 컴파일하고 hipModuleLaunchKernel로
//! 실행. 버퍼는 영속 아레나(해제 없음, ADR-0014 동일 규칙). 커널 산술은
//! core 미러(dot_row_w4a8_*_lane)와 동일 연산열 — to_bits 검증 게이트.

use cubecl_hip_sys as hip;
use std::collections::HashMap;
use std::ffi::CString;

pub mod kernels;
pub mod decode;

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
}

impl RawCtx {
    pub fn new() -> Result<Self, String> {
        unsafe {
            ck(hip::hipSetDevice(0), "hipSetDevice")?;
            let _ = hip::hipSetDeviceFlags(hip::hipDeviceScheduleSpin);
            // 컨텍스트 유지 — AmdDevice는 hanzo의 경로와 공유
            let _dev = hanzo_cubecl_hip::AmdDevice::new(0);

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
            let mut opts = vec![o1.as_ptr(), o2.as_ptr(), o3.as_ptr()];
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
            let mut module: hip::hipModule_t = std::ptr::null_mut();
            ck(hip::hipModuleLoadData(&mut module, code.as_ptr() as *const _), "ModuleLoadData")?;
            let mut fns = HashMap::new();
            for name in kernels::NAMES {
                let cname = CString::new(*name).unwrap();
                let mut f: hip::hipFunction_t = std::ptr::null_mut();
                ck(hip::hipModuleGetFunction(&mut f, module, cname.as_ptr()), "GetFunction")?;
                fns.insert(*name, f);
            }
            let mut stream: hip::hipStream_t = std::ptr::null_mut();
            ck(hip::hipStreamCreate(&mut stream), "StreamCreate")?;
            Ok(RawCtx { module, fns, stream })
        }
    }

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
            ck(hip::hipModuleLaunchKernel(f, gx, gy, 1, block, 1, 1, 0, self.stream, args.as_mut_ptr(), std::ptr::null_mut()), "launch")?;
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
        let part = self.alloc(n_out * 64 * 8)?;
        let out = self.alloc(n_out * 4)?;
        let gx = n_out.min(65535) as u32;
        let kern = match ty {
            23 => "gemm_xs",   // iq4_xs
            13 => "gemm_q5k",  // q5_K
            8 => "gemm_q8_0",  // q8_0
            12 => "gemm_q4k",  // q4_K
            14 => "gemm_q6k",  // q6_K
            20 => "gemm_nl",   // iq4_nl
            11 => "gemm_q3k",  // q3_K
            _ => return Err(format!("미지원 타입 {ty}")),
        };
        let mut xq_p = xq as *mut std::ffi::c_void;
        let mut w_p = w as *mut std::ffi::c_void;
        let mut part_p = part as *mut std::ffi::c_void;
        let mut kt_p = ktab2 as *mut std::ffi::c_void;
        let mut n_in_a = n_in as i32;
        let mut n_out_a = n_out as i32;
        let mut gx_a = gx as i32;
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
        self.launch(kern, gx, 1, 64, &mut args_v)?;
        // reduce: [n_out×64] f64 → [n_out] f32 (레인 순서 합, 1회 캐스트)
        let mut part_p2 = part as *mut std::ffi::c_void;
        let mut out_p = out as *mut std::ffi::c_void;
        let mut n_out_b = n_out as i32;
        let mut rargs = vec![
            &mut part_p2 as *mut _ as *mut std::ffi::c_void,
            &mut out_p as *mut _ as *mut std::ffi::c_void,
            &mut n_out_b as *mut _ as *mut std::ffi::c_void,
        ];
        self.launch("reduce64", n_out.div_ceil(64) as u32, 1, 64, &mut rargs)?;
        let mut res = vec![0f32; n_out];
        self.sync()?;
        self.d2h(bytemuck::cast_slice_mut(&mut res).as_mut(), out as *const u8)?;
        Ok(res)
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
        let nblk = n / 32;
        let mut x_p = x as *mut std::ffi::c_void;
        let mut xq_p = xq as *mut std::ffi::c_void;
        let mut n_a = n as i32;
        let mut args = vec![
            &mut x_p as *mut _ as *mut std::ffi::c_void,
            &mut xq_p as *mut _ as *mut std::ffi::c_void,
            &mut n_a as *mut _ as *mut std::ffi::c_void,
        ];
        self.launch("quant_q8", nblk.div_ceil(64) as u32, 1, 64, &mut args)
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
    let mut xq = ctx.alloc((n / 4 + n / 32) * 4)?;
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
