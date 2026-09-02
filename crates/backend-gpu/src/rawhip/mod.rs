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
    /// 크기별 스크래치 풀 — 해제 없는 재사용 (호출마다 신규 할당이
    /// 메모리 고갈→illegal address 유발, 2026-09-03 RCA).
    scratch: std::sync::Mutex<HashMap<usize, Vec<*mut u8>>>,
    cursors: std::sync::Mutex<HashMap<usize, usize>>,
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
            // FMA 수축 차단 — CPU 비트계약 (a+=b*c 축약이 비트 불일치,
            // 2026-09-03 AR xor RCA)
            let o4 = CString::new("-ffp-contract=off").unwrap();
            let mut opts = vec![o1.as_ptr(), o2.as_ptr(), o3.as_ptr(), o4.as_ptr()];
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
            Ok(RawCtx { module, fns, stream, scratch: std::sync::Mutex::new(HashMap::new()), cursors: std::sync::Mutex::new(HashMap::new()) })
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
            ck(hip::hipModuleLaunchKernel(f, gx, gy, 1, block, 1, 1, 0, self.stream, args.as_mut_ptr(), std::ptr::null_mut()), "launch")?;
        }
        Ok(())
    }

    /// 디버그: 첫 행 part[64] 반환 (gemv_q8과 동일 경로).
    pub fn gemv_q8_parts(
        &self,
        xq: *const u8,
        w: *const u8,
        ktab2: *const u8,
        ty: u32,
        n_in: usize,
    ) -> Result<Vec<f64>, String> {
        let n_out = 1usize;
        let kern = match ty {
            21 => "gemm_iq3s",
            _ => return Err("gemv_q8_parts: iq3s 전용".into()),
        };
        let part = self.scratch(n_out * 64 * 8)?;
        let out = self.scratch(n_out * 4)?;
        let mut xq_p = xq as *mut std::ffi::c_void;
        let mut w_p = w as *mut std::ffi::c_void;
        let mut part_p = part as *mut std::ffi::c_void;
        let mut kt_p = ktab2 as *mut std::ffi::c_void;
        let mut n_in_a = n_in as i32;
        let mut n_out_a = n_out as i32;
        let mut args_v: Vec<*mut std::ffi::c_void> = vec![
            &mut xq_p as *mut _ as *mut std::ffi::c_void,
            &mut w_p as *mut _ as *mut std::ffi::c_void,
            &mut part_p as *mut _ as *mut std::ffi::c_void,
            &mut n_in_a as *mut _ as *mut std::ffi::c_void,
            &mut n_out_a as *mut _ as *mut std::ffi::c_void,
        ];
        let gx = 1u32;
        unsafe {
            ck(hip::hipModuleLaunchKernel(
                *self.fns.get(kern).ok_or("커널 없음")?,
                gx, 1, 1, 64, 1, 1, 0, self.stream, args_v.as_mut_ptr(), std::ptr::null_mut(),
            ), "gemm")?;
        }
        let mut part_p2 = part as *mut std::ffi::c_void;
        let mut out_p = out as *mut std::ffi::c_void;
        let mut n_out_b = n_out as i32;
        let mut rargs = vec![
            &mut part_p2 as *mut _ as *mut std::ffi::c_void,
            &mut out_p as *mut _ as *mut std::ffi::c_void,
            &mut n_out_b as *mut _ as *mut std::ffi::c_void,
        ];
        self.launch("reduce64", n_out.div_ceil(64) as u32, 1, 64, &mut rargs)?;
        self.sync()?;
        let mut r = vec![0f64; 64];
        self.d2h(bytemuck::cast_slice_mut(&mut r).as_mut(), part)?;
        Ok(r)
    }


    /// 디버그: row0 특정 sub의 기여값.
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
            ck(hip::hipModuleLaunchKernel(f, gx, gy, gz, block, 1, 1, 0, self.stream, args.as_mut_ptr(), std::ptr::null_mut()), "launch3")?;
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
        let gx = n_out.min(65535) as u32;
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
    ) -> Result<(), String> {
        let part = self.scratch(n_out * 64 * 8)?;
        let gx = n_out.min(65535) as u32;
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
        self.launch(kern, gx, 1, 64, &mut args_v)?;
        let mut part_p2 = part as *mut std::ffi::c_void;
        let mut out_p = out as *mut std::ffi::c_void;
        let mut n_out_b = n_out as i32;
        let mut rargs = vec![
            &mut part_p2 as *mut _ as *mut std::ffi::c_void,
            &mut out_p as *mut _ as *mut std::ffi::c_void,
            &mut n_out_b as *mut _ as *mut std::ffi::c_void,
        ];
        self.launch("reduce64", n_out.div_ceil(64) as u32, 1, 64, &mut rargs)?;
        Ok(())
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
