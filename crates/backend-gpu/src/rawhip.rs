//! 원시 HIP 스파이크 (2026-09-03) — hipRTC 직접 컴파일 + 개별 런치 vs
//! hipGraph 리플레이 벤치마크. cubecl을 거치지 않고 cubecl-hip-sys
//! 바인딩만 사용. 디코드 실행기 원시 HIP 재작성의 사전 검증.
//!
//! 실증 (2026-09-03, gfx1151):
//! - 개별 원시 런치 2.84µs (cubecl 경로 ~10.7µs — 3.8× 저렴)
//! - 그래프(512×2노드) 리플레이 2.2µs/노드 — 런치 오버헤드 소거,
//!   측정치는 자잘한 커널의 GPU 실행 바닥(~2.2µs)

use cubecl_hip_sys as hip;
use std::ffi::CString;

const SRC: &str = r#"
extern "C" __global__ void scale1(const float* x, float* y, float s, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = x[i] * s + 1.0f;
}
extern "C" __global__ void scale2(const float* x, float* y, float s, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = x[i] * s - 0.5f;
}
"#;

fn ck(status: hip::hipError_t, what: &str) -> Result<(), String> {
    if status == hip::hipError_t_hipSuccess {
        Ok(())
    } else {
        Err(format!("{what}: {status:?}"))
    }
}

/// ① hipRTC 컴파일 ② 개별 런치 N회 ③ 그래프 캡처·리플레이 — 비교 보고.
pub fn raw_probe(iters: usize) -> Result<String, String> {
    unsafe {
        // hanzo 런타임과 동일 — hipSetDevice가 주 컨텍스트 활성화
        ck(hip::hipSetDevice(0), "hipSetDevice")?;
        let _ = hip::hipSetDeviceFlags(hip::hipDeviceScheduleSpin);
        let _dev = hanzo_cubecl_hip::AmdDevice::new(0);

        // ① hipRTC 컴파일 (hanzo context.rs와 동일 옵션)
        let src = CString::new(SRC).unwrap();
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
            return Err(format!("컴파일 실패: {log}"));
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
        let k1n = CString::new("scale1").unwrap();
        let k2n = CString::new("scale2").unwrap();
        let mut k1: hip::hipFunction_t = std::ptr::null_mut();
        let mut k2: hip::hipFunction_t = std::ptr::null_mut();
        ck(hip::hipModuleGetFunction(&mut k1, module, k1n.as_ptr()), "GetFunction1")?;
        ck(hip::hipModuleGetFunction(&mut k2, module, k2n.as_ptr()), "GetFunction2")?;

        // ② 버퍼·스트림·인자 (스택 슬롯은 런치 시점에 값 복사 — 수명 안전)
        let n = 1usize << 16;
        let mut x: *mut f32 = std::ptr::null_mut();
        let mut y: *mut f32 = std::ptr::null_mut();
        ck(hip::hipMalloc(&mut x as *mut _ as *mut _, n * 4), "Malloc x")?;
        ck(hip::hipMalloc(&mut y as *mut _ as *mut _, n * 4), "Malloc y")?;
        let mut stream: hip::hipStream_t = std::ptr::null_mut();
        ck(hip::hipStreamCreate(&mut stream), "StreamCreate")?;
        let xh: Vec<f32> = vec![0.5; n];
        ck(hip::hipMemcpyAsync(x as *mut _, xh.as_ptr() as *const _, n * 4, hip::hipMemcpyKind_hipMemcpyHostToDevice, stream), "MemcpyH2D")?;
        ck(hip::hipStreamSynchronize(stream), "upload sync")?;

        let grid = ((n + 255) / 256) as u32;
        let block = 256u32;
        let mut one = 1.0f32;
        let mut nn_a = n as i32;
        let mut args: [*mut std::ffi::c_void; 4] = [
            &mut x as *mut *mut f32 as *mut _,
            &mut y as *mut *mut f32 as *mut _,
            &mut one as *mut f32 as *mut _,
            &mut nn_a as *mut i32 as *mut _,
        ];

        // 개별 런치 — (scale1; scale2) × pairs
        let pairs = iters.max(100);
        let t0 = std::time::Instant::now();
        for _ in 0..pairs {
            ck(hip::hipModuleLaunchKernel(k1, grid, 1, 1, block, 1, 1, 0, stream, args.as_mut_ptr(), std::ptr::null_mut()), "launch k1")?;
            ck(hip::hipModuleLaunchKernel(k2, grid, 1, 1, block, 1, 1, 0, stream, args.as_mut_ptr(), std::ptr::null_mut()), "launch k2")?;
        }
        ck(hip::hipStreamSynchronize(stream), "sync1")?;
        let indiv = t0.elapsed();

        // ③ 그래프: 체인 ×CHAIN회 캡처 → 리플레이 pairs 회
        const CHAIN: usize = 512;
        ck(hip::hipStreamBeginCapture(stream, hip::hipStreamCaptureMode_hipStreamCaptureModeGlobal), "BeginCapture")?;
        for _ in 0..CHAIN {
            ck(hip::hipModuleLaunchKernel(k1, grid, 1, 1, block, 1, 1, 0, stream, args.as_mut_ptr(), std::ptr::null_mut()), "cap k1")?;
            ck(hip::hipModuleLaunchKernel(k2, grid, 1, 1, block, 1, 1, 0, stream, args.as_mut_ptr(), std::ptr::null_mut()), "cap k2")?;
        }
        let mut graph: hip::hipGraph_t = std::ptr::null_mut();
        ck(hip::hipStreamEndCapture(stream, &mut graph), "EndCapture")?;
        let mut exec: hip::hipGraphExec_t = std::ptr::null_mut();
        ck(hip::hipGraphInstantiate(&mut exec, graph, std::ptr::null_mut(), std::ptr::null_mut(), 0), "Instantiate")?;
        ck(hip::hipGraphLaunch(exec, stream), "graph warm")?;
        ck(hip::hipStreamSynchronize(stream), "sync warm")?;
        let t1 = std::time::Instant::now();
        for _ in 0..pairs {
            ck(hip::hipGraphLaunch(exec, stream), "graph launch")?;
        }
        ck(hip::hipStreamSynchronize(stream), "sync2")?;
        let graph_t = t1.elapsed();

        // 검증: 두 커널 모두 x(0.5)를 읹고 y에 씀 — 마지막 scale2: y=0.5·1−0.5=0
        ck(hip::hipModuleLaunchKernel(k2, grid, 1, 1, block, 1, 1, 0, stream, args.as_mut_ptr(), std::ptr::null_mut()), "ver k2")?;
        ck(hip::hipStreamSynchronize(stream), "ver sync")?;
        let mut yh = vec![0f32; 2];
        ck(hip::hipMemcpy(yh.as_mut_ptr() as *mut _, y as *const _, 8, hip::hipMemcpyKind_hipMemcpyDeviceToHost), "MemcpyD2H")?;
        if yh[0].abs() > 1e-6 {
            return Err(format!("값 검증 실패: y[0]={:.6} (기대 0.0)", yh[0]));
        }

        let _ = hip::hipGraphExecDestroy(exec);
        let _ = hip::hipGraphDestroy(graph);
        let _ = hip::hipStreamDestroy(stream);
        let _ = hip::hipFree(x as *mut _);
        let _ = hip::hipFree(y as *mut _);

        Ok(format!(
            "원시 HIP 스파이크 OK: 개별 {:.2}µs/런치, 그래프({CHAIN}×2노드) {:.2}µs/리플레이 = {:.3}µs/노드 → 개별 대비 {:.1}×",
            indiv.as_secs_f64() * 1e6 / (pairs * 2) as f64,
            graph_t.as_secs_f64() * 1e6 / pairs as f64,
            graph_t.as_secs_f64() * 1e6 / (pairs * CHAIN * 2) as f64,
            indiv.as_secs_f64() * CHAIN as f64 / graph_t.as_secs_f64(),
        ))
    }
}
