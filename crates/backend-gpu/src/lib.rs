//! GPU 백엔드 (cubecl + HIP/ROCm) — universal 모드.
//!
//! 커널은 Rust(cubecl 매크로)로 작성, hipRTC가 gfx1151용으로 JIT 컴파일.
//! 같은 커널이 CUDA/PTX(sm_80)로도 컴파일되어 CMP 170HX에 직결 (ADR-0009).

use cubecl::prelude::*;
use cubecl::zspace::{Shape, Strides};

#[cube(launch_unchecked)]
/// f32 GEMV: out[o] = Σ_i x[i]·W[o,i] — W: [n_out, n_in] 행 우선, 큐브당 1 출력 행.
fn gemv_f32(
    x: &Tensor<f32>,
    w: &Tensor<f32>,
    out: &mut Tensor<f32>,
    #[comptime] n_in: usize,
) {
    let o = ABSOLUTE_POS_X as usize;
    if o >= out.len() {
        terminate!();
    }
    let mut acc = 0.0f32;
    let base = o * n_in;
    for i in 0..n_in {
        acc += x[i] * w[base + i];
    }
    out[o] = acc;
}

/// 디바이스 프로브.
pub fn probe() -> Result<String, String> {
    let device = hanzo_cubecl_hip::AmdDevice::new(0);
    let client = hanzo_cubecl_hip::HipRuntime::client(&device);
    let _ = client.properties();
    Ok("hip device 0 ready".to_string())
}

/// GEMV 스모크: 결정적 입력으로 CPU 대조.
pub fn smoke_gemv(n_in: usize, n_out: usize) -> Result<Vec<f32>, String> {
    let device = hanzo_cubecl_hip::AmdDevice::new(0);
    let client = hanzo_cubecl_hip::HipRuntime::client(&device);

    let x: Vec<f32> = (0..n_in).map(|i| (i as f32 * 0.37) % 1.0 - 0.5).collect();
    let w: Vec<f32> = (0..n_in * n_out)
        .map(|i| ((i * 31 + 7) as f32 * 0.11 % 1.0) - 0.5)
        .collect();

    let x_gpu = client.create_from_slice(bytemuck::cast_slice(&x));
    let w_gpu = client.create_from_slice(bytemuck::cast_slice(&w));
    let out_gpu = client.empty(n_out * 4);

    unsafe {
        let shape_x: Shape = [n_in].into();
        let shape_w: Shape = [n_in * n_out].into();
        let shape_o: Shape = [n_out].into();
        let strides_x: Strides = [1].into();
        let strides_w: Strides = [1].into();
        let strides_o: Strides = [1].into();
        gemv_f32::launch_unchecked(
            &client,
            CubeCount::Static(n_out as u32, 1, 1),
            CubeDim::new_2d(64, 4), // 256 threads/cube — GEMV 행 병렬
            TensorArg::from_raw_parts(x_gpu, strides_x, shape_x),
            TensorArg::from_raw_parts(w_gpu, strides_w, shape_w),
            TensorArg::from_raw_parts(out_gpu.clone(), strides_o, shape_o),
            n_in,
        );
    }

    let bytes = client.read_one(out_gpu).map_err(|e| e.to_string())?;
    let v: Vec<f32> = bytemuck::cast_slice(&bytes).to_vec();

    let mut ok = true;
    for o in 0..n_out {
        let mut acc = 0.0f32;
        for i in 0..n_in {
            acc += x[i] * w[o * n_in + i];
        }
        if (acc - v[o]).abs() > 1e-3 {
            ok = false;
        }
    }
    if !ok {
        return Err("GEMV GPU 결과 불일치".into());
    }
    Ok(v)
}
