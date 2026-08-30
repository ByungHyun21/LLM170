//! GPU 백엔드 (cubecl + HIP/ROCm) — universal 모드.
//!
//! 커널은 Rust(cubecl 매크로)로 작성, hipRTC가 gfx1151용으로 JIT 컴파일.
//! 같은 커널이 CUDA/PTX(sm_80)로도 컴파일되어 CMP 170HX에 직결 (ADR-0009).
//!
//! `GpuMatmul`: `Accelerator` 구현 — 양자화 가중치 원시 바이트를 GPU에 상주시키고
//! dequant+FMA 없는 mul+add GEMM을 수행. 블록/요소 순서는 CPU 참조(quant.rs·matmul.rs)와
//! 동일한 누산 순서를 유지해 수치 일치 확률을 최대화한다 (ADR-0005 strict FP).

use cubecl::prelude::*;
use cubecl::zspace::{Shape, Strides};
use cubecl_runtime::server::Handle;
use half::f16;
use llm170_core::matmul::{Accelerator, Weight};
use llm170_gguf::GgmlType;
use llm170_profiler::profile_span;
use std::collections::HashMap;
use std::sync::Mutex;

// ---------------------------------------------------------------------------
// 커널 — 양자화 블록을 즉석 디양자화하며 GEMM.
// 그리드: x = n_out/64 (행), y = t_len/4 (토큰). 유닛 = (행, 토큰) 1:1.
// 누산: 행별 블록 순차 + 블록 내 j 순차 — CPU matmul과 동일 순서.
// ---------------------------------------------------------------------------

/// 바이트 값 → 부호 있는 f32 (ggml i8 재해석).
#[cube]
fn byte_signed(v: u32) -> f32 {
    let x = v as i32;
    if x > 127 { (x - 256) as f32 } else { x as f32 }
}

/// u32 워드에서 i바이트 추출 (리틀 엔디안) — WGSL에 u8이 없어 양자화 바이트는
/// u32 텐서로 운반하고 커널에서 언팩한다 (llama.cpp CUDA 관례와 동일).
#[cube]
fn byte(w: &Tensor<u32>, i: usize) -> u32 {
    (w[i >> 2] >> (((i & 3) * 8) as u32)) & 0xFF
}

/// f16 바이트 2개 → f32 (리틀 엔디안). WGSL에 u16/f16이 없어 u32 비트 산술로 전개 —
/// quant.rs half_to_f32와 동일 값 (부동소수 f16 → f32는 항상 정확).
#[cube]
fn f16_at(w: &Tensor<u32>, off: usize) -> f32 {
    let h = byte(w, off) | (byte(w, off + 1) << 8);
    let sign = (h & 0x8000) << 16;
    let exp = (h >> 10) & 0x1F;
    let frac = h & 0x3FF;
    if exp == 0 {
        if frac == 0 {
            f32::from_bits(sign)
        } else {
            let v = (frac as f32) * (1.0 / 16777216.0);
            if sign != 0 { -v } else { v }
        }
    } else if exp == 31 {
        f32::from_bits(sign | 0x7F800000 | (frac << 13))
    } else {
        f32::from_bits(sign | ((exp + 112) << 23) | (frac << 13))
    }
}

/// q4_K/q5_K 공용 6비트 스케일+min 추출 (get_scale_min_k4, ggml-quants.c:880).
#[cube]
fn scale_min_k4(j: usize, w: &Tensor<u32>, base: usize) -> (u32, u32) {
    if j < 4 {
        (byte(w, base + j) & 63, byte(w, base + j + 4) & 63)
    } else {
        (
            (byte(w, base + j + 4) & 0xF) | ((byte(w, base + j - 4) >> 6) << 4),
            (byte(w, base + j + 4) >> 4) | ((byte(w, base + j) >> 6) << 4),
        )
    }
}

/// 양자화 GEMM: out[t·n_out+o] = Σ_i x[t·n_in+i]·W[o,i] — W는 원시 블록 바이트.
/// qtype 값은 GgmlType 판별자 그대로 (comptime 특수화 → 타입별 1회 JIT).
#[cube(launch_unchecked)]
fn gemm_q(
    x: &Tensor<f32>,
    w: &Tensor<u32>,
    out: &mut Tensor<f32>,
    ktab: &Tensor<f32>,
    grid3: &Tensor<u32>,
    n_in: usize,
    n_out: usize,
    t_len: usize,
    #[comptime] qtype: usize,
) {
    let o = ABSOLUTE_POS_X as usize; // 행 (CubeDim::x = 64)
    let t = ABSOLUTE_POS_Y as usize; // 토큰 (CubeDim::y = 4)
    if o >= n_out || t >= t_len {
        terminate!();
    }
    let blck = if qtype == 0 {
        1
    } else if qtype == 1 {
        1
    } else if qtype == 30 {
        1
    } else if qtype == 8 {
        32
    } else if qtype == 20 {
        32
    } else {
        256
    };
    let bsize = if qtype == 0 {
        4
    } else if qtype == 1 {
        2
    } else if qtype == 30 {
        2
    } else if qtype == 8 {
        34
    } else if qtype == 20 {
        18
    } else if qtype == 12 {
        144
    } else if qtype == 13 {
        176
    } else if qtype == 14 {
        210
    } else if qtype == 11 {
        110
    } else if qtype == 23 {
        136
    } else {
        110 // iq3_s (21)
    };

    let blocks = n_in / blck;
    let row_base = o * blocks * bsize;
    let xb = t * n_in;
    let mut acc = 0.0f32;
    for b in 0..blocks {
        let wb = row_base + b * bsize;
        let xo = xb + b * blck;

        if qtype == 0 {
            // F32: 요소당 4바이트
            let bits = (byte(w, wb) as u32)
                | ((byte(w, wb + 1) as u32) << 8)
                | ((byte(w, wb + 2) as u32) << 16)
                | ((byte(w, wb + 3) as u32) << 24);
            acc += x[xo] * f32::from_bits(bits);
        } else if qtype == 1 {
            // F16
            acc += x[xo] * f16_at(w, wb);
        } else if qtype == 30 {
            // Bf16
            let h = (byte(w, wb) as u32) | ((byte(w, wb + 1) as u32) << 8);
            acc += x[xo] * f32::from_bits(h << 16);
        } else if qtype == 8 {
            // Q8_0: d(2) qs(32, i8)
            let d = f16_at(w, wb);
            for j in 0..32 {
                acc += x[xo + j] * (byte_signed(byte(w, wb + 2 + j)) * d);
            }
        } else if qtype == 12 {
            // Q4_K: d(2) dmin(2) scales(12) qs(128)
            let d = f16_at(w, wb);
            let min = f16_at(w, wb + 2);
            for sb in 0..8 {
                let (sc, m) = scale_min_k4(sb, w, wb + 4);
                let d1 = d * sc as f32;
                let mm = min * m as f32;
                for j in 0..32 {
                    let q = byte(w, wb + 16 + (sb / 2) * 32 + j);
                    let nib = if sb % 2 == 0 { q & 0xF } else { q >> 4 };
                    acc += x[xo + sb * 32 + j] * (d1 * nib as f32 - mm);
                }
            }
        } else if qtype == 13 {
            // Q5_K: d(2) dmin(2) scales(12) qh(32) qs(128)
            let d = f16_at(w, wb);
            let min = f16_at(w, wb + 2);
            for sb in 0..8 {
                let (sc, m) = scale_min_k4(sb, w, wb + 4);
                let d1 = d * sc as f32;
                let mm = min * m as f32;
                let sh = (2 * (sb / 2)) as u32;
                for j in 0..32 {
                    let qh_b = byte(w, wb + 16 + j) as u32;
                    let ql = byte(w, wb + 48 + (sb / 2) * 32 + j);
                    let bit = if sb % 2 == 0 {
                        (qh_b >> sh) & 1
                    } else {
                        (qh_b >> (sh + 1)) & 1
                    };
                    let nib = if sb % 2 == 0 { ql & 0xF } else { ql >> 4 };
                    let v = nib + bit * 16;
                    acc += x[xo + sb * 32 + j] * (d1 * v as f32 - mm);
                }
            }
        } else if qtype == 14 {
            // Q6_K: ql(128) qh(64) scales(16, i8) d(2)
            let d = f16_at(w, wb + 208);
            for h in 0..2 {
                for l in 0..32 {
                    let is = h * 8 + l / 16;
                    let qlo = byte(w, wb + h * 64 + l);
                    let qlo2 = byte(w, wb + h * 64 + 32 + l);
                    let qhb = byte(w, wb + 128 + h * 32 + l);
                    let q1 = (((qlo & 0xF) | ((qhb & 3) << 4)) as i32) - 32;
                    acc +=
                        x[xo + h * 128 + l] * (d * byte_signed(byte(w, wb + 192 + is)) * q1 as f32);
                    let q2 = (((qlo2 & 0xF) | (((qhb >> 2) & 3) << 4)) as i32) - 32;
                    acc += x[xo + h * 128 + 32 + l]
                        * (d * byte_signed(byte(w, wb + 192 + is + 2)) * q2 as f32);
                    let q3 = (((qlo >> 4) | (((qhb >> 4) & 3) << 4)) as i32) - 32;
                    acc += x[xo + h * 128 + 64 + l]
                        * (d * byte_signed(byte(w, wb + 192 + is + 4)) * q3 as f32);
                    let q4 = (((qlo2 >> 4) | (((qhb >> 6) & 3) << 4)) as i32) - 32;
                    acc += x[xo + h * 128 + 96 + l]
                        * (d * byte_signed(byte(w, wb + 192 + is + 6)) * q4 as f32);
                }
            }
        } else if qtype == 11 {
            // Q3_K: hmask(32) qs(64) scales(12) d(2)
            let d_all = f16_at(w, wb + 108);
            let a0 = (byte(w, wb + 96) as u32)
                | ((byte(w, wb + 97) as u32) << 8)
                | ((byte(w, wb + 98) as u32) << 16)
                | ((byte(w, wb + 99) as u32) << 24);
            let a1 = (byte(w, wb + 100) as u32)
                | ((byte(w, wb + 101) as u32) << 8)
                | ((byte(w, wb + 102) as u32) << 16)
                | ((byte(w, wb + 103) as u32) << 24);
            let tmp = (byte(w, wb + 104) as u32)
                | ((byte(w, wb + 105) as u32) << 8)
                | ((byte(w, wb + 106) as u32) << 16)
                | ((byte(w, wb + 107) as u32) << 24);
            let k1 = 0x03030303u32;
            let k2 = 0x0f0f0f0fu32;
            let aux2 = ((a0 >> 4) & k2) | (((tmp >> 4) & k1) << 4);
            let aux3 = ((a1 >> 4) & k2) | (((tmp >> 6) & k1) << 4);
            let aux0 = (a0 & k2) | ((tmp & k1) << 4);
            let aux1 = (a1 & k2) | (((tmp >> 2) & k1) << 4);
            for n in 0..2 {
                for si in 0..4 {
                    let shift = si * 2;
                    let bit = si as u32;
                    for half in 0..2 {
                        let ai = n * 8 + si * 2 + half;
                        let aux_v = if ai < 4 {
                            aux0
                        } else if ai < 8 {
                            aux1
                        } else if ai < 12 {
                            aux2
                        } else {
                            aux3
                        };
                        let scb = (aux_v >> (((ai % 4) * 8) as u32)) & 0xFF;
                        let dl = d_all * (byte_signed(scb) - 32.0);
                        for l in 0..16 {
                            let qv = ((byte(w, wb + 32 + n * 32 + half * 16 + l) as u32
                                >> (shift as u32))
                                & 3) as i32;
                            let sub =
                                4 - (((byte(w, wb + half * 16 + l) as u32 >> bit) & 1) as i32) * 4;
                            acc += x[xo + n * 128 + si * 32 + half * 16 + l]
                                * (dl * (qv - sub) as f32);
                        }
                    }
                }
            }
        } else if qtype == 23 {
            // IQ4_XS: d(2) scales_h(2) scales_l(4) qs(128)
            let d = f16_at(w, wb);
            let scales_h = byte(w, wb + 2) | (byte(w, wb + 3) << 8);
            for ib in 0..8 {
                let ls = ((byte(w, wb + 4 + ib / 2) >> ((4 * (ib % 2)) as u32)) & 0xF)
                    | (((scales_h >> ((2 * ib) as u32)) & 3) << 4);
                let dl = d * (ls as i32 - 32) as f32;
                for j in 0..16 {
                    let q = byte(w, wb + 8 + ib * 16 + j);
                    acc += x[xo + ib * 32 + j] * (dl * ktab[(q & 0xF) as usize]);
                    acc += x[xo + ib * 32 + 16 + j] * (dl * ktab[(q >> 4) as usize]);
                }
            }
        } else if qtype == 20 {
            // IQ4_NL: d(2) qs(16)
            let d = f16_at(w, wb);
            for j in 0..16 {
                let q = byte(w, wb + 2 + j);
                acc += x[xo + j] * (d * ktab[(q & 0xF) as usize]);
                acc += x[xo + 16 + j] * (d * ktab[(q >> 4) as usize]);
            }
        } else {
            // IQ3_S (21): d(2) qs(64) qh(8) signs(32) scales(4)
            let d = f16_at(w, wb);
            for ib in 0..4 {
                let db1 = d * (1 + 2 * (byte(w, wb + 106 + ib) & 0xF)) as f32;
                let db2 = d * (1 + 2 * (byte(w, wb + 106 + ib) >> 4)) as f32;
                let qh0 = byte(w, wb + 66 + 2 * ib) as usize;
                let qh1 = byte(w, wb + 66 + 2 * ib + 1) as usize;
                for l in 0..4 {
                    let i1 =
                        (byte(w, wb + 2 + ib * 16 + 2 * l) as usize) | ((qh0 << (8 - 2 * l)) & 256);
                    let i2 = (byte(w, wb + 2 + ib * 16 + 2 * l + 1) as usize)
                        | ((qh0 << (7 - 2 * l)) & 256);
                    for j in 0..4 {
                        let g1 = byte_signed((grid3[i1] >> ((8 * j) as u32)) & 0xFF);
                        let g2 = byte_signed((grid3[i2] >> ((8 * j) as u32)) & 0xFF);
                        let sg = byte(w, wb + 74 + ib * 8 + l);
                        let s1 = 1.0 - 2.0 * (((sg as u32 >> (j as u32)) & 1) as f32);
                        let s2 = 1.0 - 2.0 * (((sg as u32 >> ((4 + j) as u32)) & 1) as f32);
                        acc += x[xo + ib * 64 + l * 8 + j] * (db1 * g1 * s1);
                        acc += x[xo + ib * 64 + l * 8 + 4 + j] * (db1 * g2 * s2);
                    }
                }
                for l in 0..4 {
                    let i1 = (byte(w, wb + 2 + ib * 16 + 8 + 2 * l) as usize)
                        | ((qh1 << (8 - 2 * l)) & 256);
                    let i2 = (byte(w, wb + 2 + ib * 16 + 8 + 2 * l + 1) as usize)
                        | ((qh1 << (7 - 2 * l)) & 256);
                    for j in 0..4 {
                        let g1 = byte_signed((grid3[i1] >> ((8 * j) as u32)) & 0xFF);
                        let g2 = byte_signed((grid3[i2] >> ((8 * j) as u32)) & 0xFF);
                        let sg = byte(w, wb + 74 + ib * 8 + 4 + l);
                        let s1 = 1.0 - 2.0 * (((sg as u32 >> (j as u32)) & 1) as f32);
                        let s2 = 1.0 - 2.0 * (((sg as u32 >> ((4 + j) as u32)) & 1) as f32);
                        acc += x[xo + ib * 64 + 32 + l * 8 + j] * (db2 * g1 * s1);
                        acc += x[xo + ib * 64 + 32 + l * 8 + 4 + j] * (db2 * g2 * s2);
                    }
                }
            }
        }
    }
    out[t * n_out + o] = acc;
}

// ---------------------------------------------------------------------------
// f32 스모크 커널 (원형 유지 — gpu-smoke 서브커맨드)
// ---------------------------------------------------------------------------

#[cube(launch_unchecked)]
/// f32 GEMV: out[o] = Σ_i x[i]·W[o,i] — W: [n_out, n_in] 행 우선, 큐브당 1 출력 행.
fn gemv_f32(x: &Tensor<f32>, w: &Tensor<f32>, out: &mut Tensor<f32>, #[comptime] n_in: usize) {
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

/// wgpu/Vulkan 디바이스 프로브.
pub fn probe_vulkan() -> Result<String, String> {
    use cubecl::wgpu::{RuntimeOptions, Vulkan, WgpuDevice, WgpuRuntime, init_device, init_setup};
    let device = WgpuDevice::DefaultDevice;
    let setup = init_setup::<Vulkan>(&device, RuntimeOptions::default());
    let dev = init_device(setup, RuntimeOptions::default());
    let client = WgpuRuntime::client(&dev);
    let _ = client.properties();
    Ok("wgpu/vulkan device ready".to_string())
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

    // SAFETY: 커널은 ABSOLUTE_POS_X >= out.len() 시 terminate!() — OOB 없음.
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

// ---------------------------------------------------------------------------
// Accelerator 구현 — core::matmul::Accelerator (server --backend gpu)
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct DevWeight {
    h: Handle,
    ty: GgmlType,
    n_in: usize,
    n_out: usize,
    bytes: usize,
}

/// GPU matmul 가속기 — 런타임 제네릭(HIP 또는 wgpu/Vulkan).
/// 가중치는 데이터 포인터 키로 1회 업로드 후 상주. 커널 소스는 런타임 공유 (ADR-0009).
pub struct GpuMatmul<R: Runtime> {
    client: ComputeClient<R>,
    weights: Mutex<HashMap<usize, DevWeight>>,
    ktab: Handle,  // iq4_nl 룩업 (f32×16)
    grid3: Handle, // iq3_s 그리드 (u32×512)
}

impl GpuMatmul<hanzo_cubecl_hip::HipRuntime> {
    /// ROCm/HIP 런타임 (기본).
    pub fn new_hip() -> Result<Self, String> {
        let device = hanzo_cubecl_hip::AmdDevice::new(0);
        let client = hanzo_cubecl_hip::HipRuntime::client(&device);
        Self::with_client(client)
    }
}

impl GpuMatmul<cubecl::wgpu::WgpuRuntime> {
    /// wgpu/Vulkan 런타임 — 같은 커널을 WGSL/SPIR-V로 컴파일.
    pub fn new_vulkan() -> Result<Self, String> {
        use cubecl::wgpu::{
            RuntimeOptions, Vulkan, WgpuDevice, WgpuRuntime, init_device, init_setup,
        };
        // DefaultDevice = 고성능 우선 선택 — 개발기 iGPU(8060S)·CMP dGPU 모두 대응
        // (DiscreteGpu 필터는 APU에서 설계상 페닉)
        let device = WgpuDevice::DefaultDevice;
        let setup = init_setup::<Vulkan>(&device, RuntimeOptions::default());
        let dev = init_device(setup, RuntimeOptions::default());
        let client = WgpuRuntime::client(&dev);
        Self::with_client(client)
    }
}

impl<R: Runtime> GpuMatmul<R> {
    fn with_client(client: ComputeClient<R>) -> Result<Self, String> {
        let ktab: Vec<f32> = llm170_core::KVALUES_IQ4NL
            .iter()
            .map(|&v| v as f32)
            .collect();
        let grid3 = llm170_core::IQ3S_GRID;
        let ktab = client.create_from_slice(bytemuck::cast_slice(&ktab));
        let grid3 = client.create_from_slice(bytemuck::cast_slice(&grid3));
        Ok(GpuMatmul {
            client,
            weights: Mutex::new(HashMap::new()),
            ktab,
            grid3,
        })
    }

    fn dev_weight(&self, w: &Weight) -> Result<DevWeight, String> {
        let key = w.data.as_ptr() as usize;
        let mut map = self
            .weights
            .lock()
            .map_err(|_| "weights cache lock poisoned")?;
        if let Some(d) = map.get(&key) {
            return Ok(d.clone());
        }
        // WGSL에 u8이 없어 u32 워드로 운반 (byte() 헬퍼가 언팩). 4바이트 정렬 필수.
        if w.data.len() % 4 != 0 {
            return Err(format!("tensor bytes {} not 4-byte aligned", w.data.len()));
        }
        let h = self.client.create_from_slice(w.data);
        let d = DevWeight {
            h,
            ty: w.ty,
            n_in: w.n_in as usize,
            n_out: w.n_out as usize,
            bytes: w.data.len() / 4,
        };
        map.insert(key, d.clone());
        Ok(d)
    }

    /// GEMM 실행 → [t][n_out] 플랫 결과.
    fn run_gemm(&self, d: &DevWeight, xf: &[f32], t: usize) -> Result<Vec<f32>, String> {
        let (n_in, n_out) = (d.n_in, d.n_out);
        let xg = self.client.create_from_slice(bytemuck::cast_slice(xf));
        let og = self.client.empty(t * n_out * 4);
        let (blck, _) = d.ty.block_info();
        let _ = blck;
        let bytes = d.bytes;
        let gx = n_out.div_ceil(64) as u32;
        let gy = t.div_ceil(4) as u32;
        // SAFETY: 그리드가 (n_out, t_len)을 덮고 커널 시작부에서 범위 가드(terminate!) —
        // 모든 인덱스는 x[t·n_in+i]·w[바이트]·out[t·n_out+o] 상한 내. 무한루프 없음.
        unsafe {
            gemm_q::launch_unchecked(
                &self.client,
                CubeCount::Static(gx, gy, 1),
                CubeDim::new_2d(64, 4),
                TensorArg::from_raw_parts(xg, [1].into(), [t * n_in].into()),
                TensorArg::from_raw_parts(d.h.clone(), [1].into(), [d.bytes].into()),
                TensorArg::from_raw_parts(og.clone(), [1].into(), [t * n_out].into()),
                TensorArg::from_raw_parts(self.ktab.clone(), [1].into(), [16].into()),
                TensorArg::from_raw_parts(self.grid3.clone(), [1].into(), [512].into()),
                n_in,
                n_out,
                t,
                d.ty as u32 as usize,
            );
        }
        let raw = self.client.read_one(og).map_err(|e| e.to_string())?;
        Ok(bytemuck::cast_slice(&raw).to_vec())
    }
}

impl<R: Runtime> Accelerator for GpuMatmul<R> {
    fn matmul_batch(
        &self,
        xs: &[Vec<f32>],
        w: &Weight,
        outs: &mut [Vec<f32>],
    ) -> Result<(), String> {
        profile_span!("gpu::matmulB");
        let t = xs.len();
        let n_in = w.n_in as usize;
        if xs.iter().any(|r| r.len() != n_in) {
            return Err(format!("matmul_batch: x 행 길이 != n_in ({n_in})"));
        }
        let d = self.dev_weight(w)?;
        let mut xf = Vec::with_capacity(t * n_in);
        for r in xs {
            xf.extend_from_slice(r);
        }
        let res = self.run_gemm(&d, &xf, t)?;
        for (ti, out) in outs.iter_mut().enumerate() {
            out.copy_from_slice(&res[ti * d.n_out..(ti + 1) * d.n_out]);
        }
        Ok(())
    }

    fn matmul(&self, x: &[f32], w: &Weight, out: &mut [f32]) -> Result<(), String> {
        profile_span!("gpu::matmul1");
        let d = self.dev_weight(w)?;
        if x.len() != d.n_in {
            return Err(format!("matmul: x 길이 {} != n_in {}", x.len(), d.n_in));
        }
        let res = self.run_gemm(&d, x, 1)?;
        out.copy_from_slice(&res[..d.n_out]);
        Ok(())
    }
}
