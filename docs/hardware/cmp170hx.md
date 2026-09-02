# CMP 170HX — Target Hardware Specification

Deployment priority-1 device. Factual basis: local research notes
(`source/research/2026-08-30-cmp170hx-constraints.md`, source URLs included).

## Base Specification

- GA100, **sm_80**, 70 SMs / 4480 FP32 cores / 280 tensor cores. CUDA 11.0–13.x;
  works with standard Linux drivers (535-server / 570 / 610 verified) — no
  special driver needed.
- HBM2e 8 GB (stock, 7.7 GiB usable) / 4096-bit / 1493 theoretical · 1355
  measured GB/s. ECC off.
- PCIe **1.1 x4** (~0.85 GB/s), BAR1 64 MiB, no ReBAR. No NVLink.
- Headless (no outputs), passive cooling, **power = 2× 8-pin EPS 12V**
  (not PCIe 8-pin), 250 W TDP (measured 170–230 W).
- FP64 effectively absent (94–195 GFLOPS). No MIG.

## Throttle Matrix (stock eFUSE) — kernel rules for `cmp-stock` mode

| Path | Measured | Status |
|---|---|---|
| FP32 FFMA (fused) | 0.39 TFLOPS (1/32) | **forbidden** — decompose FMA into mul+add |
| FP32 mul/add separated | 6.2–6.3 TFLOPS | usable |
| **FP16 half2 (CUDA cores)** | **42 TFLOPS (83%)** | **best floating-point path** |
| BF16 CUDA cores | ~20 TFLOPS | usable |
| Tensor cores HMMA/TF32 | 6.07 / 3.04 TFLOPS (12%) | **forbidden** (slower than CUDA cores) |
| INT32 mul/add | 12.5 TIOPS | unthrottled |
| FP64 | ~0.1 TFLOPS | forbidden |

- Transcendental substitutes: `__expf` / `__fdividef` / `rsqrtf`. Community
  precedent (eastmoe patch): llama.cpp pp +105% / tg +83%.
- Performance character: **decode = bandwidth-bound (A100-class strength),
  prefill = compute-bound (RTX 2060-class)**.

## Unlock (user-confirmed premise as of 2026-08-30: 40/64 GB available)

- cmpunlocker (2026, GSP/SEC2 exploit, clean-room, driver 610.43.03 line):
  **memory unlock + compute (throttle) unlock**. A 10 GB card exposing
  40,960 MiB confirmed (2026-07-30). The 64 GB unlock is user-confirmed fact.
- Design stance: `cmp-unlocked` is a **first-class scenario**. It is
  unofficial, so `cmp-stock` operation is maintained too (mode separation
  costs nothing).
- Compute unlock may lift the FFMA/tensor-core throttles → `cmp-unlocked`
  kernels may use full-rate paths. **Until unlock measurements exist, the
  half2 path stays the default; switch after measuring** (see Verification
  items).

## Memory Budget vs Reference Models

| Config | Usable | Qwen3.8-27B | Flash-Next (111.3 GB + PLE 28 GB) |
|---|---|---|---|
| `cmp-stock` | ~7.7 GiB | even Q4 (16.5 GB) doesn't fit → tensor offload mandatory | large-scale offload |
| `cmp-unlocked` 40 GB | ~39 GiB | Q8 (31.4 GB) + KV fits | not possible — offload |
| `cmp-unlocked` 64 GB | ~63 GiB | all quantizations resident | body partially resident + PLE/some experts offloaded |

- Offload path constraint: PCIe 1.1 x4 = 0.85 GB/s. The NVMe mmap +
  page-cache pattern (the PLE offload in the Flash-Next reference run script
  is the archetype) is the sensible default.
- Loading: BAR1 64 MiB → whole-VRAM mmap impossible; pinned-buffer chunked
  copies. 8 GB card loads in ~10 s+ (40/64 GB scales proportionally — same
  bandwidth, so 60–90 s [estimated]).

## Verification Items (on hardware arrival)

1. Re-measure the post-unlock throttle matrix (FFMA/tensor-core full rate?)
   → decide the `cmp-unlocked` kernel switch.
2. Vulkan compute exposure (stock and unlocked each) — whether the universal
   path carries to the CMP directly.
3. Unlocked-memory stability soak (40/64 GB, long duration).
4. Cooling: 3000+ RPM chassis airflow or an A100 waterblock (Bykski
   N-TESLA-A100-X-V2 compatible).
