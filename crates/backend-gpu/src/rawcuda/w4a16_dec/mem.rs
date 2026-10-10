//! [R1 2026-10-10] W4a16Dec 메모리·스크래치·모니터링 — mod.rs 분해.
//! h2d/zero·스크래치 alloc/free·복사 계측·mem_stats·realloc_fields.

use super::*;

/// ensure_* 버퍼 교체 계약(G1) — 기존 전량 해제 → 전 필드 0화 → 재할당.
/// alloc이 중간에 실패하면 필드는 0 또는 그때까지의 성공분을 유지한다:
/// 재시도는 성공분만 회수하고(이중해제 없음) 처음부터 다시 할당한다.
/// cap류는 호출부가 0으로 내린 뒤 성공 시에만 갱신할 것.
/// free/alloc을 클로저로 받는 이유: CUDA 없이 모의 주입 단위 테스트(회귀).
pub(super) fn realloc_fields<const N: usize>(
    mut free: impl FnMut(CUdeviceptr) -> Result<(), String>,
    mut alloc: impl FnMut(usize) -> Result<CUdeviceptr, String>,
    mut fields: [&mut CUdeviceptr; N],
    sizes: [usize; N],
) -> Result<(), String> {
    for f in fields.iter() {
        if **f != 0 {
            free(**f)?;
        }
    }
    for f in fields.iter_mut() {
        **f = 0;
    }
    for (f, sz) in fields.into_iter().zip(sizes) {
        *f = alloc(sz)?;
    }
    Ok(())
}

impl W4a16Dec {
    pub(crate) fn h2d_chunked(cc: &CudaCtx, dst: CUdeviceptr, src: &[u8]) -> Result<(), String> {
        const CH: usize = 8 << 20;
        // 페이지러블 비동기(드라이버 스테이징 — 호출 내 스테이징 완료 계약) +
        // 8청크(64MiB)마다 sync로 스테이징 큐 상한.
        //
        // [실측 2026-10-09, 4090·Gen4 x16] 수동 핀드 링 경로(CPU→핀드 복사 +
        // 핑퐁 DMA)는 ~5GB/s — CPU의 핀드(비캐시) 기록이 병목이었다. 드라이버
        // 스테이징은 9.3GB/s(h2d-bench), 핀드 DMA는 13.8GB/s지만 그 앞단 복사가
        // 더 느리다. 합계 20GB급 업로드가 4.4s → ~2.2s.
        const INFLIGHT: usize = 64 << 20; // 스테이징 큐 상한(바이트 기준)
        let mut pending = 0usize;
        for (i, chunk) in src.chunks(CH).enumerate() {
            cc.h2d_async(dst + (i * CH) as u64, chunk)?;
            pending += chunk.len();
            if pending >= INFLIGHT {
                cc.sync()?;
                pending = 0;
            }
            // 업로드는 수 초~수십 초 — 와치독이 로드를 스텔로 오판하지 않게 심박.
            llm170_diag::watchdog::bump();
        }
        cc.sync()
    }

    pub(super) fn zero_dev(cc: &CudaCtx, ptr: CUdeviceptr, len: usize) -> Result<(), String> {
        // [2026-10-09 H] 동기 h2d 4MiB 청크 루프 → cuMemsetD8Async 1콜.
        // (종전에는 청크마다 스테이징·호출 — 게다가 캡처 경로에서 금지되는
        // 동기 복사였다.) 스트림 순서라 후속 커널·판독과 정합.
        cc.memset0_async(ptr, len)
    }

    /// 메모리 분류(모니터링) — 모델이 시스템을 어떻게 쓰는지.
    /// 반환: (VRAM 가중치, VRAM KV, CPU 오프로드 가중치, CPU PLE).
    /// - 가중치: 업로드 누적(weights_bytes — 로드 후 불변). 스트리밍 모드에서
    ///   전문가는 VRAM에 없고 호스트(mmAP 페이지 캐시)에서 토큰별로 올린다.
    /// - KV: 어텐션 캐시 2벌(K+V) — 기본 f32, [P13] KVQ 시 int8+스케일.
    /// - PLE: 미구현(W4-2) — 항상 0.
    pub fn mem_stats(&self) -> (u64, u64, u64, u64) {
        let kv = self
            .attn
            .map(|d| {
                let rows = (self.n_slots as u64) * (d.n_attn as u64) * (d.cap as u64);
                // [KVQ 채택 2026-10-10] int8 KV + 행×헤드 f32 스케일 2벌.
                rows * (d.kv_dim() as u64) + rows * (d.kv_heads as u64) * 4 * 2
            })
            .unwrap_or(0);
        let experts = self.experts_bytes;
        let (w_gpu, w_cpu) = if self.n_experts > 0 && !self.moe_resident {
            (self.weights_bytes, experts)
        } else {
            (self.weights_bytes, 0)
        };
        (w_gpu, kv, w_cpu, 0)
    }

    /// 토큰당 활성 가중치 바이트(모니터링 — 실효 대역폭 계산용).
    /// 상주 모드 = 상주 가중치 − 미선택 전문가(전문가 크기 균일 — 평균이 정확).
    /// 스트리밍 모드 = 상주 가중치(전문가는 별도 CPU 오프로드로 집계).
    pub fn active_weight_bytes(&self) -> u64 {
        let experts = self.experts_bytes;
        if self.n_experts > 0 && self.moe_resident {
            let unsel = experts / self.n_experts as u64 * (self.n_experts - self.top_k) as u64;
            self.weights_bytes.saturating_sub(unsel)
        } else {
            self.weights_bytes
        }
    }

    /// 복사 계측(모니터링) — [h2d, d2h, d2d] × (바이트, ns, 호출).
    pub fn copy_stats(&self) -> [(u64, u64, u64); 3] {
        self.cc.copy_stats()
    }

    /// 마이크로벤치 표면(진단 전용) — 스크래치 alloc/h2d/free + GEMM 발사/sync.
    /// 핀드 스크래치 할당(벤치·진단) — h2d_bench 계약.
    pub fn alloc_pinned_scratch(&self, bytes: usize) -> Result<*mut std::ffi::c_void, String> {
        self.cc.pinned_alloc(bytes)
    }

    pub fn free_pinned_scratch(&self, p: *mut std::ffi::c_void) -> Result<(), String> {
        self.cc.pinned_free(p)
    }

    pub fn alloc_scratch(&self, bytes: usize) -> Result<CUdeviceptr, String> {
        self.cc.alloc(bytes)
    }

    pub fn free_scratch(&self, p: CUdeviceptr) -> Result<(), String> {
        self.cc.free(p)
    }

    pub fn h2d_scratch(&self, dst: CUdeviceptr, src: &[u8]) -> Result<(), String> {
        self.cc.h2d(dst, src)
    }

    pub fn sync_bench(&self) -> Result<(), String> {
        self.cc.sync()
    }

    /// 지정 선형의 t≥2 GEMM 1회 발사(벤치 전용 — y는 스크래치).
    /// 진단 타이머 리포트(P8) — LLM170_TIME=1일 때 범주별 ms·비중.
    pub fn prof_report(&mut self, label: &str) -> Result<String, String> {
        let _g = self.cc.guard()?;
        self.cc.prof_report(label)
    }
}
