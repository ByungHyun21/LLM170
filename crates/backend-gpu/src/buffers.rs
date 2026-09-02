//! GPU 버퍼 아레나 — 메모리 수명의 단일 소유 주체 (리팩토링 P0).
//!
//! 규칙 1문장: **GPU 메모리는 아레나에서만 나온다; 해제하는 경로는 없다.**
//!
//! 배경 (2026-09-01 실측): drop된 `create_from_slice` 핸들의 cubecl 지연
//! 해제가 큐 잔여 커널과 경합해 가비지 판독(NaN)을 유발했다. 휘발 업로드를
//! 포함한 전 버퍼를 크기등급 풀에 영속 보관함으로써 해제 자체를 제거한다.
//! 또한 가중치 예산 초과 폴백이 dummy(4B) 핸들을 흘려보내 더미 GEMM을
//! 일으킨 사고를, `WRef` 타입으로 host/GPU를 분리해 원천 차단한다.

use crate::Handle;
use cubecl::prelude::*;
use llm170_core::matmul::Weight;
use llm170_gguf::GgmlType;
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

/// 큐 완결 — 신규 할당 직전 호출해 할당·런치 경합을 원천 차단.
/// LLM170_NO_DRAIN=1이면 생략 (격리용 — DSD 스레드와의 교차 sync 상호작용).
fn drain<R: Runtime>(client: &ComputeClient<R>) {
    if std::env::var_os("LLM170_NO_DRAIN").is_some() {
        return;
    }
    let _ = cubecl_common::future::block_on(client.sync());
}

/// 가중치 참조 — GPU 상주 또는 호스트 폴백. dummy 핸들 부류 원천 제거:
/// `gpu()`는 호스트 폴백에서 Err를 내므로 무효 핸들 GEMM이 불가능하다.
#[derive(Clone)]
pub enum WRef {
    Gpu {
        h: Handle,
        ty: GgmlType,
        n_in: usize,
        n_out: usize,
        bytes: usize,
    },
    Host {
        ty: GgmlType,
        n_in: usize,
        n_out: usize,
    },
}

impl WRef {
    pub fn gpu(&self) -> Result<&Handle, String> {
        match self {
            WRef::Gpu { h, .. } => Ok(h),
            WRef::Host { .. } => Err("WRef::gpu: 호스트 폴백 가중치 (예산 초과)".into()),
        }
    }
    pub fn shape(&self) -> (usize, usize) {
        match self {
            WRef::Gpu { n_in, n_out, .. } => (*n_in, *n_out),
            WRef::Host { n_in, n_out, .. } => (*n_in, *n_out),
        }
    }
    pub fn ty(&self) -> GgmlType {
        match self {
            WRef::Gpu { ty, .. } | WRef::Host { ty, .. } => *ty,
        }
    }
    pub fn is_host(&self) -> bool {
        matches!(self, WRef::Host { .. })
    }
    /// u32 워드 수 (원시 바이트 / 4).
    pub fn words(&self) -> usize {
        match self {
            WRef::Gpu { bytes, .. } => *bytes,
            WRef::Host { n_in, n_out, ty } => {
                let (blck, bsize) = ty.block_info();
                (n_in / blck as usize) * bsize as usize * n_out
            }
        }
    }
}

/// Weight residency store — pointer-keyed, uploaded once, persistent.
/// Budget: when `LLM170_W_CAP_GB` is not set, 40% of the memory total
/// measured at runtime for the active backend (device-agnostic).
/// Rationale (measured 2026-09-01 on a 96 GiB carve-out): long-prefill
/// scratch pool demand is ~46 GiB ≈ 48% of total, so a 40% weight budget
/// leaves room for the pool plus margin. A fixed 48 GiB budget exhausted
/// VRAM (allocation "execution error" at ~6 min).
pub struct WeightStore {
    /// Key = (data pointer, byte length) — distinguishes views sharing a
    /// base pointer (full expert stack vs expert-0 slice, 2026-09-01 bug).
    map: Mutex<HashMap<(usize, usize), WRef>>,
    dev_bytes: std::sync::atomic::AtomicUsize,
    /// Upload cap in bytes, derived from the measured memory total.
    cap_bytes: usize,
}

#[allow(clippy::new_without_default)]
impl WeightStore {
    /// `mem_total`: total bytes of the placement region for this runtime
    /// (HIP = VRAM carve-out, Vulkan/wgpu = GTT). Constructors read it from
    /// amdgpu sysfs; unknown devices fall back to a conservative default.
    pub fn new(mem_total: usize) -> Self {
        WeightStore {
            map: Mutex::new(HashMap::new()),
            dev_bytes: std::sync::atomic::AtomicUsize::new(0),
            cap_bytes: Self::derive_cap(mem_total),
        }
    }

    /// Cap derivation: explicit env override wins; otherwise 40% of the
    /// measured total, GiB-aligned, with a 1 GiB floor so small test models
    /// (tiny4) are never locked out.
    fn derive_cap(mem_total: usize) -> usize {
        if let Some(v) = std::env::var("LLM170_W_CAP_GB")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
        {
            return v << 30;
        }
        // 프레임(활성 상주 디코드)은 전문가 스택 상주가 성립 조건 —
        // 큰 배치 영역(≥64GiB ≈ HIP carve-out)에서는 95%까지 상향.
        // 작은 영역(Vulkan GTT 등)은 기존 40% 유지: 스택이 못 들어가면
        // decode1이 value 경로로 폴백한다 (2026-09-02 기본 ON 전환).
        let frame_on = std::env::var("LLM170_FRAME").is_ok_and(|v| v != "0");
        if frame_on && mem_total >= 64 << 30 {
            return ((mem_total / 20 * 19) >> 30 << 30).max(1 << 30);
        }
        ((mem_total / 5 * 2) >> 30 << 30).max(1 << 30)
    }

    /// Upload cap in bytes.
    pub fn cap(&self) -> usize {
        self.cap_bytes
    }

    /// Weight lookup — uploads on first call (within budget) or falls back
    /// to host residency. Cached afterwards. Weights travel as u32 words
    /// (WGSL has no u8; 4-byte alignment required).
    pub fn get<R: Runtime>(
        &self,
        client: &ComputeClient<R>,
        w: &Weight,
    ) -> Result<WRef, String> {
        let key = (w.data.as_ptr() as usize, w.data.len());
        let mut map = self.map.lock().map_err(|_| "weights lock poisoned")?;
        if let Some(d) = map.get(&key) {
            return Ok(d.clone());
        }
        let total = self.dev_bytes.load(std::sync::atomic::Ordering::Relaxed);
        let host = total + w.data.len() > self.cap_bytes;
        let d = if host {
            WRef::Host { ty: w.ty, n_in: w.n_in as usize, n_out: w.n_out as usize }
        } else {
            if w.data.len() % 4 != 0 {
                return Err(format!("tensor bytes {} not 4-byte aligned", w.data.len()));
            }
            // New uploads (alloc + copy) only on a drained queue — allocating
            // while launches are in flight races inside libamdhip64 (GPF at
            // fixed IPs 0x43251c/0x3fc4e5, measured 2026-09-01). Uploads are
            // once-per-weight, so the sync cost is negligible.
            drain(client);
            let h = client.create_from_slice(w.data);
            self.dev_bytes
                .fetch_add(w.data.len(), std::sync::atomic::Ordering::Relaxed);
            WRef::Gpu {
                h,
                ty: w.ty,
                n_in: w.n_in as usize,
                n_out: w.n_out as usize,
                bytes: w.data.len() / 4,
            }
        };
        map.insert(key, d.clone());
        Ok(d)
    }

    /// Resident bytes (instrumentation).
    pub fn resident(&self) -> usize {
        self.dev_bytes.load(std::sync::atomic::Ordering::Relaxed)
    }
}

/// Read a memory total from amdgpu sysfs. `kind` selects the region file
/// (`vram` or `gtt`). Generic across devices — no per-machine constants.
pub fn sysfs_mem_total(kind: &str) -> Option<usize> {
    let dir = std::fs::read_dir("/sys/class/drm").ok()?;
    for entry in dir.flatten() {
        let p = entry.path().join("device").join(format!("mem_info_{kind}_total"));
        if let Ok(s) = std::fs::read_to_string(&p) {
            if let Ok(v) = s.trim().parse::<usize>() {
                if v > 0 {
                    return Some(v);
                }
            }
        }
    }
    None
}

/// 크기등급 스크래치/휘발업로드 풀 — 인플라이트 중 재할당 방지(획득은 pop,
/// read 동기화 후 반납). 반납된 핸들은 영속 재사용, 해제 없음.
pub struct ScratchPool {
    map: Mutex<HashMap<usize, Vec<Handle>>>,
    /// 풀에 할당된 누적 바이트 (VRAM 고갈 추적 계측).
    total: std::sync::atomic::AtomicUsize,
}

impl ScratchPool {
    pub fn new() -> Self {
        ScratchPool {
            map: Mutex::new(HashMap::new()),
            total: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    /// 풀에서 획득 (없으면 신규 할당 — 누적 계측).
    pub fn acquire<R: Runtime>(
        &self,
        client: &ComputeClient<R>,
        bytes: usize,
    ) -> Result<Handle, String> {
        let mut map = self.map.lock().map_err(|_| "buf pool lock poisoned")?;
        if let Some(h) = map.get_mut(&bytes).and_then(|v| v.pop()) {
            return Ok(h);
        }
        // 신규 할당도 큐가 빈 상태에서만 (위와 같은 경합 방지).
        drop(map);
        drain(client);
        let h = client.empty(bytes);
        let mut map = self.map.lock().map_err(|_| "buf pool lock poisoned")?;
        self.total.fetch_add(bytes, std::sync::atomic::Ordering::Relaxed);
        if std::env::var_os("LLM170_GPU_TIME").is_some() {
            eprintln!(
                "# pool 신규 {bytes}B 풀누적 {:.2}GiB 상주가중치 --",
                self.total.load(std::sync::atomic::Ordering::Relaxed) as f64 / (1 << 30) as f64,
            );
        }
        Ok(h)
    }

    /// 반납 — 영속 재사용, 해제 없음 (drop·지연 해제 경합 원천 차단).
    pub fn release(&self, hs: &[(Handle, usize)]) {
        if let Ok(mut map) = self.map.lock() {
            for (h, bytes) in hs {
                map.entry(*bytes).or_default().push(h.clone());
            }
        }
    }

    /// 풀 누적 바이트 (계측용).
    pub fn total(&self) -> usize {
        self.total.load(std::sync::atomic::Ordering::Relaxed)
    }
}
