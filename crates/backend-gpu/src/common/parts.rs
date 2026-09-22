//! parts — 대형 가중치 파일 파트의 pread 스테이징 (hip·vk 공용, plans/86 §6).
//!
//! mmap 폴트 경로는 페이지 폴트당 4KB 랜덤 읽기라 20-180 MB/s에 그친다.
//! 파트 범위가 알려진 텐서는 파일에서 8 MiB 순차 pread로 채운다
//! (버퍼드 pread 실측 ~1.2 GB/s — llama.cpp의 스테이징 패치와 동일 원리).

use std::os::unix::fs::FileExt;

/// 파트 파일 소스 — mmap 베이스 주소 범위 + 파일 핸들.
#[repr(C)]
pub struct PartSource {
    /// mmap 시작 주소(파일 시작 기준).
    pub base: usize,
    /// 이 파트의 mmap 길이(바이트).
    pub len: usize,
    pub file: std::fs::File,
}

impl PartSource {
    /// [ptr, ptr+len) 을 이 파트가 포함하면 파일 오프셋, 아니면 None.
    pub fn covers(&self, ptr: usize, len: usize) -> Option<u64> {
        let end = ptr.checked_add(len)?;
        (ptr >= self.base && end <= self.base + self.len).then(|| (ptr - self.base) as u64)
    }
}

/// 8 MiB 순차 pread로 호스트 매핑 버퍼를 채운다 — vk staged_fill 의 공용판.
pub fn pread_fill(file: &std::fs::File, dst: *mut u8, mut off: u64, len: usize) -> Result<(), String> {
    const CH: usize = 8 << 20;
    let mut done = 0usize;
    while done < len {
        let n = CH.min(len - done);
        unsafe {
            file.read_exact_at(std::slice::from_raw_parts_mut(dst.add(done), n), off)
                .map_err(|e| format!("pread {off}: {e}"))?;
        }
        done += n;
        off += n as u64;
    }
    Ok(())
}
