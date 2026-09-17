//! q4acc MoE — 전문가 그룹화·순열 (q4acc/mod.rs 에서 이동, plans/78 R1).

use super::*;
use crate::rawhip::env_on;

impl Q4Acc {
    /// 진단(LLM170_MOE_HASH): moe 최종 출력(out) 해시 — DEV/HOST 경로 비교용.
    pub(super) fn moe_hash_check(&self, tag: &str, op: *mut u8, rows: usize, n_out: usize) -> Result<(), String> {
        if !env_on("LLM170_MOE_HASH") {
            return Ok(());
        }
        // out의 행 수는 토큰 수(t) — rows는 t·k_sel이므로 rows/k_sel… 대신
        // 버퍼 규약상 out은 [t][n_out]이고 t = frame t_cur.
        let t = self.t_cur().max(1);
        let n = (t * n_out).min(rows * n_out);
        let mut v = vec![0.0f32; n];
        self.ctx.d2h(bytemuck::cast_slice_mut(&mut v), op as *const u8)?;
        let mut x = 0xcbf29ce484222325u64;
        for f in v.iter() {
            x ^= f.to_bits() as u64;
            x = x.wrapping_mul(0x100000001b3);
        }
        static SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let q = SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        eprintln!("# moe-hash {tag} #{q} t={t} n_out={n_out} h={x:016x}");
        Ok(())
    }
    /// 행 순열 gather: dst[g] = src[perm[g]] (row_u32 = 행당 u32 수).
    /// 산란은 역순열을 넘겨 같은 커널로 수행한다.
    /// 디바이스 그룹화 런치 — q4_moe_group_t1(단일 블록·단일 스레드).
    /// 호스트 왕복(동기 d2h + 테이블 빌드 + h2d 3회)을 대체한다. 테이블은
    /// ids의 순수 함수이므로 결과는 호스트판과 동일(비트 동일).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn moe_group_dev(
        &self,
        ids: u64,
        ne: usize,
        rows: usize,
        off_d: u64,
        perm_d: u64,
        inv_d: u64,
        rowexp_d: u64,
        perm_pad_d: u64,
        inv_pad_d: u64,
        tilexp_d: u64,
        rows_pad_d: u64,
        bound: usize,
    ) -> Result<(), String> {
        let mut ip = self.fptr(ids)?;
        let (mut od, mut pd, mut iv) = (off_d as *mut u8, perm_d as *mut u8, inv_d as *mut u8);
        let (mut rx, mut pp, mut ipd) =
            (rowexp_d as *mut u8, perm_pad_d as *mut u8, inv_pad_d as *mut u8);
        let (mut tx, mut rpd) = (tilexp_d as *mut u8, rows_pad_d as *mut u8);
        let (mut n_e, mut rws) = (ne as i32, rows as i32);
        let mut bnd = bound as i32;
        let mut args: Vec<*mut std::ffi::c_void> = vec![
            (&mut ip) as *mut _ as *mut std::ffi::c_void,
            (&mut n_e) as *mut _ as *mut std::ffi::c_void,
            (&mut rws) as *mut _ as *mut std::ffi::c_void,
            (&mut od) as *mut _ as *mut std::ffi::c_void,
            (&mut pd) as *mut _ as *mut std::ffi::c_void,
            (&mut iv) as *mut _ as *mut std::ffi::c_void,
            (&mut rx) as *mut _ as *mut std::ffi::c_void,
            (&mut pp) as *mut _ as *mut std::ffi::c_void,
            (&mut ipd) as *mut _ as *mut std::ffi::c_void,
            (&mut tx) as *mut _ as *mut std::ffi::c_void,
            (&mut rpd) as *mut _ as *mut std::ffi::c_void,
            (&mut bnd) as *mut _ as *mut std::ffi::c_void,
        ];
        self.ctx.launch3("q4_moe_group_t1", 1, 1, 1, 128, &mut args)
    }


    pub(super) fn rows_permute(
        &self,
        src: *mut u8,
        perm: &[u32],
        dst: *mut u8,
        row_u32: usize,
        n: usize,
    ) -> Result<(), String> {
        if n == 0 || row_u32 == 0 {
            return Ok(());
        }
        let pd = {
            let mut g = self.rperm.lock().map_err(|e| e.to_string())?;
            g.ensure(&self.ctx, n * 4)?
        };
        self.ctx.h2d(pd, bytemuck::cast_slice(perm))?;
        self.rows_permute_dev(src, pd, dst, row_u32, n)
    }

    /// 디바이스 순열판 — 순열이 이미 GPU에 있으면 h2d/동기 없이 런치만 한다.
    pub(super) fn rows_permute_dev(
        &self,
        src: *mut u8,
        perm_d: *mut u8,
        dst: *mut u8,
        row_u32: usize,
        n: usize,
    ) -> Result<(), String> {
        if n == 0 || row_u32 == 0 {
            return Ok(());
        }
        let (mut a, mut b, mut c) = (src, perm_d, dst);
        let (mut ru, mut nn) = (row_u32 as i32, n as i32);
        self.kop(
            "q4_rows_permute_u32",
            n as u32,
            1,
            1,
            128,
            &mut cargs!(&mut a, &mut b, &mut c, &mut ru, &mut nn),
        )
    }
}
