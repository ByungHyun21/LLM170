//! [R1 2026-10-10] W4a16Dec 업로드·등록 — mod.rs 분해.
//! 선형/플레인/head 업로드, norm·GDN·attn·MoE 등록/테이블.

use super::*;

impl W4a16Dec {
    /// 선형 상주 업로드 — packed u32 [n][k/8] · scale f16 [n][k/128].
    pub fn upload_lin(
        &mut self,
        name: &str,
        q: &[u8],
        s: &[u8],
        n: usize,
        k: usize,
    ) -> Result<(), String> {
        self.upload_lin_g(name, q, s, n, k, 128, false)
    }

    /// [gemma4 P2-3 2026-10-14] 그룹·스케일 dtype 지정 상주 업로드 —
    /// packed u32 [n][k/8] · scale [n][k/group]×2B(f16/bf16). GEMV/GEMM 런처는
    /// `gptq4::kernel_sym`이 (group, scale_bf16)로 심볼을 고른다(런치 구성 동일 —
    /// `w4a16-gemv`/`w4a16-gemm` 게이트와 동일 표).
    pub fn upload_lin_g(
        &mut self,
        name: &str,
        q: &[u8],
        s: &[u8],
        n: usize,
        k: usize,
        group: usize,
        scale_bf16: bool,
    ) -> Result<(), String> {
        // 워커 스레드 대비 컨텍스트 가드(할당·h2d — 종전 upload_lin 무가드 클래스).
        let _g = self.cc.guard()?;
        // 커널 표에 없는 조합은 등록 단계에서 거른다(런타임 심볼 실패 방지).
        crate::rawcuda::gptq4::kernel_sym(false, group, scale_bf16)?;
        if q.len() != n * (k / 2) || s.len() != n * (k / group) * 2 {
            return Err(format!(
                "upload_lin_g {name}: q={} s={} != n{n} k{k} g{group} 계약",
                q.len(),
                s.len()
            ));
        }
        // G2: alloc 전량 성공 → h2d 성공 시에만 상주 등록. 어느 단계든 실패하면
        // 성공분을 회수한다(부분 업로드 유실 금지 — 재시도가 처음부터).
        let dq = self.cc.alloc(q.len())?;
        let ds = match self.cc.alloc(s.len()) {
            Ok(p) => p,
            Err(e) => {
                let _ = self.cc.free(dq);
                return Err(e);
            }
        };
        let r = (|| {
            Self::h2d_chunked(&self.cc, dq, q)?;
            Self::h2d_chunked(&self.cc, ds, s)
        })();
        if let Err(e) = r {
            let _ = self.cc.free(dq);
            let _ = self.cc.free(ds);
            return Err(e);
        }
        self.weights_bytes += (q.len() + s.len()) as u64;
        if let Some((oq, os, _, _)) = self.lins.insert(name.to_string(), (dq, ds, n, k)) {
            let _ = self.cc.free(oq);
            let _ = self.cc.free(os);
        }
        if group == 128 && !scale_bf16 {
            self.lin_grp.remove(name); // 기본값 — 기재 불요
        } else {
            self.lin_grp.insert(name.to_string(), (group, scale_bf16));
        }
        // [split-k marlin 디코드] 이중 버퍼 — g128·f16은 marlin 재패킹 사본을
        // 함께 상주시킨다(t=1 sk 디스패치 소비). 실패는 내부 폴백(경고만).
        // [2026-10-19] g32·bf16(gemma4 dense)도 이중 버퍼 등록 — t>16 마린
        // GEMM 소비(kernels/gemv.rs). 실패 폴백 계약은 g128과 동일.
        if (group == 128 && !scale_bf16) || (group == 32 && scale_bf16) {
            self.upload_lin_marlin(name, q, s, n, k, group, scale_bf16);
        }
        Ok(())
    }

    /// [split-k marlin 디코드 2026-10-17] 재패킹 사본 상주 등록(이중 버퍼 —
    /// 원본 row-major 유지). q = packed u32 [n][k/8] 바이트. 실패(형상 계약
    /// 위반·VRAM 부족)는 경고 후 스킵 — 디스패치가 기존 GEMV/tall로 폴백해
    /// 모델 로드는 불능이 아니게 한다(27B g128: +n*k/2B ≈ +11.7GB).
    /// [2026-10-19] g32·bf16(gemma4 dense) 일반화 — 스케일은 디코드→f32→
    /// 순열→f16 단일 경로(g128·f16은 값 동일 왕복 = 비트 순열 보존).
    #[allow(clippy::too_many_arguments)]
    fn upload_lin_marlin(
        &mut self,
        name: &str,
        q: &[u8],
        s: &[u8],
        n: usize,
        k: usize,
        group: usize,
        scale_bf16: bool,
    ) {
        let repack = llm170_core::marlin::marlin_repack_q(
            // SAFETY: q는 upload_lin_g 계약 길이(n*k/2, 4의 배수) 바이트 슬라이스.
            unsafe { std::slice::from_raw_parts(q.as_ptr() as *const u32, n * k / 8) },
            n,
            k,
        );
        let qm = match repack {
            Ok(v) => v,
            Err(e) => {
                eprintln!("[marlin] {name}: 재패킹 스킵({e}) — GEMV 폴백");
                return;
            }
        };
        // SAFETY: 로컬 Vec<u32>의 유효 수명 내 바이트 뷰(길이 = 원소수×4).
        let qb = unsafe { std::slice::from_raw_parts(qm.as_ptr() as *const u8, qm.len() * 4) };
        let dqm = match self.cc.alloc(qb.len()) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("[marlin] {name}: 재패킹 VRAM 부족({e}) — GEMV 폴백");
                return;
            }
        };
        if let Err(e) = Self::h2d_chunked(&self.cc, dqm, qb) {
            let _ = self.cc.free(dqm);
            eprintln!("[marlin] {name}: 재패킹 업로드 실패({e}) — GEMV 폴백");
            return;
        }
        self.weights_bytes += qb.len() as u64;
        self.marlin_bytes += qb.len() as u64;
        if let Some((old, _, _)) = self.lins_mk.insert(name.to_string(), (dqm, n, k)) {
            let _ = self.cc.free(old);
            self.weights_bytes = self.weights_bytes.saturating_sub((n * k / 2) as u64);
            self.marlin_bytes = self.marlin_bytes.saturating_sub((n * k / 2) as u64);
        }
        // [2026-10-19] 마린 스케일 순열 사본([kg][n] f16) — gptq 64-청크 순열.
        // 단일 경로: 디코드(f16/bf16→f32 정확) → marlin_permute_scales(gs=group)
        // → f16 인코딩. g128·f16은 값 왕복이 비트 보존(종전 인라인 비트 순열과
        // 동일 결과), g32·bf16은 marlin_moe s13/s2(perm16)과 동일 계약 —
        // 스케일 f16 정밀도 클래스(eval 게이트 판정 대상).
        let kg = k / group;
        if s.len() == n * kg * 2 {
            let sf: Vec<f32> = s
                .as_chunks::<2>()
                .0
                .iter()
                .map(|c| {
                    let b = u16::from_le_bytes(*c);
                    if scale_bf16 {
                        llm170_core::quant::deq::bf16_to_f32(b)
                    } else {
                        llm170_core::quant::deq::half_to_f32(b)
                    }
                })
                .collect();
            let so: Vec<u8> = match llm170_core::marlin::marlin_permute_scales(&sf, n, k, group) {
                Ok(pf) => {
                    let mut out = vec![0u8; pf.len() * 2];
                    for (i, v) in pf.iter().enumerate() {
                        out[i * 2..i * 2 + 2]
                            .copy_from_slice(&Self::f32_to_f16_bits(*v).to_le_bytes());
                    }
                    out
                }
                Err(_) => Vec::new(),
            };
            if !so.is_empty()
                && let Ok(dsc) = self.cc.alloc(so.len())
            {
                if Self::h2d_chunked(&self.cc, dsc, &so).is_ok() {
                    self.weights_bytes += so.len() as u64;
                    self.marlin_bytes += so.len() as u64;
                    if let Some((old, _, _)) = self.lins_msc.insert(name.to_string(), (dsc, n, kg))
                    {
                        let _ = self.cc.free(old);
                        self.weights_bytes = self.weights_bytes.saturating_sub(so.len() as u64);
                        self.marlin_bytes = self.marlin_bytes.saturating_sub(so.len() as u64);
                    }
                } else {
                    let _ = self.cc.free(dsc);
                }
            }
        }
    }

    // ── norm ──

    /// 노름 가중 상주 등록 — nw [rows][hidden] f32(행 포인터 계약 w·hidden).
    /// hidden 계약 = (0, 8192] — 커널이 1024 배수면 `norm_resid`, 아니면
    /// 행 경계 가드 변형(`norm_resid_g`, gemma4 2816)을 쓴다(선택은
    /// `norm_resid_at`). 종전 1024 배수 강제는 스트라이드 커널 전용 가정이었다.
    pub fn set_norm_weights(&mut self, nw: &[f32], rows: usize) -> Result<(), String> {
        if self.hidden == 0 || self.hidden > 8192 {
            return Err(format!(
                "norm: hidden={} — (0, 8192] 계약(v[8] 레지스터 상한)",
                self.hidden
            ));
        }
        if rows == 0 || nw.len() != rows * self.hidden {
            return Err(format!(
                "norm: nw {} != rows {rows} × hidden {}",
                nw.len(),
                self.hidden
            ));
        }
        if self.dnw != 0 {
            self.cc.free(self.dnw)?;
        }
        self.dnw = 0; // G1: 실패 시 재시도가 0을 해제하지 않게.
        self.norm_w_rows = 0;
        // SAFETY: 로컬 슬라이스의 유효 수명 내 바이트 뷰(길이 = 원소수×4).
        let b = unsafe { std::slice::from_raw_parts(nw.as_ptr() as *const u8, nw.len() * 4) };
        let d = self.cc.alloc(nw.len() * 4)?;
        if let Err(e) = Self::h2d_chunked(&self.cc, d, b) {
            let _ = self.cc.free(d); // G1: 부분 업로드 실패 시 유실 방지.
            return Err(e);
        }
        self.dnw = d;
        self.norm_w_rows = rows;
        Ok(())
    }

    // ── GDN ──

    pub fn set_gdn(
        &mut self,
        dims: GdnDims,
        cw: &[f32],
        ab: &[f32],
        alog: &[f32],
        dtb: &[f32],
        nw: &[f32],
    ) -> Result<(), String> {
        let (n, cch, hv, hd) = (dims.n_gdn, dims.conv_ch(), dims.h_v, dims.hidden);
        if cw.len() != n * cch * 4
            || ab.len() != n * 2 * hv * hd
            || alog.len() != n * hv
            || dtb.len() != n * hv
            || nw.len() != n * 128
        {
            return Err("GDN: 상수 형상 계약 위반(cw/ab/alog/dtb/nw)".into());
        }
        for q in [
            self.gdn.dcw,
            self.gdn.dab_c,
            self.gdn.dalog,
            self.gdn.ddtb,
            self.gdn.dnwg,
            self.gdn.dring,
            self.gdn.dgst,
        ] {
            if q != 0 {
                self.cc.free(q)?;
            }
        }
        (
            self.gdn.dcw,
            self.gdn.dab_c,
            self.gdn.dalog,
            self.gdn.ddtb,
            self.gdn.dnwg,
        ) = (0, 0, 0, 0, 0);
        (self.gdn.dring, self.gdn.dgst) = (0, 0);
        let b =
            // SAFETY: 로컬 슬라이스의 유효 수명 내 바이트 뷰(길이 = 원소수×4).
            |v: &[f32]| unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) };
        let dcw = self.cc.alloc(cw.len() * 4)?;
        Self::h2d_chunked(&self.cc, dcw, b(cw))?;
        let dab = self.cc.alloc(ab.len() * 4)?;
        Self::h2d_chunked(&self.cc, dab, b(ab))?;
        let dal = self.cc.alloc(alog.len() * 4)?;
        Self::h2d_chunked(&self.cc, dal, b(alog))?;
        let ddt = self.cc.alloc(dtb.len() * 4)?;
        Self::h2d_chunked(&self.cc, ddt, b(dtb))?;
        let dnw = self.cc.alloc(nw.len() * 4)?;
        Self::h2d_chunked(&self.cc, dnw, b(nw))?;
        let slots = self.n_slots;
        let dring = self.cc.alloc(slots * n * 3 * cch * 4)?;
        Self::zero_dev(&self.cc, dring, slots * n * 3 * cch * 4)?;
        // [행-스케일 2026-10-13] 라이브 상태 = f16(2B) + 행별 스케일 f32.
        let dgst = self.cc.alloc(slots * n * hv * 128 * 128 * 2)?;
        Self::zero_dev(&self.cc, dgst, slots * n * hv * 128 * 128 * 2)?;
        let dsc = self.cc.alloc(slots * n * hv * 128 * 4)?;
        let ones = vec![1.0f32; slots * n * hv * 128];
        // SAFETY: 로컬 슬라이스의 유효 수명 내 바이트 뷰(길이 = 원소수×4).
        let onesb =
            unsafe { std::slice::from_raw_parts(ones.as_ptr() as *const u8, ones.len() * 4) };
        Self::h2d_chunked(&self.cc, dsc, onesb)?;
        // [P1 2026-10-16] 단일 슬롯분 ones — reset_state의 d2d 원본(적재 1회).
        let dsc_ones = self.cc.alloc(n * hv * 128 * 4)?;
        Self::h2d_chunked(&self.cc, dsc_ones, &onesb[..n * hv * 128 * 4])?;
        (
            self.gdn.dcw,
            self.gdn.dab_c,
            self.gdn.dalog,
            self.gdn.ddtb,
            self.gdn.dnwg,
        ) = (dcw, dab, dal, ddt, dnw);
        self.gdn.dring = dring;
        self.gdn.dgst = dgst;
        self.gdn.dsc = dsc;
        self.gdn.dsc_ones = dsc_ones;
        self.gdn.gdn = Some(dims);
        Ok(())
    }

    /// [K6-3 예산] KV 아레나 예산 설정 — set_attn 전에 호출(미호출 = 종전 총량).
    /// 서버가 여유 VRAM에서 파생: 풀 총량 = min(예산/블록비용, 슬롯×ctx). 풀 총량
    /// ≥ 단일 요청 전역 ctx(입장 하드 보장)는 서버 가드 몫.
    pub fn set_kv_budget(&mut self, bytes: Option<u64>) {
        self.attn.kv_budget = bytes;
    }

    /// [2026-10-16] VRAM 티어 상한(바이트 — 서버 파생) — kv_budget 초과분은
    /// 호스트 아레나로 확장(2계층). None = 전량 VRAM(종전 동작).
    pub fn set_kv_vram_cap(&mut self, bytes: Option<u64>) {
        self.attn.kv_vram_cap = bytes;
    }

    /// 블록 id → 아레나 주소(2계층) — `id < vram_blocks` = VRAM 아레나,
    /// 이후는 호스트 아레나(핀드·디바이스 매핑). 표 기입·직접 참조 공용.
    pub(super) fn kv_block_addr(
        base_v: CUdeviceptr,
        base_h: CUdeviceptr,
        id: u64,
        stride: u64,
        vram_blocks: usize,
    ) -> u64 {
        if (id as usize) < vram_blocks {
            base_v + id * stride
        } else {
            base_h + (id - vram_blocks as u64) * stride
        }
    }

    /// 호스트 KV 아레나 해제(핀드 — cc 대장 밖이라 수동) + 필드 0화.
    /// ucache 호스트 항목도 함께 해제(재등록 경로 공용).
    pub(super) fn free_kv_host_arenas(&mut self) -> Result<(), String> {
        self.ucache.release_if_present("attn:kv:host");
        for p in [
            self.attn.hkc_h,
            self.attn.hvc_h,
            self.attn.hksc_h,
            self.attn.hvsc_h,
        ] {
            if !p.is_null() {
                self.cc.pinned_free(p)?;
            }
        }
        self.attn.hkc_h = std::ptr::null_mut();
        self.attn.hvc_h = std::ptr::null_mut();
        self.attn.hksc_h = std::ptr::null_mut();
        self.attn.hvsc_h = std::ptr::null_mut();
        (self.attn.dkc_h, self.attn.dvc_h) = (0, 0);
        (self.attn.dksc_h, self.attn.dvsc_h) = (0, 0);
        self.attn.kv_host_bytes = 0;
        self.attn.kv_vram_blocks = 0;
        Ok(())
    }

    /// [2026-10-16] 티어별 **자유** 블록 — 운영·컨트롤러 신호(배치 수가 아니라
    /// 가용: VRAM이 비었는데 호스트가 점유 중인지 = 오프로드 점착·순서 편중 관측).
    /// (VRAM 자유, 호스트 자유).
    pub fn kv_tier_free(&self) -> (usize, usize) {
        self.attn
            .kv_pool
            .as_ref()
            .map(|p| (p.free_vram_blocks(), p.free_host_blocks()))
            .unwrap_or((0, 0))
    }

    /// [2026-10-16] KV 계층 회계 — (VRAM 바이트, 호스트 바이트, 호스트 블록,
    /// 전체 블록). 오프로드율 = 호스트/(VRAM+호스트) — /stats 노출.
    pub fn kv_tier_bytes(&self) -> (u64, u64, usize, usize) {
        let total = self
            .attn
            .kv_pool
            .as_ref()
            .map(|p| p.total_blocks())
            .unwrap_or(0);
        let hb = total.saturating_sub(self.attn.kv_vram_blocks);
        (self.attn.kv_arena_bytes, self.attn.kv_host_bytes, hb, total)
    }

    /// [gemma4 2026-10-14] `layers` = 층별 어텐션 파라미터(빈 벡터 = 균일
    /// 모델 — qwen35 종전 경로와 완전 동일). QK 노름 가중은 층별 head_dim의
    /// **최대치 보폭**(d_max)으로 패딩된 배열을 받는다(균일 모델은 n×d 동일).
    pub fn set_attn(
        &mut self,
        dims: AttnDims,
        layers: Vec<AttnLayer>,
        qnw: &[f32],
        knw: &[f32],
    ) -> Result<(), String> {
        let n = dims.n_attn;
        if !layers.is_empty() && layers.len() != n {
            return Err(format!("attn: layers {} != n_attn {n}", layers.len()));
        }
        let d_max = layers.iter().map(|l| l.d).max().unwrap_or(dims.d);
        if qnw.len() != n * d_max || knw.len() != n * d_max {
            return Err(format!("attn: qnw/knw != {n}x{d_max}"));
        }
        let slots = self.n_slots;
        // [K6-3 2026-10-13] KV 물리 풀 — 평탄 아레나 4종(k/v/ksc/vsc).
        // 블록 = 512토큰 × 1층. 현 단계: 총량 = 종전 용량(slots×n×mb)과 동일·
        // 전량 선할당(동작 보존), 다음 단계에서 예산 기반 온디맨드로 전환.
        // [gemma4 2026-10-14] 아레나·블록 = **층별 최대 클래스**(층별 크기 혼합
        // 모델 대응 — 여유분 미사용. 정확 크기 클래스화는 후속 최적화).
        // 균일 모델은 스칼라 필드와 동일(무해).
        let (kvd, kvh) = if layers.is_empty() {
            (dims.kv_dim(), dims.kv_heads)
        } else {
            (
                layers
                    .iter()
                    .map(|l| l.kv_heads * l.d)
                    .max()
                    .unwrap_or(dims.kv_dim()),
                layers
                    .iter()
                    .map(|l| l.kv_heads)
                    .max()
                    .unwrap_or(dims.kv_heads),
            )
        };
        let mb = dims.cap.div_ceil(512);
        let need_blocks = slots * n * mb;
        let block_k = 512 * kvd; // 아레나 k/v 블록 바이트(1B/원소 int8)
        let block_s = 512 * kvh * 4; // 아레나 ksc/vsc 블록 바이트(f32)
        // [K6-3 예산] 풀 총량 — 예산이 있으면 블록 비용(4아레나 합)으로 환산해 상한.
        let per_block = 2 * (block_k + block_s) as u64;
        let total_blocks = match self.attn.kv_budget {
            Some(b) => ((b / per_block.max(1)) as usize).clamp(1, need_blocks),
            None => need_blocks,
        };
        // [2026-10-16] 2계층 분할 — VRAM 상한(서버 파생) 초과분은 호스트 아레나
        // (핀드·디바이스 매핑 — 커널이 표를 통해 직접 접근). VRAM 선순위(블록 id
        // 오름차순 + KvPool 티어 스택). 유계 층(슬라이딩 창) 우선 호스트 배치는
        // 차기(§9.1의 "우선 후보" — 현행은 무계 포함 허용, 요청 우선 규칙).
        // 진단 강제 = LLM170_KV_VRAM_MB — **압박 테스트 하네스·프로파일 전용**
        // (배치 파라미터 노브 — 수치 불변, 경로 선택 아님. 하네스가 설정한다).
        let diag_vram = llm170_diag::flag::val("LLM170_KV_VRAM_MB")
            .and_then(|v| v.parse::<u64>().ok())
            .map(|mb| mb << 20);
        let vram_cap = diag_vram.or(self.attn.kv_vram_cap);
        let vram_blocks = match vram_cap {
            Some(b) => ((b / per_block.max(1)) as usize).clamp(1, total_blocks),
            None => total_blocks,
        };
        let host_blocks = total_blocks - vram_blocks;
        // [§3.4 ②] KV 재등록 — 구 회계 해제(부재 시 무동작: 최초 등록).
        self.ucache.release_if_present("attn:kv");
        self.free_kv_host_arenas()?;
        for q in [
            self.attn.dqnw_a,
            self.attn.dknw_a,
            self.attn.dkc,
            self.attn.dvc,
            self.attn.dksc,
            self.attn.dvsc,
            self.attn.dblk_tab,
            self.attn.dpp,
            self.attn.dqg_a,
            self.attn.dkin_a,
            self.attn.dvin_a,
            self.attn.dqh_a,
            self.attn.doutv_a,
        ] {
            if q != 0 {
                self.cc.free(q)?;
            }
        }
        (
            self.attn.dqnw_a,
            self.attn.dknw_a,
            self.attn.dkc,
            self.attn.dvc,
            self.attn.dpp,
        ) = (0, 0, 0, 0, 0);
        (self.attn.dksc, self.attn.dvsc) = (0, 0);
        self.attn.dblk_tab = 0;
        self.attn.kv_pool = None;
        (
            self.attn.dqg_a,
            self.attn.dkin_a,
            self.attn.dvin_a,
            self.attn.dqh_a,
            self.attn.doutv_a,
        ) = (0, 0, 0, 0, 0);
        self.attn.attn_t_cap = 0;
        let b =
            // SAFETY: 로컬 슬라이스의 유효 수명 내 바이트 뷰(길이 = 원소수×4).
            |v: &[f32]| unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) };
        let dq = self.cc.alloc(qnw.len() * 4)?;
        // [2026-10-16 검토 반영] alloc 직후 즉시 필드 선점 — h2d 실패 창까지 폐쇄
        // (이후 어느 오류경로든 필드 소유·free_all/재등록 정리가 회수).
        self.attn.dqnw_a = dq;
        Self::h2d_chunked(&self.cc, dq, b(qnw))?;
        let dk = self.cc.alloc(knw.len() * 4)?;
        self.attn.dknw_a = dk;
        Self::h2d_chunked(&self.cc, dk, b(knw))?;
        let pool = super::blocktable::KvPool::new(slots, n, mb, total_blocks, vram_blocks)?;
        let (dkc, dvc) = {
            // int8 KV(1B/원소) + 스케일 2벌. 기록 attn_prep_q,
            // 판독 attn_fwd3s_part_q(병합은 f32 그대로).
            let kbytes = vram_blocks * block_k;
            let kc = self.cc.alloc(kbytes)?;
            Self::zero_dev(&self.cc, kc, kbytes)?;
            let vc = self.cc.alloc(kbytes)?;
            Self::zero_dev(&self.cc, vc, kbytes)?;
            let sbytes = vram_blocks * block_s;
            let ks = self.cc.alloc(sbytes)?;
            Self::zero_dev(&self.cc, ks, sbytes)?;
            let vs = self.cc.alloc(sbytes)?;
            Self::zero_dev(&self.cc, vs, sbytes)?;
            self.attn.dksc = ks;
            self.attn.dvsc = vs;
            (kc, vc)
        };
        // 호스트 확장 아레나(있으면) — 핀드·디바이스 매핑. 실패 시 방금 만든
        // 장치 아레나까지 회수(부분 상태 금지).
        if host_blocks > 0 {
            let hk = host_blocks * block_k;
            let hs = host_blocks * block_s;
            // 부분 실패 시 앞선 핀드 할당을 즉시 회수(대장 밖 메모리 — 누수 금지).
            let alloc4 = || -> Result<_, String> {
                let (hkc, dkc_h) = self.cc.pinned_alloc_device(hk)?;
                let (hvc, dvc_h) = match self.cc.pinned_alloc_device(hk) {
                    Ok(v) => v,
                    Err(e) => {
                        let _ = self.cc.pinned_free(hkc);
                        return Err(e);
                    }
                };
                let (hksc, dksc_h) = match self.cc.pinned_alloc_device(hs) {
                    Ok(v) => v,
                    Err(e) => {
                        let _ = self.cc.pinned_free(hkc);
                        let _ = self.cc.pinned_free(hvc);
                        return Err(e);
                    }
                };
                let (hvsc, dvsc_h) = match self.cc.pinned_alloc_device(hs) {
                    Ok(v) => v,
                    Err(e) => {
                        let _ = self.cc.pinned_free(hkc);
                        let _ = self.cc.pinned_free(hvc);
                        let _ = self.cc.pinned_free(hksc);
                        return Err(e);
                    }
                };
                Ok(((hkc, dkc_h), (hvc, dvc_h), (hksc, dksc_h), (hvsc, dvsc_h)))
            };
            match alloc4() {
                Ok(((hkc, dkc_h), (hvc, dvc_h), (hksc, dksc_h), (hvsc, dvsc_h))) => {
                    // 호스트 0 초기화(커널 기록 전 판독 안전 — 1회).
                    // SAFETY: 핀드 할당 [0, len) 쓰기.
                    unsafe {
                        std::ptr::write_bytes(hkc, 0, hk);
                        std::ptr::write_bytes(hvc, 0, hk);
                        std::ptr::write_bytes(hksc, 0, hs);
                        std::ptr::write_bytes(hvsc, 0, hs);
                    }
                    self.attn.hkc_h = hkc;
                    self.attn.hvc_h = hvc;
                    self.attn.hksc_h = hksc;
                    self.attn.hvsc_h = hvsc;
                    self.attn.dkc_h = dkc_h;
                    self.attn.dvc_h = dvc_h;
                    self.attn.dksc_h = dksc_h;
                    self.attn.dvsc_h = dvsc_h;
                    self.attn.kv_host_bytes = 2 * (hk + hs) as u64;
                }
                Err(e) => {
                    for q in [dkc, dvc, self.attn.dksc, self.attn.dvsc] {
                        if q != 0 {
                            let _ = self.cc.free(q);
                        }
                    }
                    (self.attn.dkc, self.attn.dvc) = (0, 0);
                    (self.attn.dksc, self.attn.dvsc) = (0, 0);
                    return Err(e);
                }
            }
        }
        self.attn.kv_vram_blocks = vram_blocks;
        let dpp = self.cc.alloc(slots * 4)?;
        Self::zero_dev(&self.cc, dpp, slots * 4)?;
        // [K6-3 2026-10-13] KV 블록 테이블(디바이스 u64 [slot][layer][mb]×4) —
        // 엔트리 = 풀 물리 블록 주소(KvPool 정본). 커널(attn_blk_row)은 테이블만
        // 보므로 블록 배정/회수는 호스트 갱신만으로 반영된다(블록 id가 VRAM/호스트
        // 어느 아레나인지는 표 엔트리 주소가 담는다).
        // **온디맨드**: 초기 테이블은 0(미할당) — 슬롯은 입장(kv_grow)에서 필요한
        // 블록만 배정받는다(요청별 예약·회수 — K6 3차). 미할당 엔트리는 판독
        // 전에 반드시 kv_grow가 덮는다(성장 = 사용 직전 +2 정렬 여유).
        let tab = vec![0u64; slots * n * mb * 4];
        let dblk = self.cc.alloc(tab.len() * 8)?;
        // SAFETY: 로컬 Vec<u64>의 유효 수명 내 바이트 뷰(길이 = 원소수×8).
        let tb = unsafe { std::slice::from_raw_parts(tab.as_ptr() as *const u8, tab.len() * 8) };
        Self::h2d_chunked(&self.cc, dblk, tb)?;
        // [§3.4 ② 2026-10-12] 통합 캐시 회계 등록 — KV 본체(int8 k/v 2벌 +
        // 행×헤드 스케일 2벌 + 블록 테이블). **Kv 범주는 고정(퇴거 불가)**
        // 이므로 등록 실패는 예산 부족(과점유) 경고로 그친다 — 이미
        // 물리 할당이 성공했고 KV 없이는 디코드 자체가 불가능하기 때문.
        // (VRAM 아레나 = HBM 등록 · 호스트 아레나는 별도 HostRam 항목.)
        let kv_bytes = (vram_blocks * (block_k * 2 + block_s * 2)) as u64 + tab.len() as u64 * 8;
        self.attn.kv_arena_bytes = kv_bytes;
        if let Err(e) = self.ucache.allocate(
            "attn:kv",
            CacheCategory::Kv,
            Tier::Hbm,
            kv_bytes,
            base_util::KV,
        ) {
            eprintln!("[ucache] attn:kv 등록 실패(회계 전용): {e}");
        } else {
            eprintln!("{}", self.ucache.log_line("attn:kv"));
        }
        // [2026-10-16 검토 반영] 호스트 아레나도 통합 캐시 회계 등록 — 함수/바이트
        // 단위 소비자(압박 산정)가 레지스트리만 보는 경우의 공백 제거.
        if self.attn.kv_host_bytes > 0 {
            self.ucache.release_if_present("attn:kv:host");
            if let Err(e) = self.ucache.allocate(
                "attn:kv:host",
                CacheCategory::Kv,
                Tier::HostRam,
                self.attn.kv_host_bytes,
                base_util::KV,
            ) {
                eprintln!("[ucache] attn:kv:host 등록 실패(회계 전용): {e}");
            } else {
                eprintln!("{}", self.ucache.log_line("attn:kv:host"));
            }
        }
        self.attn.dblk_tab = dblk;
        self.attn.kv_pool = Some(pool);
        // dqnw_a/dknw_a는 alloc 직후 선점(오류경로 회수 — 위 참조).
        self.attn.dkc = dkc;
        self.attn.dvc = dvc;
        self.attn.dpp = dpp;
        self.attn.layers = layers;
        self.attn.attn = Some(dims);
        // [§3.4 ② 트랜셰 2] 등록 중 퇴거(ckpt·spec·expert-lru)가 결정됐을
        // 수 있다 — KV 필드 확정 후 실제 해제까지 완료. KV 자체는 고정
        // (Kv 범주)이라 이 스윕의 대상이 아니다.
        self.ucache_sweep()?;
        Ok(())
    }

    /// [K6-3 2026-10-13] 슬롯 KV 블록 확보 — `n_tokens`(+2 정렬 여유)까지
    /// 온디맨드. 신규 블록만 디바이스 테이블에 기입(동기 h2d — 틱 경계
    /// 호출 계약: 리플레이/커널 실행 중 금지). 반환 = 신규 블록 수.
    /// 테이블 상한(층당 mb)으로 클램프 — 초과분은 ctx 가드/overflow 몫.
    pub fn kv_grow(&mut self, slot: usize, n_tokens: usize) -> Result<usize, String> {
        // [K6-3] 컨텍스트 가드 — 스케줄러/워밍업 등 외부 스레드에서도 h2d 유효.
        let _g = self.cc.guard()?;
        let dm = self.attn.attn.ok_or("kv_grow: attn 미등록")?;
        if slot >= self.n_slots {
            return Err(format!("kv_grow: 슬롯 {slot} ≥ {}", self.n_slots));
        }
        let mb = dm.cap.div_ceil(512);
        let nb = (n_tokens + 2).div_ceil(512).min(mb);
        // [gemma4 2026-10-14] 최대 KV 클래스(층별 혼합 — 등록 시와 동일 규칙).
        let (kvd, kvh) = self.attn_max_kv()?;
        let (bk, bs) = ((512 * kvd) as u64, (512 * kvh * 4) as u64);
        let (kc, vc, ks, vs) = (self.attn.dkc, self.attn.dvc, self.attn.dksc, self.attn.dvsc);
        // 2계층 주소(블록 id < kv_vram_blocks = VRAM, 이후 = 호스트 아레나).
        let (kc_h, vc_h, ks_h, vs_h) = (
            self.attn.dkc_h,
            self.attn.dvc_h,
            self.attn.dksc_h,
            self.attn.dvsc_h,
        );
        let vram_blocks = self.attn.kv_vram_blocks;
        let n_attn = dm.n_attn;
        // [2.N Stage 2] 슬롯 플로어(행) — 슬라이딩 층에서 이 행 미만 블록은
        // 확보하지 않는다(해제 유지). 풀층·qwen(균일 layers 빈 벡터)은 무영향.
        let floor = self.kv_floor.get(slot).copied().unwrap_or(0) as usize;
        let mut new = 0usize;
        // [k2 (b) 2026-10-17] 층 창 필터 훅 — 창 밖 KV 층은 이 샤드가 소유하지
        // 않는다(확보 생략). MTP 가상 층(il 매핑이 n_layers를 넘는 마지막
        // 서브층)은 최종 층 소유 창이 소유(owns_attn 내부) — 전체창 = 항상
        // 참 → 동작 불변.
        let win = self.window;
        for l in 0..n_attn {
            if !win.owns_attn(l, dm.interval, self.n_layers) {
                continue;
            }
            let sliding = self.attn.layers.get(l).map(|x| x.sliding).unwrap_or(false);
            for bi in 0..nb {
                if sliding && bi * 512 + 512 <= floor {
                    continue;
                }
                let have = self
                    .attn
                    .kv_pool
                    .as_ref()
                    .ok_or("kv_grow: KvPool 미등록")?
                    .get(slot, l, bi)
                    .is_some();
                if have {
                    continue;
                }
                let id = self
                    .attn
                    .kv_pool
                    .as_mut()
                    .ok_or("kv_grow: KvPool 미등록")?
                    .ensure(slot, l, bi)? as u64;
                new += 1;
                let mut bytes = [0u8; 32];
                for (i2, v) in [
                    Self::kv_block_addr(kc, kc_h, id, bk, vram_blocks),
                    Self::kv_block_addr(vc, vc_h, id, bk, vram_blocks),
                    Self::kv_block_addr(ks, ks_h, id, bs, vram_blocks),
                    Self::kv_block_addr(vs, vs_h, id, bs, vram_blocks),
                ]
                .iter()
                .enumerate()
                {
                    bytes[i2 * 8..(i2 + 1) * 8].copy_from_slice(&v.to_le_bytes());
                }
                let off = (((slot * n_attn + l) * mb + bi) * 4) as u64 * 8;
                self.cc.h2d(self.attn.dblk_tab + off, &bytes)?;
            }
        }
        if slot < self.kv_assigned.len() {
            self.kv_assigned[slot] = self.kv_assigned[slot].max(n_tokens);
        }
        Ok(new)
    }

    /// [K6-3] 위치 기반 확보 — `pos_end`(+2 정렬 여유)까지. 고수위 캐시로
    /// 무비용 no-op. 체인/프리필/MTP/검증 **진입 공통** — 직접 호출자
    /// (프로브·infer 포함)도 안전하다(테이블 미할당 판독 방지).
    pub(super) fn kv_ensure(&mut self, slot: usize, pos_end: usize) -> Result<(), String> {
        if slot >= self.n_slots {
            return Err(format!("kv_ensure: 슬롯 {slot} ≥ {}", self.n_slots));
        }
        if pos_end <= self.kv_assigned[slot] {
            return Ok(());
        }
        self.kv_grow(slot, pos_end)?;
        Ok(())
    }

    /// [K6-3] 슬롯 KV 반납(리셋 경로 전용 — 정상 완료는 접두 캐시로 유지).
    pub fn kv_release(&mut self, slot: usize) -> usize {
        if slot < self.kv_assigned.len() {
            self.kv_assigned[slot] = 0;
        }
        if slot < self.kv_floor.len() {
            self.kv_floor[slot] = 0;
        }
        self.attn
            .kv_pool
            .as_mut()
            .map(|p| p.release(slot))
            .unwrap_or(0)
    }

    /// 슬롯 플로어(행) 조회 — 0 = 미트림. 엔진 재사용 가드(reuse_valid) 입력.
    pub fn kv_floor_of(&self, slot: usize) -> u32 {
        self.kv_floor.get(slot).copied().unwrap_or(0)
    }

    /// [2.N Stage 2] 플로어 전진 — 슬라이딩 층의 `floor_rows` 미만 블록을
    /// 풀에 반납하고 디바이스 테이블 엔트리를 0으로 지운다(틱 경계 호출 계약:
    /// 커널 실행 중 금지 — kv_grow와 동일). 단조(전진만 — 후퇴 요청은 무시).
    /// 풀층은 손대지 않는다. 반환 = 해제 블록 수.
    ///
    /// 해제 행 판독 안전: 디코드 part는 `[lim−sw, lim)`만 읽고(커널이 하한을
    /// 강제), FA2 프리필은 타일 전 열을 읽되 마스크 밖 점수는 무조건 −1e30
    /// 대입이라 기여 0이다. 단 해제 행이 **미래에 비마스크로 읽히면** 오염 —
    /// 엔진 복원 가드(`floor ≤ l`·`floor ≤ prompt−sw`)가 그 경우를 콜드로
    /// 돌리므로 이 함수는 가드 통과 전제다. 기록도 접미사 행(≥ l ≥ floor)에만
    /// 닿는다.
    pub fn kv_release_below(&mut self, slot: usize, floor_rows: u32) -> Result<usize, String> {
        let _g = self.cc.guard()?;
        let dm = self.attn.attn.ok_or("kv_release_below: attn 미등록")?;
        if slot >= self.n_slots {
            return Err(format!("kv_release_below: 슬롯 {slot} ≥ {}", self.n_slots));
        }
        if floor_rows <= self.kv_floor_of(slot) {
            return Ok(0);
        }
        let mb = dm.cap.div_ceil(512);
        let n_attn = dm.n_attn;
        // 완전 포함 블록만(내림) — 비양자화 입력에도 행 손실 없음(엔진은 512 양자화 전달).
        let floor_bi = (floor_rows as usize) / 512;
        let sliding: Vec<bool> = (0..n_attn)
            .map(|l| self.attn.layers.get(l).map(|x| x.sliding).unwrap_or(false))
            .collect();
        let rel = self
            .attn
            .kv_pool
            .as_mut()
            .ok_or("kv_release_below: KvPool 미등록")?
            .release_below(slot, floor_bi, &sliding);
        if rel.is_empty() {
            self.kv_floor[slot] = floor_rows;
            return Ok(0);
        }
        let zero = [0u8; 32];
        for (l, bi) in &rel {
            let off = (((slot * n_attn + l) * mb + bi) * 4) as u64 * 8;
            self.cc.h2d(self.attn.dblk_tab + off, &zero)?;
        }
        self.kv_floor[slot] = floor_rows;
        Ok(rel.len())
    }

    /// [K6-3 예산] 실 KV 아레나+테이블 바이트(0 = 미등록).
    pub fn kv_pool_bytes(&self) -> u64 {
        self.attn.kv_arena_bytes
    }

    /// [K6-3] 풀 총 블록 수(용량 — 초과 예약 조기 거부).
    pub fn kv_total_blocks(&self) -> usize {
        self.attn
            .kv_pool
            .as_ref()
            .map(|p| p.total_blocks())
            .unwrap_or(0)
    }

    /// [K6-3] 풀 여유 블록 수(입장 판정·모니터링).
    pub fn kv_free_blocks(&self) -> usize {
        self.attn
            .kv_pool
            .as_ref()
            .map(|p| p.free_blocks())
            .unwrap_or(0)
    }

    /// [K6-3] n_tokens(+2 정렬 여유) 예약에 필요한 블록 수 — 입장 판정용.
    /// 예약 슬랙(8 — 스펙 전방 tmax−1)은 호출부 예약 몫(`sched::admit::RESERVE_SLACK`,
    /// `need_tokens`가 포함) — 여기의 +2는 정렬 여유일 뿐 이중 계상이 아니다.
    pub fn kv_blocks_needed(&self, n_tokens: usize) -> usize {
        match self.attn.attn {
            Some(dm) => {
                let mb = dm.cap.div_ceil(512);
                (n_tokens + 2).div_ceil(512).min(mb) * dm.n_attn
            }
            None => 0,
        }
    }

    /// [2026-10-16] 실블록 유래 단일 요청 상한(토큰) — `kv_blocks_needed(n)
    /// ≤ total`의 최대 n(= ⌊total/n_attn⌋×512 − 2; +2는 grow 정렬 여유).
    /// 풀 floor(블록 잔여)를 반영해 상한 표기·OAI 입장 전 거부를 실블록과 일치.
    pub fn kv_cap_tokens(&self) -> usize {
        match self.attn.attn {
            Some(dm) if dm.n_attn > 0 => {
                ((self.kv_total_blocks() / dm.n_attn) * 512).saturating_sub(2)
            }
            _ => 0,
        }
    }

    /// [2026-10-16] 슬롯 현재 점유 블록 — aging 예측 입력(완료 시 해제량).
    pub fn kv_used_blocks(&self, slot: usize) -> usize {
        self.attn
            .kv_pool
            .as_ref()
            .map(|p| p.used_blocks(slot))
            .unwrap_or(0)
    }

    /// 플레인 bf16 [n][k] 상주 업로드 — [n][k] 원본(w4a16_gemv_bf16 계약).
    pub fn upload_plain(
        &mut self,
        name: &str,
        data: &[u8],
        n: usize,
        k: usize,
    ) -> Result<(), String> {
        let _g = self.cc.guard()?;
        let need = n * k * 2;
        if n == 0 || k == 0 || data.len() < need {
            return Err(format!(
                "upload_plain({name}): 형상 계약 위반 n={n} k={k} bytes={} < {need}",
                data.len()
            ));
        }
        // [n][k] 원본 그대로 — w4a16_gemv_bf16(행=블록) 계약(전치 없음).
        let dw = self.cc.alloc(need)?;
        if let Err(e) = Self::h2d_chunked(&self.cc, dw, &data[..need]) {
            let _ = self.cc.free(dw);
            return Err(e);
        }
        self.weights_bytes += need as u64;
        if let Some((old, _, _)) = self.plains.insert(name.to_string(), (dw, n, k)) {
            let _ = self.cc.free(old);
        }
        Ok(())
    }

    /// 플레인 모드 전환(전문가 외 전부 bf16인 MoE 모델).
    pub fn set_plain_mode(&mut self, on: bool) {
        self.plain_weights = on;
    }

    /// MoE 구성 등록 — n_experts>0이면 체인은 플레인+MoE FFN 경로.
    pub fn set_moe(
        &mut self,
        n_experts: usize,
        top_k: usize,
        moe_ffn: usize,
        shared_ffn: usize,
        group: usize,
        scale_bf16: bool,
        glu_gelu: bool,
    ) {
        self.moe.glu_gelu = glu_gelu;
        self.moe.n_experts = n_experts;
        self.moe.top_k = top_k;
        self.moe.moe_ffn = moe_ffn;
        self.moe.shared_ffn = shared_ffn;
        self.moe.moe_group = group;
        self.moe.moe_scale_bf16 = scale_bf16;
    }

    /// 전문가 슬라이스 테이블 등록 — (packed ptr/len, scale ptr/len) × (il,e,proj).
    /// 포인터는 서버 스토어 mmap 슬라이스 — 서버가 모델을 함께 보유하는 수명 계약.
    pub fn set_expert_table(&mut self, tab: Vec<(u64, u64, u64, u64)>) {
        self.set_expert_table_hybrid_opts(tab, 0, None)
    }

    /// [2026-10-21 잠복②] 단발 진단·eval 런용 — 호스트 스테이징 **명시 해제**.
    /// 스테이징 63GiB(C2)는 장기 serve의 페이지캐시 의존 제거용이라 단발 런엔
    /// 무의미하고, 자원 가드 밖(프로브 경로)이라 OOM 실사례(오라클 동시 실행).
    /// 미스는 mmap/페이지캐시 경유(수치 불변 — 바이트 출처만 다름).
    pub fn set_expert_table_nostage(&mut self, tab: Vec<(u64, u64, u64, u64)>) {
        self.set_expert_table_hybrid_opts(tab, 0, Some(false))
    }

    /// [2026-10-16 #11] 하이브리드 변형 — 앞 hot_per개/층(핫 몫)은 스테이징
    /// 제외(VRAM 아레나 O_DIRECT 직행분), 비상주 몫만 호스트 RAM 1회 복사.
    pub fn set_expert_table_hybrid(&mut self, tab: Vec<(u64, u64, u64, u64)>, hot_per: usize) {
        self.set_expert_table_hybrid_opts(tab, hot_per, None)
    }

    /// stage = None(env `LLM170_MOE_HOST_STAGE` 기본) · Some(false) 명시 해제.
    pub fn set_expert_table_hybrid_opts(
        &mut self,
        mut tab: Vec<(u64, u64, u64, u64)>,
        hot_per: usize,
        stage: Option<bool>,
    ) {
        // [P7-fix] 기존 스테이징이 등록돼 있으면 해제(버퍼 해제 전 필수).
        if self.moe.host_stage_reg > 0 {
            // SAFETY: 등록 성공 상태(host_stage_reg>0)의 host_stage 선두 주소다.
            let _ = unsafe {
                self.cc
                    .host_unregister(self.moe.host_stage.as_ptr() as *mut std::ffi::c_void)
            };
            self.moe.host_stage_reg = 0;
        }
        // [C2 2026-10-09] 호스트 RAM 스테이징(사용자 결정) — 스트리밍 셋을
        // 로드 시 1회 호스트 RAM으로 복사한다(SSD mmap·페이지캐시 경로 의존
        // 제거). 탈출구 LLM170_MOE_HOST_STAGE=0, 할당 실패 시 mmap 유지.
        // [잠복② 2026-10-21] stage=Some(false) = 명시 해제(단발 진단·eval —
        // 프로브 경로는 자원 가드 밖이라 63GiB 스테이징이 OOM을 만들었다).
        let do_stage = stage.unwrap_or_else(|| llm170_diag::flag::ne0("LLM170_MOE_HOST_STAGE"));
        if do_stage {
            // [2026-10-16 #11] 핫 몫(앞 hot_per개/층)은 스테이징 제외 — VRAM
            // 직행분이라 호스트 복사 불요(램 절감 = 핫 몫만큼).
            let n_exp = self.moe.n_experts;
            let rows: Vec<usize> = (0..tab.len())
                .filter(|&i| n_exp == 0 || (i / 3) % n_exp >= hot_per)
                .collect();
            let total: u64 = rows.iter().map(|&i| tab[i].1 + tab[i].3).sum();
            let mut buf: Vec<u8> = Vec::new();
            if total > 0 && usize::try_from(total).is_ok_and(|n| buf.try_reserve_exact(n).is_ok()) {
                let mut offs: Vec<(usize, usize, usize)> = Vec::with_capacity(rows.len());
                for &i in &rows {
                    let (qp, ql, sp, sl) = tab[i];
                    let qo = buf.len();
                    // SAFETY: tab 항목은 set_expert_table 수명 계약(mmap 슬라이스)
                    // — 호출 내 유효, len은 계약 검증 완료분.
                    buf.extend_from_slice(unsafe {
                        std::slice::from_raw_parts(qp as *const u8, ql as usize)
                    });
                    let so = buf.len();
                    // SAFETY: 상동(scale 슬라이스).
                    buf.extend_from_slice(unsafe {
                        std::slice::from_raw_parts(sp as *const u8, sl as usize)
                    });
                    offs.push((i, qo, so));
                }
                let base = buf.as_ptr() as u64;
                for &(i, qo, so) in &offs {
                    tab[i].0 = base + qo as u64;
                    tab[i].2 = base + so as u64;
                }
                eprintln!(
                    "[moe] 호스트 RAM 스테이징 {:.2}GiB — 업로드 mmap 의존 제거(C2)",
                    total as f64 / (1u64 << 30) as f64
                );
                self.moe.host_stage = buf;
                // [P7-fix 2026-10-10] 핀드 DMA — 등록 가능한 선두 구간만
                // cuMemHostRegister(RLIMIT_MEMLOCK 한도). 등록 구간 소스 복사는
                // DMA 직행(실측 13.8GB/s), 나머지는 드라이버 스테이징 유지.
                // 예산 LLM170_MOE_PIN_MB(0=끔, 기본 4096).
                let budget_mb = llm170_diag::flag::val("LLM170_MOE_PIN_MB")
                    .and_then(|v| v.parse::<usize>().ok())
                    .unwrap_or(4096);
                let budget = budget_mb << 20;
                if budget > 0 && !self.moe.host_stage.is_empty() {
                    // memlock 한도가 작은 환경 대비 1/4씩 축소 재시도(하한 4MiB).
                    let mut bytes = self.moe.host_stage.len().min(budget);
                    // [2026-10-12 결함수정] 등록 경계가 슬라이스(q/s)를 가르면 그
                    // 슬라이스의 async 복사가 실패한다(실측: 4GiB 경계가 L9
                    // 슬라이스를 관통 — cuMemcpyHtoDAsync INVALID_VALUE, 그
                    // 전문가가 선택된 프리필/디코드가 죽음). 등록 길이를 슬라이스
                    // 경계(선행 q·s 시작 오프셋)로 내린다.
                    if let Some(edge) = offs
                        .iter()
                        .flat_map(|&(_, qo, so)| [qo, so])
                        .filter(|&b| b <= bytes)
                        .max()
                    {
                        bytes = edge;
                    }
                    loop {
                        // SAFETY: host_stage는 구축 후 불변 Vec(boxed slice)라
                        // 해제 전까지 포인터 수명이 유지된다(해제는 상단 unregister).
                        match unsafe {
                            self.cc.host_register(
                                self.moe.host_stage.as_ptr() as *mut std::ffi::c_void,
                                bytes,
                            )
                        } {
                            Ok(()) => {
                                self.moe.host_stage_reg = bytes;
                                eprintln!(
                                    "# moe: 스테이징 핀드 등록 {:.2}/{}GiB — 구간 DMA 직행(P7-fix)",
                                    bytes as f64 / (1u64 << 30) as f64,
                                    self.moe.host_stage.len() >> 30
                                );
                                break;
                            }
                            Err(e) => {
                                if bytes <= (4 << 20) {
                                    eprintln!(
                                        "# moe: 핀드 등록 실패({e}) — 드라이버 스테이징 유지(P7-fix)"
                                    );
                                    break;
                                }
                                bytes = (bytes / 4).max(4 << 20);
                            }
                        }
                    }
                }
            } else {
                eprintln!("# moe: 호스트 스테이징 {total}B 할당 실패 — mmap 스트리밍 유지");
            }
        }
        self.experts_bytes = tab.iter().map(|e| e.1 + e.3).sum();
        self.moe.moe_tab = tab;
    }

    /// 전문가 상주 업로드 — 호스트 테이블(mmAP 슬라이스)을 VRAM 아레나
    /// (proj별 packed/scale 6개)로 올리고 테이블을 VRAM 포인터로 교체한다.
    /// 상주가 가능한 기기(170HX 64GB)에서 토큰당 PCIe 스트리밍을 제거한다.
    pub fn upload_experts_resident(
        &mut self,
        host_tab: &[(u64, u64, u64, u64)],
    ) -> Result<(), String> {
        let _g = self.cc.guard()?;
        if self.moe.n_experts == 0 || host_tab.len() != self.n_layers * self.moe.n_experts * 3 {
            return Err("upload_experts_resident: 테이블 형상 위반".into());
        }
        let mut pk = [0usize; 3];
        let mut sk = [0usize; 3];
        for (i, e) in host_tab.iter().enumerate() {
            pk[i % 3] += e.1 as usize;
            sk[i % 3] += e.3 as usize;
        }
        let (dpk, dsk, allocd) = self.alloc_expert_arenas(pk, sk)?;
        let mut dev_tab = Vec::with_capacity(host_tab.len());
        let (mut opk, mut osk) = ([0usize; 3], [0usize; 3]);
        // 1) 디바이스 테이블(아레나 오프셋) — 복사 없이 주소만 계산.
        for (i, e) in host_tab.iter().enumerate() {
            let p = i % 3;
            dev_tab.push((dpk[p] + opk[p] as u64, e.1, dsk[p] + osk[p] as u64, e.3));
            opk[p] += e.1 as usize;
            osk[p] += e.3 as usize;
        }
        let (mut opk2, mut osk2) = ([0usize; 3], [0usize; 3]);
        // 2) 전송 — h2d_chunked(페이지러블 비동기 + 드라이버 스테이징).
        // SAFETY: 항목은 서버 스토어 mmap 슬라이스(set_expert_table 수명 계약).
        let r = (|| -> Result<(), String> {
            for (i, e) in host_tab.iter().enumerate() {
                let p = i % 3;
                // SAFETY: 로컬 슬라이스의 유효 수명 내 바이트 뷰(길이 = 원소수×4).
                let qb = unsafe { std::slice::from_raw_parts(e.0 as *const u8, e.1 as usize) };
                // SAFETY: 로컬 슬라이스의 유효 수명 내 바이트 뷰(길이 = 원소수×4).
                let sb = unsafe { std::slice::from_raw_parts(e.2 as *const u8, e.3 as usize) };
                Self::h2d_chunked(&self.cc, dpk[p] + opk2[p] as u64, qb)?;
                Self::h2d_chunked(&self.cc, dsk[p] + osk2[p] as u64, sb)?;
                opk2[p] += e.1 as usize;
                osk2[p] += e.3 as usize;
            }
            Ok(())
        })();
        if let Err(err) = r {
            for p in allocd {
                let _ = self.cc.free(p);
            }
            return Err(err);
        }
        // 디바이스 포인터 테이블 — (q,s) 쌍 평탄 배열(커널 간접 참조).
        let experts_bytes: u64 = host_tab.iter().map(|e| e.1 + e.3).sum();
        self.register_moe_resident(dev_tab, experts_bytes)
    }

    /// [2026-10-16] 전문가 아레나 3종×2(q/s) 할당 — (아레나, 정리용 목록).
    /// 업로드 본체(호스트 슬라이스)·스트림(_streamed) 공용.
    fn alloc_expert_arenas(
        &mut self,
        pk: [usize; 3],
        sk: [usize; 3],
    ) -> Result<([CUdeviceptr; 3], [CUdeviceptr; 3], Vec<CUdeviceptr>), String> {
        let mut dpk = [0 as CUdeviceptr; 3];
        let mut dsk = [0 as CUdeviceptr; 3];
        let mut allocd = Vec::new();
        let setup = (|| -> Result<(), String> {
            for p in 0..3 {
                dpk[p] = self.cc.alloc(pk[p])?;
                allocd.push(dpk[p]);
                dsk[p] = self.cc.alloc(sk[p])?;
                allocd.push(dsk[p]);
            }
            Ok(())
        })();
        if let Err(e) = setup {
            for p in allocd {
                let _ = self.cc.free(p);
            }
            return Err(e);
        }
        Ok((dpk, dsk, allocd))
    }

    /// [2026-10-16] 전문가 상주 마감 — 디바이스 테이블 업로드 + 등록.
    /// 아레나 할당·전송은 호출부(본체/스트림) 몫.
    fn register_moe_resident(
        &mut self,
        dev_tab: Vec<(u64, u64, u64, u64)>,
        experts_bytes: u64,
    ) -> Result<(), String> {
        let mut flat: Vec<u64> = Vec::with_capacity(dev_tab.len() * 2);
        for e in &dev_tab {
            flat.push(e.0);
            flat.push(e.2);
        }
        // SAFETY: 로컬 슬라이스의 유효 수명 내 바이트 뷰(길이 = 원소수×8).
        let fb = unsafe { std::slice::from_raw_parts(flat.as_ptr() as *const u8, flat.len() * 8) };
        if self.moe.moe_dev_tab != 0 {
            let _ = self.cc.free(self.moe.moe_dev_tab);
            self.moe.moe_dev_tab = 0;
        }
        let dtab = self.cc.alloc(flat.len() * 8)?;
        if let Err(e) = Self::h2d_chunked(&self.cc, dtab, fb) {
            let _ = self.cc.free(dtab);
            return Err(e);
        }
        self.moe.moe_dev_tab = dtab;
        self.moe.moe_tab = dev_tab;
        self.experts_bytes = experts_bytes;
        self.weights_bytes += experts_bytes;
        self.moe.moe_resident = true;
        Ok(())
    }

    /// [2026-10-16] 전문가 상주 업로드(스트림) — `lens[i] = (q,s)` 셀 길이
    /// (i%3 = proj: gate/up/down). `fill(cell, which, off, buf)`가 핀드 버퍼를
    /// 채운다(예: O_DIRECT 직결) — 호스트 물질화 없이 VRAM 아레나로.
    pub fn upload_experts_resident_streamed(
        &mut self,
        lens: &[(usize, usize)],
        fill: &mut dyn FnMut(usize, usize, u64, &mut [u8]) -> Result<(), String>,
    ) -> Result<(), String> {
        let _g = self.cc.guard()?;
        if self.moe.n_experts == 0 || lens.len() != self.n_layers * self.moe.n_experts * 3 {
            return Err("upload_experts_resident_streamed: 테이블 형상 위반".into());
        }
        let mut pk = [0usize; 3];
        let mut sk = [0usize; 3];
        for (i, &(ql, sl)) in lens.iter().enumerate() {
            pk[i % 3] += ql;
            sk[i % 3] += sl;
        }
        let (dpk, dsk, allocd) = self.alloc_expert_arenas(pk, sk)?;
        let mut dev_tab = Vec::with_capacity(lens.len());
        let (mut opk, mut osk) = ([0usize; 3], [0usize; 3]);
        for (i, &(ql, sl)) in lens.iter().enumerate() {
            let p = i % 3;
            dev_tab.push((
                dpk[p] + opk[p] as u64,
                ql as u64,
                dsk[p] + osk[p] as u64,
                sl as u64,
            ));
            opk[p] += ql;
            osk[p] += sl;
        }
        let r = (|| -> Result<(), String> {
            let mut ring = super::mem::PinnedRing::new(&self.cc, STREAM_CHUNK)?;
            let (mut cpk, mut csk) = ([0usize; 3], [0usize; 3]);
            for (i, &(ql, sl)) in lens.iter().enumerate() {
                let p = i % 3;
                if ql > 0 {
                    ring.send(dpk[p] + cpk[p] as u64, ql, &mut |off, buf| {
                        fill(i, 0, off, buf)
                    })?;
                }
                if sl > 0 {
                    ring.send(dsk[p] + csk[p] as u64, sl, &mut |off, buf| {
                        fill(i, 1, off, buf)
                    })?;
                }
                cpk[p] += ql;
                csk[p] += sl;
                llm170_diag::watchdog::bump();
            }
            ring.sync()
        })();
        if let Err(err) = r {
            for p in allocd {
                let _ = self.cc.free(p);
            }
            return Err(err);
        }
        let experts_bytes: u64 = lens.iter().map(|e| (e.0 + e.1) as u64).sum();
        // [2026-10-19] 마린 MoE 재패킹 아레나(상주 전용 — 실패는 경고·폴백).
        if let Err(e) = self.build_marlin_moe_arenas(lens, fill) {
            eprintln!("[marlin_moe] 재패킹 스킵({e}) — 레거시 커널 폴백");
        }
        self.register_moe_resident(dev_tab, experts_bytes)
    }

    /// f32 → f16 비트(RNE) — 마린 스케일 변환용(bf16→f32→f16 — bf16 값은 정확).
    fn f32_to_f16_bits(v: f32) -> u16 {
        let x = v.to_bits();
        let sign = ((x >> 16) & 0x8000) as u16;
        let exp = ((x >> 23) & 0xFF) as i32 - 127 + 15;
        let man = x & 0x7F_FFFF;
        if exp >= 0x1F {
            return sign | 0x7C00;
        }
        if exp <= 0 {
            if exp < -10 {
                return sign;
            }
            let m = (man | 0x80_0000) >> (14 - exp);
            return sign | (m as u16);
        }
        let m = man >> 13;
        let mut r = sign | ((exp as u16) << 10) | (m as u16);
        let rem = man & 0x1FFF;
        if rem > 0x1000 || (rem == 0x1000 && (m & 1) == 1) {
            r = r.wrapping_add(1);
        }
        r
    }

    /// [2026-10-19] 마린 MoE 재패킹 아레나 구축 — **상주 경로 전용**(설계:
    /// 완전 상주 시에만 마린 디스패치 — 하이브리드/스트리밍은 레거시 커널 =
    /// 골든 무접촉). w13 = gate|up 호스트 융합 → `marlin_repack_q` · w2 = down ·
    /// 스케일 [g][n] 순열+bf16→f16. 실패는 경고만(아레나 0 = 폴백).
    fn build_marlin_moe_arenas(
        &mut self,
        lens: &[(usize, usize)],
        fill: &mut dyn FnMut(usize, usize, u64, &mut [u8]) -> Result<(), String>,
    ) -> Result<(), String> {
        let (nl, ne, nff, h) = (
            self.n_layers,
            self.moe.n_experts,
            self.moe.moe_ffn,
            self.hidden,
        );
        let (k13, n13) = (h, 2 * nff);
        let (k2, n2) = (nff, h);
        if !k13.is_multiple_of(128)
            || !n13.is_multiple_of(64)
            || !k2.is_multiple_of(16)
            || !n2.is_multiple_of(64)
        {
            return Err(format!(
                "marlin_moe 계약: k13={k13} n13={n13} k2={k2} n2={n2}"
            ));
        }
        let g13 = k13 / 32;
        let g2 = k2 / 32;
        let st13 = n13 * k13 / 2; // 전문가당 B 바이트(int4 = k*n/2)
        let st2 = n2 * k2 / 2;
        let ss13 = g13 * n13 * 2; // 스케일 f16 바이트
        let ss2 = g2 * n2 * 2;
        let (a13, a2, s13, s2) = (
            self.cc.alloc(nl * ne * st13)?,
            self.cc.alloc(nl * ne * st2)?,
            self.cc.alloc(nl * ne * ss13)?,
            self.cc.alloc(nl * ne * ss2)?,
        );
        let r = (|| -> Result<(), String> {
            const RCHUNK: usize = 8 << 20;
            let mat = |fill: &mut dyn FnMut(usize, usize, u64, &mut [u8]) -> Result<(), String>,
                       i: usize,
                       which: usize,
                       len: usize|
             -> Result<Vec<u8>, String> {
                let mut b = vec![0u8; len];
                for off in (0..len).step_by(RCHUNK) {
                    let end = (off + RCHUNK).min(len);
                    fill(i, which, off as u64, &mut b[off..end])?;
                }
                Ok(b)
            };
            let perm16 = |sb: &[u8], n: usize, k: usize, gs: usize| -> Result<Vec<u8>, String> {
                // bf16 [n][k/gs] → f32 → marlin 순열 [k/gs][n] → f16.
                let g = k / gs;
                let sf: Vec<f32> = sb
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
                    .collect();
                let pf = llm170_core::marlin::marlin_permute_scales(&sf, n, k, gs)?;
                let mut out = vec![0u8; g * n * 2];
                for (i, v) in pf.iter().enumerate() {
                    out[i * 2..i * 2 + 2].copy_from_slice(&Self::f32_to_f16_bits(*v).to_le_bytes());
                }
                Ok(out)
            };
            for layer in 0..nl {
                for e in 0..ne {
                    let i0 = (layer * ne + e) * 3;
                    // ── w13: gate|up 융합 재패킹 + 스케일 융합 순열 ──
                    let (ql0, sl0) = lens[i0];
                    let (ql1, sl1) = lens[i0 + 1];
                    let (qb0, sb0) = (mat(fill, i0, 0, ql0)?, mat(fill, i0, 1, sl0)?);
                    let (qb1, sb1) = (mat(fill, i0 + 1, 0, ql1)?, mat(fill, i0 + 1, 1, sl1)?);
                    let mut qcat = qb0;
                    qcat.extend_from_slice(&qb1);
                    let q32: Vec<u32> = qcat
                        .as_chunks::<4>()
                        .0
                        .iter()
                        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                        .collect();
                    let rp = llm170_core::marlin::marlin_repack_q(&q32, n13, k13)?;
                    let rb: Vec<u8> = rp.iter().flat_map(|v| v.to_le_bytes()).collect();
                    let off = (layer * ne + e) as u64 * st13 as u64;
                    Self::h2d_chunked(&self.cc, a13 + off, &rb)?;
                    let mut scat = sb0;
                    scat.extend_from_slice(&sb1);
                    let sp = perm16(&scat, n13, k13, 32)?;
                    Self::h2d_chunked(&self.cc, s13 + (layer * ne + e) as u64 * ss13 as u64, &sp)?;
                    // ── w2: down 재패킹 + 스케일 순열 ──
                    let (ql2, sl2) = lens[i0 + 2];
                    let qb2 = mat(fill, i0 + 2, 0, ql2)?;
                    let sb2 = mat(fill, i0 + 2, 1, sl2)?;
                    let q32b: Vec<u32> = qb2
                        .as_chunks::<4>()
                        .0
                        .iter()
                        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                        .collect();
                    let rp2 = llm170_core::marlin::marlin_repack_q(&q32b, n2, k2)?;
                    let rb2: Vec<u8> = rp2.iter().flat_map(|v| v.to_le_bytes()).collect();
                    Self::h2d_chunked(&self.cc, a2 + (layer * ne + e) as u64 * st2 as u64, &rb2)?;
                    let sp2 = perm16(&sb2, n2, k2, 32)?;
                    Self::h2d_chunked(&self.cc, s2 + (layer * ne + e) as u64 * ss2 as u64, &sp2)?;
                    llm170_diag::watchdog::bump();
                }
            }
            self.cc.sync()
        })();
        if let Err(e) = r {
            for p in [a13, a2, s13, s2] {
                let _ = self.cc.free(p);
            }
            return Err(e);
        }
        self.moe_m13 = a13;
        self.moe_m2 = a2;
        self.moe_s13 = s13;
        self.moe_s2 = s2;
        let bytes = (nl * ne * (st13 + st2 + ss13 + ss2)) as u64;
        self.weights_bytes += bytes;
        self.marlin_bytes += bytes;
        Ok(())
    }

    /// [2026-10-16 #11] 하이브리드 상주 업로드 — 층당 앞 hot_per개 전문가는
    /// O_DIRECT 직행으로 VRAM 아레나(테이블 엔트리 = 아레나 주소), 나머지
    /// 엔트리는 0(스트리밍 계약: 읽기 전 mcache_insert가 기록). 디바이스
    /// 테이블은 로드 시 1회 업로드하고 hot_tab(호스트 미러)을 보관한다.
    pub fn upload_experts_hybrid_streamed(
        &mut self,
        lens: &[(usize, usize)],
        fill: &mut dyn FnMut(usize, usize, u64, &mut [u8]) -> Result<(), String>,
        hot_per: usize,
        mcache_k_plan: usize,
    ) -> Result<(), String> {
        let _g = self.cc.guard()?;
        let n_exp = self.moe.n_experts;
        if n_exp == 0 || lens.len() != self.n_layers * n_exp * 3 || hot_per == 0 || hot_per >= n_exp
        {
            return Err("upload_experts_hybrid_streamed: 인자 위반".into());
        }
        let hot_cell = |i: usize| (i / 3) % n_exp < hot_per;
        let (mut pk, mut sk) = ([0usize; 3], [0usize; 3]);
        for (i, &(ql, sl)) in lens.iter().enumerate() {
            if hot_cell(i) {
                pk[i % 3] += ql;
                sk[i % 3] += sl;
            }
        }
        let (dpk, dsk, allocd) = self.alloc_expert_arenas(pk, sk)?;
        let mut dev_tab = Vec::with_capacity(lens.len());
        let (mut opk, mut osk) = ([0usize; 3], [0usize; 3]);
        for (i, &(ql, sl)) in lens.iter().enumerate() {
            let p = i % 3;
            if hot_cell(i) {
                dev_tab.push((
                    dpk[p] + opk[p] as u64,
                    ql as u64,
                    dsk[p] + osk[p] as u64,
                    sl as u64,
                ));
                opk[p] += ql;
                osk[p] += sl;
            } else {
                // 비상주 엔트리 — 스테이징(읽기) 전 mcache_insert가 기록(0 = 미기록).
                dev_tab.push((0, ql as u64, 0, sl as u64));
            }
        }
        let r = (|| -> Result<(), String> {
            let mut ring = super::mem::PinnedRing::new(&self.cc, STREAM_CHUNK)?;
            let (mut cpk, mut csk) = ([0usize; 3], [0usize; 3]);
            for (i, &(ql, sl)) in lens.iter().enumerate() {
                let p = i % 3;
                if !hot_cell(i) {
                    continue;
                }
                if ql > 0 {
                    ring.send(dpk[p] + cpk[p] as u64, ql, &mut |off, buf| {
                        fill(i, 0, off, buf)
                    })?;
                }
                if sl > 0 {
                    ring.send(dsk[p] + csk[p] as u64, sl, &mut |off, buf| {
                        fill(i, 1, off, buf)
                    })?;
                }
                cpk[p] += ql;
                csk[p] += sl;
                llm170_diag::watchdog::bump();
            }
            ring.sync()
        })();
        if let Err(err) = r {
            for p in allocd {
                let _ = self.cc.free(p);
            }
            return Err(err);
        }
        // 디바이스 테이블 업로드(핫 엔트리 유효·비상주 0) + 호스트 미러 보관.
        let mut flat: Vec<u64> = Vec::with_capacity(dev_tab.len() * 2);
        for e in &dev_tab {
            flat.push(e.0);
            flat.push(e.2);
        }
        // SAFETY: 로컬 슬라이스의 유효 수명 내 바이트 뷰(길이 = 원소수×8).
        let fb = unsafe { std::slice::from_raw_parts(flat.as_ptr() as *const u8, flat.len() * 8) };
        if self.moe.moe_dev_tab != 0 {
            let _ = self.cc.free(self.moe.moe_dev_tab);
            self.moe.moe_dev_tab = 0;
        }
        let dtab = self.cc.alloc(flat.len() * 8)?;
        if let Err(e) = Self::h2d_chunked(&self.cc, dtab, fb) {
            let _ = self.cc.free(dtab);
            return Err(e);
        }
        self.moe.moe_dev_tab = dtab;
        let hot_bytes: u64 = lens
            .iter()
            .enumerate()
            .filter(|&(i, _)| hot_cell(i))
            .map(|(_, &(ql, sl))| (ql + sl) as u64)
            .sum();
        self.weights_bytes += hot_bytes;
        self.moe.hot_tab = dev_tab;
        self.moe.hot_per_layer = hot_per;
        self.moe.mcache_k_plan = mcache_k_plan;
        Ok(())
    }

    /// [FN PP2 2026-10-20] 마린-단독 상주(프리필 전용) — raw 아레나 없이 마린
    /// 아레나만 할당(이중 아레나 60GB/카드 한계 회피). 창 컴팩트 배치 + 절대
    /// il 인덱싱은 아레나 베이스를 -lo·ne·stride 사전 오프셋해 맞춘다(커널
    /// 불변). **계약: t≤8 GEMV 전문가 경로(raw 테이블) 불가** — 디코드는
    /// 스트리밍/상주 raw 배치와 별도 경로 필요(프리필 속도 지표 전용).
    /// 실패는 Err(호출부가 스트리밍 폴백).
    pub fn upload_experts_marlin_only_streamed(
        &mut self,
        lens_win: &[(usize, usize)],
        fill: &mut dyn FnMut(usize, usize, u64, &mut [u8]) -> Result<(), String>,
        layer_lo: usize,
    ) -> Result<(), String> {
        let _g = self.cc.guard()?;
        let (nl_all, ne, nff, h) = (
            self.n_layers,
            self.moe.n_experts,
            self.moe.moe_ffn,
            self.hidden,
        );
        let n_win = lens_win.len().checked_div(3 * ne.max(1)).unwrap_or(0);
        if ne == 0 || n_win == 0 || layer_lo + n_win > nl_all {
            return Err(format!(
                "marlin_only: lens {} → n_win {n_win} lo {layer_lo} nl {nl_all} 위반",
                lens_win.len()
            ));
        }
        let (k13, n13) = (h, 2 * nff);
        let (k2, n2) = (nff, h);
        if !k13.is_multiple_of(128)
            || !n13.is_multiple_of(64)
            || !k2.is_multiple_of(16)
            || !n2.is_multiple_of(64)
        {
            return Err(format!(
                "marlin_only 계약: k13={k13} n13={n13} k2={k2} n2={n2}"
            ));
        }
        let (g13, g2) = (k13 / 32, k2 / 32);
        let (st13, st2) = (n13 * k13 / 2, n2 * k2 / 2);
        let (ss13, ss2) = (g13 * n13 * 2, g2 * n2 * 2);
        let (a13, a2, s13, s2) = (
            self.cc.alloc(n_win * ne * st13)?,
            self.cc.alloc(n_win * ne * st2)?,
            self.cc.alloc(n_win * ne * ss13)?,
            self.cc.alloc(n_win * ne * ss2)?,
        );
        const RCHUNK: usize = 8 << 20;
        let mat = |fill: &mut dyn FnMut(usize, usize, u64, &mut [u8]) -> Result<(), String>,
                   i: usize,
                   which: usize,
                   len: usize|
         -> Result<Vec<u8>, String> {
            let mut b = vec![0u8; len];
            for off in (0..len).step_by(RCHUNK) {
                let end = (off + RCHUNK).min(len);
                fill(i, which, off as u64, &mut b[off..end])?;
            }
            Ok(b)
        };
        let perm16 = |sb: &[u8], n: usize, k: usize, gs: usize| -> Result<Vec<u8>, String> {
            let g = k / gs;
            let sf: Vec<f32> = sb
                .as_chunks::<2>()
                .0
                .iter()
                .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
                .collect();
            let pf = llm170_core::marlin::marlin_permute_scales(&sf, n, k, gs)?;
            let mut out = vec![0u8; g * n * 2];
            for (i, v) in pf.iter().enumerate() {
                out[i * 2..i * 2 + 2].copy_from_slice(&Self::f32_to_f16_bits(*v).to_le_bytes());
            }
            Ok(out)
        };
        let r = (|| -> Result<(), String> {
            for layer in 0..n_win {
                for e in 0..ne {
                    let i0 = (layer * ne + e) * 3;
                    let (ql0, sl0) = lens_win[i0];
                    let (ql1, sl1) = lens_win[i0 + 1];
                    let (qb0, sb0) = (mat(fill, i0, 0, ql0)?, mat(fill, i0, 1, sl0)?);
                    let (qb1, sb1) = (mat(fill, i0 + 1, 0, ql1)?, mat(fill, i0 + 1, 1, sl1)?);
                    let mut qcat = qb0;
                    qcat.extend_from_slice(&qb1);
                    let q32: Vec<u32> = qcat
                        .as_chunks::<4>()
                        .0
                        .iter()
                        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                        .collect();
                    let rp = llm170_core::marlin::marlin_repack_q(&q32, n13, k13)?;
                    let rb: Vec<u8> = rp.iter().flat_map(|v| v.to_le_bytes()).collect();
                    Self::h2d_chunked(&self.cc, a13 + (layer * ne + e) as u64 * st13 as u64, &rb)?;
                    let mut scat = sb0;
                    scat.extend_from_slice(&sb1);
                    let sp = perm16(&scat, n13, k13, 32)?;
                    Self::h2d_chunked(&self.cc, s13 + (layer * ne + e) as u64 * ss13 as u64, &sp)?;
                    let (ql2, sl2) = lens_win[i0 + 2];
                    let qb2 = mat(fill, i0 + 2, 0, ql2)?;
                    let sb2 = mat(fill, i0 + 2, 1, sl2)?;
                    let q32b: Vec<u32> = qb2
                        .as_chunks::<4>()
                        .0
                        .iter()
                        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                        .collect();
                    let rp2 = llm170_core::marlin::marlin_repack_q(&q32b, n2, k2)?;
                    let rb2: Vec<u8> = rp2.iter().flat_map(|v| v.to_le_bytes()).collect();
                    Self::h2d_chunked(&self.cc, a2 + (layer * ne + e) as u64 * st2 as u64, &rb2)?;
                    let sp2 = perm16(&sb2, n2, k2, 32)?;
                    Self::h2d_chunked(&self.cc, s2 + (layer * ne + e) as u64 * ss2 as u64, &sp2)?;
                    llm170_diag::watchdog::bump();
                }
            }
            self.cc.sync()
        })();
        if let Err(e) = r {
            for p in [a13, a2, s13, s2] {
                let _ = self.cc.free(p);
            }
            return Err(e);
        }
        // 절대 il 인덱싱 사전 오프셋(커널: base + il·ne·stride → 창 컴팩트 착지).
        let (off13, off2) = (
            (layer_lo as u64 * ne as u64) * st13 as u64,
            (layer_lo as u64 * ne as u64) * st2 as u64,
        );
        let (os13, os2) = (
            (layer_lo as u64 * ne as u64) * ss13 as u64,
            (layer_lo as u64 * ne as u64) * ss2 as u64,
        );
        self.moe_m13 = a13.wrapping_sub(off13);
        self.moe_m2 = a2.wrapping_sub(off2);
        self.moe_s13 = s13.wrapping_sub(os13);
        self.moe_s2 = s2.wrapping_sub(os2);
        let bytes = (n_win * ne * (st13 + st2 + ss13 + ss2)) as u64;
        self.weights_bytes += bytes;
        // raw 테이블 = 전체 크기 0패딩(길이 계약 — t>8 마린 경로만 소비).
        self.set_expert_table(vec![(0u64, 0u64, 0u64, 0u64); nl_all * ne * 3]);
        // [fix] 상주 플래그 — 미설정 시 스트리밍 스테이징(moe_stage_t)이
        // distinct 가드를 발사한다(마린이 참조할 raw 0 테이블 대상).
        self.moe.moe_resident = true;
        self.experts_bytes = bytes;
        Ok(())
    }

    /// head(output.weight) 상주 업로드 — [I8head 2026-10-11] bf16 → int8
    /// (행별 amax/127 스케일) 양자화 후 업로드.
    /// 품질 판정(eval chunk4): en PPL +0.061%·argmax 99.139% / ko +0.0002%·
    /// 99.347% — KV int8 채택 밴드 · 27B 프로브 골든 유지. int4-g128 판은
    /// 기각(en +0.59%/92.7%). 대역: 2.54GB → 1.27GB+1MB(27B) — head 2.7ms
    /// (디코드의 14%)가 절반으로.
    pub fn upload_head(&mut self, data: &[u8], n: usize, k: usize) -> Result<(), String> {
        let _g = self.cc.guard()?;
        let need = n * k * 2;
        if n == 0 || k == 0 || data.len() < need || !k.is_multiple_of(4) {
            return Err(format!(
                "upload_head: 형상 계약 위반 n={n} k={k} bytes={} < {need}",
                data.len()
            ));
        }
        // 호스트 양자화 — 행별 amax/127. 다중 스레드(1.27G원소 ~수백 ms).
        let mut q8 = vec![0i8; n * k];
        let mut scv = vec![0f32; n];
        let src = &data[..need];
        let nt = std::thread::available_parallelism()
            .map(|v| v.get())
            .unwrap_or(4)
            .min(16);
        let rows_per = n.div_ceil(nt).max(1);
        {
            let qchunks: Vec<_> = q8.chunks_mut(rows_per * k).collect();
            let schunks: Vec<_> = scv.chunks_mut(rows_per).collect();
            std::thread::scope(|sp| {
                let mut r0 = 0usize;
                for (qc, sc_c) in qchunks.into_iter().zip(schunks) {
                    let base = r0;
                    sp.spawn(move || {
                        let nrows = qc.len() / k;
                        for r in 0..nrows {
                            let row = &src[(base + r) * k * 2..(base + r + 1) * k * 2];
                            let mut amax = 0.0f32;
                            for i in 0..k {
                                let v = f32::from_bits(
                                    u32::from(u16::from_le_bytes([row[i * 2], row[i * 2 + 1]]))
                                        << 16,
                                );
                                amax = amax.max(v.abs());
                            }
                            let s = if amax > 0.0 { amax / 127.0 } else { 1.0 };
                            sc_c[r] = s;
                            for i in 0..k {
                                let v = f32::from_bits(
                                    u32::from(u16::from_le_bytes([row[i * 2], row[i * 2 + 1]]))
                                        << 16,
                                );
                                qc[r * k + i] = (v / s).round().clamp(-127.0, 127.0) as i8;
                            }
                        }
                    });
                    r0 += rows_per;
                }
            });
        }
        // [검토 반영 2026-10-16] 재적재 시 argmax_out(4B)도 해제 — 종전
        // 누락은 재업로드마다 4B 누수였다(신규 스트림 경로와 동일 정정).
        for p in [self.head_w, self.head_sc, self.head_out, self.argmax_out] {
            if p != 0 {
                self.cc.free(p)?;
            }
        }
        self.head_w = 0; // G1: 부분 상태(head_n=0)로 소비되지 않게 마지막에 대입.
        self.head_sc = 0;
        self.head_out = 0;
        self.argmax_out = 0;
        self.head_n = 0;
        self.head_k = 0;
        // int8 본체 + f32 행스케일 업로드.
        let dw = self.cc.alloc(n * k)?;
        // SAFETY: 로컬 슬라이스의 유효 수명 내 바이트 뷰(길이 = 원소수).
        let qb = unsafe { std::slice::from_raw_parts(q8.as_ptr() as *const u8, n * k) };
        if let Err(e) = Self::h2d_chunked(&self.cc, dw, qb) {
            let _ = self.cc.free(dw);
            return Err(e);
        }
        let dsc = self.cc.alloc(n * 4)?;
        // SAFETY: 로컬 슬라이스의 유효 수명 내 바이트 뷰(길이 = 원소수×4).
        let sb = unsafe { std::slice::from_raw_parts(scv.as_ptr() as *const u8, n * 4) };
        if let Err(e) = Self::h2d_chunked(&self.cc, dsc, sb) {
            let _ = self.cc.free(dw);
            let _ = self.cc.free(dsc);
            return Err(e);
        }
        // [검토 반영] alloc(4) 실패 시 alloc(n*4) 회수(and_then 체인은 p 유실).
        let dout = match self.cc.alloc(n * 4) {
            Ok(p) => match self.cc.alloc(4) {
                Ok(a) => {
                    self.argmax_out = a;
                    p
                }
                Err(e) => {
                    let _ = self.cc.free(p);
                    let _ = self.cc.free(dw);
                    let _ = self.cc.free(dsc);
                    return Err(e);
                }
            },
            Err(e) => {
                let _ = self.cc.free(dw);
                let _ = self.cc.free(dsc);
                return Err(e);
            }
        };
        self.head_w = dw;
        self.head_sc = dsc;
        self.head_out = dout;
        self.head_n = n;
        self.head_k = k;
        self.weights_bytes += (n * k + n * 4 + n * 4) as u64;
        Ok(())
    }

    /// [S3-1 2026-10-11] token_embd 상주 업로드 — 프리필 임베딩 gather 원본.
    /// f16/bf16 원본 바이트 그대로(디코드는 커널 h2f/b2f = Rust 동일 함수 —
    /// embed_row와 값 비트동일).
    pub fn upload_embed(
        &mut self,
        data: &[u8],
        n: usize,
        k: usize,
        bf16: bool,
    ) -> Result<(), String> {
        let _g = self.cc.guard()?;
        let need = n * k * 2;
        if n == 0 || k == 0 || data.len() < need {
            return Err(format!(
                "upload_embed: 형상 계약 위반 n={n} k={k} bytes={} < {need}",
                data.len()
            ));
        }
        if self.emb_dev != 0 {
            self.cc.free(self.emb_dev)?;
        }
        self.emb_dev = 0; // G1: 부분 상태로 소비되지 않게 마지막에 재대입.
        self.emb_n = 0;
        self.emb_k = 0;
        let dw = self.cc.alloc(need)?;
        if let Err(e) = Self::h2d_chunked(&self.cc, dw, &data[..need]) {
            let _ = self.cc.free(dw);
            return Err(e);
        }
        self.emb_dev = dw;
        self.emb_n = n;
        self.emb_k = k;
        self.emb_bf16 = bf16;
        self.weights_bytes += need as u64;
        Ok(())
    }

    /// [S3-1] 임베딩 상주 여부 — 프리필 gather 경로 진입 조건.
    pub fn embed_ready(&self) -> bool {
        self.emb_dev != 0 && self.emb_k == self.hidden
    }

    /// [S3-1] 임베딩 gather 런치 — toks[t] → out[t][k] f32.
    /// [MTP-CHAIN 2026-10-12 §3.3 ②] pub(super)로 넓힘 — 장치 체인 그래프가
    /// 토큰 레지스터(dbatch_am)에서 같은 커널로 임베딩 행을 조립한다.
    pub(super) fn embed_gather_launch(
        &self,
        toks: CUdeviceptr,
        out: CUdeviceptr,
        t: usize,
    ) -> Result<(), String> {
        let f = self.cc.function("w4a16_embed_gather")?;
        let (mut p_e, mut p_tk, mut p_o) = (self.emb_dev, toks, out);
        let (mut p_t, mut p_k, mut p_n, mut p_b) = (
            t as i32,
            self.emb_k as i32,
            self.emb_n as i32,
            self.emb_bf16 as i32,
        );
        let elems = t * self.emb_k;
        self.cc.launch(
            f,
            elems.div_ceil(256) as u32,
            1,
            256,
            &mut crate::rawcuda::args::l7(
                &mut p_e, &mut p_tk, &mut p_o, &mut p_t, &mut p_k, &mut p_n, &mut p_b,
            ),
        )
    }
}

// ── [2026-10-16] 스트림 상주 업로드 — O_DIRECT 직결(핀드 링) ─────────────
//
// 서버/프로브 로더가 파일 스트림(예: core::w4a16::DirectReader)을 `fill` 콜백로
// 공급하면 **핀드 링에 채워 곧바로 h2d_async**로 올린다 — 드라이버 스테이징
// 왕복이 없고(핀드 소스 = 직접 DMA), 읽기와 PCIe 복사가 창 단위로 겹친다.
// 호스트 RAM = 상수(링 2×8MiB). 수치는 바이트 동일(적재 경로 — 판정 클래스 아님).

/// 스트림 청크(핀드 버퍼 2벌 = 16MiB 상수).
const STREAM_CHUNK: usize = 8 << 20;

impl W4a16Dec {
    /// 선형 상주 업로드(스트림) — `fill(which, off, buf)`가 채운다(0=q·1=s).
    /// 검증·bookkeeping은 `upload_lin_g`와 동일(성공 시에만 상주 등록 — 부분
    /// 유실 금지).
    pub fn upload_lin_g_stream(
        &mut self,
        name: &str,
        n: usize,
        k: usize,
        group: usize,
        scale_bf16: bool,
        q_len: usize,
        s_len: usize,
        fill: &mut dyn FnMut(usize, u64, &mut [u8]) -> Result<(), String>,
    ) -> Result<(), String> {
        let _g = self.cc.guard()?;
        crate::rawcuda::gptq4::kernel_sym(false, group, scale_bf16)?;
        if q_len != n * (k / 2) || s_len != n * (k / group) * 2 {
            return Err(format!(
                "upload_lin_g_stream {name}: q={q_len} s={s_len} != n{n} k{k} g{group} 계약"
            ));
        }
        let dq = self.cc.alloc(q_len)?;
        let ds = match self.cc.alloc(s_len) {
            Ok(p) => p,
            Err(e) => {
                let _ = self.cc.free(dq);
                return Err(e);
            }
        };
        let r = (|| -> Result<(), String> {
            let mut ring = super::mem::PinnedRing::new(&self.cc, STREAM_CHUNK)?;
            ring.send(dq, q_len, &mut |off, buf| fill(0, off, buf))?;
            ring.send(ds, s_len, &mut |off, buf| fill(1, off, buf))?;
            ring.sync()
        })();
        if let Err(e) = r {
            let _ = self.cc.free(dq);
            let _ = self.cc.free(ds);
            return Err(e);
        }
        self.weights_bytes += (q_len + s_len) as u64;
        if let Some((oq, os, _, _)) = self.lins.insert(name.to_string(), (dq, ds, n, k)) {
            let _ = self.cc.free(oq);
            let _ = self.cc.free(os);
        }
        if group == 128 && !scale_bf16 {
            self.lin_grp.remove(name);
        } else {
            self.lin_grp.insert(name.to_string(), (group, scale_bf16));
        }
        // [split-k marlin 디코드] 이중 버퍼 — 재패킹은 호스트 전체 q가 필요해
        // fill(0,…)로 재료화한다(트랜지트 n*k/2B — 최대 ~151MB/가중).
        if (group == 128 && !scale_bf16) || (group == 32 && scale_bf16) {
            const RCHUNK: usize = 8 << 20;
            let mut qbuf = vec![0u8; q_len];
            let mut ok = true;
            for off in (0..q_len).step_by(RCHUNK) {
                let end = (off + RCHUNK).min(q_len);
                if let Err(e) = fill(0, off as u64, &mut qbuf[off..end]) {
                    eprintln!("[marlin] {name}: q 재료화 실패({e}) — GEMV 폴백");
                    ok = false;
                    break;
                }
            }
            // [2026-10-18] 마린 스케일 순열용 s 재료화(소형 — n*k/64B).
            let mut sbuf = vec![0u8; s_len];
            let mut ok_s = true;
            for off in (0..s_len).step_by(RCHUNK) {
                let end = (off + RCHUNK).min(s_len);
                if let Err(e) = fill(1, off as u64, &mut sbuf[off..end]) {
                    eprintln!("[marlin] {name}: s 재료화 실패({e}) — GEMV 폴백");
                    ok_s = false;
                    break;
                }
            }
            if ok && ok_s {
                self.upload_lin_marlin(name, &qbuf, &sbuf, n, k, group, scale_bf16);
            }
        }
        Ok(())
    }

    /// 플레인 bf16 상주 업로드(스트림) — `upload_plain` 미러.
    pub fn upload_plain_stream(
        &mut self,
        name: &str,
        n: usize,
        k: usize,
        len: usize,
        fill: &mut dyn FnMut(u64, &mut [u8]) -> Result<(), String>,
    ) -> Result<(), String> {
        let _g = self.cc.guard()?;
        let need = n * k * 2;
        if n == 0 || k == 0 || len < need {
            return Err(format!(
                "upload_plain_stream({name}): 형상 계약 위반 n={n} k={k} bytes={len} < {need}"
            ));
        }
        let dw = self.cc.alloc(need)?;
        let r = (|| -> Result<(), String> {
            let mut ring = super::mem::PinnedRing::new(&self.cc, STREAM_CHUNK)?;
            ring.send(dw, need, fill)?;
            ring.sync()
        })();
        if let Err(e) = r {
            let _ = self.cc.free(dw);
            return Err(e);
        }
        self.weights_bytes += need as u64;
        if let Some((old, _, _)) = self.plains.insert(name.to_string(), (dw, n, k)) {
            let _ = self.cc.free(old);
        }
        Ok(())
    }

    /// token_embd 상주 업로드(스트림) — `upload_embed` 미러.
    pub fn upload_embed_stream(
        &mut self,
        n: usize,
        k: usize,
        bf16: bool,
        len: usize,
        fill: &mut dyn FnMut(u64, &mut [u8]) -> Result<(), String>,
    ) -> Result<(), String> {
        let _g = self.cc.guard()?;
        let need = n * k * 2;
        if n == 0 || k == 0 || len < need {
            return Err(format!(
                "upload_embed_stream: 형상 계약 위반 n={n} k={k} bytes={len} < {need}"
            ));
        }
        if self.emb_dev != 0 {
            self.cc.free(self.emb_dev)?;
        }
        self.emb_dev = 0; // G1: 부분 상태로 소비되지 않게 마지막에 재대입.
        self.emb_n = 0;
        self.emb_k = 0;
        let dw = self.cc.alloc(need)?;
        let r = (|| -> Result<(), String> {
            let mut ring = super::mem::PinnedRing::new(&self.cc, STREAM_CHUNK)?;
            ring.send(dw, need, fill)?;
            ring.sync()
        })();
        if let Err(e) = r {
            let _ = self.cc.free(dw);
            return Err(e);
        }
        self.emb_dev = dw;
        self.emb_n = n;
        self.emb_k = k;
        self.emb_bf16 = bf16;
        self.weights_bytes += need as u64;
        Ok(())
    }

    /// head(output.weight) 상주 업로드(스트림) — bf16 행을 행-블록(2MiB)으로
    /// 받아 호스트 int8 양자화(행별 amax/127 — `upload_head`와 동일 산식) 후 h2d.
    /// 호스트 RAM = 블록 크기 상수(전체 2.4GB 물질화 회피).
    pub fn upload_head_stream(
        &mut self,
        n: usize,
        k: usize,
        len: usize,
        fill: &mut dyn FnMut(u64, &mut [u8]) -> Result<(), String>,
    ) -> Result<(), String> {
        let _g = self.cc.guard()?;
        let need = n * k * 2;
        if n == 0 || k == 0 || len < need || !k.is_multiple_of(4) {
            return Err(format!(
                "upload_head_stream: 형상 계약 위반 n={n} k={k} bytes={len} < {need}"
            ));
        }
        // [검토 반영 2026-10-16] 재적재 시 argmax_out(4B)도 해제(누수 정정).
        for p in [self.head_w, self.head_sc, self.head_out, self.argmax_out] {
            if p != 0 {
                self.cc.free(p)?;
            }
        }
        self.head_w = 0; // G1: 부분 상태(head_n=0)로 소비되지 않게 마지막에 대입.
        self.head_sc = 0;
        self.head_out = 0;
        self.argmax_out = 0;
        self.head_n = 0;
        self.head_k = 0;
        let dw = self.cc.alloc(n * k)?;
        let dsc = match self.cc.alloc(n * 4) {
            Ok(p) => p,
            Err(e) => {
                let _ = self.cc.free(dw);
                return Err(e);
            }
        };
        // [검토 반영] alloc(4) 실패 시 alloc(n*4) 회수(and_then 체인은 p 유실).
        let dout = match self.cc.alloc(n * 4) {
            Ok(p) => match self.cc.alloc(4) {
                Ok(a) => {
                    self.argmax_out = a;
                    p
                }
                Err(e) => {
                    let _ = self.cc.free(p);
                    let _ = self.cc.free(dw);
                    let _ = self.cc.free(dsc);
                    return Err(e);
                }
            },
            Err(e) => {
                let _ = self.cc.free(dw);
                let _ = self.cc.free(dsc);
                return Err(e);
            }
        };
        let nt = std::thread::available_parallelism()
            .map(|v| v.get())
            .unwrap_or(4)
            .min(16);
        let rows_blk = ((2 << 20) / (k * 2).max(1)).max(1);
        let mut raw = vec![0u8; rows_blk * k * 2];
        let mut q8 = vec![0i8; rows_blk * k];
        let mut scv = vec![0f32; rows_blk];
        let mut r0 = 0usize;
        while r0 < n {
            let rows = rows_blk.min(n - r0);
            let blen = rows * k * 2;
            fill((r0 * k * 2) as u64, &mut raw[..blen])?;
            {
                let rows_per = rows.div_ceil(nt).max(1);
                let qchunks: Vec<_> = q8[..rows * k].chunks_mut(rows_per * k).collect();
                let schunks: Vec<_> = scv[..rows].chunks_mut(rows_per).collect();
                std::thread::scope(|sp| {
                    let mut rr = 0usize;
                    for (qc, sc_c) in qchunks.into_iter().zip(schunks) {
                        let base = rr;
                        let raw_ref = &raw[..blen];
                        sp.spawn(move || {
                            let nrows = qc.len() / k;
                            for r in 0..nrows {
                                let row = &raw_ref[(base + r) * k * 2..(base + r + 1) * k * 2];
                                let mut amax = 0.0f32;
                                for i in 0..k {
                                    let v = f32::from_bits(
                                        u32::from(u16::from_le_bytes([row[i * 2], row[i * 2 + 1]]))
                                            << 16,
                                    );
                                    amax = amax.max(v.abs());
                                }
                                let s = if amax > 0.0 { amax / 127.0 } else { 1.0 };
                                sc_c[r] = s;
                                for i in 0..k {
                                    let v = f32::from_bits(
                                        u32::from(u16::from_le_bytes([row[i * 2], row[i * 2 + 1]]))
                                            << 16,
                                    );
                                    qc[r * k + i] = (v / s).round().clamp(-127.0, 127.0) as i8;
                                }
                            }
                        });
                        rr += rows_per;
                    }
                });
            }
            // SAFETY: 블록 슬라이스의 유효 수명 내 바이트 뷰(길이 = 원소수/4배).
            let qb = unsafe { std::slice::from_raw_parts(q8.as_ptr() as *const u8, rows * k) };
            if let Err(e) = Self::h2d_chunked(&self.cc, dw + (r0 * k) as u64, qb) {
                let _ = self.cc.free(dw);
                let _ = self.cc.free(dsc);
                let _ = self.cc.free(dout);
                return Err(e);
            }
            // SAFETY: 같은 계약(길이 = 원소수×4).
            let sb = unsafe { std::slice::from_raw_parts(scv.as_ptr() as *const u8, rows * 4) };
            if let Err(e) = Self::h2d_chunked(&self.cc, dsc + (r0 * 4) as u64, sb) {
                let _ = self.cc.free(dw);
                let _ = self.cc.free(dsc);
                let _ = self.cc.free(dout);
                return Err(e);
            }
            r0 += rows;
            llm170_diag::watchdog::bump();
        }
        self.head_w = dw;
        self.head_sc = dsc;
        self.head_out = dout;
        self.head_n = n;
        self.head_k = k;
        self.weights_bytes += (n * k + n * 4 + n * 4) as u64;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::W4a16Dec;

    /// KV 2계층 주소 — id < vram_blocks = VRAM 아레나, 이후 = 호스트 아레나
    /// (핀드·디바이스 매핑). 경계·연속성·스킵 없음.
    #[test]
    fn kv_block_addr_tiers() {
        let (bv, bh, stride) = (0x1_0000u64, 0x2_0000u64, 0x100u64);
        // VRAM부.
        assert_eq!(W4a16Dec::kv_block_addr(bv, bh, 0, stride, 10), 0x1_0000);
        assert_eq!(
            W4a16Dec::kv_block_addr(bv, bh, 9, stride, 10),
            0x1_0000 + 9 * 0x100
        );
        // 호스트부(경계 = 첫 호스트 블록).
        assert_eq!(W4a16Dec::kv_block_addr(bv, bh, 10, stride, 10), 0x2_0000);
        assert_eq!(
            W4a16Dec::kv_block_addr(bv, bh, 12, stride, 10),
            0x2_0000 + 2 * 0x100
        );
        // 전량 VRAM(호스트 0) — 호스트 기저 미사용.
        assert_eq!(
            W4a16Dec::kv_block_addr(bv, bh, 3, stride, 4),
            0x1_0000 + 3 * 0x100
        );
    }
}
