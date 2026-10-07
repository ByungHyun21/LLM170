//! EXL3 CUDA 순차 forward — 호스트 스테이징 v1(plans/cuda-port.md S5).
//! 모듈 프로브에서 확인된 GEMV·norm·GDN·attn·ew를 한 토큰씩 조립한다.
//! 잔차는 별도 디바이스 버퍼에 상주한다: gemv_host가 공유 dx를 덮어써도
//! norm의 x는 유지된다. xn만 판독, 다음 분기 ab만 업로드하는 규약이다.
//!
//! [슬롯 — plans/cuda-port.md S8] 슬롯은 GDN 링/스캔 상태·KV 캐시·pos의
//! 호스트↔디바이스 상태 집합이다. 가중치(dcw·dqnw·선형)는 전 슬롯 공유이므로
//! 커널 layer 인덱스를 유지하고, 상태 포인터만 슬롯 오프셋으로 넘긴다.

use crate::rawcuda::exl3_cuda::Exl3CudaDecoder;
use crate::rawcuda::ffi::CUdeviceptr;

impl Exl3CudaDecoder {
    /// 슬롯 1토큰 계산(순차 디코드 경로).
    pub fn forward_tok(&mut self, slot: usize, tok: u32) -> Result<Vec<f32>, String> {
        let row = self.embed_row_host(tok);
        if row.len() != self.hidden {
            return Err(format!("exl3-cuda: 임베딩 토큰 {tok} 범위 밖 또는 미적재"));
        }
        self.forward(slot, &row).map(|(logits, _)| logits)
    }

    /// 슬롯 1스텝 greedy: argmax 스캔 폭은 로짓 전체 길이(결함 8호).
    /// forward와 argmax가 같은 컨텍스트 스코프를 써야 하므로 가드를
    /// 여기로 올린다(슬롯 스레드에 전파되지 않는 current 컨텍스트 —
    /// plans/cuda-port.md S5). 중첩 가드는 재진입 가능(prev=자신).
    pub fn step_tok(&mut self, slot: usize, tok: u32) -> Result<u32, String> {
        let _g = self.cc.guard()?;
        let logits = self.forward_tok(slot, tok)?;
        self.argmax_host(&logits)
    }

    /// 슬롯 임베딩 행 1개 → (로짓, 최종 노름 이전 잔차) — plans/cuda-port.md S5.
    pub fn forward(
        &mut self,
        slot: usize,
        embed_row: &[f32],
    ) -> Result<(Vec<f32>, Vec<f32>), String> {
        self.forward_host_staged(slot, embed_row)
    }

    /// 순차 한 행 계산: 반환 (전체 로짓, 최종 노름 이전 잔차).
    /// HIP forward와 동일하게 GDN/어텐션 상태를 전진시키며, 배치는 사용하지 않는다.
    pub(crate) fn forward_host_staged(
        &mut self,
        slot: usize,
        embed_row: &[f32],
    ) -> Result<(Vec<f32>, Vec<f32>), String> {
        if self.hidden == 0 || embed_row.len() != self.hidden {
            return Err(format!(
                "exl3-cuda: 임베딩 폭 {} != hidden {}",
                embed_row.len(),
                self.hidden
            ));
        }
        let n_slots = self.n_slots.max(1);
        if slot >= n_slots {
            return Err(format!("exl3-cuda: slot={slot} >= n_slots={n_slots}"));
        }
        let pos = self.slot_pos[slot];
        let cap = self.attn_dims()?.cap;
        if pos as usize >= cap {
            return Err(format!(
                "context overflow: slot{slot} pos={pos} + 1 > kvcap={cap} (--ctx 상향 필요)"
            ));
        }
        // 위치축 한계(fwd3s 공유메모리 sarr 1024행, S9). cap이 커도 이 값을
        // 넘으면 커널이 illegal address로 죽는다 — 여기가 진짜 상한이다.
        // 조용한 오답이 아니라 조기에 명확히 거부한다.
        if pos as usize + 1 > crate::rawcuda::attn_cuda::ATTN_SCORE_SCAP {
            return Err(format!(
                "context overflow: slot{slot} pos={pos}+1 > {} \
                 (CUDA fwd3s 위치축 한계 — 위치 청크 미구현, plans/cuda-port.md S9)",
                crate::rawcuda::attn_cuda::ATTN_SCORE_SCAP
            ));
        }
        if self.norm_w_rows < 2 * self.n_layers + 1 || self.gdn.is_none() {
            return Err("exl3-cuda: 디코더 상수 미등록 — load_with_ctx 필요".into());
        }
        // GEMV가 dx를 입력 스테이징으로 재사용하므로 잔차 소유권은 별도
        // dres에 둔다(계획 S5의 잔차 프로토콜). 오류 시에도 해제한다.
        // 컨텍스트 가드는 슬롯 스레드 진입 시 필수다: 로드 스레드의
        // current 컨텍스트는 다른 스레드로 전파되지 않으므로, 가드 없이
        // alloc/h2d하면 CUresult=201(INVALID_CONTEXT)로 죽는다(plans/cuda-port.md S5).
        let _g = self.cc.guard()?;
        let dres = self.cc.alloc(self.hidden * 4)?;
        let result = (|| {
            // SAFETY: f32 [hidden]을 바이트 뷰로 복사; GPU 사용 전까지 살아 있다.
            let row = unsafe {
                std::slice::from_raw_parts(embed_row.as_ptr() as *const u8, self.hidden * 4)
            };
            self.cc.h2d(dres, row)?;
            self.forward_resident(slot, pos, dres)
        })();
        let freed = self.cc.free(dres);
        match (result, freed) {
            (Err(e), _) => Err(e),
            (_, Err(e)) => Err(e),
            (Ok(v), Ok(())) => Ok(v),
        }
    }

    fn forward_resident(
        &mut self,
        slot: usize,
        pos: u32,
        dres: CUdeviceptr,
    ) -> Result<(Vec<f32>, Vec<f32>), String> {
        let mut ab = vec![0.0f32; self.hidden];
        let mut gi = 0usize;
        for il in 0..self.loaded_layers.min(self.n_layers) {
            let lp = format!("model.language_model.layers.{il}");
            let xn = self
                .norm_resid_staged(2 * il, dres, &ab)
                .map_err(|e| format!("L{il} input norm: {e}"))?;
            ab = if il % 4 == 3 {
                // qg=(q‖gate) 인터리브; prep에서 디인터리브 + q/k norm + RoPE.
                let att = format!("{lp}.self_attn");
                let q = self
                    .gemv_host(&format!("{att}.q_proj"), &xn)
                    .map_err(|e| format!("L{il} q_proj: {e}"))?;
                let k = self
                    .gemv_host(&format!("{att}.k_proj"), &xn)
                    .map_err(|e| format!("L{il} k_proj: {e}"))?;
                let v = self
                    .gemv_host(&format!("{att}.v_proj"), &xn)
                    .map_err(|e| format!("L{il} v_proj: {e}"))?;
                let out = self
                    .attn_chain_host(slot, il / 4, 1, &q, &k, &v, pos)
                    .map_err(|e| format!("L{il} attn: {e}"))?;
                self.gemv_host(&format!("{att}.o_proj"), &out.outv)
                    .map_err(|e| format!("L{il} o_proj: {e}"))?
            } else {
                let att = format!("{lp}.linear_attn");
                let qkv = self
                    .gemv_host(&format!("{att}.in_proj_qkv"), &xn)
                    .map_err(|e| format!("L{il} in_proj_qkv: {e}"))?;
                let z = self
                    .gemv_host(&format!("{att}.in_proj_z"), &xn)
                    .map_err(|e| format!("L{il} in_proj_z: {e}"))?;
                let gated = self
                    .gdn_chain_host(slot, gi, 1, &xn, &qkv, &z, None, None)
                    .map_err(|e| format!("L{il} gdn: {e}"))?;
                gi += 1;
                self.gemv_host(&format!("{att}.out_proj"), &gated)
                    .map_err(|e| format!("L{il} out_proj: {e}"))?
            };
            let xn = self
                .norm_resid_staged(2 * il + 1, dres, &ab)
                .map_err(|e| format!("L{il} post norm: {e}"))?;
            let mlp = format!("{lp}.mlp");
            let gate = self
                .gemv_host(&format!("{mlp}.gate_proj"), &xn)
                .map_err(|e| format!("L{il} gate_proj: {e}"))?;
            let up = self
                .gemv_host(&format!("{mlp}.up_proj"), &xn)
                .map_err(|e| format!("L{il} up_proj: {e}"))?;
            let activated = self
                .ew_host(&gate, &up)
                .map_err(|e| format!("L{il} ew: {e}"))?;
            ab = self
                .gemv_host(&format!("{mlp}.down_proj"), &activated)
                .map_err(|e| format!("L{il} down_proj: {e}"))?;
        }
        // HIP forward_staged와 동일한 두 번째 반환값: 마지막 분기 ab를
        // 더하기 전의 dx. lm_head 입력 xn이나 x+ab와 혼동 금지.
        let mut hidden_bytes = vec![0u8; self.hidden * 4];
        self.cc.d2h(&mut hidden_bytes, dres)?;
        self.cc.sync()?;
        let xn = self.norm_resid_staged(2 * self.n_layers, dres, &ab)?;
        let logits = self.gemv_host("lm_head", &xn)?;
        self.slot_pos[slot] = pos + 1;
        self.attn_set_pos(slot, pos + 1)?;
        // SAFETY: d2h 동기 완료; hidden_bytes는 hidden개의 f32 LE 값이다.
        let hidden = unsafe {
            std::slice::from_raw_parts(hidden_bytes.as_ptr() as *const f32, self.hidden).to_vec()
        };
        Ok((logits, hidden))
    }

    /// 슬롯의 GDN 상태만 0화한다. KV는 위치 0부터 재기록되므로 초기화할
    /// 필요가 없다(HIP reset_state의 인과 순서 계약).
    pub fn reset_state(&mut self, slot: usize) -> Result<(), String> {
        let _g = self.cc.guard()?;
        let n_slots = self.n_slots.max(1);
        if slot >= n_slots {
            return Err(format!("exl3-cuda: slot={slot} >= n_slots={n_slots}"));
        }
        self.slot_pos[slot] = 0;
        self.attn_set_pos(slot, 0)?;
        if let Some(dims) = self.gdn {
            // SAFETY: 슬롯 최외곽 레이아웃 경계 내(S8) — 층 슬라이스와
            // 같은 stride로 블록 하나를 덮는다.
            let ring_bytes = dims.n_gdn * 3 * dims.conv_ch() * 4;
            let st_bytes = dims.n_gdn * dims.h_v * 128 * 128 * 4;
            let ring_off = (slot * dims.n_gdn * 3 * dims.conv_ch()) as u64 * 4;
            let st_off = (slot * dims.n_gdn * dims.h_v * 128 * 128) as u64 * 4;
            self.zero_state(self.dring + ring_off, ring_bytes)?;
            self.zero_state(self.dgst + st_off, st_bytes)?;
        }
        Ok(())
    }

    /// 전 슬롯 제자리 리셋(서버 워밍업 종료 후 — vk reset_states 미러).
    pub fn reset_states(&mut self) -> Result<(), String> {
        for s in 0..self.n_slots.max(1) {
            self.reset_state(s)?;
        }
        Ok(())
    }

    fn zero_state(&self, ptr: CUdeviceptr, len: usize) -> Result<(), String> {
        if ptr == 0 || len == 0 {
            return Err("exl3-cuda: 초기화되지 않은 상태 버퍼".into());
        }
        let zero = vec![0u8; (4 << 20).min(len)];
        for off in (0..len).step_by(zero.len()) {
            self.cc
                .h2d(ptr + off as u64, &zero[..zero.len().min(len - off)])?;
        }
        Ok(())
    }
}
