use super::*;

impl W4a16Dec {
    pub(super) fn ensure_norm_bufs(&mut self, t_len: usize) -> Result<(), String> {
        let need = t_len * self.hidden;
        if need > self.norm.cap {
            self.cc.capture_guard("norm_cap"); // [R12]
            self.graph_invalidate(); // [P10] 재할당 — 캡처 옛 포인터 차단.
            self.cc.sync()?; // [P10] 비행 커널의 해제 버퍼 사용 차단.
            self.norm.cap = 0; // G1: 실패 시 재진입 보장(성공 뒤에만 갱신).
            realloc_fields(
                |p| self.cc.free(p),
                |n| self.cc.alloc(n),
                [&mut self.norm.dx, &mut self.norm.dab, &mut self.norm.dxn],
                [need * 4, need * 4, need * 4],
            )?;
            self.norm.cap = need;
        }
        Ok(())
    }

    /// dx32(x32 버퍼) 용량 보장 — 융합 노름·cast_x32 공용.
    pub(super) fn ensure_dx32(&mut self, n: usize) -> Result<CUdeviceptr, String> {
        if n > self.dx32.cap {
            self.cc.capture_guard("dx32_cap"); // [R12]
            self.graph_invalidate(); // [P10]
            self.cc.sync()?; // [P10] 비행 커널의 해제 버퍼 사용 차단.
            if self.dx32.ptr != 0 {
                self.cc.free(self.dx32.ptr)?;
            }
            self.dx32.ptr = 0; // G1
            self.dx32.cap = 0;
            self.dx32.ptr = self.cc.alloc(n * 4)?;
            self.dx32.cap = n;
        }
        Ok(self.dx32.ptr)
    }

    pub(super) fn ensure_ew_bufs(&mut self, n: usize) -> Result<(), String> {
        if n > self.ew.cap {
            self.cc.capture_guard("ew_cap"); // [R12]
            self.graph_invalidate(); // [P10]
            self.cc.sync()?; // [P10] 비행 커널의 해제 버퍼 사용 차단.
            self.ew.cap = 0; // G1
            realloc_fields(
                |p| self.cc.free(p),
                |b| self.cc.alloc(b),
                [&mut self.ew.dewg, &mut self.ew.dewu, &mut self.ew.dew],
                [n * 4, n * 4, n * 4],
            )?;
            self.ew.cap = n;
        }
        Ok(())
    }

    pub(super) fn ensure_gdn_bufs(&mut self, t_len: usize) -> Result<(), String> {
        if t_len <= self.gdn_t_cap {
            return Ok(());
        }
        let dm = self.gdn.ok_or("GDN: 형상 미등록")?;
        let (hd, cch, kl, vl) = (dm.hidden, dm.conv_ch(), dm.k_len(), dm.v_len());
        self.graph_invalidate(); // [P10]
        self.cc.sync()?; // [P10] 비행 커널의 해제 버퍼 사용 차단.
        self.gdn_t_cap = 0; // G1: 실패 시 재진입 보장.
        realloc_fields(
            |p| self.cc.free(p),
            |b| self.cc.alloc(b),
            [
                &mut self.dqkv,
                &mut self.dzv,
                &mut self.dgxn,
                &mut self.dgq,
                &mut self.dgk,
                &mut self.dgv,
                &mut self.dq2,
                &mut self.dk2,
                &mut self.dv2,
                &mut self.dbg,
                &mut self.dgo,
                &mut self.dgate,
                &mut self.dgpart,
                &mut self.dgdc,
                &mut self.dakq,
            ],
            [
                t_len * cch * 4,
                t_len * vl * 4,
                t_len * hd * 4,
                t_len * kl * 4,
                t_len * kl * 4,
                t_len * vl * 4,
                t_len * kl * 4,
                t_len * kl * 4,
                t_len * vl * 4,
                t_len * dm.bg_len() * 4,
                t_len * vl * 4,
                t_len * vl * 4,
                // [A9] 토큰별 트리오 스크래치 — 배치 디코드 = 슬롯 수 상한.
                self.n_slots * dm.h_v * 8 * 256 * 4,
                self.n_slots * dm.h_v * 128 * 4,
                // [A5-4] A/KQ prepass 스크래치(최대 청크 수 기준, t_len 무관).
                dm.h_v * (CHAIN_TMAX / GDN_CS) * 2 * GDN_CS * GDN_CS * 2,
            ],
        )?;
        self.gdn_t_cap = t_len;
        Ok(())
    }

    pub(super) fn ensure_attn_bufs(&mut self, t_len: usize) -> Result<(), String> {
        if t_len <= self.attn_t_cap {
            return Ok(());
        }
        let dm = self.attn.ok_or("attn: 형상 미등록")?;
        self.graph_invalidate(); // [P10]
        self.cc.sync()?; // [P10] 비행 커널의 해제 버퍼 사용 차단.
        self.attn_t_cap = 0; // G1: 실패 시 재진입 보장.
        realloc_fields(
            |p| self.cc.free(p),
            |b| self.cc.alloc(b),
            [
                &mut self.dqg_a,
                &mut self.dkin_a,
                &mut self.dvin_a,
                &mut self.dqh_a,
                &mut self.doutv_a,
                &mut self.dattn_part,
            ],
            [
                t_len * dm.qg_dim() * 4,
                t_len * dm.kv_dim() * 4,
                t_len * dm.kv_dim() * 4,
                t_len * dm.q_dim() * 4,
                t_len * dm.q_dim() * 4,
                t_len * dm.q_heads * ATTN_SPLITS * 258 * 4,
            ],
        )?;
        self.attn_t_cap = t_len;
        Ok(())
    }

    /// 체인 작업 버퍼 보장(1회 — 폭은 선형 형상에서 산출).
    pub(super) fn ensure_chain_bufs(&mut self) -> Result<(), String> {
        if self.chain_bufs_ok {
            return Ok(());
        }
        let h = self.hidden;
        let ad = self.attn.ok_or("attn: 형상 미등록")?;
        let gd = self.gdn.ok_or("GDN: 형상 미등록")?;
        let (ff_gate, ff_up) = if self.plain_weights {
            // 플레인(MoE) 모드 — dense FFN 없음. ew 스테이징 폭은 shared 폭.
            let (_, g, _) = self.plain_spec("blk.0.moe_shared_gate.weight")?;
            let (_, u, _) = self.plain_spec("blk.0.moe_shared_up.weight")?;
            (g, u)
        } else {
            let (_, _, g, _) = self.lin_spec("blk.0.ffn_gate.weight")?;
            let (_, _, u, _) = self.lin_spec("blk.0.ffn_up.weight")?;
            (g, u)
        };
        // 슬롯 0 = qg·qkv·gate/up, 슬롯 1 = kin/vin·z·up. ew가 gate·up을
        // 동시에 읽으므로 둘 다 FFN 폭 확보(구 S10 ensure_chain_bufs 계약).
        let w0 = ad.qg_dim().max(gd.conv_ch()).max(ff_gate);
        let w1 = ad.kv_dim().max(gd.v_len()).max(ff_up);
        // 배치 프리필(t≤CHAIN_TMAX)까지 수용 — 버퍼는 t배 폭으로 잡는다
        // (t=1 경로는 오프셋 0만 사용하므로 동작 불변).
        let tb = CHAIN_TMAX;
        self.graph_invalidate(); // [P10]
        self.cc.sync()?; // [P10] 비행 커널의 해제 버퍼 사용 차단.
        let [c0, c1, c2, c3, c4] = &mut self.dchain;
        realloc_fields(
            |p| self.cc.free(p),
            |b| self.cc.alloc(b),
            [&mut self.dres, &mut self.dab_dev, c0, c1, c2, c3, c4],
            [
                tb * h * 4,
                tb * h * 4,
                tb * w0 * 4,
                tb * w1 * 4,
                tb * w1 * 4,
                tb * ff_up * 4,
                tb * h * 4,
            ],
        )?;
        Self::zero_dev(&self.cc, self.dab_dev, tb * h * 4)?;
        self.stg_w0 = w0;
        self.stg_w1 = w1;
        self.stg_w2 = ff_up;
        self.chain_bufs_ok = true;
        Ok(())
    }

    /// 배치 출력 스크래치 보장([t][max_n]).
    /// 플레인 모드(MoE)는 lins가 비어 있다 — plains·전문가 폭까지 포함해야
    /// dyt가 NULL로 남지 않는다(실측: ssm_out GEMM이 NULL에 쓰러 크래시).
    pub(super) fn ensure_dyt(&mut self, t: usize) -> Result<CUdeviceptr, String> {
        let (mut mn, mut mk) = (0usize, 0usize);
        for &(_, _, n, k) in self.lins.values() {
            mn = mn.max(n);
            mk = mk.max(k);
        }
        for &(_, n, k) in self.plains.values() {
            mn = mn.max(n);
            mk = mk.max(k);
        }
        mn = mn.max(self.hidden).max(self.n_experts);
        let need = t * mn;
        if need > self.dyt.cap {
            self.cc.capture_guard("dyt_cap"); // [R12]
            self.graph_invalidate(); // [P10]
            self.cc.sync()?; // [P10] 비행 커널의 해제 버퍼 사용 차단.
            if self.dyt.ptr != 0 {
                self.cc.free(self.dyt.ptr)?;
            }
            self.dyt.ptr = 0; // G1
            self.dyt.cap = 0;
            self.dyt.ptr = self.cc.alloc(need * 4)?;
            self.dyt.cap = need;
        }
        // 배치 캐스트 입력(xh: t×k f16)도 함께 보장.
        if t * mk > self.xh.cap {
            self.cc.capture_guard("xh_cap"); // [R12]
            self.graph_invalidate(); // [P10]
            self.cc.sync()?;
            if self.xh.ptr != 0 {
                self.cc.free(self.xh.ptr)?;
            }
            self.xh.ptr = 0;
            self.xh.cap = 0;
            self.xh.ptr = self.cc.alloc(t * mk * 2)?;
            self.xh.cap = t * mk;
        }
        Ok(self.dyt.ptr)
    }

    /// dy 버퍼 보장(n f32).
    pub(super) fn ensure_dy(&mut self, n: usize) -> Result<CUdeviceptr, String> {
        if n > self.dy.cap {
            self.cc.capture_guard("y_cap"); // [R12]
            self.graph_invalidate(); // [P10]
            self.cc.sync()?; // [P10] 비행 커널의 해제 버퍼 사용 차단.
            if self.dy.ptr != 0 {
                self.cc.free(self.dy.ptr)?;
            }
            self.dy.ptr = 0; // G1
            self.dy.cap = 0;
            self.dy.ptr = self.cc.alloc(n * 4)?;
            self.dy.cap = n;
        }
        Ok(self.dy.ptr)
    }

    /// MoE 버퍼 보장(전문가 스테이징·라우터·출력).
    pub(super) fn ensure_moe_bufs(&mut self) -> Result<(), String> {
        if self.moe_bufs_ok {
            return Ok(());
        }
        let h = self.hidden;
        let n_ff = self.moe_ffn;
        let n_exp = self.n_experts;
        if n_ff == 0 || n_exp == 0 || !self.moe_group.is_multiple_of(32) {
            return Err("moe: 구성 미등록".into());
        }
        // 프리필 t≤CHAIN_TMAX까지 수용 — 슬롯 = TMAX×top_k, 라우터 = TMAX×n_exp.
        let tmax = CHAIN_TMAX;
        // 전문가 최대 행렬 = h×n_ff(gate/up) = h×n_ff(down) — 동일 크기.
        // [P7 2026-10-09] 층 단위 배치 스테이징 — top_k×3 proj 슬라이스.
        // proj 방향과 무관하게 동일 크기(gate/up [n_ff,h] · down [h,n_ff]):
        // q = h·n_ff/2, s = h·n_ff/g·2. 슬라이스별 업로드 후 sync 1회.
        let np = self.top_k.max(1) * 3;
        let pk = h * n_ff / 2 * np;
        let sk = h * n_ff / self.moe_group * 2 * np;
        self.graph_invalidate(); // [P10]
        self.cc.sync()?; // [P10] 비행 커널의 해제 버퍼 사용 차단.
        realloc_fields(
            |p| self.cc.free(p),
            |b| self.cc.alloc(b),
            [
                &mut self.stg.q,
                &mut self.stg.s,
                &mut self.drt.ptr,
                &mut self.dmo.ptr,
            ],
            [pk, sk, tmax * n_exp * 4, tmax * h * 4],
        )?;
        self.stg.cap = (pk, sk);
        self.drt.cap = tmax * n_exp;
        self.dmo.cap = tmax * h;
        // 배치 전문가 출력([TMAX×top_k][n_ff]·[TMAX×top_k][h]) + 슬롯 idx/가중.
        let tk = tmax * self.top_k.max(1);
        // [P11] 전문가-우선 정렬 버퍼 — gslot[n_exp][tk] + cnt[n_exp].
        if self.moe_gslot == 0 || self.moe_gmax < tk {
            self.graph_invalidate();
            self.cc.sync()?;
            for p in [self.moe_gslot, self.moe_gcnt, self.moe_goff] {
                if p != 0 {
                    self.cc.free(p)?;
                }
            }
            self.moe_gslot = 0;
            self.moe_gcnt = 0;
            self.moe_goff = 0;
            self.moe_gmax = 0;
            self.moe_gslot = self.cc.alloc(n_exp * tk * 4)?;
            self.moe_gcnt = self.cc.alloc(n_exp * 4)?;
            self.moe_goff = self.cc.alloc(n_exp * 4)?;
            self.moe_gmax = tk;
        }
        if self.exp.cap < tk {
            for p in [self.exp.gate, self.exp.up, self.exp.act, self.exp.dn] {
                if p != 0 {
                    self.cc.free(p)?;
                }
            }
            self.exp.gate = 0;
            self.exp.up = 0;
            self.exp.act = 0;
            self.exp.dn = 0;
            self.exp.cap = 0;
            self.exp.gate = self.cc.alloc(tk * n_ff * 4)?;
            self.exp.up = self.cc.alloc(tk * n_ff * 4)?;
            self.exp.act = self.cc.alloc(tk * n_ff * 4)?;
            self.exp.dn = self.cc.alloc(tk * h * 4)?;
            self.exp.cap = tk;
        }
        for p in [self.moe_idx, self.moe_wt] {
            if p != 0 {
                self.cc.free(p)?;
            }
        }
        self.moe_idx = 0;
        self.moe_wt = 0;
        self.moe_idx = self.cc.alloc(tk * 4)?;
        self.moe_wt = self.cc.alloc(tk * 4)?;
        // [B-4] 다중 세그먼트 GEMV 테이블(drt/dchain 포인터 참조 — 재할당 시 재작성).
        self.ensure_chain_bufs()?;
        self.build_moe_multi_tab()?;
        self.moe_bufs_ok = true;
        Ok(())
    }

    /// 캡처 전 버퍼 워밍업 — **불변식: 체인에서 지연 할당되는 모든 버퍼는
    /// 여기서 선할당한다.** 캡처 중 `cuMemAlloc`은 금지 API라 드라이버가
    /// instantiate에서 SIGSEGV로 죽는다(2026-10-08 실측 — gemv dx32/dy 누락이
    /// 원인이었다). 체인에 새 버퍼를 추가하면 반드시 이 목록에도 추가할 것.
    /// 현행 지연 할당원: chain(dres/dab_dev/dchain) · norm(dx/dab/dxn) ·
    /// gdn 12종 · attn 5종 · gemv_dev(dx32/dy, 선형 전수 최대치) ·
    /// head(head_w/head_out — upload_head 소관).
    pub(super) fn warm_for_capture(&mut self) -> Result<(), String> {
        self.ensure_chain_bufs()?;
        if self.n_experts > 0 {
            // P1: MoE 상주 체인의 지연 할당원(drt/dexp_*/dmo/moe_idx/moe_wt).
            self.ensure_moe_bufs()?;
        }
        // [P10] t=1이 아니라 **최대 청크(CHAIN_TMAX)** 로 선할당 — 그래프는
        // 캡처 시점의 포인터를 기록하므로, 이후 프리필이 버퍼를 재할당하면
        // replay가 해제 주소를 쓴다(위 graph_invalidate 주석의 실측 결함).
        self.ensure_norm_bufs(CHAIN_TMAX)?;
        self.ensure_gdn_bufs(CHAIN_TMAX)?;
        self.ensure_attn_bufs(CHAIN_TMAX)?;
        self.ensure_dyt(CHAIN_TMAX)?;
        // [P11 fix] lins + plains **둘 다** — 플레인(MoE) 모델은 가중치가
        // plains에 있어 lins만 보면 dy/dx32가 0 → 캡처 중 ensure_dy 재할당 →
        // 재할당 경로의 sync가 캡처 금지 API(CUresult=900)로 캡처 실패(실측:
        // 35B serve가 직접 경로로 폴백 → tg 93.6→81.6).
        let (mut mk, mut mn) = (0usize, 0usize);
        for &(_, _, n, k) in self.lins.values() {
            mk = mk.max(k);
            mn = mn.max(n);
        }
        for &(_, n, k) in self.plains.values() {
            mk = mk.max(k);
            mn = mn.max(n);
        }
        // 프리필 norm 융합(ensure_dx32(t*h))과 FFN 캐스트(cast_x32(t×k))를
        // 모두 커버 — 캡처 후 재할당(재캡처·옛 포인터)을 봉인한다.
        mk = mk.max(CHAIN_TMAX * mk.max(self.hidden));
        mn = mn.max(self.hidden);
        if mk > self.dx32.cap {
            self.cc.capture_guard("dx32_cap"); // [R12]
            self.graph_invalidate(); // [P10]
            self.cc.sync()?;
            if self.dx32.ptr != 0 {
                self.cc.free(self.dx32.ptr)?;
            }
            self.dx32.ptr = 0; // G1 관례: 실패 시 재시도 이중해제 방지.
            self.dx32.cap = 0;
            self.dx32.ptr = self.cc.alloc(mk * 4)?;
            self.dx32.cap = mk;
        }
        if mk > self.dx16.cap {
            self.cc.capture_guard("dx16_cap"); // [R12]
            self.graph_invalidate(); // [P10]
            self.cc.sync()?;
            if self.dx16.ptr != 0 {
                self.cc.free(self.dx16.ptr)?;
            }
            self.dx16.ptr = 0;
            self.dx16.cap = 0;
            self.dx16.ptr = self.cc.alloc(mk * 2)?;
            self.dx16.cap = mk;
        }
        if mn > self.dy.cap {
            self.cc.capture_guard("y_cap"); // [R12]
            self.graph_invalidate(); // [P10]
            self.cc.sync()?;
            if self.dy.ptr != 0 {
                self.cc.free(self.dy.ptr)?;
            }
            self.dy.ptr = 0;
            self.dy.cap = 0;
            self.dy.ptr = self.cc.alloc(mn * 4)?;
            self.dy.cap = mn;
        }
        Ok(())
    }

    /// [A9] 배치 디코드 부속 버퍼 — 로짓[t×head_n]·argmax[t]·핀드 입출력.
    pub(super) fn ensure_batch_bufs(&mut self) -> Result<(), String> {
        if self.head_n == 0 {
            return Err("batch: head 미등록(업로드 선행)".into());
        }
        let tmax = self.n_slots.min(BATCH_DEC_MAX);
        if self.dbatch_lg == 0 {
            self.dbatch_lg = self.cc.alloc(tmax * self.head_n * 4)?;
            self.dbatch_am = self.cc.alloc(tmax * 4)?;
        }
        if self.pin_batch_tok.is_null() {
            self.pin_batch_tok = self.cc.pinned_alloc(tmax * 4)?;
            self.pin_batch_in = self.cc.pinned_alloc(tmax * self.hidden * 4)?;
            self.pin_batch_pos = self.cc.pinned_alloc(self.n_slots * 4)?;
        }
        Ok(())
    }
}
