use super::mem::realloc_fields;
use super::*;

impl W4a16Dec {
    /// [R13 2026-10-12] 단일 Buf(ptr+cap) 재할당 공용 — [R12] capture_guard +
    /// [P10] 무효화·동기 순서를 한곳에 모은다. need_bytes = 새 용량(바이트),
    /// cap_elems = 기록할 원소 수. 신규 할당을 먼저 하고 기존을 해제한다 —
    /// alloc 실패 시 기존 버퍼가 보존된다(종전 free→alloc 실패 시 ptr=0 잔).
    pub(super) fn realloc_buf(
        &mut self,
        buf: Buf,
        need_bytes: usize,
        cap_elems: usize,
        what: &str,
    ) -> Result<Buf, String> {
        self.cc.capture_guard(what); // [R12]
        self.graph_invalidate(); // [P10] 재할당 — 캡처 옛 포인터 차단.
        self.cc.sync()?; // [P10] 비행 커널의 해제 버퍼 사용 차단.
        let ptr = self.cc.alloc(need_bytes)?;
        if buf.ptr != 0 {
            self.cc.free(buf.ptr)?;
        }
        Ok(Buf {
            ptr,
            cap: cap_elems,
        })
    }

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

    /// dx32(x32 버퍼) 용량 보장 — 융합 노름·생산자(ew/gate/merge) 공용.
    pub(super) fn ensure_dx32(&mut self, n: usize) -> Result<CUdeviceptr, String> {
        if n > self.dx32.cap {
            self.dx32 = self.realloc_buf(self.dx32, n * 4, n, "dx32_cap")?;
        }
        Ok(self.dx32.ptr)
    }

    /// [4.1 2026-10-11] f16 미러(dx16) 용량 보장 — ew/gate/merge 융합 기록용
    /// (cast_x16 인라인 가드와 동일 계약: capture_guard + P10 무효화·동기).
    pub(super) fn ensure_dx16(&mut self, n: usize) -> Result<CUdeviceptr, String> {
        if n > self.dx16.cap {
            self.dx16 = self.realloc_buf(self.dx16, n * 2, n, "dx16_cap")?;
        }
        Ok(self.dx16.ptr)
    }

    /// [A2b 2026-10-15] 그룹 mma A의 (hi|lo) packed 미러(u32/원소 = 4B) —
    /// `w4a16_cast_hilo` 산출 전용. 3-pass(A hi/lo)의 스테이징 소스.
    pub(super) fn ensure_dxp(&mut self, n: usize) -> Result<CUdeviceptr, String> {
        if n > self.dxp.cap {
            self.dxp = self.realloc_buf(self.dxp, n * 4, n, "dxp_cap")?;
        }
        Ok(self.dxp.ptr)
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
        if t_len <= self.gdn.gdn_t_cap {
            return Ok(());
        }
        let dm = self.gdn.gdn.ok_or("GDN: 형상 미등록")?;
        let (hd, cch, kl, vl) = (dm.hidden, dm.conv_ch(), dm.k_len(), dm.v_len());
        self.graph_invalidate(); // [P10]
        self.cc.sync()?; // [P10] 비행 커널의 해제 버퍼 사용 차단.
        self.gdn.gdn_t_cap = 0; // G1: 실패 시 재진입 보장.
        realloc_fields(
            |p| self.cc.free(p),
            |b| self.cc.alloc(b),
            [
                &mut self.gdn.dqkv,
                &mut self.gdn.dzv,
                &mut self.gdn.dgxn,
                &mut self.gdn.dgq,
                &mut self.gdn.dgk,
                &mut self.gdn.dgv,
                &mut self.gdn.dq2,
                &mut self.gdn.dk2,
                &mut self.gdn.dv2,
                &mut self.gdn.dbg,
                &mut self.gdn.dgo,
                &mut self.gdn.dgate,
                &mut self.gdn.dgpart,
                &mut self.gdn.dgdc,
                &mut self.gdn.dakq,
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
        self.gdn.gdn_t_cap = t_len;
        Ok(())
    }

    pub(super) fn ensure_attn_bufs(&mut self, t_len: usize) -> Result<(), String> {
        if t_len <= self.attn.attn_t_cap {
            return Ok(());
        }
        let dm = self.attn.attn.ok_or("attn: 형상 미등록")?;
        self.graph_invalidate(); // [P10]
        self.cc.sync()?; // [P10] 비행 커널의 해제 버퍼 사용 차단.
        self.attn.attn_t_cap = 0; // G1: 실패 시 재진입 보장.
        realloc_fields(
            |p| self.cc.free(p),
            |b| self.cc.alloc(b),
            [
                &mut self.attn.dqg_a,
                &mut self.attn.dkin_a,
                &mut self.attn.dvin_a,
                &mut self.attn.dqh_a,
                &mut self.attn.doutv_a,
                &mut self.attn.dattn_part,
            ],
            [
                t_len * dm.qg_dim() * 4,
                t_len * dm.kv_dim() * 4,
                t_len * dm.kv_dim() * 4,
                t_len * dm.q_dim() * 4,
                t_len * dm.q_dim() * 4,
                t_len
                    * dm.q_heads
                    * (if t_len <= 8 {
                        ATTN_SPLITS_D
                    } else {
                        ATTN_SPLITS
                    })
                    * 258
                    * 4,
            ],
        )?;
        self.attn.attn_t_cap = t_len;
        Ok(())
    }

    /// 체인 작업 버퍼 보장(1회 — 폭은 선형 형상에서 산출).
    pub(super) fn ensure_chain_bufs(&mut self) -> Result<(), String> {
        if self.chain_bufs_ok {
            return Ok(());
        }
        let h = self.hidden;
        let ad = self.attn.attn.ok_or("attn: 형상 미등록")?;
        let gd = self.gdn.gdn.ok_or("GDN: 형상 미등록")?;
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
        // [S3-1 2026-10-11] 토큰 id 스테이징(gather 입력) — 캡처 전 선할당 계약.
        if self.dtg == 0 {
            self.dtg = self.cc.alloc(tb * 4)?;
        }
        // [S2 2026-10-11] 다중세그먼트 테이블(branch·gate/up) — dchain/스펙
        // 확정 후 1회. 캡처 전(warm_for_capture→본 함수) 작성이 계약.
        self.build_multi_tables()?;
        self.chain_bufs_ok = true;
        Ok(())
    }

    /// [S2 2026-10-11] 다중세그먼트 테이블 작성 — 층별 branch(attn q/k/v ≤3,
    /// GDN qkv/z 2)·gate/up(2, split 전용 — 플레인은 MoE 공용 multi_gate가
    /// 담당). split = (q,s,out,n,k) u64×5 / 플레인 = (w,out,n,k) u64×4.
    /// 폭 검사는 종전 gemv_stage_x32/plain_stage_x32 계약 미러(스테이징 초과
    /// 거부). 내용은 이후 불변(그래프 정적 계약).
    fn build_multi_tables(&mut self) -> Result<(), String> {
        let plain = self.plain_weights;
        let interval = self.attn.attn.ok_or("attn: 형상 미등록")?.interval;
        let [s0, s1, s1b, _s2, _s3] = self.dchain;
        let (w0, w1) = (self.stg_w0, self.stg_w1);
        let nl = self.n_layers;
        let span = if plain { 20usize } else { 25usize }; // u64/층
        let mut tab: Vec<u64> = vec![0; nl * span];
        let mut meta: Vec<[u32; 8]> = Vec::with_capacity(nl);
        for il in 0..nl {
            let mut br: Vec<(String, u64, usize)> = Vec::new();
            if (il + 1) % interval == 0 {
                br.push((format!("blk.{il}.attn_q.weight"), s0, w0));
                br.push((format!("blk.{il}.attn_k.weight"), s1, w1));
                br.push((format!("blk.{il}.attn_v.weight"), s1b, w1));
            } else {
                br.push((format!("blk.{il}.attn_qkv.weight"), s0, w0));
                br.push((format!("blk.{il}.attn_gate.weight"), s1, w1));
            }
            let gu: Vec<(String, u64, usize)> = if plain {
                Vec::new()
            } else {
                vec![
                    (format!("blk.{il}.ffn_gate.weight"), s0, w0),
                    (format!("blk.{il}.ffn_up.weight"), s1, w1),
                ]
            };
            let mut wr = il * span;
            let mut gm = [[0u32; 4]; 2];
            for (gi, segs) in [&br, &gu].into_iter().enumerate() {
                let off = wr as u32;
                let mut total = 0usize;
                let mut tgrid = 0usize;
                let mut cnt = 0u32;
                for (name, out, wmax) in segs {
                    let (p0, p1, n, k) = if plain {
                        let (w, n, k) = self.plain_spec(name)?;
                        (w, 0u64, n, k)
                    } else {
                        self.lin_spec(name)?
                    };
                    if n > *wmax {
                        return Err(format!("gemv_multi({name}): n={n} > 스테이징 {wmax}"));
                    }
                    tab[wr] = p0;
                    wr += 1;
                    if !plain {
                        tab[wr] = p1;
                        wr += 1;
                    }
                    tab[wr] = *out;
                    wr += 1;
                    tab[wr] = n as u64;
                    wr += 1;
                    tab[wr] = k as u64;
                    wr += 1;
                    total += n;
                    tgrid += n.div_ceil(GEMV_TR);
                    cnt += 1;
                }
                gm[gi] = [off, cnt, total as u32, tgrid as u32];
            }
            meta.push([
                gm[0][0], gm[0][1], gm[0][2], gm[1][0], gm[1][1], gm[1][2], gm[0][3], gm[1][3],
            ]);
        }
        let bytes = tab.len() * 8;
        // SAFETY: 로컬 Vec<u64>의 유효 수명 내 바이트 뷰(길이 = 원소수×8).
        let buf = unsafe { std::slice::from_raw_parts(tab.as_ptr() as *const u8, bytes) };
        if self.dmulti == 0 {
            self.dmulti = self.cc.alloc(bytes)?;
        }
        self.cc.h2d(self.dmulti, buf)?;
        self.cc.sync()?;
        self.dmulti_meta = meta;
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
        mn = mn.max(self.hidden).max(self.moe.n_experts);
        let need = t * mn;
        if need > self.dyt.cap {
            self.dyt = self.realloc_buf(self.dyt, need * 4, need, "dyt_cap")?;
        }
        // 배치 캐스트 입력(xh: t×k f16)도 함께 보장.
        if t * mk > self.xh.cap {
            self.xh = self.realloc_buf(self.xh, t * mk * 2, t * mk, "xh_cap")?;
        }
        Ok(self.dyt.ptr)
    }

    /// dy 버퍼 보장(n f32).
    pub(super) fn ensure_dy(&mut self, n: usize) -> Result<CUdeviceptr, String> {
        if n > self.dy.cap {
            self.dy = self.realloc_buf(self.dy, n * 4, n, "y_cap")?;
        }
        Ok(self.dy.ptr)
    }

    /// [gemma4 스트리밍 2026-10-14] 스트리밍 MoE 스테이징 체인 보장 — 공용
    /// `moe_experts_streaming`은 GEMV 출력 스테이징으로 dchain[0/1/3/4]
    /// (s0/s1/s2/s3)를 쓴다. qwen은 `ensure_chain_bufs`가 담당하지만 gemma4
    /// (순수 어텐션 체인)는 그 경로가 없어 0인 채로 남는다 — 그대로 쓰면
    /// 주소 0에 GEMV 출력을 써 illegal access(실측 CUresult=700). 등록 시 1회.
    pub fn ensure_moe_stage_chain(&mut self) -> Result<(), String> {
        if self.dchain[0] != 0 {
            return Ok(());
        }
        // 폭 = max(전문가 중간 n_ff, hidden)행 × f32(두 GEMV 출력 최대치).
        let w = self.hidden.max(self.moe.moe_ffn).max(1) * 4;
        self.graph_invalidate(); // [P10]
        self.cc.sync()?; // [P10] 비행 커널의 해제 버퍼 사용 차단.
        for p in [0usize, 1, 3, 4] {
            self.dchain[p] = self.cc.alloc(w)?;
        }
        Ok(())
    }

    /// MoE 버퍼 보장(전문가 스테이징·라우터·출력).
    pub(super) fn ensure_moe_bufs(&mut self) -> Result<(), String> {
        if self.moe.moe_bufs_ok {
            return Ok(());
        }
        let h = self.hidden;
        let n_ff = self.moe.moe_ffn;
        let n_exp = self.moe.n_experts;
        if n_ff == 0 || n_exp == 0 || !self.moe.moe_group.is_multiple_of(32) {
            return Err("moe: 구성 미등록".into());
        }
        // 프리필 t≤CHAIN_TMAX까지 수용 — 슬롯 = TMAX×top_k, 라우터 = TMAX×n_exp.
        let tmax = CHAIN_TMAX;
        // 전문가 최대 행렬 = h×n_ff(gate/up) = h×n_ff(down) — 동일 크기.
        // [P7 2026-10-09] 층 단위 배치 스테이징 — top_k×3 proj 슬라이스.
        // proj 방향과 무관하게 동일 크기(gate/up [n_ff,h] · down [h,n_ff]):
        // q = h·n_ff/2, s = h·n_ff/g·2. 슬라이스별 업로드 후 sync 1회.
        let np = self.moe.top_k.max(1) * 3;
        let pk = h * n_ff / 2 * np;
        let sk = h * n_ff / self.moe.moe_group * 2 * np;
        self.graph_invalidate(); // [P10]
        self.cc.sync()?; // [P10] 비행 커널의 해제 버퍼 사용 차단.
        realloc_fields(
            |p| self.cc.free(p),
            |b| self.cc.alloc(b),
            [
                &mut self.moe.stg.q,
                &mut self.moe.stg.s,
                &mut self.moe.drt.ptr,
                &mut self.moe.dmo.ptr,
            ],
            [pk, sk, tmax * n_exp * 4, tmax * h * 4],
        )?;
        self.moe.stg.cap = (pk, sk);
        self.moe.drt.cap = tmax * n_exp;
        self.moe.dmo.cap = tmax * h;
        // 배치 전문가 출력([TMAX×top_k][n_ff]·[TMAX×top_k][h]) + 슬롯 idx/가중.
        let tk = tmax * self.moe.top_k.max(1);
        // [P11] 전문가-우선 정렬 버퍼 — gslot[n_exp][tk] + cnt[n_exp].
        if self.moe.moe_gslot == 0 || self.moe.moe_gmax < tk {
            self.graph_invalidate();
            self.cc.sync()?;
            for p in [self.moe.moe_gslot, self.moe.moe_gcnt, self.moe.moe_goff] {
                if p != 0 {
                    self.cc.free(p)?;
                }
            }
            self.moe.moe_gslot = 0;
            self.moe.moe_gcnt = 0;
            self.moe.moe_goff = 0;
            self.moe.moe_gmax = 0;
            self.moe.moe_gslot = self.cc.alloc(n_exp * tk * 4)?;
            self.moe.moe_gcnt = self.cc.alloc(n_exp * 4)?;
            self.moe.moe_goff = self.cc.alloc(n_exp * 4)?;
            self.moe.moe_gmax = tk;
        }
        if self.moe.exp.cap < tk {
            for p in [
                self.moe.exp.gate,
                self.moe.exp.up,
                self.moe.exp.act,
                self.moe.exp.dn,
            ] {
                if p != 0 {
                    self.cc.free(p)?;
                }
            }
            self.moe.exp.gate = 0;
            self.moe.exp.up = 0;
            self.moe.exp.act = 0;
            self.moe.exp.dn = 0;
            self.moe.exp.cap = 0;
            self.moe.exp.gate = self.cc.alloc(tk * n_ff * 4)?;
            self.moe.exp.up = self.cc.alloc(tk * n_ff * 4)?;
            self.moe.exp.act = self.cc.alloc(tk * n_ff * 4)?;
            self.moe.exp.dn = self.cc.alloc(tk * h * 4)?;
            self.moe.exp.cap = tk;
        }
        for p in [self.moe.moe_idx, self.moe.moe_wt] {
            if p != 0 {
                self.cc.free(p)?;
            }
        }
        self.moe.moe_idx = 0;
        self.moe.moe_wt = 0;
        self.moe.moe_idx = self.cc.alloc(tk * 4)?;
        self.moe.moe_wt = self.cc.alloc(tk * 4)?;
        // [B3 2026-10-12] shared 꼬리 포크 자원 — 보조 스트림·전용 sgate
        // 슬롯(t≤8행)·층별 이벤트 쌍. 1회 생성 후 불변(캡처 그래프가 참조).
        if self.cc.stream2.is_null() {
            self.cc.create_stream2()?;
        }
        if self.moe.dsg == 0 {
            self.moe.dsg = self.cc.alloc(8 * 4)?;
        }
        // [B7] shared 게이트/업 전용 슬롯 — dchain(d0/d1) 재사용은 스트리밍
        // 전문가 스크래치와 충돌(실측 골든 파손). t≤8 경로는 [t][n] 레이아웃
        // (tmax 배) — 크기 부족 시 인접 버퍼 파손(실측).
        let dsh_need = tmax * self.moe.shared_ffn.max(1);
        if self.moe.dsh_cap < dsh_need {
            for p in self.moe.dsh {
                if p != 0 {
                    self.cc.free(p)?;
                }
            }
            self.moe.dsh = [0, 0];
            self.moe.dsh_cap = 0;
            let sz = dsh_need * 4;
            self.moe.dsh = [self.cc.alloc(sz)?, self.cc.alloc(sz)?];
            self.moe.dsh_cap = dsh_need;
        }
        if self.moe.ev_fork.is_empty() && self.n_layers > 0 {
            for _ in 0..self.n_layers * 2 {
                let ev = self.cc.event_create_nt()?;
                self.moe.ev_fork.push(ev);
            }
        }
        // [B4-2 2026-10-12] 스트리밍 전문가 LRU 캐시 — 상주면 불필요(전문가
        // VRAM 상주). 여유의 1/4 한도로 K 산정(무료 VRAM 부족 시 자동 축소/비활성).
        if !self.moe.moe_resident && self.moe.mcache_k == 0 && self.n_layers > 0 {
            let qsz = h * n_ff / 2;
            let ssz = h * n_ff / self.moe.moe_group.max(32) * 2;
            let per_slot = 3 * (qsz + ssz);
            let free = crate::cuda_mem_free().map(|(f, _)| f).unwrap_or(0);
            let kmax = ((free / 4) / (self.n_layers * per_slot).max(1) as u64) as usize;
            let k = 96usize.min(kmax);
            if k > 0 {
                let qb = self.n_layers * k * 3 * qsz;
                let sb = self.n_layers * k * 3 * ssz;
                let qr = self.cc.alloc(qb);
                let sr = self.cc.alloc(sb);
                if let (Ok(q), Ok(s)) = (qr, sr) {
                    self.moe.mcache_q = q;
                    self.moe.mcache_s = s;
                    self.moe.mcache_k = k;
                    self.moe.mcache_ids = vec![u16::MAX; self.n_layers * k];
                    self.moe.mcache_age = vec![0u32; self.n_layers * k];
                    eprintln!(
                        "[moe] 스트리밍 전문가 캐시 {k}슬롯/층 ({}MiB)",
                        (qb + sb) >> 20
                    );
                    // [§3.4 ② 2026-10-12] 통합 캐시 회계 — 전문가 LRU(q+s
                    // 합산). touch는 트랜셰 2에서 적중률 기반 utility 승격으로
                    // 이관(핫패스 = 층×토큰마다라 현재는 미연결).
                    if let Err(e) = self.ucache.allocate(
                        "moe:expert-lru",
                        CacheCategory::ExpertWeights,
                        Tier::Hbm,
                        (qb + sb) as u64,
                        base_util::EXPERT_LRU,
                    ) {
                        eprintln!("[ucache] moe:expert-lru 등록 실패(회계 전용): {e}");
                    } else {
                        eprintln!("{}", self.ucache.log_line("moe:expert-lru"));
                    }
                    // [B4-1 2026-10-12] 스트리밍 t≤8 프리필용 장치 테이블
                    // ((q,s)쌍 per 층·전문가·투영) — 캐시 삽입이 엔트리를
                    // 갱신하고, 전문가 GEMV가 idx 간접 참조로 읽는다.
                    if self.moe.moe_dev_tab == 0 && self.moe.n_experts > 0 {
                        let n = self.n_layers * self.moe.n_experts * 3 * 2;
                        let p = self.cc.alloc(n * 8)?;
                        let zeros = vec![0u8; n * 8];
                        Self::h2d_chunked(&self.cc, p, &zeros)?;
                        self.moe.moe_dev_tab = p;
                    }
                }
            }
        }
        // [B-4] 다중 세그먼트 GEMV 테이블(drt/dchain 포인터 참조 — 재할당 시 재작성).
        // [gemma4 P2-3 2026-10-14] GDN 미등록(어텐션 전용 아치)은 qwen 체인 버퍼·
        // 다중 세그먼트 테이블이 불요 — gemma4 g4 경로는 전문가 개별 런치 +
        // accum만 쓴다(테이블은 qwen 체인 전용 소비).
        if self.gdn.gdn.is_some() {
            self.ensure_chain_bufs()?;
            self.build_moe_multi_tab()?;
        }
        self.moe.moe_bufs_ok = true;
        // [§3.4 ② 트랜셰 2] 등록(allocate) 중 퇴거가 결정됐을 수 있다 —
        // 여기서 실제 해제까지 완료(구성 확정 후 소비 — 순서 계약).
        self.ucache_sweep()?;
        Ok(())
    }

    /// [§3.4 ② 2026-10-12 트랜셰 2] 레지스트리 퇴거 실행 — `take_evicted()`
    /// 가 돌려준 엔트리의 실제 디바이스 메모리를 해제한다(enforce=true
    /// 국면의 소비 지점). 퇴거 결정은 등록 지점(set_attn·enable_spec·
    /// ensure_moe_bufs·ucache_pressure)의 allocate/set_budget 내부에서만
    /// 일어나고, 실제 해제는 이 메서드가 맡는다(레지스트리는 포인터를
    /// 모른다). 안전 계약(순서 고정):
    /// 1) 그래프 전량 무효화 — 캡처 exec가 옛 포인터(dchk/dsnap/mcache)를
    ///    replay하는 것 차단(graph_invalidate와 동일 계약).
    /// 2) sync — 비행 커널의 해제 버퍼 참조 차단(동일 계약).
    /// 3) 해제 + 소유 필드 0화 + 기능 플래그 정합:
    ///    - moe:expert-lru → 전문가 LRU 폐기. moe_bufs_ok=false로 재구성
    ///      유도 — 다음 moe_ffn_dev의 ensure_moe_bufs가 여유 재산정(자가
    ///      회복, 사이 디코드는 stg 폴백·t≤8 프리필은 재구성 후 진행).
    ///    - spec:ckpt → dchk 해제. checkpoint/restore 경로는 dchk==0
    ///      가드로 자동 비활성(재계산 가능한 순수 최적화 — 기능 무손실).
    ///    - spec:state → WY(dwy_s0/dc/g/k)+dsnap_ring/dring0 해제 +
    ///      spec_on=false. 검증 라운드는 라운드 사이(호스트 순차 지점)에서만
    ///      끊기므로 실행 중 라운드 부패는 없다(스케줄러는 spec_on()으로
    ///      선판정). dsave(프로브 전용 지연 할당)가 있으면 함께 해제.
    ///    - attn:kv → **올 수 없다**(Kv 범주 고정) — 오면 부기 정합성
    ///      오류로 거부(활성 KV 해제는 요청 파괴).
    ///
    /// 반환 = 실제 해제한 바이트(0 = 퇴거 없음).
    pub(super) fn ucache_sweep(&mut self) -> Result<u64, String> {
        let evicted = self.ucache.take_evicted();
        if evicted.is_empty() {
            return Ok(0);
        }
        self.graph_invalidate();
        self.cc.sync()?;
        let mut freed = 0u64;
        for e in &evicted {
            match e.id.as_str() {
                "moe:expert-lru" => {
                    for p in [self.moe.mcache_q, self.moe.mcache_s] {
                        if p != 0 {
                            self.cc.free(p)?;
                        }
                    }
                    (self.moe.mcache_q, self.moe.mcache_s, self.moe.mcache_k) = (0, 0, 0);
                    self.moe.mcache_ids.clear();
                    self.moe.mcache_age.clear();
                    self.moe.moe_bufs_ok = false; // 자가 재구성 유도
                }
                "spec:ckpt" => {
                    if !self.gdn.dchk.is_null() {
                        self.cc.pinned_free(self.gdn.dchk)?;
                    }
                    self.gdn.dchk = std::ptr::null_mut(); // null 가드로 체크포인트 자동 비활성
                }
                "spec:state" => {
                    for p in [
                        self.gdn.dwy_s0,
                        self.gdn.dwy_dc,
                        self.gdn.dwy_g,
                        self.gdn.dwy_k,
                        self.gdn.dsnap_ring,
                        self.gdn.dring0,
                        self.gdn.dsave,
                    ] {
                        if p != 0 {
                            self.cc.free(p)?;
                        }
                    }
                    (
                        self.gdn.dwy_s0,
                        self.gdn.dwy_dc,
                        self.gdn.dwy_g,
                        self.gdn.dwy_k,
                    ) = (0, 0, 0, 0);
                    (self.gdn.dsnap_ring, self.gdn.dring0, self.gdn.dsave) = (0, 0, 0);
                    self.spec_on = false; // 검증 라운드 차단
                }
                id => {
                    return Err(format!(
                        "ucache: 퇴거 실행 불가 엔트리 \"{id}\" — Kv는 퇴거 불가 계약 위반"
                    ));
                }
            }
            freed += e.bytes;
        }
        if freed > 0 {
            eprintln!(
                "[ucache] 퇴거 실행: {}엔트리 {}MiB 실해제",
                evicted.len(),
                freed >> 20
            );
        }
        Ok(freed)
    }

    /// [§3.4 ②] 런타임 VRAM 압박 반영 — HBM 예산 축소 → 등록부 퇴거 결정
    /// (set_budget 스윕) → 실제 해제(ucache_sweep)까지 한 번에 완료.
    /// 반환 = 실제 해제한 바이트. Kv는 고정이라 예산 이하로 내려가지
    /// 않아도(Kv 단독 과점유) 오류 아니다 — 정책상 허용 국면.
    pub fn ucache_pressure(&mut self, hbm_budget: u64) -> Result<u64, String> {
        self.ucache.set_budget(Tier::Hbm, hbm_budget);
        self.ucache_sweep()
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
        if self.moe.n_experts > 0 {
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
            self.dx32 = self.realloc_buf(self.dx32, mk * 4, mk, "dx32_cap")?;
        }
        if mk > self.dx16.cap {
            self.dx16 = self.realloc_buf(self.dx16, mk * 2, mk, "dx16_cap")?;
        }
        if mn > self.dy.cap {
            self.dy = self.realloc_buf(self.dy, mn * 4, mn, "y_cap")?;
        }
        Ok(())
    }

    /// [A9] 배치 디코드 부속 버퍼 — 로짓[t×head_n]·argmax[t]·핀드 입출력.
    pub(super) fn ensure_batch_bufs(&mut self) -> Result<(), String> {
        if self.head_n == 0 {
            return Err("batch: head 미등록(업로드 선행)".into());
        }
        // [A-1] 스펙 검증은 단일 슬롯에 t≤8토큰을 통과 — 배치 버퍼를 8로.
        let tmax = self
            .n_slots
            .min(BATCH_DEC_MAX)
            .max(if self.spec_on { BATCH_DEC_MAX } else { 0 })
            .max(self.batch_tmax_min);
        if self.dbatch_lg == 0 {
            self.dbatch_lg = self.cc.alloc(tmax * self.head_n * 4)?;
            self.dbatch_am = self.cc.alloc(tmax * 4)?;
        }
        if self.graph.pin_batch_tok.is_null() {
            self.graph.pin_batch_tok = self.cc.pinned_alloc(tmax * 4)?;
            self.graph.pin_batch_in = self.cc.pinned_alloc(tmax * self.hidden * 4)?;
            self.graph.pin_batch_pos = self.cc.pinned_alloc(self.n_slots * 4)?;
            // [S11] 샘플 로짓 판독 핀드 — pageable 대비 d2h 가속.
            self.graph.pin_batch_lg = self.cc.pinned_alloc(tmax * self.head_n * 4)?;
            // [배치 행 퓨전 2026-10-14] 행별 메타 핀드(blk/pp/ring/st/sc — u64×t).
            self.graph.pin_row_blk = self.cc.pinned_alloc(tmax * 8)?;
            self.graph.pin_row_pp = self.cc.pinned_alloc(tmax * 8)?;
            self.graph.pin_row_ring = self.cc.pinned_alloc(tmax * 8)?;
            self.graph.pin_row_st = self.cc.pinned_alloc(tmax * 8)?;
            self.graph.pin_row_sc = self.cc.pinned_alloc(tmax * 8)?;
        }
        if self.drow_blk == 0 {
            self.drow_blk = self.cc.alloc(tmax * 8)?;
            self.drow_pp = self.cc.alloc(tmax * 8)?;
            self.drow_ring = self.cc.alloc(tmax * 8)?;
            self.drow_st = self.cc.alloc(tmax * 8)?;
            self.drow_sc = self.cc.alloc(tmax * 8)?;
        }
        Ok(())
    }
}
