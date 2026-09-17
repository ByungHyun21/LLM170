//! q4acc QSA — 인덱서 어텐션 raw 내부 + QsaOps (plans/78 R1).

use super::*;

impl Q4Acc {
    /// q4_qsa_attn_sel 런치 본체 — 선택 목록(오름차순 위치)만 순회한다.
    #[allow(clippy::too_many_arguments)]
    pub fn qsa_attn_sel_raw(
        &self,
        q: &[f32],
        ck: &[f32],
        cv: &[f32],
        sel_idx: &[u32],
        sel_off: &[u32],
        kq_scale: f32,
        n_head: usize,
        n_kv: usize,
        hd: usize,
        t: usize,
    ) -> Result<Vec<f32>, String> {
        let (qdev, kdev, vdev, sdev, odev, ofdev) = {
            let mut a = self.qs.lock().map_err(|e| e.to_string())?;
            let qdev = a.ensure(&self.ctx, q.len() * 4)?;
            // KV는 컨텍스트 전체를 미리 잡는다(엔진이 주입한 ctx_len). 종전에는
            // n_past가 늘 때마다 재할당해 매 스텝 주소가 바뀌었다(실측 48회/세션).
            let kv_floats = self.ctx_len.load(std::sync::atomic::Ordering::Relaxed)
                * n_kv.max(1) * hd.max(1);
            let mut b = self.ckv.lock().map_err(|e| e.to_string())?;
            let kdev = b.ensure(&self.ctx, ck.len().max(kv_floats) * 4)?;
            let mut c = self.cvv.lock().map_err(|e| e.to_string())?;
            let vdev = c.ensure(&self.ctx, cv.len().max(kv_floats) * 4)?;
            let mut d = self.msk.lock().map_err(|e| e.to_string())?;
            let sdev = d.ensure(&self.ctx, sel_idx.len().max(1) * 4)?;
            let mut e2 = self.soff.lock().map_err(|e| e.to_string())?;
            let ofdev = e2.ensure(&self.ctx, sel_off.len().max(1) * 4)?;
            let mut f2 = self.atn.lock().map_err(|e| e.to_string())?;
            let odev = f2.ensure(&self.ctx, t * n_head * hd * 4)?;
            (qdev, kdev, vdev, sdev, odev, ofdev)
        };
        // 주의(실측 2026-09-14): 아래 h2d는 **매 호출 KV 캐시 전체**를 올린다.
        // n_past 8192에서 8192 x 2 x 256 x 4B x 2(K,V) = 33.6MB/층, 12층이면
        // 403MB/스텝이고 d2h 대역 실측(17.6 GB/s)으로 ~23ms/스텝 = 장문맥 스텝의 16%다.
        // 컨텍스트 스케일링 실측이 이를 지지한다: pp2048 103.9ms/스텝 ->
        // pp8192 141.8ms/스텝(+37.9)이고 KV 증가분만 302MB/스텝 ~17ms(증가의 45%)다.
        // 정공법은 KV를 디바이스 상주로 두고 증가분만 올리는 것이다. 다만
        // (ptr, len) 기반 델타 캐시는 **정확하지 않다**: 새 시퀀스의 첫 청크가
        // 이전 캐시 길이보다 길면 len이 커져 델타 경로로 빠지고 낡은 접두가 남는다
        // (짧은 시퀀스 1024 뒤에 긴 시퀀스가 2048로 시작하는 경우). 경계 내용
        // 비교도 동일 내용이면 통과해 버린다. 따라서 **명시적 리셋 신호**가 필요하다:
        // 스테이지는 pos0을 알고 있으므로 qsa_attention_sel 시그니처에 pos0을 넣거나
        // 프레임 begin에서 리셋을 알리는 것이 최소 변경이다(트레이트+CPU 폴백 수정).
        // (_sel4_raw도 같은 블록을 쓴다. 서버 다중 시퀀스·프롬프트 교체 검증 필수.)
        self.ctx.h2d(qdev, bytemuck::cast_slice(q))?;
        self.ctx.h2d(kdev, bytemuck::cast_slice(ck))?;
        self.ctx.h2d(vdev, bytemuck::cast_slice(cv))?;
        self.ctx.h2d(sdev, bytemuck::cast_slice(sel_idx))?;
        self.ctx.h2d(ofdev, bytemuck::cast_slice(sel_off))?;
        let mut q_p = qdev as *mut std::ffi::c_void;
        let mut k_p = kdev as *mut std::ffi::c_void;
        let mut v_p = vdev as *mut std::ffi::c_void;
        let mut si_p = sdev as *mut std::ffi::c_void;
        let mut so_p = ofdev as *mut std::ffi::c_void;
        let mut o_p = odev as *mut std::ffi::c_void;
        let mut sc = kq_scale;
        let mut nh = n_head as i32;
        let mut nk = n_kv as i32;
        let mut h = hd as i32;
        let mut tt = t as i32;
        let mut args: Vec<*mut std::ffi::c_void> = vec![
            (&mut q_p) as *mut _ as *mut std::ffi::c_void,
            (&mut k_p) as *mut _ as *mut std::ffi::c_void,
            (&mut v_p) as *mut _ as *mut std::ffi::c_void,
            (&mut si_p) as *mut _ as *mut std::ffi::c_void,
            (&mut so_p) as *mut _ as *mut std::ffi::c_void,
            (&mut o_p) as *mut _ as *mut std::ffi::c_void,
            (&mut sc) as *mut _ as *mut std::ffi::c_void,
            (&mut nh) as *mut _ as *mut std::ffi::c_void,
            (&mut nk) as *mut _ as *mut std::ffi::c_void,
            (&mut h) as *mut _ as *mut std::ffi::c_void,
            (&mut tt) as *mut _ as *mut std::ffi::c_void,
        ];
        // 블록 16워프 = 16토큰(워프당 1헤드) — 목록만 순회하는 기본판.
        self.ctx.launch3(
            "q4_qsa_attn_sel",
            t.div_ceil(16) as u32,
            n_head as u32,
            1,
            512,
            &mut args,
        )?;
        let mut out = vec![0.0f32; t * n_head * hd];
        self.ctx.d2h(bytemuck::cast_slice_mut(&mut out), odev)?;
        Ok(out)
    }

    /// q4_qsa_attn_sel4 런치 본체 — 선택 목록(오름차순 위치)만 순회한다.
    #[allow(clippy::too_many_arguments)]
    pub fn qsa_attn_sel4_raw(
        &self,
        q: &[f32],
        ck: &[f32],
        cv: &[f32],
        sel_idx: &[u32],
        sel_off: &[u32],
        kq_scale: f32,
        n_head: usize,
        n_kv: usize,
        hd: usize,
        t: usize,
    ) -> Result<Vec<f32>, String> {
        let (qdev, kdev, vdev, sdev, odev, ofdev) = {
            let mut a = self.qs.lock().map_err(|e| e.to_string())?;
            let qdev = a.ensure(&self.ctx, q.len() * 4)?;
            // KV는 컨텍스트 전체를 미리 잡는다(엔진이 주입한 ctx_len). 종전에는
            // n_past가 늘 때마다 재할당해 매 스텝 주소가 바뀌었다(실측 48회/세션).
            let kv_floats = self.ctx_len.load(std::sync::atomic::Ordering::Relaxed)
                * n_kv.max(1) * hd.max(1);
            let mut b = self.ckv.lock().map_err(|e| e.to_string())?;
            let kdev = b.ensure(&self.ctx, ck.len().max(kv_floats) * 4)?;
            let mut c = self.cvv.lock().map_err(|e| e.to_string())?;
            let vdev = c.ensure(&self.ctx, cv.len().max(kv_floats) * 4)?;
            let mut d = self.msk.lock().map_err(|e| e.to_string())?;
            let sdev = d.ensure(&self.ctx, sel_idx.len().max(1) * 4)?;
            let mut e2 = self.soff.lock().map_err(|e| e.to_string())?;
            let ofdev = e2.ensure(&self.ctx, sel_off.len().max(1) * 4)?;
            let mut f2 = self.atn.lock().map_err(|e| e.to_string())?;
            let odev = f2.ensure(&self.ctx, t * n_head * hd * 4)?;
            (qdev, kdev, vdev, sdev, odev, ofdev)
        };
        self.ctx.h2d(qdev, bytemuck::cast_slice(q))?;
        self.ctx.h2d(kdev, bytemuck::cast_slice(ck))?;
        self.ctx.h2d(vdev, bytemuck::cast_slice(cv))?;
        self.ctx.h2d(sdev, bytemuck::cast_slice(sel_idx))?;
        self.ctx.h2d(ofdev, bytemuck::cast_slice(sel_off))?;
        let mut q_p = qdev as *mut std::ffi::c_void;
        let mut k_p = kdev as *mut std::ffi::c_void;
        let mut v_p = vdev as *mut std::ffi::c_void;
        let mut si_p = sdev as *mut std::ffi::c_void;
        let mut so_p = ofdev as *mut std::ffi::c_void;
        let mut o_p = odev as *mut std::ffi::c_void;
        let mut sc = kq_scale;
        let mut nh = n_head as i32;
        let mut nk = n_kv as i32;
        let mut h = hd as i32;
        let mut tt = t as i32;
        let mut args: Vec<*mut std::ffi::c_void> = vec![
            (&mut q_p) as *mut _ as *mut std::ffi::c_void,
            (&mut k_p) as *mut _ as *mut std::ffi::c_void,
            (&mut v_p) as *mut _ as *mut std::ffi::c_void,
            (&mut si_p) as *mut _ as *mut std::ffi::c_void,
            (&mut so_p) as *mut _ as *mut std::ffi::c_void,
            (&mut o_p) as *mut _ as *mut std::ffi::c_void,
            (&mut sc) as *mut _ as *mut std::ffi::c_void,
            (&mut nh) as *mut _ as *mut std::ffi::c_void,
            (&mut nk) as *mut _ as *mut std::ffi::c_void,
            (&mut h) as *mut _ as *mut std::ffi::c_void,
            (&mut tt) as *mut _ as *mut std::ffi::c_void,
        ];
        // 8워프 = 4토큰 × 2헤드묶음. 묶음당 헤드 수는 6이 기본(2026-09-14):
        // 게이트를 레지스터에서 빼면 qr[6][8]+acc[6][8]=96으로 4헤드판과 같은
        // 예산이라 K/V 행 재독이 6회 -> 4회로 준다(프리필 어텐션이 대역폭 바운드:
        // t=2048 콜당 ~34GB/236GB/s ~= 실측 101ms). 12의 배수가 아니면 4헤드판.
        let use6 = n_head.is_multiple_of(12) && std::env::var("LLM170_QSA_H6").as_deref() != Ok("0");
        let (kern, gy) = if use6 {
            ("q4_qsa_attn_sel6", (n_head / 12) as u32)
        } else {
            ("q4_qsa_attn_sel4", (n_head / 8) as u32)
        };
        self.ctx.launch3(kern, t.div_ceil(4) as u32, gy, 1, 256, &mut args)?;
        let mut out = vec![0.0f32; t * n_head * hd];
        self.ctx.d2h(bytemuck::cast_slice_mut(&mut out), odev)?;
        Ok(out)
    }

    /// plans/67 1단계: **디바이스 q판** — q가 wq의 frame_mm_group 출력(디바이스)에
    /// 이미 있을 때 h2d 없이 어텐션을 돌고 결과를 디바이스 out에 쓴다(d2h도 없음).
    /// k/v는 기존 풀 업로드 경로(실측: KV 업로드는 유의미한 비용이 아님).
    /// QSA KV 상주 풀 — ctx_len 전체를 선할당(주소 안정성: ensure 재할당이
    /// 어텐션 커널에 전달된 포인터를 무효화하지 않게 1회 확정). k/v 행은
    /// D2D로 append(왕복 0). 반환 핸들 = 풀 포인터(디바이스 주소).
    fn qsa_kv_dev_impl(
        &self,
        full_idx: usize,
        seq: usize,
        k: u64,
        v: u64,
        t: usize,
        pos0: usize,
        n_kv: usize,
        hd: usize,
    ) -> Result<(u64, u64), String> {
        let ctx_len = self.ctx_len.load(std::sync::atomic::Ordering::Relaxed);
        if ctx_len == 0 {
            return Err("qsa_kv_dev: ctx_len 미주입".into());
        }
        let bytes = ctx_len * n_kv * hd * 4;
        // 워터마크 — 이 풀에 적립된 다음 위치. 규칙:
        //   pos0 == w: 정상 순차 적립.
        //   pos0 <  w: **되감기** — 위치 p의 k/v는 (토큰 접두어, p)의 결정 함수라
        //              접두어가 불변인 되감기(벤치 워밍업 후 재시작, 스펙 롤백,
        //              슬롯 재프리필)에서 [0, pos0)의 기존 값과 새 값이 동일하다.
        //              재구축도 순서대로 돌아 과거 청크가 이번 재구축분을 덮는다.
        //   pos0 >  w: 구멍(값 경로 청크 등) — 읽을 수 없으니 업로드 경로로 폴백.
        {
            let mut wm = self.qsa_kv_pos.lock().map_err(|e| e.to_string())?;
            let w = wm.entry((full_idx, seq)).or_insert(0);
            if pos0 > *w {
                return Err(format!(
                    "qsa_kv_dev: 워터마크 구멍 w={w} pos0={pos0} — 업로드 경로로 폴백"
                ));
            }
            *w = pos0 + t;
        }
        let mut m = self.qsa_kv.lock().map_err(|e| e.to_string())?;
        let ent = m
            .entry((full_idx, seq))
            .or_insert_with(|| (GBuf::new("qsakv_k"), GBuf::new("qsakv_v")));
        if ent.0.bytes < bytes {
            ent.0.ensure(&self.ctx, bytes)?;
            ent.1.ensure(&self.ctx, bytes)?;
        }
        let (kp, vp) = (ent.0.ptr, ent.1.ptr);
        let rows = t * n_kv * hd * 4;
        let ksrc = self.fptr(k)?;
        let vsrc = self.fptr(v)?;
        self.ctx
            .d2d(unsafe { kp.add(pos0 * n_kv * hd * 4) }, ksrc, rows)?;
        self.ctx
            .d2d(unsafe { vp.add(pos0 * n_kv * hd * 4) }, vsrc, rows)?;
        Ok((kp as u64, vp as u64))
    }

    /// plans/73 공용: ik를 idx 풀에 적립하고 완성 블록의 블록키를 증분 갱신한다.
    /// 소스가 디바이스(디코드, f.qsa_ik)면 d2d, 호스트(프리필 청크)면 h2d.
    /// 워터마크 규약은 qsa_kv_dev_impl과 동일(순차 적립/접두어 되감기 허용).
    #[allow(clippy::too_many_arguments)]
    fn qsa_idx_append(
        &self,
        full_idx: usize,
        seq: usize,
        ik_dev: *const u8,
        ik_host: &[f32],
        t: usize,
        pos0: usize,
        idx_dim: usize,
        r: usize,
        ikw: &[f32],
        cs_idx: &[f32],
        eps: f32,
    ) -> Result<(*mut u8, *mut u8), String> {
        let ctx_len = self.ctx_len.load(std::sync::atomic::Ordering::Relaxed);
        if ctx_len == 0 {
            return Err("qsa_idx_append: ctx_len 미주입".into());
        }
        if r == 0 || idx_dim != 128 {
            return Err(format!("qsa_idx_append: 미지원 형상 r={r} idx_dim={idx_dim}"));
        }
        {
            let mut wm = self.qsa_idx_pos.lock().map_err(|e| e.to_string())?;
            let w = wm.entry((full_idx, seq)).or_insert(0);
            if pos0 > *w {
                return Err(format!("qsa_idx_append: 워터마크 구멍 w={w} pos0={pos0}"));
            }
            *w = pos0 + t;
        }
        let nb_max = ctx_len / r + 1;
        let (idxk_p, bk_p) = {
            let mut m = self.qsa_idxk.lock().map_err(|e| e.to_string())?;
            let idxk = m
                .entry((full_idx, seq))
                .or_insert_with(|| GBuf::new("qsa_idxk"));
            idxk.ensure(&self.ctx, ctx_len * idx_dim * 4)?;
            let mut b = self.qsa_bk.lock().map_err(|e| e.to_string())?;
            let bk = b
                .entry((full_idx, seq))
                .or_insert_with(|| GBuf::new("qsa_bk"));
            bk.ensure(&self.ctx, nb_max * idx_dim * 4)?;
            (idxk.ptr, bk.ptr)
        };
        if !ik_dev.is_null() {
            self.ctx.d2d(
                unsafe { idxk_p.add(pos0 * idx_dim * 4) },
                ik_dev,
                t * idx_dim * 4,
            )?;
        } else {
            // hk: h2d는 동기 API — 프리필 청크(드문 경로)라 비용 무의미.
            self.ctx.h2d(
                unsafe { idxk_p.add(pos0 * idx_dim * 4) },
                bytemuck::cast_slice(&ik_host[..t * idx_dim]),
            )?;
        }
        let b0 = pos0 / r;
        let b1 = (pos0 + t) / r;
        if b1 > b0 {
            let ikw_d = self.upload_hashed(&self.qsa_ikw, ikw)?;
            let cs_d = self.upload_by_ptr(&self.qsa_csidx, cs_idx)?;
            let (mut ikp, mut bkp, mut iw, mut cp) = (
                idxk_p as *mut std::ffi::c_void,
                bk_p as *mut std::ffi::c_void,
                ikw_d as *mut std::ffi::c_void,
                cs_d as *mut std::ffi::c_void,
            );
            let (mut e, mut bb0, mut rr, mut dd) =
                (eps, b0 as i32, r as i32, idx_dim as i32);
            let mut args: Vec<*mut std::ffi::c_void> = vec![
                (&mut ikp) as *mut _ as *mut std::ffi::c_void,
                (&mut bkp) as *mut _ as *mut std::ffi::c_void,
                (&mut iw) as *mut _ as *mut std::ffi::c_void,
                (&mut cp) as *mut _ as *mut std::ffi::c_void,
                (&mut e) as *mut _ as *mut std::ffi::c_void,
                (&mut bb0) as *mut _ as *mut std::ffi::c_void,
                (&mut rr) as *mut _ as *mut std::ffi::c_void,
                (&mut dd) as *mut _ as *mut std::ffi::c_void,
            ];
            self.ctx
                .launch3("q4_idx_bk_update", (b1 - b0) as u32, 1, 1, 32, &mut args)?;
        }
        Ok((idxk_p, bk_p))
    }

    /// 소형 상수 업로드 캐시 — 내용 FNV 해시가 같으면 재업로드 생략(매 스텝
    /// h2d+sync를 낳던 qn/kn/iqw/ikw 류 제거). 반환 = 디바이스 포인터.
    pub(super) fn upload_hashed(&self, slot: &std::sync::Mutex<(u64, GBuf)>, data: &[f32]) -> Result<*mut u8, String> {
        let h = fnv_hash(data);
        let mut g = slot.lock().map_err(|e| e.to_string())?;
        if g.0 != h || g.1.ptr.is_null() {
            g.1.ensure(&self.ctx, data.len().max(1) * 4)?;
            self.ctx.h2d(g.1.ptr, bytemuck::cast_slice(data))?;
            g.0 = h;
        }
        Ok(g.1.ptr)
    }

    /// (ptr,len) 키 다중 엔트리 업로드 캐시 — 층별로 다른 상수를 상주시킨다.
    /// 단일 슬롯이면 층마다 미스해 매층 동기 h2d가 발생한다(실측 3.4ms/층).
    pub(super) fn upload_map(
        &self,
        map: &std::sync::Mutex<std::collections::HashMap<(u64, usize), GBuf>>,
        name: &'static str,
        data: &[f32],
    ) -> Result<*mut u8, String> {
        // 2026-09-16 RCA(값 드리프트): 호출부(qsa_frame*)가 **층마다 재할당되는
        // 로컬 Vec** 를 건넨다 — 할당기가 같은 주소를 재사용하면 (ptr,len) 키가
        // 이전 층 버퍼에 히트해 **내용이 다른 노름 가중치를 재업로드 없이 재사용**
        // 한다. 서버 스레드 타이밍에 따라 발동 → 로짓 ~0.1-0.5 드리프트(근접 타이
        // 플립, FN 게이트 1692↔24902)의 원인. 키를 **내용 FNV 해시**로 바꾼다
        // (24KB 해싱 ~2us — 절감 40ms 대 무의미).
        let bytes = bytemuck::cast_slice::<f32, u8>(data);
        let mut h: u64 = 0xcbf29ce484222325;
        for &b in bytes {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
        let key = (h, data.len());
        let mut m = map.lock().map_err(|e| e.to_string())?;
        if let Some(b) = m.get(&key) {
            return Ok(b.ptr);
        }
        // 상한: 층 수 × 소수 항목이면 충분하다 — 넘치면 비운다(재업로드 비용 < 무한 증가).
        if m.len() >= 64 {
            m.clear();
        }
        let mut b = GBuf::new(name);
        b.ensure(&self.ctx, data.len().max(1) * 4)?;
        self.ctx.h2d(b.ptr, bytemuck::cast_slice(data))?;
        let p = b.ptr;
        m.insert(key, b);
        Ok(p)
    }

    /// 대형 상수(cs 테이블) 업로드 캐시 — (ptr, len) 키. 프레임 필드 벡터는
    /// 스텝 사이 포인터가 안정적이라 해시(4MB)보다 저렴하다.
    fn upload_by_ptr(
        &self,
        slot: &std::sync::Mutex<(usize, usize, GBuf)>,
        data: &[f32],
    ) -> Result<*mut u8, String> {
        let key = (data.as_ptr() as usize, data.len());
        let mut g = slot.lock().map_err(|e| e.to_string())?;
        if g.0 != key.0 || g.1 != key.1 || g.2.ptr.is_null() {
            g.2.ensure(&self.ctx, data.len().max(1) * 4)?;
            self.ctx.h2d(g.2.ptr, bytemuck::cast_slice(data))?;
            (g.0, g.1) = (key.0, key.1);
        }
        Ok(g.2.ptr)
    }

    /// 상주 캐시판 어텐션 — ck/cv가 디바이스 주소(업로드 없음). 커널 선택은
    /// qsa_attention_dev와 동일(t=1 분할 우선).
    fn qsa_attn_res(
        &self,
        q: u64,
        ckp: u64,
        cvp: u64,
        sel_idx: &[u32],
        sel_off: &[u32],
        kq_scale: f32,
        n_head: usize,
        n_kv: usize,
        hd: usize,
        t: usize,
        out: u64,
    ) -> Result<(), String> {
        let use_split = t == 1 && std::env::var("LLM170_QSA_SPLIT").as_deref() != Ok("0");
        let (sdev, ofdev, pdev) = {
            let mut d = self.msk.lock().map_err(|e| e.to_string())?;
            let sdev = d.ensure(&self.ctx, sel_idx.len().max(1) * 4)?;
            let mut e2 = self.soff.lock().map_err(|e| e.to_string())?;
            let ofdev = e2.ensure(&self.ctx, sel_off.len().max(1) * 4)?;
            let n_splits = if use_split {
                let list_len = sel_off
                    .get(1)
                    .copied()
                    .unwrap_or(0)
                    .saturating_sub(sel_off.first().copied().unwrap_or(0)) as usize;
                let cap = std::env::var("LLM170_QSA_SPLITS")
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(64);
                (list_len / 32).clamp(1, cap.max(1).min(512))
            } else {
                1
            };
            let mut g = self.qsp.lock().map_err(|e| e.to_string())?;
            let pdev = g.ensure(&self.ctx, n_head * n_splits * 32 * 10 * 4)?;
            (sdev, ofdev, pdev)
        };
        self.ctx.h2d(sdev, bytemuck::cast_slice(sel_idx))?;
        self.ctx.h2d(ofdev, bytemuck::cast_slice(sel_off))?;
        let qdev = self.fptr(q)?;
        let odev = self.fptr(out)?;
        if use_split {
            let list_len = sel_off
                .get(1)
                .copied()
                .unwrap_or(0)
                .saturating_sub(sel_off.first().copied().unwrap_or(0)) as usize;
            let cap = std::env::var("LLM170_QSA_SPLITS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(64);
            let n_splits: usize = (list_len / 32).clamp(1, cap.max(1).min(512));
            let mut q_p = qdev as *mut std::ffi::c_void;
            let mut k_p = ckp as *mut std::ffi::c_void;
            let mut v_p = cvp as *mut std::ffi::c_void;
            let mut si_p = sdev as *mut std::ffi::c_void;
            let mut so_p = ofdev as *mut std::ffi::c_void;
            let mut pa_p = pdev as *mut std::ffi::c_void;
            let mut ns = n_splits as i32;
            let mut sc = kq_scale;
            let mut nh = n_head as i32;
            let mut nk = n_kv as i32;
            let mut h = hd as i32;
            let mut args: Vec<*mut std::ffi::c_void> = vec![
                (&mut q_p) as *mut _ as *mut std::ffi::c_void,
                (&mut k_p) as *mut _ as *mut std::ffi::c_void,
                (&mut v_p) as *mut _ as *mut std::ffi::c_void,
                (&mut si_p) as *mut _ as *mut std::ffi::c_void,
                (&mut so_p) as *mut _ as *mut std::ffi::c_void,
                (&mut pa_p) as *mut _ as *mut std::ffi::c_void,
                (&mut ns) as *mut _ as *mut std::ffi::c_void,
                (&mut sc) as *mut _ as *mut std::ffi::c_void,
                (&mut nh) as *mut _ as *mut std::ffi::c_void,
                (&mut nk) as *mut _ as *mut std::ffi::c_void,
                (&mut h) as *mut _ as *mut std::ffi::c_void,
            ];
            self.ctx.launch3(
                "q4_qsa_attn_sel4s",
                n_splits.div_ceil(4) as u32,
                (n_head / 12) as u32,
                1,
                256,
                &mut args,
            )?;
            let mut pa_p = pdev as *mut std::ffi::c_void;
            let mut q_p = qdev as *mut std::ffi::c_void;
            let mut o_p = odev as *mut std::ffi::c_void;
            let mut ns = n_splits as i32;
            let mut nh = n_head as i32;
            let mut h = hd as i32;
            let mut args: Vec<*mut std::ffi::c_void> = vec![
                (&mut pa_p) as *mut _ as *mut std::ffi::c_void,
                (&mut q_p) as *mut _ as *mut std::ffi::c_void,
                (&mut o_p) as *mut _ as *mut std::ffi::c_void,
                (&mut ns) as *mut _ as *mut std::ffi::c_void,
                (&mut nh) as *mut _ as *mut std::ffi::c_void,
                (&mut h) as *mut _ as *mut std::ffi::c_void,
            ];
            self.ctx.launch3(
                "q4_qsa_attn_sel4s_merge",
                n_head.div_ceil(8) as u32,
                1,
                1,
                256,
                &mut args,
            )?;
            return Ok(());
        }
        // 비분할 — t>3은 sel4(K/V 4헤드 공유), t≤3은 sel. 상동 사유.
        // use6(12헤드 6분할)는 비활성 — 위 사유로 sel4/sel 판을 쓴다.
        let use6 = false;
        let _ = n_head;
        let mut q_p = qdev as *mut std::ffi::c_void;
        let mut o_p = odev as *mut std::ffi::c_void;
        let mut k_p = ckp as *mut std::ffi::c_void;
        let mut v_p = cvp as *mut std::ffi::c_void;
        let mut si_p = sdev as *mut std::ffi::c_void;
        let mut so_p = ofdev as *mut std::ffi::c_void;
        let mut sc = kq_scale;
        let mut nh = n_head as i32;
        let mut nk = n_kv as i32;
        let mut h = hd as i32;
        let mut tt = t as i32;
        let mut args: Vec<*mut std::ffi::c_void> = vec![
            (&mut q_p) as *mut _ as *mut std::ffi::c_void,
            (&mut k_p) as *mut _ as *mut std::ffi::c_void,
            (&mut v_p) as *mut _ as *mut std::ffi::c_void,
            (&mut si_p) as *mut _ as *mut std::ffi::c_void,
            (&mut so_p) as *mut _ as *mut std::ffi::c_void,
            (&mut o_p) as *mut _ as *mut std::ffi::c_void,
            (&mut sc) as *mut _ as *mut std::ffi::c_void,
            (&mut nh) as *mut _ as *mut std::ffi::c_void,
            (&mut nk) as *mut _ as *mut std::ffi::c_void,
            (&mut h) as *mut _ as *mut std::ffi::c_void,
            (&mut tt) as *mut _ as *mut std::ffi::c_void,
        ];
        let (kern, gy, blk) = if use6 {
            ("q4_qsa_attn_sel6", (n_head / 12) as u32, 256u32)
        } else {
            ("q4_qsa_attn_sel4", (n_head / 8) as u32, 256u32)
        };
        let gx = t.div_ceil(4) as u32;
        self.ctx.launch3(kern, gx, gy, 1, blk, &mut args)?;
        Ok(())
    }

    /// 산술은 `qsa_attn_sel6_raw`와 동일(같은 커널) → 비트 동일 기대.
    #[allow(clippy::too_many_arguments)]
    pub fn qsa_attn_dev_raw(
        &self,
        q: u64,
        ck: &[f32],
        cv: &[f32],
        sel_idx: &[u32],
        sel_off: &[u32],
        kq_scale: f32,
        n_head: usize,
        n_kv: usize,
        hd: usize,
        t: usize,
        out: u64,
    ) -> Result<(), String> {
        let (qdev, kdev, vdev, sdev, ofdev, _odev) = {
            let mut a = self.qs.lock().map_err(|e| e.to_string())?;
            let qdev = a.ensure(&self.ctx, t.max(1) * n_head * 2 * hd * 4)?;
            let mut b = self.ckv.lock().map_err(|e| e.to_string())?;
            let kdev = b.ensure(&self.ctx, ck.len().max(1) * 4)?;
            let mut c = self.cvv.lock().map_err(|e| e.to_string())?;
            let vdev = c.ensure(&self.ctx, cv.len().max(1) * 4)?;
            let mut d = self.msk.lock().map_err(|e| e.to_string())?;
            let sdev = d.ensure(&self.ctx, sel_idx.len().max(1) * 4)?;
            let mut e2 = self.soff.lock().map_err(|e| e.to_string())?;
            let ofdev = e2.ensure(&self.ctx, sel_off.len().max(1) * 4)?;
            let mut f2 = self.atn.lock().map_err(|e| e.to_string())?;
            let odev = f2.ensure(&self.ctx, t * n_head * hd * 4)?;
            (qdev, kdev, vdev, sdev, ofdev, odev)
        };
        let _ = qdev; // q는 인자로 받은 디바이스 버퍼를 그대로 쓴다(업로드 없음).
        self.ctx.h2d(kdev, bytemuck::cast_slice(ck))?;
        self.ctx.h2d(vdev, bytemuck::cast_slice(cv))?;
        self.ctx.h2d(sdev, bytemuck::cast_slice(sel_idx))?;
        self.ctx.h2d(ofdev, bytemuck::cast_slice(sel_off))?;
        let mut q_p = self.fptr(q)? as *mut std::ffi::c_void;
        let mut o_p = self.fptr(out)? as *mut std::ffi::c_void;
        let mut k_p = kdev as *mut std::ffi::c_void;
        let mut v_p = vdev as *mut std::ffi::c_void;
        let mut si_p = sdev as *mut std::ffi::c_void;
        let mut so_p = ofdev as *mut std::ffi::c_void;
        let mut sc = kq_scale;
        let mut nh = n_head as i32;
        let mut nk = n_kv as i32;
        let mut h = hd as i32;
        let mut tt = t as i32;
        let mut args: Vec<*mut std::ffi::c_void> = vec![
            (&mut q_p) as *mut _ as *mut std::ffi::c_void,
            (&mut k_p) as *mut _ as *mut std::ffi::c_void,
            (&mut v_p) as *mut _ as *mut std::ffi::c_void,
            (&mut si_p) as *mut _ as *mut std::ffi::c_void,
            (&mut so_p) as *mut _ as *mut std::ffi::c_void,
            (&mut o_p) as *mut _ as *mut std::ffi::c_void,
            (&mut sc) as *mut _ as *mut std::ffi::c_void,
            (&mut nh) as *mut _ as *mut std::ffi::c_void,
            (&mut nk) as *mut _ as *mut std::ffi::c_void,
            (&mut h) as *mut _ as *mut std::ffi::c_void,
            (&mut tt) as *mut _ as *mut std::ffi::c_void,
        ];
        // 프리필(t>3)은 sel4 — K/V를 4헤드가 공유(트래픽 1/4, 호스트 경로와
        // 동일 선택). sel6(6헤드)은 K/V를 묶음마다 재독해 t=2048 실측 40ms/런치
        // 까지 올라갔다(2026-09-14 KTRACE, 48런치 1.92s) — 프리필 회귀였음.
        // t≤3도 호스트 규약대로 sel(헤드당 워프)을 쓴다.
        let _ = n_head % 12;
        let (kern, gy, blk) = if t > 3 {
            ("q4_qsa_attn_sel4", (n_head / 8) as u32, 256u32)
        } else {
            ("q4_qsa_attn_sel", (n_head / 4) as u32, 128u32)
        };
        if t > 3 {
            let gx = t.div_ceil(4) as u32;
            self.ctx.launch3(kern, gx, gy, 1, blk, &mut args)?;
        } else {
            // _sel 원본 규격: 블록 16워프=16토큰(워프당 1헤드), gy=n_head.
            self.ctx
                .launch3("q4_qsa_attn_sel", t.div_ceil(16) as u32, n_head as u32, 1, 512, &mut args)?;
        }
        Ok(())
    }

    /// t=1 위치 분할판 — 선택목록을 n_splits로 쪼개 (split, 헤드묶음) 그리드로
    /// 펼친다. `_sel4`는 워프가 목록 전체를 직렬 순회해 t=1에서 지연 바운드다
    /// (실측 1.425ms/콜). 부분 (m,l,acc)를 남기고 2차 커널이 flash 규약으로
    /// 병합한다 — 합산 순서가 분할 경계에서 달라 비트 동일은 아니고 greedy
    /// 스트림 동일성으로 검증한다. LLM170_QSA_SPLITS로 분할 수(기본 64).
    #[allow(clippy::too_many_arguments)]
    pub fn qsa_attn_sel4s_raw(
        &self,
        q: &[f32],
        ck: &[f32],
        cv: &[f32],
        sel_idx: &[u32],
        sel_off: &[u32],
        kq_scale: f32,
        n_head: usize,
        n_kv: usize,
        hd: usize,
        t: usize,
    ) -> Result<Vec<f32>, String> {
        // 분할 수는 목록 길이에 맞춘다 — 짧은 문맥에서는 분할 이득이 없고
        // 부분 버퍼 쓰기·병합 비용만 늘어난다(분할당 최소 32위치).
        let list_len = sel_off
            .get(1)
            .copied()
            .unwrap_or(0)
            .saturating_sub(sel_off.first().copied().unwrap_or(0)) as usize;
        let cap = std::env::var("LLM170_QSA_SPLITS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(64);
        let n_splits: usize = (list_len / 32).clamp(1, cap.max(1).min(512));
        let (qdev, kdev, vdev, sdev, ofdev, pdev, odev) = {
            let mut a = self.qs.lock().map_err(|e| e.to_string())?;
            let qdev = a.ensure(&self.ctx, q.len().max(1) * 4)?;
            // KV는 컨텍스트 전체를 미리 잡는다(엔진이 주입한 ctx_len). 종전에는
            // n_past가 늘 때마다 재할당해 매 스텝 주소가 바뀌었다(실측 48회/세션).
            let kv_floats = self.ctx_len.load(std::sync::atomic::Ordering::Relaxed)
                * n_kv.max(1) * hd.max(1);
            let mut b = self.ckv.lock().map_err(|e| e.to_string())?;
            let kdev = b.ensure(&self.ctx, ck.len().max(kv_floats) * 4)?;
            let mut c = self.cvv.lock().map_err(|e| e.to_string())?;
            let vdev = c.ensure(&self.ctx, cv.len().max(kv_floats) * 4)?;
            let mut d = self.msk.lock().map_err(|e| e.to_string())?;
            let sdev = d.ensure(&self.ctx, sel_idx.len().max(1) * 4)?;
            let mut e2 = self.soff.lock().map_err(|e| e.to_string())?;
            let ofdev = e2.ensure(&self.ctx, sel_off.len().max(1) * 4)?;
            let mut g = self.qsp.lock().map_err(|e| e.to_string())?;
            let pdev = g.ensure(&self.ctx, n_head * n_splits * 32 * 10 * 4)?;
            let mut f2 = self.atn.lock().map_err(|e| e.to_string())?;
            let odev = f2.ensure(&self.ctx, t * n_head * hd * 4)?;
            (qdev, kdev, vdev, sdev, ofdev, pdev, odev)
        };
        self.ctx.h2d(qdev, bytemuck::cast_slice(q))?;
        self.ctx.h2d(kdev, bytemuck::cast_slice(ck))?;
        self.ctx.h2d(vdev, bytemuck::cast_slice(cv))?;
        self.ctx.h2d(sdev, bytemuck::cast_slice(sel_idx))?;
        self.ctx.h2d(ofdev, bytemuck::cast_slice(sel_off))?;
        {
            let mut q_p = qdev as *mut std::ffi::c_void;
            let mut k_p = kdev as *mut std::ffi::c_void;
            let mut v_p = vdev as *mut std::ffi::c_void;
            let mut si_p = sdev as *mut std::ffi::c_void;
            let mut so_p = ofdev as *mut std::ffi::c_void;
            let mut pa_p = pdev as *mut std::ffi::c_void;
            let mut ns = n_splits as i32;
            let mut sc = kq_scale;
            let mut nh = n_head as i32;
            let mut nk = n_kv as i32;
            let mut h = hd as i32;
            let mut args: Vec<*mut std::ffi::c_void> = vec![
                (&mut q_p) as *mut _ as *mut std::ffi::c_void,
                (&mut k_p) as *mut _ as *mut std::ffi::c_void,
                (&mut v_p) as *mut _ as *mut std::ffi::c_void,
                (&mut si_p) as *mut _ as *mut std::ffi::c_void,
                (&mut so_p) as *mut _ as *mut std::ffi::c_void,
                (&mut pa_p) as *mut _ as *mut std::ffi::c_void,
                (&mut ns) as *mut _ as *mut std::ffi::c_void,
                (&mut sc) as *mut _ as *mut std::ffi::c_void,
                (&mut nh) as *mut _ as *mut std::ffi::c_void,
                (&mut nk) as *mut _ as *mut std::ffi::c_void,
                (&mut h) as *mut _ as *mut std::ffi::c_void,
            ];
            // 묶음당 6헤드(블록당 2묶음) — 분할판도 같은 재독 절감.
            self.ctx.launch3(
                "q4_qsa_attn_sel4s",
                n_splits.div_ceil(4) as u32,
                (n_head / 12) as u32,
                1,
                256,
                &mut args,
            )?;
        }
        {
            let mut pa_p = pdev as *mut std::ffi::c_void;
            let mut q_p = qdev as *mut std::ffi::c_void;
            let mut o_p = odev as *mut std::ffi::c_void;
            let mut ns = n_splits as i32;
            let mut nh = n_head as i32;
            let mut h = hd as i32;
            let mut args: Vec<*mut std::ffi::c_void> = vec![
                (&mut pa_p) as *mut _ as *mut std::ffi::c_void,
                (&mut q_p) as *mut _ as *mut std::ffi::c_void,
                (&mut o_p) as *mut _ as *mut std::ffi::c_void,
                (&mut ns) as *mut _ as *mut std::ffi::c_void,
                (&mut nh) as *mut _ as *mut std::ffi::c_void,
                (&mut h) as *mut _ as *mut std::ffi::c_void,
            ];
            self.ctx.launch3(
                "q4_qsa_attn_sel4s_merge",
                n_head.div_ceil(8) as u32,
                1,
                1,
                256,
                &mut args,
            )?;
        }
        let mut out = vec![0.0f32; t * n_head * hd];
        self.ctx.d2h(bytemuck::cast_slice_mut(&mut out), odev)?;
        Ok(out)
    }

    /// 분할판의 **디바이스 q·출력판** (plans/67 2c) — q를 wq 출력 버퍼에서 직접
    /// 읽고 어텐션 출력도 프레임 버퍼에 쓴다(왕복 0). 산술은 sel4s와 동일
    /// 커널 쌍이라 greedy 스트림 동일. t=1 장문맥 디코드의 지연 바운드를
    /// 분할로 푼다(142.5→124.4ms/스텝 실측치의 디바이스 상속).
    #[allow(clippy::too_many_arguments)]
    pub fn qsa_attn_sel4s_dev_raw(
        &self,
        q: u64,
        ck: &[f32],
        cv: &[f32],
        sel_idx: &[u32],
        sel_off: &[u32],
        kq_scale: f32,
        n_head: usize,
        n_kv: usize,
        hd: usize,
        t: usize,
        out: u64,
    ) -> Result<(), String> {
        let _ = t; // t==1 규약(호출부가 보장) — 커널은 sel_off로 범위를 안다
        let list_len = sel_off
            .get(1)
            .copied()
            .unwrap_or(0)
            .saturating_sub(sel_off.first().copied().unwrap_or(0)) as usize;
        let cap = std::env::var("LLM170_QSA_SPLITS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(64);
        let n_splits: usize = (list_len / 32).clamp(1, cap.max(1).min(512));
        let (kdev, vdev, sdev, ofdev, pdev) = {
            let kv_floats = self.ctx_len.load(std::sync::atomic::Ordering::Relaxed)
                * n_kv.max(1) * hd.max(1);
            let mut b = self.ckv.lock().map_err(|e| e.to_string())?;
            let kdev = b.ensure(&self.ctx, ck.len().max(kv_floats) * 4)?;
            let mut c = self.cvv.lock().map_err(|e| e.to_string())?;
            let vdev = c.ensure(&self.ctx, cv.len().max(kv_floats) * 4)?;
            let mut d = self.msk.lock().map_err(|e| e.to_string())?;
            let sdev = d.ensure(&self.ctx, sel_idx.len().max(1) * 4)?;
            let mut e2 = self.soff.lock().map_err(|e| e.to_string())?;
            let ofdev = e2.ensure(&self.ctx, sel_off.len().max(1) * 4)?;
            let mut g = self.qsp.lock().map_err(|e| e.to_string())?;
            let pdev = g.ensure(&self.ctx, n_head * n_splits * 32 * 10 * 4)?;
            (kdev, vdev, sdev, ofdev, pdev)
        };
        self.ctx.h2d(kdev, bytemuck::cast_slice(ck))?;
        self.ctx.h2d(vdev, bytemuck::cast_slice(cv))?;
        self.ctx.h2d(sdev, bytemuck::cast_slice(sel_idx))?;
        self.ctx.h2d(ofdev, bytemuck::cast_slice(sel_off))?;
        let qdev = self.fptr(q)?;
        let odev = self.fptr(out)?;
        {
            let mut q_p = qdev as *mut std::ffi::c_void;
            let mut k_p = kdev as *mut std::ffi::c_void;
            let mut v_p = vdev as *mut std::ffi::c_void;
            let mut si_p = sdev as *mut std::ffi::c_void;
            let mut so_p = ofdev as *mut std::ffi::c_void;
            let mut pa_p = pdev as *mut std::ffi::c_void;
            let mut ns = n_splits as i32;
            let mut sc = kq_scale;
            let mut nh = n_head as i32;
            let mut nk = n_kv as i32;
            let mut h = hd as i32;
            let mut args: Vec<*mut std::ffi::c_void> = vec![
                (&mut q_p) as *mut _ as *mut std::ffi::c_void,
                (&mut k_p) as *mut _ as *mut std::ffi::c_void,
                (&mut v_p) as *mut _ as *mut std::ffi::c_void,
                (&mut si_p) as *mut _ as *mut std::ffi::c_void,
                (&mut so_p) as *mut _ as *mut std::ffi::c_void,
                (&mut pa_p) as *mut _ as *mut std::ffi::c_void,
                (&mut ns) as *mut _ as *mut std::ffi::c_void,
                (&mut sc) as *mut _ as *mut std::ffi::c_void,
                (&mut nh) as *mut _ as *mut std::ffi::c_void,
                (&mut nk) as *mut _ as *mut std::ffi::c_void,
                (&mut h) as *mut _ as *mut std::ffi::c_void,
            ];
            self.ctx.launch3(
                "q4_qsa_attn_sel4s",
                n_splits.div_ceil(4) as u32,
                (n_head / 12) as u32,
                1,
                256,
                &mut args,
            )?;
        }
        {
            let mut pa_p = pdev as *mut std::ffi::c_void;
            let mut q_p = qdev as *mut std::ffi::c_void;
            let mut o_p = odev as *mut std::ffi::c_void;
            let mut ns = n_splits as i32;
            let mut nh = n_head as i32;
            let mut h = hd as i32;
            let mut args: Vec<*mut std::ffi::c_void> = vec![
                (&mut pa_p) as *mut _ as *mut std::ffi::c_void,
                (&mut q_p) as *mut _ as *mut std::ffi::c_void,
                (&mut o_p) as *mut _ as *mut std::ffi::c_void,
                (&mut ns) as *mut _ as *mut std::ffi::c_void,
                (&mut nh) as *mut _ as *mut std::ffi::c_void,
                (&mut h) as *mut _ as *mut std::ffi::c_void,
            ];
            self.ctx.launch3(
                "q4_qsa_attn_sel4s_merge",
                n_head.div_ceil(8) as u32,
                1,
                1,
                256,
                &mut args,
            )?;
        }
        Ok(())
    }

    /// q4_qsa_attn 커널 런치 본체 — 가드 없음(격리 프로브·진단 전용).
    #[allow(clippy::too_many_arguments)]
    pub fn qsa_attn_raw(
        &self,
        q: &[f32],
        ck: &[f32],
        cv: &[f32],
        mask: &[u32],
        kq_scale: f32,
        n_past: usize,
        n_head: usize,
        n_kv: usize,
        hd: usize,
        t: usize,
    ) -> Result<Vec<f32>, String> {
        let (qdev, kdev, vdev, mdev, odev) = {
            let mut a = self.qs.lock().map_err(|e| e.to_string())?;
            let qdev = a.ensure(&self.ctx, q.len() * 4)?;
            // KV는 컨텍스트 전체를 미리 잡는다(엔진이 주입한 ctx_len). 종전에는
            // n_past가 늘 때마다 재할당해 매 스텝 주소가 바뀌었다(실측 48회/세션).
            let kv_floats = self.ctx_len.load(std::sync::atomic::Ordering::Relaxed)
                * n_kv.max(1) * hd.max(1);
            let mut b = self.ckv.lock().map_err(|e| e.to_string())?;
            let kdev = b.ensure(&self.ctx, ck.len().max(kv_floats) * 4)?;
            let mut c = self.cvv.lock().map_err(|e| e.to_string())?;
            let vdev = c.ensure(&self.ctx, cv.len().max(kv_floats) * 4)?;
            let mut d = self.msk.lock().map_err(|e| e.to_string())?;
            let mdev = d.ensure(&self.ctx, mask.len() * 4)?;
            let mut e2 = self.atn.lock().map_err(|e| e.to_string())?;
            let odev = e2.ensure(&self.ctx, t * n_head * hd * 4)?;
            (qdev, kdev, vdev, mdev, odev)
        };
        self.ctx.h2d(qdev, bytemuck::cast_slice(q))?;
        self.ctx.h2d(kdev, bytemuck::cast_slice(ck))?;
        self.ctx.h2d(vdev, bytemuck::cast_slice(cv))?;
        self.ctx.h2d(mdev, bytemuck::cast_slice(mask))?;
        let mut q_p = qdev as *mut std::ffi::c_void;
        let mut k_p = kdev as *mut std::ffi::c_void;
        let mut v_p = vdev as *mut std::ffi::c_void;
        let mut m_p = mdev as *mut std::ffi::c_void;
        let mut o_p = odev as *mut std::ffi::c_void;
        let mut sc = kq_scale;
        let mut np_ = n_past as i32;
        let mut nh = n_head as i32;
        let mut nk = n_kv as i32;
        let mut h = hd as i32;
        let mut tt = t as i32;
        let mut args: Vec<*mut std::ffi::c_void> = vec![
            (&mut q_p) as *mut _ as *mut std::ffi::c_void,
            (&mut k_p) as *mut _ as *mut std::ffi::c_void,
            (&mut v_p) as *mut _ as *mut std::ffi::c_void,
            (&mut m_p) as *mut _ as *mut std::ffi::c_void,
            (&mut o_p) as *mut _ as *mut std::ffi::c_void,
            (&mut sc) as *mut _ as *mut std::ffi::c_void,
            (&mut np_) as *mut _ as *mut std::ffi::c_void,
            (&mut nh) as *mut _ as *mut std::ffi::c_void,
            (&mut nk) as *mut _ as *mut std::ffi::c_void,
            (&mut h) as *mut _ as *mut std::ffi::c_void,
            (&mut tt) as *mut _ as *mut std::ffi::c_void,
        ];
        // 워프-퍼-토큰 커널 — 블록 배리어 없음 + K를 8토큰이 공유(§41).
        // 미러 대조 2.263e-4(구 커널과 동일), 토큰 동일, 프리필 −1.9%@11.75k.
        self.ctx.launch3(
            "q4_qsa_attn_wt",
            t.div_ceil(16) as u32,
            n_head as u32,
            1,
            512,
            &mut args,
        )?;
        let mut out = vec![0.0f32; t * n_head * hd];
        self.ctx.d2h(bytemuck::cast_slice_mut(&mut out), odev)?;
        if std::env::var_os("LLM170_Q4_DBG").is_some() {
            let bad = out.iter().filter(|v| !v.is_finite()).count();
            let badq = q.iter().filter(|v| !v.is_finite()).count();
            eprintln!("# qsa_attn t={t} n_past={n_past}: out 비유한={bad}/{} q 비유한={badq}", out.len());
        }
        Ok(out)
    }
}

impl llm170_core::matmul::QsaOps for Q4Acc {

    fn qsa_attention_dev(
        &self,
        q: u64,
        ck: &[f32],
        cv: &[f32],
        sel_idx: &[u32],
        sel_off: &[u32],
        kq_scale: f32,
        n_head: usize,
        n_kv: usize,
        hd: usize,
        t: usize,
        out: u64,
    ) -> Result<(), String> {
        // t=1은 위치 분할판(flash-decoding형) — 디바이스 q·출력판이 같은 커널
        // 쌍을 쓴다. 규약은 호스트 판(qsa_attention_sel)과 동일: LLM170_QSA_SPLIT=0
        // 이면 비분할 sel6/sel4로 돌아간다.
        if t == 1 && std::env::var("LLM170_QSA_SPLIT").as_deref() != Ok("0") {
            self.qsa_attn_sel4s_dev_raw(q, ck, cv, sel_idx, sel_off, kq_scale, n_head, n_kv, hd, t, out)
        } else {
            self.qsa_attn_dev_raw(q, ck, cv, sel_idx, sel_off, kq_scale, n_head, n_kv, hd, t, out)
        }
    }

    fn qsa_kv_dev(
        &self,
        full_idx: usize,
        seq: usize,
        k: u64,
        v: u64,
        t: usize,
        pos0: usize,
        n_kv: usize,
        hd: usize,
    ) -> Result<(u64, u64), String> {
        self.qsa_kv_dev_impl(full_idx, seq, k, v, t, pos0, n_kv, hd)
    }

    fn qsa_kv_check(
        &self,
        full_idx: usize,
        seq: usize,
        host_ck: &[f32],
        host_cv: &[f32],
    ) -> Result<(), String> {
        let m = self.qsa_kv.lock().map_err(|e| e.to_string())?;
        let Some((kb, vb)) = m.get(&(full_idx, seq)) else {
            return Err("qsa_kv_check: 풀 없음".into());
        };
        let n = host_ck.len().min(kb.bytes / 4);
        let mut got = vec![0.0f32; n];
        self.ctx
            .d2h(bytemuck::cast_slice_mut(&mut got), kb.ptr as *const u8)
            .map_err(|e| e.to_string())?;
        for (i, (a, b)) in got.iter().zip(host_ck[..n].iter()).enumerate() {
            if a.to_bits() != b.to_bits() {
                return Err(format!(
                    "qsa_kv_check k 불일치 @float {i}: pool={a:e} host={b:e}"
                ));
            }
        }
        let n = host_cv.len().min(vb.bytes / 4);
        let mut got = vec![0.0f32; n];
        self.ctx
            .d2h(bytemuck::cast_slice_mut(&mut got), vb.ptr as *const u8)
            .map_err(|e| e.to_string())?;
        for (i, (a, b)) in got.iter().zip(host_cv[..n].iter()).enumerate() {
            if a.to_bits() != b.to_bits() {
                return Err(format!(
                    "qsa_kv_check v 불일치 @float {i}: pool={a:e} host={b:e}"
                ));
            }
        }
        Ok(())
    }

    fn qsa_attention_dev_res(
        &self,
        q: u64,
        ck: u64,
        cv: u64,
        sel_idx: &[u32],
        sel_off: &[u32],
        kq_scale: f32,
        n_head: usize,
        n_kv: usize,
        hd: usize,
        t: usize,
        out: u64,
    ) -> Result<(), String> {
        self.qsa_attn_res(q, ck, cv, sel_idx, sel_off, kq_scale, n_head, n_kv, hd, t, out)
    }

    fn qsa_sel_dev(
        &self,
        full_idx: usize,
        seq: usize,
        iq: u64,
        ik: u64,
        t: usize,
        pos0: usize,
        idx_heads: usize,
        idx_dim: usize,
        r: usize,
        idx_top_k: usize,
        iqw: &[f32],
        ikw: &[f32],
        cs_idx: &[f32],
        eps: f32,
    ) -> Result<(u64, u64, usize), String> {
        if t != 1 {
            return Err(format!("qsa_sel_dev: t={t} (디코드 전용)"));
        }
        if r == 0 {
            return Err("qsa_sel_dev: r=0".into());
        }
        let n_past = pos0 + t;
        let n_blocks = n_past / r;
        if n_blocks > 8192 {
            return Err(format!("qsa_sel_dev: n_blocks={n_blocks} > 8192 (expand shared)"));
        }
        let iqp = self.fptr(iq)?;
        let ikp = self.fptr(ik)?;
        let (_idxk_p, bk_p) =
            self.qsa_idx_append(full_idx, seq, ikp, &[], t, pos0, idx_dim, r, ikw, cs_idx, eps)?;
        // (1) iq norm+rope — iqr 스크래치.
        let iqr = {
            let mut g = self.qsa_iqr.lock().map_err(|e| e.to_string())?;
            g.ensure(&self.ctx, t * idx_heads * idx_dim * 4)?
        };
        let iqw_d = self.upload_hashed(&self.qsa_iqw, iqw)?;
        let cs_d = self.upload_by_ptr(&self.qsa_csidx, cs_idx)?;
        {
            let (mut qp, mut op, mut iw, mut cp) = (
                iqp as *mut std::ffi::c_void,
                iqr as *mut std::ffi::c_void,
                iqw_d as *mut std::ffi::c_void,
                cs_d as *mut std::ffi::c_void,
            );
            let (mut e, mut pp, mut tt, mut ih, mut dd) = (
                eps,
                pos0 as i32,
                t as i32,
                idx_heads as i32,
                idx_dim as i32,
            );
            let mut args: Vec<*mut std::ffi::c_void> = vec![
                (&mut qp) as *mut _ as *mut std::ffi::c_void,
                (&mut op) as *mut _ as *mut std::ffi::c_void,
                (&mut iw) as *mut _ as *mut std::ffi::c_void,
                (&mut cp) as *mut _ as *mut std::ffi::c_void,
                (&mut e) as *mut _ as *mut std::ffi::c_void,
                (&mut pp) as *mut _ as *mut std::ffi::c_void,
                (&mut tt) as *mut _ as *mut std::ffi::c_void,
                (&mut ih) as *mut _ as *mut std::ffi::c_void,
                (&mut dd) as *mut _ as *mut std::ffi::c_void,
            ];
            self.ctx
                .launch3("q4_idx_q_rope", idx_heads as u32, t as u32, 1, 32, &mut args)?;
        }
        // (2) 블록 점수 — 스레드당 블록.
        let scr = {
            let mut g = self.qsa_scr.lock().map_err(|e| e.to_string())?;
            g.ensure(&self.ctx, n_blocks.max(1) * 4)?
        };
        if n_blocks > 0 {
            let (mut qp, mut bp, mut sp) = (
                iqr as *mut std::ffi::c_void,
                bk_p as *mut std::ffi::c_void,
                scr as *mut std::ffi::c_void,
            );
            let (mut nb, mut ih, mut dd) = (n_blocks as i32, idx_heads as i32, idx_dim as i32);
            let mut args: Vec<*mut std::ffi::c_void> = vec![
                (&mut qp) as *mut _ as *mut std::ffi::c_void,
                (&mut bp) as *mut _ as *mut std::ffi::c_void,
                (&mut sp) as *mut _ as *mut std::ffi::c_void,
                (&mut nb) as *mut _ as *mut std::ffi::c_void,
                (&mut ih) as *mut _ as *mut std::ffi::c_void,
                (&mut dd) as *mut _ as *mut std::ffi::c_void,
            ];
            self.ctx.launch3(
                "q4_idx_score",
                (n_blocks as u32).div_ceil(256),
                1,
                1,
                256,
                &mut args,
            )?;
        }
        // (3) top-k 순위 + 목록 전개. n_sel 산술은 stages::qsa_select 패스 B와
        // 동일(usize 정수 — 호스트에서 계산해도 무동기).
        let tail_start = n_blocks * r;
        let tail_cnt = n_past - tail_start;
        let width = n_past.min(idx_top_k + r - 1);
        let n_sel = ((width - tail_cnt) / r).min(n_blocks);
        let list_len = n_sel * r + tail_cnt;
        let (sdev, ofdev) = {
            let mut d = self.msk.lock().map_err(|e| e.to_string())?;
            let sdev = d.ensure(&self.ctx, list_len.max(1) * 4)? as u64;
            let mut e2 = self.soff.lock().map_err(|e| e.to_string())?;
            let ofdev = e2.ensure(&self.ctx, 2 * 4)? as u64;
            (sdev, ofdev)
        };
        if n_blocks > 0 && n_blocks <= 4096 && std::env::var("LLM170_QSA_TOPK").as_deref() != Ok("0")
        {
            // 비토닉 단일 블록판 — rank+expand 콤보 대비 ~20×(0.228 → ~0.01ms).
            let (mut sp, mut si, mut so) = (
                scr as *mut std::ffi::c_void,
                sdev as *mut std::ffi::c_void,
                ofdev as *mut std::ffi::c_void,
            );
            let (mut nb, mut ns, mut rr, mut np) =
                (n_blocks as i32, n_sel as i32, r as i32, n_past as i32);
            let mut args: Vec<*mut std::ffi::c_void> = vec![
                (&mut sp) as *mut _ as *mut std::ffi::c_void,
                (&mut si) as *mut _ as *mut std::ffi::c_void,
                (&mut so) as *mut _ as *mut std::ffi::c_void,
                (&mut nb) as *mut _ as *mut std::ffi::c_void,
                (&mut ns) as *mut _ as *mut std::ffi::c_void,
                (&mut rr) as *mut _ as *mut std::ffi::c_void,
                (&mut np) as *mut _ as *mut std::ffi::c_void,
            ];
            self.ctx.launch3("q4_idx_topk", 1, 1, 1, 256, &mut args)?;
            return Ok((sdev, ofdev, list_len));
        }
        {
            let selflag = {
                let mut g = self.qsa_selflag.lock().map_err(|e| e.to_string())?;
                g.ensure(&self.ctx, n_blocks.max(1) * 4)?
            };
            if n_blocks > 0 {
                let (mut sp, mut fp) = (
                    scr as *mut std::ffi::c_void,
                    selflag as *mut std::ffi::c_void,
                );
                let (mut nb, mut ns) = (n_blocks as i32, n_sel as i32);
                let mut args: Vec<*mut std::ffi::c_void> = vec![
                    (&mut sp) as *mut _ as *mut std::ffi::c_void,
                    (&mut fp) as *mut _ as *mut std::ffi::c_void,
                    (&mut nb) as *mut _ as *mut std::ffi::c_void,
                    (&mut ns) as *mut _ as *mut std::ffi::c_void,
                ];
                self.ctx.launch3(
                    "q4_idx_rank",
                    (n_blocks as u32).div_ceil(256),
                    1,
                    1,
                    256,
                    &mut args,
                )?;
            }
            let (mut fp, mut si, mut so) = (
                selflag as *mut std::ffi::c_void,
                sdev as *mut std::ffi::c_void,
                ofdev as *mut std::ffi::c_void,
            );
            let (mut nb, mut ns, mut rr, mut np) =
                (n_blocks as i32, n_sel as i32, r as i32, n_past as i32);
            let mut args: Vec<*mut std::ffi::c_void> = vec![
                (&mut fp) as *mut _ as *mut std::ffi::c_void,
                (&mut si) as *mut _ as *mut std::ffi::c_void,
                (&mut so) as *mut _ as *mut std::ffi::c_void,
                (&mut nb) as *mut _ as *mut std::ffi::c_void,
                (&mut ns) as *mut _ as *mut std::ffi::c_void,
                (&mut rr) as *mut _ as *mut std::ffi::c_void,
                (&mut np) as *mut _ as *mut std::ffi::c_void,
            ];
            self.ctx
                .launch3("q4_idx_expand", 1, 1, 1, 256, &mut args)?;
        }
        Ok((sdev, ofdev, list_len))
    }

    fn qsa_idx_append_host(
        &self,
        full_idx: usize,
        seq: usize,
        ik_host: &[f32],
        t: usize,
        pos0: usize,
        idx_dim: usize,
        r: usize,
        ikw: &[f32],
        cs_idx: &[f32],
        eps: f32,
    ) -> Result<(), String> {
        self.qsa_idx_append(
            full_idx,
            seq,
            std::ptr::null(),
            ik_host,
            t,
            pos0,
            idx_dim,
            r,
            ikw,
            cs_idx,
            eps,
        )
        .map(|_| ())
    }

    /// 인덱서 k 행 디바이스 적립 — t>1 프리필 단축 경로(항등 선택) 전용.
    fn qsa_idx_append_dev(
        &self,
        full_idx: usize,
        seq: usize,
        ik: u64,
        t: usize,
        pos0: usize,
        idx_dim: usize,
        r: usize,
        ikw: &[f32],
        cs_idx: &[f32],
        eps: f32,
    ) -> Result<(), String> {
        let ikp = self.fptr(ik)? as *const u8;
        self.qsa_idx_append(full_idx, seq, ikp, &[], t, pos0, idx_dim, r, ikw, cs_idx, eps)
            .map(|_| ())
    }

    fn qsa_host_rebuild(
        &self,
        full_idx: usize,
        seq: usize,
        pos: usize,
        kv_row: usize,
        kv_k: &mut [f32],
        kv_v: &mut [f32],
        idx_k: &mut [f32],
        bk: &mut [f32],
        r: usize,
        idx_dim: usize,
    ) -> Result<(), String> {
        let kv_ok = {
            let wm = self.qsa_kv_pos.lock().map_err(|e| e.to_string())?;
            wm.get(&(full_idx, seq)).copied().unwrap_or(0) >= pos
        };
        let idx_ok = {
            let wm = self.qsa_idx_pos.lock().map_err(|e| e.to_string())?;
            wm.get(&(full_idx, seq)).copied().unwrap_or(0) >= pos
        };
        if !kv_ok || !idx_ok {
            return Err(format!(
                "qsa_host_rebuild: 풀 워터마크 부족 kv={kv_ok} idx={idx_ok} pos={pos}"
            ));
        }
        let nb = pos / r;
        {
            let m = self.qsa_kv.lock().map_err(|e| e.to_string())?;
            let ent = m
                .get(&(full_idx, seq))
                .ok_or("qsa_host_rebuild: kv 풀 없음")?;
            self.ctx.d2h(bytemuck::cast_slice_mut(&mut kv_k[..pos * kv_row]), ent.0.ptr)?;
            self.ctx.d2h(bytemuck::cast_slice_mut(&mut kv_v[..pos * kv_row]), ent.1.ptr)?;
        }
        {
            let m = self.qsa_idxk.lock().map_err(|e| e.to_string())?;
            let p = m
                .get(&(full_idx, seq))
                .ok_or("qsa_host_rebuild: idx 풀 없음")?
                .ptr;
            self.ctx
                .d2h(bytemuck::cast_slice_mut(&mut idx_k[..pos * idx_dim]), p)?;
        }
        {
            let m = self.qsa_bk.lock().map_err(|e| e.to_string())?;
            let p = m
                .get(&(full_idx, seq))
                .ok_or("qsa_host_rebuild: bk 풀 없음")?
                .ptr;
            self.ctx
                .d2h(bytemuck::cast_slice_mut(&mut bk[..nb * idx_dim]), p)?;
        }
        Ok(())
    }

    fn qsa_attention_dev_sel(
        &self,
        q: u64,
        ck: u64,
        cv: u64,
        sel_idx: u64,
        sel_off: u64,
        list_len: usize,
        kq_scale: f32,
        n_head: usize,
        n_kv: usize,
        hd: usize,
        t: usize,
        out: u64,
    ) -> Result<(), String> {
        // sel 버퍼가 이미 디바이스에 있다 — 업로드 없이 qsa_attn_res와 동일한
        // 커널 쌍(t=1 분할 우선)을 발사한다.
        if t != 1 || std::env::var("LLM170_QSA_SPLIT").as_deref() == Ok("0") {
            return Err(format!("qsa_attention_dev_sel: t={t} 비분할은 미지원"));
        }
        let cap = std::env::var("LLM170_QSA_SPLITS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(64);
        let n_splits: usize = (list_len / 32).clamp(1, cap.max(1).min(512));
        let pdev = {
            let mut g = self.qsp.lock().map_err(|e| e.to_string())?;
            g.ensure(&self.ctx, n_head * n_splits * 32 * 10 * 4)?
        };
        let (qdev, odev) = (self.fptr(q)?, self.fptr(out)?);
        let mut q_p = qdev as *mut std::ffi::c_void;
        let mut k_p = ck as *mut std::ffi::c_void;
        let mut v_p = cv as *mut std::ffi::c_void;
        let mut si_p = sel_idx as *mut std::ffi::c_void;
        let mut so_p = sel_off as *mut std::ffi::c_void;
        let mut pa_p = pdev as *mut std::ffi::c_void;
        let mut ns = n_splits as i32;
        let mut sc = kq_scale;
        let mut nh = n_head as i32;
        let mut nk = n_kv as i32;
        let mut h = hd as i32;
        let mut args: Vec<*mut std::ffi::c_void> = vec![
            (&mut q_p) as *mut _ as *mut std::ffi::c_void,
            (&mut k_p) as *mut _ as *mut std::ffi::c_void,
            (&mut v_p) as *mut _ as *mut std::ffi::c_void,
            (&mut si_p) as *mut _ as *mut std::ffi::c_void,
            (&mut so_p) as *mut _ as *mut std::ffi::c_void,
            (&mut pa_p) as *mut _ as *mut std::ffi::c_void,
            (&mut ns) as *mut _ as *mut std::ffi::c_void,
            (&mut sc) as *mut _ as *mut std::ffi::c_void,
            (&mut nh) as *mut _ as *mut std::ffi::c_void,
            (&mut nk) as *mut _ as *mut std::ffi::c_void,
            (&mut h) as *mut _ as *mut std::ffi::c_void,
        ];
        self.ctx.launch3(
            "q4_qsa_attn_sel4s",
            n_splits.div_ceil(4) as u32,
            (n_head / 12) as u32,
            1,
            256,
            &mut args,
        )?;
        let mut pa2_p = pdev as *mut std::ffi::c_void;
        let mut q2_p = qdev as *mut std::ffi::c_void;
        let mut o2_p = odev as *mut std::ffi::c_void;
        let mut ns2 = n_splits as i32;
        let mut nh2 = n_head as i32;
        let mut h2 = hd as i32;
        let mut margs: Vec<*mut std::ffi::c_void> = vec![
            (&mut pa2_p) as *mut _ as *mut std::ffi::c_void,
            (&mut q2_p) as *mut _ as *mut std::ffi::c_void,
            (&mut o2_p) as *mut _ as *mut std::ffi::c_void,
            (&mut ns2) as *mut _ as *mut std::ffi::c_void,
            (&mut nh2) as *mut _ as *mut std::ffi::c_void,
            (&mut h2) as *mut _ as *mut std::ffi::c_void,
        ];
        self.ctx.launch3(
            "q4_qsa_attn_sel4s_merge",
            n_head.div_ceil(8) as u32,
            1,
            1,
            256,
            &mut margs,
        )?;
        Ok(())
    }

    fn qsa_sel_readback(
        &self,
        sel_idx: u64,
        sel_off: u64,
        list_len: usize,
    ) -> Result<(Vec<u32>, Vec<u32>), String> {
        let mut idx = vec![0u32; list_len];
        let mut off = vec![0u32; 2];
        self.ctx
            .d2h(bytemuck::cast_slice_mut(&mut idx), sel_idx as *const u8)?;
        self.ctx
            .d2h(bytemuck::cast_slice_mut(&mut off), sel_off as *const u8)?;
        Ok((idx, off))
    }

    #[allow(clippy::too_many_arguments)]
    fn qsa_attention(
        &self,
        q: &[f32],
        ck: &[f32],
        cv: &[f32],
        mask: &[u32],
        kq_scale: f32,
        n_past: usize,
        n_head: usize,
        n_kv: usize,
        hd: usize,
        t: usize,
    ) -> Result<Vec<f32>, String> {
        // t>128 가드 해제(2026-09-13): q4-qsa-check 프로브로 커널이 t=129·200·
        // 512(n_past 512)에서 CPU 미러와 일치함을 확인(최대 2e-5). 기존 가드는
        // 폴백을 유발했지만 호출자(qsa.rs)의 Err 경로가 CPU 재계산 없이 **빈
        // 어텐션 행**을 반환해 어텐션 자체가 누락됐다(양 경로 동일 → 자가일치
        // 검사가 통과). 유일한 강제 폴백: LLM170_QSA_CPU=1.
        if std::env::var_os("LLM170_QSA_CPU").is_some() {
            return Err(format!("q4acc: qsa_attention t={t} CPU 강제(LLM170_QSA_CPU)"));
        }
        self.qsa_attn_raw(q, ck, cv, mask, kq_scale, n_past, n_head, n_kv, hd, t)
    }

    #[allow(clippy::too_many_arguments)]
    fn qsa_attention_sel(
        &self,
        q: &[f32],
        ck: &[f32],
        cv: &[f32],
        sel_idx: &[u32],
        sel_off: &[u32],
        kq_scale: f32,
        n_head: usize,
        n_kv: usize,
        hd: usize,
        t: usize,
    ) -> Result<Vec<f32>, String> {
        if std::env::var_os("LLM170_QSA_CPU").is_some() {
            return Err(format!("q4acc: qsa_attention_sel t={t} CPU 강제"));
        }
        // 4헤드-퍼-워프판은 K/V 행을 4헤드가 공유한다(트래픽 1/4) — 프리필에서
        // −20%. 단 t가 작으면(디코드) 블록의 워프 대부분이 놀아 역효과이므로
        // t≤3은 헤드당 워프 1개인 `_sel`로 보낸다. 둘은 비트 동일(프로브 확인).
        // 실측(2026-09-14): 그 강제(LLM170_QSA_SEL4_DEC)는 디코드에서 중립이었다
        // (589.7/562.2 vs 576.3/566.2ms, 토큰 동일) — 점유율 손실이 트래픽 이득을 상쇄.
        // 장문맥 디코드의 실제 비용은 아래와 같다(pp8192, KTRACE):
        //   q4_qsa_attn_sel = 1.425 ms/콜 = 17.1 ms/스텝(커널 합 83.9ms의 20%, 최대 단일)
        //   = 8192위치 x 256 x 2(K,V) x 24헤드 = 403 MB/층 -> 236 GB/s로 1.7ms ≈ 측정치
        // 즉 **헤드 24개가 같은 K/V 행을 각자 다시 읽는 대역폭 문제**다. 4헤드 공유로는
        // 점유율 때문에 안 되고, flash-decoding형(선택목록을 블록 간 분할 + 부분 softmax
        // 병합)으로 K/V를 1회만 읽어야 한다 — 17.1ms -> ~1ms, 스텝의 ~8%.
        // t=1 분할판은 기본 ON이다(장문맥 디코드 142.5 -> 124.4 ms/스텝 = -12.7%,
        // diverse 스트림 완전 동일, 단문맥 무회귀). 비트 동일 경로 복귀는
        // LLM170_QSA_SPLIT=0, 분할 상한은 LLM170_QSA_SPLITS(기본 64, 목록/32로 적응).
        if t == 1 && std::env::var("LLM170_QSA_SPLIT").as_deref() != Ok("0") {
            // 위치 분할(flash-decoding형) — 지연 바운드인 t=1을 (split, 헤드묶음)
            // 그리드로 펼친다. 부분 소프트맥스를 2차 커널이 병합하므로 합산
            // 순서가 달라진다(greedy 스트림 동일성으로 검증, 비트 동일 아님).
            self.qsa_attn_sel4s_raw(q, ck, cv, sel_idx, sel_off, kq_scale, n_head, n_kv, hd, t)
        } else if t <= 3 {
            self.qsa_attn_sel_raw(q, ck, cv, sel_idx, sel_off, kq_scale, n_head, n_kv, hd, t)
        } else {
            self.qsa_attn_sel4_raw(q, ck, cv, sel_idx, sel_off, kq_scale, n_head, n_kv, hd, t)
        }
    }
}
