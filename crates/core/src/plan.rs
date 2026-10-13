//! 배치 플래너 v0 (§2.M "배치 = 함수" — 2026-10-13).
//!
//! 원칙(루트 AGENTS·플랜 §1-10): 배치안은 **모델명·케이스 분기가 아니라**
//! 매니페스트(범주별 바이트) × 하드웨어(VRAM·호스트 RAM·SSD) × 워크로드
//! (목표 ctx·KV/토큰·문맥 수)의 순수 함수다. v0 = 분류기 + 사다리 규칙 +
//! 배치안 산출(기동 로그·`/stats`·`llm170 plan` 드라이런). 실제 결정 배선
//! (`--moe auto`의 상주 판정 등)은 기존 실측 기반 판정 유지 — 다중 GPU·FN 시점.
//!
//! 사다리(우선순위): ① 매-토큰 dense(+head) → VRAM 필수(불가 = 기동 거부)
//! ② KV 공유 풀(문맥 추가분 GDN 상태 포함; 부족 시 실효 ctx 축소)
//! ③ 전문가 → 잔여 VRAM 핫셋(전량 수용이면 상주; 초과분 = 호스트 RAM
//! 스트리밍 → 그마저 부족하면 SSD) ④ 행-스파스 임베딩(PLE) → 기본 SSD
//! (잔여 VRAM이 전부 덮으면 상주) ⑤ embed → 호스트 mmap.

use std::path::Path;

/// 배치 범주 — 경로 규칙으로만 판정(모델 특정 이름 금지).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Category {
    /// 매-토큰 밀집(어텐션·GDN·라우터·head·shared_expert·PLE conv 등) — VRAM.
    Dense,
    /// 전문가 은행(`.mlp.experts.`) — 잔여 VRAM 핫셋/스트리밍.
    Experts,
    /// 행-스파스 임베딩(PLE `ple_embedding`) — 기본 SSD.
    RowSparse,
    /// 호스트 embed_row(`embed_tokens`/`token_embd`) — 호스트 mmap.
    Embed,
}

/// 텐서명 → 배치 범주(`shared_expert`는 매-토큰이므로 dense 우선 판정).
pub fn classify(name: &str) -> Category {
    if name.contains("shared_expert") {
        return Category::Dense;
    }
    if name.contains(".mlp.experts.") {
        return Category::Experts;
    }
    if name.contains("ple_embedding") || name.contains("ngram_embedding") {
        return Category::RowSparse;
    }
    if name.contains("embed_tokens") || name.contains("token_embd") {
        return Category::Embed;
    }
    Category::Dense
}

/// 범주별 바이트 합.
#[derive(Clone, Copy, Default, PartialEq, Eq, Debug)]
pub struct Sizes {
    pub dense: u64,
    pub experts: u64,
    pub row_sparse: u64,
    pub embed: u64,
}

impl Sizes {
    pub fn total(&self) -> u64 {
        self.dense
            .saturating_add(self.experts)
            .saturating_add(self.row_sparse)
            .saturating_add(self.embed)
    }
}

/// 텐서명→바이트 매니페스트(safetensors 헤더/index만 — 텐서 비적재).
pub struct Manifest {
    items: Vec<(String, u64)>,
}

impl Manifest {
    pub fn from_items(mut items: Vec<(String, u64)>) -> Self {
        items.sort();
        Self { items }
    }

    /// 모델 디렉터리의 safetensors index/헤더만 판독(core::st — 데이터 비적재).
    pub fn from_dir(dir: &Path) -> Result<Self, String> {
        // safetensors 헤더/index만(데이터 비적재·**dtype 비의존**: FP8 등 st 로더
        // 미지원 형식도 바이트 계상 가능 — 배치 판단에는 형상·크기만 필요).
        let idx = dir.join("model.safetensors.index.json");
        let mut shard_names: Vec<String> = Vec::new();
        if idx.exists() {
            let raw = std::fs::read_to_string(&idx).map_err(|e| format!("index.json: {e}"))?;
            let v = crate::json::Json::parse(&raw).map_err(|e| format!("index.json 파싱: {e}"))?;
            let obj = v
                .get("weight_map")
                .and_then(crate::json::Json::as_object)
                .ok_or("index.json: weight_map 없음")?;
            for (_, file) in obj {
                let file = file.as_str().ok_or("weight_map 값이 문자열이 아님")?;
                if !shard_names.iter().any(|s| s == file) {
                    shard_names.push(file.to_string());
                }
            }
        } else {
            shard_names = std::fs::read_dir(dir)
                .map_err(|e| format!("{}: {e}", dir.display()))?
                .flatten()
                .map(|e| e.file_name().to_string_lossy().into_owned())
                .filter(|n| n.ends_with(".safetensors") && n.starts_with("model"))
                .collect();
            if shard_names.is_empty() {
                return Err(format!("{}에 model*.safetensors 없음", dir.display()));
            }
        }
        shard_names.sort();
        let mut items: Vec<(String, u64)> = Vec::new();
        for shard in &shard_names {
            scan_header(&dir.join(shard), &mut items)?;
        }
        if items.is_empty() {
            return Err("매니페스트 비음".into());
        }
        Ok(Self::from_items(items))
    }

    pub fn len(&self) -> usize {
        self.items.len()
    }

    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    pub fn sums(&self) -> Sizes {
        let mut s = Sizes::default();
        for (n, b) in &self.items {
            match classify(n) {
                Category::Dense => s.dense = s.dense.saturating_add(*b),
                Category::Experts => s.experts = s.experts.saturating_add(*b),
                Category::RowSparse => s.row_sparse = s.row_sparse.saturating_add(*b),
                Category::Embed => s.embed = s.embed.saturating_add(*b),
            }
        }
        s
    }
}

/// 하드웨어 — 단일 카드 또는 총량. `vram_free`가 있으면 총량 대신 실가용 기준.
#[derive(Clone, Copy, Debug)]
pub struct Hardware {
    pub vram_total: u64,
    pub vram_free: Option<u64>,
    pub host_ram: u64,
    pub ssd: Option<u64>,
    /// 워크스페이스·스크래치 예약(운영 상수).
    pub reserve: u64,
    /// [2.M 샤드] 카드 수(1 = 단일). vram은 전 카드 합(또는 총 가용).
    pub cards: usize,
}

/// 워크로드 — 목표 문맥·문맥 수.
#[derive(Clone, Copy, Debug)]
pub struct Workload {
    pub model_max_ctx: usize,
    pub want_ctx: usize,
    pub kv_bytes_per_token: u64,
    pub state_bytes_per_ctx: u64,
    pub contexts: usize,
}

/// 배치 티어.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Tier {
    Vram,
    HostRam,
    Ssd,
}

impl Tier {
    pub fn name(self) -> &'static str {
        match self {
            Tier::Vram => "vram",
            Tier::HostRam => "host",
            Tier::Ssd => "ssd",
        }
    }
}

/// 배치안 산출.
#[derive(Clone, Debug)]
pub struct Plan {
    pub sizes: Sizes,
    /// 판정에 쓴 VRAM 기준(가용 우선).
    pub vram: u64,
    pub reserve: u64,
    pub kv_budget: u64,
    pub effective_ctx: usize,
    /// 목표 ctx = min(요청, 모델 최대).
    pub goal_ctx: usize,
    /// 실효 ctx < 목표(요청·모델 최대의 min) — 풀 부족 클램프.
    pub ctx_capped: bool,
    pub experts_tier: Tier,
    /// VRAM 핫셋(상주 시 = 전문가 총량).
    pub experts_hot: u64,
    pub row_sparse_tier: Tier,
    /// 호스트 RAM 필요(스트리밍 전문가 + embed mmap).
    pub host_ram_need: u64,
    /// SSD 필요(행-스파스 + SSD로 밀린 전문가).
    pub ssd_need: u64,
    pub refuse: Option<&'static str>,
}

impl Plan {
    pub fn fits(&self) -> bool {
        self.refuse.is_none()
    }

    /// 전량 상주(전문가까지 VRAM) — 순수 dense는 전문가 0으로 자동 성립.
    pub fn fully_resident(&self) -> bool {
        self.refuse.is_none() && self.experts_tier == Tier::Vram
    }

    /// 한 줄 요약(기동 로그·`/stats` — JSON 안전: 따옴표/역슬래시 없음).
    pub fn summary(&self) -> String {
        if let Some(r) = self.refuse {
            return format!("거부: {r}");
        }
        let gib = |b: u64| b as f64 / (1u64 << 30) as f64;
        format!(
            "dense={:.1}G experts={:.1}G row_sparse={:.1}G embed={:.1}G · vram={:.1}G kv_budget={:.1}G ctx={}/{} ({}) · experts={}(hot {:.1}G) ple={} · host_ram_need={:.1}G ssd_need={:.1}G",
            gib(self.sizes.dense),
            gib(self.sizes.experts),
            gib(self.sizes.row_sparse),
            gib(self.sizes.embed),
            gib(self.vram),
            gib(self.kv_budget),
            self.effective_ctx,
            self.goal_ctx,
            if self.ctx_capped { "capped" } else { "full" },
            self.experts_tier.name(),
            gib(self.experts_hot),
            self.row_sparse_tier.name(),
            gib(self.host_ram_need),
            gib(self.ssd_need),
        )
    }
}

/// 층별 정보(샤드 플랜 입력) — 이름의 `layers.N` 파싱.
#[derive(Clone, Copy, Default, Debug)]
pub struct LayerInfo {
    pub bytes: u64,
    /// 풀 어텐션 층 여부(`.self_attn.` 포함) — KV 비중 계산용(GDN은 KV 없음).
    pub full_attn: bool,
}

/// 매니페스트의 층 분해 — (층 목록, 층 미소속 가중치, embed(호스트)).
#[derive(Clone, Debug, Default)]
pub struct LayerSplit {
    pub layers: Vec<LayerInfo>,
    pub non_layer: u64,
    pub embed: u64,
}

impl Manifest {
    /// 텐서명 → 층 분해(`layers.{N}.`; 그룹은 embed(호스트)·비층(헤드 등)).
    pub fn layer_split(&self) -> LayerSplit {
        let layer_of = |name: &str| -> Option<usize> {
            let pos = name
                .find(".layers.")
                .map(|p| p + 8)
                .or_else(|| name.find("layers.").map(|p| p + 7))?;
            let digits: String = name[pos..]
                .chars()
                .take_while(|c| c.is_ascii_digit())
                .collect();
            digits.parse().ok()
        };
        let mut layers: Vec<LayerInfo> = Vec::new();
        let mut split = LayerSplit::default();
        for (name, bytes) in &self.items {
            match (classify(name), layer_of(name)) {
                (_, Some(li)) => {
                    if layers.len() <= li {
                        layers.resize(li + 1, LayerInfo::default());
                    }
                    layers[li].bytes = layers[li].bytes.saturating_add(*bytes);
                    if name.contains(".self_attn.") {
                        layers[li].full_attn = true;
                    }
                }
                (Category::Embed, None) => split.embed = split.embed.saturating_add(*bytes),
                (_, None) => split.non_layer = split.non_layer.saturating_add(*bytes),
            }
        }
        split.layers = layers;
        split
    }
}

/// 다중 카드 레이어 파이프라인 샤드 플랜 (§2.M — 좁은 링크에서 PP 우선:
/// CMP 실측 PP4 87.6 > TP4 60.4 t/s). 층 연속 블록 배분(링크 통과 = 경계 활성만),
/// KV·상태는 카드 로컬(층 비중만큼). **v0 = 배분·수용 계산**(실행 배선은 다중
/// GPU 도착 후 — 카드=층 블록+로컬 KV). 케이스 하드코딩 없음(층 크기·카드 수 함수).
#[derive(Clone, Debug)]
pub struct ShardPlan {
    pub cards: usize,
    /// 카드별 [start, end) 층 범위.
    pub blocks: Vec<(usize, usize)>,
    pub per_card_weights: Vec<u64>,
    pub per_card_kv: Vec<u64>,
    pub per_card_state: Vec<u64>,
    /// 필요량 = 가중치 + 로컬 KV + 상태 (+비층 가중치는 마지막 카드).
    pub per_card_need: Vec<u64>,
    /// 카드 유효 용량(예약 차감).
    pub per_card_cap: u64,
    pub worst_card: usize,
    pub fits: bool,
    pub notes: Vec<String>,
}

/// 샤드 플랜 — 순수 함수.
pub fn shard_plan(
    layers: &[LayerInfo],
    split: &LayerSplit,
    hw: &Hardware,
    wl: &Workload,
) -> ShardPlan {
    let cards = hw.cards.max(1);
    let notes = Vec::new();
    let v = hw.vram_free.unwrap_or(hw.vram_total);
    let per_card_cap = (v / cards as u64).saturating_sub(hw.reserve);
    let n_full = layers.iter().filter(|l| l.full_attn).count().max(1) as u64;
    let n_gdn = layers.iter().filter(|l| !l.full_attn).count().max(1) as u64;
    let ctx = wl.want_ctx.min(wl.model_max_ctx) as u64;
    let kv_total = wl.kv_bytes_per_token.saturating_mul(ctx);
    let state_total = wl
        .state_bytes_per_ctx
        .saturating_mul(wl.contexts.max(1) as u64);
    // 연속 블록 그리디: 목표 = (가중치 총합 + 비층)/카드수.
    let weights_total: u64 = layers.iter().map(|l| l.bytes).sum::<u64>() + split.non_layer;
    let target = weights_total / cards as u64;
    let mut blocks: Vec<(usize, usize)> = Vec::new();
    let mut per_card_weights = vec![0u64; cards];
    {
        let mut card = 0usize;
        let mut start = 0usize;
        for (i, l) in layers.iter().enumerate() {
            let last = card + 1 == cards;
            if !last
                && card < cards
                && per_card_weights[card] > 0
                && per_card_weights[card] + l.bytes > target
            {
                blocks.push((start, i));
                card += 1;
                start = i;
            }
            per_card_weights[card] = per_card_weights[card].saturating_add(l.bytes);
        }
        blocks.push((start, layers.len()));
        while blocks.len() < cards {
            let end = blocks.last().map(|b| b.1).unwrap_or(0);
            blocks.push((end, end));
        }
    }
    // 비층(헤드 등)은 마지막 카드에 가산. KV·상태는 층 비중만큼.
    if cards > 0 {
        per_card_weights[cards - 1] = per_card_weights[cards - 1].saturating_add(split.non_layer);
    }
    let mut per_card_kv = vec![0u64; cards];
    let mut per_card_state = vec![0u64; cards];
    for (c, &(s, e)) in blocks.iter().enumerate() {
        let nf = layers[s.min(e)..e].iter().filter(|l| l.full_attn).count() as u64;
        let ng = layers[s.min(e)..e].iter().filter(|l| !l.full_attn).count() as u64;
        per_card_kv[c] = kv_total.saturating_mul(nf) / n_full;
        per_card_state[c] = state_total.saturating_mul(ng) / n_gdn;
    }
    let per_card_need: Vec<u64> = (0..cards)
        .map(|c| {
            per_card_weights[c]
                .saturating_add(per_card_kv[c])
                .saturating_add(per_card_state[c])
        })
        .collect();
    let worst_card = (0..cards).max_by_key(|&c| per_card_need[c]).unwrap_or(0);
    let fits = per_card_need[worst_card] <= per_card_cap;
    ShardPlan {
        cards,
        blocks,
        per_card_weights,
        per_card_kv,
        per_card_state,
        per_card_need,
        per_card_cap,
        worst_card,
        fits,
        notes,
    }
}

/// 배치안 계산 — 순수 함수(사다리 규칙).
pub fn plan(m: &Manifest, hw: &Hardware, wl: &Workload) -> Plan {
    let s = m.sums();
    let v = hw.vram_free.unwrap_or(hw.vram_total);
    let x = hw.reserve;
    let mut refuse = None;

    // ① 매-토큰 dense(+head) — VRAM 필수. 불가 = 기동 거부 클래스.
    if s.dense.saturating_add(x) > v {
        refuse = Some("dense+예약 > 가용 VRAM");
    }
    // ② KV 공유 풀 = 가용 − dense − 예약 − 문맥 추가분 상태.
    let state = (wl.contexts.saturating_sub(1)) as u64 * wl.state_bytes_per_ctx;
    let kv_budget = v
        .saturating_sub(s.dense)
        .saturating_sub(x)
        .saturating_sub(state);
    let cap = (kv_budget / wl.kv_bytes_per_token.max(1)) as usize;
    let cap = cap / 512 * 512;
    let goal = wl.want_ctx.min(wl.model_max_ctx);
    let effective_ctx = cap.min(goal);
    if effective_ctx < 512 && refuse.is_none() {
        refuse = Some("KV 풀 < 512 토큰");
    }
    let ctx_capped = effective_ctx < goal;
    let kv_used = effective_ctx as u64 * wl.kv_bytes_per_token;
    let spare = kv_budget.saturating_sub(kv_used);

    // ③ 전문가 — 잔여 VRAM이 핫셋(전량 수용이면 상주). 스트리밍은 호스트 RAM이
    //    전문가 뱅크 전체를 받을 때만, 아니면 SSD.
    let experts_hot = s.experts.min(spare);
    let experts_tier = if s.experts <= spare {
        Tier::Vram
    } else if hw.host_ram >= s.experts.saturating_add(s.embed) {
        Tier::HostRam
    } else {
        Tier::Ssd
    };
    let leftover = spare.saturating_sub(experts_hot);

    // ④ 행-스파스(PLE) — 기본 SSD; 잔여 VRAM이 전부 덮으면 상주.
    let row_sparse_tier = if s.row_sparse <= leftover {
        Tier::Vram
    } else {
        Tier::Ssd
    };

    // ⑤ 회계 — 호스트 mmap(스트리밍 전문가 + embed), SSD(행-스파스/전문가).
    let streamed = s.experts.saturating_sub(experts_hot);
    let host_ram_need = streamed.saturating_add(s.embed);
    let mut ssd_need = if row_sparse_tier == Tier::Ssd {
        s.row_sparse
    } else {
        0
    };
    if experts_tier == Tier::Ssd {
        ssd_need = ssd_need.saturating_add(s.experts);
    }

    Plan {
        sizes: s,
        vram: v,
        reserve: x,
        kv_budget,
        effective_ctx,
        goal_ctx: goal,
        ctx_capped,
        experts_tier,
        experts_hot,
        row_sparse_tier,
        host_ram_need,
        ssd_need,
        refuse,
    }
}

/// safetensors 샤드 헤더의 텐서별 바이트(data_offsets) 합산 — dtype 무시.
fn scan_header(path: &Path, out: &mut Vec<(String, u64)>) -> Result<(), String> {
    use std::io::Read;
    let mut f = std::fs::File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut len8 = [0u8; 8];
    f.read_exact(&mut len8)
        .map_err(|e| format!("{} 헤더 길이: {e}", path.display()))?;
    let n = u64::from_le_bytes(len8) as usize;
    if n == 0 || n > (1 << 28) {
        return Err(format!("{} 헤더 길이 이상: {n}", path.display()));
    }
    let mut buf = vec![0u8; n];
    f.read_exact(&mut buf)
        .map_err(|e| format!("{} 헤더 판독: {e}", path.display()))?;
    let text = std::str::from_utf8(&buf).map_err(|e| format!("{}: {e}", path.display()))?;
    let hdr =
        crate::json::Json::parse(text).map_err(|e| format!("{} 헤더 파싱: {e}", path.display()))?;
    let obj = hdr
        .as_object()
        .ok_or_else(|| format!("{} 헤더가 객체가 아님", path.display()))?;
    for (name, ent) in obj {
        if name == "__metadata__" {
            continue;
        }
        let offs = ent
            .get("data_offsets")
            .and_then(crate::json::Json::as_num_array)
            .ok_or_else(|| format!("{name}: data_offsets 없음"))?;
        if offs.len() != 2 || offs[1] < offs[0] {
            return Err(format!("{name}: data_offsets 이상"));
        }
        out.push((name.clone(), offs[1] - offs[0]));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const G: u64 = 1 << 30;

    /// 실모델 텐서명 분류 표(27B·35B·FN 실명 샘플 — 픽스처).
    #[test]
    fn classify_real_names() {
        let cases: &[(&str, Category)] = &[
            // 27B dense
            (
                "model.language_model.layers.0.linear_attn.A_log",
                Category::Dense,
            ),
            (
                "model.language_model.layers.0.self_attn.q_proj.weight_packed",
                Category::Dense,
            ),
            ("model.language_model.embed_tokens.weight", Category::Embed),
            ("lm_head.weight", Category::Dense),
            // 35B MoE
            (
                "model.language_model.layers.0.mlp.experts.0.down_proj.weight_packed",
                Category::Experts,
            ),
            (
                "model.language_model.layers.0.mlp.experts.127.up_proj.weight_packed",
                Category::Experts,
            ),
            (
                "model.language_model.layers.0.mlp.shared_expert.down_proj.weight",
                Category::Dense,
            ),
            (
                "model.language_model.layers.0.mlp.gate.weight",
                Category::Dense,
            ),
            // FN PLE(행-스파스) + PLE dense 모듈
            (
                "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_7.weight",
                Category::RowSparse,
            ),
            (
                "model.language_model.layers.1.ple.ple_embedding.layer_multipliers",
                Category::RowSparse,
            ),
            (
                "model.language_model.layers.1.ple.conv1d.weight",
                Category::Dense,
            ),
            (
                "model.language_model.layers.1.ple.key_proj.weight",
                Category::Dense,
            ),
            // 엔진 매핑명(token_embd) 폴백
            ("token_embd.weight", Category::Embed),
        ];
        for (name, want) in cases {
            assert_eq!(classify(name), *want, "name={name}");
        }
    }

    fn man(dense: u64, experts: u64, row_sparse: u64) -> Manifest {
        Manifest::from_items(vec![
            ("model.layers.0.self_attn.q_proj.weight".into(), dense),
            (
                "model.layers.0.mlp.experts.0.down_proj.weight".into(),
                experts,
            ),
            (
                "model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight".into(),
                row_sparse,
            ),
        ])
    }

    fn hw(vram: u64, host: u64) -> Hardware {
        Hardware {
            vram_total: vram,
            vram_free: None,
            host_ram: host,
            ssd: None,
            reserve: 5 * G / 2, // 2.5GiB 운영 예약
            cards: 1,
        }
    }

    fn wl(ctx: usize) -> Workload {
        Workload {
            model_max_ctx: 262_144,
            want_ctx: ctx,
            kv_bytes_per_token: 33 * 1024,
            state_bytes_per_ctx: 151 * (1 << 20),
            contexts: 4,
        }
    }

    /// 드라이런 예시(§2.M — 목표값이 아니라 함수 출력의 검증 픽스처):
    /// 100B급→2장 상주 / 200B급→2장 상주(여유 감소) / 300B급→2장 초과·4장 상주 /
    /// 500B+→4장에서도 오프로드.
    #[test]
    fn dry_run_ladder_examples() {
        // 100B급(≈52G: dense 12 + experts 40) — 2×64G 상주.
        let p = plan(&man(12 * G, 40 * G, 0), &hw(128 * G, 62 * G), &wl(262_144));
        assert!(p.fully_resident(), "{p:?}");
        assert_eq!(p.effective_ctx, 262_144);

        // 200B급(≈104G: dense 14 + experts 90) — 2장, 여유 감소하나 상주.
        let p = plan(&man(14 * G, 90 * G, 0), &hw(128 * G, 62 * G), &wl(262_144));
        assert!(p.fully_resident(), "{p:?}");

        // 300B급(≈156G: dense 18 + experts 138) — 2장 초과 → 오프로드(호스트 부족 → SSD),
        // 4장이면 상주.
        let m300 = man(18 * G, 138 * G, 0);
        let p2 = plan(&m300, &hw(128 * G, 62 * G), &wl(262_144));
        assert!(!p2.fully_resident());
        assert_eq!(p2.experts_tier, Tier::Ssd, "{p2:?}");
        assert!(p2.experts_hot > 0); // 잔여 VRAM은 핫셋으로
        let p4 = plan(&m300, &hw(256 * G, 62 * G), &wl(262_144));
        assert!(p4.fully_resident(), "{p4:?}");

        // 500B+(≈260G: dense 25 + experts 235) — 4장에서도 오프로드(호스트 스트리밍).
        let p = plan(
            &man(25 * G, 235 * G, 0),
            &hw(256 * G, 512 * G),
            &wl(262_144),
        );
        assert!(!p.fully_resident());
        assert_eq!(p.experts_tier, Tier::HostRam, "{p:?}");
        assert_eq!(p.ssd_need, 0);
        // 호스트 몫 = 뱅크 − VRAM 핫셋.
        assert_eq!(p.host_ram_need, 235 * G - p.experts_hot);
    }

    /// 호스트 RAM이 전문가 뱅크를 못 받으면 SSD 티어로 강등.
    #[test]
    fn experts_fall_to_ssd_without_host_ram() {
        let p = plan(&man(18 * G, 138 * G, 0), &hw(128 * G, 8 * G), &wl(262_144));
        assert_eq!(p.experts_tier, Tier::Ssd, "{p:?}");
        assert!(p.ssd_need >= 138 * G);
    }

    /// dense가 안 들어가면 기동 거부 클래스.
    #[test]
    fn refuse_when_dense_exceeds() {
        let p = plan(&man(130 * G, 10 * G, 0), &hw(128 * G, 0), &wl(262_144));
        assert!(!p.fits());
        assert!(p.refuse.is_some());
    }

    /// 실효 ctx = min(목표, 모델 최대, KV 풀 파생) — 풀 부족 시 클램프·512 하한.
    #[test]
    fn effective_ctx_is_pool_derived() {
        // 큰 KV/토큰(200KiB) — 24G 카드에서 풀이 256K를 못 덮는다.
        let mut w = wl(262_144);
        w.kv_bytes_per_token = 200 * 1024;
        let p = plan(&man(10 * G, 0, 0), &hw(24 * G, 62 * G), &w);
        assert!(p.effective_ctx < 262_144);
        assert!(p.ctx_capped);
        assert_eq!(p.effective_ctx % 512, 0);
        // 모델 최대보다 큰 목표는 모델 최대가 상한(캡 아님 — 목표의 min).
        let p = plan(&man(10 * G, 0, 0), &hw(24 * G, 62 * G), &wl(1 << 20));
        assert_eq!(p.effective_ctx, 262_144);
        assert!(!p.ctx_capped);
    }

    /// 문맥 추가분(GDN 상태)이 KV 예산에서 차감된다 — 문맥 수의 함수.
    #[test]
    fn context_state_subtracts_from_kv_budget() {
        let mut w1 = wl(262_144);
        w1.contexts = 1;
        let mut w8 = wl(262_144);
        w8.contexts = 8;
        let p1 = plan(&man(10 * G, 0, 0), &hw(24 * G, 62 * G), &w1);
        let p8 = plan(&man(10 * G, 0, 0), &hw(24 * G, 62 * G), &w8);
        assert_eq!(p1.kv_budget - p8.kv_budget, 7 * w1.state_bytes_per_ctx);
    }

    /// 경계: 전문가 == 잔여 VRAM 이면 상주, +1바이트면 스트리밍.
    #[test]
    fn experts_boundary_exact_fit() {
        // want=512 → kv_used 소량, 상태 3×151M.
        let w = wl(512);
        let base = plan(&man(10 * G, 0, 0), &hw(128 * G, 512 * G), &w);
        let spare = base.kv_budget - base.effective_ctx as u64 * w.kv_bytes_per_token;
        let p = plan(&man(10 * G, spare, 0), &hw(128 * G, 512 * G), &w);
        assert!(p.fully_resident(), "{p:?}");
        let p = plan(&man(10 * G, spare + 1, 0), &hw(128 * G, 512 * G), &w);
        assert_eq!(p.experts_tier, Tier::HostRam, "{p:?}");
    }

    /// 행-스파스(PLE): 기본 SSD — 잔여 VRAM이 전부 덮는 소형이면 상주.
    #[test]
    fn row_sparse_default_ssd() {
        let p = plan(&man(10 * G, 0, 48 * G), &hw(56 * G, 62 * G), &wl(512));
        assert_eq!(p.row_sparse_tier, Tier::Ssd);
        assert!(p.ssd_need >= 48 * G);
        let p = plan(&man(10 * G, 0, 8 * G), &hw(128 * G, 62 * G), &wl(512));
        assert_eq!(p.row_sparse_tier, Tier::Vram);
    }

    /// 헤더 스캔 — index 없는 단일 파일 경로 + 미지원 dtype(FP8)도 계상.
    #[test]
    fn scan_header_accepts_unknown_dtype() {
        let dir = std::env::temp_dir().join(format!("llm170-plan-test-{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        let hdr = r#"{"a.weight":{"dtype":"F8_E4M3","shape":[2],"data_offsets":[0,2]},"b.weight":{"dtype":"F32","shape":[1],"data_offsets":[2,6]},"__metadata__":{"format":"pt"}}"#;
        let mut f = Vec::new();
        f.extend_from_slice(&(hdr.len() as u64).to_le_bytes());
        f.extend_from_slice(hdr.as_bytes());
        f.extend_from_slice(&[0u8; 6]);
        std::fs::write(dir.join("model.safetensors"), &f).unwrap();
        let m = Manifest::from_dir(&dir).unwrap();
        assert_eq!(m.len(), 2);
        assert_eq!(m.sums().dense, 6);
        let _ = std::fs::remove_dir_all(&dir);
    }

    // ── §2.M 샤드 플랜 (다중 카드 레이어 파이프라인) ──
    fn shard_layers(n: usize, bytes: u64) -> Vec<LayerInfo> {
        (0..n)
            .map(|i| LayerInfo {
                bytes,
                full_attn: i % 4 == 3, // 하이브리드 3/4 GDN + 1/4 풀어텐션
            })
            .collect()
    }

    fn shard_hw(vram: u64, host: u64, cards: usize) -> Hardware {
        let mut h = hw(vram, host);
        h.cards = cards;
        h
    }

    fn shard_split(non_layer: u64) -> LayerSplit {
        LayerSplit {
            layers: Vec::new(),
            non_layer,
            embed: 0,
        }
    }

    /// 드라이런 예시(§2.M — 함수 출력의 검증 픽스처): 100B급@2 상주 /
    /// 300B급@2 초과·@4 상주 / 500B급@4 초과(오프로드 레짐).
    #[test]
    fn shard_dry_run_examples() {
        // 100B급 ≈52G(64층×0.8125G), 2×64G(총 128G).
        let l = shard_layers(64, 812 * G / 1000);
        let p = shard_plan(
            &l,
            &shard_split(0),
            &shard_hw(128 * G, 62 * G, 2),
            &wl(262_144),
        );
        assert_eq!(p.cards, 2);
        assert!(p.fits, "{p:?}");
        // 블록 균형 — 최대/최소 카드 가중치 ≤ 1.15×.
        let (mx, mn) = (
            *p.per_card_weights.iter().max().unwrap(),
            *p.per_card_weights.iter().min().unwrap(),
        );
        assert!(mx * 100 <= mn * 115, "불균형 {mx} vs {mn}");
        // 300B급 ≈155G, 2장 초과·4장 상주.
        let l3 = shard_layers(64, 2422 * G / 1000);
        let p2 = shard_plan(
            &l3,
            &shard_split(0),
            &shard_hw(128 * G, 62 * G, 2),
            &wl(262_144),
        );
        assert!(!p2.fits);
        let p4 = shard_plan(
            &l3,
            &shard_split(0),
            &shard_hw(256 * G, 62 * G, 4),
            &wl(262_144),
        );
        assert!(p4.fits, "{p4:?}");
        // 500B급 ≈258G — 4장에서도 초과(오프로드 레짐).
        let l5 = shard_layers(64, 4032 * G / 1000);
        assert!(
            !shard_plan(
                &l5,
                &shard_split(0),
                &shard_hw(256 * G, 62 * G, 4),
                &wl(262_144)
            )
            .fits
        );
    }

    /// KV·상태가 카드 층 비중만큼 국소화되고, 경계 케이스를 뒤집을 수 있다.
    #[test]
    fn shard_kv_local_accounting() {
        // 64층×3.2G = 204.8G + 비층 40G = 244.8G, 4×64=256G — KV(256K·33.8KB=8.86G)+
        // 상태(0.6G) 포함 판정에서 경계적.
        let l = shard_layers(64, 32 * G / 10);
        let hw4 = shard_hw(256 * G, 512 * G, 4);
        let p = shard_plan(&l, &shard_split(40 * G), &hw4, &wl(262_144));
        assert_eq!(p.cards, 4);
        // 풀어텐션 층(1/4)은 해당 카드에만 KV 귀속 — KV 합 = 전체.
        let kv_sum: u64 = p.per_card_kv.iter().sum();
        let kv_expect = (33 * 1024u64) * 262_144;
        assert!(kv_sum.abs_diff(kv_expect) <= kv_expect / 100, "kv {kv_sum}");
        // 상태 총량 = state_bytes_per_ctx × contexts(=4) — 전 GDN 카드 합과 일치.
        let st_sum: u64 = p.per_card_state.iter().sum();
        let st_expect = 4 * 151 * (1 << 20);
        assert!(
            st_sum > 0 && st_sum.abs_diff(st_expect) <= st_expect / 20,
            "st {st_sum}"
        );
    }

    /// 층 파싱 — 실명 픽스처(하이브리드 GDN/풀어텐션·expert 가산·embed 분리).
    #[test]
    fn layer_split_parses_real_names() {
        let m = Manifest::from_items(vec![
            (
                "model.language_model.layers.0.linear_attn.A_log".into(),
                100,
            ),
            (
                "model.language_model.layers.0.mlp.experts.0.down_proj.weight_packed".into(),
                900,
            ),
            (
                "model.language_model.layers.1.self_attn.q_proj.weight_packed".into(),
                500,
            ),
            ("model.language_model.embed_tokens.weight".into(), 700),
            ("lm_head.weight".into(), 300),
        ]);
        let sp = m.layer_split();
        assert_eq!(sp.layers.len(), 2);
        assert_eq!(sp.layers[0].bytes, 1000);
        assert!(!sp.layers[0].full_attn);
        assert_eq!(sp.layers[1].bytes, 500);
        assert!(sp.layers[1].full_attn);
        assert_eq!(sp.embed, 700);
        assert_eq!(sp.non_layer, 300);
    }
}
