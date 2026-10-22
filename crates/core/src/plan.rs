
use std::path::Path;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Category {
    Dense,
    Experts,
    RowSparse,
    Embed,
}

pub fn classify(name: &str) -> Category {
    if name.contains("shared_expert") {
        return Category::Dense;
    }
    if name.contains(".experts.") {
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

pub struct Manifest {
    items: Vec<(String, u64)>,
}

impl Manifest {
    pub fn from_items(mut items: Vec<(String, u64)>) -> Self {
        items.sort();
        Self { items }
    }

    pub fn from_dir(dir: &Path) -> Result<Self, String> {
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

#[derive(Clone, Copy, Debug)]
pub struct Hardware {
    pub vram_total: u64,
    pub vram_free: Option<u64>,
    pub host_ram: u64,
    pub ssd: Option<u64>,
    pub reserve: u64,
    pub cards: usize,
}

#[derive(Clone, Copy, Debug)]
pub struct Workload {
    pub model_max_ctx: usize,
    pub want_ctx: usize,
    pub kv_bytes_per_token: u64,
    pub state_bytes_per_ctx: u64,
    pub contexts: usize,
    pub embed_resident: bool,
}

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

#[derive(Clone, Debug)]
pub struct Plan {
    pub sizes: Sizes,
    pub vram: u64,
    pub reserve: u64,
    pub kv_budget: u64,
    pub effective_ctx: usize,
    pub goal_ctx: usize,
    pub ctx_capped: bool,
    pub experts_tier: Tier,
    pub experts_hot: u64,
    pub row_sparse_tier: Tier,
    pub host_ram_need: u64,
    pub ssd_need: u64,
    pub kv_host_budget: u64,
    pub refuse: Option<&'static str>,
}

impl Plan {
    pub fn fits(&self) -> bool {
        self.refuse.is_none()
    }

    pub fn fully_resident(&self) -> bool {
        self.refuse.is_none() && self.experts_tier == Tier::Vram
    }

    pub fn summary(&self) -> String {
        if let Some(r) = self.refuse {
            return format!("거부: {r}");
        }
        let gib = |b: u64| b as f64 / (1u64 << 30) as f64;
        format!(
            "dense={:.1}G experts={:.1}G row_sparse={:.1}G embed={:.1}G · vram={:.1}G kv_budget={:.1}G(+host {:.1}G) ctx={}/{} ({}) · experts={}(hot {:.1}G) ple={} · host_ram_need={:.1}G ssd_need={:.1}G",
            gib(self.sizes.dense),
            gib(self.sizes.experts),
            gib(self.sizes.row_sparse),
            gib(self.sizes.embed),
            gib(self.vram),
            gib(self.kv_budget),
            gib(self.kv_host_budget),
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


#[derive(Clone, Copy, Default, Debug)]
pub struct LayerInfo {
    pub bytes: u64,
    pub row_sparse: u64,
    pub full_attn: bool,
}

impl LayerInfo {
    pub fn resident(&self) -> u64 {
        self.bytes.saturating_sub(self.row_sparse)
    }
}

#[derive(Clone, Debug, Default)]
pub struct LayerSplit {
    pub layers: Vec<LayerInfo>,
    pub non_layer: u64,
    pub embed: u64,
}

impl Manifest {
    pub fn layer_split(&self) -> LayerSplit {
        let layer_of = |name: &str| -> Option<usize> {
            if name.contains("vision_tower.")
                || name.contains("embed_vision.")
                || name.contains("audio_tower.")
                || name.contains(".visual.")
            {
                return None;
            }
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
                    if classify(name) == Category::RowSparse {
                        layers[li].row_sparse = layers[li].row_sparse.saturating_add(*bytes);
                    }
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

#[derive(Clone, Debug)]
pub struct ShardPlan {
    pub cards: usize,
    pub blocks: Vec<(usize, usize)>,
    pub per_card_weights: Vec<u64>,
    pub per_card_kv: Vec<u64>,
    pub per_card_state: Vec<u64>,
    pub per_card_need: Vec<u64>,
    pub per_card_cap: u64,
    pub worst_card: usize,
    pub fits: bool,
    pub notes: Vec<String>,
}

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
    let weights_total: u64 = layers.iter().map(|l| l.resident()).sum::<u64>() + split.non_layer;
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
                && per_card_weights[card] + l.resident() > target
            {
                blocks.push((start, i));
                card += 1;
                start = i;
            }
            per_card_weights[card] = per_card_weights[card].saturating_add(l.resident());
        }
        blocks.push((start, layers.len()));
        while blocks.len() < cards {
            let end = blocks.last().map(|b| b.1).unwrap_or(0);
            blocks.push((end, end));
        }
    }
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

pub fn plan(m: &Manifest, hw: &Hardware, wl: &Workload) -> Plan {
    let s = m.sums();
    let v = hw.vram_free.unwrap_or(hw.vram_total);
    let x = hw.reserve;
    let mut refuse = None;
    let embed_vram = if wl.embed_resident { s.embed } else { 0 };
    let embed_host = s.embed.saturating_sub(embed_vram);

    if s.dense.saturating_add(embed_vram).saturating_add(x) > v {
        refuse = Some("dense+embed 상주+예약 > 가용 VRAM");
    }
    let state = (wl.contexts.saturating_sub(1)) as u64 * wl.state_bytes_per_ctx;
    let kv_budget = v
        .saturating_sub(s.dense)
        .saturating_sub(embed_vram)
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

    let experts_hot = s.experts.min(spare);
    let experts_tier = if s.experts <= spare {
        Tier::Vram
    } else if hw.host_ram >= s.experts.saturating_add(embed_host) {
        Tier::HostRam
    } else {
        Tier::Ssd
    };
    let leftover = spare.saturating_sub(experts_hot);

    let row_sparse_tier = if s.row_sparse <= leftover {
        Tier::Vram
    } else {
        Tier::Ssd
    };

    let streamed = s.experts.saturating_sub(experts_hot);
    let host_ram_need = streamed.saturating_add(embed_host);
    let mut ssd_need = if row_sparse_tier == Tier::Ssd {
        s.row_sparse
    } else {
        0
    };
    if experts_tier == Tier::Ssd {
        ssd_need = ssd_need.saturating_add(s.experts);
    }
    let kv_host_budget = hw.host_ram.saturating_sub(host_ram_need);

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
        kv_host_budget,
        refuse,
    }
}

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

    #[test]
    fn classify_real_names() {
        let cases: &[(&str, Category)] = &[
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
            (
                "model.language_model.layers.0.experts.0.down_proj.weight_packed",
                Category::Experts,
            ),
            (
                "model.language_model.layers.29.experts.127.up_proj.weight_scale",
                Category::Experts,
            ),
            (
                "model.vision_tower.encoder.layers.0.mlp.gate_proj.linear.weight",
                Category::Dense,
            ),
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
            reserve: 5 * G / 2,
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
            embed_resident: false,
        }
    }

    #[test]
    fn embed_resident_policy() {
        let man = Manifest::from_items(vec![
            ("blk.0.attn_q.weight".to_string(), 10 << 30),
            (
                "model.language_model.embed_tokens.weight".to_string(),
                2 << 30,
            ),
        ]);
        let hw = Hardware {
            vram_total: 32 << 30,
            vram_free: Some(32 << 30),
            host_ram: 64 << 30,
            ssd: None,
            reserve: 2 << 30,
            cards: 1,
        };
        let mut wl = wl(262_144);
        wl.embed_resident = false;
        let p_host = plan(&man, &hw, &wl);
        wl.embed_resident = true;
        let p_res = plan(&man, &hw, &wl);
        assert_eq!(
            p_host.kv_budget - p_res.kv_budget,
            2 << 30,
            "embed 상주 = KV 예산에서 2G 차감(VRAM 몫 이동)"
        );
        assert_eq!(
            p_host.host_ram_need - p_res.host_ram_need,
            2 << 30,
            "호스트 필요분에서 제거"
        );
        assert_eq!(
            p_res.kv_host_budget,
            hw.host_ram - p_res.host_ram_need,
            "호스트 KV 여유 = 호스트 − 필요분"
        );
        let tight = Hardware {
            vram_total: 13 << 30,
            vram_free: Some(13 << 30),
            ..hw
        };
        assert!(
            plan(&man, &tight, &wl).refuse.is_some(),
            "dense+embed+예약 > VRAM"
        );
        wl.embed_resident = false;
        assert!(plan(&man, &tight, &wl).fits(), "비상주면 host로 통과");
    }

    #[test]
    fn dry_run_ladder_examples() {
        let p = plan(&man(12 * G, 40 * G, 0), &hw(128 * G, 62 * G), &wl(262_144));
        assert!(p.fully_resident(), "{p:?}");
        assert_eq!(p.effective_ctx, 262_144);

        let p = plan(&man(14 * G, 90 * G, 0), &hw(128 * G, 62 * G), &wl(262_144));
        assert!(p.fully_resident(), "{p:?}");

        let m300 = man(18 * G, 138 * G, 0);
        let p2 = plan(&m300, &hw(128 * G, 62 * G), &wl(262_144));
        assert!(!p2.fully_resident());
        assert_eq!(p2.experts_tier, Tier::Ssd, "{p2:?}");
        assert!(p2.experts_hot > 0);
        let p4 = plan(&m300, &hw(256 * G, 62 * G), &wl(262_144));
        assert!(p4.fully_resident(), "{p4:?}");

        let p = plan(
            &man(25 * G, 235 * G, 0),
            &hw(256 * G, 512 * G),
            &wl(262_144),
        );
        assert!(!p.fully_resident());
        assert_eq!(p.experts_tier, Tier::HostRam, "{p:?}");
        assert_eq!(p.ssd_need, 0);
        assert_eq!(p.host_ram_need, 235 * G - p.experts_hot);
    }

    #[test]
    fn experts_fall_to_ssd_without_host_ram() {
        let p = plan(&man(18 * G, 138 * G, 0), &hw(128 * G, 8 * G), &wl(262_144));
        assert_eq!(p.experts_tier, Tier::Ssd, "{p:?}");
        assert!(p.ssd_need >= 138 * G);
    }

    #[test]
    fn refuse_when_dense_exceeds() {
        let p = plan(&man(130 * G, 10 * G, 0), &hw(128 * G, 0), &wl(262_144));
        assert!(!p.fits());
        assert!(p.refuse.is_some());
    }

    #[test]
    fn effective_ctx_is_pool_derived() {
        let mut w = wl(262_144);
        w.kv_bytes_per_token = 200 * 1024;
        let p = plan(&man(10 * G, 0, 0), &hw(24 * G, 62 * G), &w);
        assert!(p.effective_ctx < 262_144);
        assert!(p.ctx_capped);
        assert_eq!(p.effective_ctx % 512, 0);
        let p = plan(&man(10 * G, 0, 0), &hw(24 * G, 62 * G), &wl(1 << 20));
        assert_eq!(p.effective_ctx, 262_144);
        assert!(!p.ctx_capped);
    }

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

    #[test]
    fn experts_boundary_exact_fit() {
        let w = wl(512);
        let base = plan(&man(10 * G, 0, 0), &hw(128 * G, 512 * G), &w);
        let spare = base.kv_budget - base.effective_ctx as u64 * w.kv_bytes_per_token;
        let p = plan(&man(10 * G, spare, 0), &hw(128 * G, 512 * G), &w);
        assert!(p.fully_resident(), "{p:?}");
        let p = plan(&man(10 * G, spare + 1, 0), &hw(128 * G, 512 * G), &w);
        assert_eq!(p.experts_tier, Tier::HostRam, "{p:?}");
    }

    #[test]
    fn row_sparse_default_ssd() {
        let p = plan(&man(10 * G, 0, 48 * G), &hw(56 * G, 62 * G), &wl(512));
        assert_eq!(p.row_sparse_tier, Tier::Ssd);
        assert!(p.ssd_need >= 48 * G);
        let p = plan(&man(10 * G, 0, 8 * G), &hw(128 * G, 62 * G), &wl(512));
        assert_eq!(p.row_sparse_tier, Tier::Vram);
    }

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

    fn shard_layers(n: usize, bytes: u64) -> Vec<LayerInfo> {
        (0..n)
            .map(|i| LayerInfo {
                bytes,
                row_sparse: 0,
                full_attn: i % 4 == 3,
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

    #[test]
    fn shard_dry_run_examples() {
        let l = shard_layers(64, 812 * G / 1000);
        let p = shard_plan(
            &l,
            &shard_split(0),
            &shard_hw(128 * G, 62 * G, 2),
            &wl(262_144),
        );
        assert_eq!(p.cards, 2);
        assert!(p.fits, "{p:?}");
        let (mx, mn) = (
            *p.per_card_weights.iter().max().unwrap(),
            *p.per_card_weights.iter().min().unwrap(),
        );
        assert!(mx * 100 <= mn * 115, "불균형 {mx} vs {mn}");
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

    #[test]
    fn shard_kv_local_accounting() {
        let l = shard_layers(64, 32 * G / 10);
        let hw4 = shard_hw(256 * G, 512 * G, 4);
        let p = shard_plan(&l, &shard_split(40 * G), &hw4, &wl(262_144));
        assert_eq!(p.cards, 4);
        let kv_sum: u64 = p.per_card_kv.iter().sum();
        let kv_expect = (33 * 1024u64) * 262_144;
        assert!(kv_sum.abs_diff(kv_expect) <= kv_expect / 100, "kv {kv_sum}");
        let st_sum: u64 = p.per_card_state.iter().sum();
        let st_expect = 4 * 151 * (1 << 20);
        assert!(
            st_sum > 0 && st_sum.abs_diff(st_expect) <= st_expect / 20,
            "st {st_sum}"
        );
    }

    #[test]
    fn shard_excludes_row_sparse() {
        let mut l = shard_layers(8, G);
        l[1].bytes += 40 * G;
        l[1].row_sparse = 40 * G;
        let p = shard_plan(&l, &shard_split(0), &shard_hw(64 * G, 62 * G, 1), &wl(4096));
        assert!(p.fits, "{p:?}");
        assert!(p.per_card_weights[0] <= 9 * G, "{:?}", p.per_card_weights);
    }

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

    #[test]
    fn layer_split_parses_gemma4_names() {
        let m = Manifest::from_items(vec![
            (
                "model.language_model.layers.0.experts.0.gate_proj.weight_packed".into(),
                800,
            ),
            (
                "model.language_model.layers.0.self_attn.q_proj.weight_packed".into(),
                200,
            ),
            (
                "model.language_model.layers.5.self_attn.q_proj.weight_packed".into(),
                300,
            ),
            ("model.language_model.embed_tokens.weight".into(), 700),
            ("model.language_model.norm.weight".into(), 50),
            (
                "model.vision_tower.encoder.layers.0.input_layernorm.weight".into(),
                111,
            ),
            ("model.embed_vision.embedding_projection.weight".into(), 60),
        ]);
        let sp = m.layer_split();
        assert_eq!(sp.layers.len(), 6);
        assert_eq!(sp.layers[0].bytes, 1000, "전문가(800)+어텐션(200)");
        assert_eq!(sp.layers[1].bytes, 0);
        assert_eq!(sp.layers[5].bytes, 300);
        assert_eq!(sp.embed, 700);
        assert_eq!(sp.non_layer, 50 + 111 + 60);
    }
}
