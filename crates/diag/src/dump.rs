
#[derive(Debug, Default)]
pub struct DumpOpts {
    pub top2: bool,
    keys: std::collections::HashSet<String>,
}

impl DumpOpts {
    pub fn key(&self, k: &str) -> bool {
        self.keys.contains(k)
    }
}

static OPTS: std::sync::LazyLock<DumpOpts> = std::sync::LazyLock::new(|| {
    let mut o = DumpOpts::default();
    let Some(v) = std::env::var("LLM170_DUMP").ok() else {
        return o;
    };
    for key in v.split(',') {
        let key = key.trim();
        if key.is_empty() {
            continue;
        }
        if key == "top2" {
            o.top2 = true;
        }
        o.keys.insert(key.to_string());
    }
    o
});

pub fn opts() -> &'static DumpOpts {
    &OPTS
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_off() {
        if std::env::var_os("LLM170_DUMP").is_none() {
            assert!(!opts().top2);
        }
    }
}
