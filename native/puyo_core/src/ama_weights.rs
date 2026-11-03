//! ama config.json のプロファイルを読み、読み込んだ原文バイトを版の印にする。
use crate::ama_eval::Weight;
use sha2::{Digest, Sha256};

pub const IMPLEMENTATION_COMMIT: &str = "dea210bcd92965ae08fbc311f23565b0fab6dbbb";
const BUNDLED: &[u8] = include_bytes!("../ama_weights.json");
const MAX_WEIGHT: i64 = 10000;

pub struct Loaded {
    pub weight: Weight,
    pub sha256: String,
}

pub fn load(path: Option<&str>, profile: &str) -> Result<Loaded, String> {
    #[cfg(feature = "ama-profile")]
    let _timer = crate::ama_profile_clock::Timer::new(6);
    if !["build", "ac"].contains(&profile) {
        return Err("profileはbuild/acです".into());
    }
    let bytes = match path {
        Some(p) => std::fs::read(p).map_err(|e| format!("重みファイル: {e}"))?,
        None => BUNDLED.to_vec(),
    };
    let value: serde_json::Value = serde_json::from_slice(&bytes).map_err(|e| e.to_string())?;
    // 他プロファイルは原典同様に許容するが、使う2プロファイルは必ず検証する。
    for name in ["build", "ac"] {
        validate(&value[name])?;
    }
    let weight = serde_json::from_value(value[profile].clone()).map_err(|e| e.to_string())?;
    Ok(Loaded { weight, sha256: format!("{:x}", Sha256::digest(&bytes)) })
}

fn validate(value: &serde_json::Value) -> Result<(), String> {
    let fields = value.as_object().ok_or("build/acのオブジェクトが必要です")?;
    let _: Weight = serde_json::from_value(value.clone()).map_err(|e| e.to_string())?;
    for (name, value) in fields {
        let number = value.as_i64().ok_or("重みは整数が必要です")?;
        if !(-MAX_WEIGHT..=MAX_WEIGHT).contains(&number) {
            return Err("重みは-10000..10000が必要です（整数演算の安全範囲）".into());
        }
        if name == "u" && number == 0 {
            continue;
        }
        if !["chain", "y", "key", "chi", "shape", "well", "bump", "form", "link_2",
            "link_3", "waste_14", "side", "nuisance", "tear", "waste"].contains(&name.as_str()) {
            return Err(format!("未対応の重み: {name}。版の移植が必要です"));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn bundled_matches_original_constants() {
        assert_eq!(load(None, "build").unwrap().weight, Weight::BUILD);
        assert_eq!(load(None, "ac").unwrap().weight, Weight::AC);
    }
}
