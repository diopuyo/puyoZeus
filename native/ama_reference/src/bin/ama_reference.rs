use std::io::{self, BufRead};
use serde_json::{json, Value};

/// 枝の並列数は環境変数 AMA_THREADS (既定 1、配信の pyd は呼び出し側が渡す)。
fn threads() -> usize {
    std::env::var("AMA_THREADS").ok().and_then(|v| v.parse().ok()).unwrap_or(1)
}

fn main() {
    let threads = threads();
    for line in io::stdin().lock().lines() {
        let out = line.map_err(|e| e.to_string()).and_then(|s| {
            serde_json::from_str::<Value>(&s).map_err(|e| e.to_string())
        }).and_then(|v| ama_reference::api::request_with_threads(&v, threads));
        println!("{}", out.unwrap_or_else(|e| json!({"error": e})));
    }
}
