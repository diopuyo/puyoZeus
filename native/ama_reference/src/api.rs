//! 保存入力と同じ JSON の node/ranked/path/plain 入口。
use crate::{chain, field::{Field, Placement, JSON_COLORS}, order, search::{self, Candidate, Config}};
use serde_json::{json, Value};

fn trace(mut f: Field) -> Value {
    let scores = chain::simulate(&mut f);
    json!({"scores": scores, "board": f.grid(), "dead": f.dead()})
}

fn node(f: Field, pair: [u8; 2]) -> Value {
    let moves: Vec<Value> = f.placements(pair).iter().map(|&p| {
        let placed = f.place(p, pair).unwrap();
        let mut item = trace(placed);
        item["score"] = json!(item["scores"].as_array().unwrap().iter().map(|s| s.as_i64().unwrap()).sum::<i64>());
        item["x"] = json!(p.x); item["r"] = json!(p.r); item["placed"] = json!(placed.grid());
        item
    }).collect();
    json!({"initial": trace(f), "moves": moves})
}

fn path_json(path: &[Placement]) -> Value {
    json!(path.iter().map(|p| [p.x, p.r]).collect::<Vec<_>>())
}

fn replay(mut f: Field, q: &[[u8; 2]], path: &[Placement]) -> Vec<Value> {
    path.iter().zip(q).map(|(&p, &pair)| {
        f = f.place(p, pair).expect("探索道筋の配置失敗");
        let mut item = trace(f);
        item["x"] = json!(p.x); item["r"] = json!(p.r);
        chain::simulate(&mut f);
        item
    }).collect()
}

fn branches_json(m: &Candidate, results: &[search::Result]) -> Vec<Value> {
    results.iter().enumerate().filter_map(|(b, r)| {
        r.candidates.iter().find(|c| c.placement == m.placement).map(|c| {
            json!({"score": c.score, "root_score": c.root_score, "path": path_json(&c.path),
                "branch": b, "search_score": c.search_score})
        })
    }).collect()
}

fn candidate_json(m: &Candidate, results: &[search::Result], f: Field,
    visible: &[[u8; 2]], config: Config, plain: bool) -> Value {
    let branches = branches_json(m, results);
    let mut item = json!({"x": m.placement.x, "r": m.placement.r, "score": m.score,
        "root_score": branches.first().map(|b| b["root_score"].clone()).unwrap_or(json!(0)), "branches": branches});
    if plain && !branches.is_empty() {
        let mut best = 0;
        for b in 1..branches.len() {
            if branches[b]["score"].as_i64() > branches[best]["score"].as_i64() { best = b; }
        }
        let branch = branches[best]["branch"].as_u64().unwrap() as usize;
        let c = results[branch].candidates.iter().find(|c| c.placement == m.placement).unwrap();
        item["representative_branch"] = json!(branch);
        item["steps"] = json!(replay(f, &search::queue(visible, branch, config.depth), &c.path));
    }
    item
}

fn run_search(f: Field, visible: &[[u8; 2]], input: &Value, op: &str, threads: usize) -> Value {
    let config = Config { width: number(input, "width", 250), depth: number(input, "depth", 16),
        trigger: number(input, "trigger", 95000) as i32 };
    let path = op == "path" || op == "plain";
    let count = number(input, "branches", if path { 1 } else { 6 }).min(6);
    let results = search::branches(f, visible, config, count, threads);
    let mut merged: Vec<Candidate> = Vec::new();
    let mut expanded = 0;
    for r in &results {
        if path || !r.candidates.is_empty() { expanded += r.expanded; }
        if r.candidates.is_empty() { continue; }
        if merged.is_empty() { merged = r.candidates.clone(); continue; }
        for c in &mut merged {
            if let Some(next) = r.candidates.iter().find(|n| n.placement == c.placement) { c.score += next.score; }
        }
    }
    order::sort(&mut merged, |a, b| a.score > b.score);
    let candidates: Vec<Value> = merged.iter().take(if op == "plain" { 1 } else { usize::MAX }).map(|c| {
        if path { candidate_json(c, &results, f, visible, config, op == "plain") }
        else { json!([[c.placement.x, c.placement.r], c.score]) }
    }).collect();
    json!({"expanded_nodes": expanded, "candidates": candidates})
}

/// 返しの見積もり用: 本物のツモ (visible に最大 40 組) を 1 枝で探索し、k 手目までに撃てた連鎖の得点の最大
/// (cummax[k-1]) を返す。途中で打ち切らない (trigger 無し)。BAGS の補完は使わない (depth はツモの組数まで)。
fn return_profile(f: Field, queue: &[[u8; 2]], input: &Value) -> Value {
    let depth = number(input, "depth", queue.len()).min(queue.len());
    let config = Config { width: number(input, "width", 250), depth, trigger: i32::MAX };
    let result = search::search(f, queue, config);
    let mut best = 0i64;
    let mut cummax: Vec<i64> = result.layer_max.iter().map(|&m| { best = best.max(m); best }).collect();
    // 置ける手が無く探索が始まらない盤 (窒息寸前) は層が無い。長さを depth にそろえ、それ以上は撃てないとする。
    cummax.resize(depth, best);
    json!({"cummax": cummax, "expanded_nodes": result.expanded})
}

fn number(input: &Value, key: &str, default: usize) -> usize {
    input[key].as_u64().map(|v| v as usize).unwrap_or(default)
}

pub fn request(input: &Value) -> Result<Value, String> {
    request_with_threads(input, 1)
}

pub fn request_with_threads(input: &Value, threads: usize) -> Result<Value, String> {
    if threads == 0 { return Err("threadsは1以上が必要".into()); }
    let board: Vec<u8> = serde_json::from_value(input["board"].clone()).map_err(|e| e.to_string())?;
    let mut f = Field::from_json(&board).map_err(str::to_owned)?;
    let pairs: Vec<[usize; 2]> = serde_json::from_value(input["visible"].clone()).map_err(|e| e.to_string())?;
    let op = input["op"].as_str().unwrap_or("");
    if pairs.is_empty() || pairs.len() > if op == "node" { 3 } else { 40 } { return Err("可視組数不正".into()); }
    if pairs.iter().flatten().any(|&v| !(1..=4).contains(&v)) { return Err("色番号不正".into()); }
    let visible: Vec<_> = pairs.iter().map(|p| [JSON_COLORS[p[0]], JSON_COLORS[p[1]]]).collect();
    if op == "plain" && (visible.len() != 3 || number(input, "branches", 0) != 6) {
        return Err("plainは可視3組・通常補完6本限定".into());
    }
    if op == "node" { return Ok(node(f, visible[0])); }
    if op == "return_profile" {
        if visible.len() < 2 { return Err("return_profileは2組以上が必要".into()); }
        return Ok(return_profile(f, &visible, input));
    }
    let initial = trace(f);
    if op == "plain" { chain::simulate(&mut f); }
    let mut out = run_search(f, &visible, input, op, threads);
    if op == "plain" || op == "path" { out["initial"] = initial; }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn plain_rejects_wrong_queue_and_node_empty_moves() {
        let board = vec![0; 78];
        assert!(request(&json!({"board": board, "visible": [[1, 2]], "op": "plain", "branches": 6})).is_err());
        let r = request(&json!({"board": board, "visible": [[1, 2]], "op": "node"})).unwrap();
        assert_eq!(r["moves"].as_array().unwrap().len(), 22);
    }

    #[test]
    fn return_profile_is_cumulative_and_matches_ranked_best() {
        // 赤 3 個の上に赤を置けば 1 手目で消える盤 (列 0 の下 3 段が赤)
        let mut board = vec![0u8; 78];
        for row in 10..13 { board[row * 6] = 1; }
        let queue = json!([[1, 2], [3, 4], [1, 1], [2, 3], [4, 4], [1, 3]]);
        let r = request(&json!({"board": board, "visible": queue, "op": "return_profile", "width": 50, "depth": 6})).unwrap();
        let cummax: Vec<i64> = r["cummax"].as_array().unwrap().iter().map(|v| v.as_i64().unwrap()).collect();
        assert_eq!(cummax.len(), 6);
        assert!(cummax[0] > 0 && cummax.windows(2).all(|w| w[0] <= w[1]));
        let ranked = request(&json!({"board": board, "visible": queue, "op": "ranked", "branches": 1, "width": 50,
                                     "depth": 6, "trigger": 1_000_000_000})).unwrap();
        let best = ranked["candidates"].as_array().unwrap().iter().map(|c| c[1].as_i64().unwrap()).max().unwrap();
        assert_eq!(*cummax.last().unwrap(), best);
        assert!(request(&json!({"board": board, "visible": [[1, 2]], "op": "return_profile"})).is_err());
    }
}
