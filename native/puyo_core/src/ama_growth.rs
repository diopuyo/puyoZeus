//! 方式Fの順位で非発火手を積み、各地点の実発火能力だけを測る。
use crate::ama_search::{self, Config, Pair};
use crate::bitboard::{simulate_chain, BitBoard};

pub const FIRE_LOOKAHEAD: usize = 2;
pub type Point = (usize, i64, i32);

fn capacity(board: BitBoard, pairs: &[Pair]) -> (i64, i32) {
    let initial = simulate_chain(&board, true);
    let mut best = (initial.exact_score, initial.chain_count);
    if best.0 > 0 { return best; }
    for p in ama_search::placements(&board, pairs[0]) {
        if crate::ama_budget::expired() { break; }
        let Some(a) = ama_search::apply(&board, p, pairs[0]) else { continue; };
        best = best.max((a.exact_score, a.chain_count));
        if a.exact_score > 0 { continue; }
        for q in ama_search::placements(&a.final_board, pairs[1]) {
            if crate::ama_budget::expired() { break; }
            if let Some(b) = ama_search::apply(&a.final_board, q, pairs[1]) {
                best = best.max((b.exact_score, b.chain_count));
            }
        }
    }
    best
}

/// kは非発火の積み増し数。最後の発火操作0〜2手はkに含めない。
pub fn curve(mut board: BitBoard, tape: &[Pair], horizon: usize, cfg: &Config, visible: usize) -> Vec<Point> {
    let mut result = Vec::new();
    for k in 0..=horizon {
        if crate::ama_budget::expired() || k + FIRE_LOOKAHEAD > tape.len() { break; }
        let (score, chain) = capacity(board, &tape[k..k + FIRE_LOOKAHEAD]);
        if crate::ama_budget::expired() { break; }
        result.push((k, score, chain));
        if k == horizon { break; }
        let end = (k + visible).min(tape.len());
        let ranked = ama_search::multi(board, &tape[k..end], cfg.depth, cfg, 0);
        if crate::ama_budget::expired() { break; }
        let next = ranked.iter().filter_map(|c| ama_search::apply(&board, c.placement, tape[k]))
            .find(|r| r.exact_score == 0);
        let Some(next) = next else { break; };
        board = next.final_board;
    }
    result
}
