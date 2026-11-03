//! v4.4専用。催促と本線の実発火候補を同じ非発火経路上で別々に残す。
use crate::ama_search::{self, Config, Pair};
use crate::bitboard::{simulate_chain, BitBoard, ChainSimResult};
use super::quick::CONSUMPTION_LIMIT;

pub type Point = (usize, i64, i32, i64, i32);
const LOOKAHEAD: usize = 2;

fn material(board: &BitBoard) -> usize {
    board.colors.iter().flatten().map(|c| c.count_ones() as usize).sum()
}

fn retain(result: &mut [(i64, i32); 2], value: ChainSimResult, count: usize) {
    if value.exact_score == 0 { return; }
    let consumed = value.total_erased as f64 / count.max(1) as f64;
    let branch = usize::from(consumed >= CONSUMPTION_LIMIT);
    result[branch] = result[branch].max((value.exact_score, value.chain_count));
}

fn capacity(board: BitBoard, pairs: &[Pair]) -> [(i64, i32); 2] {
    let mut result = [(0, 0); 2];
    let count = material(&board);
    let initial = simulate_chain(&board, true);
    if initial.exact_score > 0 {
        retain(&mut result, initial, count);
        return result;
    }
    for p in ama_search::placements(&board, pairs[0]) {
        if crate::ama_budget::expired() { break; }
        let Some(a) = ama_search::apply(&board, p, pairs[0]) else { continue; };
        retain(&mut result, a, count + 2);
        if a.exact_score > 0 { continue; }
        for q in ama_search::placements(&a.final_board, pairs[1]) {
            if crate::ama_budget::expired() { break; }
            if let Some(b) = ama_search::apply(&a.final_board, q, pairs[1]) {
                retain(&mut result, b, count + 4);
            }
        }
    }
    result
}

pub fn curve(mut board: BitBoard, tape: &[Pair], horizon: usize, cfg: &Config, visible: usize) -> Vec<Point> {
    let mut result = Vec::new();
    for k in 0..=horizon {
        if crate::ama_budget::expired() || k + LOOKAHEAD > tape.len() { break; }
        let values = capacity(board, &tape[k..k + LOOKAHEAD]);
        if crate::ama_budget::expired() { break; }
        result.push((k, values[0].0, values[0].1, values[1].0, values[1].1));
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sixty_percent_is_main() {
        let mut board = BitBoard::default();
        board.colors[0] = [1; 6];
        let value = simulate_chain(&board, true);
        let mut result = [(0, 0); 2];
        retain(&mut result, value, 10);
        assert_eq!(result[0], (0, 0));
        assert!(result[1].0 > 0);
    }

    #[test]
    fn below_sixty_percent_is_provocation() {
        let mut board = BitBoard::default();
        board.colors[0] = [1; 6];
        let value = simulate_chain(&board, true);
        let mut result = [(0, 0); 2];
        retain(&mut result, value, 11);
        assert!(result[0].0 > 0);
        assert_eq!(result[1], (0, 0));
    }
}
