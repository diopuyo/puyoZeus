//! 競争実験専用。実発火の段得点・終了盤と非発火経路を返す。
use crate::ama_search::{self, Config, Pair, Placement};
use crate::bitboard::{self, BitBoard, ChainSimResult, BOARD_COLS, COLOR_OJAMA};

pub type Fire = (usize, i32, Vec<i64>, Vec<u8>, i32, usize);
pub type State = (usize, Vec<u8>, i32);
const LOOKAHEAD: usize = 2;
const DROP_LIMIT: i32 = 30;

fn placed(board: &BitBoard, (x, r): Placement, pair: Pair) -> Option<BitBoard> {
    let (a, b, y) = match r {
        0 => (pair.0, pair.1, x), 1 => (pair.0, pair.1, x + 1),
        2 => (pair.1, pair.0, x), _ => (pair.0, pair.1, x.checked_sub(1)?),
    };
    bitboard::drop_one(&bitboard::drop_one(board, x, a)?, y, b)
}

/// ama_f::settleと同じ順序で均等落下。窒息盤は次の非発火手へ進めない。
fn land(mut board: BitBoard, pending: i32) -> (BitBoard, i32) {
    let count = pending.min(DROP_LIMIT);
    for i in 0..count as usize {
        if let Some(next) = bitboard::drop_one(&board, i % BOARD_COLS, COLOR_OJAMA) {
            board = next;
        }
    }
    (board, pending - count)
}

fn dead(board: &BitBoard) -> bool {
    bitboard::height_of(board, bitboard::DEATH_COL) > bitboard::DEATH_BIT_INDEX
}

fn retain(out: &mut Vec<Fire>, hand: usize, raw: BitBoard, sim: ChainSimResult, pending: i32, source: usize) {
    if sim.exact_score == 0 { return; }
    let scores = bitboard::simulate_chain_trace(&raw, true).iter().map(|s| s.4).collect();
    out.push((hand, sim.chain_count, scores, bitboard::board_to_grid(&sim.final_board), pending, source));
}

fn fires(board: BitBoard, tape: &[Pair], k: usize, limit: usize, pending: i32) -> Vec<Fire> {
    let mut out = Vec::new();
    let initial = bitboard::simulate_chain_ama(&board, true);
    if initial.exact_score > 0 {
        retain(&mut out, k, board, initial, pending, k);
        return out;
    }
    if k == limit || dead(&board) { return out; }
    for p in ama_search::placements(&board, tape[0]) {
        let Some(a) = ama_search::apply(&board, p, tape[0]) else { continue; };
        if a.exact_score > 0 {
            retain(&mut out, k + 1, placed(&board, p, tape[0]).unwrap(), a, pending, k);
            continue;
        }
        if k + 1 == limit { continue; }
        let (after, remaining) = land(a.final_board, pending);
        if dead(&after) { continue; }
        for q in ama_search::placements(&after, tape[1]) {
            if let Some(b) = ama_search::apply(&after, q, tape[1]) {
                retain(&mut out, k + LOOKAHEAD, placed(&after, q, tape[1]).unwrap(), b, remaining, k);
            }
        }
    }
    out
}

pub fn curve(mut board: BitBoard, tape: &[Pair], horizon: usize, cfg: &Config,
             visible: usize, mut pending: i32) -> (Vec<Fire>, Vec<State>) {
    let (mut result, mut states) = (Vec::new(), Vec::new());
    for k in 0..=horizon {
        if k + LOOKAHEAD > tape.len() { break; }
        states.push((k, bitboard::board_to_grid(&board), pending));
        result.extend(fires(board, &tape[k..], k, horizon, pending));
        if k == horizon || dead(&board) { break; }
        let end = (k + visible).min(tape.len());
        let ranked = ama_search::multi(board, &tape[k..end], cfg.depth, cfg, 0);
        let next = ranked.iter().filter_map(|c| ama_search::apply(&board, c.placement, tape[k]))
            .find(|r| r.exact_score == 0);
        let Some(next) = next else { break; };
        (board, pending) = land(next.final_board, pending);
    }
    (result, states)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn drop_is_capped_and_remainder_is_kept() {
        let (b, left) = land(BitBoard::default(), 37);
        assert_eq!(left, 7);
        assert!(b.ojama.iter().all(|p| p.count_ones() == 5));
    }
    #[test]
    fn stages_and_final_board_match_native_trace() {
        let mut b = BitBoard::default();
        b.colors[0][..4].fill(1);
        let v = fires(b, &[(1, 2), (3, 4)], 0, 0, 37);
        assert_eq!(v.len(), 1);
        assert_eq!((v[0].0, v[0].1, v[0].4), (0, 1, 37));
        assert_eq!(v[0].2, vec![40]);
        assert!(v[0].3.iter().all(|&c| c == 0));
    }
}
