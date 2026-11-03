//! 伸ばし競争の明示呼出専用。発火操作を含む手数と終了盤を保存する。
use crate::ama_search::{self, Config, Pair};
use crate::bitboard::{board_to_grid, simulate_chain_ama, BitBoard, ChainSimResult};

pub type Point = (usize, usize, i64, i32, Vec<u8>);
const FIRE_LOOKAHEAD: usize = 2;

fn retain(points: &mut Vec<Point>, start: usize, used: usize, value: ChainSimResult) {
    if value.exact_score > 0 {
        points.push((start, start + used, value.exact_score, value.chain_count,
                     board_to_grid(&value.final_board)));
    }
}

/// 非発火経路の各点から0〜2手の発火候補を全列挙。発火後は伸ばさない。
fn fires(board: BitBoard, pairs: &[Pair], start: usize, horizon: usize) -> Vec<Point> {
    let mut result = Vec::new();
    let initial = simulate_chain_ama(&board, true);
    if initial.exact_score > 0 {
        retain(&mut result, start, 0, initial);
        return result;
    }
    if start == horizon { return result; }
    for p in ama_search::placements(&board, pairs[0]) {
        let Some(a) = ama_search::apply(&board, p, pairs[0]) else { continue; };
        if a.exact_score > 0 { retain(&mut result, start, 1, a); continue; }
        if start + 1 == horizon { continue; }
        for q in ama_search::placements(&a.final_board, pairs[1]) {
            if let Some(b) = ama_search::apply(&a.final_board, q, pairs[1]) {
                retain(&mut result, start, FIRE_LOOKAHEAD, b);
            }
        }
    }
    result
}

/// 既存growthと同じama順位の非発火経路。時間期限は設けない。
pub fn curve(mut board: BitBoard, tape: &[Pair], horizon: usize,
             cfg: &Config, visible: usize) -> Vec<Point> {
    let mut result = Vec::new();
    for k in 0..=horizon {
        if k + FIRE_LOOKAHEAD > tape.len() { break; }
        result.extend(fires(board, &tape[k..], k, horizon));
        if k == horizon { break; }
        let end = (k + visible).min(tape.len());
        let ranked = ama_search::multi(board, &tape[k..end], cfg.depth, cfg, 0);
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
    use crate::bitboard::{BOARD_ROWS, BOARD_COLS};
    #[test]
    fn firing_hand_is_inside_horizon_and_keeps_end_board() {
        let mut board = BitBoard::default();
        board.colors[0][0] = 7;
        let points = fires(board, &[(1, 2), (3, 4)], 0, 1);
        assert!(!points.is_empty());
        for point in points {
            assert_eq!(point.1, 1);
            assert!(point.2 > 0);
            assert_eq!(point.4.len(), BOARD_ROWS * BOARD_COLS);
        }
    }

    #[test]
    fn zero_horizon_does_not_place_a_pair() {
        let mut board = BitBoard::default();
        board.colors[0][0] = 7;
        assert!(fires(board, &[(1, 2), (3, 4)], 0, 0).is_empty());
    }

    #[test]
    fn zero_horizon_keeps_an_already_ignited_chain() {
        let mut board = BitBoard::default();
        board.colors[0][..4].fill(1);
        let points = fires(board, &[(1, 2), (3, 4)], 0, 0);
        assert_eq!(points.len(), 1);
        assert_eq!((points[0].0, points[0].1, points[0].3), (0, 0, 1));
        assert!(points[0].4.iter().all(|&c| c == 0));
    }

    #[test]
    fn first_capacity_matches_existing_growth_without_extra_hands() {
        let mut board = BitBoard::default();
        board.colors[0][0] = 3;
        let tape = [(1, 2), (1, 3), (2, 4), (3, 4)];
        let cfg = Config { trigger: ama_search::DEFAULT_TRIGGER, width: 1,
            depth: 2, colors: [1, 2, 3, 4], weight: crate::ama_eval::Weight::BUILD };
        let old = crate::ama_f::growth::curve(board, &tape, 0, &cfg, 3);
        let new = fires(board, &tape, 0, FIRE_LOOKAHEAD);
        let best = new.iter().map(|p| (p.2, p.3)).max().unwrap_or((0, 0));
        assert_eq!((old[0].1, old[0].2), best);
        assert!(new.iter().all(|p| p.1 <= FIRE_LOOKAHEAD));
    }
}
