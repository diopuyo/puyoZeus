//! 発火起点の2手補完。Python版と同じ列挙順・多重度を保持する。
use crate::bitboard::{board_to_grid, board_from_grid, b3_origin_completion, enumerate_placements,
    is_dead, simulate_chain, BitBoard, BOARD_COLS, BOARD_ROWS, NUM_TRACKED_COLORS};
use rayon::prelude::*;
use std::collections::HashMap;
use std::sync::OnceLock;

const WORKERS: usize = 3;
type Key = ([[u16; BOARD_COLS]; NUM_TRACKED_COLORS], [u16; BOARD_COLS]);
type Weighted = Vec<(BitBoard, usize)>;
static POOL: OnceLock<rayon::ThreadPool> = OnceLock::new();

pub type Completion = (Vec<i64>, Vec<Vec<u8>>, i64, Vec<u8>, usize);

pub fn completions(board: BitBoard, pairs: &[(u8, u8)], hidden: bool,
                   parallel: bool) -> (Vec<Completion>, usize) {
    let (rows, trials) = candidates(board, pairs, hidden, parallel);
    let complete = |(grid, weight): &(Vec<u8>, usize)| {
        let mut arr = [0; BOARD_ROWS * BOARD_COLS];
        arr.copy_from_slice(grid);
        let (prefix, colors, score, final_grid) = b3_origin_completion(&board_from_grid(&arr), hidden);
        (prefix, colors, score, final_grid, *weight)
    };
    let values = if parallel {
        POOL.get().expect("候補列挙時に生成済み").install(|| rows.par_iter().map(complete).collect())
    } else { rows.iter().map(complete).collect() };
    (values, trials)
}

fn insert(rows: &mut Weighted, index: &mut HashMap<Key, usize>, board: BitBoard, weight: usize) {
    let key = (board.colors, board.ojama);
    if let Some(&i) = index.get(&key) { rows[i].1 += weight; }
    else { index.insert(key, rows.len()); rows.push((board, weight)); }
}

fn expand(board: &BitBoard, pairs: &[(u8, u8)], hidden: bool) -> Vec<(BitBoard, bool, bool)> {
    pairs.iter().flat_map(|&pair| enumerate_placements(board, pair, false))
        .map(|(_, placed)| {
            let fires = simulate_chain(&placed, hidden).chain_count > 0;
            (placed, fires, is_dead(&placed))
        }).collect()
}

pub fn candidates(board: BitBoard, pairs: &[(u8, u8)], hidden: bool,
                  parallel: bool) -> (Vec<(Vec<u8>, usize)>, usize) {
    let (mut options, mut frontier) = (Vec::new(), Vec::new());
    let (mut oi, mut fi) = (HashMap::new(), HashMap::new());
    let mut trials = 0;
    for (placed, fires, dead) in expand(&board, pairs, hidden) {
        trials += 1;
        if fires { insert(&mut options, &mut oi, placed, 1); }
        else if !dead { insert(&mut frontier, &mut fi, placed, 1); }
    }
    // indexed par_iterのcollectで入力順を維持し、併合と同点処理は直列で行う。
    let expand_one = |(first, weight): &(BitBoard, usize)| (expand(first, pairs, hidden), *weight);
    let expanded: Vec<_> = if parallel {
        POOL.get_or_init(|| rayon::ThreadPoolBuilder::new().num_threads(WORKERS)
            .build().expect("発火起点探索プール"))
            .install(|| frontier.par_iter().map(expand_one).collect())
    } else { frontier.iter().map(expand_one).collect() };
    for (rows, weight) in expanded {
        for (placed, fires, _) in rows {
            trials += weight;
            if fires { insert(&mut options, &mut oi, placed, weight); }
        }
    }
    (options.into_iter().map(|(b, w)| (board_to_grid(&b).to_vec(), w)).collect(), trials)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn parallel_preserves_order_and_weights() {
        let pairs = vec![(1, 1), (1, 2), (2, 2)];
        for hidden in [false, true] {
            let serial = candidates(BitBoard::default(), &pairs, hidden, false);
            assert_eq!(serial, candidates(BitBoard::default(), &pairs, hidden, true));
            assert!(!serial.0.is_empty());
            assert!(serial.0.iter().any(|(_, weight)| *weight > 1));
        }
    }

    #[test]
    fn completion_packs_preexisting_gaps_like_python() {
        let mut grid = [0; BOARD_ROWS * BOARD_COLS];
        for row in BOARD_ROWS - 4..BOARD_ROWS { grid[row * BOARD_COLS] = 1; }
        grid[4 * BOARD_COLS + 1] = 2;
        let (prefix, colors, score, final_grid) = b3_origin_completion(&board_from_grid(&grid), true);
        assert_eq!((prefix, colors, score), (vec![40], vec![vec![1]], 40));
        assert_eq!(final_grid[(BOARD_ROWS - 1) * BOARD_COLS + 1], 2);
        assert_eq!(final_grid.iter().filter(|&&v| v != 0).count(), 1);
    }
}
