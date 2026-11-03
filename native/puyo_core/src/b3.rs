//! B3専用。1手=2個組、未発火候補を潜在火力で保持する。
use crate::bitboard::{board_to_grid, drop_one, enumerate_placements, is_dead,
    simulate_chain_b3, BitBoard, BOARD_COLS, BOARD_ROWS, NUM_TRACKED_COLORS};
use std::collections::HashMap;

const MATCH_COLORS: u32 = 4;
const VISIBLE_MASK: u16 = (1 << (BOARD_ROWS - 1)) - 1;

/// 隠し段を除いた消去集合の有無。既存get_mask_popと同じ連結式。
#[cfg(feature = "b3-fast-prefilter")]
fn can_pop(plane: &[u16; BOARD_COLS]) -> bool {
    const COLUMN_BITS: usize = 16;
    let m = plane.iter().enumerate().fold(0u128, |v, (x, &p)|
        v | (((p & VISIBLE_MASK) as u128) << (x * COLUMN_BITS)));
    let (u, d, l, r) = (m & (m >> 1), m & (m << 1),
        m & (m << COLUMN_BITS), m & (m >> COLUMN_BITS));
    let three = (u & d & (l | r)) | (l & r & (u | d));
    let two = (u & d) | (l & r) | ((u | d) & (l | r));
    three != 0 || (two & ((two >> 1) | (two << 1)
        | (two << COLUMN_BITS) | (two >> COLUMN_BITS))) != 0
}

fn existing_pop(board: &BitBoard) -> bool {
    #[cfg(feature = "b3-fast-prefilter")]
    { board.colors.iter().any(can_pop) }
    #[cfg(not(feature = "b3-fast-prefilter"))]
    { let _ = board; true }
}

/// 任意の初期盤面には既存の4連結があり得るため、その場合は省略しない。
fn simulate_placed(original: &BitBoard, placed: &BitBoard, existing: bool)
    -> crate::bitboard::ChainSimResult {
    #[cfg(feature = "b3-fast-prefilter")]
    if !existing && !placed.colors.iter().zip(&original.colors)
        .any(|(after, before)| after != before && can_pop(after)) {
        return crate::bitboard::ChainSimResult { chain_count: 0, total_erased: 0,
            total_ojama: 0, score_approx: 0, exact_score: 0, final_board: *placed };
    }
    let _ = (original, existing);
    simulate_chain_b3(placed)
}

#[derive(Clone)]
struct Node { board: BitBoard, mask: u8, shot: i64, potential: i64, links: u32, build: bool }

// 各bitは「その色を除いた4色集合」が候補として残ることを示す。
// 同一盤面へ到達した経路の候補集合をORすれば、5色混用せず1盤面に併合できる。
fn compatible(mask: u8, pair: (u8, u8)) -> u8 {
    mask & !(1 << (pair.0 - 1)) & !(1 << (pair.1 - 1))
}

fn color_mask(board: &BitBoard) -> u8 {
    board.colors.iter().enumerate().fold(0, |m, (i, p)|
        if p.iter().any(|&v| v != 0) { m | (1 << i) } else { m })
}

fn links(board: &BitBoard) -> u32 {
    board.colors.iter().map(|p| (0..BOARD_COLS).map(|c| {
        let v = p[c] & VISIBLE_MASK;
        (v & (v >> 1)).count_ones()
            + if c + 1 < BOARD_COLS { (v & p[c + 1]).count_ones() } else { 0 }
    }).sum::<u32>()).sum()
}

fn pairs(queue: &[u8], hand: usize, colors: &[u8]) -> Vec<(u8, u8)> {
    let a = queue.get(hand * 2).copied().unwrap_or(0);
    let b = queue.get(hand * 2 + 1).copied().unwrap_or(0);
    let mut out = Vec::new();
    for &x in colors { for &y in colors {
        if (a == 0 || a == x) && (b == 0 || b == y)
            && (a != 0 || b != 0 || x <= y) { out.push((x, y)); }
    }}
    out
}

fn expand(frontier: &[Node], pairs: &[(u8, u8)], _resolve: bool,
          best: &mut (i64, i32)) -> Vec<Node> {
    let mut unique = HashMap::new();
    let mut out: Vec<Node> = Vec::new();
    for node in frontier {
      let existing = existing_pop(&node.board);
      for &pair in pairs {
        let mask = compatible(node.mask, pair);
        if mask == 0 { continue; }
        // 互換引数の値によらず、配置→全連鎖解決→窒息判定の順を守る。
        for (_, placed) in enumerate_placements(&node.board, pair, false) {
            let sim = simulate_placed(&node.board, &placed, existing);
            if is_dead(&sim.final_board) { continue; }
            if sim.exact_score > best.0 { *best = (sim.exact_score, sim.chain_count); }
            let key = (sim.final_board.colors, sim.final_board.ojama);
            if let Some(&idx) = unique.get(&key) {
                let old: &mut Node = &mut out[idx];
                old.shot = old.shot.max(sim.exact_score);
                old.mask |= mask;
                old.build |= sim.exact_score == 0;
            } else {
                unique.insert(key, out.len());
                out.push(Node { board: sim.final_board, mask, shot: sim.exact_score,
                    potential: 0, links: links(&sim.final_board), build: sim.exact_score == 0 });
            }
        }
    }}
    out
}

// 既存max_chain_after_dropsと同じ1個追加の連鎖試行。生得点で比較する。
fn potential(node: &Node, colors: &[u8]) -> i64 {
    let mut best = 0;
    let existing = existing_pop(&node.board);
    for &color in colors {
        if compatible(node.mask, (color, color)) == 0 { continue; }
        for col in 0..BOARD_COLS {
            if let Some(b) = drop_one(&node.board, col, color) {
                let sim = simulate_placed(&node.board, &b, existing);
                if !is_dead(&sim.final_board) { best = best.max(sim.exact_score); }
            }
        }
    }
    best
}

fn select(mut nodes: Vec<Node>, colors: &[u8], width: usize, shortlist: usize) -> Vec<Node> {
    // shortlist=0は全候補評価。軽量版でも積み・発火の両枠を予備選抜する。
    if shortlist > 0 && nodes.len() > width * shortlist {
        nodes.sort_by_key(|n| std::cmp::Reverse((n.links, n.shot)));
        let build = nodes.iter().find(|n| n.build).cloned();
        let mut fires: Vec<Node> = nodes.iter().filter(|n| n.shot > 0).cloned().collect();
        fires.sort_by_key(|n| std::cmp::Reverse(n.shot));
        nodes.truncate(width * shortlist);
        if !nodes.iter().any(|n| n.build) {
            if let Some(n) = build { nodes.push(n); }
        }
        for n in fires.into_iter().take(width) {
            if !nodes.iter().any(|x| x.board == n.board && x.mask == n.mask) { nodes.push(n); }
        }
    }
    for n in &mut nodes { n.potential = potential(n, colors); }
    nodes.sort_by_key(|n| std::cmp::Reverse((n.potential, n.links, n.shot)));
    // 未発火と発火残骸を最低1枠ずつ残す。Qは選抜前の全発火の最大。
    let build = nodes.iter().find(|n| n.build).cloned();
    let fire = nodes.iter().filter(|n| n.shot > 0).max_by_key(|n| n.shot).cloned();
    let mut selected = Vec::new();
    if let Some(n) = build { selected.push(n); }
    if width > 1 { if let Some(n) = fire {
        if !selected.iter().any(|x| x.board == n.board) { selected.push(n); }
    } }
    for n in nodes {
        if selected.len() == width { break; }
        if !selected.iter().any(|x| x.board == n.board && x.mask == n.mask) { selected.push(n); }
    }
    selected
}

pub type SearchOutput = (Vec<i64>, Vec<i32>, usize, bool, bool, usize);

pub fn search(board: BitBoard, queue: &[u8], colors: &[u8], hmax: usize,
              width: usize, resolve: bool, shortlist: usize, observed: &[u8]) -> SearchOutput {
    let mut mask = color_mask(&board);
    for &c in observed { mask |= 1 << (c - 1); }
    for &c in queue { if c > 0 { mask |= 1 << (c - 1); } }
    if colors.len() <= MATCH_COLORS as usize { for &c in colors { mask |= 1 << (c - 1); } }
    let palettes = ((1 << NUM_TRACKED_COLORS) - 1) & !mask;
    let mut frontier = vec![Node { board, mask: palettes, shot: 0, potential: 0, links: 0, build: true }];
    let (mut scores, mut chains) = (Vec::new(), Vec::new());
    let mut best = (0, 0);
    let mut died = false;
    for hand in 0..hmax {
        let children = expand(&frontier, &pairs(queue, hand, colors), resolve, &mut best);
        if children.is_empty() { died = true; break; }
        frontier = select(children, colors, width, shortlist);
        scores.push(best.0); chains.push(best.1);
    }
    let done = scores.len();
    let full = died || expand(&frontier, &pairs(queue, done, colors), resolve, &mut (0, 0)).is_empty();
    let empty = frontier.iter().map(|n| board_to_grid(&n.board).iter().filter(|&&c| c == 0).count()).max().unwrap_or(0);
    scores.resize(hmax, best.0); chains.resize(hmax, best.1);
    (scores, chains, done, died, full, empty)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bitboard::{board_from_grid, simulate_chain, simulate_chain_with_steps};

    const RED: u8 = 1;
    const BLUE: u8 = 2;
    const OJAMA: u8 = 9;
    const DEATH_COL: usize = 2;
    const RESCUE_SCORE: i64 = 100;

    #[cfg(feature = "b3-fast-prefilter")]
    #[test]
    fn prefilter_matches_arbitrary_and_settled_boards() {
        const CASES: usize = 4096;
        let mut seed = 0x123456789abcdefu64;
        for _ in 0..CASES {
            let grid: Vec<u8> = (0..BOARD_ROWS * BOARD_COLS).map(|_| {
                seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17;
                match seed % 8 { 6 => OJAMA, 7 => 0, n => n as u8 }
            }).collect();
            let raw = board_from_grid(&grid);
            for board in [raw, simulate_chain_b3(&raw).final_board] {
                let existing = existing_pop(&board);
                assert_eq!(existing, simulate_chain_b3(&board).chain_count > 0);
                for color in 1..=NUM_TRACKED_COLORS as u8 {
                    for col in 0..BOARD_COLS {
                        if let Some(placed) = drop_one(&board, col, color) {
                            let a = simulate_placed(&board, &placed, existing);
                            let b = simulate_chain_b3(&placed);
                            assert_eq!((a.final_board, a.exact_score, a.chain_count,
                                a.total_erased, a.total_ojama, a.score_approx),
                                (b.final_board, b.exact_score, b.chain_count,
                                b.total_erased, b.total_ojama, b.score_approx));
                        }
                    }
                }
            }
        }
    }

    fn rescue_board() -> BitBoard {
        // 他列は満杯。中央に赤赤を縦置きしたときだけ窒息を連鎖で解消できる。
        let mut grid = [OJAMA; BOARD_ROWS * BOARD_COLS];
        for row in 0..3 { grid[row * BOARD_COLS + DEATH_COL] = 0; }
        for row in 3..6 { grid[row * BOARD_COLS + DEATH_COL] = RED; }
        board_from_grid(&grid)
    }

    #[test]
    fn death_is_checked_after_chain_and_rescue_enters_q() {
        let board = rescue_board();
        let placed = enumerate_placements(&board, (RED, RED), false);
        assert!(!placed.is_empty() && placed.iter().all(|(_, b)| is_dead(b)));
        assert!(enumerate_placements(&board, (RED, RED), true).is_empty());
        for resolve in [false, true] {
            let node = Node { board, mask: 0b10000, shot: 0, potential: 0, links: 0, build: true };
            let mut best = (0, 0);
            let children = expand(&[node], &[(RED, RED)], resolve, &mut best);
            assert!(!children.is_empty() && children.iter().all(|n| !is_dead(&n.board)));
            assert_eq!(best, (RESCUE_SCORE, 1));
            let result = search(board, &[RED, RED], &[RED], 1, 4, resolve, 12, &[]);
            assert_eq!(result.0, vec![RESCUE_SCORE]);
        }
    }

    #[test]
    fn death_remaining_after_chain_is_not_a_candidate() {
        for resolve in [false, true] {
            let result = search(rescue_board(), &[BLUE, BLUE], &[RED, BLUE], 1, 4, resolve, 12, &[]);
            assert_eq!(result.0, vec![0]);
            assert_eq!(result.2, 0);
            assert!(result.3);
        }
    }

    #[test]
    fn hidden_ojama_falls_without_being_erased_only_in_b3() {
        // 隠しおじゃまの直下4赤が消える。下側のおじゃま1個だけ巻き込む。
        let mut grid = [0; BOARD_ROWS * BOARD_COLS];
        for row in 0..BOARD_ROWS { grid[row * BOARD_COLS] = OJAMA; }
        for row in 1..5 { grid[row * BOARD_COLS] = RED; }
        let board = board_from_grid(&grid);
        let sim = simulate_chain_b3(&board);
        assert_eq!(sim.total_ojama, 1);
        assert_eq!(sim.final_board.ojama[0].count_ones(), 8);
        assert_eq!(board_to_grid(&sim.final_board)[5 * BOARD_COLS], OJAMA);
        let old = simulate_chain(&board, true);
        assert_eq!(old.total_ojama, 2);
        assert_eq!(old.final_board.ojama[0].count_ones(), 7);
        assert_eq!(old.exact_score, sim.exact_score);
        assert_eq!(simulate_chain_with_steps(&board, true).0.final_board, old.final_board);
        let node = Node { board, mask: 0b10000, shot: 0, potential: 0, links: 0, build: true };
        let children = expand(&[node], &[(BLUE, BLUE)], false, &mut (0, 0));
        assert!(!children.is_empty());
        assert!(children.iter().all(|n| n.board.ojama[0].count_ones() == 8));
    }

    #[test]
    fn unknown_pairs_cover_same_and_mixed_colors() {
        let all = pairs(&[], 0, &[1, 2, 3, 4]);
        assert_eq!(all.len(), 10);
        assert!(all.contains(&(1, 1)) && all.contains(&(1, 4)));
        assert_eq!(pairs(&[0, 2], 0, &[1, 2]), vec![(1, 2), (2, 2)]);
    }

    #[test]
    fn dedup_and_two_puyos_per_hand() {
        let node = Node { board: BitBoard::default(), mask: 0b11111, shot: 0, potential: 0, links: 0, build: true };
        let out = expand(&[node.clone(), node], &[(1, 1)], false, &mut (0, 0));
        // 同色組は上下反転が同一。縦6+横5だけ残る。
        assert_eq!(out.len(), BOARD_COLS * 2 - 1);
        assert!(out.iter().all(|n| board_to_grid(&n.board).iter().filter(|&&v| v != 0).count() == 2));
    }

    #[test]
    fn nonfire_survives_shortlist_and_beam() {
        let mut fire = Node { board: BitBoard::default(), mask: 0b10000, shot: 40, potential: 0, links: 99, build: false };
        fire.board = drop_one(&fire.board, 0, 1).unwrap();
        let build = Node { board: BitBoard::default(), mask: 0b10000, shot: 0, potential: 0, links: 0, build: true };
        let out = select(vec![fire.clone(), fire.clone(), fire, build], &[1], 2, 1);
        assert!(out.iter().any(|n| n.build));
        assert!(out.iter().any(|n| n.shot > 0));
    }

    #[test]
    fn path_cannot_introduce_fifth_color() {
        let node = Node { board: BitBoard::default(), mask: 0b10000, shot: 0, potential: 0, links: 0, build: true };
        assert!(expand(&[node], &[(1, 5)], false, &mut (0, 0)).is_empty());
    }

    #[test]
    fn identical_boards_merge_compatible_palettes() {
        let a = Node { board: BitBoard::default(), mask: 0b10000, shot: 0, potential: 0, links: 0, build: true };
        let mut b = a.clone(); b.mask = 0b01000;
        let out = expand(&[a, b], &[(1, 1)], false, &mut (0, 0));
        assert_eq!(out.len(), BOARD_COLS * 2 - 1);
        assert!(out.iter().all(|n| n.mask == 0b11000));
        assert!(out.iter().all(|n| compatible(n.mask, (4, 5)) == 0));
    }
}
