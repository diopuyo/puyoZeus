//! ama ai/search/beam/{beam,layer}.cpp (dea210bc) の移植。
//! MIT License, Copyright (c) 2023 citrus610。探索表は Vec 所有で必ず解放する。
use crate::ama_eval::{evaluate, heights, Weight, DEATH_HEIGHT, VISIBLE_HEIGHT};
use crate::bitboard::{drop_one, height_of, simulate_chain_ama as simulate_chain, BitBoard, ChainSimResult, BOARD_COLS, DEATH_COL};
use crate::{
    ama_order as order,
    ama_table::{hash, Table},
};

pub const VISIBLE_PAIRS: usize = 3;
const MAX_PLACEMENTS: usize = BOARD_COLS * 4 - 2;
const PRUNE: i32 = 5000;
pub const DEFAULT_TRIGGER: i32 = 95000;
const BAGS: [[usize; 4]; 2] = [[0, 1, 2, 3], [1, 2, 0, 3]];
pub type Pair = (u8, u8);
pub type Placement = (usize, u8);

#[derive(Clone, Copy)]
pub struct Config {
    pub trigger: i32,
    pub width: usize,
    pub depth: usize,
    pub colors: [u8; 4],
    pub weight: Weight,
}

#[derive(Clone, Copy)]
struct Node {
    board: BitBoard,
    eval: i32,
    action: i32,
    index: usize,
}
fn key(n: Node) -> i32 {
    n.eval + n.action
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Candidate {
    pub placement: Placement,
    pub score: i32,
}

pub fn placements(b: &BitBoard, pair: Pair) -> Vec<Placement> {
    let h = heights(b);
    let mut result = Vec::with_capacity(MAX_PLACEMENTS);
    for r in 0..if pair.0 == pair.1 { 2 } else { 4 } {
        for x in 0..BOARD_COLS {
            let valid = match r {
                0 | 2 => h[x] <= DEATH_HEIGHT as i32,
                1 => x + 1 < BOARD_COLS && h[x] <= VISIBLE_HEIGHT as i32 && h[x + 1] <= VISIBLE_HEIGHT as i32,
                _ => x > 0 && h[x] <= VISIBLE_HEIGHT as i32 && h[x - 1] <= VISIBLE_HEIGHT as i32,
            };
            if valid {
                result.push((x, r));
            }
        }
    }
    result
}

pub fn apply(b: &BitBoard, (x, r): Placement, pair: Pair) -> Option<ChainSimResult> {
    #[cfg(feature = "ama-profile-detail")]
    let _timer = crate::ama_profile_clock::Timer::new(3);
    let (a, bcolor, y) = match r {
        0 => (pair.0, pair.1, x),
        1 => (pair.0, pair.1, x + 1),
        2 => (pair.1, pair.0, x),
        _ => (pair.0, pair.1, x.checked_sub(1)?),
    };
    let placed = drop_one(&drop_one(b, x, a)?, y, bcolor)?;
    let result = simulate_chain(&placed, true);
    if height_of(&result.final_board, DEATH_COL) > DEATH_HEIGHT {
        None
    } else {
        Some(result)
    }
}

fn expand(node: Node, pair: Pair, cfg: &Config, mut callback: impl FnMut(Node, Placement, i32)) {
    let h = heights(&node.board);
    for p in placements(&node.board, pair) {
        if crate::ama_budget::expired() { break; }
        let Some(result) = apply(&node.board, p, pair) else {
            continue;
        };
        let other = match p.1 {
            1 => p.0 + 1,
            3 => p.0 - 1,
            _ => p.0,
        };
        let action = node.action
            + i32::from(h[p.0] != h[other]) * cfg.weight.tear
            + result.chain_count * cfg.weight.waste;
        callback(Node { board: result.final_board, action, ..node }, p, result.exact_score as i32);
    }
}

struct Layer {
    data: Vec<Node>,
    table: Table,
    width: usize,
}
impl Layer {
    fn new(width: usize) -> Self {
        Self { data: Vec::with_capacity(width), table: Table::new(), width }
    }
    fn add(&mut self, node: Node) {
        if self.data.len() < self.width {
            self.data.push(node);
            if self.data.len() == self.width {
                order::make(&mut self.data, key);
            }
        } else if key(self.data[0]) < key(node) {
            order::pop(&mut self.data, key);
            *self.data.last_mut().unwrap() = node;
            order::push(&mut self.data, key);
        }
    }
    fn sort(&mut self) {
        // 原典の初期層は data.push_back され、ヒープ化されていないが同じ順序を再現。
        if self.data.len() < self.width {
            order::insertion(&mut self.data, key);
        } else {
            order::sort_heap(&mut self.data, key);
        }
    }
}

fn think(pair: Pair, candidates: &mut [Candidate], parents: &mut Layer, children: &mut Layer, cfg: &Config) {
    parents.sort();
    for &node in &parents.data {
        if crate::ama_budget::expired() { break; }
        expand(node, pair, cfg, |mut child, _, score| {
            candidates[child.index].score = candidates[child.index].score.max(score);
            if score >= PRUNE {
                return;
            }
            let eval = children.table.probe(hash(&child.board, &cfg.colors), child.action, || {
                evaluate(&child.board, &cfg.colors, &cfg.weight)
            });
            if let Some(eval) = eval {
                child.eval = eval;
                children.add(child);
            }
        });
    }
    parents.data.clear();
    parents.table.update();
}

pub fn search(board: BitBoard, queue: &[Pair], cfg: &Config) -> Vec<Candidate> {
    if queue.len() < 2 {
        return Vec::new();
    }
    let root = Node { board, eval: 0, action: 0, index: 0 };
    let (mut parents, mut children) = (Layer::new(cfg.width), Layer::new(cfg.width));
    let mut candidates = Vec::new();
    expand(root, queue[0], cfg, |mut child, placement, score| {
        child.index = candidates.len();
        child.eval = evaluate(&child.board, &cfg.colors, &cfg.weight);
        candidates.push(Candidate { placement, score });
        parents.data.push(child);
    });
    for &pair in &queue[1..] {
        if crate::ama_budget::expired() { break; }
        think(pair, &mut candidates, &mut parents, &mut children, cfg);
        if candidates.iter().any(|c| c.score >= cfg.trigger) {
            break;
        }
        std::mem::swap(&mut parents, &mut children);
    }
    candidates
}

pub fn multi(board: BitBoard, visible: &[Pair], depth: usize, cfg: &Config, cap: i32) -> Vec<Candidate> {
    let mut result: Vec<Candidate> = Vec::new();
    #[cfg(feature = "ama-reuse-identical-queues")]
    let mut previous: Option<(Vec<Pair>, Vec<Candidate>)> = None;
    for bag in BAGS {
        if crate::ama_budget::expired() { break; }
        let mut queue = visible[..visible.len().min(depth).min(VISIBLE_PAIRS)].to_vec();
        let start = queue.len();
        while queue.len() < depth {
            let offset = (queue.len() - start) % 2 * 2;
            queue.push((cfg.colors[bag[offset]], cfg.colors[bag[offset + 1]]));
        }
        #[cfg(feature = "ama-reuse-identical-queues")]
        let mut found = if crate::ama_budget::is_unbounded() {
            if let Some((_, saved)) = previous.as_ref().filter(|(q, _)| q == &queue) {
                saved.clone()
            } else {
                let fresh = search(board, &queue, cfg);
                previous = Some((queue.clone(), fresh.clone()));
                fresh
            }
        } else { search(board, &queue, cfg) };
        #[cfg(not(feature = "ama-reuse-identical-queues"))]
        let mut found = search(board, &queue, cfg);
        if cap > 0 {
            for c in &mut found {
                c.score = c.score.min(cap);
            }
        }
        if result.is_empty() {
            result = found;
        } else {
            for (a, b) in result.iter_mut().zip(found) {
                a.score += b.score;
            }
        }
    }
    order::insertion(&mut result, |c| c.score);
    result
}

pub fn last(board: &BitBoard, pair: Pair) -> Option<Placement> {
    let mut best = None;
    let mut score = -1;
    for p in placements(board, pair) {
        if let Some(r) = apply(board, p, pair) {
            if r.exact_score > score {
                score = r.exact_score;
                best = Some(p);
            }
        }
    }
    best
}

pub fn need_move(board: &BitBoard, visible: &[Pair], need: i32) -> Option<Placement> {
    let first = placements(board, visible[0]);
    for depth in 1..=2.min(visible.len()) {
        let (mut best, mut score) = (None, -1);
        for &p in &first {
            if crate::ama_budget::expired() { return None; }
            let Some(r) = apply(board, p, visible[0]) else {
                continue;
            };
            let value = if depth == 1 {
                r.exact_score
            } else if r.exact_score > 0 {
                continue;
            } else {
                placements(&r.final_board, visible[1])
                    .into_iter()
                    .filter_map(|q| apply(&r.final_board, q, visible[1]))
                    .map(|s| s.exact_score)
                    .max()
                    .unwrap_or(-1)
            };
            if value >= need as i64 && value > score {
                score = value;
                best = Some(p);
            }
        }
        if best.is_some() {
            return best;
        }
    }
    None
}

pub fn decide(
    board: &BitBoard,
    visible: &[Pair],
    remaining: usize,
    cfg: &Config,
    need: i32,
) -> Option<Placement> {
    if remaining == 1 {
        return last(board, visible[0]);
    }
    if need > 0 {
        if let Some(p) = need_move(board, &visible[..visible.len().min(remaining)], need) {
            return Some(p);
        }
    }
    multi(*board, visible, remaining.min(cfg.depth).max(2), cfg, need).first().map(|c| c.placement)
}

#[cfg(test)]
mod profile_tests {
    use super::*;

    #[test]
    fn identical_queues_keep_candidate_values_and_order() {
        let cfg = Config { trigger: DEFAULT_TRIGGER, width: 1, depth: 2,
            colors: [1, 2, 3, 4], weight: Weight::BUILD };
        let mut board = BitBoard::default();
        const CASES: usize = 64;
        for index in 0..CASES {
            let pair = ((index % 4 + 1) as u8, ((index / 4) % 4 + 1) as u8);
            let queue = [pair, (2, 3)];
            for cap in [0, 70, 10000] {
                let mut expected = search(board, &queue, &cfg);
                for candidate in &mut expected {
                    candidate.score = 2 * if cap > 0 { candidate.score.min(cap) } else { candidate.score };
                }
                order::insertion(&mut expected, |c| c.score);
                assert_eq!(multi(board, &queue, 2, &cfg, cap), expected);
            }
            let choices = placements(&board, pair);
            board = choices.get(index % choices.len().max(1))
                .and_then(|&p| apply(&board, p, pair)).map_or(BitBoard::default(), |r| r.final_board);
        }
    }
}
