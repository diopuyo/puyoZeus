//! ビームサーチ本体 (連鎖シミュレーションエンジン `bitboard.rs` の上位)。
//!
//! 探索仕様 (task 指示 2026-08-12):
//!     入力 = 盤面 + ツモ列 (既知ペアのリスト、長さ=深さ) + 幅。
//!     評価 = 各ノードで発火可能な最大スコア (running max、= その手まで
//!         のどこかで発火した最大スコアを、深さを跨いで引き継ぐ)。
//!     出力 = 最良スコア・最良手順・深さごとの best スコア配列。
//!
//! 「発火可能な最大スコア」の意味論は既存 `_near_future_known_expand`
//! (`src/indicators_v2.py:2520`) と同じ: 1手ごとにペアを置き、繋がった
//! ぷよは即座に自動発火する (発火するかどうかを選べるゲームではない)。
//! 次の手は発火後の残骸盤面から続ける。

use crate::bitboard::{enumerate_placements, simulate_chain, BitBoard, Placement};

/// ビームサーチ中の 1 ノード (探索木のスナップショット)。
#[derive(Clone, Copy)]
struct BeamNode {
    board: BitBoard,
    /// この手までのどこかで発火した最大スコア (running max)。
    running_best: i64,
    /// 直前フロンティア内でのインデックス (手順復元用、根ノードは None)。
    parent: Option<usize>,
    /// この手の設置座標 (根ノードは None)。
    placement: Option<Placement>,
}

/// ビームサーチの結果。
pub struct BeamSearchResult {
    pub best_score: i64,
    pub best_path: Vec<Placement>,
    /// 深さごとの running-max best スコア (呼び出し側が用途に応じて解釈)。
    pub best_score_per_depth: Vec<i64>,
}

/// 1 ノードから 1 手展開して候補ノード列を作る (発火まで含む)。
fn expand_node(
    idx: usize,
    node: &BeamNode,
    pair: (u8, u8),
    exclude_hidden_row_from_pop: bool,
) -> Vec<BeamNode> {
    enumerate_placements(&node.board, pair, /*filter_dead=*/ true)
        .into_iter()
        .map(|(placement, placed)| {
            let sim = simulate_chain(&placed, exclude_hidden_row_from_pop);
            let running_best = node.running_best.max(sim.score_approx);
            BeamNode {
                board: sim.final_board,
                running_best,
                parent: Some(idx),
                placement: Some(placement),
            }
        })
        .collect()
}

/// ビームサーチ本体 (深さ = `pairs.len()`、幅 = `beam_width`)。
///
/// Args:
///     initial: 探索開始盤面。
///     pairs: 既知ツモ列 (top, bot) の並び。長さ=探索深さ。
///     beam_width: 各深さで残す候補数の上限。
///     exclude_hidden_row_from_pop: 幽霊連鎖ルール (本番既定 True、
///         `src/production_config.py::GHOST_CHAIN_RULE_ENABLED` 参照)。
///     parallel: true で rayon 並列展開 (フロンティア全ノードを並列処理)。
pub fn beam_search(
    initial: &BitBoard,
    pairs: &[(u8, u8)],
    beam_width: usize,
    exclude_hidden_row_from_pop: bool,
    parallel: bool,
) -> BeamSearchResult {
    let mut history: Vec<Vec<BeamNode>> = Vec::with_capacity(pairs.len() + 1);
    history.push(vec![BeamNode {
        board: *initial,
        running_best: 0,
        parent: None,
        placement: None,
    }]);

    let mut best_score_per_depth: Vec<i64> = Vec::with_capacity(pairs.len());
    let mut global_best: i64 = 0;

    for (d, &pair) in pairs.iter().enumerate() {
        let prev = &history[d];
        if prev.is_empty() {
            // フロンティアが枯れた (全滅) — これ以降は展開不能、直前の最良を維持する
            best_score_per_depth.push(global_best);
            history.push(Vec::new());
            continue;
        }

        let mut candidates: Vec<BeamNode> = if parallel {
            use rayon::prelude::*;
            prev.par_iter()
                .enumerate()
                .flat_map(|(idx, node)| expand_node(idx, node, pair, exclude_hidden_row_from_pop))
                .collect()
        } else {
            prev.iter()
                .enumerate()
                .flat_map(|(idx, node)| expand_node(idx, node, pair, exclude_hidden_row_from_pop))
                .collect()
        };

        if candidates.is_empty() {
            best_score_per_depth.push(global_best);
            history.push(Vec::new());
            continue;
        }

        // running_best 降順ソート (同値は安定ソートで生成順を維持、近似タイブレーク)
        candidates.sort_by(|a, b| b.running_best.cmp(&a.running_best));
        candidates.truncate(beam_width);

        let depth_best = candidates
            .iter()
            .map(|n| n.running_best)
            .max()
            .unwrap_or(global_best);
        global_best = global_best.max(depth_best);
        best_score_per_depth.push(global_best);
        history.push(candidates);
    }

    // 最終フロンティア (枯れていれば最後の非空フロンティア) から最良ノードを選ぶ。
    let mut best_depth = history.len() - 1;
    while history[best_depth].is_empty() && best_depth > 0 {
        best_depth -= 1;
    }
    let last = &history[best_depth];
    let (best_local_idx, best_node) = if last.is_empty() {
        // 初手すら置けない (窒息盤面) — 空の結果を返す
        return BeamSearchResult {
            best_score: 0,
            best_path: Vec::new(),
            best_score_per_depth,
        };
    } else {
        last.iter()
            .enumerate()
            .max_by_key(|(_, n)| n.running_best)
            .unwrap()
    };
    let best_score = best_node.running_best;

    // 手順復元 (親ポインタを根まで辿る)
    let mut path: Vec<Placement> = Vec::new();
    let mut cur_depth = best_depth;
    let mut cur_idx = best_local_idx;
    while cur_depth > 0 {
        let node = &history[cur_depth][cur_idx];
        path.push(node.placement.expect("非根ノードは placement を持つ"));
        cur_idx = node.parent.expect("非根ノードは parent を持つ");
        cur_depth -= 1;
    }
    path.reverse();

    BeamSearchResult {
        best_score,
        best_path: path,
        best_score_per_depth,
    }
}
