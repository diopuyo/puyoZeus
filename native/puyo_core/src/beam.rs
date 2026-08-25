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
//!
//! ## 2026-08-21 追加 (user指示「両方採用+高速化3点」)
//!
//! `beam_search` を一般化した `beam_search_from_frontier` を新設した。
//! 単一盤面の根から探索する従来の `beam_search` はこの一般化版を
//! `initial_frontier=[(盤面,0)]` で呼ぶ薄いラッパになる。一般化の狙いは
//! 3つの高速化を1つのエンジンで実現すること:
//!
//! 1. **初期集団の質を上げる** (`exact_shallow` の完全探索結果をビーム
//!    サーチの初期フロンティアとして渡す、`final_frontier` 参照)。
//! 2. **答えを変えない打ち切り** (`early_exit_score`、running_best は
//!    深さに対して単調非減少なので、一度閾値以上になれば以降の手を
//!    計算しても「閾値以上か」の判定は変わらない)。
//! 3. 深さ1〜2限定の部分木共有 (`final_frontier` をキャッシュして
//!    再利用、呼び出し元 `scripts/mc_counter_estimator.py` 側で実装)。

use crate::bitboard::{
    enumerate_placements, max_column_height, simulate_chain, BitBoard, Placement,
};

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
    /// 探索終了時点の最終フロンティア (盤面, running_best) の組
    /// (2026-08-21 追加、continuation/シード用途。呼び出し側が使わない
    /// 場合は無視してよい、生成コストはフロンティアのコピーのみで軽微)。
    pub final_frontier: Vec<(BitBoard, i64)>,
}

/// 1 ノードから 1 手展開して候補ノード列を作る (発火まで含む)。
///
/// `use_exact_score`: true で評価に `exact_score` (連結ボーナス反映) を使う
/// (2026-08-21 追加、user指示の応手ビームロールアウト検証用)。
/// `simulate_chain` は `score_approx`/`exact_score` を常に両方計算している
/// ため (`bitboard.rs::simulate_chain_core` 参照)、この切替自体に追加の
/// シミュレーションコストは発生しない (読み出すフィールドが変わるだけ)。
fn expand_node(
    idx: usize,
    node: &BeamNode,
    pair: (u8, u8),
    exclude_hidden_row_from_pop: bool,
    use_exact_score: bool,
) -> Vec<BeamNode> {
    enumerate_placements(&node.board, pair, /*filter_dead=*/ true)
        .into_iter()
        .map(|(placement, placed)| {
            let sim = simulate_chain(&placed, exclude_hidden_row_from_pop);
            let score = if use_exact_score { sim.exact_score } else { sim.score_approx };
            let running_best = node.running_best.max(score);
            BeamNode {
                board: sim.final_board,
                running_best,
                parent: Some(idx),
                placement: Some(placement),
            }
        })
        .collect()
}

/// 1 深さ分の展開+高さ枝刈り+幅打ち切りをまとめる (可読性のため抽出)。
///
/// `max_height`: Some(h) の場合、盤面のいずれかの列の高さが `h` 以上の
/// 候補を即座に枝刈りする (ama `ai/search/dfs/attack.cpp` の防御的枝刈り
/// [`>11`] と同一意匠、2026-08-21 追加。既定 None = 枝刈りしない、
/// backwards compat。`scripts/mc_counter_estimator.py`
/// `EXACT_SHALLOW_PRUNE_HEIGHT` docstring に物理量からの導出根拠)。
fn expand_one_depth(
    prev: &[BeamNode],
    pair: (u8, u8),
    exclude_hidden_row_from_pop: bool,
    parallel: bool,
    use_exact_score: bool,
    max_height: Option<u32>,
    beam_width: usize,
) -> Vec<BeamNode> {
    let mut candidates: Vec<BeamNode> = if parallel {
        use rayon::prelude::*;
        prev.par_iter()
            .enumerate()
            .flat_map(|(idx, node)| {
                expand_node(idx, node, pair, exclude_hidden_row_from_pop, use_exact_score)
            })
            .collect()
    } else {
        prev.iter()
            .enumerate()
            .flat_map(|(idx, node)| {
                expand_node(idx, node, pair, exclude_hidden_row_from_pop, use_exact_score)
            })
            .collect()
    };

    if let Some(threshold) = max_height {
        candidates.retain(|node| max_column_height(&node.board) < threshold);
    }
    if candidates.is_empty() {
        return candidates;
    }

    // running_best 降順ソート (同値は安定ソートで生成順を維持、近似タイブレーク)
    candidates.sort_by(|a, b| b.running_best.cmp(&a.running_best));
    candidates.truncate(beam_width);
    candidates
}

/// `early_exit_score` 判定 (可読性のため抽出、2026-08-21 追加)。
fn should_early_exit(early_exit_score: Option<i64>, global_best: i64) -> bool {
    early_exit_score.map_or(false, |t| global_best >= t)
}

/// 最終フロンティアから最良ノードを選び、手順復元して結果を組み立てる
/// (可読性のため抽出)。
fn finalize_result(
    history: Vec<Vec<BeamNode>>,
    best_score_per_depth: Vec<i64>,
) -> BeamSearchResult {
    let mut best_depth = history.len() - 1;
    while history[best_depth].is_empty() && best_depth > 0 {
        best_depth -= 1;
    }
    let last = &history[best_depth];
    let (best_local_idx, best_node) = match last.iter().enumerate().max_by_key(|(_, n)| n.running_best) {
        None => {
            // 初手すら置けない (窒息盤面/空フロンティア) — 空の結果を返す
            return BeamSearchResult {
                best_score: 0,
                best_path: Vec::new(),
                best_score_per_depth,
                final_frontier: Vec::new(),
            };
        }
        Some(v) => v,
    };
    let best_score = best_node.running_best;

    // 手順復元 (親ポインタを根まで辿る)。シードされたフロンティア (根、
    // parent=None) に達すると自然にループが終わる (通常の単一盤面開始
    // ケースと同じ終了条件、深さ0の一般化として何もしなくても正しく動く)。
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

    let final_frontier = last.iter().map(|n| (n.board, n.running_best)).collect();
    BeamSearchResult {
        best_score,
        best_path: path,
        best_score_per_depth,
        final_frontier,
    }
}

/// `beam_search` の一般化版: 任意の初期フロンティア (盤面+running_bestの組)
/// から探索を続ける (2026-08-21 追加、モジュール docstring「2026-08-21
/// 追加」参照)。
///
/// Args:
///     initial_frontier: 開始フロンティア (盤面, その時点の running_best)。
///         `exact_shallow` (完全探索) の結果や、前回呼び出しの
///         `final_frontier` を渡すことで探索を継続できる。手順復元は
///         **本関数呼び出し以降の区間のみ** (継続前の手順はシード側が別途
///         保持していない設計、`mc_counter_estimator.py` は best_score
///         のみを使うため問題ない)。
///     global_best_in: フロンティアに反映されていない「これまでの
///         running_best の最大値」(通常は `initial_frontier` 内の最大値と
///         同じだが、呼び出し側の都合で別に渡せるようにしている)。
///     early_exit_score: Some(t) の場合、ある深さで `global_best >= t` に
///         達したら**以降の手を計算せず即終了**する (2026-08-21 追加、
///         user指示②)。running_best は深さに対して単調非減少なので、一度
///         t 以上になれば以降も必ず t 以上を維持する — 「t 以上に到達
///         したか」の判定は打ち切り時点で確定しており、後続の手を計算
///         しても変わらない。ただし `best_score`/`best_score_per_depth`
///         の**具体的な値**は打ち切り時点のもの (下限) になり、打ち切り
///         なしの場合の最終値とは異なる可能性がある — 呼び出し側は
///         「閾値到達確率」の用途にのみ使うこと (分布 mean/percentile には
///         使わないこと、`estimate_counter_distribution` の
///         `early_exit_at_threshold` docstring 参照)。
#[allow(clippy::too_many_arguments)]
pub fn beam_search_from_frontier(
    initial_frontier: Vec<(BitBoard, i64)>,
    global_best_in: i64,
    pairs: &[(u8, u8)],
    beam_width: usize,
    exclude_hidden_row_from_pop: bool,
    parallel: bool,
    use_exact_score: bool,
    max_height: Option<u32>,
    early_exit_score: Option<i64>,
) -> BeamSearchResult {
    let root: Vec<BeamNode> = initial_frontier
        .into_iter()
        .map(|(board, running_best)| BeamNode {
            board, running_best, parent: None, placement: None,
        })
        .collect();
    let mut history: Vec<Vec<BeamNode>> = Vec::with_capacity(pairs.len() + 1);
    history.push(root);

    let mut best_score_per_depth: Vec<i64> = Vec::with_capacity(pairs.len());
    let mut global_best: i64 = global_best_in;

    if should_early_exit(early_exit_score, global_best) {
        return finalize_result(history, best_score_per_depth);
    }

    for (d, &pair) in pairs.iter().enumerate() {
        let prev = &history[d];
        if prev.is_empty() {
            // フロンティアが枯れた (全滅) — これ以降は展開不能、直前の最良を維持する
            best_score_per_depth.push(global_best);
            history.push(Vec::new());
            continue;
        }

        let candidates = expand_one_depth(
            prev, pair, exclude_hidden_row_from_pop, parallel, use_exact_score, max_height,
            beam_width,
        );
        if candidates.is_empty() {
            best_score_per_depth.push(global_best);
            history.push(Vec::new());
            continue;
        }

        let depth_best = candidates.iter().map(|n| n.running_best).max().unwrap_or(global_best);
        global_best = global_best.max(depth_best);
        best_score_per_depth.push(global_best);
        history.push(candidates);

        if should_early_exit(early_exit_score, global_best) {
            break;
        }
    }

    finalize_result(history, best_score_per_depth)
}

// 単一盤面から探索する簡易エントリポイントは `lib.rs::run_beam_search_
// from_board` (num_threads によるプール分岐込みの薄いラッパ) が兼ねており、
// `beam_search_from_frontier` を `initial_frontier=[(盤面,0)]` で直接呼ぶ
// (2026-08-21、旧 `beam_search` 単体ラッパは呼び出し元が無くなったため削除)。
