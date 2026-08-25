//! `puyo_core` — ぷよぷよ連鎖シミュレーション + ビームサーチ Rust ネイティブ拡張。
//!
//! PyO3 バインディング。盤面は Python 側で 13x6 の numpy 配列 (色コード
//! 0=空,1-5=色,9=おじゃま,10=UNKNOWN) を保持しているが、本クレートの
//! バインディング関数は flat な `Vec<u8>` (長さ78、行優先) を受け取る
//! 薄い設計とする (numpy crate 依存を避け、ビルド時間・リスクを削減する
//! ため。numpy 配列 <-> flat list の変換は `src/puyo_core_bridge.py` 側の
//! 薄いラッパーが担う、task 指示の「numpy 盤面を受け取る薄いラッパ」は
//! Python 側ラッパーとして実現する設計判断)。
//!
//! backwards compat: 本クレートは新規追加のみで既存 Python コードには
//! 一切触れない (`src/chain_bitboard.py` 等は無変更)。

mod beam;
mod bitboard;

use std::sync::OnceLock;

// HSV セル分類のネイティブ実装 (2026-08-20 追加)
mod hsv_classify;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::ThreadPool;

use bitboard::{
    board_from_grid, board_to_grid, drop_one, enumerate_placements, is_dead as bitboard_is_dead,
    simulate_chain, simulate_chain_with_steps, BitBoard, BOARD_COLS, BOARD_ROWS,
};

/// rayon スレッドプールをプロセス全体で 1 個だけ再利用する
/// (`ThreadPoolBuilder::build()` は数msの生成コストがあり、探索1回ごとに
/// 作り直すとリアルタイム用途 (目標 10〜20ms) では無視できないオーバーヘッドに
/// なるため。初回呼び出しのスレッド数で確定する — 呼び出し側が探索ごとに
/// 異なる num_threads を渡す運用は想定しない、というシンプルな制約を明示する)。
static GLOBAL_POOL: OnceLock<ThreadPool> = OnceLock::new();

fn get_or_build_pool(num_threads: usize) -> Result<&'static ThreadPool, String> {
    if let Some(pool) = GLOBAL_POOL.get() {
        return Ok(pool);
    }
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build()
        .map_err(|e| e.to_string())?;
    // 他スレッドが先に set() していた場合は自分のプールを捨てて既存を使う
    let _ = GLOBAL_POOL.set(pool);
    Ok(GLOBAL_POOL.get().expect("直前に set したので必ず存在する"))
}

const GRID_LEN: usize = BOARD_ROWS * BOARD_COLS;

fn grid_from_pylist(grid: Vec<u8>) -> PyResult<[u8; GRID_LEN]> {
    if grid.len() != GRID_LEN {
        return Err(PyValueError::new_err(format!(
            "grid の長さが不正です: {} (期待値: {})",
            grid.len(),
            GRID_LEN
        )));
    }
    let mut arr = [0u8; GRID_LEN];
    arr.copy_from_slice(&grid);
    Ok(arr)
}

/// 1 盤面の連鎖シミュレーション結果 (Python から属性アクセス可能)。
#[pyclass]
#[derive(Clone)]
pub struct ChainSimResultPy {
    #[pyo3(get)]
    pub chain_count: i32,
    #[pyo3(get)]
    pub total_erased: i64,
    #[pyo3(get)]
    pub total_ojama: i64,
    #[pyo3(get)]
    pub score_approx: i64,
    /// 厳密得点 (連結ボーナス反映、2026-08-13 追加、後方互換: 既存フィールド無変更)。
    #[pyo3(get)]
    pub exact_score: i64,
    #[pyo3(get)]
    pub final_grid: Vec<u8>,
}

/// 1 盤面を連鎖シミュレートする (`src/chain_bitboard.py::simulate_batch_with_approx_score`
/// とビット一致のパリティが正解基準、`tests/test_puyo_core_parity.py` 参照)。
///
/// Args:
///     grid: 13x6 色コード配列を flatten した長さ78のリスト (行優先)。
///     exclude_hidden_row_from_pop: 幽霊連鎖ルール (本番既定 True)。
#[pyfunction]
fn simulate_chain_py(
    py: Python<'_>,
    grid: Vec<u8>,
    exclude_hidden_row_from_pop: bool,
) -> PyResult<ChainSimResultPy> {
    let arr = grid_from_pylist(grid)?;
    let result = py.allow_threads(|| {
        let board = board_from_grid(&arr);
        simulate_chain(&board, exclude_hidden_row_from_pop)
    });
    Ok(ChainSimResultPy {
        chain_count: result.chain_count,
        total_erased: result.total_erased,
        total_ojama: result.total_ojama,
        score_approx: result.score_approx,
        exact_score: result.exact_score,
        final_grid: board_to_grid(&result.final_board),
    })
}

/// 1 ステップ (1 消し) の同時消し詳細 (Python から属性アクセス可能、
/// 2026-08-13 追加)。XII-5 同時消しリッチネス
/// (`src/indicators_v2.py::simultaneous_pop_richness`) の native 対応用。
#[pyclass]
#[derive(Clone)]
pub struct ChainStepInfoPy {
    #[pyo3(get)]
    pub num_groups: u32,
    #[pyo3(get)]
    pub num_colors: u32,
    #[pyo3(get)]
    pub erased_count: i64,
    #[pyo3(get)]
    pub ojama_count: i64,
}

/// `simulate_chain_py` の拡張版: 連鎖シミュレーション結果に加え、ステップごとの
/// 同時消し詳細も返す (`ChainStepInfoPy`、2026-08-13 追加)。既存
/// `simulate_chain_py`/`ChainSimResultPy` は無変更 (新規関数のみの追加)。
///
/// Args:
///     grid: 13x6 色コード配列を flatten した長さ78のリスト (行優先)。
///     exclude_hidden_row_from_pop: 幽霊連鎖ルール (本番既定 True)。
///
/// Returns:
///     `(ChainSimResultPy, list[ChainStepInfoPy])` のタプル
///     (steps の長さ=chain_count、ステップ順)。
#[pyfunction]
fn simulate_chain_with_steps_py(
    py: Python<'_>,
    grid: Vec<u8>,
    exclude_hidden_row_from_pop: bool,
) -> PyResult<(ChainSimResultPy, Vec<ChainStepInfoPy>)> {
    let arr = grid_from_pylist(grid)?;
    let (result, steps) = py.allow_threads(|| {
        let board = board_from_grid(&arr);
        simulate_chain_with_steps(&board, exclude_hidden_row_from_pop)
    });
    let result_py = ChainSimResultPy {
        chain_count: result.chain_count,
        total_erased: result.total_erased,
        total_ojama: result.total_ojama,
        score_approx: result.score_approx,
        exact_score: result.exact_score,
        final_grid: board_to_grid(&result.final_board),
    };
    let steps_py = steps
        .into_iter()
        .map(|s| ChainStepInfoPy {
            num_groups: s.num_groups,
            num_colors: s.num_colors,
            erased_count: s.erased_count,
            ojama_count: s.ojama_count,
        })
        .collect();
    Ok((result_py, steps_py))
}

/// `simulate_after_drops_py` 専用の結果型 (Python から属性アクセス可能)。
///
/// `ChainSimResultPy` に加え、消去解決前 (連鎖シミュレート前) の「落下直後」
/// 盤面 `dropped_grid` を持つ。呼び出し側 (`scripts/mc_counter_estimator.py`
/// の潜在火力ビーム探索) が2手目探索の起点として「未解決の落下直後盤面」を
/// 必要とするため (既存 `src.indicators_v2._pfp_first_pass` の意味論保存、
/// 連鎖解決後の盤面ではない点に注意)。
#[pyclass]
#[derive(Clone)]
pub struct DropSimResultPy {
    #[pyo3(get)]
    pub dropped_grid: Vec<u8>,
    #[pyo3(get)]
    pub chain_count: i32,
    #[pyo3(get)]
    pub total_erased: i64,
    #[pyo3(get)]
    pub total_ojama: i64,
    #[pyo3(get)]
    pub score_approx: i64,
    #[pyo3(get)]
    pub exact_score: i64,
    #[pyo3(get)]
    pub final_grid: Vec<u8>,
}

/// 1 個ぷよを複数パターン (col, color) 落としてそれぞれ連鎖シミュレートする
/// (`scripts/mc_counter_estimator.py` の takapt定石探索 [列×色30通り] /
/// 潜在火力ビーム探索用、2026-08-13 追加)。1 回の PyO3 呼び出しで複数候補を
/// まとめて処理することで、候補ごとに Python<->Rust 境界を跨ぐ変換コスト
/// (グリッド flatten/reshape 等) を償却する (task 指示「バッチ化できる箇所は
/// バッチで」)。列が満杯で置けない場合はその要素のみ None。
///
/// Args:
///     grid: 落とす前の基準盤面 (13x6 flatten、長さ78)。
///     drops: `[(col, color), ...]` の候補リスト。
///     exclude_hidden_row_from_pop: 幽霊連鎖ルール。
///
/// Returns:
///     `drops` と同じ長さの `Option<ChainSimResultPy>` リスト
///     (None=列が満杯で置けなかった候補)。
#[pyfunction]
fn simulate_after_drops_py(
    py: Python<'_>,
    grid: Vec<u8>,
    drops: Vec<(u8, u8)>,
    exclude_hidden_row_from_pop: bool,
) -> PyResult<Vec<Option<DropSimResultPy>>> {
    let arr = grid_from_pylist(grid)?;
    let results = py.allow_threads(|| {
        let board = board_from_grid(&arr);
        drops
            .into_iter()
            .map(|(col, color)| {
                let dropped = drop_one(&board, col as usize, color)?;
                let r = simulate_chain(&dropped, exclude_hidden_row_from_pop);
                Some(DropSimResultPy {
                    dropped_grid: board_to_grid(&dropped),
                    chain_count: r.chain_count,
                    total_erased: r.total_erased,
                    total_ojama: r.total_ojama,
                    score_approx: r.score_approx,
                    exact_score: r.exact_score,
                    final_grid: board_to_grid(&r.final_board),
                })
            })
            .collect::<Vec<_>>()
    });
    Ok(results)
}

/// `simulate_after_drops_py` の軽量版: 盤面 (グリッド) を一切返さず
/// `(chain_count, exact_score)` のみ返す (2026-08-13 追加)。
///
/// 呼び出し側が盤面そのものを必要としない場面 (`current_max_chain`/
/// `potential_fire_power` の2手目お邪魔換算等) 向け。`DropSimResultPy` の
/// `dropped_grid`/`final_grid` (各78要素の Vec<u8> をPythonリストへ変換する
/// コスト) を完全に無くすことで、`simulate_after_drops_py` よりさらに
/// Python<->Rust 境界のコストを下げる。値は `simulate_after_drops_py` の
/// 対応要素と完全一致する (パリティは `tests/test_puyo_core_parity.py`)。
///
/// Returns:
///     `drops` と同じ長さの `Option<(chain_count, exact_score)>` リスト。
#[pyfunction]
fn chain_metrics_after_drops_py(
    py: Python<'_>,
    grid: Vec<u8>,
    drops: Vec<(u8, u8)>,
    exclude_hidden_row_from_pop: bool,
) -> PyResult<Vec<Option<(i32, i64)>>> {
    let arr = grid_from_pylist(grid)?;
    let results = py.allow_threads(|| {
        let board = board_from_grid(&arr);
        drops
            .into_iter()
            .map(|(col, color)| {
                let dropped = drop_one(&board, col as usize, color)?;
                let r = simulate_chain(&dropped, exclude_hidden_row_from_pop);
                Some((r.chain_count, r.exact_score))
            })
            .collect::<Vec<_>>()
    });
    Ok(results)
}

/// 複数盤面それぞれに対し、列×色30通り (`drops`) を試して到達できる最大
/// 連鎖数を返す (2026-08-13 追加)。
///
/// `scripts/mc_counter_estimator.py` の `_select_build_placement` の
/// tie-break (`_current_max_chain_value` を候補盤面ごとに個別呼び出し) を
/// 1回のバッチ呼び出しに統合するための API (候補数 [最大22程度] 分の
/// Python<->Rust 往復を1回に削減、`v3.2 選択ロジックの境界コスト削減`
/// docstring参照)。`chain_metrics_after_drops_py` を `grids` の要素数だけ
/// 個別に呼んで `chain_count` の最大を取った場合と完全一致する。
///
/// Args:
///     grids: 評価対象の盤面 (各78要素 flatten) のリスト。
///     drops: `[(col, color), ...]` の候補リスト (通常は列×色30通り)。
///     exclude_hidden_row_from_pop: 幽霊連鎖ルール。
///
/// Returns:
///     `grids` と同じ長さの最大連鎖数リスト (置ける候補が1つも無い盤面は0)。
#[pyfunction]
fn max_chain_after_drops_for_boards_py(
    py: Python<'_>,
    grids: Vec<Vec<u8>>,
    drops: Vec<(u8, u8)>,
    exclude_hidden_row_from_pop: bool,
) -> PyResult<Vec<i32>> {
    let arrs: Vec<[u8; GRID_LEN]> = grids
        .into_iter()
        .map(grid_from_pylist)
        .collect::<PyResult<Vec<_>>>()?;
    let results = py.allow_threads(|| {
        arrs.iter()
            .map(|arr| {
                let board = board_from_grid(arr);
                drops
                    .iter()
                    .filter_map(|&(col, color)| drop_one(&board, col as usize, color))
                    .map(|dropped| simulate_chain(&dropped, exclude_hidden_row_from_pop).chain_count)
                    .max()
                    .unwrap_or(0)
            })
            .collect::<Vec<_>>()
    });
    Ok(results)
}

/// 複数盤面それぞれについて、potential_fire_power (既存指標III-8、2手先
/// ビーム beam_k) の生値 (お邪魔換算前の最良2手先 exact_score) をまとめて
/// 返す (2026-08-13 追加)。
///
/// `scripts/mc_counter_estimator.py::_select_build_placement` の tie-break
/// (潜在連鎖が同値タイの候補全件に対して `_potential_fire_power_value` を
/// 呼ぶ) が実測でロールアウトの主要コストだった (tied件数の実測分布は
/// 中央値10件・最大22件、候補1件あたり最大 1+beam_k 回の native往復が
/// 必要だったため)。本関数で tied 候補全件をまとめて1回の呼び出しに
/// 統合する。
///
/// アルゴリズム (`_pfp_first_pass`/`_pfp_second_pass` と同一意味論、
/// `POTENTIAL_FIRE_POWER_MAX_ADD==2` の場合のみ呼び出し側が使う想定):
///   1手目: `drops` (列×色30通り) を試し、シミュレート後 chain_count 降順で
///     上位 `beam_k` 件の「落下直後 (連鎖解決前)」盤面を残す
///     (`_pfp_first_pass` の dropped_board と同じ、解決済み盤面ではない点に注意)。
///   2手目: 残した各盤面にさらに `drops` を試し、到達できる最大
///     `exact_score` を求める。全候補中の最大値を返す。
///
/// Args:
///     grids: 評価対象の盤面 (各78要素 flatten) のリスト。
///     drops: `[(col, color), ...]` の候補リスト (通常は列×色30通り)。
///     beam_k: 1手目で残す上位候補数 (`POTENTIAL_FIRE_POWER_BEAM_K`)。
///     exclude_hidden_row_from_pop: 幽霊連鎖ルール。
///
/// Returns:
///     `grids` と同じ長さの最良2手先 exact_score リスト (候補が無い盤面は0)。
#[pyfunction]
fn potential_fire_power_raw_for_boards_py(
    py: Python<'_>,
    grids: Vec<Vec<u8>>,
    drops: Vec<(u8, u8)>,
    beam_k: usize,
    exclude_hidden_row_from_pop: bool,
) -> PyResult<Vec<i64>> {
    let arrs: Vec<[u8; GRID_LEN]> = grids
        .into_iter()
        .map(grid_from_pylist)
        .collect::<PyResult<Vec<_>>>()?;
    let results = py.allow_threads(|| {
        arrs.iter()
            .map(|arr| {
                let board = board_from_grid(arr);
                potential_fire_power_raw_one(&board, &drops, beam_k, exclude_hidden_row_from_pop)
            })
            .collect::<Vec<_>>()
    });
    Ok(results)
}

/// `potential_fire_power_raw_for_boards_py` の1盤面分の本体
/// (可読性のため分離、50行ルール対応)。
fn potential_fire_power_raw_one(
    board: &BitBoard,
    drops: &[(u8, u8)],
    beam_k: usize,
    exclude_hidden_row_from_pop: bool,
) -> i64 {
    // 1手目: 30候補をシミュレートし、chain_count降順で上位beam_k件の
    // 「落下直後 (連鎖未解決)」盤面を残す (`_pfp_first_pass` と同一意味論)。
    let mut first_pass: Vec<(i32, BitBoard)> = drops
        .iter()
        .filter_map(|&(col, color)| drop_one(board, col as usize, color))
        .map(|dropped| {
            let r = simulate_chain(&dropped, exclude_hidden_row_from_pop);
            (r.chain_count, dropped)
        })
        .collect();
    first_pass.sort_by(|a, b| b.0.cmp(&a.0));
    first_pass.truncate(beam_k);

    // 2手目: 残した各盤面にさらに30候補を試し、最大exact_scoreを求める。
    first_pass
        .iter()
        .flat_map(|(_chain, board1)| {
            drops.iter().filter_map(move |&(col, color)| {
                let dropped2 = drop_one(board1, col as usize, color)?;
                Some(simulate_chain(&dropped2, exclude_hidden_row_from_pop).exact_score)
            })
        })
        .max()
        .unwrap_or(0)
}

/// 22 配置 (4回転×6列、横置きは5列) を列挙し、配置後盤面 (発火前) を返す。
///
/// Args:
///     grid: 13x6 flatten (長さ78)。
///     top_color / bot_color: ペアの色 (1-5)。
///     filter_dead: true で窒息する配置を除外
///         (`_enumerate_placement_boards` 準拠)。false は `_enumerate_placements` 準拠。
///
/// Returns:
///     list[(col, rotation, grid78, is_dead)] — 呼び出し側で必要な形に整形する。
#[pyfunction]
fn enumerate_placements_py(
    py: Python<'_>,
    grid: Vec<u8>,
    top_color: u8,
    bot_color: u8,
    filter_dead: bool,
) -> PyResult<Vec<(u8, u8, Vec<u8>, bool)>> {
    let arr = grid_from_pylist(grid)?;
    let results = py.allow_threads(|| {
        let board = board_from_grid(&arr);
        enumerate_placements(&board, (top_color, bot_color), filter_dead)
            .into_iter()
            .map(|(placement, placed)| {
                let dead = bitboard_is_dead(&placed);
                (placement.col, placement.rotation, board_to_grid(&placed), dead)
            })
            .collect::<Vec<_>>()
    });
    Ok(results)
}

/// 1 配置 (設置直後盤面+連鎖シミュレーション結果) の統合結果
/// (Python から属性アクセス可能、2026-08-13 追加)。
///
/// `scripts/mc_counter_estimator.py` の `_select_best_placement`/
/// `_select_build_placement` 用: 従来は `enumerate_placements_py` (1回) +
/// 配置ごとの `simulate_chain_py` (最大22回) で最大23回の Python<->Rust
/// 往復が必要だったが、本型を返す `enumerate_and_simulate_placements_py`
/// (1回) に統合することで往復を1回に減らす (境界コスト削減、意味論は
/// 個別呼び出しした場合と完全一致)。
#[pyclass]
#[derive(Clone)]
pub struct PlacementSimResultPy {
    #[pyo3(get)]
    pub col: u8,
    #[pyo3(get)]
    pub rotation: u8,
    /// 設置直後 (連鎖解決前) の盤面。
    #[pyo3(get)]
    pub placed_grid: Vec<u8>,
    /// 設置直後盤面が窒息しているか (`enumerate_placements_py` の `is_dead` と同一意味論)。
    #[pyo3(get)]
    pub is_dead: bool,
    #[pyo3(get)]
    pub chain_count: i32,
    #[pyo3(get)]
    pub total_erased: i64,
    #[pyo3(get)]
    pub total_ojama: i64,
    #[pyo3(get)]
    pub score_approx: i64,
    #[pyo3(get)]
    pub exact_score: i64,
    /// 連鎖解決後の盤面。
    #[pyo3(get)]
    pub final_grid: Vec<u8>,
}

/// 22 配置を列挙し、各配置の連鎖シミュレーション結果まで一括して返す
/// (`enumerate_placements_py` + 配置ごとの `simulate_chain_py` の統合版、
/// 2026-08-13 追加。既存2関数は無変更)。
///
/// Args:
///     grid: 13x6 flatten (長さ78)。
///     top_color / bot_color: ペアの色 (1-5)。
///     filter_dead: true で「設置直後」が窒息する配置を除外
///         (`enumerate_placements_py` と同一意味論)。
///     exclude_hidden_row_from_pop: 幽霊連鎖ルール。
///
/// Returns:
///     `PlacementSimResultPy` のリスト (列挙順=rotation昇順→col昇順、
///     `enumerate_placements_py` と同一順序)。
#[pyfunction]
fn enumerate_and_simulate_placements_py(
    py: Python<'_>,
    grid: Vec<u8>,
    top_color: u8,
    bot_color: u8,
    filter_dead: bool,
    exclude_hidden_row_from_pop: bool,
) -> PyResult<Vec<PlacementSimResultPy>> {
    let arr = grid_from_pylist(grid)?;
    let results = py.allow_threads(|| {
        let board = board_from_grid(&arr);
        enumerate_placements(&board, (top_color, bot_color), filter_dead)
            .into_iter()
            .map(|(placement, placed)| {
                let dead = bitboard_is_dead(&placed);
                let r = simulate_chain(&placed, exclude_hidden_row_from_pop);
                PlacementSimResultPy {
                    col: placement.col,
                    rotation: placement.rotation,
                    placed_grid: board_to_grid(&placed),
                    is_dead: dead,
                    chain_count: r.chain_count,
                    total_erased: r.total_erased,
                    total_ojama: r.total_ojama,
                    score_approx: r.score_approx,
                    exact_score: r.exact_score,
                    final_grid: board_to_grid(&r.final_board),
                }
            })
            .collect::<Vec<_>>()
    });
    Ok(results)
}

// ============================
// 選択ロジック融合 (`scripts/mc_counter_estimator.py::_select_build_placement`
// のRust側完全移植、2026-08-21 追加、v3.3)
// ============================
//
// 従来は「22配置列挙+シミュレーション (1回)」+「current_max_chain tie-break
// (1回)」+「potential_fire_power tie-break (1回)」の最大3回のPython<->Rust
// 往復と、途中候補ぶんのBoardオブジェクト生成 (最大44個/手、実測の主要コスト、
// memory `project_counter_reach_cost_breakdown_2026-08-21` 参照) が必要
// だった。本節は選択ロジック全体を1回のnative呼び出しに融合し、
// Python側へは最終的に選ばれた1盤面のみを返す (途中候補は一切
// Python境界を越えない)。既存3関数 (`enumerate_and_simulate_placements_py`
// 等) は無変更 (新規追加のみ、backwards compat)。

/// 全候補が発火を伴う (強制発火、`_select_build_placement` の
/// `non_dead` 分岐) 場合のフォールバック。`chain_count` が最小の配置を選び、
/// 実際に解決させた `final_board` を返す (Python `min(..., key=...)` と同じ
/// 「同値タイなら先勝ち」の順序保存)。`non_dead` が空 (置き場所が無い) 場合は
/// `None`。
fn force_fire_fallback(
    non_dead: &[(i32, BitBoard)],
    exclude_hidden_row_from_pop: bool,
) -> Option<BitBoard> {
    if non_dead.is_empty() {
        return None;
    }
    let mut best_idx = 0usize;
    for i in 1..non_dead.len() {
        if non_dead[i].0 < non_dead[best_idx].0 {
            best_idx = i;
        }
    }
    let placed = non_dead[best_idx].1;
    Some(simulate_chain(&placed, exclude_hidden_row_from_pop).final_board)
}

/// tie-break本体: まず `current_max_chain` (列×色30通り探索、`drops`) で
/// 最良の候補群 (同値タイ全件) を絞り、1件に絞れなければ
/// `potential_fire_power_raw_one` (既存、2手先ビーム) の最大値1件を選ぶ
/// (`_select_build_placement` のtie-break分岐と同一の「先勝ち」順序保存)。
fn select_by_tie_break(
    build_only: &[BitBoard],
    drops: &[(u8, u8)],
    beam_k: usize,
    exclude_hidden_row_from_pop: bool,
) -> BitBoard {
    let chain_values: Vec<i32> = build_only
        .iter()
        .map(|b| {
            drops
                .iter()
                .filter_map(|&(col, color)| drop_one(b, col as usize, color))
                .map(|d| simulate_chain(&d, exclude_hidden_row_from_pop).chain_count)
                .max()
                .unwrap_or(0)
        })
        .collect();
    let best_potential = *chain_values.iter().max().expect("build_onlyは空でない前提");
    let tied: Vec<BitBoard> = build_only
        .iter()
        .zip(chain_values.iter())
        .filter(|(_, &v)| v == best_potential)
        .map(|(b, _)| *b)
        .collect();
    if tied.len() == 1 {
        return tied[0];
    }
    let mut best_idx = 0usize;
    let mut best_pfp =
        potential_fire_power_raw_one(&tied[0], drops, beam_k, exclude_hidden_row_from_pop);
    for (i, b) in tied.iter().enumerate().skip(1) {
        let v = potential_fire_power_raw_one(b, drops, beam_k, exclude_hidden_row_from_pop);
        if v > best_pfp {
            best_pfp = v;
            best_idx = i;
        }
    }
    tied[best_idx]
}

/// `_select_build_placement` (Python) の選択ロジック全体を1回のnative
/// 呼び出しに融合した本体 (22配置列挙+連鎖判定+2段tie-breakをすべて
/// Rust内で完結させる)。列挙順序は `enumerate_placements` の自然順
/// (rotation昇順→col昇順、既存 `enumerate_and_simulate_placements_py` と
/// 同一) を保つため、Python native経路 (ソート省略) と完全に同じ結果になる。
fn select_build_placement_core(
    board: &BitBoard,
    pair: (u8, u8),
    drops: &[(u8, u8)],
    beam_k: usize,
    exclude_hidden_row_from_pop: bool,
) -> Option<BitBoard> {
    let candidates = enumerate_placements(board, pair, false);
    let mut build_only: Vec<BitBoard> = Vec::new();
    let mut non_dead: Vec<(i32, BitBoard)> = Vec::new();
    for (_placement, placed) in &candidates {
        if bitboard_is_dead(placed) {
            continue;
        }
        let chain_count = simulate_chain(placed, exclude_hidden_row_from_pop).chain_count;
        non_dead.push((chain_count, *placed));
        if chain_count == 0 {
            build_only.push(*placed);
        }
    }
    if build_only.is_empty() {
        return force_fire_fallback(&non_dead, exclude_hidden_row_from_pop);
    }
    if build_only.len() == 1 {
        return Some(build_only[0]);
    }
    Some(select_by_tie_break(&build_only, drops, beam_k, exclude_hidden_row_from_pop))
}

/// `select_build_placement_core` のPyO3バインディング (2026-08-21 追加)。
///
/// Args:
///     grid: 判定対象の盤面 (13x6 flatten、長さ78)。
///     top_color / bot_color: 設置するペアの色。
///     drops: 列×色候補リスト (通常は列×色30通り、`_DROP_CANDIDATES_30`)。
///     beam_k: potential_fire_power 1手目で残す上位候補数。
///     exclude_hidden_row_from_pop: 幽霊連鎖ルール。
///
/// Returns:
///     選択された配置後盤面 (13x6 flatten)。置き場所が全く無い場合は `None`。
///     値は `scripts/mc_counter_estimator.py::_select_build_placement` の
///     native経路 (融合前、個別呼び出し3回) と完全一致する
///     (`tests/test_puyo_core_parity.py` でパリティ確認)。
#[pyfunction]
fn select_build_placement_py(
    py: Python<'_>,
    grid: Vec<u8>,
    top_color: u8,
    bot_color: u8,
    drops: Vec<(u8, u8)>,
    beam_k: usize,
    exclude_hidden_row_from_pop: bool,
) -> PyResult<Option<Vec<u8>>> {
    let arr = grid_from_pylist(grid)?;
    let result = py.allow_threads(|| {
        let board = board_from_grid(&arr);
        select_build_placement_core(
            &board, (top_color, bot_color), &drops, beam_k, exclude_hidden_row_from_pop,
        )
    });
    Ok(result.map(|b| board_to_grid(&b)))
}

/// ビームサーチ結果 (Python から属性アクセス可能)。
#[pyclass]
#[derive(Clone)]
pub struct BeamSearchResultPy {
    #[pyo3(get)]
    pub best_score: i64,
    /// 最良手順 [(col, rotation), ...] (深さ順)。
    #[pyo3(get)]
    pub best_path: Vec<(u8, u8)>,
    /// 深さごとの running-max best スコア配列。
    #[pyo3(get)]
    pub best_score_per_depth: Vec<i64>,
}

/// ビームサーチ (深さ13〜16手 / 幅250 を想定、リアルタイム応手探索用)。
///
/// Args:
///     grid: 探索開始盤面 (13x6 flatten、長さ78)。
///     pairs: 既知ツモ列 `[(top,bot), ...]`。長さ=探索深さ。
///     beam_width: 各深さで残す候補数上限。
///     exclude_hidden_row_from_pop: 幽霊連鎖ルール (本番既定 True)。
///     num_threads: rayon 並列スレッド数。None = 並列無効 (単スレッド)。
///         0 以下は無効 (ValueError)。
///     use_exact_score: true で評価に厳密得点 `exact_score` (連結ボーナス
///         反映) を使う (既定 false = 従来の `score_approx` 近似、
///         2026-08-13 追加、backwards compat)。
///     max_height: Some(h) で盤面のいずれかの列が高さ h 以上になった候補を
///         枝刈りする (既定 None = 枝刈りしない、2026-08-21 追加)。
///     early_exit_score: Some(t) で running_best が t 以上になった時点で
///         以降の手を計算せず打ち切る (既定 None = 打ち切らない、
///         2026-08-21 追加。`beam::beam_search_from_frontier` docstring
///         参照、「閾値到達確率」用途限定で使うこと)。
#[pyfunction]
#[pyo3(signature = (
    grid, pairs, beam_width, exclude_hidden_row_from_pop, num_threads=None,
    use_exact_score=false, max_height=None, early_exit_score=None,
))]
#[allow(clippy::too_many_arguments)]
fn beam_search_py(
    py: Python<'_>,
    grid: Vec<u8>,
    pairs: Vec<(u8, u8)>,
    beam_width: usize,
    exclude_hidden_row_from_pop: bool,
    num_threads: Option<usize>,
    use_exact_score: bool,
    max_height: Option<u32>,
    early_exit_score: Option<i64>,
) -> PyResult<BeamSearchResultPy> {
    let arr = grid_from_pylist(grid)?;
    let result = run_beam_search_from_board(
        py, &arr, &pairs, beam_width, exclude_hidden_row_from_pop, num_threads, use_exact_score,
        max_height, early_exit_score,
    )?;
    Ok(beam_result_to_py(result))
}

/// `beam_search_py`/`exact_shallow_search_py` 共通のスレッドプール分岐処理
/// (可読性のため抽出、2026-08-21 追加)。単一盤面を根に置いた
/// `beam::beam_search_from_frontier` 呼び出しを num_threads に応じて
/// 単スレ/プール実行に振り分ける。
#[allow(clippy::too_many_arguments)]
fn run_beam_search_from_board(
    py: Python<'_>,
    grid: &[u8; GRID_LEN],
    pairs: &[(u8, u8)],
    beam_width: usize,
    exclude_hidden_row_from_pop: bool,
    num_threads: Option<usize>,
    use_exact_score: bool,
    max_height: Option<u32>,
    early_exit_score: Option<i64>,
) -> PyResult<beam::BeamSearchResult> {
    if let Some(n) = num_threads {
        if n == 0 {
            return Err(PyValueError::new_err("num_threads は 1 以上を指定してください"));
        }
    }
    py.allow_threads(|| -> Result<beam::BeamSearchResult, String> {
        let board = board_from_grid(grid);
        let root = vec![(board, 0i64)];
        match num_threads {
            None => Ok(beam::beam_search_from_frontier(
                root, 0, pairs, beam_width, exclude_hidden_row_from_pop,
                /*parallel=*/ false, use_exact_score, max_height, early_exit_score,
            )),
            Some(n) => {
                let pool = get_or_build_pool(n)?;
                Ok(pool.install(|| {
                    beam::beam_search_from_frontier(
                        root, 0, pairs, beam_width, exclude_hidden_row_from_pop,
                        /*parallel=*/ true, use_exact_score, max_height, early_exit_score,
                    )
                }))
            }
        }
    })
    .map_err(PyValueError::new_err)
}

fn beam_result_to_py(result: beam::BeamSearchResult) -> BeamSearchResultPy {
    BeamSearchResultPy {
        best_score: result.best_score,
        best_path: result
            .best_path
            .iter()
            .map(|p| (p.col, p.rotation))
            .collect(),
        best_score_per_depth: result.best_score_per_depth,
    }
}

/// フロンティア1件のシリアライズ可能な表現 (盤面+running_bestのみ、
/// 2026-08-21 追加、Python から属性アクセス可能)。
#[pyclass]
#[derive(Clone)]
pub struct FrontierEntryPy {
    #[pyo3(get)]
    pub grid: Vec<u8>,
    #[pyo3(get)]
    pub running_best: i64,
}

#[pymethods]
impl FrontierEntryPy {
    /// Python 側 (`src/puyo_core_bridge.py::_beam_search_dispatch`) が
    /// `beam_search_continue_py` に渡すフロンティアを構築するための
    /// コンストラクタ (2026-08-21 追加)。
    #[new]
    fn new(grid: Vec<u8>, running_best: i64) -> Self {
        FrontierEntryPy { grid, running_best }
    }
}

/// フロンティア付きビームサーチ結果 (2026-08-21 追加、`exact_shallow_
/// search_py`/`beam_search_continue_py` 用。通常の `beam_search_py` は
/// フロンティアのシリアライズコストを避けるため `BeamSearchResultPy` の
/// ままにしている)。
#[pyclass]
#[derive(Clone)]
pub struct BeamSearchWithFrontierResultPy {
    #[pyo3(get)]
    pub best_score: i64,
    #[pyo3(get)]
    pub best_path: Vec<(u8, u8)>,
    #[pyo3(get)]
    pub best_score_per_depth: Vec<i64>,
    #[pyo3(get)]
    pub final_frontier: Vec<FrontierEntryPy>,
}

fn beam_result_to_py_with_frontier(result: beam::BeamSearchResult) -> BeamSearchWithFrontierResultPy {
    BeamSearchWithFrontierResultPy {
        best_score: result.best_score,
        best_path: result.best_path.iter().map(|p| (p.col, p.rotation)).collect(),
        best_score_per_depth: result.best_score_per_depth,
        final_frontier: result
            .final_frontier
            .iter()
            .map(|(board, running_best)| FrontierEntryPy {
                grid: board_to_grid(board),
                running_best: *running_best,
            })
            .collect(),
    }
}

/// ama方式の浅い完全探索 (2026-08-21 追加、user指示「両方併用」)。
/// `beam_search_py` と異なり **幅の打ち切りを一切行わない** (`pairs` の
/// 深さだけ完全に列挙する、深さ2なら 22+22²=506通り)。結果には
/// `final_frontier` (完全探索終了時点の全候補、`beam_search_continue_py`
/// への継続用) も含む。
///
/// Args:
///     grid/pairs/exclude_hidden_row_from_pop/use_exact_score/
///     max_height/early_exit_score: `beam_search_py` と同一意味論。
#[pyfunction]
#[pyo3(signature = (
    grid, pairs, exclude_hidden_row_from_pop, use_exact_score=true, max_height=None,
    early_exit_score=None,
))]
fn exact_shallow_search_py(
    py: Python<'_>,
    grid: Vec<u8>,
    pairs: Vec<(u8, u8)>,
    exclude_hidden_row_from_pop: bool,
    use_exact_score: bool,
    max_height: Option<u32>,
    early_exit_score: Option<i64>,
) -> PyResult<BeamSearchWithFrontierResultPy> {
    let arr = grid_from_pylist(grid)?;
    let result = py.allow_threads(|| {
        let board = board_from_grid(&arr);
        beam::beam_search_from_frontier(
            vec![(board, 0)], 0, &pairs, usize::MAX, exclude_hidden_row_from_pop,
            /*parallel=*/ false, use_exact_score, max_height, early_exit_score,
        )
    });
    Ok(beam_result_to_py_with_frontier(result))
}

/// 既存フロンティアから探索を継続する (2026-08-21 追加、user指示①③:
/// `exact_shallow_search_py` の完全探索結果 [`final_frontier`] をビーム
/// サーチの初期集団として渡すことで、深さ2までは完全探索・深さ3以降を
/// ビームサーチにする「両方併用」を実現する)。深さ1〜2限定の部分木共有
/// (同じ接頭辞のロールアウトが `final_frontier` をキャッシュして再利用) も
/// 本関数で実現する (呼び出し元 `scripts/mc_counter_estimator.py` 側)。
///
/// Args:
///     frontier: 継続元のフロンティア (`exact_shallow_search_py`/
///         本関数自身の `final_frontier` から得た `FrontierEntryPy` 列)。
///     global_best_in: フロンティア構築時点までの running_best 最大値。
///     pairs: 継続する残りのツモ列。
///     beam_width/exclude_hidden_row_from_pop/num_threads/use_exact_score/
///         max_height/early_exit_score: `beam_search_py` と同一意味論。
#[pyfunction]
#[pyo3(signature = (
    frontier, global_best_in, pairs, beam_width, exclude_hidden_row_from_pop, num_threads=None,
    use_exact_score=true, max_height=None, early_exit_score=None,
))]
#[allow(clippy::too_many_arguments)]
fn beam_search_continue_py(
    py: Python<'_>,
    frontier: Vec<FrontierEntryPy>,
    global_best_in: i64,
    pairs: Vec<(u8, u8)>,
    beam_width: usize,
    exclude_hidden_row_from_pop: bool,
    num_threads: Option<usize>,
    use_exact_score: bool,
    max_height: Option<u32>,
    early_exit_score: Option<i64>,
) -> PyResult<BeamSearchWithFrontierResultPy> {
    if let Some(n) = num_threads {
        if n == 0 {
            return Err(PyValueError::new_err("num_threads は 1 以上を指定してください"));
        }
    }
    let root: Vec<(BitBoard, i64)> = frontier
        .iter()
        .map(|f| -> PyResult<(BitBoard, i64)> {
            let arr = grid_from_pylist(f.grid.clone())?;
            Ok((board_from_grid(&arr), f.running_best))
        })
        .collect::<PyResult<Vec<_>>>()?;

    let result = py.allow_threads(|| -> Result<beam::BeamSearchResult, String> {
        match num_threads {
            None => Ok(beam::beam_search_from_frontier(
                root, global_best_in, &pairs, beam_width, exclude_hidden_row_from_pop,
                /*parallel=*/ false, use_exact_score, max_height, early_exit_score,
            )),
            Some(n) => {
                let pool = get_or_build_pool(n)?;
                Ok(pool.install(|| {
                    beam::beam_search_from_frontier(
                        root, global_best_in, &pairs, beam_width, exclude_hidden_row_from_pop,
                        /*parallel=*/ true, use_exact_score, max_height, early_exit_score,
                    )
                }))
            }
        }
    })
    .map_err(PyValueError::new_err)?;

    Ok(beam_result_to_py_with_frontier(result))
}

/// PyO3 モジュール定義。
#[pymodule]
fn puyo_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(simulate_chain_py, m)?)?;
    m.add_function(wrap_pyfunction!(simulate_chain_with_steps_py, m)?)?;
    m.add_function(wrap_pyfunction!(simulate_after_drops_py, m)?)?;
    m.add_function(wrap_pyfunction!(chain_metrics_after_drops_py, m)?)?;
    m.add_function(wrap_pyfunction!(max_chain_after_drops_for_boards_py, m)?)?;
    m.add_function(wrap_pyfunction!(potential_fire_power_raw_for_boards_py, m)?)?;
    m.add_function(wrap_pyfunction!(enumerate_placements_py, m)?)?;
    m.add_function(wrap_pyfunction!(enumerate_and_simulate_placements_py, m)?)?;
    m.add_function(wrap_pyfunction!(select_build_placement_py, m)?)?;
    m.add_function(wrap_pyfunction!(beam_search_py, m)?)?;
    m.add_function(wrap_pyfunction!(exact_shallow_search_py, m)?)?;
    m.add_function(wrap_pyfunction!(beam_search_continue_py, m)?)?;
    m.add_function(wrap_pyfunction!(hsv_classify::classify_cells_hsv, m)?)?;
    m.add_class::<ChainSimResultPy>()?;
    m.add_class::<ChainStepInfoPy>()?;
    m.add_class::<DropSimResultPy>()?;
    m.add_class::<PlacementSimResultPy>()?;
    m.add_class::<BeamSearchResultPy>()?;
    m.add_class::<FrontierEntryPy>()?;
    m.add_class::<BeamSearchWithFrontierResultPy>()?;
    Ok(())
}
