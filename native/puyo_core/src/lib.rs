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

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::ThreadPool;

use bitboard::{
    board_from_grid, board_to_grid, enumerate_placements, is_dead as bitboard_is_dead,
    simulate_chain, BOARD_COLS, BOARD_ROWS,
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
        final_grid: board_to_grid(&result.final_board),
    })
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
#[pyfunction]
#[pyo3(signature = (grid, pairs, beam_width, exclude_hidden_row_from_pop, num_threads=None))]
fn beam_search_py(
    py: Python<'_>,
    grid: Vec<u8>,
    pairs: Vec<(u8, u8)>,
    beam_width: usize,
    exclude_hidden_row_from_pop: bool,
    num_threads: Option<usize>,
) -> PyResult<BeamSearchResultPy> {
    let arr = grid_from_pylist(grid)?;
    if let Some(n) = num_threads {
        if n == 0 {
            return Err(PyValueError::new_err("num_threads は 1 以上を指定してください"));
        }
    }

    let result = py.allow_threads(|| -> Result<beam::BeamSearchResult, String> {
        let board = board_from_grid(&arr);
        match num_threads {
            None => Ok(beam::beam_search(
                &board,
                &pairs,
                beam_width,
                exclude_hidden_row_from_pop,
                /*parallel=*/ false,
            )),
            Some(n) => {
                let pool = get_or_build_pool(n)?;
                Ok(pool.install(|| {
                    beam::beam_search(
                        &board,
                        &pairs,
                        beam_width,
                        exclude_hidden_row_from_pop,
                        /*parallel=*/ true,
                    )
                }))
            }
        }
    })
    .map_err(PyValueError::new_err)?;

    Ok(BeamSearchResultPy {
        best_score: result.best_score,
        best_path: result
            .best_path
            .iter()
            .map(|p| (p.col, p.rotation))
            .collect(),
        best_score_per_depth: result.best_score_per_depth,
    })
}

/// PyO3 モジュール定義。
#[pymodule]
fn puyo_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(simulate_chain_py, m)?)?;
    m.add_function(wrap_pyfunction!(enumerate_placements_py, m)?)?;
    m.add_function(wrap_pyfunction!(beam_search_py, m)?)?;
    m.add_class::<ChainSimResultPy>()?;
    m.add_class::<BeamSearchResultPy>()?;
    Ok(())
}
