//! 既知3組までの全列挙。得点による候補削減を行わない。
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use crate::bitboard::{board_from_grid, board_to_grid, enumerate_placements, is_dead,
                     simulate_chain, BitBoard, BOARD_COLS, BOARD_ROWS};

const MAX_KNOWN_PAIRS: usize = 3;
const CELLS: usize = BOARD_COLS * BOARD_ROWS;
type Path = Vec<(u8, u8)>;
type Terminal = (Vec<u8>, i64, i32, Path);

/// 静かな配置だけを再帰展開し、最初の発火・死亡・末端をすべて残す。
fn walk(board: &BitBoard, pairs: &[(u8, u8)], path: &mut Path,
        hidden: bool, out: &mut Vec<Terminal>) {
    let placements = enumerate_placements(board, pairs[path.len()], false);
    if placements.is_empty() && !path.is_empty() {
        out.push((board_to_grid(board), 0, 0, path.clone()));
    }
    for (placement, placed) in placements {
        let chain = simulate_chain(&placed, hidden);
        path.push((placement.col, placement.rotation));
        if chain.chain_count > 0 || is_dead(&chain.final_board) || path.len() == pairs.len() {
            out.push((board_to_grid(&chain.final_board), chain.exact_score,
                      chain.chain_count, path.clone()));
        } else {
            walk(&chain.final_board, pairs, path, hidden, out);
        }
        path.pop();
    }
}

/// Python境界は探索全体で1往復。評価前のK制限はAPI自体に存在しない。
#[pyfunction]
pub fn prefire_terminals_py(py: Python<'_>, grid: Vec<u8>, pairs: Vec<(u8, u8)>,
                            exclude_hidden_row_from_pop: bool) -> PyResult<Vec<Terminal>> {
    if grid.len() != CELLS || pairs.is_empty() || pairs.len() > MAX_KNOWN_PAIRS {
        return Err(PyValueError::new_err("盤面78セル・既知1〜3組が必要"));
    }
    if pairs.iter().any(|(a,b)| !(1..=5).contains(a) || !(1..=5).contains(b)) {
        return Err(PyValueError::new_err("未知色を既知ツモとして列挙できない"));
    }
    let mut raw = [0u8; CELLS];
    raw.copy_from_slice(&grid);
    Ok(py.allow_threads(|| {
        let mut out = Vec::new();
        walk(&board_from_grid(&raw), &pairs, &mut Vec::new(), exclude_hidden_row_from_pop, &mut out);
        out
    }))
}
