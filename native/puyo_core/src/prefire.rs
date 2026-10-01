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
fn walk<F: FnMut(&BitBoard, i64, i32, &Path)>(board: &BitBoard, pairs: &[(u8, u8)], path: &mut Path,
        hidden: bool, emit: &mut F) {
    let placements = enumerate_placements(board, pairs[path.len()], false);
    if placements.is_empty() && !path.is_empty() {
        emit(board, 0, 0, path);
    }
    for (placement, placed) in placements {
        let chain = simulate_chain(&placed, hidden);
        path.push((placement.col, placement.rotation));
        if chain.chain_count > 0 || is_dead(&chain.final_board) || path.len() == pairs.len() {
            emit(&chain.final_board, chain.exact_score, chain.chain_count, path);
        } else {
            walk(&chain.final_board, pairs, path, hidden, emit);
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
        walk(&board_from_grid(&raw), &pairs, &mut Vec::new(), exclude_hidden_row_from_pop,
             &mut |board, score, chains, path| out.push((board_to_grid(board), score, chains, path.clone())));
        out
    }))
}

/// 案C用。同じ全列挙を走査し、各手数の最大得点と連鎖数だけを返す。
#[pyfunction]
pub fn prefire_threat_maxima_py(py: Python<'_>, grid: Vec<u8>, pairs: Vec<(u8, u8)>,
                                  exclude_hidden_row_from_pop: bool) -> PyResult<Vec<(i64, i32, usize)>> {
    if grid.len() != CELLS || pairs.is_empty() || pairs.len() > MAX_KNOWN_PAIRS {
        return Err(PyValueError::new_err("盤面78セル・既知1〜3組が必要"));
    }
    if pairs.iter().any(|(a,b)| !(1..=5).contains(a) || !(1..=5).contains(b)) {
        return Err(PyValueError::new_err("未知色を既知ツモとして列挙できない"));
    }
    let mut raw = [0u8; CELLS];
    raw.copy_from_slice(&grid);
    Ok(py.allow_threads(|| {
        let mut best: Vec<Option<(i64, i32, Path)>> = vec![None; pairs.len()];
        walk(&board_from_grid(&raw), &pairs, &mut Vec::new(), exclude_hidden_row_from_pop,
             &mut |_, score, chains, path| {
                 if chains == 0 { return; }
                 let slot = &mut best[path.len()-1];
                 // 得点換算は単調。同得点は5Bのpath辞書順で選び、連鎖長も一致させる。
                 if slot.as_ref().map_or(true, |old| score > old.0 || (score == old.0 && path < &old.2)) {
                     *slot = Some((score, chains, path.clone()));
                 }
             });
        best.into_iter().flatten().map(|(score, chains, path)| (score, chains, path.len())).collect()
    }))
}
