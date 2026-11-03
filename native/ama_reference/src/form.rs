//! 正の既定カタログの上下分割と鏡像選択。
use crate::{field::{bit, Field}, form_data::*};

const ERROR: i32 = -100;
const NUISANCE: u8 = 1;
/// 左下 4 列 × 4 段 (おじゃまがあれば形の評価をしない範囲)。
const LOWER_LEFT: u128 = {
    let mut m = 0;
    let mut x = 0;
    while x < 4 {
        let mut y = 0;
        while y < 4 { m |= bit(x, y); y += 1; }
        x += 1;
    }
    m
};
const CELLS: [[[i32; 6]; 6]; 3] = [GTR_CELLS, FRON_CELLS, SGTR_CELLS];
const MATRIX: [[[i32; 8]; 8]; 3] = [GTR_MATRIX, FRON_MATRIX, SGTR_MATRIX];

/// 形の評価が見る下 6 段の色番号 ([列][段])。盤から 1 回だけ読み出す。
type Lower = [[u8; ROWS]; 6];
const ROWS: usize = 6;

fn lower_cells(f: &Field) -> Lower {
    let mut out = [[0; ROWS]; 6];
    for (x, column) in out.iter_mut().enumerate() {
        for (y, cell) in column.iter_mut().enumerate() { *cell = f.cell(x, y); }
    }
    out
}

fn pattern(f: &Field, lower: &Lower, index: usize, upper: bool, mirrored: bool) -> i32 {
    let (mut result, mut colors) = (0, [0u8; 8]);
    let start = if upper { 3 } else { 0 };
    let matrix = &MATRIX[index];
    for x in 0..6 {
        for y in start..start + 3 {
            if f.heights[x] <= y { break; }
            let group = CELLS[index][y][if mirrored { 5 - x } else { x }] as usize;
            if group == 0 { continue; }
            let color = lower[x][y];
            if colors[group] == 0 { colors[group] = color; }
            else if colors[group] != color && matrix[group][group] > 0 { return ERROR; }
            result += matrix[group][group];
        }
    }
    for i in 1..8 {
        if colors[i] == 0 { continue; }
        for k in i + 1..8 {
            if colors[k] == 0 { continue; }
            let relation = matrix[i][k];
            if relation < 0 && colors[i] == colors[k] { return ERROR; }
            if relation > 0 {
                if colors[i] != colors[k] { return ERROR; }
                result += relation;
            }
        }
    }
    result
}

pub fn evaluate(f: &Field) -> (i32, bool) {
    if f.mask(NUISANCE) & LOWER_LEFT != 0 { return (0, false); }
    let lower = lower_cells(f);
    let (mut best, mut base) = (0, ERROR);
    for i in 0..6 {
        let score = pattern(f, &lower, i % 3, false, i >= 3);
        if i == 0 || score > base { base = score; best = i; }
    }
    let mirrored = best >= 3;
    let mut upper = ERROR;
    for i in 0..3 {
        let boundary_ok = (0..6).all(|x| {
            let cx = if mirrored { 5 - x } else { x };
            CELLS[best % 3][2][cx] == 0 || CELLS[i][3][cx] == 0 ||
                f.heights[x] <= 3 || lower[x][2] != lower[x][3]
        });
        if boundary_ok { upper = upper.max(pattern(f, &lower, i, true, mirrored)); }
    }
    (base + upper, mirrored)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn empty_form_and_nuisance() {
        let mut f = Field::default();
        assert_eq!(evaluate(&f), (0, false));
        f.drop_color(0, 1, false);
        assert_eq!(evaluate(&f), (0, false));
    }
}
