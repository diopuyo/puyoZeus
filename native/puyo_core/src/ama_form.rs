//! ama ai/search/beam/form.{h,cpp} (dea210bc) より。MIT, Copyright (c) 2023 citrus610。
use crate::bitboard::{BitBoard, BOARD_COLS};
const FORM_HEIGHT: usize = 6;
const FORM_ERROR: i32 = -100;
const GTR_FORM: [[u8; 6]; 6] = [
    [0, 0, 0, 0, 0, 0],
    [4, 4, 4, 0, 0, 0],
    [3, 3, 3, 4, 0, 0],
    [1, 2, 5, 0, 0, 0],
    [1, 1, 2, 5, 0, 0],
    [2, 2, 5, 0, 0, 0],
];
const GTR_MATRIX: [[i32; 8]; 8] = [
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 2, -1, -1, 0, 0, 0, 0],
    [0, -1, 1, -1, 0, -1, 0, 0],
    [0, -1, -1, 2, -1, -1, 0, 0],
    [0, 0, 0, -1, 0, 0, 0, 0],
    [0, 0, -1, -1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
];
const FRON_FORM: [[u8; 6]; 6] = [
    [0, 0, 0, 0, 0, 0],
    [5, 5, 5, 0, 0, 0],
    [4, 4, 4, 5, 0, 0],
    [1, 1, 3, 6, 0, 0],
    [1, 2, 2, 7, 0, 0],
    [3, 3, 2, 3, 6, 0],
];
const FRON_MATRIX: [[i32; 8]; 8] = [
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 2, -1, -1, -1, 0, 0, 0],
    [0, -1, 1, -1, 0, 0, 0, -1],
    [0, -1, -1, 1, -1, 0, -1, 0],
    [0, -1, 0, -1, 2, -1, 0, 0],
    [0, 0, 0, 0, -1, 0, 0, 0],
    [0, 0, 0, -1, 0, 0, 0, 0],
    [0, 0, -1, 0, 0, 0, 0, 0],
];
const SGTR_FORM: [[u8; 6]; 6] = [
    [0, 0, 0, 0, 0, 0],
    [5, 5, 5, 0, 0, 0],
    [4, 4, 4, 5, 0, 0],
    [1, 1, 3, 6, 0, 0],
    [1, 2, 2, 3, 6, 0],
    [2, 3, 3, 6, 0, 0],
];
const SGTR_MATRIX: [[i32; 8]; 8] = [
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 2, -1, -1, -1, 0, 0, 0],
    [0, -1, 1, -1, 0, 0, 0, 0],
    [0, -1, -1, 1, -1, 0, -1, 0],
    [0, -1, 0, -1, 2, -1, 0, 0],
    [0, 0, 0, 0, -1, 0, 0, 0],
    [0, 0, 0, -1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
];

fn match_form(
    b: &BitBoard,
    h: &[i32; BOARD_COLS],
    form: &[[u8; BOARD_COLS]; FORM_HEIGHT],
    matrix: &[[i32; 8]; 8],
) -> i32 {
    let mut map = [0u8; 8];
    let mut result = 0;
    for x in 0..BOARD_COLS {
        for y in 0..FORM_HEIGHT.min(h[x] as usize) {
            let index = form[FORM_HEIGHT - 1 - y][x] as usize;
            if index == 0 {
                continue;
            }
            let color = (0..b.colors.len())
                .find(|&c| b.colors[c][x] & (1 << y) != 0)
                .map_or(if b.ojama[x] & (1 << y) != 0 { 9 } else { 0 }, |c| c as u8 + 1);
            if map[index] == 0 {
                map[index] = color;
            } else if map[index] != color && matrix[index][index] > 0 {
                return FORM_ERROR;
            }
            result += matrix[index][index];
        }
    }
    for i in 1..map.len() {
        for k in i + 1..map.len() {
            if map[i] == 0 || map[k] == 0 {
                continue;
            }
            let rel = matrix[i][k];
            if (rel < 0 && map[i] == map[k]) || (rel > 0 && map[i] != map[k]) {
                return FORM_ERROR;
            }
            if rel > 0 {
                result += rel;
            }
        }
    }
    result
}

pub fn evaluate(b: &BitBoard, h: &[i32; BOARD_COLS]) -> i32 {
    // 原典のマスクは左4列の下4段だけを検査する。
    const GARBAGE_MASK: u16 = 0xf;
    if b.ojama[..4].iter().any(|v| v & GARBAGE_MASK != 0) {
        return 0;
    }
    [
        match_form(b, h, &GTR_FORM, &GTR_MATRIX),
        match_form(b, h, &FRON_FORM, &FRON_MATRIX),
        match_form(b, h, &SGTR_FORM, &SGTR_MATRIX),
    ]
    .into_iter()
    .max()
    .unwrap_or(FORM_ERROR)
}
