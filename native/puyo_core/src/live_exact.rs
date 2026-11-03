//! 単盤面の厳密計算専用C ABI。既存bitboardのUNKNOWN空扱いとは分離する。
#![no_std]

extern "C" { fn abort() -> !; }

// panic=abort専用。外部例外が境界を越えた場合も復帰しない。
#[no_mangle]
pub extern "C" fn rust_eh_personality() -> ! {
    unsafe { abort() }
}

#[panic_handler]
fn panic(_: &core::panic::PanicInfo<'_>) -> ! {
    unsafe { abort() }
}

const ROWS: usize = 13;
const COLS: usize = 6;
const CELLS: usize = ROWS * COLS;
const EMPTY: u8 = 0;
const OJAMA: u8 = 9;
const UNKNOWN: u8 = 10;
const MIN_POP: usize = 4;
const RECORD: usize = CELLS * 3;

fn neighbors(i: usize, hidden: usize) -> impl Iterator<Item = usize> {
    let (r, c) = (i / COLS, i % COLS);
    [if r > hidden { Some(i-COLS) } else { None },
     if r+1 < ROWS { Some(i+COLS) } else { None },
     if c > 0 { Some(i-1) } else { None },
     if c+1 < COLS { Some(i+1) } else { None }].into_iter().flatten()
}

fn component(grid: &[u8], start: usize, hidden: usize, seen: &mut [bool; CELLS]) -> ([usize; CELLS], usize) {
    let mut cells = [0; CELLS];
    cells[0] = start;
    let mut size = 1;
    seen[start] = true;
    let mut cursor = 0;
    while cursor < size {
        for other in neighbors(cells[cursor], hidden) {
            if !seen[other] && grid[other] == grid[start] {
                seen[other] = true;
                cells[size] = other;
                size += 1;
            }
        }
        cursor += 1;
    }
    (cells, size)
}

fn groups(grid: &[u8], hidden: usize) -> [u8; CELLS] {
    let (mut seen, mut labels, mut label) = ([false; CELLS], [0; CELLS], 0);
    for start in hidden*COLS..CELLS {
        if seen[start] || [EMPTY, OJAMA, UNKNOWN].contains(&grid[start]) { continue; }
        let (cells, size) = component(grid, start, hidden, &mut seen);
        if size >= MIN_POP {
            label += 1;
            for &cell in &cells[..size] { labels[cell] = label; }
        }
    }
    labels
}

fn gravity(grid: &mut [u8; CELLS]) {
    for col in 0..COLS {
        let mut floor = ROWS;
        for row in (0..ROWS).rev() {
            let i = row*COLS+col;
            let value = grid[i];
            if value == UNKNOWN { floor = row; }
            else if value != EMPTY {
                floor -= 1;
                grid[i] = EMPTY;
                grid[floor*COLS+col] = value;
            }
        }
    }
}

/// 呼出側は78セルとhidden=0/1を検証する。
#[no_mangle]
pub unsafe extern "C" fn live_stable(input: *const u8, hidden: usize) -> bool {
    !groups(core::slice::from_raw_parts(input, CELLS), hidden).iter().any(|&v| v != 0)
}

/// 局所判定の入力色を勝手に制限せず、既存Pythonと同じ連結を調べる。
#[no_mangle]
pub unsafe extern "C" fn live_local(input: *const u8, starts: *const u8, count: usize, hidden: usize) -> bool {
    let grid = core::slice::from_raw_parts(input, CELLS);
    let mut seen = [false; CELLS];
    for &cell in core::slice::from_raw_parts(starts, count) {
        let start = cell as usize;
        if start >= hidden*COLS && !seen[start]
            && component(grid, start, hidden, &mut seen).1 >= MIN_POP { return true; }
    }
    false
}

/// 各段の消去前・消去後・グループ番号を記録する。候補・探索は扱わない。
#[no_mangle]
pub unsafe extern "C" fn live_simulate(input: *const u8, hidden: usize, output: *mut u8) -> usize {
    let mut grid = [0; CELLS];
    grid.copy_from_slice(core::slice::from_raw_parts(input, CELLS));
    let mut steps = 0;
    loop {
        let labels = groups(&grid, hidden);
        if !labels.iter().any(|&v| v != 0) { break; }
        let record = core::slice::from_raw_parts_mut(output.add(steps*RECORD), RECORD);
        record[..CELLS].copy_from_slice(&grid);
        record[CELLS*2..].copy_from_slice(&labels);
        let mut erase = [false; CELLS];
        for i in 0..CELLS {
            if labels[i] == 0 { continue; }
            erase[i] = true;
            for other in neighbors(i, 0) {
                if grid[other] == OJAMA { erase[other] = true; }
            }
        }
        for i in 0..CELLS { if erase[i] { grid[i] = EMPTY; } }
        gravity(&mut grid);
        record[CELLS..CELLS*2].copy_from_slice(&grid);
        steps += 1;
    }
    steps
}
