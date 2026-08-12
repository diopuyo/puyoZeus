//! ビットボード連鎖判定エンジン (`src/chain_bitboard.py` の Rust 移植)。
//!
//! 出典・帰属 (MIT License):
//!     消去判定式 (`get_mask_pop` の m3/m2 定式) は citrus610/ama
//!     (https://github.com/citrus610/ama, MIT License, Copyright (c) 2023 citrus610)
//!     の `core/fieldbit.cpp` のアルゴリズム設計に対する帰属を明記する。
//!     本ファイルは `src/chain_bitboard.py` (numpy 移植版) をさらに Rust の
//!     スカラー演算へ移植したもの (SIMD 命令列は未使用)。
//!
//! 盤面ビット表現 (`src/chain_bitboard.py` docstring と同一):
//!     1 列 = 1 個の u16 (下位 13 bit を使用)。
//!     bit i は物理行 (BOARD_ROWS - 1 - i) に対応する
//!     (bit0 = 最下段 row=12、bit12 = 最上段(隠し段) row=0)。
//!
//! **正解基準**: `src/chain_bitboard.py::simulate_batch_with_approx_score` と
//! ビット一致のパリティ (`tests/test_puyo_core_parity.py` で担保)。

// ============================
// 定数 (マジックナンバー禁止 → 全て名前付き定数)
// ============================

/// 盤面の行数 (`src/board.py::BOARD_ROWS` と同一)。
pub const BOARD_ROWS: usize = 13;
/// 盤面の列数 (`src/board.py::BOARD_COLS` と同一)。
pub const BOARD_COLS: usize = 6;
/// 13bit 全体を使うフルマスク (= 0x1FFF)。
pub const FULL_MASK: u16 = (1u16 << BOARD_ROWS) - 1;
/// 隠し段 (row0 = 13段目) に対応するビット (bit12)。
pub const HIDDEN_ROW_BIT: u16 = 1u16 << (BOARD_ROWS - 1);
/// 隠し段を除いた 12 段分のマスク (幽霊連鎖ルール用、`POP_MASK_12BIT`)。
pub const POP_MASK_12BIT: u16 = FULL_MASK & !HIDDEN_ROW_BIT;
/// 連鎖ループの安全弁 (`chain_bitboard.MAX_CHAIN_STEPS` と同一)。
pub const MAX_CHAIN_STEPS: usize = 19;

/// 窒息判定行 (可視最上段、隠し段row0は含まない。`src/board.py::DEATH_ROW`)。
pub const DEATH_ROW: usize = 1;
/// 窒息判定列 (`src/board.py::DEATH_COL`)。
pub const DEATH_COL: usize = 2;
/// 窒息判定に使うビット位置 (row1 に対応する bit index = 12-1 = 11)。
pub const DEATH_BIT_INDEX: u32 = (BOARD_ROWS - 1 - DEATH_ROW) as u32;

/// 色コード (src/board.py と同一の値をそのまま使う)。
pub const COLOR_EMPTY: u8 = 0;
pub const COLOR_RED: u8 = 1;
// COLOR_BLUE/GREEN/YELLOW は release ビルド本体では COLOR_RED..COLOR_PURPLE の
// 範囲チェックのみで使うため直接参照されないが、単体テスト (#[cfg(test)]) や
// 将来の呼び出し元での可読性のために名前付き定数として保持する。
#[allow(dead_code)]
pub const COLOR_BLUE: u8 = 2;
#[allow(dead_code)]
pub const COLOR_GREEN: u8 = 3;
#[allow(dead_code)]
pub const COLOR_YELLOW: u8 = 4;
pub const COLOR_PURPLE: u8 = 5;
pub const COLOR_OJAMA: u8 = 9;
#[allow(dead_code)] // 参照用定数 (エンジンには入れない値である、という仕様の明示)
pub const COLOR_UNKNOWN: u8 = 10;
/// 連結判定対象の色数 (RED/BLUE/GREEN/YELLOW/PURPLE の 5 色)。
pub const NUM_TRACKED_COLORS: usize = 5;

/// 1ぷよあたりの基本得点 (`src/scoring.py::BASE_SCORE_PER_PUYO`)。
pub const BASE_SCORE_PER_PUYO: i64 = 10;
/// ボーナス係数の最小値・最大値 (`src/scoring.py` と同一)。
pub const MIN_BONUS_MULTIPLIER: i64 = 1;
pub const MAX_BONUS_MULTIPLIER: i64 = 999;
/// 連鎖ボーナステーブル (1連鎖目 index0、`src/scoring.py::CHAIN_POWER_TABLE`)。
pub const CHAIN_POWER_TABLE: [i64; 19] = [
    0, 8, 16, 32, 64, 96, 128, 160, 192, 224,
    256, 288, 320, 352, 384, 416, 448, 480, 512,
];
/// 19連鎖超の線形延長幅 (`src/scoring.py::CHAIN_POWER_INCREMENT`)。
pub const CHAIN_POWER_INCREMENT: i64 = 32;
/// 色数ボーナス LUT (index=同時消去色数0-5、`chain_bitboard._COLOR_BONUS_LUT` と同一)。
pub const COLOR_BONUS_LUT: [i64; 6] = [0, 0, 3, 6, 12, 24];
/// 連結ボーナス上限 (11連結以上、`src/scoring.py::CONNECTION_BONUS_MAX`)。
pub const CONNECTION_BONUS_MAX: i64 = 10;

/// 1 列分のビット平面 (6 列)。
pub type ColPlanes = [u16; BOARD_COLS];

/// 盤面のビットボード表現。
///
/// `colors[i]` は色 `i+1` (RED=1..PURPLE=5) のビット平面。
/// `src/chain_bitboard.py::TRACKED_COLORS` と同じ順序 (RED,BLUE,GREEN,YELLOW,PURPLE)。
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct BitBoard {
    pub colors: [ColPlanes; NUM_TRACKED_COLORS],
    pub ojama: ColPlanes,
}

// ============================
// シフト演算 (ama の get_expand / get_mask_pop 相当)
// ============================

/// 縦方向 (盤面内の上下) シフト。`chain_bitboard._shift_vertical` のスカラー版。
fn shift_vertical(p: &ColPlanes, toward_lsb: bool) -> ColPlanes {
    let mut r: ColPlanes = [0; BOARD_COLS];
    for c in 0..BOARD_COLS {
        r[c] = if toward_lsb {
            (p[c] >> 1) & FULL_MASK
        } else {
            (p[c] << 1) & FULL_MASK
        };
    }
    r
}

/// 横方向 (列方向) シフト。`chain_bitboard._shift_horizontal` のスカラー版。
///
/// toward_low_col=true: 列 c <- 列 c+1 (右を見る)。false: 列 c <- 列 c-1 (左を見る)。
fn shift_horizontal(p: &ColPlanes, toward_low_col: bool) -> ColPlanes {
    let mut r: ColPlanes = [0; BOARD_COLS];
    if toward_low_col {
        for c in 0..BOARD_COLS - 1 {
            r[c] = p[c + 1];
        }
    } else {
        for c in 1..BOARD_COLS {
            r[c] = p[c - 1];
        }
    }
    r
}

fn plane_and(a: &ColPlanes, b: &ColPlanes) -> ColPlanes {
    let mut r: ColPlanes = [0; BOARD_COLS];
    for c in 0..BOARD_COLS {
        r[c] = a[c] & b[c];
    }
    r
}

fn plane_or(a: &ColPlanes, b: &ColPlanes) -> ColPlanes {
    let mut r: ColPlanes = [0; BOARD_COLS];
    for c in 0..BOARD_COLS {
        r[c] = a[c] | b[c];
    }
    r
}

fn plane_mask(p: &ColPlanes, mask: u16) -> ColPlanes {
    let mut r: ColPlanes = [0; BOARD_COLS];
    for c in 0..BOARD_COLS {
        r[c] = p[c] & mask;
    }
    r
}

fn plane_not_masked(p: &ColPlanes, mask: u16) -> ColPlanes {
    let mut r: ColPlanes = [0; BOARD_COLS];
    for c in 0..BOARD_COLS {
        r[c] = (!p[c]) & mask;
    }
    r
}

fn plane_any_nonzero(p: &ColPlanes) -> bool {
    p.iter().any(|&v| v != 0)
}

fn plane_popcount(p: &ColPlanes) -> i64 {
    p.iter().map(|&v| v.count_ones() as i64).sum()
}

/// ama `get_expand`: 自身 | 上下左右シフト (1マス膨張、self含む)。
fn expand(p: &ColPlanes) -> ColPlanes {
    let u = shift_vertical(p, true);
    let d = shift_vertical(p, false);
    let r = shift_horizontal(p, true);
    let l = shift_horizontal(p, false);
    plane_or(&plane_or(&plane_or(p, &u), &d), &plane_or(&r, &l))
}

/// ama `get_mask_pop` のスカラー移植 (1色分)。
///
/// `chain_bitboard._get_mask_pop_batch` とビット単位で同一の演算列。
fn get_mask_pop(plane: &ColPlanes) -> ColPlanes {
    let m = *plane; // 呼び出し側で既にマスク済みの前提 (pop_mask 適用後)
    let u = plane_and(&shift_vertical(&m, true), &m);
    let d = plane_and(&shift_vertical(&m, false), &m);
    let r = plane_and(&shift_horizontal(&m, true), &m);
    let l = plane_and(&shift_horizontal(&m, false), &m);

    let ud_and = plane_and(&u, &d);
    let lr_and = plane_and(&l, &r);
    let ud_or = plane_or(&u, &d);
    let lr_or = plane_or(&l, &r);

    let m3 = plane_or(&plane_and(&ud_and, &lr_or), &plane_and(&lr_and, &ud_or));
    let m2 = plane_or(&plane_or(&ud_and, &lr_and), &plane_and(&ud_or, &lr_or));

    let m2_r = plane_and(&shift_horizontal(&m2, true), &m2);
    let m2_l = plane_and(&shift_horizontal(&m2, false), &m2);
    let m2_u = plane_and(&shift_vertical(&m2, true), &m2);
    let m2_d = plane_and(&shift_vertical(&m2, false), &m2);

    let core = plane_or(&plane_or(&m3, &m2_r), &plane_or(&m2_l, &plane_or(&m2_u, &m2_d)));
    plane_and(&expand(&core), &m)
}

// ============================
// PEXT 相当 (重力コンパクション)
// ============================

/// PEXT (parallel bit extract) のスカラー版。
///
/// `keep_mask` のうち立っているビット位置の `value` ビットを、LSB側へ
/// 順に詰めて返す (`chain_bitboard._pext_batch` のスカラー等価版)。
fn pext16(value: u16, keep_mask: u16) -> u16 {
    let mut result: u16 = 0;
    let mut out_pos: u32 = 0;
    for bit in 0..BOARD_ROWS as u32 {
        let bit_flag = 1u16 << bit;
        if (keep_mask & bit_flag) != 0 {
            if (value & bit_flag) != 0 {
                result |= 1u16 << out_pos;
            }
            out_pos += 1;
        }
    }
    result
}

fn pext_plane(value: &ColPlanes, keep_mask: &ColPlanes) -> ColPlanes {
    let mut r: ColPlanes = [0; BOARD_COLS];
    for c in 0..BOARD_COLS {
        r[c] = pext16(value[c], keep_mask[c]);
    }
    r
}

// ============================
// 連鎖ボーナス計算 (`src/scoring.py::chain_power` の Rust 移植)
// ============================

/// n 連鎖目の連鎖ボーナスを返す (1-indexed)。
pub fn chain_power(chain_idx_1based: usize) -> i64 {
    if chain_idx_1based == 0 {
        return 0;
    }
    let zero_idx = chain_idx_1based - 1;
    if zero_idx < CHAIN_POWER_TABLE.len() {
        CHAIN_POWER_TABLE[zero_idx]
    } else {
        let extra = (zero_idx - (CHAIN_POWER_TABLE.len() - 1)) as i64;
        CHAIN_POWER_TABLE[CHAIN_POWER_TABLE.len() - 1] + extra * CHAIN_POWER_INCREMENT
    }
}

/// 1 グループの連結ボーナスを返す (`src/scoring.py::connection_bonus` の Rust 移植)。
pub fn connection_bonus(group_size: u32) -> i64 {
    match group_size {
        0..=3 => 0,
        4 => 0,
        5 => 2,
        6 => 3,
        7 => 4,
        8 => 5,
        9 => 6,
        10 => 7,
        _ => CONNECTION_BONUS_MAX,
    }
}

// ============================
// 厳密得点用: 連結成分分解 (連結ボーナスはグループサイズ依存のため、
// m2/m3 のバッチマスク演算だけでは求まらない。消去確定マスクに対してのみ
// フラッドフィルする、盤面全体で最大78セルなので計算量は無視できる)
// ============================

/// (col, bit) の4連結近傍座標を返す (`src/chain.py::NEIGHBOR_DELTAS` と同一の
/// 上下左右、ビット表現なので上下=同列の隣接bit、左右=隣接列の同bit)。
fn neighbor_positions(col: usize, bit: u32) -> [(Option<usize>, Option<u32>); 4] {
    [
        (Some(col), bit.checked_add(1).filter(|&b| b < BOARD_ROWS as u32)),
        (Some(col), bit.checked_sub(1)),
        (col.checked_add(1).filter(|&c| c < BOARD_COLS), Some(bit)),
        (col.checked_sub(1), Some(bit)),
    ]
}

/// 1 色分の消去マスクを 4 連結成分に分解し、各成分のセル数を返す
/// (`src/chain.py::ChainSimulator._flood_fill` と同一結果になることが前提、
/// パリティテストで担保する)。
fn connected_component_sizes(mask: &ColPlanes) -> Vec<u32> {
    let mut visited: ColPlanes = [0u16; BOARD_COLS];
    let mut sizes: Vec<u32> = Vec::new();
    for start_col in 0..BOARD_COLS {
        for start_bit in 0..BOARD_ROWS as u32 {
            let flag = 1u16 << start_bit;
            if (mask[start_col] & flag) == 0 || (visited[start_col] & flag) != 0 {
                continue;
            }
            visited[start_col] |= flag;
            let mut stack: Vec<(usize, u32)> = vec![(start_col, start_bit)];
            let mut size: u32 = 0;
            while let Some((col, bit)) = stack.pop() {
                size += 1;
                for (ncol_opt, nbit_opt) in neighbor_positions(col, bit) {
                    let (ncol, nbit) = match (ncol_opt, nbit_opt) {
                        (Some(c), Some(b)) => (c, b),
                        _ => continue,
                    };
                    let nflag = 1u16 << nbit;
                    if (mask[ncol] & nflag) != 0 && (visited[ncol] & nflag) == 0 {
                        visited[ncol] |= nflag;
                        stack.push((ncol, nbit));
                    }
                }
            }
            sizes.push(size);
        }
    }
    sizes
}

// ============================
// 連鎖シミュレーション本体
// ============================

/// 1 盤面分の連鎖シミュレーション結果。
#[derive(Clone, Copy, Debug)]
pub struct ChainSimResult {
    pub chain_count: i32,
    pub total_erased: i64,
    pub total_ojama: i64,
    /// 近似得点 (連結ボーナス0近似。`chain_bitboard.BitboardChainScoreResult.score_approx` と同一土俵)。
    pub score_approx: i64,
    /// 厳密得点 (連結ボーナス反映、`src/scoring.py::calculate_chain_score` と同一土俵。
    /// 2026-08-13 追加、後方互換: 既存フィールドは無変更)。
    pub exact_score: i64,
    pub final_board: BitBoard,
}

/// 連鎖シミュレーション (`chain_bitboard.simulate_batch_with_approx_score` の
/// 1 盤面版とビット一致)。
///
/// Args:
///     board: 判定対象の盤面 (破壊しない、コピーして処理)。
///     exclude_hidden_row_from_pop: true で隠し段 (13段目) を消去判定から除外する
///         (幽霊連鎖ルール、本番既定 `GHOST_CHAIN_RULE_ENABLED=True`、
///         `src/production_config.py:160` 参照)。
pub fn simulate_chain(board: &BitBoard, exclude_hidden_row_from_pop: bool) -> ChainSimResult {
    let pop_mask: u16 = if exclude_hidden_row_from_pop {
        POP_MASK_12BIT
    } else {
        FULL_MASK
    };
    let mut colors = board.colors;
    let mut ojama = board.ojama;

    let mut chain_count: i32 = 0;
    let mut total_erased: i64 = 0;
    let mut total_ojama: i64 = 0;
    let mut score_approx: i64 = 0;
    let mut exact_score: i64 = 0;

    for step in 0..MAX_CHAIN_STEPS {
        let mut color_pop_masks: [ColPlanes; NUM_TRACKED_COLORS] = Default::default();
        for i in 0..NUM_TRACKED_COLORS {
            let masked = plane_mask(&colors[i], pop_mask);
            color_pop_masks[i] = get_mask_pop(&masked);
        }
        let mut color_union: ColPlanes = [0; BOARD_COLS];
        for i in 0..NUM_TRACKED_COLORS {
            color_union = plane_or(&color_union, &color_pop_masks[i]);
        }
        if !plane_any_nonzero(&color_union) {
            break;
        }

        let ojama_cleared = plane_and(&expand(&color_union), &ojama);
        let full_pop_mask = plane_or(&color_union, &ojama_cleared);

        let erased_count = plane_popcount(&color_union);
        let ojama_count = plane_popcount(&ojama_cleared);
        total_erased += erased_count;
        total_ojama += ojama_count;
        chain_count += 1;

        let num_colors_step = color_pop_masks
            .iter()
            .filter(|m| plane_any_nonzero(m))
            .count();
        let chain_bonus_this_step = chain_power(step + 1);
        let raw_bonus = chain_bonus_this_step + COLOR_BONUS_LUT[num_colors_step];
        let multiplier = raw_bonus.clamp(MIN_BONUS_MULTIPLIER, MAX_BONUS_MULTIPLIER);
        let step_score = erased_count * BASE_SCORE_PER_PUYO * multiplier;
        score_approx += step_score;

        // 厳密得点: 連結ボーナスはグループサイズ依存なので、消去確定マスク
        // (color_pop_masks[i]、m2/m3演算で既に「このステップで消える色iのセル」
        // だけに絞られている) を連結成分分解して初めて求まる
        // (`src/scoring.py::calculate_step_score` の conn 相当)。
        let exact_conn_bonus_this_step: i64 = color_pop_masks
            .iter()
            .flat_map(connected_component_sizes)
            .map(connection_bonus)
            .sum();
        let exact_raw_bonus =
            chain_bonus_this_step + exact_conn_bonus_this_step + COLOR_BONUS_LUT[num_colors_step];
        let exact_multiplier = exact_raw_bonus.clamp(MIN_BONUS_MULTIPLIER, MAX_BONUS_MULTIPLIER);
        let exact_step_score = erased_count * BASE_SCORE_PER_PUYO * exact_multiplier;
        exact_score += exact_step_score;

        let keep_mask = plane_not_masked(&full_pop_mask, FULL_MASK);
        for i in 0..NUM_TRACKED_COLORS {
            colors[i] = pext_plane(&colors[i], &keep_mask);
        }
        ojama = pext_plane(&ojama, &keep_mask);
    }

    ChainSimResult {
        chain_count,
        total_erased,
        total_ojama,
        score_approx,
        exact_score,
        final_board: BitBoard { colors, ojama },
    }
}

// ============================
// Board <-> グリッド (13x6 色コード配列) 変換
// ============================

/// 色コードグリッド (行優先、13*6=78要素) からビットボードを構築する。
///
/// COLOR_UNKNOWN (10) はエンジンに入れない (空扱い、task指示通り)。
pub fn board_from_grid(grid: &[u8]) -> BitBoard {
    debug_assert_eq!(grid.len(), BOARD_ROWS * BOARD_COLS);
    let mut b = BitBoard::default();
    for row in 0..BOARD_ROWS {
        let bit_index = (BOARD_ROWS - 1 - row) as u32;
        for col in 0..BOARD_COLS {
            let v = grid[row * BOARD_COLS + col];
            if v >= COLOR_RED && v <= COLOR_PURPLE {
                b.colors[(v - COLOR_RED) as usize][col] |= 1u16 << bit_index;
            } else if v == COLOR_OJAMA {
                b.ojama[col] |= 1u16 << bit_index;
            }
            // COLOR_EMPTY / COLOR_UNKNOWN はビットを立てない (= 空扱い)
        }
    }
    b
}

/// ビットボードから色コードグリッド (行優先、78要素) を復元する。
pub fn board_to_grid(b: &BitBoard) -> Vec<u8> {
    let mut grid = vec![COLOR_EMPTY; BOARD_ROWS * BOARD_COLS];
    for col in 0..BOARD_COLS {
        for bit in 0..BOARD_ROWS as u32 {
            let row = BOARD_ROWS - 1 - bit as usize;
            let mask = 1u16 << bit;
            for ci in 0..NUM_TRACKED_COLORS {
                if b.colors[ci][col] & mask != 0 {
                    grid[row * BOARD_COLS + col] = COLOR_RED + ci as u8;
                }
            }
            if b.ojama[col] & mask != 0 {
                grid[row * BOARD_COLS + col] = COLOR_OJAMA;
            }
        }
    }
    grid
}

// ============================
// 設置列挙 (`src/indicators_v2.py::_enumerate_placements` / `_place_pair_to_board` 準拠)
// ============================

/// 縦置き回転種別 (TOP=ペア第1要素が上、BOT=ペア第1要素が下)。
pub const ROTATION_VERTICAL_TOP_UP: u8 = 0;
pub const ROTATION_HORIZONTAL_TOP_LEFT: u8 = 1;
pub const ROTATION_VERTICAL_BOT_UP: u8 = 2;
pub const ROTATION_HORIZONTAL_BOT_LEFT: u8 = 3;
/// 22 配置 (4 回転 × 6 列、横置きは 5 列)。
pub const NUM_ROTATIONS: u8 = 4;

/// 指定列の高さ (積み上げ数) を返す。
///
/// `src/board.py::Board.height_of` と同一の意味論 (トップの物理行から算出、
/// popcount ではなく最上位セットビット位置を使う。列に浮きぷよ由来の
/// ギャップがあっても Python 版と一致させるため)。
fn height_of(b: &BitBoard, col: usize) -> u32 {
    let mut occ: u16 = 0;
    for ci in 0..NUM_TRACKED_COLORS {
        occ |= b.colors[ci][col];
    }
    occ |= b.ojama[col];
    if occ == 0 {
        0
    } else {
        (16 - occ.leading_zeros()) as u32
    }
}

fn set_cell(b: &mut BitBoard, col: usize, bit: u32, color: u8) {
    let mask = 1u16 << bit;
    if color >= COLOR_RED && color <= COLOR_PURPLE {
        b.colors[(color - COLOR_RED) as usize][col] |= mask;
    } else if color == COLOR_OJAMA {
        b.ojama[col] |= mask;
    }
}

/// col 列の積み上がり最上段に color を1個落とす
/// (`src/indicators_v2.py::_drop_one_color`+`_drop_row`、
/// `scripts/mc_counter_estimator.py::_drop_one_puyo` と同一仕様)。
/// 列が満杯 (height>=BOARD_ROWS) なら None。
pub fn drop_one(board: &BitBoard, col: usize, color: u8) -> Option<BitBoard> {
    let height = height_of(board, col);
    if height >= BOARD_ROWS as u32 {
        return None;
    }
    let mut b = *board;
    set_cell(&mut b, col, height, color);
    Some(b)
}

/// 窒息判定 (`src/board.py::Board.is_dead` と同一の意味論、隠し段は含まない)。
pub fn is_dead(b: &BitBoard) -> bool {
    let mask = 1u16 << DEATH_BIT_INDEX;
    for ci in 0..NUM_TRACKED_COLORS {
        if b.colors[ci][DEATH_COL] & mask != 0 {
            return true;
        }
    }
    b.ojama[DEATH_COL] & mask != 0
}

/// 1 設置の座標 (呼び出し側の手順復元用)。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Placement {
    pub col: u8,
    pub rotation: u8,
}

/// ペア (top, bot) を盤面に配置する (`_place_pair_to_board` 準拠)。
///
/// Returns: 配置後盤面。列が満杯等で置けない場合は None。
fn place_pair(board: &BitBoard, pair: (u8, u8), col: usize, rotation: u8) -> Option<BitBoard> {
    let (top, bot) = pair;
    if top == COLOR_EMPTY || bot == COLOR_EMPTY {
        return None;
    }
    let mut b = *board;
    match rotation {
        ROTATION_VERTICAL_TOP_UP | ROTATION_VERTICAL_BOT_UP => {
            if col >= BOARD_COLS {
                return None;
            }
            let (upper, lower) = if rotation == ROTATION_VERTICAL_TOP_UP {
                (top, bot)
            } else {
                (bot, top)
            };
            // _drop_two_in_column: height <= BOARD_ROWS-2 (2つ分の空きが必要)
            let h = height_of(&b, col);
            if h > (BOARD_ROWS - 2) as u32 {
                return None;
            }
            set_cell(&mut b, col, h, lower);
            set_cell(&mut b, col, h + 1, upper);
            Some(b)
        }
        ROTATION_HORIZONTAL_TOP_LEFT | ROTATION_HORIZONTAL_BOT_LEFT => {
            if col >= BOARD_COLS - 1 {
                return None;
            }
            let (left, right) = if rotation == ROTATION_HORIZONTAL_TOP_LEFT {
                (top, bot)
            } else {
                (bot, top)
            };
            let h1 = height_of(&b, col);
            if h1 >= BOARD_ROWS as u32 {
                return None;
            }
            set_cell(&mut b, col, h1, left);
            let h2 = height_of(&b, col + 1);
            if h2 >= BOARD_ROWS as u32 {
                return None;
            }
            set_cell(&mut b, col + 1, h2, right);
            Some(b)
        }
        _ => None,
    }
}

/// 22 配置すべてを試し、配置後盤面 (発火前) と設置座標のペアを返す。
///
/// Args:
///     filter_dead: true で窒息する配置を除外する
///         (`_enumerate_placement_boards` 準拠、既定でビームサーチが使う経路)。
///         false は `_enumerate_placements` 準拠 (窒息盤面も含む)。
pub fn enumerate_placements(
    board: &BitBoard,
    pair: (u8, u8),
    filter_dead: bool,
) -> Vec<(Placement, BitBoard)> {
    let mut results = Vec::with_capacity(22);
    for rotation in 0..NUM_ROTATIONS {
        let max_col = if rotation == ROTATION_VERTICAL_TOP_UP || rotation == ROTATION_VERTICAL_BOT_UP {
            BOARD_COLS
        } else {
            BOARD_COLS - 1
        };
        for col in 0..max_col {
            if let Some(placed) = place_pair(board, pair, col, rotation) {
                if filter_dead && is_dead(&placed) {
                    continue;
                }
                results.push((
                    Placement {
                        col: col as u8,
                        rotation,
                    },
                    placed,
                ));
            }
        }
    }
    results
}

// ============================
// 単体テスト (cargo test、Python パリティテストの前段の最低限の回帰保護)
// ============================

#[cfg(test)]
mod tests {
    use super::*;

    fn empty_grid() -> [u8; BOARD_ROWS * BOARD_COLS] {
        [0u8; BOARD_ROWS * BOARD_COLS]
    }

    #[test]
    fn chain_power_matches_table() {
        assert_eq!(chain_power(1), 0);
        assert_eq!(chain_power(2), 8);
        assert_eq!(chain_power(19), 512);
        // 19連鎖超は線形延長 (+32)
        assert_eq!(chain_power(20), 544);
    }

    #[test]
    fn four_connected_puyo_erase_with_score_40() {
        // 2x2 に同色を並べると4連結で消える (連鎖1、色数1、ボーナス=0、40点)
        let mut grid = empty_grid();
        grid[12 * BOARD_COLS + 0] = COLOR_RED;
        grid[11 * BOARD_COLS + 0] = COLOR_RED;
        grid[12 * BOARD_COLS + 1] = COLOR_RED;
        grid[11 * BOARD_COLS + 1] = COLOR_RED;
        let board = board_from_grid(&grid);
        let result = simulate_chain(&board, false);
        assert_eq!(result.chain_count, 1);
        assert_eq!(result.total_erased, 4);
        assert_eq!(result.score_approx, 40);
        // 消えた後は全セル空
        assert_eq!(board_to_grid(&result.final_board), vec![0u8; BOARD_ROWS * BOARD_COLS]);
    }

    #[test]
    fn five_connected_puyo_exact_score_includes_connection_bonus() {
        // 縦3+横2 (計5連結) の1グループ、size=5 → connection_bonus=2。
        // score_approx (連結ボーナス0近似) は 50、exact_score は 100 になるはず
        // (`src/scoring.py::connection_bonus(5) == 2`,
        // raw_bonus=0(chain)+2(conn)+0(color)=2 → multiplier=2 → 5*10*2=100)。
        let mut grid = empty_grid();
        grid[12 * BOARD_COLS + 0] = COLOR_RED;
        grid[11 * BOARD_COLS + 0] = COLOR_RED;
        grid[10 * BOARD_COLS + 0] = COLOR_RED;
        grid[12 * BOARD_COLS + 1] = COLOR_RED;
        grid[11 * BOARD_COLS + 1] = COLOR_RED;
        let board = board_from_grid(&grid);
        let result = simulate_chain(&board, false);
        assert_eq!(result.chain_count, 1);
        assert_eq!(result.total_erased, 5);
        assert_eq!(result.score_approx, 50, "近似 (連結ボーナス0) は 5*10*1=50");
        assert_eq!(result.exact_score, 100, "厳密 (連結ボーナス2) は 5*10*2=100");
    }

    #[test]
    fn connected_component_sizes_splits_two_separate_groups() {
        // 同色でも盤面上非隣接なら別成分として分解されること
        // (connection_bonus 計算の前提の直接確認)。
        let mut mask: ColPlanes = [0u16; BOARD_COLS];
        mask[0] = 0b1111; // col0: bit0-3 (4連結、下段側)
        mask[3] = 0b1_0000_0000; // col3: bit8 (孤立1セル、離れた列)
        let mut sizes = connected_component_sizes(&mask);
        sizes.sort_unstable();
        assert_eq!(sizes, vec![1, 4]);
    }

    #[test]
    fn drop_one_lands_on_top_of_existing_stack() {
        let mut grid = empty_grid();
        grid[12 * BOARD_COLS + 0] = COLOR_RED;
        grid[11 * BOARD_COLS + 0] = COLOR_RED;
        let board = board_from_grid(&grid);
        let dropped = drop_one(&board, 0, COLOR_BLUE).expect("列0はまだ空きがある");
        let grid2 = board_to_grid(&dropped);
        assert_eq!(grid2[10 * BOARD_COLS + 0], COLOR_BLUE, "積み上げ最上段の1つ上に乗るはず");
        assert_eq!(grid2[12 * BOARD_COLS + 0], COLOR_RED, "既存の下2段は変化しない");
    }

    #[test]
    fn drop_one_on_full_column_returns_none() {
        let mut grid = empty_grid();
        for row in 0..BOARD_ROWS {
            grid[row * BOARD_COLS + 0] = COLOR_RED;
        }
        let board = board_from_grid(&grid);
        assert!(drop_one(&board, 0, COLOR_BLUE).is_none(), "満杯列には置けない");
    }

    #[test]
    fn three_connected_puyo_do_not_erase() {
        let mut grid = empty_grid();
        grid[12 * BOARD_COLS + 0] = COLOR_RED;
        grid[11 * BOARD_COLS + 0] = COLOR_RED;
        grid[12 * BOARD_COLS + 1] = COLOR_RED;
        let board = board_from_grid(&grid);
        let result = simulate_chain(&board, false);
        assert_eq!(result.chain_count, 0);
        assert_eq!(result.total_erased, 0);
    }

    #[test]
    fn ghost_chain_rule_hides_hidden_row_from_pop() {
        // 13段目 (row0) に4連結を作っても幽霊連鎖ルール有効時は消えない
        let mut grid = empty_grid();
        grid[0 * BOARD_COLS + 0] = COLOR_RED;
        grid[0 * BOARD_COLS + 1] = COLOR_RED;
        grid[1 * BOARD_COLS + 0] = COLOR_RED;
        grid[1 * BOARD_COLS + 1] = COLOR_RED;
        let board = board_from_grid(&grid);
        let with_ghost = simulate_chain(&board, true);
        assert_eq!(with_ghost.chain_count, 0, "幽霊連鎖ルール有効時は13段目は消えない");
        let without_ghost = simulate_chain(&board, false);
        assert_eq!(without_ghost.chain_count, 1, "従来ルールでは13段目も消える");
    }

    #[test]
    fn is_dead_detects_death_row_col() {
        let mut grid = empty_grid();
        grid[DEATH_ROW * BOARD_COLS + DEATH_COL] = COLOR_RED;
        let board = board_from_grid(&grid);
        assert!(is_dead(&board));
    }

    #[test]
    fn enumerate_placements_returns_22_on_empty_board() {
        let board = board_from_grid(&empty_grid());
        let results = enumerate_placements(&board, (COLOR_RED, COLOR_RED + 1), false);
        assert_eq!(results.len(), 22, "空盤面では4回転×6列(横は5列)=22通り");
    }

    #[test]
    fn l_shape_connection_erases() {
        // L字型 (4連結) も消える確認 (m2/m3 定式の縦横混在パス)
        let mut grid = empty_grid();
        grid[12 * BOARD_COLS + 0] = COLOR_RED; // (row12,col0)
        grid[11 * BOARD_COLS + 0] = COLOR_RED; // (row11,col0)
        grid[10 * BOARD_COLS + 0] = COLOR_RED; // (row10,col0)
        grid[10 * BOARD_COLS + 1] = COLOR_RED; // (row10,col1)
        let board = board_from_grid(&grid);
        let result = simulate_chain(&board, false);
        assert_eq!(result.chain_count, 1);
        assert_eq!(result.total_erased, 4);
    }

    #[test]
    fn ojama_adjacent_to_erased_group_is_cleared() {
        // 4連結の周囲に置いたお邪魔が巻き込み消去されるか
        let mut grid = empty_grid();
        grid[12 * BOARD_COLS + 1] = COLOR_RED;
        grid[11 * BOARD_COLS + 1] = COLOR_RED;
        grid[12 * BOARD_COLS + 2] = COLOR_RED;
        grid[11 * BOARD_COLS + 2] = COLOR_RED;
        // 左隣 (col0) にお邪魔を隣接配置 (row12)
        grid[12 * BOARD_COLS + 0] = COLOR_OJAMA;
        // 消去グループから遠い (隣接しない) お邪魔は残るはず
        grid[9 * BOARD_COLS + 5] = COLOR_OJAMA;
        let board = board_from_grid(&grid);
        let result = simulate_chain(&board, false);
        assert_eq!(result.chain_count, 1);
        assert_eq!(result.total_erased, 4);
        assert_eq!(result.total_ojama, 1, "隣接お邪魔1個だけ巻き込み消去される");
        let final_grid = board_to_grid(&result.final_board);
        // 遠いお邪魔 (元 row9,col5) は落下して残っているはず (盤面全体でお邪魔1個)
        let remaining_ojama = final_grid.iter().filter(|&&v| v == COLOR_OJAMA).count();
        assert_eq!(remaining_ojama, 1);
    }

    #[test]
    fn multi_step_chain_counts_two_steps() {
        // 1段目の赤4連結消去 → col1 の青2個が落下 → col2 の青2個と
        // 高さが揃って隣接し4連結になる (2段目発火)、という古典的な
        // 「土台を抜いて上をつなげる」2連鎖の最小構成。
        //
        // col0: 高さ0,1 = R,R (それだけ)
        // col1: 高さ0,1 = R,R / 高さ2,3 = B,B (1段目消去後に高さ0,1へ落下)
        // col2: 高さ0,1 = B,B (最初から)
        let mut grid = empty_grid();
        grid[12 * BOARD_COLS + 0] = COLOR_RED; // col0 高さ0
        grid[11 * BOARD_COLS + 0] = COLOR_RED; // col0 高さ1
        grid[12 * BOARD_COLS + 1] = COLOR_RED; // col1 高さ0
        grid[11 * BOARD_COLS + 1] = COLOR_RED; // col1 高さ1
        grid[10 * BOARD_COLS + 1] = COLOR_BLUE; // col1 高さ2
        grid[9 * BOARD_COLS + 1] = COLOR_BLUE; // col1 高さ3
        grid[12 * BOARD_COLS + 2] = COLOR_BLUE; // col2 高さ0
        grid[11 * BOARD_COLS + 2] = COLOR_BLUE; // col2 高さ1
        let board = board_from_grid(&grid);
        let result = simulate_chain(&board, false);
        assert_eq!(result.chain_count, 2, "1段目消去後に落下した青がcol2の青と繋がって2連鎖");
        assert_eq!(result.total_erased, 8);
    }

    #[test]
    fn enumerate_placements_filter_dead_excludes_death_placement() {
        // col2 (DEATH_COL) を row2 (DEATH_ROWの1つ下、height=10) まで積んで
        // 縦置きで積むと2つ目が DEATH_ROW に到達し窒息する配置を作る
        let mut grid = empty_grid();
        for row in 3..BOARD_ROWS {
            grid[row * BOARD_COLS + DEATH_COL] = COLOR_RED;
        }
        let board = board_from_grid(&grid);
        let all = enumerate_placements(&board, (COLOR_BLUE, COLOR_GREEN), false);
        let filtered = enumerate_placements(&board, (COLOR_BLUE, COLOR_GREEN), true);
        assert!(
            filtered.len() < all.len(),
            "filter_dead=true は窒息する配置を除外するはず (all={}, filtered={})",
            all.len(),
            filtered.len()
        );
        for (_placement, placed) in &filtered {
            assert!(!is_dead(placed), "filter_dead=true の結果に窒息盤面が残っている");
        }
    }
}
