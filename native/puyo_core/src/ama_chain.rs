//! ama専用の固定長連鎖計算。消去式は既存bitboardと同一。
//! 参照: puyoai core/field_bits.h の expand/findVanishingBits。
//! MIT: Copyright (c) 2013- mayah, peria, test_lockit, shinh, pascal,
//! and other puyoai contributors. 全文は ../licenses/puyoai-MIT.txt。
//! 既存消去式の帰属: Copyright (c) 2023 citrus610 (../licenses/ama-MIT.txt)。
use super::*;

const COLUMN_BITS: usize = 16;
const PLANE_MASK: u128 = 0x1fff_1fff_1fff_1fff_1fff_1fff;

fn pack(p: &ColPlanes) -> u128 {
    p.iter().enumerate().fold(0, |a, (x, &v)| a | ((v as u128) << (x * COLUMN_BITS)))
}

fn unpack(p: u128) -> ColPlanes {
    std::array::from_fn(|x| (p >> (x * COLUMN_BITS)) as u16)
}

fn adjacent(p: u128) -> u128 {
    ((p << 1) | (p >> 1) | (p << COLUMN_BITS) | (p >> COLUMN_BITS)) & PLANE_MASK
}

/// quietが必要とする「4個以上か」だけを、参照のexpand4同様に3回で判定する。
#[cfg(feature = "ama-fast-connected")]
pub(super) fn connected_four(plane: &ColPlanes, x: usize, y: usize) -> u32 {
    const EXPANSIONS: usize = 3;
    let mask = pack(&plane.map(|v| v & POP_MASK_12BIT));
    let mut group = 1u128 << (x * COLUMN_BITS + y);
    for _ in 0..EXPANSIONS { group = (group | adjacent(group)) & mask; }
    group.count_ones().min((EXPANSIONS + 1) as u32)
}

fn vanish(m: u128) -> u128 {
    let (u, d, l, r) = ((m >> 1) & m, (m << 1) & m,
        (m << COLUMN_BITS) & m, (m >> COLUMN_BITS) & m);
    let three = (u & d & (l | r)) | (l & r & (u | d));
    let two = (u & d) | (l & r) | ((u | d) & (l | r));
    let core = three | (adjacent(two) & two);
    (core | adjacent(core)) & m
}

/// サイズ列を確保せず、同じ連結成分から整数ボーナスを加算する。
fn bonus(mut remaining: u128) -> i64 {
    let mut result = 0;
    while remaining != 0 {
        let mut group = remaining & remaining.wrapping_neg();
        loop {
            let next = (group | adjacent(group)) & remaining;
            if next == group { break; }
            group = next;
        }
        result += connection_bonus(group.count_ones());
        remaining &= !group;
    }
    result
}

fn masks(colors: &[ColPlanes; NUM_TRACKED_COLORS], mask: u16) -> [u128; NUM_TRACKED_COLORS] {
    #[cfg(all(feature = "ama-fast-avx2", target_arch = "x86_64"))]
    if std::arch::is_x86_feature_detected!("avx2") {
        return unsafe { masks_avx2(colors, mask) };
    }
    colors.map(|p| vanish(pack(&p.map(|v| v & mask))))
}

/// 128bitの独立した盤面を2色ずつ処理する。列外は常にゼロ。
#[cfg(all(feature = "ama-fast-avx2", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn masks_avx2(colors: &[ColPlanes; NUM_TRACKED_COLORS], mask: u16) -> [u128; NUM_TRACKED_COLORS] {
    use std::arch::x86_64::*;
    let mut result = [0; NUM_TRACKED_COLORS];
    for first in (0..NUM_TRACKED_COLORS).step_by(2) {
        let pair = [pack(&colors[first]), colors.get(first + 1).map_or(0, pack)];
        let m = _mm256_and_si256(_mm256_loadu_si256(pair.as_ptr().cast()), _mm256_set1_epi16(mask as i16));
        let u = _mm256_and_si256(_mm256_srli_epi16::<1>(m), m);
        let d = _mm256_and_si256(_mm256_slli_epi16::<1>(m), m);
        let l = _mm256_and_si256(_mm256_slli_si256::<2>(m), m);
        let r = _mm256_and_si256(_mm256_srli_si256::<2>(m), m);
        let ud = _mm256_and_si256(u, d);
        let lr = _mm256_and_si256(l, r);
        let udo = _mm256_or_si256(u, d);
        let lro = _mm256_or_si256(l, r);
        let three = _mm256_or_si256(_mm256_and_si256(ud, lro), _mm256_and_si256(lr, udo));
        let two = _mm256_or_si256(_mm256_or_si256(ud, lr), _mm256_and_si256(udo, lro));
        let core = _mm256_or_si256(three, _mm256_and_si256(edges_avx2(two), two));
        let popped = _mm256_and_si256(_mm256_or_si256(core, edges_avx2(core)), m);
        let mut out = [0u128; 2];
        _mm256_storeu_si256(out.as_mut_ptr().cast(), popped);
        result[first] = out[0];
        if first + 1 < NUM_TRACKED_COLORS { result[first + 1] = out[1]; }
    }
    result
}

#[cfg(all(feature = "ama-fast-avx2", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn edges_avx2(m: std::arch::x86_64::__m256i) -> std::arch::x86_64::__m256i {
    use std::arch::x86_64::*;
    _mm256_or_si256(_mm256_or_si256(_mm256_slli_epi16::<1>(m), _mm256_srli_epi16::<1>(m)),
        _mm256_or_si256(_mm256_slli_si256::<2>(m), _mm256_srli_si256::<2>(m)))
}

fn step<const PROTECT_HIDDEN_OJAMA: bool>(board: &mut BitBoard, hidden: bool,
    index: usize, result: &mut ChainSimResult) -> bool {
    let mask = if hidden { POP_MASK_12BIT } else { FULL_MASK };
    let pops = masks(&board.colors, mask);
    let union = pops.iter().fold(0, |a, &p| a | p);
    if union == 0 { return false; }
    let garbage_plane = if PROTECT_HIDDEN_OJAMA {
        pack(&board.ojama.map(|p| p & mask))
    } else { pack(&board.ojama) };
    let garbage = (union | adjacent(union)) & garbage_plane;
    let erased = union.count_ones() as i64;
    let colors = pops.iter().filter(|&&p| p != 0).count();
    let base = chain_power(index + 1) + COLOR_BONUS_LUT[colors];
    let connection: i64 = pops.iter().map(|&p| bonus(p)).sum();
    result.chain_count += 1;
    result.total_erased += erased;
    result.total_ojama += garbage.count_ones() as i64;
    result.score_approx += erased * BASE_SCORE_PER_PUYO * base.clamp(MIN_BONUS_MULTIPLIER, MAX_BONUS_MULTIPLIER);
    result.exact_score += erased * BASE_SCORE_PER_PUYO * (base + connection).clamp(MIN_BONUS_MULTIPLIER, MAX_BONUS_MULTIPLIER);
    let keep = unpack(PLANE_MASK & !(union | garbage));
    gravity(board, &keep);
    true
}

fn gravity(board: &mut BitBoard, keep: &ColPlanes) {
    #[cfg(all(feature = "ama-fast-gravity", target_arch = "x86_64"))]
    if std::arch::is_x86_feature_detected!("bmi2") {
        unsafe { gravity_bmi2(board, keep); }
        return;
    }
    for p in &mut board.colors { *p = pext_plane(p, keep); }
    board.ojama = pext_plane(&board.ojama, keep);
}

/// 命令判定と関数境界を各セルから一回の落下へまとめる。処理対象は同じ36値。
#[cfg(all(feature = "ama-fast-gravity", target_arch = "x86_64"))]
#[target_feature(enable = "bmi2")]
unsafe fn gravity_bmi2(board: &mut BitBoard, keep: &ColPlanes) {
    use std::arch::x86_64::_pext_u32;
    for x in 0..BOARD_COLS {
        let mask = (keep[x] & FULL_MASK) as u32;
        for p in &mut board.colors { p[x] = _pext_u32(p[x] as u32, mask) as u16; }
        board.ojama[x] = _pext_u32(board.ojama[x] as u32, mask) as u16;
    }
}

pub(super) fn simulate(board: &BitBoard, hidden: bool) -> ChainSimResult {
    simulate_inner::<false>(board, hidden)
}

#[cfg(feature = "b3-fast-chain")]
pub(super) fn simulate_b3(board: &BitBoard) -> ChainSimResult {
    simulate_inner::<true>(board, true)
}

fn simulate_inner<const PROTECT_HIDDEN_OJAMA: bool>(board: &BitBoard, hidden: bool) -> ChainSimResult {
    let mut work = *board;
    let mut result = ChainSimResult { chain_count: 0, total_erased: 0, total_ojama: 0,
        score_approx: 0, exact_score: 0, final_board: work };
    for index in 0..MAX_CHAIN_STEPS {
        if !step::<PROTECT_HIDDEN_OJAMA>(&mut work, hidden, index, &mut result) { break; }
    }
    result.final_board = work;
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn random_chain_all_fields_match() {
        #[cfg(target_arch = "x86_64")]
        eprintln!("avx2={} bmi2={}", std::arch::is_x86_feature_detected!("avx2"),
            std::arch::is_x86_feature_detected!("bmi2"));
        const CASES: usize = 100_000;
        let mut state = 0x123456789abcdefu64;
        for _ in 0..CASES {
            let mut grid = [0; BOARD_ROWS * BOARD_COLS];
            for cell in &mut grid {
                state ^= state << 13; state ^= state >> 7; state ^= state << 17;
                *cell = match state % 8 { 6 => COLOR_OJAMA, 7 => 0, c => c as u8 };
            }
            let board = board_from_grid(&grid);
            #[cfg(feature = "b3-fast-chain")]
            {
                let a = simulate_chain_core::<true>(&board, true, false).0;
                let b = simulate_b3(&board);
                assert_eq!((a.chain_count, a.total_erased, a.total_ojama, a.score_approx, a.exact_score, a.final_board),
                    (b.chain_count, b.total_erased, b.total_ojama, b.score_approx, b.exact_score, b.final_board));
            }
            for hidden in [false, true] {
                let a = simulate_chain(&board, hidden);
                let b = simulate(&board, hidden);
                assert_eq!((a.chain_count, a.total_erased, a.total_ojama, a.score_approx, a.exact_score, a.final_board),
                    (b.chain_count, b.total_erased, b.total_ojama, b.score_approx, b.exact_score, b.final_board));
            }
        }
    }
}
