//! ama dea210bc の ai/search/beam/{eval,quiet}.cpp の Rust 移植。
//! MIT License, Copyright (c) 2023 citrus610。全文は ../licenses/ama-MIT.txt。
//! 評価値は学習特徴量ではなく原典互換の探索内部整数（正規化しない）。
use crate::bitboard::{
    height_of, simulate_chain_ama as simulate_chain, BitBoard, BOARD_COLS, BOARD_ROWS, DEATH_BIT_INDEX, DEATH_COL, POP_MASK_12BIT,
};

pub const QUIET_DROP: usize = 3;
pub const VISIBLE_HEIGHT: usize = BOARD_ROWS - 1;
pub const DEATH_HEIGHT: u32 = DEATH_BIT_INDEX;

#[derive(Clone, Copy, Debug, PartialEq, serde::Deserialize)]
pub struct Weight {
    pub chain: i32,
    pub y: i32,
    pub key: i32,
    pub chi: i32,
    pub shape: i32,
    pub well: i32,
    pub bump: i32,
    #[serde(default)]
    pub form: i32,
    pub link_2: i32,
    pub link_3: i32,
    pub waste_14: i32,
    pub side: i32,
    pub nuisance: i32,
    pub tear: i32,
    pub waste: i32,
}

impl Weight {
    pub const BUILD: Self = Self {
        chain: 1000,
        y: 289,
        key: -200,
        chi: 200,
        shape: -100,
        well: -100,
        bump: -100,
        form: 50,
        link_2: 150,
        link_3: 250,
        waste_14: -50,
        side: 0,
        nuisance: -250,
        tear: -250,
        waste: -250,
    };
    pub const AC: Self = Self {
        chain: 500,
        y: 100,
        key: -200,
        chi: 100,
        shape: 0,
        well: 0,
        bump: 0,
        form: 0,
        link_2: 50,
        link_3: 150,
        waste_14: -20,
        side: 100,
        nuisance: -250,
        tear: -100,
        waste: -100,
    };
}

pub fn heights(b: &BitBoard) -> [i32; BOARD_COLS] {
    std::array::from_fn(|x| height_of(b, x) as i32)
}

pub fn expand(p: &[u16; BOARD_COLS]) -> [u16; BOARD_COLS] {
    std::array::from_fn(|x| {
        p[x] | (p[x] << 1)
            | (p[x] >> 1)
            | if x > 0 { p[x - 1] } else { 0 }
            | if x + 1 < BOARD_COLS { p[x + 1] } else { 0 }
    })
}

pub fn links(b: &BitBoard) -> (i32, i32) {
    let (mut two, mut three) = (0, 0);
    for plane in &b.colors {
        let m = plane.map(|v| v & POP_MASK_12BIT);
        let u: [u16; BOARD_COLS] = m.map(|v| (v >> 1) & v);
        let d: [u16; BOARD_COLS] = m.map(|v| (v << 1) & v);
        let l: [u16; BOARD_COLS] = std::array::from_fn(|x| if x > 0 { m[x - 1] & m[x] } else { 0 });
        let r: [u16; BOARD_COLS] =
            std::array::from_fn(|x| if x + 1 < BOARD_COLS { m[x + 1] & m[x] } else { 0 });
        let l3 = std::array::from_fn(|x| ((u[x] | d[x]) & (l[x] | r[x])) | (u[x] & d[x]) | (l[x] & r[x]));
        let e = expand(&l3);
        for x in 0..BOARD_COLS {
            three += l3[x].count_ones() as i32;
            two += ((u[x] | l[x]) & !e[x]).count_ones() as i32;
        }
    }
    (two, three)
}

fn chi(h: &[i32; BOARD_COLS], x: usize) -> i32 {
    let mut value = 0;
    for strict in [false, true] {
        for dir in [-1isize, 1] {
            let mut i = x as isize + dir;
            while (0..BOARD_COLS as isize).contains(&i) {
                if h[i as usize] > h[x] || (strict && h[i as usize] == h[x]) {
                    break;
                }
                value += 1;
                i += dir;
            }
        }
    }
    value
}

fn connected(plane: &[u16; BOARD_COLS], x: usize, y: usize) -> u32 {
    #[cfg(feature = "ama-profile-detail")]
    let _timer = crate::ama_profile_clock::Timer::new(4);
    #[cfg(feature = "ama-fast-connected")]
    { crate::bitboard::connected_ama_four(plane, x, y) }
    #[cfg(not(feature = "ama-fast-connected"))]
    { connected_scalar(plane, x, y) }
}

#[cfg(any(test, not(feature = "ama-fast-connected")))]
fn connected_scalar(plane: &[u16; BOARD_COLS], x: usize, y: usize) -> u32 {
    let mut mask = [0; BOARD_COLS];
    mask[x] = 1 << y;
    loop {
        let expanded = expand(&mask);
        let next = std::array::from_fn(|c| expanded[c] & plane[c] & POP_MASK_12BIT);
        if next == mask {
            return mask.iter().map(|v| v.count_ones()).sum();
        }
        mask = next;
    }
}

fn quiet(b: &BitBoard, h: &[i32; BOARD_COLS], colors: &[u8; 4], w: &Weight) -> i32 {
    #[cfg(feature = "ama-profile-detail")]
    let _timer = crate::ama_profile_clock::Timer::new(1);
    let (mut lo, mut hi) = (DEATH_COL, DEATH_COL);
    while lo > 0 && h[lo - 1] < VISIBLE_HEIGHT as i32 {
        lo -= 1;
    }
    while hi + 1 < BOARD_COLS && h[hi + 1] < VISIBLE_HEIGHT as i32 {
        hi += 1;
    }
    let mut best = i32::MIN;
    for x in lo..=hi {
        let max_drop = QUIET_DROP.min(VISIBLE_HEIGHT.saturating_sub(h[x] as usize));
        for &color in colors {
            let mut plan = *b;
            for need in 1..=max_drop {
                plan.colors[color as usize - 1][x] |= 1 << (h[x] as usize + need - 1);
                if connected(&plan.colors[color as usize - 1], x, h[x] as usize) < 4 {
                    continue;
                }
                let result = {
                    #[cfg(feature = "ama-profile-detail")]
                    let _timer = crate::ama_profile_clock::Timer::new(5);
                    simulate_chain(&plan, true)
                };
                if result.chain_count > 1 {
                    let (l2, l3) = links(&result.final_board);
                    best = best.max(
                        result.chain_count * w.chain
                            + h[x] * w.y
                            + need as i32 * w.key
                            + chi(h, x) * w.chi
                            + l2 * w.link_2
                            + l3 * w.link_3,
                    );
                }
                break;
            }
        }
    }
    if best == i32::MIN {
        0
    } else {
        best
    }
}

pub fn waste_14(row14: u8) -> i32 {
    let mut space = 1;
    for x in 3..BOARD_COLS {
        if row14 & (1 << x) != 0 {
            break;
        }
        space += 1;
    }
    for x in (0..2).rev() {
        if row14 & (1 << x) != 0 {
            break;
        }
        space += 1;
    }
    BOARD_COLS as i32 - space
}

pub fn evaluate(b: &BitBoard, colors: &[u8; 4], w: &Weight) -> i32 {
    #[cfg(feature = "ama-profile-detail")]
    let _timer = crate::ama_profile_clock::Timer::new(0);
    let h = heights(b);
    let avg = h.iter().sum::<i32>() / BOARD_COLS as i32;
    let shape: i32 = (0..BOARD_COLS).map(|x| (h[x] - avg - if x < 3 { 1 } else { -1 }).abs()).sum();
    let mut well = (h[1] - h[0]).max(0) + (h[4] - h[5]).max(0);
    let mut bump = 0;
    for x in 1..BOARD_COLS - 1 {
        well += (h[x - 1].min(h[x + 1]) - h[x]).max(0);
        bump += (h[x] - h[x - 1].max(h[x + 1])).max(0);
    }
    let (l2, l3) = links(b);
    let garbage = b.ojama.iter().map(|v| v.count_ones() as i32).sum::<i32>();
    let form = if w.form > 0 { super::ama_form::evaluate(b, &h) } else { 0 };
    // 自由22配置では14段目へ置かないため row14=0、waste_14=0。
    quiet(b, &h, colors, w)
        + shape * w.shape
        + well * w.well
        + bump * w.bump
        + l2 * w.link_2
        + l3 * w.link_3
        + garbage * w.nuisance
        + form * w.form
        + ((h[0] + h[1]).max(h[3] + h[4] + h[5]) - h[2]) * w.side
        + waste_14(0) * w.waste_14
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "ama-fast-connected")]
    #[test]
    fn capped_connection_preserves_quiet_predicate() {
        const CASES: usize = 2048;
        let mut state = 0x123456789abcdefu64;
        for _ in 0..CASES {
            let plane = std::array::from_fn(|_| {
                state ^= state << 13; state ^= state >> 7; state ^= state << 17;
                state as u16 & POP_MASK_12BIT
            });
            for x in 0..BOARD_COLS {
                for y in 0..VISIBLE_HEIGHT {
                    if plane[x] & (1 << y) == 0 { continue; }
                    assert_eq!(connected(&plane, x, y), connected_scalar(&plane, x, y).min(4));
                }
            }
        }
    }
    #[test]
    fn empty_and_links() {
        let mut b = BitBoard::default();
        assert_eq!(evaluate(&b, &[1, 2, 3, 4], &Weight::BUILD), -600);
        b.colors[0][0] = 3;
        assert_eq!(links(&b), (1, 0));
        b.colors[0][0] = 7;
        assert_eq!(links(&b), (0, 1));
        assert_eq!(connected(&b.colors[0], 0, 0), 3);
        assert_eq!(waste_14(0), 0);
        assert_eq!(waste_14((1 << 1) | (1 << 3)), 5);
    }
}
