//! 正の既定評価。項の順は EvalRawFeatures と同じ。
use crate::{chain, field::{self, Field, VISIBLE, WIDTH}, form};

pub const WEIGHTS: [i32; 15] = [1000, 289, -200, 200, -100, -100, -100, 50,
    150, 250, -50, 0, -250, -250, -250];
const QUIET_DROP: usize = 3;
const CENTER: usize = 2;
const NUISANCE: u8 = 1;
const VANISH: u32 = 4;

fn chi(h: &[usize; WIDTH], x: usize) -> i32 {
    // 右と左へ、x 以下 (と x 未満) の高さが続く列数。
    let side = |indices: &mut dyn Iterator<Item = usize>, strict: bool| -> i32 {
        indices.take_while(|&i| if strict { h[i] < h[x] } else { h[i] <= h[x] }).count() as i32
    };
    side(&mut (x + 1..WIDTH), false) + side(&mut (x + 1..WIDTH), true)
        + side(&mut (0..x).rev(), false) + side(&mut (0..x).rev(), true)
}

/// start から mask の中でつながる個数が n 以上か。幅優先で広げ、n に届いたら打ち切る
/// (C++ の countConnectedPuyosMax4 と同じ判定)。
fn connected_at_least(start: u128, mask: u128, n: u32) -> bool {
    chain::reaches(start, mask, n)
}

fn quiet_bounds(f: &Field) -> (usize, usize) {
    let (mut lo, mut hi) = (CENTER, CENTER);
    for x in CENTER + 1..WIDTH {
        if f.heights[x] > VISIBLE - 1 { break; }
        hi = x;
    }
    for x in (0..CENTER).rev() {
        if f.heights[x] > VISIBLE - 1 { break; }
        lo = x;
    }
    (lo, hi)
}

fn quiet_features(f: &Field) -> [i32; 15] {
    let mut best = None;
    let mut raw = [0; 15];
    let (lo, hi) = quiet_bounds(f);
    let masks = chain::planes(f);
    for x in lo..=hi {
        let h = f.heights[x];
        let drops = QUIET_DROP.min(VISIBLE.saturating_sub(h));
        for c in 4..=7 {
            let mut plan = *f;
            let mut connected = masks[(c - 4) as usize];
            for key in 1..=drops {
                plan.drop_color(x, c, false);
                connected |= field::bit(x, h + key - 1);
                if !connected_at_least(field::bit(x, h), connected, VANISH) { continue; }
                let mut remain = plan;
                let (chains, _) = chain::simulate_stats(&mut remain);
                if chains <= 1 { break; }
                let links = chain::links(&remain);
                let mut value = [0; 15];
                value[0] = chains as i32; value[1] = h as i32;
                value[2] = key as i32; value[3] = chi(&f.heights, x);
                value[8] = links[0]; value[9] = links[1];
                let score = total(&value);
                if best.is_none_or(|v| score > v) { best = Some(score); raw = value; }
                break;
            }
        }
    }
    raw
}

pub fn features(f: &Field) -> [i32; 15] {
    let (form_score, mirrored) = form::evaluate(f);
    let mut raw = quiet_features(f);
    let h = f.heights.map(|v| v as i32);
    let average = h.iter().sum::<i32>() / WIDTH as i32;
    for (i, &height) in h.iter().enumerate() {
        let index = if mirrored { WIDTH - 1 - i } else { i };
        raw[4] += (height - average - if index < 3 { 1 } else { -1 }).abs();
    }
    raw[5] = (h[1] - h[0]).max(0) + (h[4] - h[5]).max(0);
    for i in 1..WIDTH - 1 {
        raw[5] += (h[i - 1].min(h[i + 1]) - h[i]).max(0);
        raw[6] += (h[i] - h[i - 1].max(h[i + 1])).max(0);
    }
    raw[7] = form_score;
    let links = chain::links(f);
    raw[8] += links[0]; raw[9] += links[1];
    raw[11] = (h[0] + h[1]).max(h[3] + h[4] + h[5]) - h[2];
    raw[12] = f.mask(NUISANCE).count_ones() as i32;
    raw
}

pub fn total(raw: &[i32; 15]) -> i32 {
    raw.iter().zip(WEIGHTS).map(|(&v, w)| v * w).sum()
}

pub fn evaluate(f: &Field) -> i32 { total(&features(f)) }

pub fn action(tear: bool, chains: usize) -> i32 {
    i32::from(tear) * WEIGHTS[13] + chains as i32 * WEIGHTS[14]
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn empty_evaluation() {
        assert_eq!(evaluate(&Field::default()), -600);
        assert_eq!(action(true, 2), -750);
    }
}
