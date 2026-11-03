//! 可視12段の消去、13段を含む落下、段別得点。盤面のビット列 (field.rs) をそのまま使う。
use crate::bits::V;
use crate::field::{column_of, with_column, Field, VISIBLE_MASK, WIDTH};

const CHAIN: [i32; 20] = [0, 0, 8, 16, 32, 64, 96, 128, 160, 192,
    224, 256, 288, 320, 352, 384, 416, 448, 480, 512];
const COLOR: [i32; 5] = [0, 0, 3, 6, 12];
const LONG: [i32; 12] = [0, 0, 0, 0, 0, 2, 3, 4, 5, 6, 7, 10];
/// 色番号 4〜7 のビット (seen の数え方)。
const COLOR_BITS: u32 = 0xF0;
/// 1 回の連鎖の段数の上限 (盤のぷよ 78 個 ÷ 4)。
pub const MAX_CHAIN: usize = 20;

#[inline(always)]
fn near(b: V) -> V { (b.down() | b.up() | b.left() | b.right()) & V::new(VISIBLE_MASK) }

pub fn adjacent(bits: u128) -> u128 { near(V::new(bits)).get() }

pub fn expand(bits: u128, mask: u128) -> u128 {
    let (mut b, m) = (V::new(bits), V::new(mask));
    loop {
        let next = (b | near(b)) & m;
        if next.same(b) { return b.get(); }
        b = next;
    }
}

/// start から mask の中でつながる個数が n 以上か。幅優先で広げ、n に届いたら打ち切る。
pub fn reaches(start: u128, mask: u128, n: u32) -> bool {
    let (mut b, m) = (V::new(start), V::new(mask));
    loop {
        let next = (b | near(b)) & m;
        if has_at_least(next.get(), n) { return true; }
        if next.same(b) { return false; }
        b = next;
    }
}

/// 立っているビットが n 個以上か (下位ビットを n−1 回消して残りを見る。ビット数えより速い)。
fn has_at_least(mut bits: u128, n: u32) -> bool {
    for _ in 1..n { bits &= bits.wrapping_sub(1); }
    bits != 0
}

/// 可視部の赤・青・黄・緑・おじゃまの集合 (色番号 4〜7 は 3 ビット目が立つ)。
pub fn planes(f: &Field) -> [u128; 5] {
    let [p0, p1, p2] = f.planes;
    let (low, high) = (p2 & !p1 & VISIBLE_MASK, p2 & p1 & VISIBLE_MASK);
    [low & !p0, low & p0, high & !p0, high & p0, p0 & !p1 & !p2 & VISIBLE_MASK]
}

/// 可視部の色ぷよが、上・下・左・右の隣と同じ色かどうか (全色まとめて)。
#[derive(Clone, Copy)]
pub struct SameColor { u: V, d: V, l: V, r: V }

pub fn same_color(f: &Field) -> SameColor {
    let [p0, p1, p2] = f.planes.map(V::new);
    let colored = p2 & V::new(VISIBLE_MASK);
    // 色ぷよ同士なら 3 ビット目は等しいので、残り 2 ビットが等しければ同じ色。
    let u = (colored & colored.up()).andnot((p0 ^ p0.up()) | (p1 ^ p1.up()));
    let l = (colored & colored.left()).andnot((p0 ^ p0.left()) | (p1 ^ p1.left()));
    SameColor { u, d: u.down(), l, r: l.right() }
}

/// bits から同じ色の隣をたどって広げる。
#[inline(always)]
fn grow_v(mut b: V, k: &SameColor) -> V {
    loop {
        let next = b | (b.up() & k.u) | (b.down() & k.d) | (b.left() & k.l) | (b.right() & k.r);
        if next.same(b) { return b; }
        b = next;
    }
}

pub fn grow(bits: u128, k: &SameColor) -> u128 { grow_v(V::new(bits), k).get() }

pub fn links(f: &Field) -> [i32; 2] {
    let k = same_color(f);
    let three = ((k.u | k.d) & (k.l | k.r)) | (k.u & k.d) | (k.l & k.r);
    let two = (k.u | k.l).andnot(grow_v(three, &k));
    [two.get().count_ones() as i32, three.get().count_ones() as i32]
}

/// 4 個以上のかたまり全部 (全色)。核 = 同色の隣が 3 つ以上のセル、または隣が 2 つ以上のセル同士が
/// 同色で隣り合う所。4 個以上のかたまりには必ず核があり、3 個以下には無い (格子に三角形は無い)。
fn vanishing(k: &SameColor) -> u128 {
    let (u, d, l, r) = (k.u, k.d, k.l, k.r);
    let two = (u & d) | (l & r) | ((u | d) & (l | r));
    let three = (u & d & (l | r)) | (l & r & (u | d));
    let pair = (u & two.up()) | (d & two.down()) | (l & two.left()) | (r & two.right());
    let seed = three | (two & pair);
    if seed.is_zero() { 0 } else { grow_v(seed, k).get() }
}

/// かたまりが 2 つあれば 8 個以上なので、7 個以下は 1 つのかたまり。
const SINGLE_GROUP_MAX: u32 = 7;

fn erase(f: &Field, nth: usize) -> (u128, i32) {
    let k = same_color(f);
    let erased = vanishing(&k);
    if erased == 0 { return (0, 0); }
    let total = erased.count_ones();
    let (count, mut bonus, colors) = if total <= SINGLE_GROUP_MAX {
        (total as i32, LONG[total as usize], 1)
    } else {
        split_groups(f, erased, &k)
    };
    bonus = (bonus + CHAIN[nth.min(CHAIN.len() - 1)] + COLOR[colors]).clamp(1, 999);
    let [p0, p1, p2] = f.planes;
    let nuisance = p0 & !p1 & !p2 & VISIBLE_MASK;
    (erased | (adjacent(erased) & nuisance), count * 10 * bonus)
}

/// 消えるセルをかたまりに分け、(個数, 長さの加点, 色数) を返す。
fn split_groups(f: &Field, erased: u128, k: &SameColor) -> (i32, i32, usize) {
    let (mut count, mut bonus, mut seen) = (0, 0, 0u8);
    let mut remaining = erased;
    while remaining != 0 {
        let low = remaining & remaining.wrapping_neg();
        let group = grow(low, k);
        remaining &= !group;
        let size = group.count_ones() as usize;
        count += size as i32;
        bonus += LONG[size.min(LONG.len() - 1)];
        seen |= 1 << group_color(f, low);
    }
    (count, bonus, (seen as u32 & COLOR_BITS).count_ones() as usize)
}

/// 1 ビットの位置のセルの色番号。
fn group_color(f: &Field, low: u128) -> u32 {
    let s = low.trailing_zeros();
    (0..3).fold(0, |c, i| c | ((((f.planes[i] >> s) & 1) as u32) << i))
}

/// 列の値 col から、gone の位置を抜いて下へ詰める (上の空きも含めて 1 段ずつ下げる)。
fn squeeze(mut col: u32, mut gone: u32) -> u32 {
    while gone != 0 {
        let b = 31 - gone.leading_zeros();
        gone &= !(1 << b);
        let below = (1u32 << b) - 1;
        col = (col & below) | ((col >> 1) & !below);
    }
    col
}

/// 消えた位置を列ごとに詰める (浮きは浮いたまま)。
fn drop_erased(f: &mut Field, erased: u128) {
    for x in 0..WIDTH {
        let gone = column_of(erased, x);
        if gone == 0 { continue; }
        for p in &mut f.planes { *p = with_column(*p, x, squeeze(column_of(*p, x), gone)); }
    }
    f.refresh_heights();
}

/// 連鎖を最後まで進め、段数と合計得点を返す。段ごとの得点は scores に入れる。
pub fn simulate_into(f: &mut Field, scores: &mut [i32; MAX_CHAIN]) -> (usize, i32) {
    let (mut n, mut total) = (0, 0);
    loop {
        let (erased, score) = erase(f, n + 1);
        if score == 0 {
            // CoreField::simulate/vanishDrop は消去がなくても全ぷよ数から高さを更新する。
            f.count_heights();
            return (n, total);
        }
        if n < MAX_CHAIN { scores[n] = score; }
        n += 1;
        total += score;
        drop_erased(f, erased);
    }
}

pub fn simulate_stats(f: &mut Field) -> (usize, i32) {
    simulate_into(f, &mut [0; MAX_CHAIN])
}

pub fn simulate(f: &mut Field) -> Vec<i32> {
    let mut scores = [0; MAX_CHAIN];
    let (n, _) = simulate_into(f, &mut scores);
    scores[..n.min(MAX_CHAIN)].to_vec()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::{HEIGHT, STRIDE, VISIBLE};

    #[test]
    fn hidden_row_drops_but_does_not_vanish() {
        let mut f = Field::default();
        for x in 0..4 { f.set_cell(x, HEIGHT - 1, 4); }
        assert!(simulate(&mut f).is_empty());
        f = Field::default();
        for _ in 0..4 { f.drop_color(0, 4, false); }
        f.drop_color(0, 5, false);
        assert_eq!(simulate(&mut f), vec![40]);
        assert_eq!(f.cell(0, 0), 5);
        assert_eq!(f.heights[0], 1);
    }

    #[test]
    fn floating_cells_recount_even_without_a_chain() {
        let mut f = Field::default();
        for y in 1..HEIGHT { f.set_cell(2, y, 4 + (y % 4) as u8); }
        assert!(simulate(&mut f).is_empty());
        assert_eq!(f.heights[2], VISIBLE);
        assert!(f.dead());
    }

    #[test]
    fn erase_keeps_gaps_and_other_columns() {
        // 列 1: 下から 赤 赤 赤 赤 青 (空き) 黄。列 0 と 2 は触らない。
        let mut f = Field::default();
        for y in 0..4 { f.set_cell(1, y, 4); }
        f.set_cell(1, 4, 5);
        f.set_cell(1, 6, 6);
        f.set_cell(0, 0, 7);
        f.set_cell(2, 0, 5);
        let scores = simulate(&mut f);
        assert_eq!(scores, vec![40]);
        assert_eq!((f.cell(1, 0), f.cell(1, 1), f.cell(1, 2)), (5, 0, 6));
        assert_eq!((f.cell(0, 0), f.cell(2, 0)), (7, 5));
        assert_eq!(f.heights[1], 2);
    }

    /// 全部のかたまりを塗りつぶして 4 個以上を残す素朴な方法 (元の実装)。
    fn vanish_naive(m: u128) -> u128 {
        let (mut remaining, mut out) = (m, 0);
        while remaining != 0 {
            let group = expand(remaining & remaining.wrapping_neg(), remaining);
            remaining &= !group;
            if group.count_ones() >= 4 { out |= group; }
        }
        out
    }

    /// 元の実装の links (色ごと)。
    fn links_naive(f: &Field) -> [i32; 2] {
        let mut counts = [0, 0];
        for m in planes(f).into_iter().take(4) {
            let (u, d, l, r) = ((m >> 1) & m, (m << 1) & m, (m << STRIDE) & m, (m >> STRIDE) & m);
            let three = ((u | d) & (l | r)) | (u & d) | (l & r);
            let two = (u | l) & !expand(three, m);
            counts[0] += two.count_ones() as i32;
            counts[1] += three.count_ones() as i32;
        }
        counts
    }

    #[test]
    fn vanishing_and_links_match_naive_on_random_fields() {
        let codes = [0u8, 0, 1, 4, 5, 6, 7, 4, 5, 6, 7];
        let mut state = 0x2545_f491_4f6c_dd1du64;
        for round in 0..100_000 {
            let mut f = Field::default();
            // 色の数を 2〜4 に絞って、かたまりが出やすい盤も混ぜる
            let palette = 2 + round % 3;
            for x in 0..6 {
                for y in 0..13 {
                    state ^= state << 13; state ^= state >> 7; state ^= state << 17;
                    let c = codes[(state % codes.len() as u64) as usize];
                    let c = if c >= 4 { 4 + (c - 4) % palette as u8 } else { c };
                    f.set_cell(x, y, c);
                }
            }
            let naive = planes(&f).into_iter().take(4).fold(0, |a, m| a | vanish_naive(m));
            assert_eq!(vanishing(&same_color(&f)), naive, "round={round}");
            assert_eq!(links(&f), links_naive(&f), "round={round}");
        }
    }
}
