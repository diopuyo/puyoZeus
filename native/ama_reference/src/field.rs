//! puyoai CoreField と ama_reference/placement.cc の盤面・配置規則。
//! MIT: Copyright (c) puyoai contributors. 帰属は NOTICE.md を参照。
//! 盤面は C++ の BitField と同じく、色番号の 3 ビットを 3 枚のビット列で持つ
//! (列 x・行 y のセルは位置 (x+1)*16 + (y+1)。壁は持たず hash の時だけ足す)。
pub const WIDTH: usize = 6;
pub const HEIGHT: usize = 13;
pub const VISIBLE: usize = 12;
pub const JSON_COLORS: [u8; 5] = [0, 4, 6, 7, 5];
pub const STRIDE: usize = 16;
pub const PLANES: usize = 3;
const COLUMN_CELLS: u128 = (1 << HEIGHT) - 1;
const COLUMN_VISIBLE: u128 = (1 << VISIBLE) - 1;
const WALL_CODE_PLANE: usize = 1;
const HASH_ADD: u64 = 0x9e3779b97f4a7c15;
const HASH_MUL1: u64 = 0xbf58476d1ce4e5b9;
const HASH_MUL2: u64 = 0x94d049bb133111eb;
const CHECK: [&[usize]; WIDTH] = [&[1, 0], &[1], &[], &[3], &[3, 4], &[3, 4, 5]];
const CHECK12: [&[usize]; WIDTH] = [
    &[1, 2, 3, 4, 5], &[2, 3, 4, 5], &[], &[2, 1, 0], &[3, 2, 1, 0], &[4, 3, 2, 1, 0],
];
/// 1 手で置ける組の最大数 (22 = 6 列 × 4 向き − 端の 2)。
pub const MAX_PLACEMENTS: usize = 22;

/// 列 x の最下段 (行 0) の位置。
pub const fn base(x: usize) -> usize { (x + 1) * STRIDE + 1 }

/// セル (x, y) の位置のビット。
pub const fn bit(x: usize, y: usize) -> u128 { 1 << (base(x) + y) }

/// ビット列 bits の列 x の 13 段 (行 0 が最下位)。
pub fn column_of(bits: u128, x: usize) -> u32 { ((bits >> base(x)) & COLUMN_CELLS) as u32 }

/// 列 x の 13 段を value に置き換える。
pub fn with_column(bits: u128, x: usize, value: u32) -> u128 {
    (bits & !(COLUMN_CELLS << base(x))) | ((value as u128 & COLUMN_CELLS) << base(x))
}

const fn rows_mask(column: u128) -> u128 {
    let mut m = 0;
    let mut x = 0;
    while x < WIDTH { m |= column << base(x); x += 1; }
    m
}

/// 全セル (13 段) と可視部 (12 段) の位置。
pub const CELLS: u128 = rows_mask(COLUMN_CELLS);
pub const VISIBLE_MASK: u128 = rows_mask(COLUMN_VISIBLE);

/// 壁 (色番号 2) の位置。列 0 と 7、各列の行 0 と 15。
const WALL: u128 = {
    let mut m = 0u128;
    let mut i = 0;
    while i < 128 {
        let (x, y) = (i / STRIDE, i % STRIDE);
        if x == 0 || x == 7 || y == 0 || y == 15 { m |= 1 << i; }
        i += 1;
    }
    m
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Field {
    pub planes: [u128; PLANES],
    pub heights: [usize; WIDTH],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Placement { pub x: usize, pub r: usize }

impl Default for Field {
    fn default() -> Self { Self { planes: [0; PLANES], heights: [0; WIDTH] } }
}

impl Placement {
    pub fn child(self) -> Option<usize> {
        match self.r {
            0 | 2 => Some(self.x),
            1 if self.x + 1 < WIDTH => Some(self.x + 1),
            3 if self.x > 0 => Some(self.x - 1),
            _ => None,
        }
    }
}

impl Field {
    pub fn from_json(board: &[u8]) -> Result<Self, &'static str> {
        if board.len() != WIDTH * HEIGHT { return Err("盤面要素数不正"); }
        let mut result = Self::default();
        for (i, &color) in board.iter().enumerate() {
            let internal = match color {
                0..=4 => JSON_COLORS[color as usize],
                9 => 1,
                _ => return Err("色番号不正"),
            };
            result.set_cell(i % WIDTH, HEIGHT - 1 - i / WIDTH, internal);
        }
        result.refresh_heights();
        Ok(result)
    }

    pub fn grid(&self) -> Vec<u8> {
        (0..WIDTH * HEIGHT).map(|i| {
            let c = self.cell(i % WIDTH, HEIGHT - 1 - i / WIDTH);
            if c == 1 { 9 } else { JSON_COLORS.iter().position(|&v| v == c).unwrap_or(0) as u8 }
        }).collect()
    }

    /// セル (x, y) の色番号 (0 空・1 おじゃま・4〜7 色)。
    pub fn cell(&self, x: usize, y: usize) -> u8 {
        let s = base(x) + y;
        (0..PLANES).fold(0, |c, i| c | ((((self.planes[i] >> s) & 1) as u8) << i))
    }

    pub fn set_cell(&mut self, x: usize, y: usize, c: u8) {
        let b = bit(x, y);
        for i in 0..PLANES {
            if (c >> i) & 1 != 0 { self.planes[i] |= b; } else { self.planes[i] &= !b; }
        }
    }

    /// 色番号 c のセルの位置 (13 段すべて)。
    pub fn mask(&self, c: u8) -> u128 {
        (0..PLANES).fold(CELLS, |m, i| m & if (c >> i) & 1 != 0 { self.planes[i] } else { !self.planes[i] })
    }

    pub fn occupied(&self) -> u128 { self.planes[0] | self.planes[1] | self.planes[2] }

    /// 列 x の 13 段の占有ビット (行 0 が最下位)。
    pub fn column(&self, x: usize) -> u32 { column_of(self.occupied(), x) }

    /// 高さ = 下から最初の空きの位置 (無ければ 13)。
    pub fn refresh_heights(&mut self) {
        let occupied = self.occupied();
        for x in 0..WIDTH {
            self.heights[x] = (column_of(occupied, x).trailing_ones() as usize).min(HEIGHT);
        }
    }

    /// 高さ = 列のぷよの個数 (CoreField の消去後と同じ)。
    pub fn count_heights(&mut self) {
        let occupied = self.occupied();
        for x in 0..WIDTH { self.heights[x] = column_of(occupied, x).count_ones() as usize; }
    }

    pub fn dead(&self) -> bool { self.heights[2] > VISIBLE - 1 }

    pub fn hash(&self) -> u64 {
        // BitField の壁を含めた3平面を 64bit×6 語として順に混合する。
        let mut h = 0u64;
        for i in 0..PLANES {
            let plane = if i == WALL_CODE_PLANE { self.planes[i] | WALL } else { self.planes[i] };
            for word in [plane as u64, (plane >> 64) as u64] {
                h = h.wrapping_add(word.wrapping_add(HASH_ADD));
                h = (h ^ (h >> 30)).wrapping_mul(HASH_MUL1);
                h = (h ^ (h >> 27)).wrapping_mul(HASH_MUL2);
                h ^= h >> 31;
            }
        }
        h
    }

    pub fn legal(&self, p: Placement) -> bool {
        if p.x >= WIDTH || p.r >= 4 { return false; }
        let Some(_) = p.child() else { return false; };
        if self.heights[p.x] + usize::from(p.r == 2) > VISIBLE { return false; }
        let mut check_x = p.x;
        if p.r == 1 && p.x >= 2 { check_x += 1; }
        if p.r == 3 && p.x <= 2 { check_x -= 1; }
        let mut first12 = None;
        for &x in CHECK[check_x] {
            if self.heights[x] > VISIBLE { return false; }
            if self.heights[x] == VISIBLE && first12.is_none() { first12 = Some(x); }
        }
        let Some(first) = first12 else { return true; };
        if self.heights[1] > VISIBLE - 1 && self.heights[3] > VISIBLE - 1 { return true; }
        for &x in CHECK12[first] {
            if self.heights[x] > VISIBLE - 1 { break; }
            if self.heights[x] == VISIBLE - 1 { return true; }
        }
        false
    }

    pub fn drop_color(&mut self, x: usize, c: u8, allow14: bool) -> bool {
        if self.heights[x] >= HEIGHT { return allow14; }
        self.set_cell(x, self.heights[x], c);
        self.heights[x] += 1;
        true
    }

    pub fn place(&self, p: Placement, pair: [u8; 2]) -> Option<Self> {
        if !self.legal(p) || !pair.iter().all(|c| (4..=7).contains(c)) { return None; }
        let child = p.child()?;
        let mut next = *self;
        let ok = if p.r == 2 {
            next.drop_color(child, pair[1], true) && next.drop_color(p.x, pair[0], false)
        } else {
            next.drop_color(p.x, pair[0], false) && next.drop_color(child, pair[1], true)
        };
        ok.then_some(next)
    }

    /// 置ける手と置いた後の盤を placements と同じ順に渡す。同じ手を 2 回置かないための入口。
    pub fn each_child(&self, pair: [u8; 2], mut visit: impl FnMut(Placement, Field)) {
        let rotations = if pair[0] == pair[1] { 2 } else { 4 };
        for r in 0..rotations {
            for x in 0..WIDTH {
                let p = Placement { x, r };
                if let Some(f) = self.place(p, pair) { visit(p, f); }
            }
        }
    }

    pub fn placements(&self, pair: [u8; 2]) -> Vec<Placement> {
        let mut out = Vec::with_capacity(MAX_PLACEMENTS);
        self.each_child(pair, |p, _| out.push(p));
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn row14_child_disappears() {
        let mut f = Field::default();
        for _ in 0..VISIBLE { f.drop_color(2, 4, false); }
        let next = f.place(Placement { x: 2, r: 0 }, [5, 6]).unwrap();
        assert_eq!(next.heights[2], HEIGHT);
        assert_eq!(next.cell(2, HEIGHT - 1), 5);
        assert!(!f.legal(Placement { x: 2, r: 2 }));
    }

    #[test]
    fn empty_order_and_json_roundtrip() {
        let f = Field::from_json(&[0; WIDTH * HEIGHT]).unwrap();
        assert_eq!(f.placements([4, 5]).len(), 22);
        assert_eq!(f.placements([4, 4]).len(), 11);
        assert_eq!(f.placements([4, 5])[6], Placement { x: 0, r: 1 });
        assert_eq!(Field::from_json(&f.grid()).unwrap(), f);
    }

    #[test]
    fn cell_mask_and_set_roundtrip() {
        let mut f = Field::default();
        for (i, c) in [0u8, 1, 4, 5, 6, 7].into_iter().enumerate() { f.set_cell(i, 3, c); }
        for (i, c) in [0u8, 1, 4, 5, 6, 7].into_iter().enumerate() {
            assert_eq!(f.cell(i, 3), c);
            if c != 0 { assert_eq!(f.mask(c), bit(i, 3)); }
        }
        assert_eq!(f.mask(0).count_ones() as usize, WIDTH * HEIGHT - 5);
    }
}
