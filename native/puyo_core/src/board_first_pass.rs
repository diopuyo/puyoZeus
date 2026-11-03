//! read_board の 1st pass (背景パッチ NCC・HSV 中央値・ガード分類) のネイティブ実装 (2026-10-25)。
//!
//! `src/live_board_vectorized.py` の `_tier1_pending` / `_first_pass_patch_fp` が、盤面 1 枚 (72 セル) を
//! Python のセル単位ループ (NCC・中央値・指紋・ガード分類で 1 コマ 100 回超の numpy/OpenCV 呼出) で
//! 判定していたものを、1 回の Rust 呼出にまとめる。判定の式と分岐の順序は Python 版のまま。
//!
//! ## 判定の流れ (Python 版と同じ順序)
//!
//! 1. tier1: 背景パッチとの NCC が閾値以上のセルを「空の候補」とする (`skip_tier1` のときは省略)。
//! 2. 候補のうち、HSV 単独分類でもぷよ色でないものだけを tier1 空とする (W13 根治 案 2、旗で省略可)。
//! 3. 白ハイライトがあるセルは tier1 空を取り消す (旗で省略可)。
//! 4. 残り全セルの HSV 中央値 3 値と背景 FP の距離を求める (これが 2nd pass へ渡る bg 距離)。
//! 5. 距離が閾値未満のセルは tier2 の候補。HSV 単独分類でもぷよ色でなければ tier2 空。
//!
//! ## 出力の一致について (numpy との差)
//!
//! - 整数だけで決まる値 (HSV 中央値・距離・ガード分類・ハイライト) は Python 版とビット一致する。
//! - NCC は numpy の `mean` (ペアワイズ総和) と `std` の経路を同じ順序で再現する。ただし
//!   `np.dot` は OpenBLAS の CPU ごとのカーネルで総和の順序が変わる (Python 側も機種で最終桁が変わる)
//!   ため、内積 2 つ (`dot(a_c, a_c)` と `dot(a_c, b_c)`) だけは最終桁がずれうる (相対 1e-15 程度)。
//!   判定 (NCC >= 閾値) が反転するのは閾値から 1e-15 以内の場合に限られ、検証台が全セルの差の最大と
//!   閾値からの最小の余裕を記録する。
//! - cvtColor は移植しない。HSV は Python 側 (cv2) が作った画像をそのまま受け取る。
//!
//! ## 前提 (満たさなければ Err を返し、Python 側が従来経路へ戻す)
//!
//! frame / hsv は同じ形の C 連続 (H, W, 3) uint8。矩形はすべて画像内で面積 1 以上。
//! 背景データの長さは矩形の面積の総和と一致する。

use numpy::ndarray::ArrayView3;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::hsv_classify::{
    classify_cell_from_stats, collect_cell_stats, median_u8, CellStats, HsvParams, PatchStats, RangeRow,
    COLOR_EMPTY,
};

/// 色コード UNKNOWN (src/board.py の COLOR_UNKNOWN)。ガードは EMPTY / UNKNOWN を「ぷよ色でない」とする。
const COLOR_UNKNOWN: i32 = 10;
/// チャンネル数 (HSV)。
const CHANNELS: usize = 3;
/// 出力の状態: 2nd pass へ進む (空ではない)。
pub const STATUS_CLASSIFY: u8 = 0;
/// 出力の状態: tier1 で空と確定。
pub const STATUS_TIER1_EMPTY: u8 = 1;
/// 出力の状態: tier2 で空と確定。
pub const STATUS_TIER2_EMPTY: u8 = 2;
/// 背景フラグのビット: tier1 を実行してよい (bg パッチの V 中央値が採取失敗の下限以上)。
const FLAG_TIER1_ENABLED: u8 = 1;
/// 背景フラグのビット: 整形後の bg パッチが有効 (採取失敗のゼロパッチではない)。
const FLAG_BG_VALID: u8 = 2;
/// numpy のペアワイズ総和が 8 本のアキュムレータで処理するブロックの上限 (PW_BLOCKSIZE)。
const PW_BLOCK_SIZE: usize = 128;
/// numpy のペアワイズ総和の展開幅。
const PW_UNROLL: usize = 8;
/// 内積の展開幅 (アキュムレータ数)。
const DOT_LANES: usize = 4;
/// 色相の周期 (H は 0-180 で循環)。
const HUE_PERIOD: i32 = 180;

/// 判定パラメータ。Python 側の定数・設定を単一情報源にするため、値はすべて呼出側から受け取る。
#[derive(FromPyObject)]
#[pyo3(from_item_all)]
pub struct FirstPassParams {
    pub ncc_threshold: f64,
    pub ncc_std_min: f64,
    pub ncc_uniform_fallback: f64,
    pub bg_threshold: f64,
    pub h_weight: f64,
    pub s_weight: f64,
    pub v_weight: f64,
    pub skip_tier1: bool,
    pub tier1_guard: bool,
    pub highlight_override: bool,
    pub highlight_v_min: f64,
    pub highlight_s_max: f64,
    pub highlight_y_top: f64,
    pub highlight_y_bottom: f64,
    pub highlight_min_ratio: f64,
}

#[derive(Clone, Copy)]
struct Rect {
    x1: usize,
    y1: usize,
    x2: usize,
    y2: usize,
}

impl Rect {
    fn height(&self) -> usize {
        self.y2 - self.y1
    }
    fn width(&self) -> usize {
        self.x2 - self.x1
    }
    fn area(&self) -> usize {
        self.height() * self.width()
    }
}

/// numpy の `add.reduce` (連続 float64) と同じ順序の総和。
///
/// numpy の `pairwise_sum` を写したもの: 8 未満は逐次、128 以下は 8 本のアキュムレータ、
/// それ以上は 8 の倍数で二分して再帰する。`np.mean` / `np.std` の総和と最終桁まで一致する。
pub fn pairwise_sum(a: &[f64]) -> f64 {
    pairwise_sum_by(0, a.len(), &|i| a[i])
}

/// `pairwise_sum` の本体。要素を関数 `value(i)` で与える (配列を作らずに「平均を引いて二乗」などを足せる)。
/// 足す順序は numpy と同じ。`start` は添字の起点、`n` は個数。
fn pairwise_sum_by<F: Fn(usize) -> f64>(start: usize, n: usize, value: &F) -> f64 {
    if n < PW_UNROLL {
        let mut res = -0.0_f64;
        for i in 0..n {
            res += value(start + i);
        }
        return res;
    }
    if n <= PW_BLOCK_SIZE {
        let mut r = [0.0_f64; PW_UNROLL];
        for (lane, acc) in r.iter_mut().enumerate() {
            *acc = value(start + lane);
        }
        let mut i = PW_UNROLL;
        while i < n - (n % PW_UNROLL) {
            for (lane, acc) in r.iter_mut().enumerate() {
                *acc += value(start + i + lane);
            }
            i += PW_UNROLL;
        }
        let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        while i < n {
            res += value(start + i);
            i += 1;
        }
        return res;
    }
    let mut half = n / 2;
    half -= half % PW_UNROLL;
    pairwise_sum_by(start, half, value) + pairwise_sum_by(start + half, n - half, value)
}

/// 内積 `Σ (a[i] - mean) * b[i]`。4 本のアキュムレータで足す
/// (OpenBLAS の ddot も複数本で足すが、順序は CPU で異なる)。`a` は u8 の画素 (float64 へ上げて引く)。
fn centered_dot(a: &[u8], mean: f64, b: &[f64]) -> f64 {
    let mut acc = [0.0_f64; DOT_LANES];
    let chunks = a.len() / DOT_LANES;
    for c in 0..chunks {
        for lane in 0..DOT_LANES {
            let i = c * DOT_LANES + lane;
            acc[lane] += (f64::from(a[i]) - mean) * b[i];
        }
    }
    let mut tail = 0.0_f64;
    for i in chunks * DOT_LANES..a.len() {
        tail += (f64::from(a[i]) - mean) * b[i];
    }
    ((acc[0] + acc[1]) + (acc[2] + acc[3])) + tail
}

/// 背景 1 セル分の事前計算 (Python の `CellPatchFingerprint._as_bg_for` の結果)。
struct BgCell<'a> {
    centered: &'a [f64],
    selfdot: f64,
    std: f64,
    flags: u8,
    median: [i32; CHANNELS],
}

/// 作業用バッファ (セルごとに確保し直さない)。
#[derive(Default)]
struct Scratch {
    pixels: Vec<u8>,
}

/// HSV パッチを (H, W, 3) の行優先順で 1 本の列に並べる (`ravel()` と同じ順)。
fn gather_u8(hsv: &[u8], width: usize, rect: Rect, out: &mut Vec<u8>) {
    out.clear();
    for y in rect.y1..rect.y2 {
        let start = (y * width + rect.x1) * CHANNELS;
        let end = (y * width + rect.x2) * CHANNELS;
        out.extend_from_slice(&hsv[start..end]);
    }
}

/// `_compute_ncc_prepared` と同じ式の NCC。背景が無効なら 0.0 (呼出側で判定)。
fn ncc_prepared(
    hsv: &[u8],
    width: usize,
    rect: Rect,
    bg: &BgCell,
    p: &FirstPassParams,
    scratch: &mut Scratch,
) -> f64 {
    if bg.flags & FLAG_BG_VALID == 0 {
        return 0.0;
    }
    gather_u8(hsv, width, rect, &mut scratch.pixels);
    let pixels = &scratch.pixels;
    let count = pixels.len() as f64;
    // 画素は 0..=255 の整数なので総和は整数で正確。numpy のペアワイズ総和も同じ値になる (順序に依らない)。
    let integer_sum: u64 = pixels.iter().map(|&v| u64::from(v)).sum();
    let mean = integer_sum as f64 / count;
    // 分散の総和は小数になるので、numpy と同じ順序 (ペアワイズ) で足す。配列は作らず要素を式で与える。
    let square_sum = pairwise_sum_by(0, pixels.len(), &|i| {
        let d = f64::from(pixels[i]) - mean;
        d * d
    });
    let a_std = (square_sum / count).sqrt();
    if a_std < p.ncc_std_min || bg.std < p.ncc_std_min {
        return p.ncc_uniform_fallback;
    }
    let denom = (square_sum * bg.selfdot).sqrt();
    if denom == 0.0 {
        return p.ncc_uniform_fallback;
    }
    let ncc = centered_dot(pixels, mean, bg.centered) / denom;
    if ncc.is_nan() {
        p.ncc_uniform_fallback
    } else {
        ncc
    }
}

/// 白ハイライトの有無 (`detect_highlight_blob` と同じ式)。
fn has_highlight(hsv: &[u8], width: usize, rect: Rect, p: &FirstPassParams) -> bool {
    let (h, w) = (rect.height(), rect.width());
    let total = h * w;
    if total == 0 {
        return false;
    }
    let y_start = (h as f64 * p.highlight_y_top) as usize;
    let y_end = (h as f64 * p.highlight_y_bottom) as usize;
    if y_start >= y_end {
        return false;
    }
    let mut white = 0usize;
    for y in (rect.y1 + y_start)..(rect.y1 + y_end) {
        let row = (y * width + rect.x1) * CHANNELS;
        for x in 0..w {
            let s = f64::from(hsv[row + x * CHANNELS + 1]);
            let v = f64::from(hsv[row + x * CHANNELS + 2]);
            if v >= p.highlight_v_min && s <= p.highlight_s_max {
                white += 1;
            }
        }
    }
    (white as f64 / total as f64) >= p.highlight_min_ratio
}

/// HSV 3 チャンネルの中央値 (`_median_hsv_3ch` と同値。偶数個は中央 2 値の平均を 0 方向へ切り捨て)。
/// 分類と共有する統計 (素の H・S・V のヒストグラム) から求める。
fn median_hsv(st: &PatchStats) -> [i32; CHANNELS] {
    [
        median_u8(&st.h_hist, st.n) as i32,
        median_u8(&st.s_hist, st.n) as i32,
        median_u8(&st.v_hist, st.n) as i32,
    ]
}

/// 重み付き HSV 距離 (`CellFingerprint.distance_to` と同じ演算順序)。
fn fingerprint_distance(cur: [i32; CHANNELS], bg: [i32; CHANNELS], p: &FirstPassParams) -> f64 {
    let mut dh = (cur[0] - bg[0]).abs();
    dh = dh.min(HUE_PERIOD - dh);
    let ds = (cur[1] - bg[1]).abs();
    let dv = (cur[2] - bg[2]).abs();
    p.h_weight * f64::from(dh) + p.s_weight * f64::from(ds) + p.v_weight * f64::from(dv)
}

/// ガード分類: HSV 単独の分類色がぷよ色でなければ true。統計は呼出側が共有する。
fn guard_says_not_puyo(
    st: &CellStats,
    bgr: &ArrayView3<u8>,
    hsv: &ArrayView3<u8>,
    rect: Rect,
    ranges: &[RangeRow],
    hp: &HsvParams,
) -> bool {
    let color = classify_cell_from_stats(st, bgr, hsv, rect.y1, rect.y2, rect.x1, rect.x2, ranges, hp);
    color == COLOR_EMPTY || color == COLOR_UNKNOWN
}

/// セルごとの HSV 統計 (遅延計算)。tier1 のガードと tier2 の中央値・ガードが同じ走査を共有する。
struct StatsCache<'a> {
    hsv: &'a ArrayView3<'a, u8>,
    params: &'a HsvParams,
    items: Vec<Option<Box<CellStats>>>,
}

impl<'a> StatsCache<'a> {
    fn new(hsv: &'a ArrayView3<'a, u8>, params: &'a HsvParams, count: usize) -> Self {
        StatsCache { hsv, params, items: (0..count).map(|_| None).collect() }
    }

    fn get(&mut self, index: usize, rect: Rect) -> &CellStats {
        if self.items[index].is_none() {
            let st = collect_cell_stats(self.hsv, rect.y1, rect.y2, rect.x1, rect.x2, self.params);
            self.items[index] = Some(Box::new(st));
        }
        self.items[index].as_deref().expect("直前に作成済み")
    }
}

/// 引数の形を検査して矩形を読む。
fn read_rects(rects: &PyReadonlyArray2<i32>, height: usize, width: usize) -> PyResult<Vec<Rect>> {
    let view = rects.as_array();
    if view.shape()[1] != 4 {
        return Err(PyValueError::new_err("rects は (N, 4) ではない"));
    }
    let mut out = Vec::with_capacity(view.shape()[0]);
    for i in 0..view.shape()[0] {
        let values: Vec<i64> = (0..4).map(|k| i64::from(view[[i, k]])).collect();
        let in_range = values[0] >= 0
            && values[1] >= 0
            && values[0] < values[2]
            && values[1] < values[3]
            && values[2] <= width as i64
            && values[3] <= height as i64;
        if !in_range {
            return Err(PyValueError::new_err("矩形が画像の外か面積 0"));
        }
        out.push(Rect {
            x1: values[0] as usize,
            y1: values[1] as usize,
            x2: values[2] as usize,
            y2: values[3] as usize,
        });
    }
    Ok(out)
}

/// tier1 の候補 (NCC が閾値以上のセル) を求める。ncc_out には計算したセルだけ値を書く。
fn tier1_candidates(
    hsv: &[u8],
    width: usize,
    rects: &[Rect],
    bgs: &[BgCell],
    p: &FirstPassParams,
    ncc_out: &mut [f64],
) -> Vec<usize> {
    let mut scratch = Scratch::default();
    let mut pending = Vec::new();
    if p.skip_tier1 {
        return pending;
    }
    for (i, rect) in rects.iter().enumerate() {
        if bgs[i].flags & FLAG_TIER1_ENABLED == 0 {
            continue;
        }
        let ncc = ncc_prepared(hsv, width, *rect, &bgs[i], p, &mut scratch);
        ncc_out[i] = ncc;
        if ncc >= p.ncc_threshold {
            pending.push(i);
        }
    }
    pending
}

/// 背景データの配列の長さを矩形と照合し、セルごとの `BgCell` へ割り付ける。
fn build_bg_cells<'a>(
    cells: &[Rect],
    centered_all: &'a [f64],
    selfdot: &PyReadonlyArray1<f64>,
    stds: &PyReadonlyArray1<f64>,
    flags: &PyReadonlyArray1<u8>,
    medians: &PyReadonlyArray2<i32>,
) -> PyResult<Vec<BgCell<'a>>> {
    let n = cells.len();
    let (selfdot, stds, flags, medians) = (selfdot.as_array(), stds.as_array(), flags.as_array(), medians.as_array());
    let total_area: usize = cells.iter().map(|r| r.area() * CHANNELS).sum();
    if centered_all.len() != total_area
        || selfdot.len() != n
        || stds.len() != n
        || flags.len() != n
        || medians.shape() != [n, CHANNELS]
    {
        return Err(PyValueError::new_err("背景データの長さが矩形と一致しない"));
    }
    let mut bgs: Vec<BgCell> = Vec::with_capacity(n);
    let mut offset = 0usize;
    for (i, rect) in cells.iter().enumerate() {
        let len = rect.area() * CHANNELS;
        bgs.push(BgCell {
            centered: &centered_all[offset..offset + len],
            selfdot: selfdot[i],
            std: stds[i],
            flags: flags[i],
            median: [medians[[i, 0]], medians[[i, 1]], medians[[i, 2]]],
        });
        offset += len;
    }
    Ok(bgs)
}

/// 色レンジの配列 (R, 7) を行の列にする。
fn read_range_rows(ranges: &PyReadonlyArray2<i32>) -> Vec<RangeRow> {
    let view = ranges.as_array();
    (0..view.shape()[0])
        .map(|i| {
            let mut row = [0i32; 7];
            for (k, slot) in row.iter_mut().enumerate() {
                *slot = view[[i, k]];
            }
            row
        })
        .collect()
}

/// 判定に使う入力の束。
struct TierInputs<'a> {
    bgr: &'a ArrayView3<'a, u8>,
    hsv: &'a ArrayView3<'a, u8>,
    hsv_flat: &'a [u8],
    width: usize,
    cells: &'a [Rect],
    bgs: &'a [BgCell<'a>],
    ranges: &'a [RangeRow],
    hsv_params: &'a HsvParams,
    params: &'a FirstPassParams,
}

/// tier1 → tier2 の判定。戻りは (状態, bg 距離, NCC)。
fn judge_tiers(inp: &TierInputs) -> (Vec<u8>, Vec<f64>, Vec<f64>) {
    let n = inp.cells.len();
    let mut status = vec![STATUS_CLASSIFY; n];
    let mut distance = vec![f64::NAN; n];
    let mut ncc_out = vec![f64::NAN; n];
    let mut stats = StatsCache::new(inp.hsv, inp.hsv_params, n);
    // tier1: NCC の候補 → ガード → ハイライトによる取り消し
    let pending = tier1_candidates(inp.hsv_flat, inp.width, inp.cells, inp.bgs, inp.params, &mut ncc_out);
    for &i in &pending {
        if inp.params.tier1_guard {
            let st = stats.get(i, inp.cells[i]);
            if !guard_says_not_puyo(st, inp.bgr, inp.hsv, inp.cells[i], inp.ranges, inp.hsv_params) {
                continue;
            }
        }
        if inp.params.highlight_override && has_highlight(inp.hsv_flat, inp.width, inp.cells[i], inp.params) {
            continue;
        }
        status[i] = STATUS_TIER1_EMPTY;
    }
    // tier2: tier1 空でないセルの距離と、距離が近いセルのガード
    for i in 0..n {
        if status[i] == STATUS_TIER1_EMPTY {
            continue;
        }
        let st = stats.get(i, inp.cells[i]);
        let d = fingerprint_distance(median_hsv(&st.full), inp.bgs[i].median, inp.params);
        distance[i] = d;
        if d < inp.params.bg_threshold
            && guard_says_not_puyo(st, inp.bgr, inp.hsv, inp.cells[i], inp.ranges, inp.hsv_params)
        {
            status[i] = STATUS_TIER2_EMPTY;
        }
    }
    (status, distance, ncc_out)
}

/// 盤面 1 枚の 1st pass。戻り値は (状態 u8[N], 2nd pass へ渡す bg 距離 f64[N], NCC f64[N])。
///
/// 距離は tier1 で空と確定したセルだけ NaN。NCC は計算しなかったセルが NaN。
#[pyfunction]
#[pyo3(signature = (frame, hsv, rects, bg_centered, bg_selfdot, bg_std, bg_flags, bg_median, ranges, hsv_params, params))]
#[allow(clippy::too_many_arguments)]
pub fn read_board_first_pass(
    py: Python<'_>,
    frame: PyReadonlyArray3<u8>,
    hsv: PyReadonlyArray3<u8>,
    rects: PyReadonlyArray2<i32>,
    bg_centered: PyReadonlyArray1<f64>,
    bg_selfdot: PyReadonlyArray1<f64>,
    bg_std: PyReadonlyArray1<f64>,
    bg_flags: PyReadonlyArray1<u8>,
    bg_median: PyReadonlyArray2<i32>,
    ranges: PyReadonlyArray2<i32>,
    hsv_params: HsvParams,
    params: FirstPassParams,
) -> PyResult<(Py<PyArray1<u8>>, Py<PyArray1<f64>>, Py<PyArray1<f64>>)> {
    let bgr_view = frame.as_array();
    let hsv_view = hsv.as_array();
    if bgr_view.shape() != hsv_view.shape() || bgr_view.shape()[2] != CHANNELS {
        return Err(PyValueError::new_err("frame と hsv は同じ (H, W, 3) ではない"));
    }
    let (height, width) = (bgr_view.shape()[0], bgr_view.shape()[1]);
    let hsv_flat = hsv_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("hsv が C 連続ではない"))?;
    let cells = read_rects(&rects, height, width)?;
    let centered_all = bg_centered
        .as_slice()
        .map_err(|_| PyValueError::new_err("bg_centered が連続ではない"))?;
    let bgs = build_bg_cells(&cells, centered_all, &bg_selfdot, &bg_std, &bg_flags, &bg_median)?;
    let range_rows = read_range_rows(&ranges);
    let inputs = TierInputs {
        bgr: &bgr_view,
        hsv: &hsv_view,
        hsv_flat,
        width,
        cells: &cells,
        bgs: &bgs,
        ranges: &range_rows,
        hsv_params: &hsv_params,
        params: &params,
    };
    let (status, distance, ncc_out) = judge_tiers(&inputs);
    Ok((
        status.into_pyarray_bound(py).unbind(),
        distance.into_pyarray_bound(py).unbind(),
        ncc_out.into_pyarray_bound(py).unbind(),
    ))
}

/// numpy の総和と同じ順序の総和 (検証台が numpy の実装差を検出するための公開)。
#[pyfunction]
pub fn pairwise_sum_f64(values: PyReadonlyArray1<f64>) -> PyResult<f64> {
    let slice = values
        .as_slice()
        .map_err(|_| PyValueError::new_err("連続した配列ではない"))?;
    Ok(pairwise_sum(slice))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pairwise_sum_matches_naive_for_small_integers() {
        for n in 0..300usize {
            let values: Vec<f64> = (0..n).map(|i| (i % 7) as f64).collect();
            let naive: f64 = values.iter().sum();
            assert_eq!(pairwise_sum(&values), naive, "n={n}");
        }
    }

    #[test]
    fn centered_dot_matches_naive_for_small_integers() {
        let a: Vec<u8> = (0..37).map(|i| (i * 3 % 200) as u8).collect();
        let b: Vec<f64> = (0..37).map(|i| (i % 5) as f64).collect();
        let naive: f64 = a.iter().zip(&b).map(|(x, y)| (f64::from(*x) - 2.0) * y).sum();
        assert_eq!(centered_dot(&a, 2.0, &b), naive);
    }

    #[test]
    fn pairwise_by_equals_slice_version() {
        let values: Vec<f64> = (0..1728).map(|i| ((i * 37 % 101) as f64) * 0.37).collect();
        assert_eq!(pairwise_sum_by(0, values.len(), &|i| values[i]), pairwise_sum(&values));
    }

    #[test]
    fn fingerprint_distance_wraps_hue() {
        let p = FirstPassParams {
            ncc_threshold: 0.92,
            ncc_std_min: 1e-6,
            ncc_uniform_fallback: 1.0,
            bg_threshold: 50.0,
            h_weight: 0.5,
            s_weight: 1.0,
            v_weight: 1.0,
            skip_tier1: false,
            tier1_guard: true,
            highlight_override: true,
            highlight_v_min: 220.0,
            highlight_s_max: 50.0,
            highlight_y_top: 0.15,
            highlight_y_bottom: 0.55,
            highlight_min_ratio: 0.04,
        };
        // H の差 170 は循環して 10。0.5*10 + 3 + 4 = 12
        assert_eq!(fingerprint_distance([175, 13, 24], [5, 10, 20], &p), 12.0);
    }
}
