//! HSV セル分類のネイティブ実装 (2026-08-20)。
//!
//! `src/image_reader.py` の `ColorClassifier.classify` /
//! `_classify_single_patch_no_subregion` / `_compute_stable_h_median` /
//! `_compute_specular_robust_s` / `_median_fast` を **bit-identical に**移植する。
//!
//! ## なぜ Rust 化するか (実測)
//!
//! 認識 1 frame 34.69ms のうち **classify 起因が 19.64ms (57%)**。1 回 0.223ms
//! で中盤では 88 回/frame 呼ばれる。中身 (median) は既に `np.partition` で
//! 最適化済みなので、残っているのは **Python/numpy の呼び出しオーバーヘッド**
//! (median だけで 1,146 回/frame)。1 呼び出しで全セルを処理すればこれが消える。
//!
//! ## bit-identical のための約束 (崩すと認識結果が変わる)
//!
//! 1. **median は np.median と同一定義**: 奇数長は k=n/2 番目の順序統計量、
//!    偶数長は中央 2 値の f64 平均。`int()` の 0 方向切り捨ても同じ。
//!    counting sort で順序統計量を求めるので np.partition と同値。
//! 2. **cvtColor は移植しない**。HSV は Python 側 (cv2) から受け取る。
//!    OpenCV の整数丸めを再現すると 1LSB のズレが median を変え、閾値ぎわで
//!    色判定が反転しうるため、この経路は構造的に避ける。
//! 3. **色範囲は Python の dict 挿入順**で渡す。判定は先勝ちなので順序が結果を変える。
//! 4. **サブ領域 vote の同数タイは出現順で先勝ち** (`Counter.most_common(1)` の
//!    Python 挙動)。厳密に `>` 比較で走査して再現する。
//! 5. `eff_s_min = int(s_min * scale)` の f64 乗算 + 切り捨ても同式。
//!
//! ## classify と _classify_single_patch_no_subregion の差異 (要注意)
//!
//! 赤の拡張範囲 (H 11-18) で R-G 差が不足したときの制御が**違う**:
//!   - `classify`: `red_skipped = True; break` → 外側ループで `continue`
//!     (= その色の残りレンジを飛ばし、次の色へ)
//!   - `_classify_single_patch_no_subregion`: `continue`
//!     (= 同じ色の次のレンジを試す)
//! 統合してはいけない。別関数として移植する。

use numpy::ndarray::ArrayView3;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray2, PyReadonlyArray3};
use pyo3::prelude::*;

/// 色コード (src/board.py と一致させる)
const COLOR_EMPTY: i32 = 0;
const COLOR_RED: i32 = 1;
const COLOR_OJAMA: i32 = 9;

/// Python 側から渡す判定パラメータ束。
/// マジックナンバーを Rust に埋め込まず、Python の定数を単一情報源にする。
/// `from_item_all` により dict のキーから読む (呼び出し側を素の dict で書ける)。
#[derive(FromPyObject)]
#[pyo3(from_item_all)]
pub struct HsvParams {
    pub s_min_scale: f64,
    pub empty_v_threshold: i32,
    pub ojama_s_threshold: i32,
    pub ojama_v_min: i32,
    pub red_green_diff_for_red: i32,
    pub red_hue_wrap_threshold: i32,
    pub red_hue_wrap_corrected_max: i32,
    pub red_hue_low_max: i32,
    pub red_bimodal_min_ratio: f64,
    pub red_extend_h_min: i32,
    pub red_extend_h_max: i32,
    pub specular_v_min: i32,
    pub specular_s_max: i32,
    pub specular_fallback_min_ratio: f64,
    pub enable_red_hue_wrap_fix: bool,
    pub enable_specular_robust_saturation: bool,
    pub subregion_min_h: usize,
    pub subregion_min_w: usize,
}

/// u8 列の median を np.median と同値で返す (f64)。
///
/// counting sort でヒストグラムを作り k 番目の順序統計量を取る。
/// 偶数長は中央 2 値の f64 平均 — `_median_fast` と同式。
fn median_u8(hist: &[u32; 256], n: usize) -> f64 {
    debug_assert!(n > 0);
    let k = n / 2;
    if n % 2 == 1 {
        // k 番目 (0-origin) の順序統計量
        let mut acc = 0usize;
        for (v, c) in hist.iter().enumerate() {
            acc += *c as usize;
            if acc > k {
                return v as f64;
            }
        }
        unreachable!("ヒストグラムの総和が n と一致しない");
    }
    // 偶数長: k-1 番目と k 番目の平均
    let mut acc = 0usize;
    let mut lo: Option<f64> = None;
    for (v, c) in hist.iter().enumerate() {
        if *c == 0 {
            continue;
        }
        acc += *c as usize;
        if lo.is_none() && acc > k - 1 {
            lo = Some(v as f64);
        }
        if acc > k {
            let l = lo.expect("lo は先に決まっている");
            return (l + v as f64) / 2.0;
        }
    }
    unreachable!("ヒストグラムの総和が n と一致しない");
}

/// i16 列 (折り返し補正後の H) の median。値域は -180..=179。
fn median_i16(vals: &mut Vec<i16>) -> f64 {
    let n = vals.len();
    debug_assert!(n > 0);
    let k = n / 2;
    if n % 2 == 1 {
        let (_, m, _) = vals.select_nth_unstable(k);
        *m as f64
    } else {
        let (lo_part, m, _) = vals.select_nth_unstable(k);
        let hi = *m as f64;
        // k-1 番目は左側パーティションの最大値
        let lo = *lo_part.iter().max().expect("k>=1 なので非空") as f64;
        (lo + hi) / 2.0
    }
}

/// 1 パッチ分の H/S/V ヒストグラムと画素数。
struct PatchStats {
    h_hist: [u32; 256],
    s_hist: [u32; 256],
    v_hist: [u32; 256],
    /// specular 除外後の S ヒストグラム (有効画素のみ)
    s_hist_valid: [u32; 256],
    n_specular: usize,
    /// H の低域/高域カウント (赤2峰判定用)
    n_low: usize,
    n_high: usize,
    n: usize,
}

/// HSV パッチ (ROI 内の矩形) を 1 パスで走査して統計を作る。
fn collect_stats(
    hsv: &ArrayView3<u8>,
    y1: usize,
    y2: usize,
    x1: usize,
    x2: usize,
    p: &HsvParams,
) -> PatchStats {
    let mut st = PatchStats {
        h_hist: [0; 256],
        s_hist: [0; 256],
        v_hist: [0; 256],
        s_hist_valid: [0; 256],
        n_specular: 0,
        n_low: 0,
        n_high: 0,
        n: 0,
    };
    for y in y1..y2 {
        for x in x1..x2 {
            let h = hsv[[y, x, 0]];
            let s = hsv[[y, x, 1]];
            let v = hsv[[y, x, 2]];
            st.h_hist[h as usize] += 1;
            st.s_hist[s as usize] += 1;
            st.v_hist[v as usize] += 1;
            if h as i32 <= p.red_hue_low_max {
                st.n_low += 1;
            }
            if h as i32 >= p.red_hue_wrap_threshold {
                st.n_high += 1;
            }
            // specular = 明るく (V高) かつ白っぽい (S低)
            if v as i32 >= p.specular_v_min && s as i32 <= p.specular_s_max {
                st.n_specular += 1;
            } else {
                st.s_hist_valid[s as usize] += 1;
            }
            st.n += 1;
        }
    }
    st
}

/// H の安定 median (`_compute_stable_h_median` と同値)。
fn stable_h_median(
    st: &PatchStats,
    hsv: &ArrayView3<u8>,
    y1: usize,
    y2: usize,
    x1: usize,
    x2: usize,
    p: &HsvParams,
) -> i32 {
    let plain = median_u8(&st.h_hist, st.n) as i32; // int() 切り捨ては as i32 と同じ (非負)
    if !p.enable_red_hue_wrap_fix {
        return plain;
    }
    // n_total = max(1, size) と同じ扱い (size==0 は呼び出し側で除外済み)
    let n_total = st.n.max(1) as f64;
    let low_ratio = st.n_low as f64 / n_total;
    if low_ratio < p.red_bimodal_min_ratio {
        return plain;
    }
    let high_ratio = st.n_high as f64 / n_total;
    if high_ratio < p.red_bimodal_min_ratio {
        return plain;
    }
    // 赤2峰確定: H>=threshold の画素を -180 して median を取る
    let mut wrapped: Vec<i16> = Vec::with_capacity(st.n);
    for y in y1..y2 {
        for x in x1..x2 {
            let h = hsv[[y, x, 0]] as i16;
            wrapped.push(if h as i32 >= p.red_hue_wrap_threshold {
                h - 180
            } else {
                h
            });
        }
    }
    let med_wrapped = median_i16(&mut wrapped);
    if med_wrapped <= p.red_hue_wrap_corrected_max as f64 {
        // int(max(0, med_wrapped)) — f64 の max のあと 0 方向切り捨て
        let m = if med_wrapped < 0.0 { 0.0 } else { med_wrapped };
        return m as i32;
    }
    plain
}

/// S の光沢ロバスト median (`_compute_specular_robust_s` と同値)。
fn specular_robust_s(st: &PatchStats, p: &HsvParams) -> i32 {
    let plain = median_u8(&st.s_hist, st.n) as i32;
    if !p.enable_specular_robust_saturation {
        return plain;
    }
    let n_total = st.n.max(1);
    let n_valid = st.n - st.n_specular;
    // int(n_total * ratio) の切り捨てと同式
    let min_valid = (n_total as f64 * p.specular_fallback_min_ratio) as usize;
    if n_valid < min_valid {
        return plain;
    }
    if st.n_specular == 0 || st.n_specular == st.n {
        return plain;
    }
    median_u8(&st.s_hist_valid, n_valid) as i32
}

/// BGR チャンネルの median (赤/黄の R-G 差判定用)。
fn bgr_channel_median(
    bgr: &ArrayView3<u8>,
    ch: usize,
    y1: usize,
    y2: usize,
    x1: usize,
    x2: usize,
) -> i32 {
    let mut hist = [0u32; 256];
    let mut n = 0usize;
    for y in y1..y2 {
        for x in x1..x2 {
            hist[bgr[[y, x, ch]] as usize] += 1;
            n += 1;
        }
    }
    // Python 側は int(np.median(...)) — 偶数長平均のあと 0 方向切り捨て
    median_u8(&hist, n) as i32
}

/// 色レンジ 1 行 = [color_code, h_min, h_max, s_min, s_max, v_min, v_max]
type RangeRow = [i32; 7];

/// 閾値照合の共通部 (レンジ走査)。`with_red_break` で 2 実装の差を切り替える。
///
/// - `with_red_break = true`  : `classify` 相当。赤の拡張範囲で R-G 差不足なら
///   その色の残りレンジを飛ばして次の色へ (break + 外側 continue)。
/// - `with_red_break = false` : `_classify_single_patch_no_subregion` 相当。
///   同じ色の次のレンジを試す (continue)。
#[allow(clippy::too_many_arguments)]
fn match_ranges(
    ranges: &[RangeRow],
    h: i32,
    s: i32,
    v: i32,
    p: &HsvParams,
    with_red_break: bool,
    rg_diff: &mut dyn FnMut() -> i32,
) -> Option<i32> {
    let mut i = 0usize;
    while i < ranges.len() {
        let color_code = ranges[i][0];
        // 同じ color_code の連続区間を 1 グループとして扱う (Python の
        // dict[color] = [ranges...] 構造を平坦化したもの)
        let group_start = i;
        let mut group_end = i;
        while group_end < ranges.len() && ranges[group_end][0] == color_code {
            group_end += 1;
        }
        let mut red_skipped = false;
        for r in &ranges[group_start..group_end] {
            let (h_min, h_max, s_min, s_max, v_min, v_max) =
                (r[1], r[2], r[3], r[4], r[5], r[6]);
            let eff_s_min = if p.s_min_scale < 1.0 {
                (s_min as f64 * p.s_min_scale) as i32
            } else {
                s_min
            };
            if h_min <= h && h <= h_max && eff_s_min <= s && s <= s_max && v_min <= v && v <= v_max
            {
                if color_code == COLOR_RED && p.red_extend_h_min <= h && h <= p.red_extend_h_max {
                    if rg_diff() >= p.red_green_diff_for_red {
                        return Some(COLOR_RED);
                    }
                    if with_red_break {
                        red_skipped = true;
                        break; // その色の残りレンジを飛ばす
                    }
                    continue; // 同じ色の次のレンジを試す
                }
                return Some(color_code);
            }
        }
        let _ = red_skipped; // break 後は外側 continue と同義 (次の色へ)
        i = group_end;
    }
    None
}

/// サブ領域 vote を行わない純 median 分類
/// (`_classify_single_patch_no_subregion` と同値)。
#[allow(clippy::too_many_arguments)]
fn classify_no_subregion(
    bgr: &ArrayView3<u8>,
    hsv: &ArrayView3<u8>,
    y1: usize,
    y2: usize,
    x1: usize,
    x2: usize,
    ranges: &[RangeRow],
    p: &HsvParams,
) -> i32 {
    if y2 <= y1 || x2 <= x1 {
        return COLOR_EMPTY;
    }
    let st = collect_stats(hsv, y1, y2, x1, x2, p);
    let h = stable_h_median(&st, hsv, y1, y2, x1, x2, p);
    let s = specular_robust_s(&st, p);
    let v = median_u8(&st.v_hist, st.n) as i32;
    if v < p.empty_v_threshold {
        return COLOR_EMPTY;
    }
    let mut rg = || {
        let g = bgr_channel_median(bgr, 1, y1, y2, x1, x2);
        let r = bgr_channel_median(bgr, 2, y1, y2, x1, x2);
        r - g
    };
    if let Some(c) = match_ranges(ranges, h, s, v, p, false, &mut rg) {
        return c;
    }
    if s < p.ojama_s_threshold && v >= p.ojama_v_min {
        return COLOR_OJAMA;
    }
    COLOR_EMPTY
}

/// 1 セルを分類する (`ColorClassifier.classify` と同値、vote_mode=False 経路)。
#[allow(clippy::too_many_arguments)]
fn classify_cell(
    bgr: &ArrayView3<u8>,
    hsv: &ArrayView3<u8>,
    y1: usize,
    y2: usize,
    x1: usize,
    x2: usize,
    ranges: &[RangeRow],
    p: &HsvParams,
) -> i32 {
    if y2 <= y1 || x2 <= x1 {
        return COLOR_EMPTY;
    }
    let st = collect_stats(hsv, y1, y2, x1, x2, p);
    let h = stable_h_median(&st, hsv, y1, y2, x1, x2, p);
    let s = specular_robust_s(&st, p);
    let v = median_u8(&st.v_hist, st.n) as i32;
    if v < p.empty_v_threshold {
        return COLOR_EMPTY;
    }
    let mut rg = || {
        let g = bgr_channel_median(bgr, 1, y1, y2, x1, x2);
        let r = bgr_channel_median(bgr, 2, y1, y2, x1, x2);
        r - g
    };
    if let Some(c) = match_ranges(ranges, h, s, v, p, true, &mut rg) {
        return c;
    }
    if s < p.ojama_s_threshold && v >= p.ojama_v_min {
        return COLOR_OJAMA;
    }
    // サブ領域 vote (cycle 69-B): 中央 median が EMPTY のとき 4 分割で救済
    let ph = y2 - y1;
    let pw = x2 - x1;
    if ph >= p.subregion_min_h && pw >= p.subregion_min_w {
        let h2 = ph / 2;
        let w2 = pw / 2;
        // Python のスライス順序と同一: [:h2,:w2], [:h2,w2:], [h2:,:w2], [h2:,w2:]
        let subs = [
            (y1, y1 + h2, x1, x1 + w2),
            (y1, y1 + h2, x1 + w2, x2),
            (y1 + h2, y2, x1, x1 + w2),
            (y1 + h2, y2, x1 + w2, x2),
        ];
        // Counter.most_common(1) の再現: 出現順を保った候補列に対し、
        // 厳密な `>` 比較で最大を取る (同数なら先に出た色が勝つ)
        let mut order: Vec<i32> = Vec::with_capacity(4);
        let mut count: Vec<usize> = Vec::with_capacity(4);
        for (sy1, sy2, sx1, sx2) in subs {
            if sy2 <= sy1 || sx2 <= sx1 {
                continue; // sp.size == 0 に相当
            }
            let c = classify_no_subregion(bgr, hsv, sy1, sy2, sx1, sx2, ranges, p);
            if c == COLOR_EMPTY || c == 10 {
                continue; // COLOR_EMPTY / COLOR_UNKNOWN は候補にしない
            }
            match order.iter().position(|&x| x == c) {
                Some(idx) => count[idx] += 1,
                None => {
                    order.push(c);
                    count.push(1);
                }
            }
        }
        if !order.is_empty() {
            let mut best = 0usize;
            for i in 1..order.len() {
                if count[i] > count[best] {
                    best = i;
                }
            }
            return order[best];
        }
    }
    COLOR_EMPTY
}

/// 盤面 ROI 内の複数セルを一括分類する。
///
/// Python/Rust 境界を 1 回だけ越えることが高速化の本体。
/// `bgr_roi` / `hsv_roi` は同一形状 (H, W, 3)、`cell_rects` は (N, 4) で
/// ROI ローカル座標 [x1, y1, x2, y2]。
#[pyfunction]
pub fn classify_cells_hsv(
    py: Python<'_>,
    bgr_roi: PyReadonlyArray3<u8>,
    hsv_roi: PyReadonlyArray3<u8>,
    cell_rects: PyReadonlyArray2<i32>,
    ranges_flat: PyReadonlyArray2<i32>,
    params: HsvParams,
) -> PyResult<Py<PyArray1<i32>>> {
    let bgr = bgr_roi.as_array();
    let hsv = hsv_roi.as_array();
    if bgr.shape() != hsv.shape() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "bgr_roi と hsv_roi の形状が一致しない",
        ));
    }
    let (rh, rw) = (bgr.shape()[0], bgr.shape()[1]);
    let rects = cell_rects.as_array();
    let rng = ranges_flat.as_array();
    let ranges: Vec<RangeRow> = (0..rng.shape()[0])
        .map(|i| {
            [
                rng[[i, 0]],
                rng[[i, 1]],
                rng[[i, 2]],
                rng[[i, 3]],
                rng[[i, 4]],
                rng[[i, 5]],
                rng[[i, 6]],
            ]
        })
        .collect();

    let n_cells = rects.shape()[0];
    let mut out: Vec<i32> = Vec::with_capacity(n_cells);
    for i in 0..n_cells {
        // ROI 外にはみ出す矩形は Python 側のスライスと同じ挙動でクリップする
        let x1 = rects[[i, 0]].max(0) as usize;
        let y1 = rects[[i, 1]].max(0) as usize;
        let x2 = (rects[[i, 2]].max(0) as usize).min(rw);
        let y2 = (rects[[i, 3]].max(0) as usize).min(rh);
        if y2 <= y1 || x2 <= x1 {
            out.push(COLOR_EMPTY); // bgr_patch.size == 0 → COLOR_EMPTY
            continue;
        }
        out.push(classify_cell(&bgr, &hsv, y1, y2, x1, x2, &ranges, &params));
    }
    Ok(out.into_pyarray_bound(py).unbind())
}
