//! per-pixel 投票方式の HSV 分類のネイティブ実装 (2026-10-25)。
//!
//! `src/image_reader.py` の `ColorClassifier._classify_by_vote` (`vote_mode=True`) を移植する。
//! 配信の LiveImages (発火前画像の盤面読み) が盤面 1 枚あたり 72 セルをこの方式で numpy のマスク演算
//! (セルごとに 20 回前後の小さな配列演算) で分類しており、連鎖の終わり付近のコマで 20ms 超の山を作っていた。
//!
//! ## 一致のための約束
//!
//! 画素の比較・票の数え上げ・しきい値はすべて整数 (比率の積だけ float64 の掛け算と切り捨て) なので、
//! Python 版とビット一致する。cvtColor は移植しない (HSV は Python 側 cv2 が作った画像を受け取る)。
//! 色レンジは Python の dict 挿入順で渡し、票が最多の色を選ぶ比較は `>` (同数なら先の色) のまま。

use numpy::ndarray::ArrayView3;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray2, PyReadonlyArray3};
use pyo3::prelude::*;

use crate::hsv_classify::{RangeRow, COLOR_EMPTY};

/// 色コード (src/board.py と一致させる)。
const COLOR_RED: i32 = 1;
const COLOR_OJAMA: i32 = 9;
/// 色レンジの 1 行を持つ数: h_min, h_max, s_min(補正後), s_max, v_min, v_max。
const RANGE_FIELDS: usize = 6;

/// 判定パラメータ。Python 側の定数・設定を単一情報源にするため、値は呼出側から受け取る。
#[derive(FromPyObject)]
#[pyo3(from_item_all)]
pub struct VoteParams {
    pub s_min_scale: f64,
    pub ojama_s_threshold: i32,
    pub ojama_v_min: i32,
    pub red_extend_h_min: i32,
    pub red_extend_h_max: i32,
    pub vote_red_rg_diff: i32,
    pub vote_puyo_min_ratio: f64,
    pub vote_ojama_min_ratio: f64,
}

/// 同じ色コードのレンジの束 (補正後の S 下限つき)。
struct ColorGroup {
    code: i32,
    ranges: Vec<[i32; RANGE_FIELDS]>,
}

/// レンジ行を、色コードごとの束へ並べ直す (dict の挿入順)。S 下限は Python と同じ式で補正する。
fn group_ranges(rows: &[RangeRow], p: &VoteParams) -> Vec<ColorGroup> {
    let mut groups: Vec<ColorGroup> = Vec::new();
    for row in rows {
        let s_min = if p.s_min_scale < 1.0 {
            (f64::from(row[3]) * p.s_min_scale) as i32
        } else {
            row[3]
        };
        let entry = [row[1], row[2], s_min, row[4], row[5], row[6]];
        match groups.last_mut() {
            Some(last) if last.code == row[0] => last.ranges.push(entry),
            _ => groups.push(ColorGroup { code: row[0], ranges: vec![entry] }),
        }
    }
    groups
}

fn in_range(r: &[i32; RANGE_FIELDS], h: i32, s: i32, v: i32) -> bool {
    r[0] <= h && h <= r[1] && r[2] <= s && s <= r[3] && r[4] <= v && v <= r[5]
}

/// 1 セルを投票で分類する。`_classify_by_vote` と同じ判定順 (puyo > おじゃま > 空)。
#[allow(clippy::too_many_arguments)]
fn vote_cell(
    bgr: &ArrayView3<u8>,
    hsv: &ArrayView3<u8>,
    y1: usize,
    y2: usize,
    x1: usize,
    x2: usize,
    groups: &[ColorGroup],
    p: &VoteParams,
) -> i32 {
    let total = (y2 - y1) * (x2 - x1);
    let mut votes = vec![0usize; groups.len()];
    let mut ojama_votes = 0usize;
    for y in y1..y2 {
        for x in x1..x2 {
            let (h, s, v) = (i32::from(hsv[[y, x, 0]]), i32::from(hsv[[y, x, 1]]), i32::from(hsv[[y, x, 2]]));
            for (g, group) in groups.iter().enumerate() {
                if !group.ranges.iter().any(|r| in_range(r, h, s, v)) {
                    continue;
                }
                // 赤の拡張範囲 (黄との境界) は BGR の R-G 差が足りない画素を票から落とす
                if group.code == COLOR_RED && p.red_extend_h_min <= h && h <= p.red_extend_h_max {
                    let rg = i32::from(bgr[[y, x, 2]]) - i32::from(bgr[[y, x, 1]]);
                    if rg < p.vote_red_rg_diff {
                        continue;
                    }
                }
                votes[g] += 1;
            }
            ojama_votes += usize::from(s < p.ojama_s_threshold && v >= p.ojama_v_min);
        }
    }
    let (mut top_color, mut top_votes) = (COLOR_EMPTY, 0usize);
    for (g, group) in groups.iter().enumerate() {
        if votes[g] > top_votes {
            top_votes = votes[g];
            top_color = group.code;
        }
    }
    let puyo_min = ((total as f64 * p.vote_puyo_min_ratio) as usize).max(1);
    let ojama_min = ((total as f64 * p.vote_ojama_min_ratio) as usize).max(1);
    if top_votes >= puyo_min && top_votes > ojama_votes {
        return top_color;
    }
    if ojama_votes >= ojama_min {
        return COLOR_OJAMA;
    }
    COLOR_EMPTY
}

/// 盤面 ROI 内の複数セルを投票方式で一括分類する (`classify_cells_hsv` の投票版)。
#[pyfunction]
pub fn classify_cells_vote(
    py: Python<'_>,
    bgr_roi: PyReadonlyArray3<u8>,
    hsv_roi: PyReadonlyArray3<u8>,
    cell_rects: PyReadonlyArray2<i32>,
    ranges_flat: PyReadonlyArray2<i32>,
    params: VoteParams,
) -> PyResult<Py<PyArray1<i32>>> {
    let bgr = bgr_roi.as_array();
    let hsv = hsv_roi.as_array();
    if bgr.shape() != hsv.shape() {
        return Err(pyo3::exceptions::PyValueError::new_err("bgr_roi と hsv_roi の形状が一致しない"));
    }
    let (rh, rw) = (bgr.shape()[0], bgr.shape()[1]);
    let rects = cell_rects.as_array();
    let range_view = ranges_flat.as_array();
    let rows: Vec<RangeRow> = (0..range_view.shape()[0])
        .map(|i| {
            let mut row = [0i32; 7];
            for (k, slot) in row.iter_mut().enumerate() {
                *slot = range_view[[i, k]];
            }
            row
        })
        .collect();
    let groups = group_ranges(&rows, &params);
    let mut out: Vec<i32> = Vec::with_capacity(rects.shape()[0]);
    for i in 0..rects.shape()[0] {
        // ROI 外にはみ出す矩形は Python 側のスライスと同じ挙動でクリップする
        let x1 = rects[[i, 0]].max(0) as usize;
        let y1 = rects[[i, 1]].max(0) as usize;
        let x2 = (rects[[i, 2]].max(0) as usize).min(rw);
        let y2 = (rects[[i, 3]].max(0) as usize).min(rh);
        if y2 <= y1 || x2 <= x1 {
            out.push(COLOR_EMPTY); // size == 0 は classify が先に COLOR_EMPTY を返す
            continue;
        }
        out.push(vote_cell(&bgr, &hsv, y1, y2, x1, x2, &groups, &params));
    }
    Ok(out.into_pyarray_bound(py).unbind())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params() -> VoteParams {
        VoteParams {
            s_min_scale: 1.0,
            ojama_s_threshold: 20,
            ojama_v_min: 100,
            red_extend_h_min: 11,
            red_extend_h_max: 18,
            vote_red_rg_diff: 80,
            vote_puyo_min_ratio: 0.10,
            vote_ojama_min_ratio: 0.40,
        }
    }

    #[test]
    fn groups_keep_dict_order_and_scale_s_min() {
        let rows: Vec<RangeRow> = vec![[1, 0, 10, 100, 255, 100, 255], [1, 170, 180, 100, 255, 100, 255], [2, 100, 130, 25, 255, 60, 255]];
        let mut p = params();
        p.s_min_scale = 0.7;
        let groups = group_ranges(&rows, &p);
        assert_eq!(groups.len(), 2);
        assert_eq!((groups[0].code, groups[0].ranges.len()), (1, 2));
        assert_eq!(groups[0].ranges[0][2], 70); // int(100 * 0.7) の切り捨て
        assert_eq!(groups[1].ranges[0][2], 17); // int(25 * 0.7) = int(17.5) の切り捨て
    }
}
