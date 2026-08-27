"""物差しv2 (55盤面) 正解ラベル監査 (2026-08-17、デバッガ計装スクリプト)。

## 目的
c13 r9c3 (sheet 026_c13_2P_f17462) で確定した「正解ラベル自体の誤り」と
同型のラベル誤りを、55盤面全体で機械的にスクリーニングする。

## 方法 (2段)
1. 構成F (直近の統一測定構成、data/indicators_v2/yardstick_v2_boards_f_2026-08-17)
   の単一アンカーフレーム予測と、labels_final_from_user.json の正解を全セル突合し、
   不一致セルを列挙する (= 既知の誤認リストの再現チェックにもなる)。
2. 独立検査 A: 同一 (video_id, side) で近接時刻の姉妹シート同士の正解ラベルを比較。
   極短時間差 (<0.5秒、数フレーム) で実ゲームが変化しえない間隔にもかかわらず
   ラベル値が食い違うセルは、ラベル誤りの直接証拠 (r9c3型)。
   独立検査 B: 各不一致セルについて、保存済みアンカーROI PNGから該当セルの
   HSV実測 (H/S/V中央値) を取り、簡易色分類器で「アンカーフレーム時点のピクセルは
   どちらに近いか (ラベル値 or 予測値)」を出す (静止1フレームの参考シグナル、
   持続性の主張はしない)。

本スクリプトは src/ を変更しない。読み取りのみ。
"""
from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
YARDSTICK_DIR = _ROOT / "data" / "verify" / "yardstick_v2_2026-08-14"
MANIFEST_PATH = YARDSTICK_DIR / "manifest.json"
LABELS_PATH = YARDSTICK_DIR / "labels_final_from_user.json"
NPZ_DIR_F = _ROOT / "data" / "indicators_v2" / "yardstick_v2_boards_f_2026-08-17"
OUT_DIR = _ROOT / "data" / "verify" / "yardstick_v2_label_audit_2026-08-17"
PNG_DIR = OUT_DIR / "suspect_frames"

_score_mod = importlib.import_module("scripts._score_yardstick_v2_2026-08-14")

UNKNOWN_VALUE = 10
COLOR_NAME = {0: "空", 1: "赤", 2: "青", 3: "緑", 4: "黄", 5: "紫", 9: "おじゃま", 10: "不明"}
# 近接時刻の姉妹シート判定閾値 [秒]。この時間内で色↔おじゃまが自然反転することは
# 物理的にありえない (おじゃま着弾はconnected chainの完了後にのみ起きる)。
SISTER_DT_SEC_THRESHOLD: float = 0.5

# 参考HSV分類 (診断専用の粗い色分類器、本体src/のロジックとは独立。
# 独立シグナルとして使うことが目的なので本番の分類器は使わない)。
_HSV_REF_CENTERS: dict[int, tuple[float, float, float]] = {
    1: (0, 180, 200),     # 赤
    2: (105, 180, 220),   # 青
    3: (55, 150, 200),    # 緑
    4: (28, 180, 230),    # 黄
    5: (140, 140, 190),   # 紫
    9: (0, 25, 235),      # おじゃま (低彩度・高輝度)
    0: (0, 0, 60),        # 空 (低輝度)
}


def load_manifest() -> list[dict[str, Any]]:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def load_labels() -> dict[str, dict[str, Any]]:
    labels = json.loads(LABELS_PATH.read_text(encoding="utf-8"))
    return {s["sheet_id"]: s for s in labels["sheets"]}


def corrected_grid(entry: dict[str, Any], label_row: dict[str, Any]) -> np.ndarray:
    grid = [row[:] for row in entry["init_grid"]]
    for wc in label_row["wrong_cells"]:
        m = _score_mod.WRONG_CELL_RE.match(wc)
        r, c, v = int(m[1]), int(m[2]), int(m[3])
        grid[r][c] = v
    return np.array(grid, dtype=np.int64)


def cell_hsv_median(roi_png_path: Path, r: int, c: int) -> tuple[float, float, float] | None:
    """board_roi_png (6列x12行=可視段のみの盤面ROI、DEFAULT_P1/P2_REGION=384x720=
    64x60px/セル) からセル中央部のHSV中央値を取る。

    【重要】board_roi_png は隠し段 (行0) を含まない可視12行のみ (image_reader.py の
    DEFAULT_P1_REGION.height=720 / 12行 = 60px/行 で確認)。ラベル配列側の行インデックス
    r (1〜12、1=画面最上段の可視行) を画像行 (r-1) にオフセットしないと1行分ズレる
    (診断スクリプト自身のバグとして2026-08-17に発見・本行で修正)。r=0 (隠し段) は
    画像に写らないため呼び出し禁止 (assert)。
    """
    assert r >= 1, "行0(隠し段)はROI画像に写らないため呼べない"
    img = cv2.imread(str(roi_png_path))
    if img is None:
        return None
    h, w = img.shape[:2]
    cell_h, cell_w = h / 12.0, w / 6.0
    y0, y1 = int((r - 1) * cell_h), int(r * cell_h)
    x0, x1 = int(c * cell_w), int((c + 1) * cell_w)
    # 内側60%のみ使う (セル境界の混色を避ける)
    pad_y, pad_x = int((y1 - y0) * 0.2), int((x1 - x0) * 0.2)
    crop = img[y0 + pad_y:y1 - pad_y, x0 + pad_x:x1 - pad_x]
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    return (float(np.median(hsv[:, :, 0])), float(np.median(hsv[:, :, 1])), float(np.median(hsv[:, :, 2])))


def classify_hsv(hsv: tuple[float, float, float]) -> int:
    """参考HSV分類器 (診断専用、粗い最近傍)。低彩度優先でおじゃま/空を先に判定。"""
    h, s, v = hsv
    if s < 40:
        return 9 if v > 120 else 0
    best_val, best_dist = 0, 1e18
    for val, (ch, cs, cv_) in _HSV_REF_CENTERS.items():
        if val in (0, 9):
            continue
        dh = min(abs(h - ch), 180 - abs(h - ch)) * 2.0  # 色相は円環、重み付け
        dist = dh ** 2 + (s - cs) ** 2 * 0.3 + (v - cv_) ** 2 * 0.1
        if dist < best_dist:
            best_val, best_dist = val, dist
    return best_val


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PNG_DIR.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest()
    labels = load_labels()
    by_id = {e["sheet_id"]: e for e in manifest}

    valid_sheets = []
    for e in manifest:
        lab = labels[e["sheet_id"]]
        if lab["status"] == "not_a_board":
            continue
        valid_sheets.append(e)
    print(f"[info] 有効盤面数: {len(valid_sheets)}")

    # --- 検査1: 構成Fとの単一フレーム突合 ---
    index_f = _score_mod.load_npz_index(NPZ_DIR_F)
    mismatches: list[dict[str, Any]] = []
    n_miss = 0
    for e in valid_sheets:
        lab = labels[e["sheet_id"]]
        grid = corrected_grid(e, lab)
        gt = {"video_id": e["video_id"], "side": e["side"], "frame_idx": e["frame_idx"], "t_sec": e["t_sec"]}
        rec, method = _score_mod.match_record(gt, index_f)
        if rec is None:
            n_miss += 1
            continue
        pred = np.array(rec["grid"])
        for r in range(1, 13):
            for c in range(6):
                corr = int(grid[r, c])
                if corr == UNKNOWN_VALUE:
                    continue
                pv = int(pred[r, c])
                if pv != corr:
                    mismatches.append({
                        "sheet_id": e["sheet_id"], "video_id": e["video_id"], "side": e["side"],
                        "r": r, "c": c, "label": corr, "pred_f": pv,
                        "match_method": method,
                    })
    print(f"[検査1] npz-F突合miss={n_miss}/{len(valid_sheets)}、不一致セル数={len(mismatches)}")

    # --- 検査2: 姉妹シート (近接時刻) の正解ラベル内部矛盾 ---
    from collections import defaultdict
    by_vs: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for e in valid_sheets:
        by_vs[(e["video_id"], e["side"])].append(e)

    sister_findings: list[dict[str, Any]] = []
    for key, entries in by_vs.items():
        entries = sorted(entries, key=lambda e: e["t_sec"])
        for i in range(len(entries) - 1):
            e_a, e_b = entries[i], entries[i + 1]
            dt = e_b["t_sec"] - e_a["t_sec"]
            if dt > SISTER_DT_SEC_THRESHOLD:
                continue
            ga = corrected_grid(e_a, labels[e_a["sheet_id"]])
            gb = corrected_grid(e_b, labels[e_b["sheet_id"]])
            for r in range(1, 13):
                for c in range(6):
                    va, vb = int(ga[r, c]), int(gb[r, c])
                    if va == UNKNOWN_VALUE or vb == UNKNOWN_VALUE:
                        continue
                    if va != vb:
                        sister_findings.append({
                            "sheet_a": e_a["sheet_id"], "sheet_b": e_b["sheet_id"],
                            "dt_sec": round(dt, 4), "r": r, "c": c,
                            "label_a": va, "label_b": vb,
                            "implausible": (va == 9) != (vb == 9),  # 色<->おじゃま反転は物理的にありえない
                        })
    print(f"[検査2] 極近接姉妹シートペア内でのラベル不一致: {len(sister_findings)}件")
    for f in sister_findings:
        flag = " ★物理的にありえない(色<->おじゃま反転)" if f["implausible"] else ""
        print(f"   {f['sheet_a']} vs {f['sheet_b']} (dt={f['dt_sec']}s) r{f['r']}c{f['c']}: "
              f"{f['label_a']}({COLOR_NAME.get(f['label_a'],'?')}) vs {f['label_b']}({COLOR_NAME.get(f['label_b'],'?')}){flag}")

    # --- 検査3: 不一致セルに対する単フレームHSV参考シグナル ---
    hsv_checks: list[dict[str, Any]] = []
    for mm in mismatches:
        e = by_id[mm["sheet_id"]]
        roi_path = _ROOT / e["board_roi_png"]
        hsv = cell_hsv_median(roi_path, mm["r"], mm["c"])
        if hsv is None:
            continue
        hsv_pred = classify_hsv(hsv)
        hsv_checks.append({**mm, "hsv_h": round(hsv[0], 1), "hsv_s": round(hsv[1], 1), "hsv_v": round(hsv[2], 1),
                            "hsv_ref_classify": hsv_pred,
                            "hsv_agrees_with_pred_f": hsv_pred == mm["pred_f"],
                            "hsv_agrees_with_label": hsv_pred == mm["label"]})

    # HSV参考分類器が「予測(pred_f)に同意・ラベルに非同意」= ラベル疑いの弱シグナル
    label_suspects_by_hsv = [c for c in hsv_checks if c["hsv_agrees_with_pred_f"] and not c["hsv_agrees_with_label"]]
    print(f"[検査3] 単フレームHSV参考分類がpred_fに同意・ラベルに非同意 (弱シグナル): {len(label_suspects_by_hsv)}/{len(hsv_checks)}")

    out = {
        "n_valid_sheets": len(valid_sheets),
        "npz_f_miss": n_miss,
        "mismatches_vs_label": mismatches,
        "sister_pair_findings": sister_findings,
        "hsv_single_frame_checks": hsv_checks,
        "hsv_label_suspects_weak_signal": label_suspects_by_hsv,
    }
    (OUT_DIR / "audit_raw.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] -> {OUT_DIR / 'audit_raw.json'}")


if __name__ == "__main__":
    main()
