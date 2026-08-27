# -*- coding: utf-8 -*-
"""端点修正の軽量検証 (最終試合のみ、2026-08-19)。

CPU競合回避のため軽量化: 収集なし・game0 前方探索なし (game0 結果は
1回目の全量検証 TSV で確定済み)・最終試合の終点読取のみを新方式
(遡り上限=試合開始まで + 終点メディアン合成) で再計算する。並列 3。

score系統 (npz 由来) との 2 系統一致で new_label を出し、旧方式 (実収集
npz の won = 旧コード出力) と比較する。
"""
from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.match_winner import MatchWinnerDetector  # noqa: E402

NEW_DIR = ROOT / "data" / "indicators_v2" / "boards_lean_subset50_2026-08-19"
FRAMES_DIR = ROOT / "data" / "frames"
OUT_TSV = ROOT / "logs" / "_validate_winner_lastgame_light_2026-08-19.tsv"
EVIDENCE_DIR = ROOT / "data" / "verify" / "winner_endpoint_fix_2026-08-19"

TARGETS: dict[str, str] = {
    "29": "video_29.mp4", "31": "video_31.mp4", "32": "video_32.mp4",
    "33": "video_33.mp4", "34": "video_34.mp4", "35": "video_35.mp4",
    "37": "video_37.mp4", "38": "video_38.mp4", "39": "video_39.mp4",
    "c109": "video_c109.mp4", "c132": "video_c132.mp4",
}
EVIDENCE_TARGETS = ("31", "c109")
_DEATH_ROW, _DEATH_COL = 1, 2
OFFSET_BEFORE = 1.0


def _score_winner(d: dict, rows: np.ndarray) -> str | None:
    finals: dict[str, int | None] = {"1P": None, "2P": None}
    for side in ("1P", "2P"):
        srows = rows[d["side"][rows] == side]
        if len(srows):
            s = int(d["score"][srows[-1]])
            finals[side] = s if s >= 0 else None
    s1, s2 = finals["1P"], finals["2P"]
    if s1 is not None and s2 is not None and s1 != s2:
        return "1P" if s1 > s2 else "2P"
    return None


def _survival_winner(d: dict, rows: np.ndarray) -> str | None:
    choked: dict[str, bool | None] = {"1P": None, "2P": None}
    for side in ("1P", "2P"):
        srows = rows[d["side"][rows] == side]
        if len(srows):
            choked[side] = bool(d["grids"][srows[-1]][_DEATH_ROW, _DEATH_COL] != 0)
    c1, c2 = choked["1P"], choked["2P"]
    if c1 is None or c2 is None:
        return None
    if c1 and not c2:
        return "2P"
    if c2 and not c1:
        return "1P"
    return None


def process(tid: str) -> str:
    d = dict(np.load(NEW_DIR / f"{tid}.npz", allow_pickle=False))
    gidxs = sorted(set(int(g) for g in d["game_idx"]))
    g_last = gidxs[-1]
    rows = np.where(d["game_idx"] == g_last)[0]
    t_start = float(d["t_sec"][rows[0]])  # 最終試合の開始近似 (試合中は数値不変)
    cap = cv2.VideoCapture(str(FRAMES_DIR / TARGETS[tid]))
    if not cap.isOpened():
        return f"{tid}\tOPEN_FAIL"
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    last_observable = max(cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps - 0.5,
                          float(d["t_sec"][-1]))
    det = MatchWinnerDetector.load_default()
    floor = t_start + OFFSET_BEFORE
    scan_back = max(900.0, last_observable - floor)
    t_end = det._find_panel_visible_time(
        cap, last_observable, scan_back_max=scan_back, not_before_sec=floor,
    )
    if t_end is not None and t_end < floor:
        t_end = None
    if t_end is None:
        panel, unavailable = None, True
    else:
        r = det._detect_last_winner(cap, floor, t_end, floor)
        panel, unavailable = r.winner, False
        if tid in EVIDENCE_TARGETS:
            EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
            for tag, t in (("lastgame_start", floor), ("lastgame_end", t_end)):
                frame = det._read_frame(cap, t)
                if frame is not None:
                    crop = frame[940:1030, 780:1140]
                    big = cv2.resize(crop, (crop.shape[1] * 3, crop.shape[0] * 3),
                                     interpolation=cv2.INTER_NEAREST)
                    cv2.imwrite(str(EVIDENCE_DIR / f"{tid}_{tag}_t{t:.1f}_panelcrop.png"), big)
    cap.release()
    sw = _score_winner(d, rows)
    vw = _survival_winner(d, rows)
    score_sys = sw if sw is not None else vw
    if unavailable:
        new_label = vw
    else:
        new_label = score_sys if (score_sys is not None and score_sys == panel) else None
    w1 = d["won"][rows[d["side"][rows] == "1P"]]
    stored = None
    if len(w1) and not np.isnan(w1[-1]):
        stored = "1P" if float(w1[-1]) == 1.0 else "2P"
    return (f"{tid}\t{g_last}\t{t_start:.1f}\t{t_end}\t{stored}\t{sw}\t{vw}\t"
            f"{'UNAVAILABLE' if unavailable else panel}\t{new_label}")


def main() -> None:
    header = "video\tgidx_last\tstart\tt_end\tstored\tscore_w\tsurvival_w\tnew_panel\tnew_label"
    lines = [header]
    with ProcessPoolExecutor(max_workers=3) as ex:
        lines.extend(ex.map(process, sorted(TARGETS)))
    OUT_TSV.write_text("\n".join(lines) + "\n", encoding="utf-8")
    labeled = sum(1 for ln in lines[1:] if ln.split("\t")[-1] in ("1P", "2P"))
    print(f"最終試合: 旧欠損 11/11 -> 新ラベル付与 {labeled}/11")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
