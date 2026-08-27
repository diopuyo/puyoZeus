# -*- coding: utf-8 -*-
"""勝敗ラベル端点修正 (game0 起点補正 + 最終試合遡り延長) の実データ検証 (2026-08-19)。

subset50 のうち手元に動画が残る 11 本 (29,31-35,37-39,c109,c132) について:
- 旧アルゴリズム (遡り30秒上限・game0 起点=動画冒頭+1秒) を忠実に再現
- 新アルゴリズム (src/match_winner.py 修正後の detect_all_winners)
を同一の match_starts 近似値 (npz の game 別最初の snapshot 時刻) で走らせ、
2 系統一致後の最終ラベル (score 系統は npz からオフライン再現) を新旧比較する。

さらに目視突合用に、代表動画の game0 / 最終試合の読取フレームを PNG 保存する。

出力:
- logs/_validate_winner_endpoint_fix_2026-08-19.tsv (全 game 詳細)
- data/verify/winner_endpoint_fix_2026-08-19/*.png (目視用フレーム)
- 標準出力サマリ (回復数)
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
OUT_TSV = ROOT / "logs" / "_validate_winner_endpoint_fix_2026-08-19.tsv"
EVIDENCE_DIR = ROOT / "data" / "verify" / "winner_endpoint_fix_2026-08-19"

# 手元に動画が残っている target_id → 動画ファイル名
TARGETS: dict[str, str] = {
    "29": "video_29.mp4", "31": "video_31.mp4", "32": "video_32.mp4",
    "33": "video_33.mp4", "34": "video_34.mp4", "35": "video_35.mp4",
    "37": "video_37.mp4", "38": "video_38.mp4", "39": "video_39.mp4",
    "c109": "video_c109.mp4", "c132": "video_c132.mp4",
}
EVIDENCE_TARGETS = ("29", "c109", "c132")

_DEATH_ROW, _DEATH_COL = 1, 2
OFFSET_BEFORE = 1.0


def _score_winner_offline(d: dict, rows: np.ndarray) -> str | None:
    """npz 末尾 snapshot の score から score 系統勝者を再現 (同点/欠損は None)。"""
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


def _survival_winner_offline(d: dict, rows: np.ndarray) -> str | None:
    """_winner_by_survival 相当 (窒息セル row=1,col=2)。"""
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


def _stored_label(d: dict, rows: np.ndarray) -> str | None:
    """実収集 npz に保存済みの won ラベル (旧コード実走の結果)。"""
    w1 = d["won"][rows[d["side"][rows] == "1P"]]
    if len(w1) and not np.isnan(w1[-1]):
        return "1P" if float(w1[-1]) == 1.0 else "2P"
    return None


def _old_detect_all(
    det: MatchWinnerDetector, cap: cv2.VideoCapture,
    match_starts: list[float], last_observable_sec: float,
) -> list:
    """修正前アルゴリズムの忠実再現 (遡り30秒・細step固定・起点補正なし)。"""
    panel_visible = det._find_panel_visible_time(
        cap, last_observable_sec,
        scan_back_max=30.0, step=0.3, coarse_after_sec=30.0,
    )
    last_t = panel_visible if panel_visible is not None else last_observable_sec
    boundaries = list(match_starts) + [last_t]
    results = []
    for i in range(len(match_starts)):
        results.append(det.detect_winner(
            cap, boundaries[i], boundaries[i + 1],
            offset_before=OFFSET_BEFORE,
            offset_after=OFFSET_BEFORE if i < len(match_starts) - 1 else 0.0,
        ))
    return results


def _save_evidence(
    tid: str, cap: cv2.VideoCapture, det: MatchWinnerDetector,
    tag: str, t: float | None,
) -> None:
    if t is None:
        return
    frame = det._read_frame(cap, t)
    if frame is None:
        return
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(EVIDENCE_DIR / f"{tid}_{tag}_t{t:.1f}.png"), frame)
    # 数値パネル周辺の拡大クロップ (y=940-1030, x=780-1140)
    crop = frame[940:1030, 780:1140]
    big = cv2.resize(crop, (crop.shape[1] * 3, crop.shape[0] * 3),
                     interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(str(EVIDENCE_DIR / f"{tid}_{tag}_t{t:.1f}_panelcrop.png"), big)


def process_video(tid: str) -> list[str]:
    """1 動画分の新旧比較。TSV 行のリストを返す。"""
    npz_path = NEW_DIR / f"{tid}.npz"
    video_path = FRAMES_DIR / TARGETS[tid]
    d = dict(np.load(npz_path, allow_pickle=False))
    if len(d["game_idx"]) == 0:
        return []
    gidxs = sorted(set(int(g) for g in d["game_idx"]))
    rows_by_g = {g: np.where(d["game_idx"] == g)[0] for g in gidxs}
    # match_starts 近似: game0=0.0 (実走の start_sec)、以降=各 game の最初の
    # snapshot 時刻 (試合中はパネル数値が不変のため読取基準として等価)
    match_starts = [0.0] + [float(d["t_sec"][rows_by_g[g][0]]) for g in gidxs[1:]]
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return [f"{tid}\tOPEN_FAIL"]
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    last_observable = max(n_frames / fps - 0.5, float(d["t_sec"][-1]))
    det = MatchWinnerDetector.load_default()
    old_results = _old_detect_all(det, cap, match_starts, last_observable)
    new_results = det.detect_all_winners(cap, match_starts, last_observable)
    # 目視用フレーム: 代表動画の game0 / 最終試合の読取時刻
    if tid in EVIDENCE_TARGETS:
        reads = det._resolve_read_times(cap, match_starts, last_observable, OFFSET_BEFORE)
        _save_evidence(tid, cap, det, "game0_start", reads[0])
        _save_evidence(tid, cap, det, "game0_end", reads[1] if len(reads) > 1 else None)
        _save_evidence(tid, cap, det, "lastgame_start", reads[-2])
        _save_evidence(tid, cap, det, "lastgame_end", reads[-1])
    cap.release()
    lines = []
    for pos, g in enumerate(gidxs):
        rows = rows_by_g[g]
        sw = _score_winner_offline(d, rows)
        vw = _survival_winner_offline(d, rows)
        score_sys = sw if sw is not None else vw  # assign の score 系統 (窒息フォールバック込み)
        stored = _stored_label(d, rows)
        old_panel = old_results[pos].winner
        old_label = score_sys if (score_sys is not None and score_sys == old_panel) else None
        nr = new_results[pos]
        if nr.panel_unavailable:
            new_label = vw  # 窒息判定のみ (score 単独緩和はしない)
            new_panel = "UNAVAILABLE"
        else:
            new_panel = nr.winner
            new_label = score_sys if (score_sys is not None and score_sys == nr.winner) else None
        kind = "game0" if pos == 0 else ("last" if pos == len(gidxs) - 1 else "mid")
        lines.append(
            f"{tid}\t{g}\t{kind}\t{match_starts[pos]:.1f}\t{stored}\t"
            f"{sw}\t{vw}\t{old_panel}\t{old_label}\t{new_panel}\t{new_label}"
        )
    return lines


def main() -> None:
    header = ("video\tgidx\tkind\tstart_approx\tstored_label\tscore_w\tsurvival_w\t"
              "old_panel\told_label\tnew_panel\tnew_label")
    all_lines = [header]
    with ProcessPoolExecutor(max_workers=6) as ex:
        for lines in ex.map(process_video, sorted(TARGETS)):
            all_lines.extend(lines)
    OUT_TSV.parent.mkdir(exist_ok=True)
    OUT_TSV.write_text("\n".join(all_lines) + "\n", encoding="utf-8")
    # サマリ集計
    rec = {"game0": [0, 0, 0], "last": [0, 0, 0], "mid": [0, 0, 0]}  # [旧欠損, 回復, 新旧食い違い]
    for ln in all_lines[1:]:
        f = ln.split("\t")
        if len(f) < 11:
            continue
        kind, old_label, new_label = f[2], f[8], f[10]
        if old_label == "None":
            rec[kind][0] += 1
            if new_label != "None":
                rec[kind][1] += 1
        elif new_label != old_label:
            rec[kind][2] += 1
    for kind, (miss, recov, flip) in rec.items():
        print(f"{kind}: 旧欠損={miss} 回復={recov} 新旧食い違い(旧ラベルあり)={flip}")
    print(f"TSV -> {OUT_TSV}")


if __name__ == "__main__":
    main()
