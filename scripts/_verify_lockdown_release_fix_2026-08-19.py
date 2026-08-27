"""ラッチ解除修正 (数値スコア化) のオフライン検証 + match_end NCC 分布収集。

対象: 手元にある subset50 動画 (再DL分含む)。各動画を実効 ~15fps で走査し、
  1. 旧ラッチ (解除=score_zero_both テンプレのみ) の再現
  2. 新ラッチ (解除に数値スコア両側0 OR + score移動 OR を追加) の再現
  3. match_end NCC 検出エピソードの分布 (閾値0.80 引き上げ判断の材料)
を記録する。npz (boards_lean_subset50_2026-08-19) の snapshot 行と時刻で
突合し、「同一行集合での locked 行比率」の before/after を出す。

閾値分布の正解づけ: エピソード終了後 90 秒以内に両者スコアの新規 0 読取り
(=次試合開始の物理的証拠) があれば「本物の試合終了」、なければ偽検出候補。

出力: logs/_verify_lockdown_release_fix_2026-08-19/<vid>.json
"""
from __future__ import annotations

import json
import sys
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.board_motion import (  # noqa: E402
    REAL_GAMEPLAY_BOARD_STD_THRESHOLD,
    board_roi_gray,
    board_roi_std,
)
from src.match_end_detector import MatchEndDetector  # noqa: E402
from src.score_ocr import ScoreOcr  # noqa: E402
from src.score_zero import ScoreZeroDetector  # noqa: E402

NPZ_DIR = PROJECT_ROOT / "data" / "indicators_v2" / "boards_lean_subset50_2026-08-19"
FRAMES_DIR = PROJECT_ROOT / "data" / "frames"
OUT_DIR = PROJECT_ROOT / "logs" / "_verify_lockdown_release_fix_2026-08-19"

# 本番定数 (recognition_pipeline.py と同値)
PERSIST_SEC = 30 / 60  # CHAIN_BAN_SEC_AFTER_MATCH_START
VALVE_SEC = 45.0       # POST_MATCH_LOCKDOWN_MAX_SEC
SCORE_MOVE_WINDOW_SEC = 1.0
SCORE_MOVE_MIN_DELTA = 5
MATCH_END_LOCKDOWN_SEC = 5.0
NCC_DETECT_THRESHOLD = 0.55  # 既定閾値 (分布収集は 0.50 以上を全記録)
EPISODE_GAP_SEC = 10.0       # 検出が10秒以上途切れたら別エピソード
REAL_END_SCORE_ZERO_WINDOW_SEC = 90.0  # 終了後この秒数内の両者0読取り=本物

TARGET_EFFECTIVE_FPS = 15.0


class _LatchReplay:
    """ラッチ状態機械のオフライン再現 (旧/新の解除条件を切替可能)。"""

    def __init__(self, numeric_release: bool, moving_release: bool) -> None:
        self.numeric = numeric_release
        self.moving = moving_release
        self.active = False
        self.prev_me_locked = False
        self.started = -1.0
        self.raw_since = -1.0
        self.on_count = 0
        self.releases: list[tuple[float, str]] = []

    def step(
        self, t: float, me_locked: bool, sz_both: bool,
        s1: int | None, s2: int | None, gameplay: bool, moving: bool,
    ) -> bool:
        if me_locked and not self.prev_me_locked:
            if not self.active:
                self.on_count += 1
            self.active = True
            self.started = t
            self.raw_since = -1.0
        self.prev_me_locked = me_locked
        if not self.active:
            return False
        zero = sz_both or (self.numeric and s1 == 0 and s2 == 0)
        if zero:
            if self.raw_since < 0.0:
                self.raw_since = t
            if t - self.raw_since >= PERSIST_SEC and gameplay:
                self.active = False
                self.releases.append((t, "zero"))
        else:
            self.raw_since = -1.0
        if self.active and self.moving and moving and not me_locked and gameplay:
            self.active = False
            self.releases.append((t, "moving"))
        if self.active and self.started >= 0.0 and t - self.started >= VALVE_SEC:
            self.active = False
            self.releases.append((t, "valve45s"))
        return self.active


def process_video(stem: str) -> dict:
    cv2.setNumThreads(1)
    video = FRAMES_DIR / f"video_{stem}.mp4"
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    stride = max(1, int(round(fps / TARGET_EFFECTIVE_FPS)))
    me = MatchEndDetector.load_default()
    sz = ScoreZeroDetector.load_default()
    ocr = ScoreOcr.load_default()

    old_latch = _LatchReplay(numeric_release=False, moving_release=False)
    new_latch = _LatchReplay(numeric_release=True, moving_release=True)
    me_last_detect_t = -1e9

    last1: int | None = None
    last2: int | None = None
    score_hist: list[tuple[float, int]] = []  # (t, s1+s2は別々に見る必要あり)
    hist1: list[tuple[float, int]] = []
    hist2: list[tuple[float, int]] = []

    # 時系列サンプル (npz 突合用): (t, old_lock, new_lock)
    samples_t: list[float] = []
    samples_old: list[int] = []
    samples_new: list[int] = []
    # match_end 検出 (NCC>=0.50) の生ログ: (t, ncc, template)
    detections: list[tuple[float, float, str]] = []
    # 両者スコアの新規0読取り時刻 (本物終了の正解づけ用)
    fresh_zero_times: list[float] = []
    prev_pair_zero = False

    fi = -1
    while True:
        ok = cap.grab()
        if not ok:
            break
        fi += 1
        if fi % stride != 0:
            continue
        ok, frame = cap.retrieve()
        if not ok:
            break
        t = fi / fps
        if frame.shape[:2] != (1080, 1920):
            frame = cv2.resize(frame, (1920, 1080))
        res = me.detect(frame)
        if res.score >= 0.50:
            detections.append((round(t, 2), round(res.score, 4),
                               res.template_name or ""))
        me_locked_now = res.score >= NCC_DETECT_THRESHOLD
        if me_locked_now:
            me_last_detect_t = t
        me_locked = (t - me_last_detect_t) <= MATCH_END_LOCKDOWN_SEC
        sz_both = False
        try:
            sz_both = bool(sz.detect(frame).both_zero)
        except Exception:
            pass
        v1, _c1 = ocr.read_side(frame, "1P")
        v2, _c2 = ocr.read_side(frame, "2P")
        if v1 is not None:
            last1 = v1
            hist1.append((t, v1))
        if v2 is not None:
            last2 = v2
            hist2.append((t, v2))
        hist1 = [(tt, vv) for tt, vv in hist1 if t - tt <= SCORE_MOVE_WINDOW_SEC]
        hist2 = [(tt, vv) for tt, vv in hist2 if t - tt <= SCORE_MOVE_WINDOW_SEC]
        moving = False
        for h in (hist1, hist2):
            vals = [vv for _, vv in h]
            if len(vals) >= 2 and max(vals) - min(vals) >= SCORE_MOVE_MIN_DELTA:
                moving = True
                break
        pair_zero = last1 == 0 and last2 == 0
        if pair_zero and not prev_pair_zero:
            fresh_zero_times.append(round(t, 2))
        prev_pair_zero = pair_zero
        gameplay = (
            board_roi_std(board_roi_gray(frame, "1P"))
            >= REAL_GAMEPLAY_BOARD_STD_THRESHOLD
            and board_roi_std(board_roi_gray(frame, "2P"))
            >= REAL_GAMEPLAY_BOARD_STD_THRESHOLD
        )
        o = old_latch.step(t, me_locked, sz_both, last1, last2, gameplay, moving)
        n = new_latch.step(t, me_locked, sz_both, last1, last2, gameplay, moving)
        samples_t.append(t)
        samples_old.append(int(o))
        samples_new.append(int(n))
    cap.release()

    # match_end エピソード化 + 本物/偽の正解づけ
    episodes: list[dict] = []
    for t, ncc, tmpl in detections:
        if episodes and t - episodes[-1]["t_end"] <= EPISODE_GAP_SEC:
            ep = episodes[-1]
            ep["t_end"] = t
            if ncc > ep["max_ncc"]:
                ep["max_ncc"] = ncc
                ep["template"] = tmpl
        else:
            episodes.append({
                "t_start": t, "t_end": t, "max_ncc": ncc, "template": tmpl,
            })
    for ep in episodes:
        ep["followed_by_fresh_zero"] = any(
            ep["t_end"] < z <= ep["t_end"] + REAL_END_SCORE_ZERO_WINDOW_SEC
            for z in fresh_zero_times
        )

    # npz snapshot 行との突合 (同一行集合での locked 行比率 before/after)
    st = np.asarray(samples_t)
    so = np.asarray(samples_old, dtype=np.int8)
    sn = np.asarray(samples_new, dtype=np.int8)
    npz_path = NPZ_DIR / f"{stem}.npz"
    joined = {}
    if npz_path.exists() and len(st) > 0:
        d = np.load(npz_path, allow_pickle=False)
        tt = np.asarray(d["t_sec"], dtype=np.float64)
        idx = np.clip(np.searchsorted(st, tt, side="right") - 1, 0, len(st) - 1)
        joined = {
            "npz_rows": int(len(tt)),
            "npz_lock_col_ratio": float(np.mean(
                np.asarray(d["post_match_lockdown_active"]) == 1)),
            "replay_old_row_ratio": float(np.mean(so[idx])),
            "replay_new_row_ratio": float(np.mean(sn[idx])),
        }

    out = {
        "video": stem,
        "fps": fps,
        "stride": stride,
        "n_samples": len(samples_t),
        "time_locked_old_pct": float(np.mean(so) * 100) if len(so) else -1,
        "time_locked_new_pct": float(np.mean(sn) * 100) if len(sn) else -1,
        "latch_on_count_old": old_latch.on_count,
        "latch_on_count_new": new_latch.on_count,
        "releases_old": [(round(t, 1), r) for t, r in old_latch.releases],
        "releases_new": [(round(t, 1), r) for t, r in new_latch.releases],
        "fresh_zero_times": fresh_zero_times,
        "match_end_episodes": episodes,
        "joined": joined,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / f"{stem}.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8",
    )
    print(f"[done] {stem}: old={out['time_locked_old_pct']:.1f}% "
          f"new={out['time_locked_new_pct']:.1f}% episodes={len(episodes)}",
          flush=True)
    return out


def main() -> None:
    stems = sys.argv[1:] if len(sys.argv) > 1 else None
    if not stems:
        stems = [p.stem.replace("video_", "") for p in FRAMES_DIR.glob("video_*.mp4")
                 if (NPZ_DIR / (p.stem.replace("video_", "") + ".npz")).exists()]
    stems = [s for s in stems if (FRAMES_DIR / f"video_{s}.mp4").exists()]
    print("targets:", stems, flush=True)
    with Pool(processes=min(8, len(stems))) as pool:
        pool.map(process_video, stems)


if __name__ == "__main__":
    main()
