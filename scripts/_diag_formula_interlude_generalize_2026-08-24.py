"""幕間 (段と段の間の通常スコア表示) の一般性を別動画で確かめる (2026-08-24)。

## なぜ要るか

Q-01 の修正は「掛け算式の段の区切りは幕間で決まる」という観測に立っている。
根拠となった実測は

    logs/_diag_formula_fix_e2e_2026-08-24/trace_on.jsonl
    (video_zenchi_c0BQoMJwwQU、1P の 15 連鎖・13 境界)

の **1 動画・1 連鎖だけ**である。

    1 段の表示   = 各 28 フレーム (0.933 秒) で一定
    段間の幕間   = 13〜19 フレーム (0.433〜0.634 秒)、通常スコアが表示される
    幕間中の得点 = 直前段の「左×右」と完全一致 (12/12)

`docs/KNOWN_WEAKNESSES.md` W34 の「残る限界 ②」に
**「採用前に別動画 2〜3 連鎖で同じフレーム単位トレースを取り直すこと」**を
検収条件として明記した。本スクリプトはそれを行う。

## 何を測るか

指定した動画・区間・サイドについて、フレームごとに

  - 掛け算式が読めたか (`valid` / `left` / `right`)
  - 通常スコアが読めたか (= 幕間)
  - 段累積器の段数と素点

を出す。後段の解析で

  1. 1 段が何フレーム表示されるか
  2. 段と段の間に幕間が何フレーム入るか
  3. 幕間中のスコア増分が直前段の「左×右」と一致するか

を数える。**c0BQoMJwwQU で得た数値が他動画でも成り立つか**が焦点。

使い方:
    python scripts/_diag_formula_interlude_generalize_2026-08-24.py \
        <video_path> <t0> <t1> <side:1P|2P> <tag>

出力: logs/_diag_formula_interlude_generalize_2026-08-24/trace_<tag>.jsonl
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import cv2  # noqa: E402

cv2.setNumThreads(1)

import scripts.visualize_advantage_overlay as vao  # noqa: E402
from src.recognition_pipeline import RecognitionPipeline  # noqa: E402

VIDEO = sys.argv[1]
T0 = float(sys.argv[2])
T1 = float(sys.argv[3])
SIDE = sys.argv[4]
TAG = sys.argv[5]
assert SIDE in ("1P", "2P"), SIDE

WARMUP_SEC = 30.0
LOG_DIR = PROJECT_ROOT / "logs/_diag_formula_interlude_generalize_2026-08-24"
LOG_DIR.mkdir(parents=True, exist_ok=True)
TRACE_PATH = LOG_DIR / f"trace_{TAG}.jsonl"

# 幕間の観測が目的なので、掛け算式の実読と段累積を有効にする。
# `enable_formula_step_interlude` は**あえて OFF** にする:
# 幕間を段の区切りに使う前の生の観測 (幕間が本当に入るか) を見たいため。
FLAGS = dict(
    enable_chain_formula_read_verify=True,
    enable_formula_chain_count_update=True,
    enable_formula_value_read=True,
)

_state: dict = {"f": None}
_orig_load_default = RecognitionPipeline.load_default.__func__


def _patched_load_default(cls, *args, **kwargs):
    kwargs.update(FLAGS)
    return _orig_load_default(cls, *args, **kwargs)


_orig_update = RecognitionPipeline.update


def _patched_update(self, frame_idx: int, time_sec: float, frame):
    r = _orig_update(self, frame_idx, time_sec, frame)
    if not (T0 <= time_sec <= T1) or _state["f"] is None:
        return r
    is1p = SIDE == "1P"
    fr = self._formula_last_read_1p if is1p else self._formula_last_read_2p
    acc = self._formula_accum_1p if is1p else self._formula_accum_2p
    tracker = self._score_tracker_1p if is1p else self._score_tracker_2p
    ev = self._active_chain_1p if is1p else self._active_chain_2p
    _state["f"].write(json.dumps({
        "t": round(time_sec, 3),
        "state": (r.p1 if is1p else r.p2).state.value,
        # 通常スコアが読めた = 掛け算式は非表示 = 幕間。
        "score": tracker.last_score if tracker is not None else None,
        "formula": None if fr is None else {
            "valid": bool(getattr(fr, "valid", False)),
            "left": getattr(fr, "left", None),
            "right": getattr(fr, "right", None),
            "reject": getattr(fr, "reject_reason", None),
        },
        "accum": None if acc is None else {
            "steps": acc.step_count, "power": acc.total_power,
        },
        "active": None if ev is None else {
            "cc": ev.chain_count, "mech": ev.mechanism, "score": ev.total_score,
        },
    }, ensure_ascii=False) + "\n")
    return r


def main() -> None:
    RecognitionPipeline.load_default = classmethod(_patched_load_default)
    RecognitionPipeline.update = _patched_update
    try:
        with TRACE_PATH.open("w", encoding="utf-8") as f:
            _state["f"] = f
            vao.generate(
                video=Path(VIDEO), out=LOG_DIR / f"_dummy_{TAG}.mp4",
                max_sec=0.0, sample_interval=0.0,
                start_sec=max(0.0, T0 - WARMUP_SEC), end_sec=T1,
                warmup_sec=WARMUP_SEC,
                model_dir=PROJECT_ROOT / "data/verify/retrain_model62_2026-08-21",
                layout="panel", panel_subtitle_h=0,
                render=False, dump_timeline_path=None,
                normalize_fps_30=True, use_production_recognition=True,
                resize_1080p=True, enable_counter_reach=False,
                enable_slide_exit_min_display_guard=True,
                force_in_match=False,
            )
    finally:
        _state["f"] = None
        RecognitionPipeline.load_default = classmethod(_orig_load_default)
        RecognitionPipeline.update = _orig_update
    print(f"[保存] {TRACE_PATH}")


if __name__ == "__main__":
    main()
