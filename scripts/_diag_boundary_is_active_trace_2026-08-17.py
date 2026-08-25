"""問1計装: RecognitionPipeline.update をラップし、is_match_active /
match_end_locked をフレーム単位で記録する (本体コード非変更・計装ラッパー)。

同時に、同じフレームに対して score_zero_detector / match_end_detector の
生NCCスコアを (パイプライン内部で使っているのと同一インスタンスで) 再計測し、
「閾値に届いていないのか、瞬間的に届いても is_match_active の
False→True 立ち上がりに繋がっていないのか」を切り分ける。

対象窓: c109 g43 (t=3590-3810, W21実例の周辺、367サンプル相当)。
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import src.recognition_pipeline as rp  # noqa: E402
from scripts.collect_boards_lean import collect_lean  # noqa: E402

OUT_DIR = Path("data/verify/diag_boundary_2026-08-17")
OUT_DIR.mkdir(parents=True, exist_ok=True)

VIDEO = Path("data/frames/video_c109.mp4")
START_SEC = 3590.0
MAX_SEC = 220.0

_rows: list[dict] = []
_orig_update = rp.RecognitionPipeline.update


def _patched_update(self, frame_idx, time_sec, frame):
    result = _orig_update(self, frame_idx, time_sec, frame)
    # 独立再計測 (パイプライン内部の同一 detector インスタンスを使用、
    # 二重ロードなし)。
    sz_s1 = sz_s2 = sz_both = -1.0
    me_score = -1.0
    me_name = ""
    try:
        if self._score_zero_detector is not None:
            sz = self._score_zero_detector.detect(frame)
            sz_s1, sz_s2 = sz.score_1p, sz.score_2p
            sz_both = float(sz.both_zero)
    except Exception:
        pass
    try:
        if self._match_end_detector is not None:
            me = self._match_end_detector.detect(frame)
            me_score = me.score
            me_name = me.template_name or ""
    except Exception:
        pass
    _rows.append({
        "frame_idx": frame_idx,
        "t_sec": round(time_sec, 3),
        "is_match_active": int(result.is_match_active),
        "match_end_locked": int(result.match_end_locked),
        "sz_s1": round(sz_s1, 4),
        "sz_s2": round(sz_s2, 4),
        "sz_both": sz_both,
        "me_score": round(me_score, 4),
        "me_name": me_name,
    })
    return result


def main() -> int:
    rp.RecognitionPipeline.update = _patched_update
    out_npz = OUT_DIR / "c109_g43_trace.npz"
    n = collect_lean(
        VIDEO, out_npz,
        start_sec=START_SEC, max_sec=MAX_SEC,
        enable_boundary_multisignal=True,
    )
    print(f"snapshots={n}", flush=True)

    out_csv = OUT_DIR / "c109_g43_is_active_trace.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(_rows[0].keys()))
        writer.writeheader()
        writer.writerows(_rows)
    print(f"rows={len(_rows)} -> {out_csv}", flush=True)

    # 遷移カウント (False->True, True->False)
    prev = None
    rises = 0
    falls = 0
    for r in _rows:
        cur = r["is_match_active"]
        if prev is not None:
            if prev == 0 and cur == 1:
                rises += 1
            if prev == 1 and cur == 0:
                falls += 1
        prev = cur
    n_true = sum(r["is_match_active"] for r in _rows)
    n_locked_true = sum(r["match_end_locked"] for r in _rows)
    print(
        f"total_frames={len(_rows)} is_active_True={n_true} "
        f"rises(False->True)={rises} falls(True->False)={falls} "
        f"match_end_locked_True={n_locked_true}",
        flush=True,
    )
    print("ALL_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
