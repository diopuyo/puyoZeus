"""C-1a 検証: review_demo_2026-08-12.mp4 の連鎖場面で chain_count_truth を実測する。

DEMO_REVIEW_2026-08-13.md に記録された連鎖場面の周辺 (source時刻) で
1P/2P の score を **逐次デコード** (シークの多用は大容量mp4で著しく遅いため
1 window につき 1 回だけシークし、以後は連続 read() で走査する) して
発火 window (trigger_sec〜end_sec) を特定し、delta_score と ChainCountOcr の
テロップ読みを chain_count_truth.py に通した結果をログファイルへ書き出す。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.chain_count_ocr import (  # noqa: E402
    ChainCountOcr,
    ChainCountReadResult,
    ChainCountWindowResult,
    _extract_monotonic_max_chain_count,
)
from src.chain_count_truth import resolve_chain_count_truth  # noqa: E402
from src.score_ocr import ScoreOcr  # noqa: E402

VIDEO = _ROOT / "data/frames/review_demo_2026-08-12.mp4"
OUT_DIR = Path(
    r"C:\Users\ryouj\AppData\Local\Temp\claude\C--Users-ryouj--gemini-antigravity-scratch-puyo-analyzer"
    r"\9e5c8d51-7ebe-4211-9717-e19e8d042b40\scratchpad\chain_truth_verify"
)
LOG_PATH = OUT_DIR / "verify_log.txt"

# DEMO_REVIEW_2026-08-13.md の指摘場面の周辺 (source時刻、広めのwindow)。
SCAN_WINDOWS: list[tuple[str, float, float]] = [
    ("scene2_34s", 20.0, 50.0),
    ("scene3_235s", 220.0, 260.0),
    ("scene4_347s", 330.0, 365.0),
]


def _log(f, msg: str) -> None:
    print(msg)
    f.write(msg + "\n")
    f.flush()


def _scan_sequential(
    cap: cv2.VideoCapture, t0: float, t1: float, score_ocr: ScoreOcr,
) -> list[tuple[float, int | None, int | None]]:
    """[t0, t1] を 1 回シーク + 連続 read() で走査し (t, score_1p, score_2p) を返す。"""
    cap.set(cv2.CAP_PROP_POS_MSEC, t0 * 1000.0)
    out: list[tuple[float, int | None, int | None]] = []
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if t > t1:
            break
        s1, _ = score_ocr.read_side(frame, "1P")
        s2, _ = score_ocr.read_side(frame, "2P")
        out.append((t, s1, s2))
    return out


def _find_jump(
    samples: list[tuple[float, int | None]],
) -> tuple[float, float, int] | None:
    """最大の score jump 区間 (trigger_sec, end_sec, delta) を返す (簡易・貪欲)。"""
    valid = [(t, s) for t, s in samples if s is not None]
    if len(valid) < 2:
        return None
    best = None
    for i in range(len(valid) - 1):
        t0, s0 = valid[i]
        for j in range(i + 1, min(i + 60, len(valid))):
            t1, s1 = valid[j]
            delta = s1 - s0
            if delta > 0 and (best is None or delta > best[2]):
                best = (t0, t1, delta)
    return best


def _telop_window_sequential(
    cap: cv2.VideoCapture, side: str, ocr: ChainCountOcr, trig: float, end: float,
) -> ChainCountWindowResult:
    """[trig, end+1.0] を 1 回シーク + 連続 read() でテロップ読みする。"""
    cap.set(cv2.CAP_PROP_POS_MSEC, trig * 1000.0)
    hits: list[tuple[float, int]] = []
    samples: list[ChainCountReadResult] = []
    n_hits = 0
    t_end = end + 1.0
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if t > t_end:
            break
        res = ocr.read_side(frame, side)
        samples.append(res)
        if res.chain_count is not None:
            hits.append((t, res.chain_count))
            n_hits += 1
    max_count = _extract_monotonic_max_chain_count(hits)
    return ChainCountWindowResult(
        max_chain_count=max_count, samples=tuple(samples), n_hits=n_hits,
    )


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(VIDEO))
    if not cap.isOpened():
        print(f"動画を開けない: {VIDEO}")
        return 1
    score_ocr = ScoreOcr.load_default()
    chain_ocr = ChainCountOcr.load_default()

    with LOG_PATH.open("w", encoding="utf-8") as f:
        for label, t0, t1 in SCAN_WINDOWS:
            t_start = time.time()
            _log(f, f"\n=== {label}: t={t0}-{t1} ===")
            both = _scan_sequential(cap, t0, t1, score_ocr)
            _log(f, f"  score走査完了 ({time.time() - t_start:.1f}秒, {len(both)}サンプル)")
            for side_idx, side in enumerate(("1P", "2P")):
                samples = [(t, s[side_idx]) for t, *s in both]
                jump = _find_jump(samples)
                if jump is None:
                    _log(f, f"  {side}: score jump 検出できず")
                    continue
                trig, end, delta = jump
                _log(f, f"  {side}: trigger={trig:.2f}s end={end:.2f}s delta_score={delta}")
                telop_window = _telop_window_sequential(cap, side, chain_ocr, trig, end)
                result = resolve_chain_count_truth(telop_window, delta_score=delta)
                _log(
                    f,
                    f"    telop={result.telop_chain_count} (n_hits={result.telop_n_hits}) "
                    f"score={result.score_chain_count} (ratio={result.score_ratio}) "
                    f"-> truth={result.chain_count} reason={result.reason}",
                )
                mid = (trig + end) / 2.0
                cap.set(cv2.CAP_PROP_POS_MSEC, mid * 1000.0)
                ok, frame = cap.read()
                if ok:
                    out_path = OUT_DIR / f"{label}_{side}_{mid:.1f}s.jpg"
                    cv2.imwrite(str(out_path), frame)
                    _log(f, f"    crop保存: {out_path}")
    cap.release()
    _log_done = LOG_PATH
    print(f"\n完了: {_log_done}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
