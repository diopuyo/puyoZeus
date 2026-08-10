"""ChainEvent.end_sec が実際の連鎖終了とどれだけズレるかを全域で測る (2026-08-09).

## 背景
連鎖数表示のリセット基準に `end_sec` (連鎖終了の予測時刻) を使ったところ、
連鎖の途中で連鎖数が消える不具合が出た。 実測では t=24.70 発火・end=25.30 に
対し、 状態機械は t=27.43 まで chain を維持していた (**約 2 秒の過小**)。

その場の対処 (リセット基準を表示上の連鎖終了へ変更) は済んでいるが、
**1 シーンの観察で終わらせない**という規律 ([[feedback-overfitting-awareness]])
に従い、 ズレの分布を全域で確定させる。 end_sec は発火直後の期待ダメージ
見積もりにも使われるため、 系統的な過小があるなら影響範囲が広い。

## 測り方
各動画を認識に通し、 ChainEvent が立った時刻とその end_sec を記録し、
**状態機械が実際に chain / gravity_settle から抜けた時刻**と比較する。
    gap = (実際の連鎖終了) - (end_sec)
gap > 0 なら end_sec は過小 (実際の連鎖の方が長い)。

読み取り専用。 認識・評価には一切影響しない。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.board_state_machine import BoardState  # noqa: E402
from src.console_init import init_console  # noqa: E402

init_console()

from src.production_config import RECOGNITION_ADOPTED  # noqa: E402
from src.recognition_pipeline import RecognitionPipeline  # noqa: E402

OUT_JSONL = _ROOT / "data" / "verify" / "chain_end_sec_gap_2026-08-09.jsonl"
FRAMES_DIR = Path.home() / "frames"
# 連鎖とみなす状態 (段間の重力待ちも連鎖の一部)。
CHAIN_STATES = frozenset({BoardState.CHAIN, BoardState.GRAVITY_SETTLE})


def _flag_kwargs() -> dict:
    """本番構成のフラグを load_default の引数名へ変換する。"""
    kwargs: dict = {}
    for f in RECOGNITION_ADOPTED:
        parts = f.flag.split()
        name = parts[0].lstrip("-").replace("-", "_")
        kwargs[name] = float(parts[1]) if len(parts) > 1 else True
    return kwargs


def _measure(video: Path, video_id: str) -> list[dict]:
    """1 動画分の (発火時刻, end_sec, 実際の終了, gap) を返す。"""
    pipeline = RecognitionPipeline.load_default(
        force_in_match=True, **_flag_kwargs(),
    )
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    out: list[dict] = []
    # side -> 進行中の連鎖情報
    pending: dict[str, dict] = {}
    fi = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = fi / fps
        r = pipeline.update(fi, t, frame)
        for side, sr in (("1P", r.p1), ("2P", r.p2)):
            ev = getattr(sr, "chain_event", None)
            in_chain = sr.state in CHAIN_STATES
            if ev is not None and in_chain:
                end = getattr(ev, "end_sec", None)
                trig = getattr(ev, "trigger_sec", None)
                cur = pending.get(side)
                if cur is None or (trig is not None and trig != cur["trigger"]):
                    pending[side] = {
                        "trigger": float(trig) if trig is not None else t,
                        "end": float(end) if end is not None else float("nan"),
                        "chain_count": getattr(ev, "chain_count", None),
                        "mechanism": getattr(ev, "mechanism", None),
                    }
                elif end is not None and end > cur["end"]:
                    cur["end"] = float(end)  # 再検知で伸びた分を反映
            if not in_chain and side in pending:
                info = pending.pop(side)
                out.append({
                    "video": video_id, "side": side,
                    "trigger_sec": round(info["trigger"], 3),
                    "end_sec": round(info["end"], 3),
                    "actual_end_sec": round(t, 3),
                    "gap_sec": round(t - info["end"], 3),
                    "chain_count": info["chain_count"],
                    "mechanism": info["mechanism"],
                })
        fi += 1
    cap.release()
    return out


def main() -> int:
    videos = sorted(FRAMES_DIR.glob("video_*.mp4"))
    if not videos:
        print(f"動画が無い: {FRAMES_DIR}")
        return 1
    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    n_total = 0
    with OUT_JSONL.open("w", encoding="utf-8") as f:
        for v in videos:
            vid = v.stem.replace("video_", "")
            print(f"[gap] {vid} ...", flush=True)
            try:
                rows = _measure(v, vid)
            except Exception as e:
                print(f"  ERROR {vid}: {e}")
                continue
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
            f.flush()
            n_total += len(rows)
            if rows:
                gaps = np.array([r["gap_sec"] for r in rows], dtype=float)
                gaps = gaps[~np.isnan(gaps)]
                if len(gaps):
                    print(f"  n={len(rows)} gap中央値={np.median(gaps):+.2f}s "
                          f"p90={np.percentile(gaps, 90):+.2f}s", flush=True)
    print(f"\n合計 {n_total} 件 -> {OUT_JSONL}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
