"""user 定義の「試合時間」を実測し、 現行の game_idx とのズレを出す (2026-08-09).

## user 定義 (2026-08-09)
> 試合時間としてカウントしたいのは **一手目からばたんきゅー・やったまで**、
> 補助として **本数遷移** をつかう

- 開始 = **最初の1手** (最初のツモ設置) — オープニング演出は含めない
- 終了 = **試合終了テロップ** (やった! / ばたんきゅー)
- 補助 = WIN★パネルの数値変化

## なぜ測るか
現行の `game_idx` は「score が 500 点以上減った」だけで境界を切っており、
実測で c54 の game=0 が **199.9 秒** (実際はオープニング演出 + 1 試合) に
なっていた。 user 伝授では **実際の試合はほぼ 96 秒以下**。
つまり現行の試合長は非試合時間で水増しされている。

本スクリプトは定義通りの試合時間を測り、 現行との差を数値で出す。
**読み取り専用** (認識・学習には一切影響しない)。

出力: data/verify/true_match_time_2026-08-09.tsv
"""
from __future__ import annotations

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

OUT_TSV = _ROOT / "data" / "verify" / "true_match_time_2026-08-09.tsv"
FRAMES_DIR = Path.home() / "frames"
# 検証対象 (現行 game_idx で「長い試合」と出ていた動画を含める)
TARGETS: tuple[str, ...] = ("c54", "c44", "c32")
# 1 動画あたりの処理秒 (全長は重いので先頭のみ)
MAX_SEC: float = 600.0


def _flag_kwargs() -> dict:
    kwargs: dict = {}
    for f in RECOGNITION_ADOPTED:
        parts = f.flag.split()
        name = parts[0].lstrip("-").replace("-", "_")
        kwargs[name] = float(parts[1]) if len(parts) > 1 else True
    return kwargs


def _measure(video: Path) -> list[dict]:
    """1 動画分の試合区間を (最初の1手, 終了テロップ) で切り出す。"""
    pipeline = RecognitionPipeline.load_default(
        force_in_match=True, **_flag_kwargs(),
    )
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    matches: list[dict] = []
    first_move: float | None = None
    prev_end_locked = False
    prev_tsumo1 = prev_tsumo2 = 0
    fi = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = fi / fps
        if t > MAX_SEC:
            break
        # 認識は 1920x1080 前提。 バーストガード等が frame の ROI を直接参照
        # するため、 サイズが違うと空パッチになり cv2 が落ちる (2026-08-09)。
        if frame.shape[1] != 1920 or frame.shape[0] != 1080:
            frame = cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_AREA)
        r = pipeline.update(fi, t, frame)
        # 最初の1手 = ツモ設置数が 0 -> 1 になった時刻 (どちらかの side)
        t1 = pipeline.tsumo_count("1P")
        t2 = pipeline.tsumo_count("2P")
        if first_move is None and (t1 > prev_tsumo1 or t2 > prev_tsumo2):
            first_move = t
        prev_tsumo1, prev_tsumo2 = t1, t2
        # 終了テロップ (やった! / ばたんきゅー) の立ち上がり
        end_locked = bool(getattr(pipeline, "_last_match_end_locked", False))
        if end_locked and not prev_end_locked and first_move is not None:
            matches.append({
                "start_sec": round(first_move, 2),
                "end_sec": round(t, 2),
                "duration_sec": round(t - first_move, 2),
            })
            first_move = None
            prev_tsumo1 = prev_tsumo2 = 0
        prev_end_locked = end_locked
        fi += 1
    cap.release()
    return matches


def main() -> int:
    lines = ["video\tidx\tstart_sec\tend_sec\tduration_sec"]
    all_dur: list[float] = []
    for vid in TARGETS:
        src = FRAMES_DIR / f"video_{vid}.mp4"
        if not src.exists():
            print(f"skip {vid} (動画なし)")
            continue
        print(f"[true] {vid} 測定中...", flush=True)
        ms = _measure(src)
        for i, m in enumerate(ms, start=1):
            lines.append(
                f"{vid}\t{i}\t{m['start_sec']}\t{m['end_sec']}\t{m['duration_sec']}"
            )
            all_dur.append(m["duration_sec"])
        if ms:
            d = np.array([m["duration_sec"] for m in ms])
            print(f"  試合 {len(ms)} 件 / 長さ 中央値 {np.median(d):.1f}s "
                  f"最大 {d.max():.1f}s", flush=True)
        else:
            print("  試合を検出できず (終了テロップ未検出の可能性)", flush=True)
    OUT_TSV.parent.mkdir(parents=True, exist_ok=True)
    OUT_TSV.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if all_dur:
        a = np.array(all_dur)
        print(f"\n合計 {len(a)} 試合")
        print(f"  中央値 {np.median(a):.1f}s / p90 {np.percentile(a, 90):.1f}s "
              f"/ 最大 {a.max():.1f}s")
        print(f"  96 秒を超えた試合: {int((a > 96).sum())} 件 "
              f"({(a > 96).mean():.1%})")
    print(f"出力: {OUT_TSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
