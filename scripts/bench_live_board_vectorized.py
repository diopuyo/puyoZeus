"""盤面読取の一括経路 (live_board_vectorized) の ON/OFF の時間を、同じ実コマで交互に測る。

- OFF・ON の 2 本のパイプラインを同じプロセスに作り、連続コマを両方へ順に与える。
  1 コマごとに先に回す側を入れ替える (順序効果・背景負荷の単調変化が片側へ乗らないように)。
- 暖機のコマ (--warmup) は集計から除く。
- 測るもの: RecognitionPipeline.update 全体、ImageReader.read_both_boards、read_board_hsv_only。
  どれも perf_counter の壁時計 (ms)。P50 / P95 / 平均と、コマごとの差 (ON - OFF) の平均。
- 速度のために出力を変えない前提の機構なので、同一性は scripts/verify_live_board_vectorized.py で別に確認する。

使い方 (配布 Python・配布 app を --assets に):
  python run_mod.py scripts.bench_live_board_vectorized --assets APP --config recognition_config.json \
      --video VIDEO --start 4602 --count 1000 --warmup 100 --output result.json
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
from pathlib import Path
from time import perf_counter
from typing import Any, Iterator

import cv2
import numpy as np

from scripts.verify_live_board_vectorized import build_pipeline, iter_frames

PERCENTILES = (50, 95)
TIMED_METHODS = ("read_both_boards", "read_board_hsv_only")


def attach_timers(pipe: Any) -> dict[str, list[float]]:
    """リーダーの呼出時間 (ms) をインスタンスごとに貯める (クラスは書き換えない)。"""
    store: dict[str, list[float]] = {name: [] for name in TIMED_METHODS}
    reader = pipe._reader
    for name in TIMED_METHODS:
        original = getattr(reader, name)

        def wrapper(*args: Any, __original: Any = original, __name: str = name, **kwargs: Any) -> Any:
            started = perf_counter()
            try:
                return __original(*args, **kwargs)
            finally:
                store[__name].append((perf_counter() - started) * 1000)
        setattr(reader, name, wrapper)
    return store


def summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return dict(n=0)
    return dict(n=len(values), mean_ms=statistics.fmean(values),
                p50_ms=float(np.percentile(values, PERCENTILES[0])),
                p95_ms=float(np.percentile(values, PERCENTILES[1])),
                max_ms=max(values))


def cpu_clock_mhz() -> int | None:
    """計測前後の CPU 周波数 (MHz)。速度の A/B で周波数張り付きを疑えるよう記録する。"""
    try:
        import subprocess
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "(Get-CimInstance Win32_Processor | Select-Object -First 1).CurrentClockSpeed"],
            capture_output=True, text=True, timeout=20).stdout.strip()
        return int(out)
    except Exception:
        return None


def source_frames(options: argparse.Namespace) -> Iterator[tuple[int, float, np.ndarray]]:
    """動画の連続コマ、または PNG の一覧 (--repeat 回くり返す。コマ番号は通し)。"""
    if not options.png:
        yield from iter_frames(options)
        return
    frames = [frame for _, _, frame in iter_frames(options)]
    for number in range(len(frames) * options.repeat):
        yield number, number / 30.0, frames[number % len(frames)]


def run(options: argparse.Namespace) -> dict[str, Any]:
    pipes = {"off": build_pipeline(options.config, False, options.native),
             "on": build_pipeline(options.config, True, options.native)}
    stores = {name: attach_timers(pipe) for name, pipe in pipes.items()}
    update_ms: dict[str, list[float]] = {"off": [], "on": []}
    paired: list[float] = []
    clock_before = cpu_clock_mhz()
    for count, (index, time_sec, frame) in enumerate(source_frames(options)):
        order = ("off", "on") if count % 2 == 0 else ("on", "off")
        if count == options.warmup:
            for name in pipes:
                update_ms[name].clear()
                for values in stores[name].values():
                    values.clear()
        elapsed: dict[str, float] = {}
        for name in order:
            started = perf_counter()
            pipes[name].update(index, time_sec, frame)
            elapsed[name] = (perf_counter() - started) * 1000
            update_ms[name].append(elapsed[name])
        if count >= options.warmup:
            paired.append(elapsed["on"] - elapsed["off"])
    result: dict[str, Any] = dict(
        frames_measured=len(paired), warmup=options.warmup, native=options.native,
        source=str(options.video or options.png), start=options.start,
        cpu_mhz_before=clock_before, cpu_mhz_after=cpu_clock_mhz(),
        python=sys.version.split()[0], platform=platform.platform(),
        paired_update_diff_mean_ms=statistics.fmean(paired) if paired else None)
    for name in pipes:
        result[name] = dict(update=summarize(update_ms[name]),
                            **{method: summarize(values) for method, values in stores[name].items()})
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--video", type=Path)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--png", type=Path, nargs="*", default=[])
    parser.add_argument("--repeat", type=int, default=20, help="PNG 一覧のくり返し回数")
    parser.add_argument("--obs-normalize", action="store_true")
    parser.add_argument("--no-native", dest="native", action="store_false")
    parser.add_argument("--output", type=Path)
    options = parser.parse_args(argv)
    cv2.setNumThreads(1)
    for name in ("config", "video", "output"):
        value = getattr(options, name)
        if value is not None:
            setattr(options, name, value.resolve())
    options.png = [p.resolve() for p in options.png]
    os.chdir(options.assets)
    report = run(options)
    text = json.dumps(report, indent=2, ensure_ascii=False)
    if options.output:
        options.output.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
