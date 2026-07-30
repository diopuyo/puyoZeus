"""cv2.matchTemplate を呼び出し元タグ付きでラップし、時間配分を実測する。

背景 (2026-07-30):
    回数内訳では UI マスク判定が 75.0% (981回/フレーム) で支配的だが、
    回数比率 ≠ 時間比率。match_end/telop は 800x600 / 720x400 の大きな
    ROI を走査するため、回数がわずか 0.6% でも時間では支配的な可能性がある。
    「回数75%のUIマスクを潰しても時間が半分も減らない」罠を避けるため、
    削減着手前に呼び出し元ごとの累計時間を確定させる。

方針 (厳守事項):
    - src/ は一切変更しない。本スクリプト内で cv2.matchTemplate を
      モンキーパッチし、sys._getframe で呼び出し元を特定する。
    - cProfile は使わない。time.perf_counter で直接測る。
    - 計測オーバーヘッド自体 (baseline 無パッチ vs 計装あり) を報告する。
    - cv2.setNumThreads(1) と既定値の両方で測る。

実行例 (WSL):
    nice -n 19 ./venv/bin/python -m scripts._diag_matchtemplate_by_caller_2026-07-30 \
        --video data/frames/video_c60.mp4 --start-sec 1451 --frames 60
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field

import cv2
import numpy as np

from src.recognition_pipeline import RecognitionPipeline

# --- 定数 (マジックナンバー禁止のため名前を与える) ---
TARGET_W: int = 1920
TARGET_H: int = 1080
WARMUP_FRAMES: int = 10          # 起動直後のモデルロード等を除くための助走
CALL_SITE_STACK_DEPTH: int = 12  # 呼び出し元特定で遡るフレーム数の上限
SINGLE_THREAD: int = 1           # 比較対象の「1スレッド」設定


@dataclass
class CallSiteStat:
    """1 呼び出し元あたりの累積統計 (回数・累計秒・ROI/テンプレ面積)。"""

    count: int = 0
    total_sec: float = 0.0
    roi_area_sum: int = 0
    tmpl_area_sum: int = 0
    roi_shape_seen: tuple[int, int] | None = None
    tmpl_shape_seen: tuple[int, int] | None = None


def _short(path: str) -> str:
    """フルパスをファイル名のみに短縮する (Windows/POSIX 両対応)。"""
    return path.split("/")[-1].split("\\")[-1]


def _find_call_site(immediate_frame: "object") -> tuple[str, int, str, int]:
    """呼び出し元の (直接呼び出し元 file:line, 経路を辿った外側 file:line) を返す。

    直接の呼び出し元 (ui_mask.py 等、matchTemplate を直書きしている行) は
    ROI/テンプレのサイズを特定するのに使う。同じ直接呼び出し元でも複数の
    経路 (例: telop_detector.detect() が recognition_pipeline 経由 /
    image_reader 経由の両方から呼ばれる) を区別するため、直接呼び出し元と
    ファイルが異なる最初の祖先フレームも併せて返す。
    """
    immediate_file = immediate_frame.f_code.co_filename
    immediate_line = immediate_frame.f_lineno
    outer = immediate_frame
    for _ in range(CALL_SITE_STACK_DEPTH):
        nxt = outer.f_back
        if nxt is None:
            break
        outer = nxt
        if nxt.f_code.co_filename != immediate_file:
            break
    return (
        _short(immediate_file), immediate_line,
        _short(outer.f_code.co_filename), outer.f_lineno,
    )


class MatchTemplatePatcher:
    """cv2.matchTemplate をモンキーパッチして呼び出し元別に集計する。

    mode="off": パッチしない (baseline、無計装の実時間測定用)。
    mode="timing_only": 呼び出し元特定をせず、時間と回数だけ集計 (単一バケツ)。
        → 「ラップするだけ」のオーバーヘッドを測るための中間モード。
    mode="full_site": sys._getframe で呼び出し元を特定して個別集計する。
    """

    def __init__(self, mode: str) -> None:
        self.mode = mode
        self.stats: dict[str, CallSiteStat] = defaultdict(CallSiteStat)
        self._original = cv2.matchTemplate
        self._matchtemplate_total_sec: float = 0.0

    def __enter__(self) -> "MatchTemplatePatcher":
        if self.mode != "off":
            cv2.matchTemplate = self._wrapped
        return self

    def __exit__(self, *_exc: object) -> None:
        cv2.matchTemplate = self._original

    def _wrapped(self, *args: object, **kwargs: object) -> np.ndarray:
        """cv2.matchTemplate の代替。実処理は元関数に委譲し前後で時間を測る。"""
        t0 = time.perf_counter()
        result = self._original(*args, **kwargs)
        elapsed = time.perf_counter() - t0
        self._matchtemplate_total_sec += elapsed
        key = "ALL" if self.mode == "timing_only" else self._site_key()
        st = self.stats[key]
        st.count += 1
        st.total_sec += elapsed
        self._record_shapes(st, args)
        return result

    def _site_key(self) -> str:
        """full_site モード用: 呼び出し元を特定してラベル化する。"""
        caller = sys._getframe(2)  # 0=_site_key, 1=_wrapped, 2=実呼び出し元
        i_file, i_line, o_file, o_line = _find_call_site(caller)
        return f"{i_file}:{i_line} <- {o_file}:{o_line}"

    @staticmethod
    def _record_shapes(st: CallSiteStat, args: tuple[object, ...]) -> None:
        """ROI/テンプレの shape を記録する (面積計算・サイズ表示用)。"""
        if len(args) >= 2:
            roi, tmpl = args[0], args[1]
            if hasattr(roi, "shape") and hasattr(tmpl, "shape"):
                st.roi_shape_seen = tuple(roi.shape[:2])
                st.tmpl_shape_seen = tuple(tmpl.shape[:2])
                st.roi_area_sum += roi.shape[0] * roi.shape[1]
                st.tmpl_area_sum += tmpl.shape[0] * tmpl.shape[1]

    @property
    def matchtemplate_total_sec(self) -> float:
        return self._matchtemplate_total_sec


def build_pipeline() -> RecognitionPipeline:
    """本番(レンダ)と同じ設定でパイプラインを作る。"""
    return RecognitionPipeline.load_default(
        stable_frame_count=3,
        load_score_ocr=True,
        enable_chain_tracker=True,
        temporal_smoothing=1,
        force_in_match=True,
    )


def run_frames(pipeline: RecognitionPipeline, video: str,
               start_sec: float, n_frames: int) -> int:
    """指定区間を全フレーム処理する。処理できたフレーム数を返す。"""
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    start_frame = int(start_sec * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    done = 0
    for i in range(n_frames):
        ok, frame = cap.read()
        if not ok:
            break
        if frame.shape[:2] != (TARGET_H, TARGET_W):
            frame = cv2.resize(frame, (TARGET_W, TARGET_H),
                               interpolation=cv2.INTER_AREA)
        fi = start_frame + i
        pipeline.update(fi, fi / fps, frame)
        done += 1
    cap.release()
    return done


@dataclass
class RunResult:
    """1回の計測run (baseline/instrumented) の結果。"""

    mode: str
    threads: int
    done_frames: int
    wall_sec: float
    matchtemplate_total_sec: float = 0.0
    site_stats: dict[str, CallSiteStat] = field(default_factory=dict)


def measure_one(video: str, start_sec: float, n_frames: int,
                 threads: int, mode: str) -> RunResult:
    """1構成 (スレッド数 x モード) を warmup + 計測してRunResultを返す。"""
    cv2.setNumThreads(threads)
    pipeline = build_pipeline()
    # warmup はパッチ無しで実施 (モデルロード等の初回コストを除く)
    run_frames(pipeline, video, start_sec, WARMUP_FRAMES)
    with MatchTemplatePatcher(mode=mode) as patcher:
        t0 = time.perf_counter()
        done = run_frames(pipeline, video, start_sec + 1.0, n_frames)
        wall = time.perf_counter() - t0
    return RunResult(
        mode=mode, threads=threads, done_frames=done, wall_sec=wall,
        matchtemplate_total_sec=patcher.matchtemplate_total_sec,
        site_stats=dict(patcher.stats),
    )


def print_run_summary(label: str, r: RunResult) -> None:
    """1 run の要約 (fps, matchTemplate 比率) を表示する。"""
    fps = r.done_frames / r.wall_sec if r.wall_sec > 0 else 0.0
    per_frame_ms = r.wall_sec / max(r.done_frames, 1) * 1000
    mt_pct = (r.matchtemplate_total_sec / r.wall_sec * 100
              if r.wall_sec > 0 else 0.0)
    print(f"\n--- {label} (threads={r.threads}, mode={r.mode}) ---")
    print(f"  フレーム数 {r.done_frames} / 総時間 {r.wall_sec:.3f}秒 "
          f"= 1フレーム {per_frame_ms:.1f}ms → {fps:.2f} fps 相当")
    if r.mode != "off":
        other_sec = r.wall_sec - r.matchtemplate_total_sec
        print(f"  matchTemplate 累計 {r.matchtemplate_total_sec:.3f}秒 "
              f"({mt_pct:.1f}%) / それ以外 {other_sec:.3f}秒 "
              f"({100 - mt_pct:.1f}%)")


def print_site_breakdown(r: RunResult) -> None:
    """full_site モードの呼び出し元別内訳を表示する。"""
    if not r.site_stats:
        return
    total = sum(st.total_sec for st in r.site_stats.values())
    rows = sorted(r.site_stats.items(), key=lambda kv: -kv[1].total_sec)
    print(f"\n  === 呼び出し元別内訳 (threads={r.threads}) ===")
    for key, st in rows:
        avg_us = st.total_sec / max(st.count, 1) * 1e6
        pct_of_mt = st.total_sec / total * 100 if total > 0 else 0.0
        pct_of_wall = st.total_sec / r.wall_sec * 100 if r.wall_sec > 0 else 0.0
        roi = st.roi_shape_seen
        tmpl = st.tmpl_shape_seen
        print(f"    {pct_of_mt:5.1f}%(mt) {pct_of_wall:5.1f}%(全体) "
              f"{st.total_sec:7.3f}秒  {st.count:>5}回  "
              f"平均{avg_us:8.1f}us  roi={roi} tmpl={tmpl}  {key}")


def print_overhead(baseline: RunResult, instrumented: RunResult) -> None:
    """baseline (無パッチ) と計装ありの壁時計差からオーバーヘッドを報告する。"""
    delta = instrumented.wall_sec - baseline.wall_sec
    pct = delta / baseline.wall_sec * 100 if baseline.wall_sec > 0 else 0.0
    print(f"\n  [オーバーヘッド] threads={baseline.threads}: "
          f"baseline {baseline.wall_sec:.3f}秒 → 計装後 "
          f"{instrumented.wall_sec:.3f}秒 (差分 {delta:+.3f}秒, {pct:+.1f}%)")


def run_thread_setting(video: str, start_sec: float, n_frames: int,
                        threads: int) -> None:
    """指定スレッド数で baseline / timing_only / full_site の3通りを測る。"""
    baseline = measure_one(video, start_sec, n_frames, threads, mode="off")
    timing_only = measure_one(video, start_sec, n_frames, threads,
                               mode="timing_only")
    full_site = measure_one(video, start_sec, n_frames, threads,
                             mode="full_site")
    print_run_summary("baseline(無パッチ)", baseline)
    print_run_summary("timing_only(単純ラップ)", timing_only)
    print_run_summary("full_site(呼び出し元特定込み)", full_site)
    print_overhead(baseline, timing_only)
    print_overhead(baseline, full_site)
    print_site_breakdown(full_site)


def main() -> None:
    """1スレッド / 既定スレッドの両方で呼び出し元別の時間配分を測る。"""
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--start-sec", type=float, default=1451.0)
    ap.add_argument("--frames", type=int, default=60)
    args = ap.parse_args()

    default_threads = cv2.getNumThreads()  # 何もいじらない既定値 (通常16)
    print(f"既定 cv2 スレッド数 = {default_threads}")

    print(f"\n########## threads = {SINGLE_THREAD} ##########")
    run_thread_setting(args.video, args.start_sec, args.frames, SINGLE_THREAD)

    print(f"\n########## threads = {default_threads} (既定) ##########")
    run_thread_setting(args.video, args.start_sec, args.frames,
                        default_threads)


if __name__ == "__main__":
    main()
