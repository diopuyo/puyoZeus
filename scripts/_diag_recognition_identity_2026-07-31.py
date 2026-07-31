"""認識出力の全項目が2つの設定で同一かを検証する汎用検証器 (2026-07-31)。

## なぜ作ったか

2026-07-30〜31 の高速化7件はいずれも「確定盤面の差分ゼロ」で採否を判断したが、
**比較していた項目が狭かった**:
  - 比較していた: confirmed_board、一部で score
  - **比較していなかった: next_pair / dnext_pair / cnn_board / chain_event /
    state / drift / erasure_alerts / おじゃま関連**

next_detector のバッチ化のように **ネクスト値を変えうる変更**では、
盤面だけ比べても何も検出できない。窓も各動画 t=1500 前後の 10 秒程度に偏っていた。

## 何を比較するか

`SideResult` の主要フィールドを全て tuple 化して frame 単位で突き合わせ、
**どのフィールドが何フレーム違うか**を項目別に出す。
「盤面は同じだがネクストが違う」を見逃さないため、必ず項目別に報告する。

## 窓の選び方

`--windows` で局面種別を指定する。既定は 4 種:
  match_start : 試合開始直後 (勝利数パネルの start_sec から)
  mid_match   : 試合中盤 (start_sec + 30s)
  match_end   : 試合終了前後 (end_sec をまたぐ)
  deep        : 動画後半の固定時刻 (収集窓の外 = 未検証領域)

## 使い方

    # 2つのコミット間で比較 (worktree を使う)
    PYTHONPATH=<old_worktree>:. ... で旧を、PYTHONPATH=. で新を別々に走らせるのではなく、
    本スクリプトは **同一プロセス内でフラグを切り替えて** 比較する。
    フラグ名は --flag で指定する (bool のパイプライン引数、またはモジュール定数)。

    PYTHONPATH=. ./venv/bin/python -m scripts._diag_recognition_identity_2026-07-31 \
        --flag enable_score_ocr_matmul --videos video_c56 video_c26 video_c58
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

TARGET_W, TARGET_H = 1920, 1080
WARMUP_FRAMES: int = 5
# 各窓のフレーム数 (30fps で 10 秒)
WINDOW_FRAMES: int = 300
# 試合開始/中盤窓のオフセット [秒]
MID_MATCH_OFFSET_SEC: float = 30.0
# 「収集窓の外」を突くための動画後半の時刻 [秒] (評価は最初26分に限定されていた)
DEEP_WINDOW_SEC: float = 2400.0

# 比較する SideResult のフィールド (盤面系は grid を bytes 化)
BOARD_FIELDS: tuple[str, ...] = ("cnn_board", "inferred_board", "confirmed_board")
SCALAR_FIELDS: tuple[str, ...] = (
    "state", "score", "score_delta", "next_pair", "dnext_pair",
    "chain_event", "erasure_alerts", "transition_drop_alerts",
)
# 結果全体 (side に属さない) のフィールド
TOP_FIELDS: tuple[str, ...] = ("is_match_active",)


def _grid_bytes(board: Any) -> bytes | None:
    """Board から grid の bytes を取り出す (None 安全)。"""
    if board is None:
        return None
    grid = getattr(board, "grid", None)
    return None if grid is None else np.asarray(grid).tobytes()


def _snapshot(result: Any) -> dict[str, Any]:
    """1 フレームの認識出力を項目別 dict にする。"""
    snap: dict[str, Any] = {}
    for f in TOP_FIELDS:
        snap[f] = repr(getattr(result, f, None))
    for side, attr in (("1P", "side_1p"), ("2P", "side_2p")):
        sr = getattr(result, attr, None)
        for f in BOARD_FIELDS:
            snap[f"{side}.{f}"] = _grid_bytes(
                getattr(sr, f, None) if sr is not None else None,
            )
        for f in SCALAR_FIELDS:
            snap[f"{side}.{f}"] = repr(
                getattr(sr, f, None) if sr is not None else None,
            )
    return snap


def _read_frames(video: Path, start_sec: float, n: int) -> list[np.ndarray]:
    """動画から連続フレームを読み出す (1920x1080 に正規化)。"""
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        return []
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_sec * fps))
    out: list[np.ndarray] = []
    for _ in range(n):
        ok, frame = cap.read()
        if not ok:
            break
        if frame.shape[1] != TARGET_W or frame.shape[0] != TARGET_H:
            frame = cv2.resize(
                frame, (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA,
            )
        out.append(frame)
    cap.release()
    return out


def _build_windows(
    video_name: str, panel_dir: Path, kinds: list[str],
) -> list[tuple[str, float]]:
    """局面種別ごとに (ラベル, 開始秒) を組み立てる。"""
    path = panel_dir / f"{video_name}.json"
    games: list[dict] = []
    if path.exists():
        games = json.loads(path.read_text(encoding="utf-8")).get("games", [])
    out: list[tuple[str, float]] = []
    if not games:
        # 境界データがない動画でも deep 窓だけは測れる
        if "deep" in kinds:
            out.append(("deep", DEEP_WINDOW_SEC))
        return out
    mid_game = games[len(games) // 2]
    for kind in kinds:
        if kind == "match_start":
            out.append(("match_start", float(mid_game["start_sec"])))
        elif kind == "mid_match":
            out.append(
                ("mid_match", float(mid_game["start_sec"]) + MID_MATCH_OFFSET_SEC),
            )
        elif kind == "match_end":
            # 終了時刻の 5 秒前から (終了演出をまたぐ)
            out.append(("match_end", max(0.0, float(mid_game["end_sec"]) - 5.0)))
        elif kind == "deep":
            out.append(("deep", DEEP_WINDOW_SEC))
    return out


def _run(
    frames: list[np.ndarray], flag: str, value: bool,
) -> tuple[list[dict[str, Any]], float]:
    """指定フラグを切り替えてパイプラインを走らせる。

    パイプライン引数として渡せるフラグと、モジュール定数のフラグの両方に対応。
    """
    import src.background_fingerprint as bgfp
    from src.recognition_pipeline import RecognitionPipeline

    import src.next_detector as ndmod

    module_flags = {
        "ENABLE_DIRECT_PEARSON_NCC": bgfp,
        # next_detector の CNN バッチ化 (2026-07-31)。フラグではなく実装変更なので、
        # False 側は「detect_both を旧実装 (サイド別に単発 classify) に差し替える」
        # 形で比較する (下の _patch_next_detector_legacy)。
        "NEXT_CNN_BATCH": ndmod,
    }
    kwargs: dict[str, Any] = {}
    saved: tuple[Any, str, Any] | None = None
    legacy_saved: tuple[Any, str, Any] | None = None
    if flag == "NEXT_CNN_BATCH":
        # value=False のとき detect_both を旧実装 (サイドごとに単発 CNN) に戻す。
        # value=True は現行実装 (8枚を1バッチ) をそのまま使う。
        if not value:
            from src.next_detector import NextDetectionBothResult, NextDetector

            original_detect_both = NextDetector.detect_both

            def _legacy_detect_both(self: Any, frame: np.ndarray) -> Any:
                """旧実装: 1P/2P を別々に処理し CNN も 1 枚ずつ呼ぶ。"""
                self._check_resolution(frame)
                out = []
                for rois, side in ((self.ROIS_1P, "1P"), (self.ROIS_2P, "2P")):
                    prepared = self._prepare_side(frame, rois)
                    codes = [
                        self._classifier.classify(p[2]) for p in prepared
                    ]
                    out.append(self._vote_side(prepared, codes, side=side))
                return NextDetectionBothResult(p1=out[0], p2=out[1])

            NextDetector.detect_both = _legacy_detect_both
            legacy_saved = (NextDetector, "detect_both", original_detect_both)
    elif flag in module_flags:
        mod = module_flags[flag]
        saved = (mod, flag, getattr(mod, flag))
        setattr(mod, flag, value)
    else:
        kwargs[flag] = value
    try:
        pipe = RecognitionPipeline.load_default(**kwargs)
        snaps: list[dict[str, Any]] = []
        times: list[float] = []
        for idx, frame in enumerate(frames):
            t0 = time.perf_counter()
            res = pipe.update(idx, idx / 30.0, frame)
            times.append((time.perf_counter() - t0) * 1000.0)
            snaps.append(_snapshot(res))
    finally:
        if saved is not None:
            setattr(saved[0], saved[1], saved[2])
        if legacy_saved is not None:
            setattr(legacy_saved[0], legacy_saved[1], legacy_saved[2])
    arr = np.asarray(times)
    steady = arr[WARMUP_FRAMES:] if arr.size > WARMUP_FRAMES else arr
    return snaps, float(np.median(steady))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--flag", required=True, help="比較する bool フラグ名")
    ap.add_argument(
        "--videos", nargs="+",
        default=["video_c56", "video_c60", "video_c65", "video_c26", "video_c58"],
        help="既定は既存3本 + score OCR が破綻している既知の難物 c26/c58",
    )
    ap.add_argument("--video-dir", type=Path, default=Path("data/frames"))
    ap.add_argument(
        "--panel-dir", type=Path,
        default=Path("data/verify/winners_panel_diff_2026-07-26"),
    )
    ap.add_argument(
        "--windows", nargs="+",
        default=["match_start", "mid_match", "match_end", "deep"],
    )
    ap.add_argument("--frames", type=int, default=WINDOW_FRAMES)
    args = ap.parse_args()

    cv2.setNumThreads(1)
    print(f"比較フラグ: {args.flag} (False vs True)")
    print(f"動画: {', '.join(args.videos)}")
    print(f"局面: {', '.join(args.windows)}  各 {args.frames} フレーム\n")

    field_diffs: dict[str, int] = defaultdict(int)
    total_frames = 0
    n_windows = 0
    speed_gains: list[float] = []
    for name in args.videos:
        path = args.video_dir / f"{name}.mp4"
        if not path.exists():
            print(f"[skip] 動画不在: {name}")
            continue
        for label, start_sec in _build_windows(name, args.panel_dir, args.windows):
            frames = _read_frames(path, start_sec, args.frames)
            if len(frames) < 30:
                print(f"[skip] {name}/{label} t={start_sec:.0f}s: フレーム不足")
                continue
            off, ms_off = _run(frames, args.flag, False)
            on, ms_on = _run(frames, args.flag, True)
            n = min(len(off), len(on))
            win_diffs: dict[str, int] = defaultdict(int)
            for i in range(n):
                for key in off[i]:
                    if off[i][key] != on[i][key]:
                        win_diffs[key] += 1
                        field_diffs[key] += 1
            total_frames += n
            n_windows += 1
            if ms_off:
                speed_gains.append(100.0 * (ms_off - ms_on) / ms_off)
            status = (
                "一致" if not win_diffs
                else "差分: " + ", ".join(
                    f"{k}={v}" for k, v in sorted(
                        win_diffs.items(), key=lambda kv: -kv[1],
                    )[:4]
                )
            )
            print(
                f"{name:<12} {label:<12} t={start_sec:>7.0f}s {n:>4}f  "
                f"{ms_off:>6.1f}→{ms_on:>6.1f}ms  {status}"
            )

    print(f"\n=== 合計 ({n_windows} 窓 / {total_frames} フレーム) ===")
    if speed_gains:
        print(f"速度: 中央 {float(np.median(speed_gains)):+.1f}%")
    if not field_diffs:
        print("**全項目が完全一致。** 認識出力に一切の差がない。")
        return
    print("**差分あり。項目別:**")
    for key, cnt in sorted(field_diffs.items(), key=lambda kv: -kv[1]):
        print(
            f"  {key:<32}{cnt:>6} フレーム "
            f"({100.0 * cnt / max(1, total_frames):.2f}%)"
        )
    print(
        "\n※盤面が一致していてもネクスト等が違えば下流 (おじゃま会計・"
        "near_future 火力) に波及する。項目別に見ること。"
    )


if __name__ == "__main__":
    main()
