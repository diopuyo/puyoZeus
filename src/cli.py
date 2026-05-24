"""
CLI インターフェース

各機能をコマンドラインから呼び出すエントリポイント。

サブコマンド:
    analyze-frame IMAGE         画像 1 枚を解析して JSON 出力
    analyze-video INPUT         動画を盤面シーケンスに変換 (video_processor)
    composite    INPUT OUTPUT   動画にオーバーレイを合成
    stream                      配信オーバーレイサーバを起動
    timeline INPUT              動画を時系列解析して試合別タイムライン JSON を出力

Usage:
    python -m src.cli analyze-frame path/to/img.png
    python -m src.cli composite in.mp4 out.mp4 --interval 0.5
    python -m src.cli stream --port 8765
    python -m src.cli timeline data/frames/video_02.mp4 \
        --boundaries data/verify/match_boundaries_v4/video_02/matches.tsv \
        --out timeline.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import cv2

from src.analyzer import Analyzer
from src.overlay import OverlayRenderer
from src.stream_overlay import (
    DEFAULT_HOST,
    DEFAULT_PORT,
    StreamOverlayServer,
)
from src.video_compositer import CompositeOptions, VideoCompositor

# ============================
# 定数定義
# ============================

EXIT_OK: int = 0
EXIT_USAGE: int = 2
EXIT_ERROR: int = 1

DEFAULT_COMPOSITE_INTERVAL_SEC: float = 0.5

SUBCMD_ANALYZE_FRAME: str = "analyze-frame"
SUBCMD_ANALYZE_VIDEO: str = "analyze-video"
SUBCMD_COMPOSITE: str = "composite"
SUBCMD_STREAM: str = "stream"
SUBCMD_CALIBRATE: str = "calibrate"
SUBCMD_TIMELINE: str = "timeline"


# ============================
# エントリポイント
# ============================


def main(argv: Sequence[str] | None = None) -> int:
    """
    CLI エントリポイント。

    Args:
        argv: 引数リスト (None なら sys.argv[1:])。

    Returns:
        int: 終了コード (0=成功)。
    """
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return EXIT_USAGE

    handlers = {
        SUBCMD_ANALYZE_FRAME: _cmd_analyze_frame,
        SUBCMD_ANALYZE_VIDEO: _cmd_analyze_video,
        SUBCMD_COMPOSITE:     _cmd_composite,
        SUBCMD_STREAM:        _cmd_stream,
        SUBCMD_CALIBRATE:     _cmd_calibrate,
        SUBCMD_TIMELINE:      _cmd_timeline,
    }
    return handlers[args.command](args)


# ============================
# パーサ構築
# ============================


def _build_parser() -> argparse.ArgumentParser:
    """argparse パーサを構築する。"""
    parser = argparse.ArgumentParser(
        prog="puyo-analyzer",
        description="ぷよぷよeスポーツ 有利不利判定ツール",
    )
    sub = parser.add_subparsers(dest="command")

    p1 = sub.add_parser(
        SUBCMD_ANALYZE_FRAME,
        help="画像1枚を解析して JSON 出力",
    )
    p1.add_argument("image", help="入力画像パス")
    p1.add_argument(
        "--output", "-o", default=None,
        help="JSON 出力先 (省略時は標準出力)",
    )

    p2 = sub.add_parser(
        SUBCMD_ANALYZE_VIDEO,
        help="動画を盤面シーケンスに変換",
    )
    p2.add_argument("input", help="入力動画パス")
    p2.add_argument(
        "--output", "-o", default=None,
        help="JSON 出力先",
    )
    p2.add_argument(
        "--interval", type=float, default=1.0,
        help="フレーム抽出間隔 (秒)",
    )

    p3 = sub.add_parser(
        SUBCMD_COMPOSITE,
        help="動画にオーバーレイを合成",
    )
    p3.add_argument("input", help="入力動画パス")
    p3.add_argument("output", help="出力動画パス")
    p3.add_argument(
        "--interval", type=float, default=DEFAULT_COMPOSITE_INTERVAL_SEC,
        help="解析サンプル間隔 (秒)",
    )
    p3.add_argument(
        "--no-audio", action="store_true",
        help="音声結合をスキップする",
    )

    p4 = sub.add_parser(
        SUBCMD_STREAM,
        help="配信オーバーレイサーバを起動",
    )
    p4.add_argument("--host", default=DEFAULT_HOST, help="バインドアドレス")
    p4.add_argument("--port", type=int, default=DEFAULT_PORT, help="ポート")

    p5 = sub.add_parser(
        SUBCMD_CALIBRATE,
        help="参照フレームから盤面座標・HSV閾値を抽出",
    )
    p5.add_argument("frame", help="参照フレーム画像パス")
    p5.add_argument("annotation", help="アノテーション JSON パス")
    p5.add_argument(
        "--output", "-o", default="models/calibration.json",
        help="キャリブレーション結果 JSON 出力先",
    )

    p6 = sub.add_parser(
        SUBCMD_TIMELINE,
        help="動画を時系列解析して試合ごとの ScorePoint/ChainEvent を JSON 出力",
    )
    p6.add_argument("input", help="入力動画パス")
    p6.add_argument(
        "--boundaries", default=None,
        help="試合区間 TSV (matches.tsv 互換)。未指定なら自動検出。",
    )
    p6.add_argument(
        "--out", "-o", default=None,
        help="JSON 出力先 (省略時は標準出力)",
    )
    p6.add_argument(
        "--cnn", default=None,
        help="NextDetector 用 CNN 重みパス (任意)",
    )
    p6.add_argument(
        "--calib", default=None,
        help="ImageReader/MatchStateDetector のキャリブレーション JSON (任意)",
    )

    return parser


# ============================
# サブコマンド実装
# ============================


def _cmd_analyze_frame(args: argparse.Namespace) -> int:
    """単一フレーム画像を解析する。"""
    image_path = Path(args.image)
    if not image_path.exists():
        print(f"[ERROR] 画像が存在しません: {image_path}", file=sys.stderr)
        return EXIT_ERROR

    frame = cv2.imread(str(image_path))
    if frame is None:
        print(f"[ERROR] 画像を読み込めません: {image_path}", file=sys.stderr)
        return EXIT_ERROR

    analyzer = Analyzer()
    result = analyzer.analyze_frame(frame)
    _write_json(result.to_dict(), args.output)
    return EXIT_OK


def _cmd_analyze_video(args: argparse.Namespace) -> int:
    """動画を盤面シーケンスに変換する (video_processor 連携)。"""
    # 循環 import 回避のため遅延 import
    from src.video_processor import VideoProcessor

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"[ERROR] 動画が存在しません: {input_path}", file=sys.stderr)
        return EXIT_ERROR

    processor = VideoProcessor(frame_interval_sec=args.interval)
    analysis = processor.process_video_file(input_path)

    output = args.output or f"data/boards/{input_path.stem}.json"
    processor.save_analysis(analysis, Path(output))
    print(f"[OK] 出力: {output}")
    return EXIT_OK


def _cmd_composite(args: argparse.Namespace) -> int:
    """動画にオーバーレイを合成する。"""
    comp = VideoCompositor(
        analyzer=Analyzer(),
        renderer=OverlayRenderer(),
    )
    opts = CompositeOptions(
        sampling_interval_sec=args.interval,
        mux_audio=not args.no_audio,
        progress_callback=_make_progress_callback(),
    )
    try:
        result = comp.composite(
            input_path=args.input,
            output_path=args.output,
            options=opts,
        )
    except FileNotFoundError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return EXIT_ERROR

    print()  # 進捗プログレスの改行
    print(
        f"[OK] {result.output_path} "
        f"({result.total_frames}フレーム / 解析 {result.analyzed_frames}回)"
    )
    return EXIT_OK


def _cmd_calibrate(args: argparse.Namespace) -> int:
    """参照フレームと annotation からキャリブレーション設定を生成する。"""
    from src.calibration import CalibrationAnnotation, CalibrationHelper

    frame_path = Path(args.frame)
    ann_path = Path(args.annotation)
    if not frame_path.exists():
        print(f"[ERROR] フレームが存在しません: {frame_path}", file=sys.stderr)
        return EXIT_ERROR
    if not ann_path.exists():
        print(f"[ERROR] annotation が存在しません: {ann_path}", file=sys.stderr)
        return EXIT_ERROR

    frame = cv2.imread(str(frame_path))
    if frame is None:
        print(f"[ERROR] 画像を読み込めません: {frame_path}", file=sys.stderr)
        return EXIT_ERROR

    ann = CalibrationAnnotation.from_json(ann_path)
    config = CalibrationHelper().calibrate_from_reference(frame, ann)
    config.save(Path(args.output))
    print(f"[OK] キャリブレーション結果: {args.output}")
    print(
        f"     1P: x={config.p1_region.x} y={config.p1_region.y} "
        f"{config.p1_region.width}x{config.p1_region.height}"
    )
    print(f"     色範囲: {sorted(config.color_ranges.keys())}")
    return EXIT_OK


def _cmd_timeline(args: argparse.Namespace) -> int:
    """動画を時系列解析する (timeline_analyzer 連携)。"""
    from src.timeline_analyzer import TimelineAnalyzer, to_json

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"[ERROR] 動画が存在しません: {input_path}", file=sys.stderr)
        return EXIT_ERROR

    cnn = Path(args.cnn) if args.cnn else None
    calib = Path(args.calib) if args.calib else None
    boundaries = Path(args.boundaries) if args.boundaries else None

    analyzer = TimelineAnalyzer(cnn_path=cnn, calib_path=calib)
    result = analyzer.analyze_video(input_path, boundaries_tsv=boundaries)

    if args.out is None:
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    else:
        out_path = Path(args.out)
        to_json(result, out_path)
        print(
            f"[OK] 出力: {out_path} "
            f"(試合数={len(result.match_segments)})"
        )
    return EXIT_OK


def _cmd_stream(args: argparse.Namespace) -> int:
    """配信オーバーレイサーバを起動し、Ctrl-C まで待機する。"""
    srv = StreamOverlayServer(host=args.host, port=args.port)
    srv.start()
    host, port = srv.address()
    print(f"[stream] http://{host}:{port}/ で配信オーバーレイを配信中")
    print("[stream] Ctrl-C で停止")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        srv.stop()
    print("\n[stream] 停止しました")
    return EXIT_OK


# ============================
# ユーティリティ
# ============================


def _write_json(payload: dict[str, Any], output: str | None) -> None:
    """JSON を指定先 (or 標準出力) に書き出す。"""
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if output is None:
        print(text)
        return
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    print(f"[OK] 出力: {path}")


def _make_progress_callback():
    """進捗表示用コールバックを返す。"""
    last = [0.0]

    def cb(payload: dict[str, Any]) -> None:
        now = time.time()
        if now - last[0] < 0.5:
            return
        last[0] = now
        cur = payload.get("current_frame", 0)
        total = payload.get("total_frames", 0) or 0
        if total > 0:
            pct = cur / total * 100
            sys.stdout.write(
                f"\r[progress] {cur}/{total} ({pct:5.1f}%)"
            )
        else:
            sys.stdout.write(f"\r[progress] {cur} frames")
        sys.stdout.flush()

    return cb


# ============================
# スクリプトエントリ
# ============================


if __name__ == "__main__":
    sys.exit(main())
