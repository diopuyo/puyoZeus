"""
動画を走査して match_state の遷移点（開始・終了判定点）を検出し、
未加工フレームを両側で保存する。人の目で判定の妥当性を確認するため。

使い方:
    ./venv/bin/python scripts/validate_match_boundaries.py \\
        --video data/frames/video_02.mp4 --interval 2

処理:
    1. `--interval` 秒ごとにフレーム抽出
    2. 各フレームの MatchStateDetector.detect() を実行
    3. 状態遷移のたびに「直前のフレーム」と「直後のフレーム」の
       生画像を data/verify/match_boundaries/<stem>/ 配下に保存
    4. tsv サマリも同ディレクトリに出力

出力例:
    data/verify/match_boundaries/video_02/start_00210s_before.png
    data/verify/match_boundaries/video_02/start_00210s_after.png
    data/verify/match_boundaries/video_02/end_00460s_before.png
    data/verify/match_boundaries/video_02/end_00460s_after.png
    data/verify/match_boundaries/video_02/summary.tsv

TSV 列:
    t_sec  state  bg_value  transition
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import cv2
import numpy as np

from src.match_state import MatchState, MatchStateDetector


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--interval", type=float, default=2.0,
                        help="サンプリング間隔（秒）")
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--end", type=float, default=None,
                        help="終了秒（省略で動画末尾まで）")
    parser.add_argument("--enter-v", type=float, default=130.0,
                        help="in_match 遷移 bg_value 上限（ヒステリシス下）")
    parser.add_argument("--exit-v", type=float, default=170.0,
                        help="not_in_match 遷移 bg_value 下限（ヒステリシス上）")
    parser.add_argument("--confirm", type=int, default=2,
                        help="遷移確定に必要な連続一致サンプル数")
    parser.add_argument("--out-root", default="data/verify/match_boundaries")
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"動画が存在しません: {video_path}", file=sys.stderr)
        return 1

    detector = MatchStateDetector.load_default()

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    end_t = args.end if args.end is not None else duration
    print(f"動画: {video_path.name}  duration={duration:.0f}s  interval={args.interval}s")

    out_dir = Path(args.out_root) / video_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[tuple[float, str, float]] = []  # (t, confirmed_state, bg_value)
    # 2 段閾値 + 連続一致確認でチャタリング抑制
    confirmed_state = MatchState.NOT_IN_MATCH
    # 候補状態と連続回数
    pending_state: MatchState | None = None
    pending_count = 0
    prev_frame: np.ndarray | None = None
    transition_frames: dict[float, np.ndarray] = {}  # t -> frame (保存候補用)

    t = args.start
    transitions: list[tuple[str, float, np.ndarray | None, np.ndarray]] = []
    while t < end_t:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok, frame = cap.read()
        if not ok or frame is None:
            t += args.interval
            continue
        if frame.shape[:2] != (1080, 1920):
            frame = cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_AREA)
        bg = detector.bg_value(frame)

        # ヒステリシス: 現在の確定状態に応じて閾値を切り替え
        if confirmed_state == MatchState.NOT_IN_MATCH:
            raw_state = MatchState.IN_MATCH if bg < args.enter_v else MatchState.NOT_IN_MATCH
        else:
            raw_state = MatchState.NOT_IN_MATCH if bg > args.exit_v else MatchState.IN_MATCH

        if raw_state == confirmed_state:
            pending_state = None
            pending_count = 0
        else:
            if pending_state == raw_state:
                pending_count += 1
            else:
                pending_state = raw_state
                pending_count = 1
            if pending_count >= args.confirm:
                # 遷移確定
                old = confirmed_state
                confirmed_state = raw_state
                pending_state = None
                pending_count = 0
                if old == MatchState.NOT_IN_MATCH and confirmed_state == MatchState.IN_MATCH:
                    kind = "start"
                elif old == MatchState.IN_MATCH and confirmed_state == MatchState.NOT_IN_MATCH:
                    kind = "end"
                else:
                    kind = "other"
                transitions.append((kind, t, prev_frame, frame.copy()))
                print(f"  [{kind}] t={t:.1f}s  {old.value} -> {confirmed_state.value}  bg={bg:.1f}")

        rows.append((t, confirmed_state.value, bg))
        prev_frame = frame.copy()
        t += args.interval

    cap.release()

    # 遷移点の画像を保存
    for kind, t, before, after in transitions:
        tag = f"{kind}_{int(t):05d}s"
        if before is not None:
            cv2.imwrite(str(out_dir / f"{tag}_before.png"), before)
        cv2.imwrite(str(out_dir / f"{tag}_after.png"), after)

    # サマリ TSV
    tsv_path = out_dir / "summary.tsv"
    with tsv_path.open("w", encoding="utf-8") as f:
        f.write("t_sec\tstate\tbg_value\ttransition\n")
        prev_s: str | None = None
        for t_sec, state, bg in rows:
            trans = ""
            if prev_s is not None and state != prev_s:
                trans = f"{prev_s}->{state}"
            f.write(f"{t_sec:.1f}\t{state}\t{bg:.1f}\t{trans}\n")
            prev_s = state

    print(f"\n遷移検出: {len(transitions)} 件")
    print(f"出力: {out_dir}/")
    print(f"サマリ: {tsv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
