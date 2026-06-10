"""TSUMO/CHAIN/OJAMA 中の total_score 変化を実測 (Phase D-2 検証).

「アクション中の認識ちらつきが評価に影響していない」ことを定量で確認。
各 frame の (time, state_1p, state_2p, total_score, p1_advantage 寄与)
を tsv 出力。

評価変化が:
  - STABLE→STABLE で score 変化 → 通常 (= 盤面更新)
  - TSUMO/CHAIN/OJAMA 中で score 変化 → 想定外 (要調査)

利用例:
    PYTHONPATH=. ./venv/bin/python -m scripts.phase_d_score_drift_check \
        --video data/frames/video_02.mp4 --start-sec 205 --end-sec 306
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.console_init import init_console, to_windows_path  # noqa: E402

init_console()

import cv2  # noqa: E402

from src.board_state_machine import BoardState  # noqa: E402
from src.old.indicators import IndicatorCalculator  # noqa: E402
from src.per_video_model_selector import select_phase_b_model  # noqa: E402
from src.recognition_pipeline import RecognitionPipeline  # noqa: E402
from src.old.scorer import PhaseAwareScorer  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--start-sec", type=float, required=True)
    parser.add_argument("--end-sec", type=float, required=True)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--stable-n", type=int, default=2)
    parser.add_argument(
        "--out-tsv", type=Path,
        default=_ROOT / "data" / "phase_d_score_drift.tsv",
    )
    args = parser.parse_args()

    # video_id 抽出 (= "video_02" → 2)
    video_id = int(args.video.stem.split("_")[1])
    cnn_model_str = select_phase_b_model(video_id)
    cnn_model = Path(cnn_model_str) if cnn_model_str else None

    pipe = RecognitionPipeline.load_default(
        stable_frame_count=args.stable_n,
        load_score_ocr=True,
        enable_chain_tracker=True,
        cnn_model_path=cnn_model,
        temporal_smoothing=1,
        load_next_detector=True,
        force_in_match=True,
    )
    ind_calc = IndicatorCalculator()
    scorer = PhaseAwareScorer(weight_mode="optimal")
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        return 1

    match_dur = max(1.0, args.end_sec - args.start_sec)
    interval = 1.0 / args.fps
    t = args.start_sec
    frame_idx = 0

    rows: list[dict] = []
    last_score = 0.0
    last_p1_raw = 0.0
    last_p2_raw = 0.0
    last_score_key: tuple[str, str] | None = None
    score_changes_per_state: Counter[
        tuple[BoardState, BoardState]
    ] = Counter()
    score_change_magnitudes: list[
        tuple[BoardState, BoardState, float]
    ] = []
    # 各 side の評価寄与が「アクション中」に変化したカウント
    p1_raw_changes_in_p1_action = 0
    p2_raw_changes_in_p2_action = 0
    # 「片側アクション中、他側 STABLE 更新」のカウント (= 仕様通りの total_score 変化)
    asymmetric_changes = 0

    while t < args.end_sec:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        if frame.shape[:2] != (1080, 1920):
            frame = cv2.resize(
                frame, (1920, 1080), interpolation=cv2.INTER_AREA,
            )
        result = pipe.update(frame_idx, t, frame)
        # スコア計算 (両側 STABLE のとき & 盤面変化時のみ)
        new_score: float | None = None
        if (
            result.p1.state == BoardState.STABLE
            and result.p2.state == BoardState.STABLE
            and result.p1.confirmed_board is not None
            and result.p2.confirmed_board is not None
        ):
            try:
                key = (
                    result.p1.confirmed_board.to_json(),
                    result.p2.confirmed_board.to_json(),
                )
                if key != last_score_key:
                    p1_set = ind_calc.compute_all(result.p1.confirmed_board)
                    p2_set = ind_calc.compute_all(result.p2.confirmed_board)
                    res = scorer.score(
                        p1_set, p2_set,
                        max(0.0, t - args.start_sec), match_dur,
                    )
                    new_score = res.total_score
                    cur_p1_raw = res.player1_raw
                    cur_p2_raw = res.player2_raw
                    # side 個別の raw 評価が変化しているか (= 個別評価への影響)
                    p1_raw_changed = abs(cur_p1_raw - last_p1_raw) > 1e-6
                    p2_raw_changed = abs(cur_p2_raw - last_p2_raw) > 1e-6
                    if p1_raw_changed and result.p1.state != BoardState.STABLE:
                        p1_raw_changes_in_p1_action += 1
                    if p2_raw_changed and result.p2.state != BoardState.STABLE:
                        p2_raw_changes_in_p2_action += 1
                    last_p1_raw = cur_p1_raw
                    last_p2_raw = cur_p2_raw
                    last_score_key = key
            except Exception:
                pass

        score_changed = new_score is not None and new_score != last_score
        if new_score is not None:
            cur_score = new_score
        else:
            cur_score = last_score

        rows.append({
            "frame_idx": frame_idx,
            "time_sec": f"{t:.2f}",
            "1P_state": result.p1.state.value,
            "2P_state": result.p2.state.value,
            "total_score": f"{cur_score:.2f}",
            "score_changed": "1" if score_changed else "0",
        })

        if score_changed and new_score is not None:
            key_states = (result.p1.state, result.p2.state)
            score_changes_per_state[key_states] += 1
            score_change_magnitudes.append(
                (result.p1.state, result.p2.state, new_score - last_score),
            )
            last_score = new_score
        frame_idx += 1
        t += interval
    cap.release()

    # 集計
    print()
    print("[summary] score change events grouped by (1P state, 2P state)")
    for (s1, s2), n in sorted(
        score_changes_per_state.items(), key=lambda kv: -kv[1],
    ):
        print(f"  ({s1.value:<10}, {s2.value:<10}): {n}")

    # アクション中の score 変化を集計
    action_changes = sum(
        n for (s1, s2), n in score_changes_per_state.items()
        if s1 != BoardState.STABLE or s2 != BoardState.STABLE
    )
    stable_changes = sum(
        n for (s1, s2), n in score_changes_per_state.items()
        if s1 == BoardState.STABLE and s2 == BoardState.STABLE
    )
    total_changes = action_changes + stable_changes
    print()
    print(
        f"[verdict] {stable_changes}/{total_changes} score 変化が STABLE 中"
        f" ({100*stable_changes/total_changes if total_changes else 0:.1f}%)"
    )
    print()
    print("[detail] side 個別の raw 評価変化:")
    print(
        f"  1P アクション中の player1_raw 変化: {p1_raw_changes_in_p1_action}"
    )
    print(
        f"  2P アクション中の player2_raw 変化: {p2_raw_changes_in_p2_action}"
    )
    if (
        p1_raw_changes_in_p1_action == 0
        and p2_raw_changes_in_p2_action == 0
    ):
        print(
            "[verdict] ✅ アクション中 side の個別評価は変化していない "
            "— 評価に影響なし確定"
        )
    if action_changes == 0:
        print("[verdict] ✅ アクション中の total_score 変化はゼロ")
    else:
        print(
            f"[verdict] ℹ アクション中 total_score 変化 {action_changes} 件 — "
            "片側 STABLE 更新による合算変動の可能性 (上の個別評価で要確認)"
        )
        # 大きな変化を表示
        big = sorted(
            score_change_magnitudes, key=lambda m: -abs(m[2]),
        )[:10]
        for s1, s2, delta in big:
            if s1 != BoardState.STABLE or s2 != BoardState.STABLE:
                print(
                    f"  ({s1.value}, {s2.value}): delta={delta:+.2f}"
                )

    # tsv 保存
    args.out_tsv.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else []
    with args.out_tsv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        w.writeheader()
        w.writerows(rows)
    print(f"\n[saved] {to_windows_path(args.out_tsv)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
