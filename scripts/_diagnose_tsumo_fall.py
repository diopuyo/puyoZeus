"""STEP1 診断: TSUMO_FALL 状態出現頻度と drain 信号の根因調査。

調査内容:
  1. video_124_4min (最大 180s) で各 BoardState の出現フレーム数 (1P/2P別)
  2. TSUMO_FALL -> STABLE 遷移回数 (= drain が発火すべき回数)
  3. tsumo_count の増分タイミングと信号源
  4. drain が 0 になる根因の特定

使い方:
    python -m scripts._diagnose_tsumo_fall \
        --video data/frames/video_124_4min.mp4 \
        --max-sec 180
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.board_state_machine import BoardState  # noqa: E402
from src.recognition_pipeline import RecognitionPipeline  # noqa: E402

TARGET_W: int = 1920
TARGET_H: int = 1080
DEFAULT_FPS: float = 30.0


def diagnose(video_path: Path, max_sec: float = 180.0) -> None:
    """state 遷移を集計して診断レポートを出力する。"""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[ERROR] cannot open: {video_path}", file=sys.stderr)
        return
    fps = cap.get(cv2.CAP_PROP_FPS) or DEFAULT_FPS
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_sec > 0:
        n_frames = min(n_frames, int(max_sec * fps))

    pipeline = RecognitionPipeline.load_default(
        stable_frame_count=3,
        load_score_ocr=True,
        enable_chain_tracker=True,
        temporal_smoothing=1,
        load_next_detector=True,
        force_in_match=True,
    )

    # 集計
    state_count_1p: Counter[str] = Counter()
    state_count_2p: Counter[str] = Counter()
    # TSUMO_FALL -> STABLE 遷移カウント
    tsumo_to_stable_1p: int = 0
    tsumo_to_stable_2p: int = 0
    # 前フレームの state
    prev_state_1p = BoardState.MENU
    prev_state_2p = BoardState.MENU
    # tsumo_count の時刻別スナップ (0.5秒毎)
    tsumo_snapshots: list[dict] = []
    prev_tsumo_1p: int = 0
    prev_tsumo_2p: int = 0
    # TSUMO_FALL の連続フレーム追跡
    tsumo_fall_start_1p: int | None = None
    tsumo_fall_start_2p: int | None = None
    tsumo_fall_durations_1p: list[int] = []
    tsumo_fall_durations_2p: list[int] = []
    # 全遷移を記録 (最初の 200 遷移のみ詳細)
    transition_log_1p: list[tuple[float, str, str]] = []  # (t_sec, prev, curr)
    transition_log_2p: list[tuple[float, str, str]] = []
    MAX_TRANSITION_LOG = 200

    for fi in range(n_frames):
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        if frame.shape[:2] != (TARGET_H, TARGET_W):
            frame = cv2.resize(frame, (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA)
        t_sec = fi / fps
        result = pipeline.update(fi, t_sec, frame)

        curr_1p = result.p1.state
        curr_2p = result.p2.state

        # state カウント
        state_count_1p[curr_1p.value] += 1
        state_count_2p[curr_2p.value] += 1

        # 遷移記録
        if curr_1p != prev_state_1p:
            if len(transition_log_1p) < MAX_TRANSITION_LOG:
                transition_log_1p.append((t_sec, prev_state_1p.value, curr_1p.value))
            # TSUMO_FALL -> STABLE カウント
            if prev_state_1p == BoardState.TSUMO_FALL and curr_1p == BoardState.STABLE:
                tsumo_to_stable_1p += 1
            # TSUMO_FALL 継続フレーム数計測
            if prev_state_1p != BoardState.TSUMO_FALL and curr_1p == BoardState.TSUMO_FALL:
                tsumo_fall_start_1p = fi
            elif prev_state_1p == BoardState.TSUMO_FALL and curr_1p != BoardState.TSUMO_FALL:
                if tsumo_fall_start_1p is not None:
                    tsumo_fall_durations_1p.append(fi - tsumo_fall_start_1p)
                tsumo_fall_start_1p = None

        if curr_2p != prev_state_2p:
            if len(transition_log_2p) < MAX_TRANSITION_LOG:
                transition_log_2p.append((t_sec, prev_state_2p.value, curr_2p.value))
            if prev_state_2p == BoardState.TSUMO_FALL and curr_2p == BoardState.STABLE:
                tsumo_to_stable_2p += 1
            if prev_state_2p != BoardState.TSUMO_FALL and curr_2p == BoardState.TSUMO_FALL:
                tsumo_fall_start_2p = fi
            elif prev_state_2p == BoardState.TSUMO_FALL and curr_2p != BoardState.TSUMO_FALL:
                if tsumo_fall_start_2p is not None:
                    tsumo_fall_durations_2p.append(fi - tsumo_fall_start_2p)
                tsumo_fall_start_2p = None

        prev_state_1p = curr_1p
        prev_state_2p = curr_2p

        # tsumo_count スナップ (0.5秒毎)
        if fi % int(fps * 0.5) == 0:
            tc1 = pipeline.tsumo_count("1P")
            tc2 = pipeline.tsumo_count("2P")
            delta1 = tc1 - prev_tsumo_1p
            delta2 = tc2 - prev_tsumo_2p
            tsumo_snapshots.append({
                "t_sec": round(t_sec, 1),
                "fi": fi,
                "tsumo_1p": tc1,
                "tsumo_2p": tc2,
                "delta_1p": delta1,
                "delta_2p": delta2,
                "state_1p": curr_1p.value,
                "state_2p": curr_2p.value,
            })
            prev_tsumo_1p = tc1
            prev_tsumo_2p = tc2

    cap.release()

    total_frames = n_frames
    print("=" * 70)
    print(f"[STEP1] 診断: {video_path.name}  max_sec={max_sec}  total_frames={total_frames}")
    print("=" * 70)

    # --- state 出現フレーム数 ---
    print("\n[1] BoardState 出現フレーム数")
    all_states = sorted(
        set(state_count_1p.keys()) | set(state_count_2p.keys())
    )
    print(f"  {'state':<20} {'1P frames':>12} {'1P %':>8} {'2P frames':>12} {'2P %':>8}")
    for s in all_states:
        c1 = state_count_1p[s]
        c2 = state_count_2p[s]
        p1 = 100.0 * c1 / total_frames if total_frames > 0 else 0.0
        p2 = 100.0 * c2 / total_frames if total_frames > 0 else 0.0
        print(f"  {s:<20} {c1:>12,} {p1:>8.1f}% {c2:>12,} {p2:>8.1f}%")

    # --- TSUMO_FALL -> STABLE 遷移 ---
    print(f"\n[2] TSUMO_FALL -> STABLE 遷移回数")
    print(f"  1P: {tsumo_to_stable_1p} 回")
    print(f"  2P: {tsumo_to_stable_2p} 回")
    print(f"  ※ drain 発火すべき回数 = 上記回数")

    # --- TSUMO_FALL 継続フレーム統計 ---
    print(f"\n[3] TSUMO_FALL 継続フレーム数統計 (@{fps:.0f}fps)")
    for label, durations in [("1P", tsumo_fall_durations_1p), ("2P", tsumo_fall_durations_2p)]:
        if durations:
            avg = sum(durations) / len(durations)
            mn = min(durations)
            mx = max(durations)
            print(f"  {label}: count={len(durations)} avg={avg:.1f}f ({avg/fps*1000:.0f}ms) "
                  f"min={mn}f max={mx}f")
        else:
            print(f"  {label}: TSUMO_FALL 継続データなし (0 件)")

    # --- tsumo_count スナップ (最初の 60 秒分) ---
    print(f"\n[4] tsumo_count 増分スナップ (0.5秒毎、~60s)")
    print(f"  {'t_sec':>8} {'fi':>6} {'tc1':>6} {'d1':>5} {'tc2':>6} {'d2':>5}  st1/st2")
    for snap in tsumo_snapshots:
        if snap["t_sec"] > 60.0:
            break
        d1_mark = "▲" if snap["delta_1p"] > 0 else " "
        d2_mark = "▲" if snap["delta_2p"] > 0 else " "
        print(f"  {snap['t_sec']:>8.1f} {snap['fi']:>6} "
              f"{snap['tsumo_1p']:>6}{d1_mark}{snap['delta_1p']:>4} "
              f"{snap['tsumo_2p']:>6}{d2_mark}{snap['delta_2p']:>4}  "
              f"{snap['state_1p'][:8]}/{snap['state_2p'][:8]}")

    # --- 遷移ログ詳細 (最初の 40 遷移) ---
    print(f"\n[5] 1P 状態遷移ログ (最初の 40 件)")
    for i, (t, p, c) in enumerate(transition_log_1p[:40]):
        tsumo_mark = " *** TSUMO->STABLE ***" if (p == "tsumo_fall" and c == "stable") else ""
        print(f"  {i+1:>3}. t={t:>7.2f}s  {p:<20} -> {c:<20}{tsumo_mark}")

    print(f"\n[6] 2P 状態遷移ログ (最初の 40 件)")
    for i, (t, p, c) in enumerate(transition_log_2p[:40]):
        tsumo_mark = " *** TSUMO->STABLE ***" if (p == "tsumo_fall" and c == "stable") else ""
        print(f"  {i+1:>3}. t={t:>7.2f}s  {p:<20} -> {c:<20}{tsumo_mark}")

    # --- 根因サマリ ---
    print("\n[7] 根因サマリ")
    tsumo_1p_frames = state_count_1p.get("tsumo_fall", 0)
    tsumo_2p_frames = state_count_2p.get("tsumo_fall", 0)
    if tsumo_1p_frames == 0 and tsumo_2p_frames == 0:
        print("  *** 根因 A: TSUMO_FALL state が一度も出現しない ***")
        print("  -> TsumoPhaseDetector が TSUMO_FALL を判定していない")
        print("  -> drain の駆動源 (TSUMO_FALL->STABLE) が絶対に発火しない")
    elif tsumo_to_stable_1p == 0 and tsumo_to_stable_2p == 0:
        print("  *** 根因 B: TSUMO_FALL は出現するが TSUMO_FALL->STABLE 遷移が0 ***")
        print(f"  1P TSUMO_FALL frames={tsumo_1p_frames}, 2P TSUMO_FALL frames={tsumo_2p_frames}")
        print("  -> TSUMO_FALL から STABLE への遷移が別経路 (CHAIN等) に分岐している")
    else:
        print(f"  1P: TSUMO_FALL->STABLE {tsumo_to_stable_1p} 回 / TSUMO_FALL frames={tsumo_1p_frames}")
        print(f"  2P: TSUMO_FALL->STABLE {tsumo_to_stable_2p} 回 / TSUMO_FALL frames={tsumo_2p_frames}")
        tc1_final = tsumo_snapshots[-1]["tsumo_1p"] if tsumo_snapshots else 0
        tc2_final = tsumo_snapshots[-1]["tsumo_2p"] if tsumo_snapshots else 0
        print(f"  最終 tsumo_count: 1P={tc1_final} 2P={tc2_final}")
        ratio_1p = tsumo_to_stable_1p / max(tc1_final, 1) * 100
        ratio_2p = tsumo_to_stable_2p / max(tc2_final, 1) * 100
        print(f"  TSUMO->STABLE / tsumo_count 比: 1P={ratio_1p:.0f}% 2P={ratio_2p:.0f}%")
        if ratio_1p < 80 or ratio_2p < 80:
            print("  *** 根因 C: TSUMO_FALL->STABLE 遷移数が tsumo_count より大幅に少ない ***")
            print("  -> tsumo_count は NEXT 変化ベースで増えるが、")
            print("     TSUMO_FALL state は多くのツモで発生していない")


def main() -> int:
    parser = argparse.ArgumentParser(description="TSUMO_FALL 診断スクリプト")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--max-sec", type=float, default=180.0)
    args = parser.parse_args()
    diagnose(args.video, args.max_sec)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
