"""不具合A 修正スモークテスト: v89_match01 t=35-41 の CHAIN 状態持続を比較する。

フラグ OFF (現挙動) と ON (クールダウン有効) で
1P CHAIN 状態の on/off 遷移を t=33-43 区間で出力し、
ON 時に t=38.77 タイムアウト後の CHAIN 再延長が起きないことを確認する。

使用方法:
    PYTHONPATH=. python scripts/smoke_chain_refire_cooldown.py
"""

from __future__ import annotations

import cv2
from pathlib import Path

from src.board_state_machine import BoardState
from src.recognition_pipeline import RecognitionPipeline

VIDEO_PATH = Path("data/match_clips/v89/v89_match01.mp4")
T_START = 33.0   # 連鎖が始まる付近
T_END = 43.0     # 次連鎖開始まで余裕をもたせた終端


def run_smoke(enable_cooldown: bool) -> list[tuple[float, str]]:
    """指定フラグで pipeline を動かし、1P CHAIN 遷移ログを返す。

    Returns:
        (time_sec, chain_state_str) のリスト。
        chain_state_str: "CHAIN_ON" or "CHAIN_OFF"
    """
    pipe = RecognitionPipeline.load_default(
        force_in_match=True,
        enable_chain_formula_detection=True,
        enable_game_event_chain_exit=True,
        enable_chain_refire_cooldown=enable_cooldown,
    )

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    log: list[tuple[float, str]] = []
    prev_chain_on = False

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        time_sec = frame_idx / fps
        if time_sec < T_START:
            frame_idx += 1
            continue
        if time_sec > T_END:
            break

        res = pipe.update(frame_idx, time_sec, frame)
        # _active_chain_1p が None でなければ CHAIN 状態とみなす
        chain_on = pipe._active_chain_1p is not None

        # 遷移時のみ記録 (CHAIN ON→OFF, OFF→ON)
        if chain_on != prev_chain_on:
            state_str = "CHAIN_ON" if chain_on else "CHAIN_OFF"
            log.append((round(time_sec, 3), state_str))
            prev_chain_on = chain_on

        frame_idx += 1

    cap.release()
    return log


def main() -> None:
    if not VIDEO_PATH.exists():
        print(f"[smoke] 動画が見つかりません: {VIDEO_PATH}")
        return

    print(f"[smoke] 動画: {VIDEO_PATH}")
    print(f"[smoke] 区間: t={T_START}s - {T_END}s")
    print()

    log_off = run_smoke(enable_cooldown=False)
    log_on = run_smoke(enable_cooldown=True)

    print("=== フラグ OFF (現挙動) ===")
    for t, s in log_off:
        print(f"  t={t:.3f}s  {s}")

    print()
    print("=== フラグ ON (クールダウン有効) ===")
    for t, s in log_on:
        print(f"  t={t:.3f}s  {s}")

    print()
    # CHAIN_ON 期間を計算
    def calc_chain_duration(log: list[tuple[float, str]]) -> float:
        total = 0.0
        start = None
        for t, s in log:
            if s == "CHAIN_ON":
                start = t
            elif s == "CHAIN_OFF" and start is not None:
                total += t - start
                start = None
        return total

    dur_off = calc_chain_duration(log_off)
    dur_on = calc_chain_duration(log_on)
    print(f"CHAIN 合計時間 OFF={dur_off:.2f}s  ON={dur_on:.2f}s")
    if dur_on < dur_off:
        print("[smoke] OK: フラグ ON で CHAIN 持続時間が短縮された (不具合A 修正)")
    else:
        print("[smoke] INFO: CHAIN 持続時間に変化なし (区間に再発火がなかった可能性あり)")


if __name__ == "__main__":
    main()
