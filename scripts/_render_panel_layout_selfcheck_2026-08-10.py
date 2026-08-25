"""パネルレイアウト(字幕余白入り)の自己検収用静止画1枚を高速生成する。

user レビュー用ではなく自分の確認用 (レイアウト崩れがないかの目視自己検収)。
認識パイプライン・モデル学習を一切通さず、_draw_panel_layout を合成データで
直接叩くことで数秒で1枚を書き出す (実データでの60秒サンプルは不要という
2026-08-10 user変更指示に対応)。
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import scripts.visualize_advantage_overlay as vao  # noqa: E402


def _fake_frame() -> np.ndarray:
    """OUT_W x OUT_H のダミー映像フレーム(市松模様、映像領域の伸縮確認用)。"""
    frame = np.zeros((vao.OUT_H, vao.OUT_W, 3), dtype=np.uint8)
    frame[:, :] = (40, 90, 40)
    cv2.rectangle(frame, (0, 0), (vao.OUT_W - 1, vao.OUT_H - 1), (255, 255, 255), 6)
    cv2.putText(frame, "DUMMY GAME FRAME", (60, vao.OUT_H // 2), cv2.FONT_HERSHEY_SIMPLEX,
                1.6, (255, 255, 255), 3)
    return frame


def main() -> None:
    """合成データでパネルレイアウトを1枚描画し PNG に書き出す。"""
    history = [(t, 60.0 * np.sin(t / 5.0)) for t in np.linspace(0.0, 60.0, 200)]
    drivers = [("board_ojama_count", 0.42), ("current_max_chain", -0.18),
               ("death_margin", 0.09)]
    frame_out = vao._draw_panel_layout(
        _fake_frame(), adv=32.0, p1=0.66, drivers=drivers, waiting=False,
        history=history, t_rel=45.0, total=60.0,
        state1="STABLE", state2="CHAIN",
        counter_text="応手確率  1P 20%  /  2P 75%", elapsed_sec=45.0,
    )
    out = Path("data/verify/youtube_demo_2026-08-07/release/sample_panel_frame.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), frame_out)
    print(f"[done] {frame_out.shape} -> {out}")


if __name__ == "__main__":
    main()
