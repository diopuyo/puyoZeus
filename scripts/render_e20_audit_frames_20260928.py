"""E20の実手数を目視照合するため、受け側盤面の時系列画像を保存する。"""
from __future__ import annotations
import json
from pathlib import Path
import cv2
import numpy as np

SOURCE = Path("/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer/data/frames/video_zenchi_c0BQoMJwwQU.mp4")
OUT = Path("logs/e20/frames")
SAMPLES_PER_SECOND, COLUMNS = 4, 6
# 確定盤面のおじゃま増加候補まで見るための監査窓。推論には使わない。
AUDIT_ENDS = {2: 2698., 6: 2894.4, 8: 2979.4, 9: 3012.,
              10: 3047.8, 14: 3269.4, 15: 3305., 16: 3353.4}


def main() -> None:
    """原映像を固定時刻で取り出し、設置・着弾前後をまとめる。"""
    OUT.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(SOURCE))
    groups = json.loads(Path("logs/e19/CLASSIFICATION.json").read_text())["groups"]
    rows = [dict(game=g["game_idx"], receiver=g["receiver"],
                 decision_sec=g["evidence"]["decision"]["t_sec"], landing_sec=g["evidence"]["landing_sec"])
            for g in groups if g["direction"] == "正→誤" and
            g["evidence"]["category"] == "可能予測だが着弾前の新発火なし"]
    for row in rows:
        tiles = []
        times = np.arange(row["decision_sec"]-.4, AUDIT_ENDS[row["game"]], 1/SAMPLES_PER_SECOND)
        for stamp in times:
            cap.set(cv2.CAP_PROP_POS_MSEC, stamp*1000)
            ok, frame = cap.read()
            assert ok
            frame = cv2.resize(frame, (960, 540))
            if row == rows[0] and not tiles:
                cv2.imwrite(str(OUT/"full.jpg"), frame)
            x = 50 if row["receiver"] == "1P" else 555
            crop = frame[70:470, x:x+350]
            tile = cv2.copyMakeBorder(crop, 25, 0, 0, 0, cv2.BORDER_CONSTANT)
            cv2.putText(tile, f"{stamp:.3f}", (5, 18), cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 1)
            tiles.append(tile)
        while len(tiles) % COLUMNS:
            tiles.append(np.zeros_like(tiles[0]))
        image = np.vstack([np.hstack(tiles[i:i+COLUMNS]) for i in range(0, len(tiles), COLUMNS)])
        cv2.imwrite(str(OUT/f"actual_game_{row['game']}.jpg"), image)
    cap.release()


if __name__ == "__main__":
    main()
