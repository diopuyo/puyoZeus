"""W10 段階1: NEXT 読取の赤/紫誤読を実フレームで検証する (2026-08-17)。

再DL動画 (video_c11/c23、640x360) から着地イベント直前の 2P NEXT 窓を
切り出し、本番と同一構成 (models/cnn_global_best.pt、centroid無し) の
NextDetector で分類する。同時に人手目視用の crop 画像を保存する。

再DL動画は元動画 (削除済) と内容が一致する保証がない (W8既知リスク) ため、
本スクリプトは「厳密な同一フレーム照合」ではなく「同時間帯の NEXT 表示の
傾向確認」として扱う (目視での最終判定が必須)。

出力: data/verify/diag_w10_next_2026-08-17/{video}/next_history.csv
      data/verify/diag_w10_next_2026-08-17/{video}/crops/*.png
"""
from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np
import torch

from src.next_detector import NextDetector, ROI_2P_NEXT_TOP, ROI_2P_NEXT_BOT
from src.patch_classifier import CnnPatchClassifier

_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = _ROOT / "data" / "verify" / "diag_w10_next_2026-08-17"

# (動画パス, 表示名, 着地イベント時刻, 検証窓 [開始,終了] 秒, ステップ秒)
TARGETS = [
    ("/tmp/w10_diag/c11_full.mp4", "c11", 898.467, 892.0, 900.0, 0.2),
    ("/tmp/w10_diag/c23_full.mp4", "c23", 1405.12, 1398.0, 1407.0, 0.2),
]

PROD_RESOLUTION = (1920, 1080)  # 本番想定解像度 (幅, 高さ)


def _load_next_detector() -> NextDetector:
    """本番構成 (recognition_pipeline.py:3003-3024) を再現。centroid 無し。"""
    cnn = CnnPatchClassifier()
    gbest = _ROOT / "models" / "cnn_global_best.pt"
    if gbest.exists():
        state = torch.load(str(gbest), map_location="cpu", weights_only=True)
        cnn._model.load_state_dict(state)
    return NextDetector(classifier=cnn)


def main() -> None:
    det = _load_next_detector()
    for video_path, name, event_t, t_start, t_end, step in TARGETS:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[warn] 開けません: {video_path}")
            continue
        fps = cap.get(cv2.CAP_PROP_FPS)
        out_dir = OUT_DIR / name
        crops_dir = out_dir / "crops"
        crops_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        t = t_start
        while t <= t_end:
            frame_idx = int(round(t * fps))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if not ok or frame is None:
                t += step
                continue
            frame_1080 = cv2.resize(frame, PROD_RESOLUTION, interpolation=cv2.INTER_LINEAR)
            result = det.detect_2p(frame_1080)
            y1, y2, x1, x2 = ROI_2P_NEXT_TOP
            crop_top = frame_1080[y1:y2, x1:x2]
            y1b, y2b, x1b, x2b = ROI_2P_NEXT_BOT
            crop_bot = frame_1080[y1b:y2b, x1b:x2b]
            crop = np.vstack([crop_top, crop_bot])
            tag = "EVENT" if abs(t - event_t) < step / 2 else ""
            crop_path = crops_dir / f"t{t:.3f}_next{tag}.png"
            cv2.imwrite(str(crop_path), crop)
            rows.append({
                "t_sec": round(t, 3),
                "is_event": tag == "EVENT",
                "next_top": result.next_top,
                "next_bot": result.next_bot,
                "dnext_top": result.dnext_top,
                "dnext_bot": result.dnext_bot,
                "crop": str(crop_path.relative_to(_ROOT)),
            })
            t += step
        cap.release()
        csv_path = out_dir / "next_history.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
            w.writeheader()
            w.writerows(rows)
        print(f"[done] {name}: {len(rows)} 件 -> {csv_path}")
        for r in rows:
            mark = " <== 着地イベント" if r["is_event"] else ""
            print(
                f"  t={r['t_sec']:.3f} next=({r['next_top']},{r['next_bot']}) "
                f"dnext=({r['dnext_top']},{r['dnext_bot']}){mark}"
            )


if __name__ == "__main__":
    main()
