"""Step0 検証用: c系4動画の長さと既存 winners_probe の整合性を確認する使い捨てスクリプト。"""
import cv2

for v in ["c1", "c4", "c34", "c82"]:
    path = f"data/frames/video_{v}.mp4"
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"video_{v}: OPEN FAILED")
        continue
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    tot = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    dur = tot / fps if fps else 0.0
    print(f"video_{v}: duration={dur:.1f}s ({dur/60:.1f}min) fps={fps:.2f}")
    cap.release()
