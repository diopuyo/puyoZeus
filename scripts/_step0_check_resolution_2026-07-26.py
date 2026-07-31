"""Step0 検証用: c系4動画の解像度を確認する使い捨てスクリプト。"""
import cv2

for v in ["c1", "c4", "c34", "c82"]:
    path = f"data/frames/video_{v}.mp4"
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"video_{v}: OPEN FAILED")
        continue
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"video_{v}: {w}x{h}")
    cap.release()
