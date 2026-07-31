import cv2
names = ["c8","c15","c17","c23","c31","c45","c50","c58","c70","c78",
         "c85","c86","c89","c92","c95","33","36"]
for n in names:
    fn = f"data/frames/video_{n}.mp4"
    cap = cv2.VideoCapture(fn)
    if not cap.isOpened():
        print(f"{fn}: OPEN FAIL")
        continue
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    dur = frames/fps if fps else 0
    ok, _ = cap.read()
    print(f"{fn}: fps={fps:.1f} dur={dur:.1f}s read_ok={ok}")
    cap.release()
