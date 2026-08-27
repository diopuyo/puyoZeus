"""測定2 (代表動画選定) の下ごしらえ: video_51.mp4 (マスター確認済) 全編を粗くスキャンし、
score のリセット (試合境界) のおおよその位置を見つける (2026-08-25)。

**本番コードは読むだけ。変更しない。** 出力は診断用ログのみ。

粗いストライドで RecognitionPipeline.update を回し、p1/p2 score を記録する。
stride 間引きは「盤面 (board) の継続性」を壊す既知の事故要因
(memory `feedback_frame_sampling_corrupts_boards_2026-07-30`) だが、
本スクリプトは盤面を評価対象にしない (score の値だけを見る)。
score OCR はフレーム単体で完結する読み取りのため、この用途に限り粗い
ストライドで問題ない。試合境界の**おおよその位置**が分かればよく、
正確な境界確認は別途フルレートで細かく見直す。
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import cv2  # noqa: E402

cv2.setNumThreads(1)

from src.recognition_pipeline import RecognitionPipeline  # noqa: E402

VIDEO = PROJECT_ROOT / "data/frames/video_51.mp4"
STRIDE = 30  # 0.5 秒おき


def main() -> None:
    cap = cv2.VideoCapture(str(VIDEO))
    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pipeline = RecognitionPipeline.load_default(
        stable_frame_count=3, load_score_ocr=True, enable_chain_tracker=True,
        temporal_smoothing=1, load_next_detector=False, force_in_match=True,
    )
    prev_p1 = None
    prev_p2 = None
    fi = 0
    while fi < n_frames:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        if fi % STRIDE == 0:
            t_sec = fi / fps
            result = pipeline.update(fi, t_sec, frame)
            p1s, p2s = result.p1.score, result.p2.score
            if (
                prev_p1 is not None and p1s is not None and p2s is not None
                and (p1s < prev_p1 - 500 or p2s < prev_p2 - 500)
            ):
                print(f"[boundary?] t={t_sec:.2f} p1 {prev_p1}->{p1s} p2 {prev_p2}->{p2s}", flush=True)
            if fi % (STRIDE * 200) == 0:
                print(f"[progress] t={t_sec:.2f} p1={p1s} p2={p2s}", flush=True)
            if p1s is not None and p2s is not None:
                prev_p1, prev_p2 = p1s, p2s
        fi += 1
    cap.release()


if __name__ == "__main__":
    main()
