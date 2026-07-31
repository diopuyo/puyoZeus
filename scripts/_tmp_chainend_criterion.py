"""連鎖終了基準の検証 (scratch): 「掛け算式が出た後、次ツモ or お邪魔落下で連鎖終了」。

各サイドで:
  - 掛け算式検知 (_check_formula_detected) または CHAIN 状態 → 連鎖アクティブ開始
  - アクティブ中に TSUMO_FALL or OJAMA_FALL を観測 → その瞬間を連鎖終了とし1イベント確定
  - 連鎖開始前の最後の読めるスコア → 終了直後の読めるスコア で得点差を記録
この基準で連鎖が綺麗に数えられるか(特に t=142 で2Pの2連鎖を分離できるか)を見る。
"""
from __future__ import annotations

import sys

import cv2

sys.path.insert(0, ".")

from src.board_state_machine import BoardState
from src.recognition_pipeline import RecognitionPipeline

VIDEO = "data/frames/video_124_4min.mp4"
TARGET_W, TARGET_H = 1920, 1080
END_STATES = {BoardState.TSUMO_FALL, BoardState.OJAMA_FALL}
DETAIL = (136.0, 146.0)


class _Det:
    def __init__(self, side: str) -> None:
        self.side = side
        self.active = False
        self.start_score: int | None = None
        self.start_t = 0.0
        self.last_score: int | None = None
        self.events: list[tuple] = []  # (t_start, t_end, s_start, s_end, end_reason)

    def feed(self, state, formula: bool, score: int | None, t: float) -> None:
        if score is not None:
            self.last_score = score
        chain_signal = formula or state == BoardState.CHAIN
        if chain_signal and not self.active:
            # 連鎖開始: 直前の読めるスコアを基準に
            self.active = True
            self.start_score = self.last_score
            self.start_t = t
        elif self.active and state in END_STATES:
            # 連鎖終了: 次ツモ or お邪魔落下
            self.events.append((round(self.start_t, 2), round(t, 2),
                                self.start_score, self.last_score,
                                "tsumo" if state == BoardState.TSUMO_FALL else "ojama"))
            self.active = False
            self.start_score = None


def main() -> None:
    pipe = RecognitionPipeline.load_default(
        stable_frame_count=3, load_score_ocr=True, enable_chain_tracker=True,
        temporal_smoothing=1, load_next_detector=True, force_in_match=True,
    )
    if hasattr(pipe, "set_video_id"):
        pipe.set_video_id("video_124")
    cap = cv2.VideoCapture(VIDEO)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    d1, d2 = _Det("1P"), _Det("2P")
    last1 = last2 = None
    for fi in range(int(240 * fps)):
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        if frame.shape[:2] != (TARGET_H, TARGET_W):
            frame = cv2.resize(frame, (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA)
        t = fi / fps
        r = pipe.update(fi, t, frame)
        if r.p1.score is not None:
            last1 = r.p1.score
        if r.p2.score is not None:
            last2 = r.p2.score
        f1 = RecognitionPipeline._check_formula_detected(frame, pipe._score_ocr, "1P", last1)
        f2 = RecognitionPipeline._check_formula_detected(frame, pipe._score_ocr, "2P", last2)
        d1.feed(r.p1.state, f1, r.p1.score, t)
        d2.feed(r.p2.state, f2, r.p2.score, t)
    cap.release()

    for d in (d1, d2):
        print(f"\n=== {d.side}: 連鎖イベント {len(d.events)}件 ===")
        print(" t_start  t_end  s_start s_end  rise  end")
        for (ts, te, ss, se, reason) in d.events:
            rise = (se - ss) if (ss is not None and se is not None) else None
            print(f"{ts:7.2f} {te:6.2f} {str(ss):>6s} {str(se):>6s} {str(rise):>5s}  {reason}")


if __name__ == "__main__":
    main()
