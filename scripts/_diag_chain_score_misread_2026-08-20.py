"""連鎖中のスコア欄が境界を誤発火させていないかを実測する (2026-08-20)。

39番で「連鎖の真っ最中に試合境界が引かれ、その試合の勝敗ラベルが丸ごと
落ちる」現象が見つかった。連鎖中はスコア欄が「消したぷよ数×倍率」の
掛け算表示 (例: 40×16 / 70×260) に変わる既知の仕様
(memory reference_chain_phase_detection_spec) があるため、これを累積スコアと
して読むと巨大な減少に見え、境界検知の判定
(collect_boards_lean: prev_score - score >= SCORE_RESET_THRESHOLD=500) を
誤発火させうる、という仮説を検証する。

ただし ScoreOcr.score_1p は「8桁全て読めた時のみ非 None」を返す仕様なので、
掛け算表示は None になり境界判定をスキップする可能性もある。どちらが実際に
起きているかを、動画の該当区間を細かく読ませて確定させる。修正はしない。
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2

cv2.setNumThreads(1)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.score_ocr import ScoreOcr  # noqa: E402

_FRAMES_DIR = PROJECT_ROOT / "data" / "frames"
_SCORE_RESET_THRESHOLD = 500  # collect_boards_lean と同値


def _scan(video: Path, t_from: float, t_to: float, step: float) -> None:
    """区間を刻んで両サイドのスコアを読み、リセット判定の発火を再現する。"""
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        print(f"[error] 動画を開けない: {video}")
        return
    ocr = ScoreOcr.load_default()
    prev: dict[str, int | None] = {"1P": None, "2P": None}
    print(f"\n=== {video.name} {t_from:.1f}s 〜 {t_to:.1f}s (刻み{step}s) ===")
    print(f"{'時刻':>8} {'1P':>10} {'2P':>10}  {'判定':>28}")
    print("-" * 62)
    t = t_from
    fired = 0
    try:
        while t <= t_to:
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
            ok, frame = cap.read()
            if not ok or frame is None:
                t += step
                continue
            if frame.shape[:2] != (1080, 1920):
                frame = cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_AREA)
            r = ocr.read(frame)
            cur = {"1P": r.score_1p, "2P": r.score_2p}
            notes = []
            for sd in ("1P", "2P"):
                p, c = prev[sd], cur[sd]
                if p is not None and c is not None and p - c >= _SCORE_RESET_THRESHOLD:
                    notes.append(f"{sd}リセット誤発火({p}->{c})")
                    fired += 1
            # 境界判定は score が None のフレームでは何もしない (prev も更新されない)
            for sd in ("1P", "2P"):
                if cur[sd] is not None:
                    prev[sd] = cur[sd]
            s1 = "None" if cur["1P"] is None else str(cur["1P"])
            s2 = "None" if cur["2P"] is None else str(cur["2P"])
            print(f"{t:8.1f} {s1:>10} {s2:>10}  {' / '.join(notes):>28}")
            t += step
    finally:
        cap.release()
    print(f"→ この区間でのリセット誤発火: {fired}回")


def main() -> int:
    """usage: <target_id> <t_from> <t_to> [step]"""
    if len(sys.argv) < 4:
        print("usage: <target_id> <t_from> <t_to> [step]")
        return 1
    vid = _FRAMES_DIR / f"video_{sys.argv[1]}.mp4"
    step = float(sys.argv[4]) if len(sys.argv) > 4 else 0.25
    _scan(vid, float(sys.argv[2]), float(sys.argv[3]), step)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
