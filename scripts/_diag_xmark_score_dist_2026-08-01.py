"""×印のUIマスクNCCスコア分布を多数動画で実測する (2026-08-01)。

## 目的
満杯盤面で×印がぷよ化する誤りが 2/2 で発生 (v2-30, v3-027)。
実測で c23型 (0.720、閾値0.75まで0.03) と c11型 (最大0.597、テンプレ不一致) が判明。
閾値を 0.75→0.70 に下げる案の採否には**広い分布**が必要:
  1. 試合中の r1c2 スコア分布 → 0.75未満率 = 検出漏れの規模
  2. 0.70-0.75 帯の実態 → 下げた時に何が新たに検出されるか
  3. c11 型 (テンプレ不一致動画) がどれだけあるか

×印は試合中常時表示なので、試合中フレームの r1c2 スコアがそのまま検出力。
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.image_reader import DEFAULT_P1_REGION, DEFAULT_P2_REGION  # noqa: E402
from src.ui_mask import UiMaskMatcher  # noqa: E402

SAMPLES_PER_VIDEO: int = 8
PANEL_DIR = Path("data/verify/winners_panel_diff_2026-07-26")

def main() -> None:
    cv2.setNumThreads(1)
    m = UiMaskMatcher.load_default()
    videos = sorted(Path("data/frames").glob("video_c*.mp4"))[:24]
    print(f"閾値={m._threshold} 対象{len(videos)}動画 x {SAMPLES_PER_VIDEO}フレーム x 2side")
    all_scores: list[float] = []
    per_video_max: dict[str, float] = {}
    rows = []
    for vp in videos:
        vid = vp.stem
        pj = PANEL_DIR / f"{vid}.json"
        if not pj.exists():
            continue
        games = json.loads(pj.read_text(encoding="utf-8")).get("games", [])
        if not games:
            continue
        cap = cv2.VideoCapture(str(vp))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        # 試合中央の時刻を等間隔に選ぶ
        mids = [(g["start_sec"] + g["end_sec"]) / 2 for g in games]
        step = max(1, len(mids) // SAMPLES_PER_VIDEO)
        scores_v = []
        for t in mids[::step][:SAMPLES_PER_VIDEO]:
            cap.set(cv2.CAP_PROP_POS_FRAMES, float(int(t * fps)))
            ok, f = cap.read()
            if not ok:
                continue
            if f.shape[:2] != (1080, 1920):
                f = cv2.resize(f, (1920, 1080), interpolation=cv2.INTER_AREA)
            for reg in (DEFAULT_P1_REGION, DEFAULT_P2_REGION):
                x1, y1, x2, y2 = reg.cell_sample_rect(1, 2)
                r = m.match(f[y1:y2, x1:x2])
                scores_v.append(r.score)
        cap.release()
        if scores_v:
            arr = np.asarray(scores_v)
            all_scores.extend(scores_v)
            per_video_max[vid] = float(arr.max())
            rows.append((vid, float(np.median(arr)), float(arr.max()),
                         float((arr >= 0.75).mean()), float((arr >= 0.70).mean())))
    print(f"\n{'動画':<12}{'中央':>7}{'最大':>7}{'>=0.75率':>9}{'>=0.70率':>9}")
    for vid, med, mx, r75, r70 in rows:
        mark = " ★テンプレ不一致疑い" if mx < 0.70 else ""
        print(f"{vid:<12}{med:>7.3f}{mx:>7.3f}{r75:>9.2f}{r70:>9.2f}{mark}")
    a = np.asarray(all_scores)
    print(f"\n全体 n={a.size}: 検出率(>=0.75)={100*(a>=0.75).mean():.1f}%  "
          f"0.70化した場合={100*(a>=0.70).mean():.1f}%  "
          f"0.70-0.75帯={int(((a>=0.70)&(a<0.75)).sum())}件")
    n_bad = sum(1 for v in per_video_max.values() if v < 0.70)
    print(f"テンプレ不一致疑い (最大<0.70): {n_bad}/{len(per_video_max)} 動画")

if __name__ == "__main__":
    main()
