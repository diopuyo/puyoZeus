"""
勝敗パネルと数値変化で試合境界を判定する v2。

判定:
    - WinPanelDetector でパネル存在判定。パネルなし = 「対戦動画でない」区間
    - パネルある区間内で、左右の数値画像が変化した瞬間 = 1 試合終了点
    - 起動直後の数値初見は「試合開始」扱い

出力:
    data/verify/match_boundaries_v2/<video_stem>/
        panel_start_XXXXXs_before.png  パネル出現（動画セクション入り口）
        panel_start_XXXXXs_after.png
        panel_end_XXXXXs_before.png    パネル消失（セクション終わり）
        panel_end_XXXXXs_after.png
        match_end_XXXXXs_before.png    数値変化直前
        match_end_XXXXXs_after.png     数値変化直後
        summary.tsv                    t, panel_present, L_hash8, R_hash8, event
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import cv2
import numpy as np

from src.win_panel import WinPanelDetector

# 数値画像が「同じ」とみなすハミング距離閾値（16×16 bin 化後）
DIGIT_SAME_HAMMING: int = 40
# 変化を確定する連続一致サンプル数
CONFIRM_SAMPLES: int = 3


def _digit_signature(patch: np.ndarray) -> np.ndarray:
    """16×16 のバイナリ指紋を返す（大津しきい値）。"""
    if patch is None or patch.size == 0:
        return np.zeros(256, dtype=np.uint8)
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (16, 16), interpolation=cv2.INTER_AREA)
    _, bw = cv2.threshold(small, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return bw.flatten().astype(np.uint8)


def _hamming(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.sum(a != b))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--interval", type=float, default=2.0)
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--end", type=float, default=None)
    parser.add_argument("--out-root", default="data/verify/match_boundaries_v2")
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"動画なし: {video_path}", file=sys.stderr)
        return 1

    detector = WinPanelDetector.load_default()
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    end_t = args.end if args.end is not None else duration
    print(f"動画: {video_path.name}  duration={duration:.0f}s")

    out_dir = Path(args.out_root) / video_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[tuple[float, bool, str, str, str]] = []
    # 確定状態（チャタリング抑制のため連続一致で更新）
    confirmed_panel: bool | None = None
    confirmed_L: np.ndarray | None = None
    confirmed_R: np.ndarray | None = None

    # 候補と連続一致カウント
    pending_panel: bool | None = None
    panel_count = 0
    pending_L: np.ndarray | None = None
    pending_R: np.ndarray | None = None
    digit_count = 0

    prev_frame: np.ndarray | None = None
    events: list[tuple[str, float, np.ndarray | None, np.ndarray]] = []

    t = args.start
    while t < end_t:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok, frame = cap.read()
        if not ok or frame is None:
            t += args.interval
            continue
        if frame.shape[:2] != (1080, 1920):
            frame = cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_AREA)
        result = detector.detect(frame)
        sigL = _digit_signature(result.digit_left_roi) if result.present else np.zeros(256, dtype=np.uint8)
        sigR = _digit_signature(result.digit_right_roi) if result.present else np.zeros(256, dtype=np.uint8)
        hashL = "".join(str(x) for x in sigL[:32].tolist())
        hashR = "".join(str(x) for x in sigR[:32].tolist())
        event = ""

        # ---- パネル有無の確定 (連続一致) ----
        if confirmed_panel is None:
            # 初回
            confirmed_panel = result.present
        elif result.present != confirmed_panel:
            if pending_panel == result.present:
                panel_count += 1
            else:
                pending_panel = result.present
                panel_count = 1
            if panel_count >= CONFIRM_SAMPLES:
                old = confirmed_panel
                confirmed_panel = result.present
                pending_panel = None
                panel_count = 0
                event = "panel_start" if confirmed_panel else "panel_end"
                events.append((event, t, prev_frame, frame.copy()))
                print(f"  [{event}] t={t:.1f}s  score={result.score:.3f}")
                # パネル新規出現時に digit を初期化
                if event == "panel_start":
                    confirmed_L, confirmed_R = sigL, sigR
                    pending_L = pending_R = None
                    digit_count = 0
                else:
                    confirmed_L = confirmed_R = None
        else:
            pending_panel = None
            panel_count = 0

        # ---- 数値変化（パネル確定中のみ）----
        if confirmed_panel and confirmed_L is not None and confirmed_R is not None and result.present:
            dL = _hamming(sigL, confirmed_L)
            dR = _hamming(sigR, confirmed_R)
            changed = (dL >= DIGIT_SAME_HAMMING or dR >= DIGIT_SAME_HAMMING)
            if changed:
                # 候補としても前回と一致しているか
                if pending_L is not None and pending_R is not None:
                    dL2 = _hamming(sigL, pending_L)
                    dR2 = _hamming(sigR, pending_R)
                    if dL2 < DIGIT_SAME_HAMMING and dR2 < DIGIT_SAME_HAMMING:
                        digit_count += 1
                    else:
                        pending_L, pending_R = sigL, sigR
                        digit_count = 1
                else:
                    pending_L, pending_R = sigL, sigR
                    digit_count = 1
                if digit_count >= CONFIRM_SAMPLES:
                    # 数値変化確定 = 試合終了
                    confirmed_L, confirmed_R = sigL, sigR
                    pending_L = pending_R = None
                    digit_count = 0
                    event = "match_end"
                    events.append(("match_end", t, prev_frame, frame.copy()))
                    print(f"  [match_end] t={t:.1f}s  dL={dL} dR={dR}")
            else:
                pending_L = pending_R = None
                digit_count = 0

        rows.append((t, result.present, hashL, hashR, event))
        prev_frame = frame.copy()
        t += args.interval

    cap.release()

    # 画像保存
    for kind, t_sec, before, after in events:
        tag = f"{kind}_{int(t_sec):05d}s"
        if before is not None:
            cv2.imwrite(str(out_dir / f"{tag}_before.png"), before)
        cv2.imwrite(str(out_dir / f"{tag}_after.png"), after)

    # サマリ
    tsv = out_dir / "summary.tsv"
    with tsv.open("w", encoding="utf-8") as f:
        f.write("t_sec\tpanel\tL_hash\tR_hash\tevent\n")
        for t_sec, panel, hL, hR, ev in rows:
            f.write(f"{t_sec:.1f}\t{int(panel)}\t{hL[:16]}...\t{hR[:16]}...\t{ev}\n")

    panel_starts = sum(1 for e in events if e[0] == "panel_start")
    panel_ends = sum(1 for e in events if e[0] == "panel_end")
    matches = sum(1 for e in events if e[0] == "match_end")
    print(f"\nパネル出現: {panel_starts}  パネル消失: {panel_ends}  試合終了検出: {matches}")
    print(f"出力: {out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
