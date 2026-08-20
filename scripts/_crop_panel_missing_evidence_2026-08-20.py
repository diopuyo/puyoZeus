"""勝敗ラベルが付かなかった試合の実画面を切り出す (2026-08-20、user レビュー用)。

39番で「パネルは映っているのに勝者を決められない」試合が9件あった。その
判定に実際に使われた2時点 (試合の始まり側 t_a と 終わり側 t_b) のフレームを
切り出し、WIN★パネル領域を拡大して並べる。パネルの数字が本当に変化して
いないのか、映っているが読めないのかを目視できるようにする。

比較のため、正常にラベルが付いた試合も同じ形式で出す (対照)。

memory feedback_review_actual_screen_frames_2026-07-24: レビューは実ゲーム
画面フレームを貼る。memory feedback_review_image_links: 画像は Windows パスで。
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

cv2.setNumThreads(1)  # 収集ジョブと競合させない

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.match_winner import MatchWinnerDetector  # noqa: E402
from src.win_panel import PANEL_X_RANGE, PANEL_Y_RANGE  # noqa: E402

_NPZ_DIR = PROJECT_ROOT / "data" / "indicators_v2"
_FRAMES_DIR = PROJECT_ROOT / "data" / "frames"
_OUT_DIR = PROJECT_ROOT / "data" / "verify" / "panel_missing_evidence_2026-08-20"

_PANEL_ZOOM = 3  # パネル拡大率 (320x60 -> 960x180)
_LABEL_H = 34


def _read(cap: cv2.VideoCapture, t_sec: float) -> np.ndarray | None:
    """指定秒のフレームを 1920x1080 で読む。"""
    cap.set(cv2.CAP_PROP_POS_MSEC, t_sec * 1000.0)
    ok, frame = cap.read()
    if not ok or frame is None:
        return None
    if frame.shape[:2] != (1080, 1920):
        frame = cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_AREA)
    return frame


def _panel_strip(frame: np.ndarray, label: str) -> np.ndarray:
    """パネル領域を拡大し、上に説明ラベルを載せた帯を作る。"""
    y1, y2 = PANEL_Y_RANGE
    x1, x2 = PANEL_X_RANGE
    panel = frame[y1:y2, x1:x2]
    panel = cv2.resize(
        panel, (panel.shape[1] * _PANEL_ZOOM, panel.shape[0] * _PANEL_ZOOM),
        interpolation=cv2.INTER_NEAREST,
    )
    bar = np.zeros((_LABEL_H, panel.shape[1], 3), dtype=np.uint8)
    cv2.putText(bar, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([bar, panel])


def _dump_game(
    cap: cv2.VideoCapture, gid: int, t_a: float | None, t_b: float | None,
    verdict: str, out_dir: Path,
) -> None:
    """1試合ぶん: 判定に使った2時点のパネル比較 + 全画面を保存する。"""
    strips = []
    for tag, ts in (("試合の始まり側", t_a), ("試合の終わり側", t_b)):
        if ts is None:
            continue
        fr = _read(cap, ts)
        if fr is None:
            continue
        strips.append(_panel_strip(fr, f"game{gid} {tag} t={ts:.1f}s"))
        # 全画面も1枚だけ (終わり側=勝敗が確定した時点) 残す
        if tag.startswith("試合の終わり"):
            cv2.imwrite(str(out_dir / f"game{gid:03d}_{verdict}_full_{ts:.0f}s.png"), fr)
    if strips:
        w = max(s.shape[1] for s in strips)
        strips = [
            cv2.copyMakeBorder(s, 0, 0, 0, w - s.shape[1], cv2.BORDER_CONSTANT, value=0)
            for s in strips
        ]
        cv2.imwrite(str(out_dir / f"game{gid:03d}_{verdict}_panel.png"), np.vstack(strips))


def main() -> int:
    """usage: <npz_dir> <target_id> [最大出力試合数]"""
    if len(sys.argv) < 3:
        print("usage: <npz_dir> <target_id> [max_games]")
        return 1
    npz = _NPZ_DIR / sys.argv[1] / f"{sys.argv[2]}.npz"
    vid = _FRAMES_DIR / f"video_{sys.argv[2]}.mp4"
    max_games = int(sys.argv[3]) if len(sys.argv) > 3 else 4

    d = np.load(npz, allow_pickle=True)
    won = np.asarray(d["won"], dtype=float)
    game = np.asarray(d["game_idx"])
    t = np.asarray(d["t_sec"], dtype=float)
    games = sorted(np.unique(game).tolist())
    starts = [float(t[game == g].min()) for g in games]

    out_dir = _OUT_DIR / sys.argv[2]
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(vid))
    if not cap.isOpened():
        print(f"[error] 動画を開けない: {vid}")
        return 1
    try:
        det = MatchWinnerDetector.load_default()
        # 判定に実際に使われた読み取り時刻を取り出す (detect_all_winners と同じ入口)
        reads = det._resolve_read_times(cap, starts, float(t.max()), 1.0)
        results = det.detect_all_winners(cap, starts, float(t.max()))

        # 欠損した試合は理由を問わず全て拾う (2026-08-20 変更: 当初は
        # 「パネルは映っているが勝者不定」だけを対象にしていたが、
        # 得点との食い違い (winner は出ているが score と不一致) も
        # 目視で決着させたいため条件を緩めた)。
        missing, normal = [], []
        for i, (g, r) in enumerate(zip(games, results)):
            sel = game == g
            if bool(np.isnan(won[sel]).all()):
                missing.append((i, g, r))
            elif r.winner is not None:
                normal.append((i, g, r))

        print(f"[dump] 判定不能で欠損 {len(missing)}件 / 正常 {len(normal)}件")
        for i, g, r in missing[:max_games]:
            print(f"  欠損 game{g}: t_a={reads[i]} t_b={reads[i+1]} "
                  f"左変化={r.left_changed}({r.left_hamming}) "
                  f"右変化={r.right_changed}({r.right_hamming})")
            _dump_game(cap, g, reads[i], reads[i + 1], "MISSING", out_dir)
        for i, g, r in normal[:2]:
            print(f"  正常 game{g}: t_a={reads[i]} t_b={reads[i+1]} "
                  f"左変化={r.left_changed}({r.left_hamming}) "
                  f"右変化={r.right_changed}({r.right_hamming}) 勝者={r.winner}")
            _dump_game(cap, g, reads[i], reads[i + 1], f"OK_{r.winner}", out_dir)
    finally:
        cap.release()

    print(f"[out] {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
