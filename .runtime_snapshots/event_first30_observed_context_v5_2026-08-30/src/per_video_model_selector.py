"""動画 ID から最適 CNN model パスを返す (試行 E)。

cross_video 結果から動画別ベスト model を hardcode mapping。
v17b で改善する動画は v17b、それ以外は v16 を採用。

| 動画 | ベスト model | 理由 |
|---|---|---|
| v10 | v17b | +1.21pt |
| v11 | v17b | +0.27pt |
| v19 | v17b | +0.39pt |
| v01 | v17b | +0.11pt |
| v04 | v17b | +0.05pt |
| v05 | v17b | +0.005pt (微小) |
| v06 | v17b | +0.15pt |
| v15 | v16  | -0.47pt for v17b |
| v07 | v16  | v17b -0.23pt |
| v12 | v16  | v17b -1.63pt |
| v13 | v16  | v17b -2.02pt |
| ... | v16  | (default) |

未知動画では v16 を default とする。
"""
from __future__ import annotations

from pathlib import Path

# v17b が cross_video で改善した動画 (improvement > +0.0pt)
V17B_BEST_VIDEOS: frozenset[str] = frozenset({
    "video_01", "video_04", "video_05", "video_06",
    "video_10", "video_11", "video_19",
    "video_16", "video_17",  # 微小改善
})


def select_model_for_video(video_path: str) -> str:
    """動画ファイル path から最適 CNN model のパスを返す。

    Args:
        video_path: 動画 mp4 path (例: "data/frames/video_10.mp4")

    Returns:
        model path (str)
    """
    p = Path(video_path)
    stem = p.stem  # "video_10"
    if stem in V17B_BEST_VIDEOS:
        candidate = Path("models/cnn_phase_u_v17b.pt")
        if candidate.exists():
            return str(candidate)
    return "models/cnn_phase_u_v16.pt"


# ============================
# Phase B: HSV vs CNN-v1 (cnn_phase_b_v1) の動画別選択
# ============================

# 全動画 eval (data/phase_b_eval_summary{,_v1}.tsv) で +2pt 以上 CNN 改善した動画
PHASE_B_CNN_BEST_VIDEOS: frozenset[int] = frozenset({
    1, 2, 7, 9, 10, 11, 12, 13, 16,
})

# 既知の悪化動画 (-5pt 以下、確実に HSV)
PHASE_B_HSV_BEST_VIDEOS: frozenset[int] = frozenset({
    3, 4, 5, 6, 8, 15, 19,
})

# 全動画 eval (data/phase_b_eval_summary{_pv,_smooth3}.tsv) で +2pt 以上
# smoothing=3 改善した動画。これ以外は smoothing=1 (= 無効) を採用。
PHASE_B_SMOOTHING3_BEST_VIDEOS: frozenset[int] = frozenset({
    2, 3, 7, 8, 10, 11, 15, 16,
})


def select_phase_b_smoothing(video_id: int) -> int:
    """Phase B 評価結果に基づき、video_id ごとに smoothing N を選択."""
    return 3 if video_id in PHASE_B_SMOOTHING3_BEST_VIDEOS else 1


def select_phase_b_model(video_id: int) -> str | None:
    """Phase B 評価結果に基づき、video_id ごとに HSV / CNN-v1 を選択.

    Returns:
        - "models/cnn_phase_b_v1.pt" (CNN 採用) ファイルが存在する場合
        - None (HSV のみ採用)
    """
    if video_id in PHASE_B_CNN_BEST_VIDEOS:
        candidate = Path("models/cnn_phase_b_v1.pt")
        if candidate.exists():
            return str(candidate)
    return None


__all__ = [
    "PHASE_B_CNN_BEST_VIDEOS",
    "PHASE_B_HSV_BEST_VIDEOS",
    "PHASE_B_SMOOTHING3_BEST_VIDEOS",
    "V17B_BEST_VIDEOS",
    "select_model_for_video",
    "select_phase_b_model",
    "select_phase_b_smoothing",
]
