"""Phase Z-1: 動画区間の連続 frame レビュー UI。

99.5% 認識率達成に向けた動画ベース評価ツール:
    - 0.1s ごとに frame 抽出 (5 倍密度)
    - StatePipeline で GameState 抽出
    - 0.5s ごとに 5 frame の cell 多数決で「確定盤面」生成
    - ChainPhaseDetector で連鎖中区間を識別
    - 連鎖中は ChainSimulator(prev_stable_board) 予測を suspicious 判定の reference に
    - suspicious cell 判定 (3 ルール OR):
        1. 物理ルール違反 (1 cell 落下、色→色直接遷移、浮遊)
        2. 連鎖中の simulator 予測との差分
        3. フリッカー (5 frame で 2 色以上に振動)

出力 (`out_dir/`):
    - frames/{ms:06d}.png   : 0.5s ごとの確定盤面 overlay (P1+P2)
    - field_sheet.png       : 全確定盤面を grid 配置
    - suspicious_sheet.png  : suspicious cells のみ抽出した cell sheet
    - labels.csv            : 全 cell (12*6*2*N_consolidated) の recognized + your_answer
    - suspicious.csv        : suspicious cells のみ + 判定理由
    - summary.html          : 一覧
    - accuracy.tsv          : your_answer 入力後に再計算する集計

利用例:
    PYTHONPATH=. ./venv/bin/python -m scripts.phase_z_review_ui \
        --video data/frames/video_18.mp4 \
        --start 30 --end 60 \
        --bg-fp-time 30 \
        --out-dir data/verify/phase_z_review/v18_m03_30_60
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.console_init import init_console, to_windows_path  # noqa: E402
init_console()

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from src.board import (  # noqa: E402
    BOARD_COLS, BOARD_ROWS, Board, COLOR_BLUE, COLOR_EMPTY,
    COLOR_GREEN, COLOR_OJAMA, COLOR_PURPLE, COLOR_RED,
    COLOR_UNKNOWN, COLOR_YELLOW, HIDDEN_ROWS,
)
from src.chain_phase_detector import ChainPhaseDetector  # noqa: E402
from src.image_reader import (  # noqa: E402
    DEFAULT_P1_REGION, DEFAULT_P2_REGION,
)
from src.state_pipeline import StatePipeline  # noqa: E402

# ============================
# 定数
# ============================

FRAME_INTERVAL_SEC: float = 0.1
CONSOLIDATED_INTERVAL_SEC: float = 0.5
WINDOW_SIZE: int = 5  # = CONSOLIDATED_INTERVAL_SEC / FRAME_INTERVAL_SEC

# Z-1.6: CNN 確信度しきい値 (max softmax)。これ未満で suspicious 化
CNN_CONF_THRESHOLD: float = 0.60
# Z-1.7: EM 判定なのに HSV 中央が高彩度なら suspicious (検出漏れの典型シグナル)
EM_SATURATION_THRESHOLD: int = 60   # 中央 S 平均がこれ以上 + EM 判定で suspicious
EM_VALUE_THRESHOLD: int = 70         # 中央 V 平均がこれ以上で「色がある」
# Z-1.8: OJM (灰色) 判定用 — recognized=EM/?? かつ S 低 + V 中域 で suspicious
OJM_S_MAX: int = 55                  # OJM は彩度低
OJM_V_MIN: int = 110                 # 真の EM 背景より明るい (V 中域以上)
OJM_V_MAX: int = 220

COLOR_LABEL: dict[int, str] = {
    COLOR_EMPTY: "EM", COLOR_RED: "R", COLOR_BLUE: "B",
    COLOR_GREEN: "G", COLOR_YELLOW: "Y", COLOR_PURPLE: "P",
    COLOR_OJAMA: "O", COLOR_UNKNOWN: "?",
}
COLOR_FULL: dict[int, str] = {
    COLOR_EMPTY: "EM", COLOR_RED: "RED", COLOR_BLUE: "BLUE",
    COLOR_GREEN: "GRN", COLOR_YELLOW: "YEL", COLOR_PURPLE: "PUR",
    COLOR_OJAMA: "OJM", COLOR_UNKNOWN: "??",
}
COLOR_BORDER: dict[int, tuple[int, int, int]] = {
    COLOR_EMPTY: (60, 60, 60),
    COLOR_RED: (60, 60, 240),
    COLOR_BLUE: (240, 100, 60),
    COLOR_GREEN: (60, 220, 60),
    COLOR_YELLOW: (60, 240, 240),
    COLOR_PURPLE: (220, 60, 220),
    COLOR_OJAMA: (190, 190, 190),
    COLOR_UNKNOWN: (120, 120, 200),
}
SUSPICIOUS_BORDER: tuple[int, int, int] = (0, 200, 255)  # 黄色枠


@dataclass
class FrameState:
    """1 frame の抽出結果 (0.1s 単位)。"""
    t_sec: float
    board_p1: Board
    board_p2: Board
    next_p1: tuple[int, int] | None
    next_p2: tuple[int, int] | None
    dnext_p1: tuple[int, int] | None
    dnext_p2: tuple[int, int] | None
    score_p1: int | None
    score_p2: int | None
    pending_ojama_p1: int
    pending_ojama_p2: int
    is_chain_p1: bool
    is_chain_p2: bool
    predicted_p1: Board | None  # 連鎖中の simulator 予測
    predicted_p2: Board | None
    frame_bgr: np.ndarray  # 元画像 (overlay 用)
    # Z-1.6: cell ごとの CNN max softmax (12 visible rows × 6 cols)
    conf_p1: np.ndarray | None = None
    conf_p2: np.ndarray | None = None
    # Z-1.7: cell 中央 50% の HSV S/V 平均 (12 × 6)
    sat_p1: np.ndarray | None = None
    sat_p2: np.ndarray | None = None
    val_p1: np.ndarray | None = None
    val_p2: np.ndarray | None = None
    # Z-1.8: HSV 単独判定の color grid (12 × 6 int)
    hsv_color_p1: np.ndarray | None = None
    hsv_color_p2: np.ndarray | None = None


@dataclass
class ConsolidatedState:
    """0.5s ごとの確定盤面 + 補助情報。"""
    t_sec: float
    board_p1: Board  # majority vote 後 (連鎖中は predicted_p1 採用)
    board_p2: Board
    next_p1: tuple[int, int] | None
    next_p2: tuple[int, int] | None
    dnext_p1: tuple[int, int] | None
    dnext_p2: tuple[int, int] | None
    score_p1: int | None
    score_p2: int | None
    pending_ojama_p1: int
    pending_ojama_p2: int
    is_chain_p1: bool
    is_chain_p2: bool
    # Z-1.6: window 平均の CNN 確信度 (12 visible rows × 6 cols)
    conf_p1: np.ndarray | None = None
    conf_p2: np.ndarray | None = None
    # Z-1.7: window 平均の HSV S/V (12 × 6)
    sat_p1: np.ndarray | None = None
    sat_p2: np.ndarray | None = None
    val_p1: np.ndarray | None = None
    val_p2: np.ndarray | None = None
    # Z-1.8: HSV 単独判定の最頻 color grid (12 × 6 int)
    hsv_color_p1: np.ndarray | None = None
    hsv_color_p2: np.ndarray | None = None
    suspicious_p1: dict[tuple[int, int], list[str]] = field(default_factory=dict)
    suspicious_p2: dict[tuple[int, int], list[str]] = field(default_factory=dict)
    representative_frame: np.ndarray | None = None  # window 中央の元画像


# ============================
# パイプライン構築
# ============================


def build_pipeline(
    cnn_model: str,
    all_refiners_on: bool,
    use_online_hsv: bool = False,
    use_cell_anomaly: bool = False,
    use_hsv_anomaly: bool = False,
    use_connectivity_outlier: bool = False,
    use_stability: bool = False,
) -> StatePipeline:
    """補正レイヤー全部 ON の StatePipeline を構築。"""
    return StatePipeline(
        cnn_model_path=cnn_model,
        use_per_video_calibrator=all_refiners_on,
        use_temporal_voting=all_refiners_on,
        use_score_eraser=all_refiners_on,
        use_pair_landing_check=all_refiners_on,
        use_enhanced_tracker=all_refiners_on,
        use_online_hsv=use_online_hsv,
        use_cell_anomaly=use_cell_anomaly,
        use_hsv_anomaly=use_hsv_anomaly,
        use_connectivity_outlier=use_connectivity_outlier,
        use_stability=use_stability,
    )


def extract_confidence_grid(
    frame: np.ndarray, region, classifier,
) -> np.ndarray | None:
    """1 frame の region について cell ごとの CNN max softmax を返す。

    Z-3C: 144 cells を 1 バッチで GPU 推論 (個別呼び出しから 5-20x 高速化)。

    Returns:
        shape=(12, 6) の float32 array、各 cell の max softmax prob。
        classifier が predict_proba 非対応なら None。
    """
    if classifier is None or not hasattr(classifier, "predict_proba"):
        return None
    h, w = frame.shape[:2]
    patches: list[np.ndarray] = []
    positions: list[tuple[int, int]] = []
    for vrow in range(12):
        row = vrow + HIDDEN_ROWS
        for col in range(BOARD_COLS):
            x1, y1, x2, y2 = region.cell_sample_rect(row, col)
            x1 = max(0, min(x1, w - 1))
            x2 = max(x1 + 1, min(x2, w))
            y1 = max(0, min(y1, h - 1))
            y2 = max(y1 + 1, min(y2, h))
            patch = frame[y1:y2, x1:x2]
            if patch.size == 0:
                # zero patch でバッチ整合
                patch = np.zeros((1, 1, 3), dtype=np.uint8)
            patches.append(patch)
            positions.append((vrow, col))

    grid = np.zeros((12, BOARD_COLS), dtype=np.float32)
    try:
        if hasattr(classifier, "predict_proba_batch"):
            probs_batch = classifier.predict_proba_batch(patches)
            for (vrow, col), probs in zip(positions, probs_batch):
                grid[vrow, col] = float(np.max(probs))
        else:
            # fallback: 個別呼び出し
            for (vrow, col), patch in zip(positions, patches):
                probs = classifier.predict_proba(patch)
                grid[vrow, col] = float(np.max(probs))
    except Exception:
        pass
    return grid


def extract_hsv_grid(
    frame: np.ndarray, region,
    hsv_full: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """各 cell 中央 50% の (S 平均, V 平均) を返す (shape=(12, 6) 各)。

    Z-3C: hsv_full を渡せば cvtColor を skip (P1/P2 で共有可)。
    """
    h, w = frame.shape[:2]
    if hsv_full is None:
        hsv_full = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    sat = np.zeros((12, BOARD_COLS), dtype=np.float32)
    val = np.zeros((12, BOARD_COLS), dtype=np.float32)
    for vrow in range(12):
        row = vrow + HIDDEN_ROWS
        for col in range(BOARD_COLS):
            x1, y1, x2, y2 = region.cell_sample_rect(row, col)
            x1 = max(0, min(x1, w - 1))
            x2 = max(x1 + 1, min(x2, w))
            y1 = max(0, min(y1, h - 1))
            y2 = max(y1 + 1, min(y2, h))
            ph = y2 - y1
            pw = x2 - x1
            if ph <= 0 or pw <= 0:
                continue
            # 中央 50% を indexing で取り出す (cvtColor 不要)
            iy0 = y1 + ph // 4
            iy1 = y2 - ph // 4
            ix0 = x1 + pw // 4
            ix1 = x2 - pw // 4
            if iy1 <= iy0 or ix1 <= ix0:
                continue
            inner = hsv_full[iy0:iy1, ix0:ix1]
            sat[vrow, col] = float(np.mean(inner[:, :, 1]))
            val[vrow, col] = float(np.mean(inner[:, :, 2]))
    return sat, val


def extract_hsv_color_grid(
    frame: np.ndarray, region, hsv_classifier,
) -> np.ndarray | None:
    """HSV 単独判定の color code grid (12x6)。CNN/HSV 不一致検出用。

    Z-3C: classify_batch があれば一括判定 (HybridClassifier の HSV 部分は
    ColorClassifier、現状 batch API なし。CNN だけバッチ化)。
    """
    if hsv_classifier is None:
        return None
    h, w = frame.shape[:2]
    patches: list[np.ndarray] = []
    positions: list[tuple[int, int]] = []
    for vrow in range(12):
        row = vrow + HIDDEN_ROWS
        for col in range(BOARD_COLS):
            x1, y1, x2, y2 = region.cell_sample_rect(row, col)
            x1 = max(0, min(x1, w - 1))
            x2 = max(x1 + 1, min(x2, w))
            y1 = max(0, min(y1, h - 1))
            y2 = max(y1 + 1, min(y2, h))
            patch = frame[y1:y2, x1:x2]
            if patch.size == 0:
                patch = np.zeros((1, 1, 3), dtype=np.uint8)
            patches.append(patch)
            positions.append((vrow, col))

    grid = np.zeros((12, BOARD_COLS), dtype=np.int32)
    if hasattr(hsv_classifier, "classify_batch"):
        try:
            colors = hsv_classifier.classify_batch(patches)
            for (vrow, col), c in zip(positions, colors):
                grid[vrow, col] = int(c)
            return grid
        except Exception:
            pass
    for (vrow, col), patch in zip(positions, patches):
        try:
            grid[vrow, col] = int(hsv_classifier.classify(patch))
        except Exception:
            grid[vrow, col] = COLOR_EMPTY
    return grid


def extract_frame_states(
    video_path: str,
    start_sec: float,
    end_sec: float,
    pipeline: StatePipeline,
    bg_fp_time: float,
) -> list[FrameState]:
    """0.1s ごとに frame を抽出し GameState + ChainPhase 判定 + CNN 確信度を返す。

    Z-3C: cap.set + cap.read を 0.1s 刻みで呼ぶ代わりに、連続 frame を
    順次読み込んで decode コストを削減 (read_frames_sequential)。
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"video open failed: {video_path}")
    if bg_fp_time >= 0:
        pipeline.set_background_fingerprints_from_video(cap, bg_fp_time)
    pipeline.reset(match_start_sec=start_sec)
    cap.release()  # set_background_fingerprints のためだけに開いた cap は閉じる

    classifier = getattr(pipeline._image_reader, "_classifier", None)
    hsv_classifier = getattr(classifier, "_hsv", None) if classifier else None

    chain_detector = ChainPhaseDetector()
    frames: list[FrameState] = []

    # 0.1s 刻みの時刻リストを作成し、frame_reader で連続読み込み
    from src.frame_reader import read_frames_sequential
    times = []
    t = start_sec
    while t <= end_sec + 1e-6:
        times.append(t)
        t += FRAME_INTERVAL_SEC
    samples = read_frames_sequential(video_path, times)

    for sample in samples:
        if sample is None:
            continue
        t = sample.t_sec
        frame = sample.frame
        gs = pipeline.extract(frame, t)
        chain_res = chain_detector.update(
            t, gs.board_p1, gs.board_p2, gs.score_p1, gs.score_p2,
        )
        # Z-1.6: CNN 確信度を取得 (calibrator を通した frame で評価)
        target = frame
        if (pipeline._calibrator is not None
                and pipeline._calibrator.n_calib_frames > 0):
            target = pipeline._calibrator.apply(frame)
        conf_p1 = extract_confidence_grid(
            target, DEFAULT_P1_REGION, classifier,
        )
        conf_p2 = extract_confidence_grid(
            target, DEFAULT_P2_REGION, classifier,
        )
        # Z-3C: HSV 化を 1 度だけ行い P1/P2 で共有
        hsv_full = cv2.cvtColor(target, cv2.COLOR_BGR2HSV)
        sat_p1, val_p1 = extract_hsv_grid(
            target, DEFAULT_P1_REGION, hsv_full=hsv_full,
        )
        sat_p2, val_p2 = extract_hsv_grid(
            target, DEFAULT_P2_REGION, hsv_full=hsv_full,
        )
        hsv_color_p1 = extract_hsv_color_grid(
            target, DEFAULT_P1_REGION, hsv_classifier,
        )
        hsv_color_p2 = extract_hsv_color_grid(
            target, DEFAULT_P2_REGION, hsv_classifier,
        )
        frames.append(FrameState(
            t_sec=t,
            board_p1=gs.board_p1.copy(),
            board_p2=gs.board_p2.copy(),
            next_p1=gs.next_p1, next_p2=gs.next_p2,
            dnext_p1=gs.dnext_p1, dnext_p2=gs.dnext_p2,
            score_p1=gs.score_p1, score_p2=gs.score_p2,
            pending_ojama_p1=gs.pending_ojama_p1,
            pending_ojama_p2=gs.pending_ojama_p2,
            is_chain_p1=chain_res.is_chain_p1,
            is_chain_p2=chain_res.is_chain_p2,
            predicted_p1=chain_res.predicted_p1,
            predicted_p2=chain_res.predicted_p2,
            frame_bgr=frame,
            conf_p1=conf_p1,
            conf_p2=conf_p2,
            sat_p1=sat_p1, sat_p2=sat_p2,
            val_p1=val_p1, val_p2=val_p2,
            hsv_color_p1=hsv_color_p1,
            hsv_color_p2=hsv_color_p2,
        ))
    return frames


# ============================
# 0.5s ごとの多数決
# ============================


def consolidate(frames: list[FrameState]) -> list[ConsolidatedState]:
    """5 frame ずつ window で多数決 → 確定盤面を生成。

    連鎖中 frame は予測盤面を採用し、それ以外は cell 単位 majority vote。
    CNN 確信度は window 平均を保持。
    """
    out: list[ConsolidatedState] = []
    for i in range(0, len(frames), WINDOW_SIZE):
        window = frames[i:i + WINDOW_SIZE]
        if not window:
            continue
        center = window[len(window) // 2]
        any_chain_p1 = any(f.is_chain_p1 for f in window)
        any_chain_p2 = any(f.is_chain_p2 for f in window)
        board_p1 = _majority_or_predicted(
            [f.board_p1 for f in window],
            [f.predicted_p1 for f in window if f.is_chain_p1 and f.predicted_p1 is not None],
        )
        board_p2 = _majority_or_predicted(
            [f.board_p2 for f in window],
            [f.predicted_p2 for f in window if f.is_chain_p2 and f.predicted_p2 is not None],
        )
        # Z-1.6: CNN 確信度の window 平均
        conf_p1 = _mean_grid([f.conf_p1 for f in window])
        conf_p2 = _mean_grid([f.conf_p2 for f in window])
        # Z-1.7: HSV S/V の window 平均
        sat_p1 = _mean_grid([f.sat_p1 for f in window])
        sat_p2 = _mean_grid([f.sat_p2 for f in window])
        val_p1 = _mean_grid([f.val_p1 for f in window])
        val_p2 = _mean_grid([f.val_p2 for f in window])
        # Z-1.8: HSV 判定 color の最頻値
        hsv_color_p1 = _mode_color_grid([f.hsv_color_p1 for f in window])
        hsv_color_p2 = _mode_color_grid([f.hsv_color_p2 for f in window])
        out.append(ConsolidatedState(
            t_sec=center.t_sec,
            board_p1=board_p1,
            board_p2=board_p2,
            next_p1=_mode_optional([f.next_p1 for f in window]),
            next_p2=_mode_optional([f.next_p2 for f in window]),
            dnext_p1=_mode_optional([f.dnext_p1 for f in window]),
            dnext_p2=_mode_optional([f.dnext_p2 for f in window]),
            score_p1=_last_non_none([f.score_p1 for f in window]),
            score_p2=_last_non_none([f.score_p2 for f in window]),
            pending_ojama_p1=window[-1].pending_ojama_p1,
            pending_ojama_p2=window[-1].pending_ojama_p2,
            is_chain_p1=any_chain_p1,
            is_chain_p2=any_chain_p2,
            conf_p1=conf_p1,
            conf_p2=conf_p2,
            sat_p1=sat_p1, sat_p2=sat_p2,
            val_p1=val_p1, val_p2=val_p2,
            hsv_color_p1=hsv_color_p1,
            hsv_color_p2=hsv_color_p2,
            representative_frame=center.frame_bgr,
        ))
    return out


def _mode_color_grid(grids: list) -> np.ndarray | None:
    """各 cell について最頻 color を返す (None grid は除外)。"""
    valid = [g for g in grids if g is not None]
    if not valid:
        return None
    out = np.zeros_like(valid[0])
    for vrow in range(out.shape[0]):
        for col in range(out.shape[1]):
            counter: Counter = Counter()
            for g in valid:
                counter[int(g[vrow, col])] += 1
            out[vrow, col] = counter.most_common(1)[0][0]
    return out


def _mean_grid(grids: list) -> np.ndarray | None:
    """None でない grid の cell 単位平均。全部 None なら None。"""
    valid = [g for g in grids if g is not None]
    if not valid:
        return None
    return np.mean(np.stack(valid), axis=0)


def _majority_or_predicted(
    boards: list[Board], predicted: list[Board],
) -> Board:
    """連鎖中の predicted があれば最新予測を採用、無ければ cell 単位多数決。"""
    if predicted:
        return predicted[-1].copy()
    return _cell_majority(boards)


def _cell_majority(boards: list[Board]) -> Board:
    """cell 単位で最頻色を採用。同点は最新値優先。"""
    if not boards:
        return Board()
    out = boards[-1].copy()
    for row in range(BOARD_ROWS):
        for col in range(BOARD_COLS):
            counter: Counter = Counter()
            for b in boards:
                counter[int(b.get(row, col))] += 1
            best_color, _ = counter.most_common(1)[0]
            out.set(row, col, best_color)
    return out


def _mode_optional(values: list) -> object | None:
    """None を除いた最頻値、無ければ None。"""
    filtered = [v for v in values if v is not None]
    if not filtered:
        return None
    counter: Counter = Counter(filtered)
    return counter.most_common(1)[0][0]


def _last_non_none(values: list[int | None]) -> int | None:
    for v in reversed(values):
        if v is not None:
            return v
    return None


# ============================
# Suspicious 判定
# ============================


def detect_suspicious(
    consolidated: list[ConsolidatedState],
    frames: list[FrameState],
) -> None:
    """consolidated[i].suspicious_p* を埋める。複数ルール OR。"""
    for i, cs in enumerate(consolidated):
        prev_cs = consolidated[i - 1] if i > 0 else None
        window_frames = frames[i * WINDOW_SIZE: (i + 1) * WINDOW_SIZE]
        cs.suspicious_p1 = _detect_side(
            cs.board_p1,
            prev_cs.board_p1 if prev_cs else None,
            cs.next_p1,
            cs.is_chain_p1,
            [f.board_p1 for f in window_frames],
            cs.conf_p1, cs.sat_p1, cs.val_p1, cs.hsv_color_p1,
        )
        cs.suspicious_p2 = _detect_side(
            cs.board_p2,
            prev_cs.board_p2 if prev_cs else None,
            cs.next_p2,
            cs.is_chain_p2,
            [f.board_p2 for f in window_frames],
            cs.conf_p2, cs.sat_p2, cs.val_p2, cs.hsv_color_p2,
        )


def _detect_side(
    cur: Board,
    prev: Board | None,
    next_pair: tuple[int, int] | None,
    is_chain: bool,
    window: list[Board],
    conf_grid: np.ndarray | None = None,
    sat_grid: np.ndarray | None = None,
    val_grid: np.ndarray | None = None,
    hsv_color_grid: np.ndarray | None = None,
) -> dict[tuple[int, int], list[str]]:
    """1 board あたりの suspicious cell + 理由リスト。

    連鎖中 (is_chain=True) は ChainSimulator 予測が正解扱いのため
    判定対象外 (空 dict を返す)。
    """
    suspicious: dict[tuple[int, int], list[str]] = {}
    if is_chain:
        return suspicious

    # Z-1.8: recognized=UNKNOWN は不確定マーカーとして常に review 対象
    for vrow in range(12):
        row = vrow + HIDDEN_ROWS
        for col in range(BOARD_COLS):
            if int(cur.get(row, col)) == COLOR_UNKNOWN:
                suspicious.setdefault((row, col), []).append("unknown_recognized")

    # Z-1.6: CNN 低確信度 cell
    if conf_grid is not None:
        for vrow in range(12):
            row = vrow + HIDDEN_ROWS
            for col in range(BOARD_COLS):
                conf = float(conf_grid[vrow, col])
                if conf < CNN_CONF_THRESHOLD:
                    suspicious.setdefault((row, col), []).append(
                        f"low_conf({conf:.2f})",
                    )

    # Z-1.7: EM/UNKNOWN 判定なのに HSV 中央 S が高い → 検出漏れの典型シグナル
    # CNN が間違って自信を持つケース (v18_m03 で頻発) を救う
    if sat_grid is not None and val_grid is not None:
        for vrow in range(12):
            row = vrow + HIDDEN_ROWS
            for col in range(BOARD_COLS):
                color = int(cur.get(row, col))
                if color not in (COLOR_EMPTY, COLOR_UNKNOWN):
                    continue
                s = float(sat_grid[vrow, col])
                v = float(val_grid[vrow, col])
                if s >= EM_SATURATION_THRESHOLD and v >= EM_VALUE_THRESHOLD:
                    suspicious.setdefault((row, col), []).append(
                        f"em_but_saturated(S={s:.0f},V={v:.0f})",
                    )
                # Z-1.8: OJM 候補 (低彩度 + 中域 V)。真の EM 背景より明るい灰色
                elif (s < OJM_S_MAX
                        and OJM_V_MIN <= v <= OJM_V_MAX):
                    suspicious.setdefault((row, col), []).append(
                        f"em_but_grayish(S={s:.0f},V={v:.0f})",
                    )

    # Z-1.8: HSV 単独判定と recognized が異なる color (両方が puyo クラス)
    # CNN が puyo 色を誤認しているケース (色 swap) を拾う
    if hsv_color_grid is not None:
        for vrow in range(12):
            row = vrow + HIDDEN_ROWS
            for col in range(BOARD_COLS):
                rec = int(cur.get(row, col))
                hsv = int(hsv_color_grid[vrow, col])
                # 両方が EMPTY/UNKNOWN ではない puyo 色で、かつ違う
                if (rec not in (COLOR_EMPTY, COLOR_UNKNOWN)
                        and hsv not in (COLOR_EMPTY, COLOR_UNKNOWN)
                        and rec != hsv):
                    suspicious.setdefault((row, col), []).append(
                        f"hsv_disagree(cnn={COLOR_LABEL[rec]},"
                        f"hsv={COLOR_LABEL[hsv]})",
                    )

    # ルール 1: 浮遊 (直下 EMPTY なのに puyo) — 検出漏れ or 誤検出の典型
    for row in range(HIDDEN_ROWS, BOARD_ROWS - 1):
        for col in range(BOARD_COLS):
            color = int(cur.get(row, col))
            if color in (COLOR_EMPTY, COLOR_UNKNOWN):
                continue
            below = int(cur.get(row + 1, col))
            if below == COLOR_EMPTY:
                suspicious.setdefault((row, col), []).append("airborne")

    # ルール 4: hidden_below (浮遊の逆) — 直上に色があるのに自身 EM
    # 重力違反による検出漏れの典型
    for row in range(HIDDEN_ROWS + 1, BOARD_ROWS):
        for col in range(BOARD_COLS):
            color = int(cur.get(row, col))
            if color != COLOR_EMPTY:
                continue
            above = int(cur.get(row - 1, col))
            if above not in (COLOR_EMPTY, COLOR_UNKNOWN):
                suspicious.setdefault((row, col), []).append(
                    f"hidden_below(above={COLOR_LABEL[above]})",
                )

    # ルール 5: empty_in_stack — 同列の上下両方に色があり間に EM (穴)
    # スタック中の穴は重力で潰れるはずなので、検出漏れの可能性大
    for col in range(BOARD_COLS):
        # 列の最上端 puyo 行を探す
        top_filled = -1
        for row in range(HIDDEN_ROWS, BOARD_ROWS):
            if int(cur.get(row, col)) not in (COLOR_EMPTY, COLOR_UNKNOWN):
                top_filled = row
                break
        if top_filled < 0:
            continue
        # top_filled 以下で EM があれば穴
        for row in range(top_filled + 1, BOARD_ROWS):
            if int(cur.get(row, col)) == COLOR_EMPTY:
                suspicious.setdefault((row, col), []).append("empty_in_stack")

    if prev is not None:
        # ルール 1b: 1 cell だけ新規出現 (ペアは 2 cell 同時)
        new_cells: list[tuple[int, int]] = []
        for row in range(HIDDEN_ROWS, BOARD_ROWS):
            for col in range(BOARD_COLS):
                p = int(prev.get(row, col))
                c = int(cur.get(row, col))
                if p in (COLOR_EMPTY, COLOR_UNKNOWN) and c not in (
                    COLOR_EMPTY, COLOR_UNKNOWN,
                ):
                    new_cells.append((row, col))
        if len(new_cells) == 1:
            suspicious.setdefault(new_cells[0], []).append("solo_appearance")

        # ルール 1c: 色 → 別色直接遷移
        for row in range(HIDDEN_ROWS, BOARD_ROWS):
            for col in range(BOARD_COLS):
                p = int(prev.get(row, col))
                c = int(cur.get(row, col))
                if (p not in (COLOR_EMPTY, COLOR_UNKNOWN, COLOR_OJAMA)
                        and c not in (COLOR_EMPTY, COLOR_UNKNOWN, COLOR_OJAMA)
                        and p != c):
                    suspicious.setdefault((row, col), []).append(
                        f"color_swap({COLOR_LABEL[p]}->{COLOR_LABEL[c]})",
                    )

        # ルール 6: disappearance (prev=色 → cur=EM)
        # 連鎖以外で puyo が消えるのは検出漏れの典型
        for row in range(HIDDEN_ROWS, BOARD_ROWS):
            for col in range(BOARD_COLS):
                p = int(prev.get(row, col))
                c = int(cur.get(row, col))
                if (p not in (COLOR_EMPTY, COLOR_UNKNOWN)
                        and c == COLOR_EMPTY):
                    suspicious.setdefault((row, col), []).append(
                        f"disappearance({COLOR_LABEL[p]}->EM)",
                    )

        # ルール 7: unknown_drop (prev=色 → cur=??)
        for row in range(HIDDEN_ROWS, BOARD_ROWS):
            for col in range(BOARD_COLS):
                p = int(prev.get(row, col))
                c = int(cur.get(row, col))
                if (p not in (COLOR_EMPTY, COLOR_UNKNOWN)
                        and c == COLOR_UNKNOWN):
                    suspicious.setdefault((row, col), []).append(
                        f"unknown_drop({COLOR_LABEL[p]}->??)",
                    )

    # ルール 2: ペア色不整合 (新規 cell が prev next_pair の色集合に含まれない)
    if prev is not None and next_pair is not None:
        allowed = set(next_pair)
        for row in range(HIDDEN_ROWS, BOARD_ROWS):
            for col in range(BOARD_COLS):
                p = int(prev.get(row, col))
                c = int(cur.get(row, col))
                if (p in (COLOR_EMPTY, COLOR_UNKNOWN)
                        and c not in (COLOR_EMPTY, COLOR_UNKNOWN, COLOR_OJAMA)
                        and c not in allowed):
                    suspicious.setdefault((row, col), []).append(
                        f"pair_mismatch(next={next_pair}, got={COLOR_LABEL[c]})",
                    )

    # ルール 3: フリッカー (window 5 frame 内で 2 色以上に振動、EMPTY 含めず)
    if len(window) >= 3:
        for row in range(HIDDEN_ROWS, BOARD_ROWS):
            for col in range(BOARD_COLS):
                colors = {int(b.get(row, col)) for b in window}
                colors.discard(COLOR_EMPTY)
                colors.discard(COLOR_UNKNOWN)
                if len(colors) >= 2:
                    suspicious.setdefault((row, col), []).append(
                        f"flicker({len(colors)}colors)",
                    )

    return suspicious


# ============================
# 描画
# ============================


def crop_field(
    frame: np.ndarray, region, pad: int = 4,
) -> tuple[np.ndarray, tuple[int, int]]:
    h, w = frame.shape[:2]
    x1, y1, _, _ = region.cell_sample_rect(HIDDEN_ROWS, 0)
    _, _, x2, y2 = region.cell_sample_rect(HIDDEN_ROWS + 11, BOARD_COLS - 1)
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad)
    y2 = min(h, y2 + pad)
    return frame[y1:y2, x1:x2].copy(), (x1, y1)


def annotate_field(
    field_img: np.ndarray,
    region,
    board: Board,
    side: str,
    frame_xy: tuple[int, int],
    suspicious: dict[tuple[int, int], list[str]],
) -> np.ndarray:
    out = field_img.copy()
    x_off, y_off = frame_xy
    for vrow in range(12):
        row = vrow + HIDDEN_ROWS
        for col in range(BOARD_COLS):
            color = int(board.get(row, col))
            x1, y1, x2, y2 = region.cell_sample_rect(row, col)
            x1, x2 = x1 - x_off, x2 - x_off
            y1, y2 = y1 - y_off, y2 - y_off
            border = COLOR_BORDER.get(color, (60, 60, 60))
            label = COLOR_LABEL.get(color, "?")
            sus_reasons = suspicious.get((row, col))
            if sus_reasons:
                # suspicious cell は 太い黄色枠 + ラベルを赤系に
                cv2.rectangle(out, (x1, y1), (x2, y2), SUSPICIOUS_BORDER, 3)
            elif color != COLOR_EMPTY:
                cv2.rectangle(out, (x1, y1), (x2, y2), border, 2)
            else:
                cv2.rectangle(out, (x1, y1), (x2, y2), (40, 40, 40), 1)
            if color != COLOR_EMPTY and label:
                (tw, th), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_DUPLEX, 0.6, 1,
                )
                tx = x1 + 2
                ty = y1 + th + 2
                cv2.rectangle(
                    out, (tx - 1, ty - th - 1),
                    (tx + tw + 1, ty + 2), (0, 0, 0), -1,
                )
                cv2.putText(
                    out, label, (tx, ty),
                    cv2.FONT_HERSHEY_DUPLEX, 0.6, border, 1, cv2.LINE_AA,
                )
    cv2.putText(
        out, side, (4, 18),
        cv2.FONT_HERSHEY_DUPLEX, 0.7, (240, 240, 240), 1, cv2.LINE_AA,
    )
    return out


def render_consolidated_overview(cs: ConsolidatedState) -> np.ndarray:
    """1 つの 0.5s 確定状態を 1 枚の PNG にレンダ。

    レイアウト: P1 field | info banner | P2 field
    """
    if cs.representative_frame is None:
        return np.zeros((100, 100, 3), dtype=np.uint8)
    p1_img, p1_off = crop_field(cs.representative_frame, DEFAULT_P1_REGION)
    p2_img, p2_off = crop_field(cs.representative_frame, DEFAULT_P2_REGION)
    p1_ann = annotate_field(
        p1_img, DEFAULT_P1_REGION, cs.board_p1, "1P", p1_off, cs.suspicious_p1,
    )
    p2_ann = annotate_field(
        p2_img, DEFAULT_P2_REGION, cs.board_p2, "2P", p2_off, cs.suspicious_p2,
    )
    h = max(p1_ann.shape[0], p2_ann.shape[0])
    info_w = 280
    info = np.full((h, info_w, 3), 30, dtype=np.uint8)
    lines = [
        f"t={cs.t_sec:.2f}s",
        f"chain1={cs.is_chain_p1} 2={cs.is_chain_p2}",
        f"next1={cs.next_p1}",
        f"dnxt1={cs.dnext_p1}",
        f"next2={cs.next_p2}",
        f"dnxt2={cs.dnext_p2}",
        f"score1={cs.score_p1}",
        f"score2={cs.score_p2}",
        f"pend1={cs.pending_ojama_p1}",
        f"pend2={cs.pending_ojama_p2}",
        f"sus1={len(cs.suspicious_p1)} sus2={len(cs.suspicious_p2)}",
    ]
    for k, line in enumerate(lines):
        cv2.putText(
            info, line, (8, 18 + k * 22),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA,
        )
    p1_pad = _pad_to_height(p1_ann, h)
    p2_pad = _pad_to_height(p2_ann, h)
    return np.hstack([p1_pad, info, p2_pad])


def _pad_to_height(img: np.ndarray, h: int) -> np.ndarray:
    if img.shape[0] >= h:
        return img
    pad = np.full((h - img.shape[0], img.shape[1], 3), 30, dtype=np.uint8)
    return np.vstack([img, pad])


def render_field_sheet(images: list[np.ndarray], cols: int = 2) -> np.ndarray:
    """確定状態 PNG の grid 配置。"""
    if not images:
        return np.zeros((100, 100, 3), dtype=np.uint8)
    rows = (len(images) + cols - 1) // cols
    cell_w = max(img.shape[1] for img in images)
    cell_h = max(img.shape[0] for img in images)
    sheet = np.full(
        (rows * cell_h, cols * cell_w, 3), 18, dtype=np.uint8,
    )
    for k, img in enumerate(images):
        r = k // cols
        c = k % cols
        ih, iw = img.shape[:2]
        sheet[r * cell_h: r * cell_h + ih, c * cell_w: c * cell_w + iw] = img
    return sheet


# ============================
# CSV / HTML 出力
# ============================


def write_labels_csv(
    consolidated: list[ConsolidatedState], path: Path,
) -> int:
    """全 cell の (id, t, side, row, col, recognized, your_answer, conf, is_chain, suspicious_reasons)。"""
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "id", "time", "side", "row", "col",
            "recognized", "your_answer", "conf",
            "is_chain", "suspicious_reasons",
        ])
        sample_id = 0
        for cs in consolidated:
            for side, board, sus, is_chain, conf in (
                ("1P", cs.board_p1, cs.suspicious_p1, cs.is_chain_p1, cs.conf_p1),
                ("2P", cs.board_p2, cs.suspicious_p2, cs.is_chain_p2, cs.conf_p2),
            ):
                for vrow in range(12):
                    row = vrow + HIDDEN_ROWS
                    for col in range(BOARD_COLS):
                        color = int(board.get(row, col))
                        sample_id += 1
                        reasons = ";".join(sus.get((row, col), []))
                        conf_val = (
                            f"{float(conf[vrow, col]):.3f}"
                            if conf is not None else ""
                        )
                        writer.writerow([
                            sample_id, f"{cs.t_sec:.2f}", side,
                            vrow, col,
                            COLOR_FULL.get(color, "?"),
                            "",
                            conf_val,
                            "1" if is_chain else "0",
                            reasons,
                        ])
    return sample_id


def write_suspicious_csv(
    consolidated: list[ConsolidatedState], path: Path,
) -> int:
    """suspicious cells のみ抽出した CSV。レビュー対象を絞り込む。"""
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "id", "time", "side", "row", "col",
            "recognized", "your_answer", "reasons",
        ])
        sample_id = 0
        for cs in consolidated:
            for side, board, sus in (
                ("1P", cs.board_p1, cs.suspicious_p1),
                ("2P", cs.board_p2, cs.suspicious_p2),
            ):
                for (row, col), reasons in sus.items():
                    sample_id += 1
                    color = int(board.get(row, col))
                    writer.writerow([
                        sample_id, f"{cs.t_sec:.2f}", side,
                        row - HIDDEN_ROWS, col,
                        COLOR_FULL.get(color, "?"),
                        "",
                        ";".join(reasons),
                    ])
    return sample_id


def write_summary_html(
    out_dir: Path,
    consolidated: list[ConsolidatedState],
    n_total: int, n_suspicious: int,
) -> None:
    html = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'>",
        "<title>Phase Z-1 Review</title>",
        "<style>body{font-family:sans-serif;background:#222;color:#ddd;}",
        "img{max-width:100%;border:1px solid #444;margin:4px;}",
        "table{border-collapse:collapse;}",
        "td,th{border:1px solid #444;padding:4px 8px;}</style>",
        "</head><body>",
        "<h1>Phase Z-1 Review UI</h1>",
        f"<p>確定状態数: {len(consolidated)} / 全 cell: {n_total} / suspicious: {n_suspicious}</p>",
        "<h2>Quick links</h2>",
        "<ul>",
        "<li><a href='field_sheet.png'>field_sheet.png (全確定状態)</a></li>",
        "<li><a href='labels.csv'>labels.csv (全 cell)</a></li>",
        "<li><a href='suspicious.csv'>suspicious.csv (要レビュー cell のみ)</a></li>",
        "</ul>",
        "<h2>Frames (0.5s ごと)</h2>",
        "<table>",
        "<tr><th>t (s)</th><th>chain</th><th>sus 1P</th><th>sus 2P</th><th>image</th></tr>",
    ]
    for cs in consolidated:
        ms = int(cs.t_sec * 1000)
        chain_str = (
            "P1+P2" if cs.is_chain_p1 and cs.is_chain_p2
            else "P1" if cs.is_chain_p1
            else "P2" if cs.is_chain_p2 else ""
        )
        html.append(
            f"<tr><td>{cs.t_sec:.2f}</td><td>{chain_str}</td>"
            f"<td>{len(cs.suspicious_p1)}</td>"
            f"<td>{len(cs.suspicious_p2)}</td>"
            f"<td><a href='frames/{ms:06d}.png'>"
            f"<img src='frames/{ms:06d}.png' style='max-width:600px'></a></td></tr>",
        )
    html.append("</table></body></html>")
    (out_dir / "summary.html").write_text(
        "\n".join(html), encoding="utf-8",
    )


# ============================
# main
# ============================


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--start", type=float, required=True)
    parser.add_argument("--end", type=float, required=True)
    parser.add_argument("--bg-fp-time", type=float, default=-1.0)
    parser.add_argument(
        "--cnn-model", default="models/cnn_phase_u_v16.pt",
    )
    parser.add_argument(
        "--no-refiners", action="store_true",
        help="補正レイヤーを全 OFF (CNN+HSV のみで生 accuracy 確認)",
    )
    parser.add_argument(
        "--use-online-hsv", action="store_true",
        help="Z-3I: 試合中に動画別 HSV 範囲を自動学習 (未知動画対応)",
    )
    parser.add_argument(
        "--use-cell-anomaly", action="store_true",
        help="Z-3J: cell hash anomaly 検出で連鎖アニメ・落下中の不安定 cell 救済",
    )
    parser.add_argument(
        "--use-hsv-anomaly", action="store_true",
        help="Z-3J': HSV mean anomaly (pHash 版の改良、自然変動許容)",
    )
    parser.add_argument(
        "--ensemble", action="store_true",
        help="Z-X: v16+v17b の Multi-CNN ensemble",
    )
    parser.add_argument(
        "--auto-roi", action="store_true",
        help="D: 試合開始 frame から ROI offset を auto 算出して補正",
    )
    parser.add_argument(
        "--use-connectivity", action="store_true",
        help="A: 孤立 cell を周囲色に補正",
    )
    parser.add_argument(
        "--use-stability", action="store_true",
        help="G: cell HSV σ で不安定検出 + 補正",
    )
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "frames").mkdir(exist_ok=True)

    print(f"[Z-1] video={args.video} {args.start:.1f}-{args.end:.1f}s")
    pipeline = build_pipeline(
        cnn_model=args.cnn_model,
        all_refiners_on=not args.no_refiners,
        use_online_hsv=args.use_online_hsv,
        use_cell_anomaly=args.use_cell_anomaly,
        use_hsv_anomaly=args.use_hsv_anomaly,
        use_connectivity_outlier=args.use_connectivity,
        use_stability=args.use_stability,
    )
    # Z-X: Multi-CNN ensemble に差し替え
    if args.ensemble:
        from src.multi_cnn_ensemble import load_ensemble_v16_v17b
        from src.hybrid_classifier import HybridClassifier
        ensemble = load_ensemble_v16_v17b()
        new_hybrid = HybridClassifier(cnn_classifier=ensemble)
        pipeline._image_reader._classifier = new_hybrid
        print("[Z-X] ensemble (v16+v17b) を classifier に注入")

    # D: ROI auto-calibration
    if args.auto_roi:
        from dataclasses import replace
        from src.roi_auto_calibrator import detect_roi_offsets
        cap = cv2.VideoCapture(args.video)
        cap.set(cv2.CAP_PROP_POS_MSEC, args.bg_fp_time * 1000)
        ok, frame = cap.read()
        cap.release()
        if ok and frame is not None:
            if frame.shape[:2] != (1080, 1920):
                frame = cv2.resize(
                    frame, (1920, 1080), interpolation=cv2.INTER_AREA,
                )
            calib = detect_roi_offsets(frame)
            ir = pipeline._image_reader
            ir._p1_region = replace(
                ir._p1_region,
                x=ir._p1_region.x + calib.p1_offset[0],
                y=ir._p1_region.y + calib.p1_offset[1],
            )
            ir._p2_region = replace(
                ir._p2_region,
                x=ir._p2_region.x + calib.p2_offset[0],
                y=ir._p2_region.y + calib.p2_offset[1],
            )
            print(
                f"[D] ROI offset P1={calib.p1_offset} "
                f"P2={calib.p2_offset} conf={calib.confidence:.2f}"
            )
    print("[Z-1] pipeline ready")

    frames = extract_frame_states(
        args.video, args.start, args.end, pipeline, args.bg_fp_time,
    )
    print(f"[Z-1] extracted {len(frames)} frames @ 0.1s")

    consolidated = consolidate(frames)
    print(f"[Z-1] consolidated to {len(consolidated)} states @ 0.5s")

    detect_suspicious(consolidated, frames)
    n_sus = sum(
        len(cs.suspicious_p1) + len(cs.suspicious_p2)
        for cs in consolidated
    )
    print(f"[Z-1] suspicious cells: {n_sus}")

    overview_imgs: list[np.ndarray] = []
    for cs in consolidated:
        ms = int(cs.t_sec * 1000)
        img = render_consolidated_overview(cs)
        cv2.imwrite(str(out_dir / "frames" / f"{ms:06d}.png"), img)
        overview_imgs.append(img)

    sheet = render_field_sheet(overview_imgs, cols=2)
    cv2.imwrite(str(out_dir / "field_sheet.png"), sheet)
    print(f"[Z-1] saved: {to_windows_path(out_dir / 'field_sheet.png')}")

    n_total = write_labels_csv(consolidated, out_dir / "labels.csv")
    n_suspicious_rows = write_suspicious_csv(
        consolidated, out_dir / "suspicious.csv",
    )
    write_summary_html(out_dir, consolidated, n_total, n_suspicious_rows)

    print(f"[Z-1] labels.csv ({n_total} cells)")
    print(f"[Z-1] suspicious.csv ({n_suspicious_rows} cells)")
    print(f"[Z-1] summary: {to_windows_path(out_dir / 'summary.html')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
