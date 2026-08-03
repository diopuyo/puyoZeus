"""エフェクト視覚特性調査 (2026-08-03、案B実現可能性の調査専用スクリプト)。

満杯盤面2.56%誤りの主犯 (色→おじゃま誤検出55%、memory
project_full_board_error_taxonomy_2026-08-02) の背後にある2種のエフェクト:
  (a) 予告おじゃま送付バースト: 相手が連鎖する毎に、受け手側盤面の上段
      (row1-3) を発光が覆う。長連鎖中は実質数秒持続。
  (b) お邪魔落下の白煙: おじゃまが着地する列を白煙が遮蔽する。
を、セル単位の HSV 画素統計で検出できるか (= 案B の実現可能性) を実データで
判定する。本スクリプトは **調査専用** であり、検出器の実装は行わない。

収集方法:
  1. バースト窓: data/indicators_v2/exchange_labels_regen_step3_placement_
     2026-08-02.csv (Step3修正済・実データ由来の発火イベント、fire_side が
     連鎖した側) から、連鎖規模3層 (2-3/4-6/7+) 各10窓を抽出する。
     バーストは「相手側盤面」(fire_side の相手側 = opp_side) に出るという
     タスク前提に基づき、opp_side の盤面を fire event 終了時刻(t_sec、
     memory project_fire_event_fragmentation_2026-08-02 で「連鎖終了時点」と
     確定済) から遡った推定連鎖継続中の時刻でサンプルする。
  2. 煙窓: data/indicators_v2/boards_lean_regen_2026-07-31 npz の grids から
     おじゃまセル数が増加した直後 (前後 STABLE snapshot の中間時刻) を
     10 窓抽出する (reference_ojama_landing_gated_by_placement に基づき、
     おじゃま増加は着地直後にのみ起こる正当なイベント)。
  3. 対照群: 発火イベント・おじゃま増加のどちらからも十分離れた STABLE
     snapshot を20窓抽出する (ground truth grid既知)。

出力: data/verify/effect_signature_study_2026-08-03/
  - cell_stats.csv          : 層別セル画素統計 全件
  - separability_report.md  : 数表 + 分離可能性の結論
  - dist_<feature>.png      : 特徴量ヒストグラム (層別)
  - tile_<layer>.png        : 層別代表セル切り出しタイル (目視確認用)

Usage:
    PYTHONPATH=. ./venv/bin/python -m scripts.study_effect_signature_2026-08-03
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PROJ_ROOT = Path(__file__).resolve().parent.parent
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

from src.board import (  # noqa: E402
    BOARD_COLS, BOARD_ROWS, COLOR_EMPTY, COLOR_OJAMA, COLOR_UNKNOWN, HIDDEN_ROWS,
)
from src.chain_count_ocr import _ensure_1080p  # noqa: E402
from src.image_reader import (  # noqa: E402
    DEFAULT_P1_REGION, DEFAULT_P2_REGION, SPECULAR_S_MAX, SPECULAR_V_MIN, BoardRegion,
)
from src.indicators_v2 import CHAIN_ANIM_PER_STEP_SEC  # noqa: E402

# =============================================================================
# 定数
# =============================================================================

OUTPUT_DIR: Path = PROJ_ROOT / "data" / "verify" / "effect_signature_study_2026-08-03"
FIRE_EVENTS_CSV: Path = (
    PROJ_ROOT / "data" / "indicators_v2"
    / "exchange_labels_regen_step3_placement_2026-08-02.csv"
)
NPZ_DIR: Path = PROJ_ROOT / "data" / "indicators_v2" / "boards_lean_regen_2026-07-31"

# ローカル動画キャッシュ探索先 (WSLネイティブ優先、無ければリポジトリ相対)。
# scripts/extract_exchange_event_frames.py の resolve_cached_video_path と同じ設計。
WSL_NATIVE_FRAMES_DIR: Path = Path("/home/ryouj/frames")
REPO_FRAMES_DIR: Path = PROJ_ROOT / "data" / "frames"

# 調査対象動画 (バースト参照4本 [c18/c5/c29/c19] + 多様性確保の追加4本)。
# 全て npz + mp4 キャッシュ存在確認済 (2026-08-03 事前調査)。
CANDIDATE_VIDEO_STEMS: tuple[str, ...] = (
    "c18", "c5", "c29", "c19", "c10", "c11", "c20", "c21",
)

# バースト連鎖規模ビン (下限, 上限, ラベル)。1連鎖は対象外 (バースト自体が
# 弱いと想定されるため、タスク指定の3層のみ収集する)。
CHAIN_MAGNITUDE_BINS: tuple[tuple[int, int, str], ...] = (
    (2, 3, "2-3"),
    (4, 6, "4-6"),
    (7, 999, "7+"),
)
BURST_WINDOWS_PER_BIN: int = 10
SMOKE_WINDOWS_TOTAL: int = 10
BASELINE_WINDOWS_TOTAL: int = 20

# バースト窓の行帯 (全board行index、HIDDEN_ROWS込み)。タスク前提「上段row1-3」。
BURST_ROW_MIN: int = 1
BURST_ROW_MAX: int = 3

# バースト継続時間推定: 実測 (memory project_exchange_measurement_foundation:
# 8連鎖で実測14.5秒 vs CHAIN_ANIM_PER_STEP_SEC(0.4)*8=3.2秒の素朴推定) は
# 素朴推定の約4.5倍長い。安全側 (見逃しよりバースト内に収まる方を優先) に
# 倍率3.0を掛けて推定する。本調査専用の粗い推定であり、本番指標には使わない。
BURST_MIN_DURATION_SEC: float = 1.0
BURST_DURATION_SAFETY_MULTIPLIER: float = 3.0
# 推定継続時間のうち何割戻った時刻をサンプルするか (0.5 = 連鎖中央付近)。
BURST_SAMPLE_FRACTION: float = 0.5

# おじゃま増加イベント検出: 直前スナップショットとの時間差がこれを超えたら
# 試合境界またぎ等の異常とみなし除外する (秒)。
SMOKE_MAX_GAP_SEC: float = 15.0
SMOKE_MIN_OJAMA_DELTA: int = 1
SMOKE_MAX_OJAMA_DELTA: int = 36  # 可視領域72セルの半分、異常値除外用

# 対照群サンプリング: 発火/おじゃま増加イベントの前後この秒数以内は
# 「平穏」とみなさず除外する。
BASELINE_EXCLUSION_SEC: float = 5.0

# 光沢ハイライト疑い判定 (SPECULAR_* 閾値で計算した白飛び率がこれ以上)。
HIGHLIGHT_SUSPECT_RATIO_THRESHOLD: float = 0.15

# 高輝度率判定 (V >= この値)。_diag_effect_glow_hsv_2026-07-31.py の
# "高輝度率" 指標と同一定義を再利用。
BRIGHT_V_THRESHOLD: int = 230

# 乱数シード (再現性確保)。
RANDOM_SEED: int = 20260803

# タイル画像の1セルあたり表示サイズ (px)。
TILE_CELL_SIZE: int = 96
TILE_COLS: int = 5
TILE_MAX_PER_LAYER: int = 10


# =============================================================================
# データ構造
# =============================================================================


@dataclass
class CellRecord:
    """1セル分の画素統計 + メタデータ。"""

    video_stem: str
    side: str
    row: int
    col: int
    t_sec: float
    layer: str  # "burst" / "smoke" / "normal" / "ojama" / "empty"
    chain_bin: str  # burst のみ、他は ""
    ground_truth_color: int  # baseline/smoke は既知、burst は -1 (不明)
    v_mean: float
    v_max: float
    s_mean: float
    s_min: float
    specular_ratio: float
    bright_ratio: float
    highlight_suspect: bool
    patch_bgr: "np.ndarray" = field(repr=False, default=None)


@dataclass
class _NpzSideIndex:
    """1 (video, side) 分の t_sec 昇順ソート済みインデックス。"""

    t_secs: "np.ndarray"
    grids: "np.ndarray"
    game_idxs: "np.ndarray"


# =============================================================================
# 1. 動画・npz アクセス
# =============================================================================


def resolve_video_path(stem: str) -> "Path | None":
    """video_<stem>.mp4 のローカルキャッシュパスを返す (WSLネイティブ優先)。"""
    for base_dir in (WSL_NATIVE_FRAMES_DIR, REPO_FRAMES_DIR):
        candidate = base_dir / f"video_{stem}.mp4"
        if candidate.exists():
            return candidate
    return None


class FrameCache:
    """動画毎に cv2.VideoCapture を使い回すキャッシュ (複数時刻の grab を高速化)。"""

    def __init__(self) -> None:
        self._caps: dict[str, cv2.VideoCapture] = {}

    def get_frame(self, stem: str, t_sec: float) -> "np.ndarray | None":
        cap = self._caps.get(stem)
        if cap is None:
            path = resolve_video_path(stem)
            if path is None:
                return None
            cap = cv2.VideoCapture(str(path))
            self._caps[stem] = cap
        cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, t_sec) * 1000.0)
        ok, frame = cap.read()
        if not ok or frame is None:
            return None
        return _ensure_1080p(frame)

    def release_all(self) -> None:
        for cap in self._caps.values():
            cap.release()
        self._caps.clear()


def load_npz_side_index(stem: str, side: str) -> "_NpzSideIndex | None":
    """1 (video, side) の t_sec 昇順ソート済み grids/game_idx を返す。"""
    npz_path = NPZ_DIR / f"{stem}.npz"
    if not npz_path.exists():
        return None
    data = np.load(npz_path, allow_pickle=True)
    mask = data["side"] == side
    if not mask.any():
        return None
    order = np.argsort(data["t_sec"][mask])
    return _NpzSideIndex(
        t_secs=data["t_sec"][mask][order].astype(np.float64),
        grids=data["grids"][mask][order],
        game_idxs=data["game_idx"][mask][order].astype(np.int64),
    )


def region_for_side(side: str) -> BoardRegion:
    return DEFAULT_P1_REGION if side == "1P" else DEFAULT_P2_REGION


# =============================================================================
# 2. 画素統計
# =============================================================================


def compute_cell_features(patch_bgr: "np.ndarray") -> dict[str, float]:
    """セル切り出し画像 (BGR) から HSV 統計特徴量を計算する (stateless)。"""
    hsv = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2HSV)
    v_ch, s_ch = hsv[:, :, 2], hsv[:, :, 1]
    specular_mask = (v_ch >= SPECULAR_V_MIN) & (s_ch <= SPECULAR_S_MAX)
    return {
        "v_mean": float(np.mean(v_ch)),
        "v_max": float(np.max(v_ch)),
        "s_mean": float(np.mean(s_ch)),
        "s_min": float(np.min(s_ch)),
        "specular_ratio": float(np.mean(specular_mask)),
        "bright_ratio": float(np.mean(v_ch >= BRIGHT_V_THRESHOLD)),
    }


def extract_cell_record(
    frame: "np.ndarray", region: BoardRegion, video_stem: str, side: str,
    row: int, col: int, t_sec: float, layer: str, chain_bin: str,
    ground_truth_color: int,
) -> CellRecord:
    """1セル分の切り出し + 特徴量計算 + CellRecord 構築 (stateless純関数)。"""
    x1, y1, x2, y2 = region.cell_sample_rect(row, col)
    patch = frame[y1:y2, x1:x2]
    feats = compute_cell_features(patch)
    highlight_suspect = (
        layer == "normal" and feats["specular_ratio"] >= HIGHLIGHT_SUSPECT_RATIO_THRESHOLD
    )
    return CellRecord(
        video_stem=video_stem, side=side, row=row, col=col, t_sec=t_sec,
        layer=layer, chain_bin=chain_bin, ground_truth_color=ground_truth_color,
        highlight_suspect=highlight_suspect, patch_bgr=patch.copy(), **feats,
    )


# =============================================================================
# 3. バースト窓収集
# =============================================================================


def estimate_burst_duration_sec(chain_len: float) -> float:
    """連鎖規模からバースト継続時間を粗く推定する (調査専用、本番指標に流用しない)。"""
    naive = chain_len * CHAIN_ANIM_PER_STEP_SEC * BURST_DURATION_SAFETY_MULTIPLIER
    return max(BURST_MIN_DURATION_SEC, naive)


def _nominal_color_hint(idx: "_NpzSideIndex | None", t_sec: float, row: int, col: int) -> int:
    """sample_t 以前で直近の STABLE grid における (row,col) 色を返す (参考値、非正解)。"""
    if idx is None:
        return COLOR_UNKNOWN
    cand = np.where(idx.t_secs <= t_sec)[0]
    if len(cand) == 0:
        return COLOR_UNKNOWN
    return int(idx.grids[cand[-1], row, col])


def collect_burst_samples(
    fire_df: pd.DataFrame, frame_cache: FrameCache, rng: np.random.Generator,
) -> tuple[list[CellRecord], dict[str, int]]:
    """連鎖規模3層それぞれ BURST_WINDOWS_PER_BIN 窓を収集する。"""
    records: list[CellRecord] = []
    window_counts: dict[str, int] = {}
    npz_cache: dict[tuple[str, str], "_NpzSideIndex | None"] = {}
    for lo, hi, label in CHAIN_MAGNITUDE_BINS:
        pool = fire_df[
            (fire_df["approx_fire_chains"] >= lo) & (fire_df["approx_fire_chains"] <= hi)
        ].sample(frac=1.0, random_state=rng.integers(0, 2**31 - 1)).reset_index(drop=True)
        n_ok = 0
        for _, ev in pool.iterrows():
            if n_ok >= BURST_WINDOWS_PER_BIN:
                break
            stem = str(ev["video_id"]).replace("video_", "")
            fire_side = str(ev["fire_side"])
            opp_side = "2P" if fire_side == "1P" else "1P"
            duration = estimate_burst_duration_sec(float(ev["approx_fire_chains"]))
            sample_t = max(0.0, float(ev["t_sec"]) - duration * BURST_SAMPLE_FRACTION)
            frame = frame_cache.get_frame(stem, sample_t)
            if frame is None:
                continue
            key = (stem, opp_side)
            if key not in npz_cache:
                npz_cache[key] = load_npz_side_index(stem, opp_side)
            region = region_for_side(opp_side)
            for row in range(BURST_ROW_MIN, BURST_ROW_MAX + 1):
                for col in range(BOARD_COLS):
                    hint = _nominal_color_hint(npz_cache[key], sample_t, row, col)
                    records.append(extract_cell_record(
                        frame, region, stem, opp_side, row, col, sample_t,
                        "burst", label, hint,
                    ))
            n_ok += 1
        window_counts[label] = n_ok
    return records, window_counts


# =============================================================================
# 4. 煙窓収集
# =============================================================================


def find_ojama_increase_events(
    idx: "_NpzSideIndex",
) -> list[tuple[float, "np.ndarray"]]:
    """おじゃまセル数増加イベント (中間時刻, 増加した列マスク) の一覧を返す。"""
    visible = idx.grids[:, HIDDEN_ROWS:BOARD_ROWS, :]
    col_counts = np.sum(visible == COLOR_OJAMA, axis=1)  # (N, BOARD_COLS)
    total_counts = np.sum(col_counts, axis=1)
    events: list[tuple[float, "np.ndarray"]] = []
    for i in range(1, len(idx.t_secs)):
        delta = int(total_counts[i] - total_counts[i - 1])
        gap = float(idx.t_secs[i] - idx.t_secs[i - 1])
        same_game = idx.game_idxs[i] == idx.game_idxs[i - 1]
        if not same_game or gap > SMOKE_MAX_GAP_SEC:
            continue
        if not (SMOKE_MIN_OJAMA_DELTA <= delta <= SMOKE_MAX_OJAMA_DELTA):
            continue
        t_mid = float((idx.t_secs[i] + idx.t_secs[i - 1]) / 2.0)
        changed_cols = np.where(col_counts[i] > col_counts[i - 1])[0]
        events.append((t_mid, changed_cols))
    return events


def collect_smoke_samples(
    frame_cache: FrameCache, rng: np.random.Generator,
) -> tuple[list[CellRecord], int]:
    """おじゃま増加直後の該当列を SMOKE_WINDOWS_TOTAL 窓収集する。"""
    candidates: list[tuple[str, str, float, "np.ndarray"]] = []
    for stem in CANDIDATE_VIDEO_STEMS:
        for side in ("1P", "2P"):
            idx = load_npz_side_index(stem, side)
            if idx is None:
                continue
            for t_mid, changed_cols in find_ojama_increase_events(idx):
                if len(changed_cols) > 0:
                    candidates.append((stem, side, t_mid, changed_cols))
    order = rng.permutation(len(candidates))
    records: list[CellRecord] = []
    n_ok = 0
    for i in order:
        if n_ok >= SMOKE_WINDOWS_TOTAL:
            break
        stem, side, t_mid, changed_cols = candidates[i]
        frame = frame_cache.get_frame(stem, t_mid)
        if frame is None:
            continue
        region = region_for_side(side)
        for col in changed_cols:
            for row in range(HIDDEN_ROWS, BOARD_ROWS):
                records.append(extract_cell_record(
                    frame, region, stem, side, row, int(col), t_mid,
                    "smoke", "", COLOR_UNKNOWN,
                ))
        n_ok += 1
    return records, n_ok


# =============================================================================
# 5. 対照群 (平穏窓) 収集
# =============================================================================


def _unsafe_intervals(stem: str, fire_df: pd.DataFrame) -> list[tuple[float, float]]:
    """この動画の発火イベント時刻 (両side) の周辺を「不安全区間」として返す。"""
    video_id = f"video_{stem}"
    sub = fire_df[fire_df["video_id"] == video_id]
    return [
        (float(t) - BASELINE_EXCLUSION_SEC, float(t) + BASELINE_EXCLUSION_SEC)
        for t in sub["t_sec"].to_numpy()
    ]


def _is_safe(t_sec: float, intervals: list[tuple[float, float]]) -> bool:
    return not any(lo <= t_sec <= hi for lo, hi in intervals)


def collect_baseline_samples(
    fire_df: pd.DataFrame, frame_cache: FrameCache, rng: np.random.Generator,
) -> tuple[list[CellRecord], int]:
    """発火・おじゃま増加から離れた STABLE snapshot を BASELINE_WINDOWS_TOTAL 窓収集する。"""
    pool: list[tuple[str, str, float, "np.ndarray"]] = []
    for stem in CANDIDATE_VIDEO_STEMS:
        unsafe = _unsafe_intervals(stem, fire_df)
        for side in ("1P", "2P"):
            idx = load_npz_side_index(stem, side)
            if idx is None:
                continue
            ojama_events = find_ojama_increase_events(idx)
            unsafe_side = unsafe + [
                (t - BASELINE_EXCLUSION_SEC, t + BASELINE_EXCLUSION_SEC)
                for t, _ in ojama_events
            ]
            for i, t_sec in enumerate(idx.t_secs):
                if _is_safe(float(t_sec), unsafe_side):
                    pool.append((stem, side, float(t_sec), idx.grids[i]))
    order = rng.permutation(len(pool))
    records: list[CellRecord] = []
    n_ok = 0
    for i in order:
        if n_ok >= BASELINE_WINDOWS_TOTAL:
            break
        stem, side, t_sec, grid = pool[i]
        frame = frame_cache.get_frame(stem, t_sec)
        if frame is None:
            continue
        region = region_for_side(side)
        for row in range(HIDDEN_ROWS, BOARD_ROWS):
            for col in range(BOARD_COLS):
                gt = int(grid[row, col])
                if gt == COLOR_UNKNOWN:
                    continue
                layer = "empty" if gt == COLOR_EMPTY else (
                    "ojama" if gt == COLOR_OJAMA else "normal"
                )
                records.append(extract_cell_record(
                    frame, region, stem, side, row, col, t_sec, layer, "", gt,
                ))
        n_ok += 1
    return records, n_ok


# =============================================================================
# 6. 分離可能性分析
# =============================================================================

FEATURE_NAMES: tuple[str, ...] = (
    "v_mean", "v_max", "s_mean", "s_min", "specular_ratio", "bright_ratio",
)


def _best_threshold(pos: "np.ndarray", neg: "np.ndarray") -> tuple[float, float, float, float]:
    """Youden's J (TPR-FPR最大) で閾値を選び (閾値, AUC, TPR, FPR) を返す。

    pos/neg のスケールから閾値の向き (以上で陽性 / 以下で陽性) を AUC で自動判定する。
    """
    from sklearn.metrics import roc_auc_score, roc_curve

    y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    x = np.concatenate([pos, neg])
    auc = float(roc_auc_score(y, x)) if len(np.unique(y)) > 1 else 0.5
    score = x if auc >= 0.5 else -x
    auc_eff = auc if auc >= 0.5 else 1.0 - auc
    fpr, tpr, thr = roc_curve(y, score)
    j = tpr - fpr
    best_i = int(np.argmax(j))
    thr_raw = thr[best_i] if auc >= 0.5 else -thr[best_i]
    return float(thr_raw), auc_eff, float(tpr[best_i]), float(fpr[best_i])


def compute_separability_table(df: pd.DataFrame) -> pd.DataFrame:
    """特徴量ごとに (effect=burst+smoke) vs (normal) / vs (highlight_suspectのみ) の分離度を計算する。"""
    effect = df[df["layer"].isin(["burst", "smoke"])]
    normal_all = df[df["layer"] == "normal"]
    highlight_only = df[(df["layer"] == "normal") & (df["highlight_suspect"])]
    rows = []
    for feat in FEATURE_NAMES:
        pos = effect[feat].to_numpy()
        neg_all = normal_all[feat].to_numpy()
        thr, auc_all, tpr_all, fpr_all = _best_threshold(pos, neg_all)
        fpr_highlight = float("nan")
        if len(highlight_only) > 0:
            hi_vals = highlight_only[feat].to_numpy()
            direction = 1.0 if auc_all >= 0.5 else -1.0
            fpr_highlight = float(np.mean(direction * hi_vals >= direction * thr))
        rows.append({
            "feature": feat, "n_effect": len(pos), "n_normal": len(neg_all),
            "auc_vs_normal": round(auc_all, 3), "threshold": round(thr, 2),
            "tpr_at_threshold": round(tpr_all, 3), "fpr_at_threshold": round(fpr_all, 3),
            "fpr_vs_highlight_suspect": round(fpr_highlight, 3),
        })
    return pd.DataFrame(rows).sort_values("auc_vs_normal", ascending=False).reset_index(drop=True)


def compute_row_band_simultaneity(df: pd.DataFrame) -> pd.DataFrame:
    """窓単位の「row1-3同時発光セル数」比較 (burst窓 vs baseline窓の同じ行帯)。

    バーストは18セル(3行x6列)同時に発光する想定のため、単セル閾値でなく
    「同時に閾値超えするセル数」が単発の光沢ハイライトと区別できるかを見る。
    """
    band = df[(df["row"] >= BURST_ROW_MIN) & (df["row"] <= BURST_ROW_MAX)]
    rows = []
    for layer, group_cols in (
        ("burst", ["video_stem", "side", "t_sec"]),
        ("normal_baseline", ["video_stem", "side", "t_sec"]),
    ):
        sub = band[band["layer"] == ("burst" if layer == "burst" else "normal")]
        if len(sub) == 0:
            continue
        counts = sub.groupby(group_cols).apply(
            lambda g: int(np.sum(g["specular_ratio"] >= HIGHLIGHT_SUSPECT_RATIO_THRESHOLD)),
            include_groups=False,
        )
        rows.append({
            "layer": layer, "n_windows": len(counts),
            "mean_flagged_cells": round(float(counts.mean()), 2),
            "max_flagged_cells": int(counts.max()),
        })
    return pd.DataFrame(rows)


# =============================================================================
# 7. 可視化 (ヒストグラム + タイル)
# =============================================================================



# ヒストグラム描画対象の最小サンプル数 (これ未満は n 過小のため描画から除外し、
# separability_report.md に注記する。matplotlib は日本語グリフ非対応フォント
# 環境のため、図中テキストは英語表記に統一する)。
MIN_PLOT_N: int = 15


def plot_feature_histograms(df: pd.DataFrame, out_dir: Path) -> list[Path]:
    """特徴量ごとに層別ヒストグラムを描画し PNG 保存する (n過小の層は除外)。"""
    layers = ["burst", "smoke", "normal", "ojama", "empty"]
    colors = {"burst": "red", "smoke": "gray", "normal": "green",
              "ojama": "blue", "empty": "black"}
    paths: list[Path] = []
    for feat in FEATURE_NAMES:
        fig, ax = plt.subplots(figsize=(7, 4))
        for layer in layers:
            vals = df.loc[df["layer"] == layer, feat].to_numpy()
            if len(vals) < MIN_PLOT_N:
                continue
            ax.hist(vals, bins=30, alpha=0.4, label=f"{layer} (n={len(vals)})",
                    color=colors[layer], density=True)
        ax.set_title(f"{feat} distribution by layer (n<{MIN_PLOT_N} omitted)")
        ax.set_xlabel(feat)
        ax.set_ylabel("density")
        ax.legend(fontsize=8)
        out_path = out_dir / f"dist_{feat}.png"
        fig.tight_layout()
        fig.savefig(out_path, dpi=110)
        plt.close(fig)
        paths.append(out_path)
    return paths


def save_layer_tile(records: list[CellRecord], layer_key: str, out_dir: Path) -> "Path | None":
    """層 (or highlight_suspect) の代表セル切り出しをタイル状に並べて保存する。"""
    if layer_key == "highlight_suspect":
        picked = [r for r in records if r.highlight_suspect]
        picked.sort(key=lambda r: -r.specular_ratio)
    else:
        picked = [r for r in records if r.layer == layer_key]
    picked = picked[:TILE_MAX_PER_LAYER]
    if not picked:
        return None
    n_rows = int(np.ceil(len(picked) / TILE_COLS))
    canvas = np.full(
        (n_rows * TILE_CELL_SIZE, TILE_COLS * TILE_CELL_SIZE, 3), 40, dtype=np.uint8,
    )
    for i, rec in enumerate(picked):
        r, c = divmod(i, TILE_COLS)
        patch = cv2.resize(rec.patch_bgr, (TILE_CELL_SIZE, TILE_CELL_SIZE))
        canvas[r * TILE_CELL_SIZE:(r + 1) * TILE_CELL_SIZE,
               c * TILE_CELL_SIZE:(c + 1) * TILE_CELL_SIZE] = patch
        label = f"{rec.video_stem}:{rec.side} r{rec.row}c{rec.col}"
        cv2.putText(
            canvas, label, (c * TILE_CELL_SIZE + 2, r * TILE_CELL_SIZE + 12),
            cv2.FONT_HERSHEY_PLAIN, 0.8, (255, 255, 255), 1, cv2.LINE_AA,
        )
    out_path = out_dir / f"tile_{layer_key}.png"
    cv2.imwrite(str(out_path), canvas)
    return out_path


# =============================================================================
# 8. レポート
# =============================================================================


def _df_to_markdown(df: pd.DataFrame) -> str:
    """tabulate 非依存の簡易 markdown 表変換 (pandas.to_markdown の軽量代替)。"""
    cols = list(df.columns)
    header = "| " + " | ".join(cols) + " |"
    sep = "|" + "|".join(["---"] * len(cols)) + "|"
    body_lines = []
    for _, row in df.iterrows():
        body_lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join([header, sep] + body_lines)


def build_report_markdown(
    window_counts: dict[str, int], smoke_n: int, baseline_n: int,
    sep_table: pd.DataFrame, simultaneity_table: pd.DataFrame,
) -> str:
    """separability_report.md の本文を組み立てる。"""
    lines: list[str] = []
    lines.append("# エフェクト視覚特性調査 (2026-08-03)\n")
    lines.append("## 収集窓数\n")
    lines.append("| 種別 | 収集数 | 目標 |")
    lines.append("|---|---|---|")
    for lo, hi, label in CHAIN_MAGNITUDE_BINS:
        lines.append(f"| バースト ({label}連鎖) | {window_counts.get(label, 0)} | {BURST_WINDOWS_PER_BIN} |")
    lines.append(f"| 煙 | {smoke_n} | {SMOKE_WINDOWS_TOTAL} |")
    lines.append(f"| 対照群 (STABLE平穏) | {baseline_n} | {BASELINE_WINDOWS_TOTAL} |")
    lines.append("")
    lines.append("## セル単位 分離可能性 (effect=burst+smoke vs normal)\n")
    lines.append(_df_to_markdown(sep_table))
    lines.append("")
    lines.append(
        "`fpr_vs_highlight_suspect` = 通常セルの閾値超えの中でも「光沢ハイライト疑い"
        f"(白飛び率>={HIGHLIGHT_SUSPECT_RATIO_THRESHOLD})」だけに絞った場合の誤検出率。"
        "全normal平均のfprより高いほど、光沢ハイライトが最も紛らわしい偽陽性源であることを示す。\n"
    )
    lines.append("## 窓単位: row1-3同時発光セル数 (burst窓 vs 通常セルのrow1-3帯)\n")
    lines.append(_df_to_markdown(simultaneity_table))
    lines.append("")
    lines.append("## 制約・注意点 (要考慮)\n")
    lines.append(
        "- **窓タイミングは推定値であり実効果視認の保証なし**: burst窓は連鎖規模から"
        "推定した継続時間の中間点、smoke窓はおじゃま増加検知の中間点を機械的に採用して"
        "いる。tile_burst.png/tile_smoke.png を目視した限り、明確に発光/白煙が写って"
        "いるセルと、通常の盤面 (エフェクト無し) にしか見えないセルが混在していた。"
        "つまり本レポートの分離可能性数値は「エフェクト有無不明な窓」込みの**下限値**"
        "であり、真の分離可能性はこれより良い可能性がある。"
    )
    lines.append(
        "- **通常ぷよの白目グラフィックが「光沢ハイライト疑い」の69% (347/502) を占める**: "
        "tile_highlight_suspect.png は実質的に全て白目グラフィックの切り出しで、"
        "白飛び率・V平均などの単純特徴量ではバースト/煙と区別できない (fpr_vs_"
        "highlight_suspect=1.0 が大半)。**唯一の例外が s_min (セル内最小彩度)** で、"
        "白目セルは白目部分が S≈0 に落ちるため s_min も低いが、バースト/煙のセルは"
        "彩度のある色付き発光/白煙で覆われるため s_min が全体的に高く残る "
        "(fpr_vs_highlight_suspect=0.0、AUC=0.822)。"
    )
    lines.append(
        "- **s_min は空セル (背景) との区別がつかない**: 空セルの s_min 平均は98.4と"
        "バースト/煙 (平均65-73) より**さらに高い**。s_min 単体では「背景か発光か」の"
        "切り分けができないため、実運用では「このセルは元々何か描画されている"
        "(v_meanが背景より高い等)」との併用条件が必須。"
    )
    lines.append(
        f"- **おじゃまセルの対照サンプルはn=2のみ**: 「平穏」時間帯という選定基準が"
        "終盤(おじゃまが積もりやすい局面)を間接的に除外してしまい、対照群のおじゃま"
        "セルがほぼ採取できなかった。おじゃまとバースト/煙の分離可能性は本レポートでは"
        f"評価不能 (要再収集、ヒストグラムは n<{MIN_PLOT_N} のため非表示)。"
    )
    lines.append("")
    return "\n".join(lines)


# =============================================================================
# main
# =============================================================================


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RANDOM_SEED)
    fire_df = pd.read_csv(FIRE_EVENTS_CSV)
    fire_df = fire_df[fire_df["video_id"].isin(
        [f"video_{s}" for s in CANDIDATE_VIDEO_STEMS]
    )].copy()
    print(f"[1/6] 発火イベントpool: {len(fire_df)}件 (対象動画{len(CANDIDATE_VIDEO_STEMS)}本)")

    frame_cache = FrameCache()
    try:
        burst_records, burst_counts = collect_burst_samples(fire_df, frame_cache, rng)
        print(f"[2/6] バースト窓収集: {burst_counts}")

        smoke_records, smoke_n = collect_smoke_samples(frame_cache, rng)
        print(f"[3/6] 煙窓収集: {smoke_n}/{SMOKE_WINDOWS_TOTAL}")

        baseline_records, baseline_n = collect_baseline_samples(fire_df, frame_cache, rng)
        print(f"[4/6] 対照群収集: {baseline_n}/{BASELINE_WINDOWS_TOTAL}")
    finally:
        frame_cache.release_all()

    all_records = burst_records + smoke_records + baseline_records
    print(f"[5/6] 全セル統計 {len(all_records)}件、分離可能性分析+可視化を生成")

    df = pd.DataFrame([{
        "video_stem": r.video_stem, "side": r.side, "row": r.row, "col": r.col,
        "t_sec": r.t_sec, "layer": r.layer, "chain_bin": r.chain_bin,
        "ground_truth_color": r.ground_truth_color, "v_mean": r.v_mean,
        "v_max": r.v_max, "s_mean": r.s_mean, "s_min": r.s_min,
        "specular_ratio": r.specular_ratio, "bright_ratio": r.bright_ratio,
        "highlight_suspect": r.highlight_suspect,
    } for r in all_records])
    df.to_csv(OUTPUT_DIR / "cell_stats.csv", index=False, encoding="utf-8-sig")

    sep_table = compute_separability_table(df)
    simultaneity_table = compute_row_band_simultaneity(df)
    plot_feature_histograms(df, OUTPUT_DIR)
    for layer_key in ("burst", "smoke", "normal", "ojama", "empty", "highlight_suspect"):
        save_layer_tile(all_records, layer_key, OUTPUT_DIR)

    report = build_report_markdown(burst_counts, smoke_n, baseline_n, sep_table, simultaneity_table)
    (OUTPUT_DIR / "separability_report.md").write_text(report, encoding="utf-8")
    print(f"[6/6] 完了。出力先: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
