"""
多動画 × 時刻別の大規模特徴量データセット生成スクリプト (段階 1)

目的:
    DEFAULT_WEIGHTS holdout test=0.720 / Grid 50 試合=0.520 / LR 50 試合 K-fold=0.340
    という過適合の根本原因 (50 試合 × 16 特徴量) を解消するため、
    3 動画 (~148 試合) × 5 時刻 サンプリングで ~740 行のデータセットを構築する。

サンプリング時刻 (1 試合あたり 5 点):
    - midpoint        : (start + end) / 2
    - midpoint - 20s  : 中央より 20 秒早い (中盤序盤)
    - midpoint + 20s  : 中央より 20 秒遅い (終盤手前)
    - end - 5s        : 試合終了 5 秒前 (決着直前)
    - start + 20s     : 試合開始 20 秒後 (序盤)

特徴量:
    16 個の指標差分 (1P - 2P)。
    ALL_INDICATOR_NAMES (8) + EXTRA_INDICATOR_NAMES (8) を全て使う。

ラベル:
    +1 if winner == "1P" else -1 (UNKNOWN は除外)。

出力:
    data/training/match_features.csv
        列: video_id, match_idx, time_phase, <16 特徴量>, label

実行例:
    python -m scripts.generate_training_dataset
    python -m scripts.generate_training_dataset --videos 1 --time-phases midpoint end5
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2

# プロジェクトルートを sys.path に追加
_PROJ_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))

from src.image_reader import ImageReader  # noqa: E402
from src.old.indicators import (  # noqa: E402
    ALL_INDICATOR_NAMES,
    EXTRA_INDICATOR_NAMES,
    IndicatorCalculator,
    IndicatorSet,
)
from src.next_detector import NextDetector  # noqa: E402

# ============================
# 定数
# ============================

# 全 45 特徴量名 (ALL=8 + EXTRA=37)
# Phase H1 (2026-05-08) で機能 7 + 戦況 8 + 形分類 1 = 16 個追加し
# 29 → 45 に拡張。
FEATURE_NAMES: tuple[str, ...] = tuple(ALL_INDICATOR_NAMES) + tuple(
    EXTRA_INDICATOR_NAMES,
)

# 5 時刻フェーズの識別子
TIME_PHASE_START_PLUS: str = "start_plus_20"
TIME_PHASE_MID_MINUS: str = "mid_minus_20"
TIME_PHASE_MIDPOINT: str = "midpoint"
TIME_PHASE_MID_PLUS: str = "mid_plus_20"
TIME_PHASE_END_MINUS: str = "end_minus_5"

DEFAULT_TIME_PHASES: tuple[str, ...] = (
    TIME_PHASE_START_PLUS,
    TIME_PHASE_MID_MINUS,
    TIME_PHASE_MIDPOINT,
    TIME_PHASE_MID_PLUS,
    TIME_PHASE_END_MINUS,
)

# 各フェーズのオフセット秒 (None=midpoint or end ベース)
PHASE_OFFSET_FROM_START: float = 20.0
PHASE_OFFSET_FROM_MID: float = 20.0
PHASE_OFFSET_FROM_END: float = 5.0

# 試合の最低時間 (これ未満の試合はサンプル不足のためスキップ)
MIN_MATCH_DURATION_SEC: float = 25.0

# 動画 ID と各種 TSV パスの既定マッピング
DEFAULT_VIDEO_DIR: Path = Path("data/frames")
DEFAULT_BOUNDARY_DIR: Path = Path("data/verify/match_boundaries_v4")
DEFAULT_WINNERS_DIR: Path = Path("data/verify")
DEFAULT_OUTPUT_CSV: Path = Path("data/training/match_features.csv")

VIDEO_IDS: tuple[str, ...] = ("01", "02", "03")


# ============================
# データクラス
# ============================


@dataclass(frozen=True)
class MatchMeta:
    """1 試合のメタ情報。"""
    video_id: str
    match_idx: int
    start_sec: float
    end_sec: float
    winner: str  # "1P" / "2P"


@dataclass(frozen=True)
class FeatureRow:
    """1 サンプル分の出力行 (CSV 1 行に対応)。"""
    video_id: str
    match_idx: int
    time_phase: str
    features: dict[str, float]
    label: int  # +1 / -1


# ============================
# I/O
# ============================


def load_winners(tsv_path: Path) -> dict[int, str]:
    """match_winners TSV から idx → winner を読み込む。"""
    out: dict[int, str] = {}
    if not tsv_path.exists():
        return out
    for line in tsv_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("idx"):
            continue
        cols = line.split("\t")
        if len(cols) < 4:
            continue
        try:
            idx = int(cols[0])
        except ValueError:
            continue
        winner = cols[3].strip()
        if winner not in ("1P", "2P"):
            continue
        out[idx] = winner
    return out


def load_boundaries(tsv_path: Path) -> dict[int, tuple[float, float]]:
    """matches.tsv (idx, start_sec, end_sec, ...) を読み込む。"""
    out: dict[int, tuple[float, float]] = {}
    if not tsv_path.exists():
        return out
    for line in tsv_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("idx"):
            continue
        cols = line.split("\t")
        if len(cols) < 3:
            continue
        try:
            idx = int(cols[0])
            start_sec = float(cols[1])
            end_sec = float(cols[2])
        except ValueError:
            continue
        out[idx] = (start_sec, end_sec)
    return out


def collect_match_meta(
    video_id: str,
    boundary_dir: Path = DEFAULT_BOUNDARY_DIR,
    winners_dir: Path = DEFAULT_WINNERS_DIR,
) -> list[MatchMeta]:
    """1 動画分の試合メタ情報リストを取得する。"""
    boundaries = load_boundaries(
        boundary_dir / f"video_{video_id}" / "matches.tsv",
    )
    winners = load_winners(
        winners_dir / f"match_winners_v{video_id}.tsv",
    )
    metas: list[MatchMeta] = []
    for idx in sorted(boundaries.keys() & winners.keys()):
        start, end = boundaries[idx]
        if end - start < MIN_MATCH_DURATION_SEC:
            continue
        metas.append(MatchMeta(
            video_id=video_id,
            match_idx=idx,
            start_sec=start,
            end_sec=end,
            winner=winners[idx],
        ))
    return metas


# ============================
# サンプリング時刻計算
# ============================


def compute_sample_time(
    meta: MatchMeta,
    phase: str,
) -> float:
    """指定フェーズに対応する試合内秒数 (動画上の絶対秒) を返す。"""
    midpoint = (meta.start_sec + meta.end_sec) / 2.0
    if phase == TIME_PHASE_START_PLUS:
        t = meta.start_sec + PHASE_OFFSET_FROM_START
    elif phase == TIME_PHASE_MID_MINUS:
        t = midpoint - PHASE_OFFSET_FROM_MID
    elif phase == TIME_PHASE_MIDPOINT:
        t = midpoint
    elif phase == TIME_PHASE_MID_PLUS:
        t = midpoint + PHASE_OFFSET_FROM_MID
    elif phase == TIME_PHASE_END_MINUS:
        t = meta.end_sec - PHASE_OFFSET_FROM_END
    else:
        raise ValueError(f"未知の time_phase: {phase}")
    # 試合範囲内に丸める (境界からはみ出す場合に備えて)
    t = max(meta.start_sec + 1.0, min(meta.end_sec - 1.0, t))
    return t


# ============================
# 指標抽出
# ============================


def _indicator_value(indicator_set: IndicatorSet, name: str) -> float:
    """IndicatorSet から指標値 (results 優先・属性フォールバック) を取得。"""
    if name in indicator_set.results:
        return float(indicator_set.results[name].score)
    attr_map = {
        "next_acceptance": "next_acceptance",
        "shape_score": "shape_score",
        "touching_density": "touching_density",
        "tail_height": "tail_height_score",
        "color_variance": "color_variance_score",
        "key_flexibility": "key_flexibility",
        "sub_chain_independence": "sub_chain_independence",
        "chain_timing_pressure": "chain_timing_pressure",
    }
    attr = attr_map.get(name)
    if attr is None:
        return 0.0
    return float(getattr(indicator_set, attr, 0.0))


def extract_feature_diff(
    indicator_set_1p: IndicatorSet,
    indicator_set_2p: IndicatorSet,
) -> dict[str, float]:
    """1P / 2P 指標セットから 16 特徴量の差分 (p1 - p2) を計算する。"""
    out: dict[str, float] = {}
    for name in FEATURE_NAMES:
        v1 = _indicator_value(indicator_set_1p, name)
        v2 = _indicator_value(indicator_set_2p, name)
        out[name] = v1 - v2
    return out


def _detect_next_pairs(
    next_detector: NextDetector | None,
    frame,
) -> tuple[
    tuple[int, int] | None, tuple[int, int] | None,
    tuple[int, int] | None, tuple[int, int] | None,
]:
    """1P/2P 両側の next/dnext を読み取る。失敗時はそれぞれ None。

    Returns:
        (p1_next, p1_dnext, p2_next, p2_dnext)
    """
    if next_detector is None:
        return (None, None, None, None)
    try:
        both = next_detector.detect_both(frame)
    except Exception:
        return (None, None, None, None)
    return (
        both.p1.next_pair,
        both.p1.dnext_pair,
        both.p2.next_pair,
        both.p2.dnext_pair,
    )


def extract_one_sample(
    cap: cv2.VideoCapture,
    reader: ImageReader,
    calc: IndicatorCalculator,
    meta: MatchMeta,
    phase: str,
    next_detector: NextDetector | None = None,
) -> FeatureRow | None:
    """1 試合 × 1 フェーズの特徴量行を抽出する。失敗時 None。"""
    t_sec = compute_sample_time(meta, phase)
    cap.set(cv2.CAP_PROP_POS_MSEC, t_sec * 1000.0)
    ok, frame = cap.read()
    if not ok or frame is None:
        return None
    # NextDetector はフレーム単位で失敗してもサンプル全体を捨てない (graceful)。
    p1_next, p1_dnext, p2_next, p2_dnext = _detect_next_pairs(
        next_detector, frame,
    )
    try:
        b1, b2 = reader.read_both_boards(frame)
        # Phase J: opponent_board を相手側として渡す (凝視指標)
        ind1 = calc.compute_all(
            b1, next_pair=p1_next, dnext_pair=p1_dnext,
            opponent_board=b2,
        )
        ind2 = calc.compute_all(
            b2, next_pair=p2_next, dnext_pair=p2_dnext,
            opponent_board=b1,
        )
    except Exception:
        return None
    features = extract_feature_diff(ind1, ind2)
    return FeatureRow(
        video_id=meta.video_id,
        match_idx=meta.match_idx,
        time_phase=phase,
        features=features,
        label=1 if meta.winner == "1P" else -1,
    )


# ============================
# 動画単位のループ
# ============================


def extract_video_rows(
    video_id: str,
    video_path: Path,
    metas: list[MatchMeta],
    time_phases: Iterable[str],
    log_every: int = 10,
) -> list[FeatureRow]:
    """1 動画から複数試合 × 複数フェーズの行を抽出する。"""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"動画を開けません: {video_path}")
    reader = ImageReader()
    calc = IndicatorCalculator()
    # NextDetector の構築失敗時は None で graceful fallback (compute_all は
    # next_pair=None で従来動作)。
    try:
        next_detector: NextDetector | None = NextDetector.load_default()
    except Exception as exc:
        print(
            f"  [video {video_id}] NextDetector 構築失敗、"
            f"next_pair=None で続行: {exc}",
            flush=True,
        )
        next_detector = None
    rows: list[FeatureRow] = []
    phases = list(time_phases)
    for n, meta in enumerate(metas, start=1):
        for phase in phases:
            row = extract_one_sample(
                cap, reader, calc, meta, phase,
                next_detector=next_detector,
            )
            if row is not None:
                rows.append(row)
        if n % log_every == 0:
            print(
                f"  [video {video_id}] {n}/{len(metas)} 試合処理 "
                f"(累計 {len(rows)} 行)",
                flush=True,
            )
    cap.release()
    return rows


# ============================
# CSV 書き出し
# ============================


def write_csv(rows: list[FeatureRow], out_path: Path) -> None:
    """FeatureRow リストを CSV に書き出す。"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    header = ["video_id", "match_idx", "time_phase"]
    header.extend(FEATURE_NAMES)
    header.append("label")
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for r in rows:
            line: list[object] = [r.video_id, r.match_idx, r.time_phase]
            for name in FEATURE_NAMES:
                line.append(f"{r.features.get(name, 0.0):.6f}")
            line.append(r.label)
            writer.writerow(line)


# ============================
# main エントリ
# ============================


def main() -> int:
    parser = argparse.ArgumentParser(
        description="多動画 × 時刻別 特徴量データセット生成",
    )
    parser.add_argument(
        "--videos", nargs="+", default=list(VIDEO_IDS),
        help="対象動画 ID (例: 01 02 03)",
    )
    parser.add_argument(
        "--time-phases", nargs="+", default=list(DEFAULT_TIME_PHASES),
    )
    parser.add_argument(
        "--video-dir", type=Path, default=DEFAULT_VIDEO_DIR,
    )
    parser.add_argument(
        "--out", type=Path, default=DEFAULT_OUTPUT_CSV,
    )
    parser.add_argument(
        "--max-matches-per-video", type=int, default=None,
        help="動画あたりの最大試合数 (テスト用)",
    )
    args = parser.parse_args()

    all_rows: list[FeatureRow] = []
    for video_id in args.videos:
        video_path = args.video_dir / f"video_{video_id}.mp4"
        if not video_path.exists():
            print(f"[skip] 動画が無い: {video_path}", file=sys.stderr)
            continue
        metas = collect_match_meta(video_id)
        if args.max_matches_per_video is not None:
            metas = metas[: args.max_matches_per_video]
        print(
            f"[video {video_id}] {len(metas)} 試合 × "
            f"{len(args.time_phases)} 時刻 = "
            f"{len(metas) * len(args.time_phases)} サンプル予定",
            flush=True,
        )
        rows = extract_video_rows(
            video_id, video_path, metas, args.time_phases,
        )
        all_rows.extend(rows)
        print(
            f"[video {video_id}] 抽出完了: {len(rows)} 行",
            flush=True,
        )

    write_csv(all_rows, args.out)
    print(
        f"\n[save] {args.out}: {len(all_rows)} 行 "
        f"(列数 {3 + len(FEATURE_NAMES) + 1})",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
