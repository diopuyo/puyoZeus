"""
重みチューニング script (Sprint 4 - 拡張指標導入後)

目的:
    data/verify/match_winners_v02.tsv の実勝敗ラベルと、各試合終了直前の
    指標差分 (1P - 2P) を比較し、勝者を当てる重みを grid search で求める。

重要な観察 (2026-04-24 video_02 実測):
    試合終了 SAMPLE_OFFSET_SEC=3 秒前のフレームでは
        - 敗者側の盤面が「窒息間近で本線が崩壊・形が崩れている」状態
        - 勝者側は既に発火終了 → 盤面はガラ空きで指標 0 に近い
    という現象が起き、
    死亡リスクや本線完成度などのほとんどの指標は 「敗者の方が高い」 反転傾向。

    そのため grid search 結果も主要指標すべてに「負」の重みを置く解になり、
    baseline 0.08 → grid 0.92 と大幅改善するが、これは 「試合最終局面」 の
    特殊な状況に過適合しただけで、ライブ判定向け重みではない。

    実用重みは「試合中盤」(end - SAMPLE_OFFSET_SEC を 30 秒以上に設定 or
    試合中央付近) で再 tuning する必要がある。本 script はその枠組みを提供。

入出力:
    入力: data/verify/match_winners_v02.tsv, match_boundaries_v4/video_02/matches.tsv
    入力動画: data/frames/video_02.mp4
    出力: data/verify/tune_weights_v02.json
        - per_indicator_accuracy: 各指標の単独勝者一致率 (baseline)
        - best_weights: grid search で見つけた最良重み
        - best_accuracy: 最良重みの勝者一致率
        - per_match_features: 各試合の (winner, p1指標値辞書, p2指標値辞書)

使い方:
    python -m scripts.tune_weights --video data/frames/video_02.mp4 \
        --winners data/verify/match_winners_v02.tsv \
        --boundaries data/verify/match_boundaries_v4/video_02/matches.tsv \
        --out data/verify/tune_weights_v02.json
    python -m scripts.tune_weights --dry-run    # 入出力だけテスト

実装方針:
    - 各試合 end_sec - SAMPLE_OFFSET 秒のフレームを 1 枚読んで指標抽出
      (50 試合 × 1 フレーム ≈ 60 秒で完結)
    - 個々の指標スコアを 1 度抽出してキャッシュし、grid search は CPU のみで実行
    - 重みは [-1.5, -1.0, -0.5, 0, 0.5, 1.0, 1.5] の 7 段階で coarse search
      ALL_INDICATOR_NAMES (8) のみ探索、拡張指標は既定値固定で OK
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any

import cv2

# プロジェクトルートを sys.path に追加 (script として直接呼ぶ場合のため)
_PROJ_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))

from src.image_reader import ImageReader  # noqa: E402
from src.indicators import (  # noqa: E402
    ALL_INDICATOR_NAMES,
    EXTRA_INDICATOR_NAMES,
    IndicatorCalculator,
)
from src.scorer import DEFAULT_WEIGHTS, Scorer  # noqa: E402

# ============================
# 定数
# ============================

# 試合終了直前から何秒前のフレームを評価対象にするか (試合の最終局面を捉える)
SAMPLE_OFFSET_SEC: float = 3.0

# grid search の重み候補 (粗い探索)
WEIGHT_GRID: tuple[float, ...] = (-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5)

# grid search 対象の指標 (ALL_INDICATOR_NAMES のサブセット;
# 全 8 指標を 7^8 = 5,764,801 で探すと重いので、勝負を分ける主要 4 指標に絞る)
GRID_SEARCH_TARGETS: tuple[str, ...] = (
    "main_chain_maturity",
    "offset_power",
    "death_risk",
    "extension_potential",
)


@dataclass(frozen=True)
class MatchSample:
    """1 試合分のサンプル: 指標スコア辞書と勝者ラベル。"""
    idx: int
    end_sec: float
    winner: str  # "1P" or "2P"
    p1_scores: dict[str, float]
    p2_scores: dict[str, float]


# ============================
# I/O
# ============================


def load_winners(tsv_path: Path) -> dict[int, tuple[float, str]]:
    """match_winners TSV から idx → (end_sec, winner) を返す。"""
    out: dict[int, tuple[float, str]] = {}
    for line in tsv_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("idx"):
            continue
        cols = line.split("\t")
        if len(cols) < 4:
            continue
        try:
            idx = int(cols[0])
            end_sec = float(cols[2])
            winner = cols[3].strip()
        except ValueError:
            continue
        if winner not in ("1P", "2P"):
            continue
        out[idx] = (end_sec, winner)
    return out


# ============================
# サンプル抽出
# ============================


def extract_samples(
    video_path: Path,
    winners: dict[int, tuple[float, str]],
    boundaries: dict[int, tuple[float, float]] | None = None,
    sample_mode: str = "end",
    offset_sec: float = SAMPLE_OFFSET_SEC,
) -> list[MatchSample]:
    """各試合の指定タイミング (end / midpoint) のフレームから指標スコアを抽出。

    Args:
        video_path: 入力動画。
        winners: idx → (end_sec, winner)。
        boundaries: idx → (start_sec, end_sec)。midpoint モードで必要。
        sample_mode: "end" (end - offset_sec) or "midpoint" ((start+end)/2)。
        offset_sec: end モードでの試合終了からの遡り秒数。
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"動画を開けません: {video_path}")
    reader = ImageReader()
    calc = IndicatorCalculator()
    samples: list[MatchSample] = []
    for idx, (end_sec, winner) in sorted(winners.items()):
        if sample_mode == "midpoint" and boundaries and idx in boundaries:
            start, end = boundaries[idx]
            t = (start + end) / 2.0
        else:
            t = max(0.0, end_sec - offset_sec)
        sample = _extract_one(cap, reader, calc, idx, end_sec, winner, t)
        if sample is not None:
            samples.append(sample)
    cap.release()
    return samples


def load_boundaries(tsv_path: Path) -> dict[int, tuple[float, float]]:
    """matches.tsv (idx, start, end, ...) → idx→(start,end) を返す。"""
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
            start = float(cols[1])
            end = float(cols[2])
        except ValueError:
            continue
        out[idx] = (start, end)
    return out


def _extract_one(
    cap: cv2.VideoCapture,
    reader: ImageReader,
    calc: IndicatorCalculator,
    idx: int,
    end_sec: float,
    winner: str,
    t_sec: float,
) -> MatchSample | None:
    """1 試合分のサンプルを抽出する。"""
    cap.set(cv2.CAP_PROP_POS_MSEC, t_sec * 1000.0)
    ok, frame = cap.read()
    if not ok or frame is None:
        return None
    try:
        b1, b2 = reader.read_both_boards(frame)
        ind1 = calc.compute_all(b1)
        ind2 = calc.compute_all(b2)
    except Exception:
        return None
    return MatchSample(
        idx=idx,
        end_sec=end_sec,
        winner=winner,
        p1_scores=_indicator_scores_dict(ind1),
        p2_scores=_indicator_scores_dict(ind2),
    )


def _indicator_scores_dict(indicator_set) -> dict[str, float]:
    """IndicatorSet を name→score 辞書に変換 (拡張指標含む)。"""
    out: dict[str, float] = {}
    for name in ALL_INDICATOR_NAMES:
        out[name] = float(indicator_set.results[name].score)
    for name in EXTRA_INDICATOR_NAMES:
        if name in indicator_set.results:
            out[name] = float(indicator_set.results[name].score)
    return out


# ============================
# 評価ロジック
# ============================


def per_indicator_accuracy(
    samples: list[MatchSample],
) -> dict[str, float]:
    """各指標単独で「(p1>p2) == (winner=='1P')」となる比率を返す。"""
    out: dict[str, float] = {}
    if not samples:
        return out
    all_names = list(ALL_INDICATOR_NAMES) + list(EXTRA_INDICATOR_NAMES)
    for name in all_names:
        correct = 0
        total = 0
        for s in samples:
            v1 = s.p1_scores.get(name)
            v2 = s.p2_scores.get(name)
            if v1 is None or v2 is None or v1 == v2:
                continue
            predicted = "1P" if v1 > v2 else "2P"
            # death_risk は 高いほど不利なので反転
            if name == "death_risk":
                predicted = "2P" if v1 > v2 else "1P"
            if predicted == s.winner:
                correct += 1
            total += 1
        out[name] = correct / total if total > 0 else 0.0
    return out


def evaluate_weights(
    samples: list[MatchSample],
    weights: dict[str, float],
) -> float:
    """指定重みでの勝者一致率を返す。"""
    if not samples:
        return 0.0
    correct = 0
    for s in samples:
        diff = 0.0
        for name, w in weights.items():
            v1 = s.p1_scores.get(name, 0.0)
            v2 = s.p2_scores.get(name, 0.0)
            diff += (v1 - v2) * w
        if diff == 0.0:
            continue
        predicted = "1P" if diff > 0 else "2P"
        if predicted == s.winner:
            correct += 1
    return correct / len(samples)


def grid_search(
    samples: list[MatchSample],
    base_weights: dict[str, float],
    targets: tuple[str, ...] = GRID_SEARCH_TARGETS,
    grid: tuple[float, ...] = WEIGHT_GRID,
) -> tuple[dict[str, float], float]:
    """重み grid search で最良重みと一致率を返す。"""
    best_acc = -1.0
    best_w = dict(base_weights)
    for combo in product(grid, repeat=len(targets)):
        w = dict(base_weights)
        for i, name in enumerate(targets):
            w[name] = combo[i]
        acc = evaluate_weights(samples, w)
        if acc > best_acc:
            best_acc = acc
            best_w = dict(w)
    return best_w, best_acc


# ============================
# main エントリ
# ============================


def main() -> int:
    parser = argparse.ArgumentParser(
        description="重みチューニング (拡張指標 + grid search)",
    )
    parser.add_argument("--video", type=Path)
    parser.add_argument("--winners", type=Path,
                        default=Path("data/verify/match_winners_v02.tsv"))
    parser.add_argument(
        "--boundaries", type=Path,
        default=Path("data/verify/match_boundaries_v4/video_02/matches.tsv"),
    )
    parser.add_argument(
        "--out", type=Path,
        default=Path("data/verify/tune_weights_v02.json"),
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="動画を読まず、TSV のロードだけ確認")
    parser.add_argument(
        "--sample-mode", choices=("end", "midpoint"), default="end",
        help="サンプリングタイミング (end=終了直前, midpoint=試合中央)",
    )
    parser.add_argument(
        "--offset-sec", type=float, default=SAMPLE_OFFSET_SEC,
        help="end モードでの遡り秒数",
    )
    args = parser.parse_args()

    winners = load_winners(args.winners)
    print(f"[load] winners: {len(winners)} 試合")

    if args.dry_run:
        print("[dry-run] 動画読み込みをスキップ")
        return 0

    if args.video is None or not args.video.exists():
        print(f"[error] --video が必要 (見つからない: {args.video})", file=sys.stderr)
        return 1

    boundaries = load_boundaries(args.boundaries)
    samples = extract_samples(
        args.video, winners,
        boundaries=boundaries,
        sample_mode=args.sample_mode,
        offset_sec=args.offset_sec,
    )
    print(f"[extract] サンプル: {len(samples)} / {len(winners)} "
          f"(mode={args.sample_mode}, offset={args.offset_sec}s)")

    base_acc = evaluate_weights(samples, DEFAULT_WEIGHTS)
    print(f"[baseline] DEFAULT_WEIGHTS 一致率: {base_acc:.3f}")

    per_acc = per_indicator_accuracy(samples)
    for name, acc in sorted(per_acc.items(), key=lambda x: -x[1]):
        print(f"  {name:<25s} {acc:.3f}")

    best_w, best_acc = grid_search(samples, DEFAULT_WEIGHTS)
    print(f"[grid] best 一致率: {best_acc:.3f}")
    for name in GRID_SEARCH_TARGETS:
        print(f"  {name}: {best_w[name]:+.2f}")

    out = {
        "n_samples": len(samples),
        "sample_mode": args.sample_mode,
        "offset_sec": args.offset_sec,
        "baseline_accuracy": base_acc,
        "per_indicator_accuracy": per_acc,
        "best_weights": best_w,
        "best_accuracy": best_acc,
        "grid_targets": list(GRID_SEARCH_TARGETS),
        "weight_grid": list(WEIGHT_GRID),
        "per_match_features": [
            {
                "idx": s.idx, "end_sec": s.end_sec, "winner": s.winner,
                "p1": s.p1_scores, "p2": s.p2_scores,
            }
            for s in samples
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(f"[save] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
