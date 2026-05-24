"""Phase Z-3: 連続 frame 自動評価ハーネス。

Phase Z-1/Z-2 で構築したパイプラインの labels.csv から、suspicious_reasons
を解析して連続 frame 全体の品質を自動計測する。

GT を必要とせず、物理ルール違反 (連鎖外消失、浮遊、ペア不整合) や
HSV/CNN 不一致を「真の誤りの強い候補」として集計する。

メトリック (suspicious 種別ごと、cell 単位):
    - hard_violations: 真の誤りである確率が極めて高い (色 swap、消失、浮遊)
    - soft_warnings: 補正後に false positive な可能性が高い (em_but_*, low_conf)
    - 違反率 = hard_violations / 全 cell

補正レイヤー入りの labels.csv に対して実行することで、
「連続 frame 全体で 99.5% 維持できているか」を GT なしで間接評価する。

利用例:
    PYTHONPATH=. ./venv/bin/python -m scripts.phase_z_continuous_eval \
        --labels data/verify/phase_z_review/v18_m03_30_60/labels.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# 「真の誤り」である確率が極めて高い suspicious 理由
HARD_VIOLATION_REASONS: tuple[str, ...] = (
    "color_swap",          # 色 → 別の色 (連鎖外で発生は不可能)
    "disappearance",       # 色 → EM (連鎖外で発生は不可能)
    "airborne",            # 浮遊 (重力違反)
    "hidden_below",        # 上に色があるのに自身 EM (重力違反)
    "empty_in_stack",      # 列内に EM 穴 (重力違反)
    "pair_mismatch",       # 新規 cell が next_pair と不整合
    "solo_appearance",     # 1 cell だけ新規出現 (ペアは 2 cell)
    "hsv_disagree",        # CNN/HSV の puyo 色が異なる
    "unknown_drop",        # 色 → ?? (検出失敗の典型)
)

# 「補正後 false positive」の可能性が高い soft 警告
SOFT_WARNING_REASONS: tuple[str, ...] = (
    "em_but_saturated",    # EM 判定だが彩度高い (補正で puyo 化済の可能性)
    "em_but_grayish",      # EM 判定だが OJM 候補 (補正済の可能性)
    "low_conf",            # CNN 低確信度 (補正後でも残る)
    "unknown_recognized",  # 認識結果 ??
    "flicker",             # 連続 frame で色が振動 (補正で安定済の可能性)
)


def parse_reasons(reasons_str: str) -> set[str]:
    """suspicious_reasons の base name (引数前の名前) を集合で返す。"""
    out: set[str] = set()
    for r in reasons_str.split(";"):
        r = r.strip()
        if not r:
            continue
        # "em_but_saturated(S=139,V=234)" → "em_but_saturated"
        base = r.split("(", 1)[0]
        out.add(base)
    return out


def classify_cell(reasons: set[str]) -> str:
    """cell の suspicious 状態を 3 カテゴリに分類。

    Returns:
        "clean": suspicious なし → 信頼できる
        "hard": hard 違反あり → 真の誤りの可能性大
        "soft_only": soft 警告のみ → false positive の可能性大
    """
    if not reasons:
        return "clean"
    has_hard = any(r in HARD_VIOLATION_REASONS for r in reasons)
    if has_hard:
        return "hard"
    return "soft_only"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", required=True, type=Path)
    args = parser.parse_args()

    if not args.labels.exists():
        print(f"ERROR: {args.labels} not found")
        return 1

    # 全 cell を分類
    total = 0
    chain_total = 0  # 連鎖中 cell (除外対象)
    counts: Counter = Counter()  # clean / hard / soft_only
    hard_reason_counts: Counter = Counter()
    soft_reason_counts: Counter = Counter()
    per_frame: dict[str, dict[str, int]] = defaultdict(
        lambda: {"clean": 0, "hard": 0, "soft_only": 0, "chain": 0},
    )

    with args.labels.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            total += 1
            t = r["time"]
            is_chain = r.get("is_chain") == "1"
            if is_chain:
                chain_total += 1
                per_frame[t]["chain"] += 1
                continue
            reasons = parse_reasons(r.get("suspicious_reasons", ""))
            category = classify_cell(reasons)
            counts[category] += 1
            per_frame[t][category] += 1
            if category == "hard":
                for reason in reasons:
                    if reason in HARD_VIOLATION_REASONS:
                        hard_reason_counts[reason] += 1
            elif category == "soft_only":
                for reason in reasons:
                    if reason in SOFT_WARNING_REASONS:
                        soft_reason_counts[reason] += 1

    non_chain_total = total - chain_total
    print("===== 連続 frame 自動評価 =====")
    print(f"labels: {args.labels}")
    print(f"全 cell: {total}")
    print(f"連鎖中 cell (除外): {chain_total} "
          f"({100.0 * chain_total / total:.1f}%)")
    print(f"評価対象 cell: {non_chain_total}")
    print()
    print(f"clean (suspicious なし): {counts['clean']:6d} "
          f"({100.0 * counts['clean'] / non_chain_total:.2f}%)")
    print(f"hard violations:         {counts['hard']:6d} "
          f"({100.0 * counts['hard'] / non_chain_total:.2f}%)")
    print(f"soft only:               {counts['soft_only']:6d} "
          f"({100.0 * counts['soft_only'] / non_chain_total:.2f}%)")
    print()
    estimated_acc = 100.0 * (1 - counts["hard"] / non_chain_total)
    print(f"推定 accuracy (hard 違反のみカウント): {estimated_acc:.3f}%")
    print(f"目標 99.5% との差分: {estimated_acc - 99.5:+.3f}pt")
    print()

    if hard_reason_counts:
        print("=== hard violation 内訳 ===")
        for reason, n in sorted(
            hard_reason_counts.items(), key=lambda x: -x[1],
        ):
            print(f"  {reason:20s} : {n}")
        print()

    if soft_reason_counts:
        print("=== soft warning 内訳 (補正済の可能性大) ===")
        for reason, n in sorted(
            soft_reason_counts.items(), key=lambda x: -x[1],
        ):
            print(f"  {reason:20s} : {n}")
        print()

    # 違反が集中する frame top 10
    frame_violations = sorted(
        per_frame.items(),
        key=lambda x: -x[1]["hard"],
    )[:10]
    if frame_violations and frame_violations[0][1]["hard"] > 0:
        print("=== hard 違反が多い frame top 10 ===")
        print(f"  {'time':<8} {'chain':<6} {'clean':<6} "
              f"{'hard':<5} {'soft':<5}")
        for t, c in frame_violations:
            if c["hard"] == 0:
                break
            print(
                f"  {t:<8} {c['chain']:<6} {c['clean']:<6} "
                f"{c['hard']:<5} {c['soft_only']:<5}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
