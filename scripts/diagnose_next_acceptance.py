"""
段階 4-1: next_acceptance 定数列疑い検証

目的:
    - match_features_v2.csv の next_acceptance 列が定数(=0)になっている
      可能性を統計から検証する。
    - generate_training_dataset.extract_one_sample が next_pair を渡さず
      compute_all(board) を呼ぶため、next_acceptance は中立値 0.5 で固定 →
      差分 (1P-2P) は常に 0 となる仮説を確認する。
    - timeline_analyzer.py に NextDetector が統合されているかをコードで確認。

入力:
    data/training/match_features_v2.csv

出力:
    data/verify/diagnose_next_acceptance.json
        - 統計値 (mean / std / min / max / unique 数 / zero 比率)
        - 原因コードパスの説明
        - 結論 (CONSTANT_ZERO / VARYING)
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

# プロジェクトルートを sys.path に追加
_PROJ_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))

# ============================
# 定数
# ============================

DEFAULT_INPUT_CSV: Path = Path("data/training/match_features_v2.csv")
DEFAULT_OUTPUT_JSON: Path = Path("data/verify/diagnose_next_acceptance.json")
TARGET_COLUMN: str = "next_acceptance"
CONSTANT_TOL: float = 1e-9


# ============================
# CSV 読み込み
# ============================


def load_column_values(csv_path: Path, column: str) -> np.ndarray:
    """指定列の値を float ndarray で取得する。"""
    rows: list[float] = []
    with csv_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if column not in (reader.fieldnames or []):
            raise ValueError(f"列 {column} が見つからない: {csv_path}")
        for r in reader:
            rows.append(float(r.get(column, "0") or 0.0))
    return np.asarray(rows, dtype=np.float64)


# ============================
# 統計計算
# ============================


def compute_stats(values: np.ndarray) -> dict[str, Any]:
    """配列の基本統計を返す。"""
    n = int(values.size)
    if n == 0:
        return {
            "n": 0, "mean": 0.0, "std": 0.0,
            "min": 0.0, "max": 0.0, "unique": 0,
        }
    return {
        "n": n,
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "max": float(values.max()),
        "unique": int(np.unique(np.round(values, decimals=10)).size),
        "abs_mean": float(np.abs(values).mean()),
        "zero_count": int(np.sum(np.abs(values) < CONSTANT_TOL)),
        "zero_ratio": float(np.mean(np.abs(values) < CONSTANT_TOL)),
    }


def is_constant(stats: dict[str, Any]) -> bool:
    """統計から「実質定数」と判定する (std が極小かつ unique<=2)。"""
    return stats["std"] < CONSTANT_TOL and stats["unique"] <= 2


# ============================
# コードパス検証
# ============================


def verify_integration() -> dict[str, Any]:
    """timeline_analyzer / generate_training_dataset で next 検出統合を確認。"""
    paths = {
        "timeline_analyzer": Path("src/timeline_analyzer.py"),
        "generator_v1": Path("scripts/generate_training_dataset.py"),
        "generator_v2": Path("scripts/generate_training_dataset_v2.py"),
    }
    out: dict[str, Any] = {}
    for key, p in paths.items():
        if not p.exists():
            out[key] = {"exists": False}
            continue
        text = p.read_text(encoding="utf-8")
        out[key] = {
            "exists": True,
            "uses_next_detector": "NextDetector" in text or "next_detector" in text,
            "passes_next_pair": "next_pair=" in text,
            "calls_compute_all_no_next": "compute_all(b1)" in text or "compute_all(board)" in text,
        }
    return out


# ============================
# main
# ============================


def main() -> int:
    parser = argparse.ArgumentParser(description="next_acceptance 定数列検証")
    parser.add_argument("--csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT_JSON)
    args = parser.parse_args()

    values = load_column_values(args.csv, TARGET_COLUMN)
    stats = compute_stats(values)
    integration = verify_integration()
    constant = is_constant(stats)

    cause = (
        "extract_one_sample が next_pair=None で compute_all を呼ぶため "
        "next_acceptance は NEUTRAL=0.5 固定 → 差分 (1P-2P) が常に 0"
        if constant else
        "差分が変動。NextDetector 統合が機能している可能性"
    )
    conclusion = "CONSTANT_ZERO" if constant else "VARYING"

    out: dict[str, Any] = {
        "target_column": TARGET_COLUMN,
        "csv_path": str(args.csv),
        "stats": stats,
        "is_constant": constant,
        "conclusion": conclusion,
        "root_cause": cause,
        "integration_check": integration,
        "recommendation": (
            "EXTRA_INDICATOR_NAMES から next_acceptance を外すか、Scorer 重みを 0 "
            "に固定する。あるいは NextDetector 統合した Generator v3 を別途実装"
            if constant else
            "そのまま使用可。VIF 検査を再度実施"
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(f"[stats] mean={stats['mean']:.6f} std={stats['std']:.6f} "
          f"unique={stats['unique']} zero_ratio={stats['zero_ratio']:.4f}")
    print(f"[conclusion] {conclusion}")
    print(f"[save] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
