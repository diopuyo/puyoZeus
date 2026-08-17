"""認識強化の統一測定 (2026-08-17) の後段一括実行ドライバ。

B/C/D の npz 収集 (scripts/_collect_yardstick_v2_r2w10_2026-08-17.py) が
完了した後に実行する。以下を順に実行する:
    1. score_a/b/c/d の採点 (scripts/_score_yardstick_v2_r2w10_2026-08-17.py)
    2. 4構成比較 (--compare)
    3. 持続誤認診断 (scripts/_diag_persistent_misread_2026-08-17.py --all)

使い方:
    PYTHONPATH=. ./venv/bin/python -m scripts._run_unified_measurement_2026-08-17
"""
from __future__ import annotations

import importlib

_score = importlib.import_module("scripts._score_yardstick_v2_r2w10_2026-08-17")
_persist = importlib.import_module("scripts._diag_persistent_misread_2026-08-17")


def main() -> None:
    print("=== 1. 採点 (a/b/c/d/e) ===")
    _score._ensure_score_a()
    for tag in ("b", "c", "d", "e"):
        _score.score_one(tag)

    print("\n=== 2. 5構成比較 ===")
    _score.compare_all()

    print("\n=== 3. 持続誤認診断 (1次目標②) ===")
    for tag in ("a", "b", "c", "d", "e"):
        result = _persist.analyze_tag(tag)
        out_path = _persist.OUT_DIR / f"persistent_misread_{tag}.json"
        _persist.OUT_DIR.mkdir(parents=True, exist_ok=True)
        import json
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(
            f"[{tag}] 誤りセル総数={result['n_wrong_cells_total']} "
            f"持続誤認(>=5f)={result['n_persistent']} "
            f"反映遅延別枠(<=8f)={result['n_reflection_delay']}"
        )

    print("\n[done] 全出力完了")


if __name__ == "__main__":
    main()
