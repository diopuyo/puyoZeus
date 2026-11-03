"""封印セットの一括開封 CLI (事前登録 §6)。**既定は dry-run: 固定ファイルの照合と token の表示だけ。採点しない。**

実開封は統括の指示後に --execute --token <表示された token> --scorer module:function で行う。
scorer は set 名 -> 結果 dict の関数で、全セット分を 1 回の呼出で一括採点する (途中停止・差し替えなし)。
このセッションでは実行しない (封印の成績は計算しない)。
"""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

from src.b3_eval_sealed import OpeningPlan, build_manifest, open_sealed

PINNED_DEFAULT = ("TARGETS.json", "TIER_EVIDENCE.json", "AUDIT.json", "SEALED.md", "receipts/43.json",
                  "receipts/sealed01.json", "receipts/sealed02.json")
SET_NAMES = ("v43", "sealed01", "sealed02")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True, help="sealed_eval_2026-10-03 ディレクトリ")
    ap.add_argument("--manifest", type=Path, required=True, help="固定 manifest (relpath->sha256) の JSON。無ければ --write-manifest")
    ap.add_argument("--write-manifest", action="store_true", help="現状の hash で manifest を新規作成 (既存なら拒否)")
    ap.add_argument("--plan", type=Path, help="OpeningPlan の JSON (候補/R/合格条件/ラベルの hash・実行者・目的)")
    ap.add_argument("--record", type=Path, help="開封記録の出力先 (既存なら再開封を拒否)")
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--token")
    ap.add_argument("--scorer", help="module:function")
    a = ap.parse_args()
    if a.write_manifest:
        a.manifest.write_text(json.dumps(build_manifest(a.root, PINNED_DEFAULT), indent=1), encoding="utf-8")
        print("manifest 作成:", a.manifest)
        return
    manifest = json.loads(a.manifest.read_text(encoding="utf-8"))
    plan = OpeningPlan(**json.loads(a.plan.read_text(encoding="utf-8")))
    scorer = None
    if a.execute:
        if not (a.scorer and a.token and a.record):
            ap.error("--execute には --scorer / --token / --record が必要")
        mod, fn = a.scorer.split(":")
        scorer = getattr(importlib.import_module(mod), fn)
    out = open_sealed(a.root, manifest, plan, SET_NAMES, scorer or (lambda n: {}), a.record or Path("opening_record.json"),
                      token=a.token, dry_run=not a.execute)
    print(json.dumps({k: v for k, v in out.items() if k != "results"}, ensure_ascii=False, indent=1, default=str))


if __name__ == "__main__":
    main()
