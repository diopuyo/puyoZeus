"""E6の記録再生と事前登録済みE5比ゲートを保存する。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts import aggregate_e4_exchange_eval_20260926 as aggregate
from scripts.replay_exchange_event_20260926 import replay
from scripts.run_e3_exchange_eval_20260926 import SOURCES, save_json

M1_LIMIT = .20
M2_TOLERANCE_SEC = .2
LOSS_TOLERANCE = .005
FLIP_RATIO_LIMIT = 1.1
FRESHNESS_TOLERANCE_SEC = 1.0
BASELINE = Path("logs/e5/replay/metrics")


def read(path: Path) -> dict | list:
    """既存集計をそのまま比較する。"""
    return json.loads(path.read_text(encoding="utf-8"))


def gates(out: Path) -> dict:
    """固定閾値を再集計結果へ適用する。"""
    before, after = [read(p / "pooled.json") for p in (BASELINE, out / "metrics")]
    old, new = [read(p / "summary.json") for p in (BASELINE, out / "metrics")]
    result = dict(M1=after["M1"]["revoked_fraction"] <= M1_LIMIT,
        M2=float(after["M2"]["median_seconds"]["on"]) <=
           float(before["M2"]["median_seconds"]["on"]) + M2_TOLERANCE_SEC,
        M3=new[0]["M3"]["on"]["groups"]["all"]["log_loss"] <=
           old[0]["M3"]["on"]["groups"]["all"]["log_loss"] + LOSS_TOLERANCE,
        M4=after["M4"]["on"]["flips_per_minute"] <=
           before["M4"]["on"]["flips_per_minute"] * FLIP_RATIO_LIMIT,
        freshness=all(row["F4"]["on"]["longest_equal_seconds"] <=
                      row["F4"]["off"]["longest_equal_seconds"] + FRESHNESS_TOLERANCE_SEC
                      for row in new))
    result["all_pass"] = all(result.values())
    return result


def main() -> None:
    """全編レンダせず三本を直列再生する。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("logs/e6/final"))
    options = parser.parse_args()
    for source in SOURCES:
        status = replay(Path("logs/e5/renders") / source / "on/inputs.jsonl.gz",
                        options.out / "renders" / source / "on")
        print(source, status, flush=True)
    aggregate.OUT = options.out
    aggregate.main()
    result = gates(options.out)
    save_json(options.out / "gates.json", result)
    print(result, flush=True)


if __name__ == "__main__":
    main()
