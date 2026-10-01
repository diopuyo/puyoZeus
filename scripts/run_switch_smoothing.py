"""評価器切替の表示平滑 (--exchange-event-switch-smoothing) と E19 の再生・採点 (認識は不変、保存記録の再生のみ)。

現本番 (pending_expiry の E36b 構成) と同じ再生・同じ採点器で、条件だけを差し替える。
出力は logs/switch_smoothing/<variant>/ に隔離し、既存 logs は書き換えない。
使い方: python -m scripts.run_switch_smoothing --variant off|a|b|e19|a_e19 --step replay|report [--source S]
事前登録: exev docs/agent_coordination/DECISIONS.md (2026-10-01 事前登録)。
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from dataclasses import dataclass
from typing import Any

from scripts import replay_exchange_event_20260926 as replay_module
from scripts import run_d5, run_e36, run_e36b
from scripts.run_e3_exchange_eval_20260926 import save_json
from src.exchange_display_smoothing import SwitchAwareDisplayEMA
from src.exchange_event_record import encode

# 入力の保存記録 (現本番の収集、読取専用)。環境変数で差し替え可能。
EXEV_ROOT = Path(os.environ.get(
    "SWITCH_EXEV_ROOT", "/mnt/c/Users/ryouj/.codex/worktrees/exev/puyo_analyzer"))
COLLECT_RECORDS = EXEV_ROOT / "logs/pending_expiry/full/records"
OUT_ROOT = Path("logs/switch_smoothing")
SOURCES = run_e36.SOURCES
# 変種ごとの追加オプション (再生関数の引数名)。off は現本番そのもの。
VARIANTS: dict[str, dict[str, bool]] = {
    "off": {},
    "a": dict(switch_smoothing=True),
    "b": dict(switch_smoothing=True),   # 構成B: raw_waiting=True の平滑 (A が門で落ちた場合のみ)
    "e19": dict(landing_counter_prob=True),
    "a_e19": dict(switch_smoothing=True, landing_counter_prob=True),
}


@dataclass
class ConfigB(SwitchAwareDisplayEMA):
    """事前登録の構成B: waiting_confirmed は旧評価器の生値を表示 (EMA状態だけ合わせる)。"""

    raw_waiting: bool = True


def configure(variant: str) -> Path:
    """E36b の再生器を出力先・記録・オプションだけ差し替えて構成する。"""
    out = OUT_ROOT / variant
    out.mkdir(parents=True, exist_ok=True)
    extra = VARIANTS[variant]
    replay_module.SwitchAwareDisplayEMA = ConfigB if variant.startswith("b") else SwitchAwareDisplayEMA
    run_e36.OUT = out
    run_e36.RECORDS = COLLECT_RECORDS
    run_e36.options = lambda: dict(run_e36b.options(), **extra)
    return out


def replay(variant: str, source: str) -> None:
    """1 記録を 1 変種で再生する (run_e36.worker と同一手順。ただし証明器は現行 src を使う)。

    run_e36.worker は 0f28a04 版の証明器を固定して読むが、現行の PostCounterDeathBound は
    early_exit 引数を要求するため併用できない。26358a5 (E35b) は「判定不変・E36b 5記録で
    display/events バイト一致」と記録されており、本ランナーでも OFF 再生が保存済み出力と
    バイト一致することを採点前の健全性条件にしている。
    """
    out = configure(variant)
    prior = run_e36.baseline.e31.prior
    prior.OUT, prior.AuditTrace = out, run_d5.Trace

    def enriched(record: Path, dest: Path, *args: Any, **kwargs: Any) -> dict:
        result = replay_module.replay(COLLECT_RECORDS / record.name, dest, *args, **kwargs)
        save_json(dest / "snapshot_final_audit.json", args[2].summary())
        save_json(dest / "scene_timeline.json", encode(args[2].timeline))
        return result
    prior.replay = enriched
    prior.locked_worker(run_e36.VARIANT, source, run_e36.options())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=tuple(VARIANTS), required=True)
    parser.add_argument("--step", choices=("replay",), required=True)
    parser.add_argument("--source", choices=SOURCES, required=True)
    args = parser.parse_args()
    replay(args.variant, args.source)


if __name__ == "__main__":
    main()
