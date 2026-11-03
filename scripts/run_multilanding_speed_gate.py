"""遅延対策 (logs/multilanding_speed) のゲート: 保存記録の再生だけで、現本番と同じ門で採点する。

認識は変えないので収集は行わず、現本番の保存記録 (logs/pending_expiry/full/records) を再利用する。
現行 src (E35b 高速版の証明器を含む) をそのまま読み、0f28a04 固定版への差し替え (d5_runtime) は行わない。
段: replay (1 記録 1 プロセス) → report (E36b の門: q / zenchi / 誤発火 / 場面 / 第14試合)。
variant:
  exact    現本番と同一のオプション。出力が現本番と完全一致するかの確認用 (門は現本番と同じ)。
  bounded  B1: 既定OFFの遅延対策を全て ON (docs/agent_coordination/DECISIONS.md の事前登録の値)。
  bounded2 B2: B1 から応手全探索のノード上限を外したもの (B1 が門を落とした場合だけ使う)。
使い方: python -m scripts.run_multilanding_speed_gate --variant bounded --step replay --source S
        python -m scripts.run_multilanding_speed_gate --variant bounded --step report
"""
from __future__ import annotations

import argparse
from pathlib import Path

from scripts import run_c65_guard_full as base
from scripts import run_e36b
from scripts import run_pending_expiry_full as expiry

OUT_ROOT = Path('logs/multilanding_speed')
# 事前登録 (docs/agent_coordination/DECISIONS.md 2026-09-30) の値。実走前に固定する。
NODE_LIMIT = 5000       # 応手全探索の総ノード上限 (従来 20000)
SCENARIO_CAP = 256      # 隠し段の得点候補の上限 (従来は全組合せ)
VARIANTS = ('exact', 'bounded', 'bounded2')


def out_dir(variant: str) -> Path:
    """variant ごとの隔離出力先。既存 logs は書き換えない。"""
    return OUT_ROOT / f'gate_{variant}'


def options(variant: str) -> dict:
    """再生オプション。exact は現本番 (E36b) と同一。"""
    current = run_e36b.options()
    if variant == 'exact':
        return current
    bounded = dict(current, post_counter_early_exit=True, hidden_scenario_cap=SCENARIO_CAP)
    return dict(bounded, multilanding_node_limit=NODE_LIMIT) if variant == 'bounded' else bounded


def replay(variant: str, source: str) -> None:
    """現行 src を固定版に差し替えずに再生する。"""
    from scripts import d5_runtime, run_e36
    expiry.configure()
    d5_runtime.install = lambda: None  # 0f28a04 固定版へ差し替えない (現行 src の証明器を使う)
    run_e36.OUT, run_e36.options = out_dir(variant), lambda: options(variant)
    run_e36.RECORDS = expiry.COLLECT_OUT / 'records'
    run_e36.worker(source)


def report(variant: str) -> None:
    """E36b の採点器を出力先だけ差し替えて実行する。"""
    from scripts import report_e36b
    report_e36b.OUT = out_dir(variant)
    report_e36b.report()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', choices=VARIANTS, required=True)
    parser.add_argument('--step', choices=('replay', 'report'), required=True)
    parser.add_argument('--source', choices=base.SOURCES)
    args = parser.parse_args()
    if args.step == 'replay':
        assert args.source, '--source が必要'
        replay(args.variant, args.source)
    else:
        report(args.variant)


if __name__ == '__main__':
    main()
