"""描画と再生で共通の本番撃ち合い評価オプション。"""
from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path

from src import production_config


RECONCILE_OPTION = "--placement-signal-reconcile"


def parse_exchange_event_args(
    parser: argparse.ArgumentParser, argv: list[str] | None = None,
) -> argparse.Namespace:
    """指定時だけ本番構成を実行時に読み、同じ項目の個別指定より優先する。"""
    parser.add_argument(
        "--exchange-event-static-gfe-dir", type=Path, default=None,
        help="通常静止のG_feだけを指定モデルへ置換（仮想着弾は従来モデル）",
    )
    parser.add_argument(
        "--production-exchange-event", action="store_true",
        help="採用済み撃ち合い評価を一括適用（モデル等の個別指定より優先）",
    )
    arguments = list(sys.argv[1:] if argv is None else argv)
    parser.add_argument('--completion-score-coherent', action='store_true', default=False,
                        help='A1: 完走の得点・段数・盤面を同一候補から取得し、予測不能を因果補完')
    parser.add_argument('--completion-score-max-chain', action='store_true', default=False,
                        help='A1b: 最大段数・同段数なら最大確率の候補を選び、A1と同じ補完を使う')
    options = parser.parse_args(arguments)
    _validate_options(parser, options)
    if options.production_exchange_event:
        arguments.extend(shlex.split(production_config.exchange_event_flags()))
        if RECONCILE_OPTION in parser._option_string_actions:
            # 描画CLIだけが認識を動かすので、受け付けるパーサーにだけ追加する。
            arguments.extend(shlex.split(production_config.placement_reconcile_flags()))
        options = parser.parse_args(arguments)
        _validate_options(parser, options)
    validate_v2_options(parser, options)
    return options


def _validate_options(parser: argparse.ArgumentParser, options: argparse.Namespace) -> None:
    """描画・再生の候補上限をAPIと同じ契約で検証する。"""
    from src.exchange_hidden_row_probability import validate_scenario_cap
    try:
        validate_scenario_cap(getattr(options, "hidden_scenario_cap", None))
    except ValueError as exc:
        parser.error(str(exc))


def validate_v2_options(parser: argparse.ArgumentParser, options: argparse.Namespace) -> None:
    """本番構成適用後にE35依存を検査し、無効な実験指定を拒否する。"""
    if getattr(options, 'a2_safety_review', False) and not getattr(options, 'death_integrity_v2h2', False):
        parser.error('--a2-safety-reviewには--death-integrity-v2h2が必要')
    if getattr(options, 'death_history_boundary_start', False) and not getattr(options, 'death_history_integrity', False):
        parser.error('--death-history-boundary-startには--death-history-integrityが必要')
    if getattr(options, 'death_integrity_v2h2', False) and not all(getattr(options, n, False) for n in (
            'death_history_integrity', 'death_history_boundary_start', 'post_counter_completion_recent_integrity')):
        parser.error('--death-integrity-v2h2にはH1・H1′・E35⁗′が必要')
    if getattr(options, 'death_freshness_cap', False) and not getattr(options, 'death_integrity_v2h2', False):
        parser.error('--death-freshness-capには--death-integrity-v2h2が必要')
    names = ('score_attribution_v2', 'post_counter_input_integrity_v2',
             'post_counter_current_puyo_integrity', 'post_counter_current_puyo_integrity_v2c')
    if any(getattr(options, name, False) for name in names):
        if not getattr(options, 'post_counter_death_bound', False):
            parser.error('D-v2系フラグには--post-counter-death-boundが必要')
    if all(getattr(options, name, False) for name in names[-2:]):
        parser.error('E35′とE35″は同時指定できない')
    if getattr(options, 'post_counter_completion_prefix_integrity', False) and not (
            getattr(options, 'post_counter_death_bound', False)
            and getattr(options, 'post_counter_current_puyo_integrity_v2c', False)):
        parser.error('E35‴には--post-counter-death-boundと--post-counter-current-puyo-integrity-v2cが必要')
    if getattr(options, 'post_counter_completion_step_integrity', False):
        if not (getattr(options, 'post_counter_death_bound', False)
                and getattr(options, 'post_counter_current_puyo_integrity_v2c', False)):
            parser.error('E35⁗には--post-counter-death-boundと--post-counter-current-puyo-integrity-v2cが必要')
        if getattr(options, 'post_counter_completion_prefix_integrity', False):
            parser.error('E35‴とE35⁗は同時指定できない')
    if getattr(options, 'post_counter_completion_recent_integrity', False):
        if not (getattr(options, 'post_counter_death_bound', False)
                and getattr(options, 'post_counter_current_puyo_integrity_v2c', False)):
            parser.error('E35⁗′には--post-counter-death-boundと--post-counter-current-puyo-integrity-v2cが必要')
        if any(getattr(options, name, False) for name in (
                'post_counter_completion_prefix_integrity', 'post_counter_completion_step_integrity')):
            parser.error('E35⁗′と他の途中経過照合は同時指定できない')
