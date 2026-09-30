"""描画と再生で共通の本番撃ち合い評価オプション。"""
from __future__ import annotations

import argparse
import shlex
import sys

from src import production_config


RECONCILE_OPTION = "--placement-signal-reconcile"


def parse_exchange_event_args(
    parser: argparse.ArgumentParser, argv: list[str] | None = None,
) -> argparse.Namespace:
    """指定時だけ本番構成を実行時に読み、同じ項目の個別指定より優先する。"""
    parser.add_argument(
        "--production-exchange-event", action="store_true",
        help="採用済み撃ち合い評価を一括適用（モデル等の個別指定より優先）",
    )
    arguments = list(sys.argv[1:] if argv is None else argv)
    options = parser.parse_args(arguments)
    if options.production_exchange_event:
        arguments.extend(shlex.split(production_config.exchange_event_flags()))
        if RECONCILE_OPTION in parser._option_string_actions:
            # 描画CLIだけが認識を動かすので、受け付けるパーサーにだけ追加する。
            arguments.extend(shlex.split(production_config.placement_reconcile_flags()))
        options = parser.parse_args(arguments)
    return options
