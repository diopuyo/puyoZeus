"""R1b・E35bの描画ファイルを編集せず、D5既定OFFフラグを接続する。"""
from __future__ import annotations

import argparse
from functools import partial
import sys


def main() -> None:
    """D5の指定だけ消費し、E35と本番共通CLIへ引き継ぐ。"""
    from scripts import visualize_advantage_overlay as render
    from scripts import visualize_e35_overlay as e35
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--single-death-proof-guard', action='store_true', default=False)
    parser.add_argument('--single-death-proof-negative-only', action='store_true', default=False)
    options, remaining = parser.parse_known_args()
    if options.single_death_proof_guard:
        render.ExchangeEventOverlay = partial(render.ExchangeEventOverlay, single_death_proof_guard=True,
            single_death_proof_negative_only=options.single_death_proof_negative_only)
    sys.argv = [sys.argv[0], *remaining]
    e35.main()


if __name__ == '__main__':
    main()
