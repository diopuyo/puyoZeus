"""R1bの描画実装を編集せず、E35の明示フラグを既存描画へ接続する。"""
from __future__ import annotations

import argparse
from functools import partial
import sys


def main() -> None:
    """専用ラッパーのフラグだけを消費し、残りは本番共通CLIへ渡す。"""
    from scripts import visualize_advantage_overlay as render
    from scripts.run_e3_exchange_eval_20260926 import SEED
    import random
    import numpy as np
    import torch
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--post-counter-death-bound', action='store_true', default=False)
    options, remaining = parser.parse_known_args()
    if options.post_counter_death_bound:
        render.ExchangeEventOverlay = partial(render.ExchangeEventOverlay, post_counter_death_bound=True)
    sys.argv = [sys.argv[0], *remaining]
    render.main()


if __name__ == '__main__':
    main()
