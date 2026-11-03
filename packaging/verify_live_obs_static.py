"""保存された実盤面で、親側R静的特徴量の全列・会計合成の厳密同値を確認する。"""
from __future__ import annotations
from dataclasses import asdict
import gzip
import json
import os
from pathlib import Path
import pickle
import sys
from types import SimpleNamespace
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from src.board import Board
from src.phase_j.live_cache import bounded_chain_caches
from scripts import visualize_advantage_overlay as overlay


def boards() -> list:
    result, seen = [], set()
    path = Path('D:/puyo_analyzer/packaging/build_20261019a/packets/80_7.jsonl.gz')
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        for line in stream:
            for raw in json.loads(line)['raw']:
                value = raw.get('confirmed_board')
                if value is None:
                    continue
                grid = value['board']['array']
                key = repr(grid)
                if key not in seen:
                    seen.add(key)
                    result.append(Board.from_list(grid))
    return result


def evaluate(values: list, exact: bool) -> list:
    for flag in ('NATIVE', 'CONNECTIVITY'):
        os.environ['PUYO_LIVE_EXACT_'+flag] = str(int(exact))
    os.environ['PUYO_LIVE_NEAR_FUTURE_CACHE'] = str(int(exact))
    overlay._exchange_board_features.cache_clear()
    with bounded_chain_caches(future_search=exact):
        return [asdict(overlay._exchange_static_input((board, values[-1-i]),
                SimpleNamespace(net_balance_capped=i-30, forecast_p1=i), i/3, .5))
                for i, board in enumerate(values)]


def main() -> None:
    out = Path(sys.argv[1])
    values = boards()
    chosen = values  # 保存試合に現れる全ての異なる確定盤面。
    expected, actual = evaluate(chosen, False), evaluate(chosen, True)
    differences = [i for i, (a, b) in enumerate(zip(expected, actual)) if pickle.dumps(a) != pickle.dumps(b)]
    receipt = dict(unique_boards=len(values), compared=len(chosen), all_static_fields_exact=not differences,
                   mismatches=differences, scope='同一保存盤面・会計・経過時刻。モデル・探索候補・予算は変更しない。')
    out.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding='utf-8')
    print(receipt, flush=True)
    assert not differences


if __name__ == '__main__':
    main()
