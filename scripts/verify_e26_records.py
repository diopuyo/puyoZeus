"""追加した予測用盤面以外がE22固定入力と完全一致することを検査する。"""
from __future__ import annotations

from itertools import zip_longest
import json
from pathlib import Path
from src.exchange_event_record import read_records, encode
from scripts.run_e3_exchange_eval_20260926 import SOURCES, save_json, digest

OUT = Path('logs/e26')


def verify(source: str) -> dict:
    """得点・状態・確定盤面・ラベル・静止特徴を全行照合する。"""
    before = Path('logs/e16/records')/f'{source}.jsonl.gz'
    after = OUT/'records'/before.name
    rows, observations = 0, 0
    for left, right in zip_longest(read_records(before), read_records(after)):
        assert left is not None and right is not None
        if right['kind'] == 'update':
            for side in (right['args'][0].p1, right['args'][0].p2):
                observations += side.midchain_board is not None
                del side.midchain_board
        # 欠測NaNの位置も比較する（NaN同士の数値比較は常にFalseになる）。
        assert json.dumps(encode(left), sort_keys=True) == json.dumps(encode(right), sort_keys=True), (source, rows)
        rows += 1
    return dict(rows=rows, observations=observations, original_fields='identical',
                before_sha256=digest(before), after_sha256=digest(after))


def main() -> None:
    """補完記録5本の入力不変証拠を保存する。"""
    save_json(OUT/'RECORD_EQUIVALENCE.json', {s: verify(s) for s in (*SOURCES, 'zenchi', 'review')})


if __name__ == '__main__':
    main()
