"""入力待機中もamaの最後の同一試合値と実時計の保持秒数を公開する。"""
from __future__ import annotations
from typing import Any


class AmaPublication:
    def __init__(self) -> None:
        self.game: Any = None
        self.value: dict | None = None
        self.captured = 0.

    def observe(self, row: dict, now: float) -> None:
        game = row.get('game')
        if game != self.game:
            self.game, self.value = game, None
        if row.get('ama_plain') is not None:
            self.value = dict(row['ama_plain'])
            self.captured = row.get('captured_at',now)

    def project(self, row: dict, now: float) -> dict:
        if self.value is None or row.get('game') != self.game:
            return row
        value = dict(self.value)
        if value.get('age_sec') is not None:
            value['age_sec'] += max(0.,now-self.captured)
        if row.get('input_calibration',{}).get('phase','ready') != 'ready':
            value['reason'] = '入力確認中・直前値を保持'
        elif row.get('evaluation_error'):
            value['reason'] = '評価処理エラー・直前値を保持'
        return dict(row,ama_plain=value)
