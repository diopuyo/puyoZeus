"""最後の有効な値を無期限保持する独立表示器。既定OFF、試合ごとに新規生成。"""
from __future__ import annotations


class Display:
    def __init__(self, enabled: bool = False) -> None:
        self.enabled = enabled
        self.latest: dict | None = None
        self.valid: dict | None = None
        self.last = float('-inf')

    def observe(self, result: dict) -> None:
        if result['input_sec'] < self.last:
            raise ValueError('評価入力の時刻逆転')
        self.last = result['input_sec']
        if not self.enabled:
            return
        self.latest = result
        if result.get('value') is not None:
            self.valid = result

    def at(self, stamp: float, states: list) -> dict:
        if stamp < self.last:
            raise ValueError('未来の評価入力')
        valid, latest = self.valid, self.latest
        if states == ['CHAIN', 'CHAIN']:
            reason = '両側連鎖中・片側終了待ち'
        elif latest and latest.get('missing'):
            reason = latest['missing']
        elif 'MENU' in states:
            reason = '試合外'
        elif 'CHAIN' in states:
            reason = '連鎖中・相手の設置待ち'
        elif 'OJAMA_FALL' in states:
            reason = 'おじゃま着地待ち'
        else:
            reason = '次の確定盤待ち'
        return dict(probability=valid['value']['probability'] if valid else None,
            input_sec=valid['input_sec'] if valid else None,
            attempted_sec=latest['input_sec'] if latest else None,
            hold_sec=stamp-valid['input_sec'] if valid else None,
            hold_reason=reason, result=valid,
            missing_reason=latest.get('missing') if latest else '初回入力待ち')
