"""評価値を変更せず、同じ試合の観測済み決着だけを短時間公開する。"""
from __future__ import annotations

from typing import Any
from src.production_config import TERMINAL_HOLD_INPUT_RECOVERY_DEFAULT

# user決定 2026-10-12: 次の試合の始まり（試合番号の変化）で解除する。
# 区切りを取りこぼした時の誤表示は上限5秒。再観測では延長しない。
TERMINAL_DISPLAY_HOLD_SEC = 5.0


class TerminalDisplayHold:
    """試合番号と機器を境界にし、期限切れの同一試合は再武装しない。"""

    def __init__(self, input_recovery: bool = TERMINAL_HOLD_INPUT_RECOVERY_DEFAULT) -> None:
        self.input_recovery = input_recovery
        self.carried = False
        self.game: int | None = None
        self.device: str | None = None
        self.epoch: int | None = None
        self.row: dict[str, Any] | None = None
        self.deadline: float | None = None
        self.invalidated_at = float('-inf')
        self.armed = True
        self.last_conf = False

    def input_changed(self, status: dict[str, Any]) -> None:
        """同じ機器の検証保留は許容し、不明な世代変更は安全側で解除する。"""
        device, epoch = status.get('device_id'), status.get('epoch')
        changed = (self.device is not None and device != self.device) or (
            self.epoch is not None and epoch != self.epoch
            and (device is None or device != self.device))
        if changed:
            self.row, self.deadline, self.game = None, None, None
            self.invalidated_at = status['at']
            self.armed = False
            self.carried = False
        self.device, self.epoch = device, epoch

    def observe(self, row: dict[str, Any], now: float) -> None:
        """latest上書き前に確定を取り込み、古い機器・古い試合の遅着は無視する。"""
        if not self.accepts(row):
            return
        game = row['game']
        normal = (row.get('source') != 'confirmed_death' and row.get('raw_probability') is not None
                  and row.get('match_active', True) and not row.get('evaluation_error'))
        if game != self.game:
            # 旧決着の持越しだけを遮断し、入力保留で増えた番号の初確定は許可する。
            # 保持期限が切れていても、旧試合で保持済みなら通常評価まで再武装しない。
            if self.game is not None:
                self.armed = not (self.last_conf or self.deadline is not None)
            carry = (self.input_recovery and self.active(now) and not normal
                     and row.get('terminal_boundary_reason') == 'input_recovery')
            self.game = game
            self.carried = carry
            if not carry:
                self.row, self.deadline = None, None
        if self.carried and (normal or row.get('terminal_boundary_reason') == 'scored'):
            self.row, self.deadline, self.carried = None, None, False
        if normal:
            self.armed = True
        if (self.armed and self.deadline is None and row.get('source') == 'confirmed_death'
                and row.get('probability') in (0., 1.)
                and row.get('raw_probability') is not None):
            self.row = dict(row)
            self.deadline = now + TERMINAL_DISPLAY_HOLD_SEC
        self.last_conf = row.get('source') == 'confirmed_death'

    def accepts(self, row: dict[str, Any]) -> bool:
        """古い入力は公開番号も巻き戻さない。"""
        return (row['captured_at'] > self.invalidated_at
                and (self.game is None or row['game'] >= self.game))

    def active(self, now: float) -> bool:
        return self.row is not None and self.deadline is not None and now < self.deadline

    def project(self, row: dict[str, Any], now: float) -> dict[str, Any]:
        """公開用コピーだけを作る。期限後の確定再到着も表示しない。"""
        if row['game'] == self.game and self.active(now):
            return dict(self.row, _terminal_display_held=True)
        if (row.get('source') == 'confirmed_death' and (self.deadline is not None or not self.armed)
                or row['captured_at'] <= self.invalidated_at
                or self.game is not None and row['game'] < self.game):
            return dict(row, raw_probability=None, match_active=False)
        return row
