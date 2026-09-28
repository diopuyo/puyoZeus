"""E33: 観測した段の時刻と絶対終了信号だけで長すぎる候補を除く。"""
from __future__ import annotations

from typing import Any
from src.chain import ChainSimulator
from src.exchange_hidden_row_belief import HiddenRowPrefire
from src.score_ocr import FORMULA_STEP_CONFIRM_FRAMES

# 23動画418イベントの実測式 2.61 + 1.17×N の段当たり時間。
# 出典: docs/EXCHANGE_EPISODE_SPEC_2026-08-24.md、chain_id_resolver.py。
STAGE_INTERVAL_SEC = 1.17
# score_ocr.py の実測幕間上限0.634秒を余裕として丸め上げる。
# 加えて30Hz認識での式確定2フレームを待つ。対象シーンからは調整しない。
INTERSTAGE_MARGIN_SEC = 0.65
OBSERVATION_HZ = 30.0
STAGE_TIMEOUT_SEC = STAGE_INTERVAL_SEC + INTERSTAGE_MARGIN_SEC + FORMULA_STEP_CONFIRM_FRAMES / OBSERVATION_HZ
ABSOLUTE_END_REASONS = ('next', 'slide', 'ojama')


class StageTimeoutPrefire(HiddenRowPrefire):
    """E32の得点照合後に、次段の不在という因果観測を追加する。"""

    def __init__(self, simulator: ChainSimulator, timeout_only: bool = False) -> None:
        super().__init__(simulator)
        self.timeout_only = timeout_only

    def _filter(self, entry: dict, key: tuple, stamp: float) -> None:
        """同じ段の再通知では時計を延長しない。"""
        previous = entry['last']
        super()._filter(entry, key, stamp)
        if previous is None or key[0] > previous[0]:
            entry['stage_sec'] = stamp

    def observe(self, overlay: Any, result: Any, stamp: float) -> None:
        """式の更新を先に処理し、同じフレームの新段を除外しない。"""
        super().observe(overlay, result, stamp)
        for idx, label in enumerate(('1P', '2P')):
            chain = overlay.tracker.latest_chain(label)
            entry = self.entries.get(chain.chain_id) if chain else None
            if entry is None or not entry['options'] or 'stage_sec' not in entry:
                continue
            absolute = (not self.timeout_only and chain.end_signal_sec is not None and chain.end_signal_sec <= stamp
                        and chain.end_reason in ABSOLUTE_END_REASONS)
            visible = overlay._last_formula[idx] == stamp
            expired = stamp > entry['stage_sec'] + STAGE_TIMEOUT_SEC
            if absolute or (expired and not visible):
                self._exclude_later(entry, stamp, chain.end_reason if absolute else 'stage_timeout')

    def _exclude_later(self, entry: dict, stamp: float, reason: str) -> None:
        """全滅は従来予測へ撤回し、初回採用得点は書き換えない。"""
        count = entry['last'][0]
        before = entry['options']
        kept = [v for v in before if len(v['prefix']) <= count]
        if len(kept) == len(before):
            return
        entry['options'] = kept
        self._normalize(kept)
        entry['audit'].setdefault('stage_exclusions', []).append(dict(
            t_sec=stamp, reason=reason, observed_count=count, stage_sec=entry['stage_sec'],
            deadline_sec=entry['stage_sec'] + STAGE_TIMEOUT_SEC,
            removed=len(before)-len(kept), remaining=len(kept),
            before_score=entry['stats']['mean_score']))
        if not kept:
            entry['audit'].update(withdrawn=True, withdraw_sec=stamp, withdraw_reason=reason)
        self.revision += 1
        value = self._publish(entry)
        entry['audit']['stage_exclusions'][-1]['after_score'] = value['mean_score']
