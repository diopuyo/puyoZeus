"""終了・置きのない短い再通知で連鎖と台帳の起点を保つ試作（既定OFF）。"""
from __future__ import annotations

import os
ENV = 'PUYO_EX_CHAIN_IDENTITY'
INPUT_FPS = 30
# 保存記録の段数増加の空白は最大19コマ。1コマの余裕だけを設ける。
MAX_GAP_FRAMES = 20
MAX_GAP_SEC = MAX_GAP_FRAMES / INPUT_FPS
TIME_EPSILON = 1e-6
MIN_COUNT = 2
MIN_ADVANCING_COUNT = 3
STATE_KEY = '_identity_observation'


def enabled() -> bool:
    return os.environ.get(ENV, '0') == '1'


def remember(chain: dict, event: dict, stamp: float, source: dict) -> None:
    """OFFでは既存の状態辞書にも項目を追加しない。"""
    if enabled():
        chain[STATE_KEY] = dict(t=stamp, trigger=event['trigger_sec'],
                               count=event['chain_count'], board=event.get('before_board'),
                               tsumo=source.get('tsumo_count'))


def continuation(chain: dict, previous: dict, event: dict, stamp: float, source: dict) -> bool:
    """初期段の増加は新発火と区別できないため、3段以上の進みだけ許す。"""
    old_count, count = previous['count'], event['chain_count']
    gap = stamp - previous['t']
    old_tsumo, tsumo = previous['tsumo'], source.get('tsumo_count')
    placed = old_tsumo is not None and tsumo is not None and tsumo > old_tsumo
    return (not chain.get('finished', False) and not chain.get('external', False)
            and (MIN_COUNT <= count == old_count
                 or MIN_ADVANCING_COUNT <= old_count < count)
            and 0 <= gap <= MAX_GAP_SEC + TIME_EPSILON
            and not placed and not source.get('next_slide_motion', False))


def normalize(chain: dict | None, event: dict, stamp: float, source: dict) -> dict:
    """入力を変更せず、同じ連鎖と確認できた通知の発火時刻だけを保つ。"""
    if not enabled() or chain is None or chain.get('external') or event['trigger_sec'] > stamp:
        return event
    previous = chain.get(STATE_KEY)
    keep = previous is not None and (
        previous['trigger'] == event['trigger_sec']
        or continuation(chain, previous, event, stamp, source))
    remember(chain, event, stamp, source)
    if keep:
        return dict(event, trigger_sec=chain['signature'][0])
    return event
