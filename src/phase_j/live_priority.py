"""遅い PC で表示の遅れを抑える優先度の組み方。既定 OFF (2026-11-15 Claude、最低の目安 i5-8250U 級のまね)。

配布の既定 (evaluation_nice=10) は評価のまとめ (状態の更新・表示) を低い優先度にし、重い探索の子は通常のまま。
コアの少ない PC で OBS とコアを取り合うと、遅れに効く評価側が後回しになる
(PC1 の省電力コア 4 個に OBS と同居: 表示の遅れ 95% 1,023ms、e2e_user/runs/p067a_m)。
ON: 評価のまとめは下げず、重い探索の子 (b3-state・b3-compute・live-ex-state・live-ex-compute) を低くする
(同じ条件で 350ms・OBS の描画の取りこぼし 0・試合の取り逃し 0、e2e_user/runs/p067c_m)。
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import sys

ENV = 'PUYO_LIVE_LATENCY_PRIORITY'
EVENT = 'live_latency_priority'
WINDOWS_BELOW_NORMAL = 0x4000
POSIX_BACKGROUND_NICE = 10


def enabled() -> bool:
    from src.production_config import LIVE_LATENCY_PRIORITY_DEFAULT
    return os.environ.get(ENV, str(int(LIVE_LATENCY_PRIORITY_DEFAULT))) == '1'


def lower_background() -> None:
    """重い探索の子の起動時に呼ぶ。旗 OFF なら何もしない。失敗しても配信は続け、結果を stderr に残す。"""
    if not enabled():
        return
    status, error = 'applied', None
    try:
        if sys.platform == 'win32':
            from .live_cpu import windows_priority
            windows_priority(WINDOWS_BELOW_NORMAL)
        else:
            os.nice(POSIX_BACKGROUND_NICE)
    except Exception as exc:
        status, error = 'failed', repr(exc)
    record = dict(event=EVENT, pid=os.getpid(), process=mp.current_process().name, status=status, error=error)
    try:
        print(json.dumps(record, ensure_ascii=True), file=sys.stderr, flush=True)
    except (OSError, AttributeError):
        pass  # 出力先がない窓なし起動でも、配信は継続する。
