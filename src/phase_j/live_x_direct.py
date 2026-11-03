"""段1 直接公開 (live_x_direct_publish、既定OFF) の共有部品。

撃った連鎖の火力 X が v4.9f の結果に初めて載ったら、次の認識通知や B3 完了を待たずに
評価所有者を起こし、既存の B3 完了と同じ経路で即公開する (設計 live_x_timing_20261022 §2)。
OFF では本モジュールの部品を生成せず、既存の pump・公開・判定 ID の経路をそのまま通る。

「X の初出」= 値つきの結果で、側ごとに新しい発火 trigger の X (predicted_scores) が初めて載ったもの。
通知候補・画像予測 (origins) のどちらの経路で届いても同じに扱う。設計 §2 の字義「origins つきの結果で起床」は
採らない: 21c では最初の当該 X 57件中 53件が origins なしで届き (起床しても M3 は 0.704→0.691 秒)、
入力が健全な A2 (20分) では origins つきの値が 1,237件あり公開を約1.5倍に増やすため。
"""
from __future__ import annotations

import multiprocessing as mp
import os
from threading import Event
from typing import Any, Mapping

ENV_NAME = 'PUYO_LIVE_X_DIRECT_PUBLISH'
SIDES = (0, 1)
WAKE_BYTE = b'x'


def direct_publish_enabled(environ: Mapping[str, str] | None = None) -> bool:
    """spawn 子にも環境変数で届く。未設定なら production_config の既定 (False)。"""
    from src.production_config import LIVE_X_DIRECT_PUBLISH_DEFAULT
    value = (os.environ if environ is None else environ).get(ENV_NAME)
    return LIVE_X_DIRECT_PUBLISH_DEFAULT if value is None else value == '1'


class WakeSignal:
    """受信 thread → 評価所有者の片方向 pipe。multiprocessing.connection.wait で待てる。

    未読の起床は常に高々1件 (pending で合流)。評価所有者が読まない構成でも pipe が詰まって
    受信 thread が止まることはない。結果は起床より先に出力 queue へ入れるので、合流しても
    次の読取りで全件が取り込まれる。
    """

    def __init__(self) -> None:
        self.reader, self.writer = mp.Pipe(duplex=False)
        self.pending = Event()
        self.sent = 0          # pipe へ実際に書いた起床の累計 (未読と合流した通知は数えない・失敗通知は数える)

    def notify(self) -> None:
        if not self.pending.is_set():
            self.pending.set()
            self.writer.send_bytes(WAKE_BYTE)
            self.sent += 1

    def drain(self) -> int:
        """未読の起床を読み捨て、件数 (0 か 1) を返す。先に pending を下ろし、以後の通知を再び許す。"""
        self.pending.clear()
        count = 0
        while self.reader.poll():
            self.reader.recv_bytes()
            count += 1
        return count

    def close(self) -> None:
        for connection in (self.reader, self.writer):
            connection.close()


class XFirstTracker:
    """X の初出を見分ける外部 wrapper。判定の状態は直近の X つき base と、側ごとの通知済み発火だけ。

    fresh_results / fresh_fires は記録用の累計 (入力世代・試合が変わっても戻さない)。判定には使わない。
    """

    def __init__(self) -> None:
        self.identity: tuple | None = None
        self.base: dict | None = None
        self.signaled: list[float | None] = [None for _ in SIDES]
        self.fresh_results = 0     # X の初出と判定した結果の数 (= 起床を要求した回数 x_fresh)
        self.fresh_fires = 0       # 初めて X が載った (側, 発火) の数。1 つの結果で両側なら 2

    def observe(self, result: Mapping[str, Any], identity: tuple) -> bool:
        """入力世代・試合が変われば忘れる。値つきで X が初めて載った結果なら True。"""
        if identity != self.identity:
            self.identity, self.base = identity, None
            self.signaled = [None for _ in SIDES]
        base = result.get('base')
        if base and 'predicted_scores' in base:
            self.base = base
        value = result.get('value')
        if value and value.get('board_only'):
            self.base = dict(chain_ids=[side['trigger'] if side else None
                                        for side in value['board_firepower']])
        if value is None or self.base is None:
            return False
        fresh = False
        for side in SIDES:
            trigger = self.base['chain_ids'][side]
            if trigger is not None and value['predicted_scores'][side] and trigger != self.signaled[side]:
                self.signaled[side] = trigger
                self.fresh_fires += 1
                fresh = True
        self.fresh_results += fresh
        return fresh
