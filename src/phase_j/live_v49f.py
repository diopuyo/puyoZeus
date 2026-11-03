"""未採用のOBS試験欄。既存B3の勝率・履歴へ値を戻さない。"""
from __future__ import annotations

from collections import deque
import json
import os
from pathlib import Path
from queue import Queue, Empty, Full
from threading import Thread
from time import perf_counter
from typing import Any

from .live_v49f_async import AsyncV49F, QUEUE_CAPACITY, CLOSE_TIMEOUT, AUDIT_CLOSE_TIMEOUT
from .live_x_direct import WakeSignal, XFirstTracker, direct_publish_enabled

ASSETS = Path('models/v49f')
PUMP_JOIN_SEC = 5.


class LiveV49F:
    def __init__(self, model: Path, table: Path, output: Path) -> None:
        self.output = output
        self.submitted, self.received = 0, 0
        assets = (model, table, ASSETS/'policy.json', ASSETS/'pre_model.json')
        self.worker = AsyncV49F(assets, output)
        self.inputs: Queue = Queue(QUEUE_CAPACITY)
        self.outputs: Queue = Queue(QUEUE_CAPACITY)
        # 段1 直接公開 (既定OFF)。OFF は従来の1 thread pump のまま。
        self.direct = direct_publish_enabled()
        if self.direct:
            self.start_direct()
        else:
            self.thread = Thread(target=self.pump, daemon=True)
            self.thread.start()
        self.latest: dict | None = None
        self.generation: int | None = None
        self.game: int | None = None
        self.reason = '入力待ち'
        self.failure: str | None = None
        self.c = json.loads((ASSETS/'model.json').read_text(encoding='utf-8'))['c']
        self.prefire = self.create_prefire(output)

    @staticmethod
    def create_prefire(output: Path) -> Any:
        """段3 の記録・別欄表示。既定OFFでは作らない (従来の update・decorate と同一)。"""
        from .live_v49f_prefire import flags
        shadow, display = flags()
        if not shadow:
            return None
        from .live_v49f_prefire import PrefireView
        return PrefireView(output, display)

    def pump(self) -> None:
        """Pipeの待機は専用スレッドへ隔離し、B3所有スレッドを止めない。"""
        try:
            while (packet := self.inputs.get()) is not None:
                self.worker.submit(packet)
                result = self.worker.connection.recv()
                self.received += 1
                self.outputs.put(dict(result, generation=packet['generation']))
                if result.get('fatal'):
                    return
        except Exception as exc:
            self.outputs.put(dict(fatal=f'{type(exc).__name__}: {exc}'))

    def start_direct(self) -> None:
        """段1: 送信と受信を別 thread にし、受信側は X の初出で評価所有者を起こす。"""
        self.wake = WakeSignal()
        self.tracker = XFirstTracker()
        self.pending: deque = deque()
        self.closing = False
        self.x_completed_at: float | None = None
        self.thread = Thread(target=self.send_loop, name='v49f-send', daemon=True)
        self.receiver = Thread(target=self.receive_loop, name='v49f-receive', daemon=True)
        self.thread.start()
        self.receiver.start()

    def send_loop(self) -> None:
        """前の計算の完了を待たずに送る。作業側は1件ずつ FIFO で処理する。書込みはこの thread だけ。"""
        try:
            while (packet := self.inputs.get()) is not None:
                self.pending.append(packet['generation'])
                self.worker.submit(packet)
            self.worker.connection.send(None)
        except Exception as exc:
            self.deliver(dict(fatal=f'{type(exc).__name__}: {exc}'))

    def receive_loop(self) -> None:
        """結果を受けた順に渡す。X の初出 (live_x_direct.XFirstTracker) なら渡した後で起こす。"""
        try:
            while not (result := self.worker.connection.recv()).get('fatal'):
                self.received += 1
                generation = self.pending.popleft()
                fresh = self.tracker.observe(result, (generation, result['game']))
                self.deliver(dict(result, generation=generation, x_fresh=fresh), fresh)
            self.deliver(result)
        except (EOFError, OSError) as exc:
            if not self.closing:
                self.deliver(dict(fatal=f'{type(exc).__name__}: {exc}'))
        except Exception as exc:
            self.deliver(dict(fatal=f'{type(exc).__name__}: {exc}'))

    def deliver(self, result: dict, wake: bool = True) -> None:
        self.outputs.put(result)
        if wake:
            self.wake.notify()

    def update(self, frame: dict, generation: int) -> None:
        from src.ledger_x_options import boundary_game
        game = boundary_game(frame)
        if (generation, game) != (self.generation, self.game):
            self.generation, self.game = generation, game
            self.latest = None
            if getattr(self, 'direct', False):
                self.x_completed_at = None
        self.receive()
        packet = frame.get('_v49f')
        prefire = getattr(self, 'prefire', None)
        if packet is not None and prefire is not None:
            packet = prefire.take(packet, game)
        if packet is None or self.failure:
            self.reason = self.failure or '撃ち合い入力待ち'
            return
        self.reason = packet.get('capture_reason') or '計算完了待ち'
        try:
            self.inputs.put_nowait(dict(packet, id=frame['frame'], generation=generation))
            self.submitted += 1
        except Full:
            self.failure = '評価待ちが上限に到達・再起動が必要'

    def receive(self) -> None:
        while True:
            try:
                result = self.outputs.get_nowait()
            except Empty:
                return
            if result.get('fatal'):
                self.failure = result['fatal']
                if os.environ.get('PUYO_OBS_FAST_AUDIT') == '1':
                    (self.output/'v49f_failure.json').write_text(
                        json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
                continue
            if (result['generation'], result['game']) != (self.generation, self.game):
                continue
            if result['value'] is not None:
                self.latest = result
                self.reason = '更新済み'
                if result.get('x_fresh'):
                    # 段1: 判定IDへ足す X 完了時刻。表示値と同じ所有者 thread で更新する。
                    self.x_completed_at = result['completed_at']

    def decorate(self, row: dict, baseline: float | None = None, separate_baseline: bool = False) -> None:
        from src.prefire_v49d import display
        self.receive()
        b3 = baseline if separate_baseline else row.get('raw_probability')
        probability, dead, stamp = None, [], None
        if self.latest is not None and b3 is not None:
            value = self.latest['value']
            stamp = value['input_sec']
            if not value.get('board_only'):
                probability = display(b3, b3, value, self.c, value.get('pre_delta_full', 0), enabled=True)
            dead = value['death_sides'] if value['active'] else []
        if not row.get('match_active', True):
            probability, dead, stamp = None, [], None
        age = max(0., row['t_sec']-stamp) if stamp is not None else None
        reason = self.failure or self.reason
        if age is not None and age > 0 and reason == '更新済み':
            reason = '入力変化待ち'
        row['v49f'] = dict(label='撃ち合い勝率 v4.9f', p1=probability,
            p2=None if probability is None else 1-probability, age_sec=age,
            reason=reason, death_sides=dead, input_sec=stamp,
            completed_at=self.latest['completed_at'] if self.latest else None)
        if self.latest and self.latest['value'].get('board_only') and row.get('match_active', True):
            row['v49f']['board_firepower'] = self.latest['value']['board_firepower']
            row['v49f']['reason'] = '正味未確定（盤面火力のみ）'
        if os.environ.get('PUYO_LIVE_WINRATE_SWITCH') == '1':
            row['_v49f_active'] = bool(self.latest and self.latest['value']['active']
                                       and row.get('match_active', True))
        if getattr(self, 'prefire', None) is not None:
            self.prefire.decorate(row)

    def close(self) -> None:
        if getattr(self, 'direct', False):
            self.close_direct()
            return
        try:
            self.inputs.put_nowait(None)
        except Full:
            pass
        self.thread.join(5.)
        self.worker.close()
        self.close_prefire()
        if os.environ.get('PUYO_OBS_FAST_AUDIT') == '1':
            (self.output/'v49f_display_status.json').write_text(json.dumps(dict(
                submitted=self.submitted, received=self.received, failure=self.failure,
                pump_alive=self.thread.is_alive()), indent=2), encoding='utf-8')

    def close_prefire(self) -> None:
        """段3 の記録を閉じて summary を書く。通常 close・直接公開 close の両方から呼ぶ。"""
        if getattr(self, 'prefire', None) is not None:
            self.prefire.close()

    def close_direct(self) -> None:
        """送信 thread が終了要求まで送り切り、受信 thread が残りを読み切ってから接続を閉じる。"""
        audit = os.environ.get('PUYO_OBS_FAST_AUDIT') == '1'
        timeout = AUDIT_CLOSE_TIMEOUT if audit else CLOSE_TIMEOUT
        self.closing = True
        try:
            self.inputs.put_nowait(None)
        except Full:
            pass
        self.thread.join(max(PUMP_JOIN_SEC, timeout))
        self.receiver.join(timeout)
        self.worker.close()
        self.receive()   # 送受信の終了後に最後の結果・異常を吸い出し、それから終了票を数える。
        self.wake.close()
        self.close_prefire()   # 統合: 段1 ON でも段3 の記録・summary を閉じる (閉じ忘れ防止)
        if audit:
            (self.output/'v49f_display_status.json').write_text(json.dumps(dict(
                submitted=self.submitted, received=self.received, failure=self.failure,
                pump_alive=self.thread.is_alive() or self.receiver.is_alive(),
                direct_publish=True, **self.x_wake_counts()), indent=2), encoding='utf-8')

    def x_wake_counts(self) -> dict[str, int]:
        """段1 ON だけの起床の記録 (OFF の記録には出さない)。「増える公開は発火 1 回あたり」の分母に使う。

        x_fresh: X の初出で起床を要求した回数 / x_fresh_fires: 初めて X が載った (側, 発火) の数 /
        x_wakes: 起床 pipe へ実際に書いた回数 (未読と合流した要求は含まない。失敗通知の起床は含む)。
        """
        return dict(x_fresh=self.tracker.fresh_results, x_fresh_fires=self.tracker.fresh_fires,
                    x_wakes=self.wake.sent)
