"""カメラの再接続。映像が読めない状態が続いたら VideoCapture を作り直し、起動時に開けなくても試し続ける。

既定 OFF (設定 live_capture_reconnect)。OFF では DirectShowSource は従来どおり 1 回だけ開く
(make_reconnector は None を返し、入力ループは何も足さない)。

2026-10-24 の精査で確認した従来の欠け:
  - 起動時に開けないと、保留 (入力確認中) のまま永久に動かない。
  - read が失敗し続けても、機器を開き直さない (OBS の再起動やカメラの抜き差しから戻れない)。

作り直す時機の決め方 (定数の根拠):
  - RECONNECT_AFTER_FAIL_SEC = 10 秒。故障注入が想定する映像停止 (live_faults.STALL_DURATION_SEC = 5 秒) は
    FrameContinuity の再確認だけで自力復帰する設計なので、それを超えて 2 倍続いたときだけ作り直す。
    短い失敗で作り直すと、DirectShow の再 open の無映像と色較正・画面確認のやり直しを自分で起こしてしまう。
  - OPEN_RETRY_INTERVAL_SEC = 5 秒。開けない間の再試行間隔。DirectShow の open を頻繁に叩くと機器を握る
    相手 (OBS) の邪魔になりうるため。機器が現れてから最大 5 秒で映像が戻る。
作り直しのたびに間隔を空けるので、機器が壊れたままでも再接続の嵐にはならない。
"""
from __future__ import annotations

from datetime import datetime
import json
import logging
import os
from pathlib import Path
import time
from typing import Any, Callable

from src.production_config import LIVE_CAPTURE_RECONNECT_DEFAULT

FLAG_ENV = 'PUYO_LIVE_CAPTURE_RECONNECT'
OUTPUT_ENV = 'PUYO_LIVE_CAPTURE_RECONNECT_OUTPUT'
LOG_FILE = 'capture_reconnect.jsonl'
# live_faults.STALL_DURATION_SEC と同値 (live_faults は cv2 を import するためここでは複製し、試験で一致を固定)。
STALL_REFERENCE_SEC = 5.0
RECONNECT_AFTER_FAIL_SEC = 2 * STALL_REFERENCE_SEC
OPEN_RETRY_INTERVAL_SEC = STALL_REFERENCE_SEC
EVENT_ENABLED, EVENT_OPEN_FAILED = 'enabled', 'open_failed'
EVENT_ATTEMPT, EVENT_RECOVERED = 'reconnect_attempt', 'recovered'


def enabled() -> bool:
    return os.environ.get(FLAG_ENV, str(int(LIVE_CAPTURE_RECONNECT_DEFAULT))) == '1'


def configure_output(active: bool, output: Path | None) -> None:
    """起動ごとに記録先 (output フォルダ) を子 process へ配送する。OFF・output 無しでは前回の値も残さない。"""
    os.environ.pop(OUTPUT_ENV, None)
    if active and output is not None:
        os.environ[OUTPUT_ENV] = str(Path(output).resolve())


def write_record(row: dict[str, Any]) -> None:
    """再接続の記録を 1 行追記する。保存の失敗で入力ループを止めない。"""
    output = os.environ.get(OUTPUT_ENV)
    if not output:
        return
    try:
        with (Path(output) / LOG_FILE).open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
    except OSError:
        logging.getLogger(__name__).exception('カメラ再接続の記録を保存できませんでした')


def say(text: str) -> None:
    """利用者に見えるコンソールへ状態を出す (認識 process の標準出力は起動した窓へ出る)。"""
    print(text, flush=True)


class CaptureReconnector:
    """read の成否の列から作り直す時機を決め、再接続の回数と時刻を記録する。
    状態を持つ wrapper (観測指標ではない)。時計は呼び出し側 (DirectShowSource の clock) の値を受け取る。"""

    def __init__(self, emit: Callable[[dict[str, Any]], None] = write_record,
                 tell: Callable[[str], None] = say,
                 wall: Callable[[], float] = time.time) -> None:
        self.emit, self.tell, self.wall = emit, tell, wall
        self.attempts = 0          # 作り直した回数
        self.recovered = 0         # 作り直した後に映像が戻った回数
        self.failing_since: float | None = None
        self.last_attempt: float | None = None

    def due(self, read_ok: bool, opened: bool, now: float) -> bool:
        """今、作り直すべきか。read_ok=True なら失敗の記録を閉じて False。"""
        if read_ok:
            self._close_episode(now)
            return False
        if self.failing_since is None:
            self.failing_since = now
            if not opened:
                self._record(EVENT_OPEN_FAILED, now,
                             f'[カメラ] 映像入力を開けません。{OPEN_RETRY_INTERVAL_SEC:g} 秒おきに再試行します')
        anchor = self.failing_since if self.last_attempt is None else self.last_attempt
        wait = RECONNECT_AFTER_FAIL_SEC if opened else OPEN_RETRY_INTERVAL_SEC
        return now - anchor >= wait

    def record_attempt(self, opened: bool, now: float) -> None:
        """作り直しを実行した直後に呼ぶ (opened = 作り直した機器が開けたか)。"""
        self.attempts += 1
        self.last_attempt = now
        self._record(EVENT_ATTEMPT, now, f'[カメラ] 映像入力を作り直しました ({self.attempts} 回目、'
                     f'{"開けました" if opened else "まだ開けません"})', opened=opened)

    def _close_episode(self, now: float) -> None:
        since, self.failing_since = self.failing_since, None
        attempted, self.last_attempt = self.last_attempt is not None, None
        if since is None or not attempted:
            return    # 短い失敗の自力復帰は記録しない (作り直していない)
        self.recovered += 1
        self._record(EVENT_RECOVERED, now, f'[カメラ] 映像が戻りました (再接続 {self.attempts} 回、'
                     f'{now - since:.1f} 秒ぶり)', failing_sec=now - since)

    def _record(self, event: str, now: float, text: str, **extra: Any) -> None:
        wall = self.wall()
        self.tell(text)
        self.emit(dict(event=event, attempts=self.attempts, recovered=self.recovered,
                       at_mono=now, at_wall=wall, at_local=datetime.fromtimestamp(wall).isoformat(timespec='seconds'),
                       **extra))


def make_reconnector() -> CaptureReconnector | None:
    """フラグ OFF では None (入力ループは従来どおり)。ON では有効になったことを記録に残す
    (記録ファイルが無い = 再接続が不要だった、と、フラグが効いていなかった、を区別するため)。"""
    if not enabled():
        return None
    reconnector = CaptureReconnector()
    wall = reconnector.wall()
    reconnector.emit(dict(event=EVENT_ENABLED, reconnect_after_fail_sec=RECONNECT_AFTER_FAIL_SEC,
                          open_retry_interval_sec=OPEN_RETRY_INTERVAL_SEC, at_wall=wall,
                          at_local=datetime.fromtimestamp(wall).isoformat(timespec='seconds')))
    return reconnector
