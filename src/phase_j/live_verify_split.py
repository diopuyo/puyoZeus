"""画面の確認の失敗を (a) 入力の異常 / (b) 一時的な見え方の不足 に分け、(b) は期限つきで状態を捨てない。

従来は確認が連続で外れて保留に入ると、必ず盤面・台帳・試合の状態を捨てていた (epoch+1・pipe.reset)。
24b の保存記録では、試合中のこの破棄が約 0.9〜1.0 回/試合起き、直後に必ず内部の試合開始が増えていた。
本モジュールは「どの失敗なら捨てずに待ってよいか」の判定と、待つ期限だけを持つ (状態は持ち出さない)。
フラグは既定 OFF。OFF では他のモジュールから一切呼ばれず、従来の出力と 1 ビットも変わらない。
設計: <work>/codex_jobs/verify_hold_split_20261024/DESIGN.md
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import os
import time
from typing import Any, Callable

from src.production_config import LIVE_VERIFY_SOFT_HOLD_DEFAULT

FLAG_ENV = 'PUYO_LIVE_VERIFY_SOFT_HOLD'

# 保留の最長秒数。保存記録 (24b NEW/NEW24A) の再生で、期限ごとに「全て (b) のまま復帰できる reset」を数えた。
# 5 秒 68/61、8 秒 69/64、10 秒 69/64、15 秒 69/64 (/91, /86)。8 秒で飽和するので、それより長くしない。
# 長いほど復帰時の欠落が大きくなる (コマ落ち耐性の想定は数刻み〜数十刻み) ため、飽和点を上限にする。
SOFT_HOLD_MAX_SEC = 8.0
# (b) とみなす外枠の失敗辺の最大数。辺は独立なので、1 辺だけなら画面は同一で一部が隠れているだけ。
# 2 辺以上が同時に見えないのは画面全体が変わった (暗転・別画面・終了演出) 場合で、観測では被覆 0.20 未満に集中した。
MAX_TRANSIENT_FAILED_EDGES = 1

STAGE_FEEDBACK = 'feedback'
STAGE_SCORE = 'score'
STAGE_GRID_POSITION = 'grid_position'
STAGE_BORDER = 'border'
# (b) になりうる段。得点の読み落としは試合中の破棄の 2〜4 件にしか効かず、待機画面でも同じ症状が出る (DESIGN §2.3) ため
# 初版は入れない。足すときはここに STAGE_SCORE を加えるだけ。
TRANSIENT_STAGES: tuple[str, ...] = (STAGE_BORDER,)

KIND_INPUT = 'input'          # (a) 入力そのものの異常。従来どおり保留 → reset。
KIND_TRANSIENT = 'transient'  # (b) 一時的な見え方の不足。期限内は状態を捨てない。
KIND_KEY = 'failure_kind'     # 入力診断 (input_diagnostics) に載せるキー
DETAIL_KEY = 'failure'


def enabled() -> bool:
    """既定 OFF。環境変数で ON (子 process は親の環境を継承する)。"""
    return os.environ.get(FLAG_ENV, str(int(LIVE_VERIFY_SOFT_HOLD_DEFAULT))) == '1'


@dataclass(frozen=True)
class VerifyFailure:
    """確認が外れた段と、外枠なら失敗した辺の数。判定専用の最小記録 (ログ OFF でも残す)。"""
    stage: str
    side: str = ''
    failed_edges: int | None = None


def classify_failure(failure: VerifyFailure | None, transport_ready: bool = True) -> str:
    """失敗を (a)/(b) に分ける。情報が足りない場合は (a) (従来どおり) に倒す。"""
    if not transport_ready or not isinstance(failure, VerifyFailure):
        return KIND_INPUT
    if failure.stage not in TRANSIENT_STAGES:
        return KIND_INPUT
    if failure.stage == STAGE_BORDER and (
            failure.failed_edges is None or failure.failed_edges > MAX_TRANSIENT_FAILED_EDGES):
        return KIND_INPUT
    return KIND_TRANSIENT


def diagnostics(failure: VerifyFailure | None, transport_ready: bool) -> dict[str, Any]:
    """入力診断へ足す欄。ON のときだけ呼ぶ (OFF の診断は従来と同じキー集合のまま)。"""
    return {KIND_KEY: classify_failure(failure, transport_ready),
            DETAIL_KEY: asdict(failure) if isinstance(failure, VerifyFailure) else None}


class SoftHoldTimer:
    """保留の起点を覚え、期限内かだけを答える。盤面・台帳などの状態には触れない。"""

    def __init__(self, limit_sec: float = SOFT_HOLD_MAX_SEC,
                 clock: Callable[[], float] = time.perf_counter) -> None:
        self.limit_sec, self.clock = limit_sec, clock
        self.since: float | None = None

    def within_limit(self) -> bool:
        """初回の呼び出しで起点を記録する。起点から limit_sec 以内なら真。"""
        now = self.clock()
        if self.since is None:
            self.since = now
        return now - self.since <= self.limit_sec

    def clear(self) -> None:
        """復帰・従来の reset・入力切替で起点を消す。次の保留は新しい起点から数える。"""
        self.since = None
