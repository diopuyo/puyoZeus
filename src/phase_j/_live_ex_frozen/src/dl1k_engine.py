"""B3 推論を呼ばない v4.15 系エンジンと 1 要求の計算 (dl1k 研究専用・既定 OFF)。

状態の進め方は原器 (PostB3.observe: 境界→現在盤→連鎖通知→撃ち合いの閉じ) のまま。
B3 評価 (`_evaluate` / `calculate_record` / `continuous`) は呼ばない。予測器は呼ばれたら止まる。
1 要求 = 予告の射影 → v4.9g の確定死亡証明 (LandingV49G.apply) → 文脈 → 理論 minimax。
"""
from __future__ import annotations

from dataclasses import dataclass
import os
from time import perf_counter
from typing import Any, Callable

from src.phase_j._live_ex_frozen.src.dl1k_theory import BUDGET_MS, solve
from src.phase_j._live_ex_frozen.src.prefire_v415 import PostV415

MS = 1000.
SIDES = (0, 1)
CERTIFIED_TERMINAL = 'certified_terminal'
CONSERVATION_MESSAGE = 'おじゃま保存則違反'
FUTURE_MESSAGE = '未来の'
B3_CALLED_MESSAGE = 'B3 推論を呼ばない'
RETR_ENV = 'PUYO_EX_RETR'


class ForbiddenPredictor:
    """B3 推論の呼出しを検出したら止める (抽出器へ学習済み判断を混ぜない保証)。"""
    evaluators: tuple = ()

    def predict(self, row: Any) -> float:
        raise RuntimeError('dl1k の抽出器は B3 推論を呼ばない')


@dataclass(frozen=True)
class Probe:
    """撃ち合い中か (どちらかの攻めの盤予測が有効) と、同じ撃ち合いの鍵。"""
    fight: bool
    key: tuple


class ExEngine(PostV415):
    """観測状態は外部ラッパー (この engine) が持つ。判定関数には時点固定の値だけを渡す。"""

    def __init__(self, policy: dict, native: Any, seed_lookup: Any = None, epoch: int = 0,
                 ledger_hook: Callable[[Any], None] | None = None) -> None:
        self.epoch, self.ledger_hook = epoch, ledger_hook
        super().__init__(ForbiddenPredictor(), None, policy, native, True, None, BUDGET_MS, seed_lookup)

    def _boundary(self, stamp: float, game: int) -> None:
        super()._boundary(stamp, game)
        if self.ledger_hook is not None:
            self.ledger_hook(self)

    def _evaluate(self, stamp: float, began: float, extra_ms: float = 0.) -> None:
        return None

    def calculate_record(self, record: dict, raw: list[dict]) -> None:
        raise RuntimeError('元器の B3 経路 (calculate_record) は dl1k では使わない')

    def step(self, t: float, game: int, raw: list[dict]) -> None:
        self.observe(t, game, raw)
        if retr_enabled():   # retR の I 用の撃った盤 (scene_68 の直し)。EX の状態には触らない
            if not hasattr(self, 'retr_fire'):
                from src.phase_j._live_ex_frozen.src.dl1k_retr import fire_tracker
                self.retr_fire = fire_tracker(self.native)
            self.retr_fire.observe(t, game, raw)

    def probe(self) -> Probe:
        """原器が評価する条件 (撃ち合い中かつ両側の盤がある) だけを要求点にする。"""
        ids = tuple(c['signature'][0] if c else None for c in self.chains)
        boards = all(s.board is not None for s in self.state.sides)
        fight = boards and any(self._resolving(i) for i in SIDES)
        return Probe(fight, (self.epoch, self.state.game_idx, ids))

    def close(self) -> None:
        self.pool.shutdown()


def proof_of(engine: ExEngine, value: dict) -> dict:
    """v4.9g の死亡証明だけを計算する。失敗は欠測 (death_sides を空にして続行)。"""
    from src.phase_j._live_ex_frozen.src.exchange_native_chain import native_chains
    from src.phase_j._live_ex_frozen.src.prefire_landing_v49d import StageClock
    try:
        engine.tracker.refresh_verdicts(engine.chains, value['reconciliation'], engine.state.sl.core)
        with native_chains(engine.native):
            engine.landing.apply(engine, engine.tracker, value, StageClock())
    except Exception as exc:  # noqa: BLE001 - 件数を受領票へ残し、0 件合格と読み替えない
        value.update(landing_proofs=[], death_sides=[])
        return dict(ok=False, death_sides=[], error=repr(exc))
    return dict(ok=True, death_sides=list(value['death_sides']),
                reasons=[a.get('reason') for a in value['landing_proofs']])


def search_of(engine: ExEngine, value: dict) -> tuple[dict | None, str | None, list[int] | None]:
    """文脈を作り、確定死亡が片側だけなら探索しない (v4.15 と同じ)。"""
    from src.phase_j._live_ex_frozen.src.exchange_native_chain import native_chains
    from src.phase_j._live_ex_frozen.src.prefire_context_v410 import context
    ctx, reason = context(engine, value)
    if len(value.get('death_sides', [])) == 1:
        ctx, reason = None, CERTIFIED_TERMINAL
    if ctx is None:
        return None, reason, None
    knowledge = engine.knowledge.snapshot(ctx)
    evidence = engine.placements.snapshot(ctx, engine.state)
    with native_chains(engine.native):
        output = solve(engine.native, ctx, knowledge, evidence)
    receiver = engine.chains[ctx.receiver]
    observed = list(receiver['best']['scores']) if ctx.triggers[ctx.receiver] is not None else None
    output['seed_known'] = [s.get('seed') is not None for s in knowledge['sides']]
    return output, None, observed


def classify_error(exc: BaseException) -> str:
    """保存則・因果の違反と、その他の計算失敗を分ける。"""
    text = str(exc)
    if B3_CALLED_MESSAGE in text:
        return 'b3_called'
    if CONSERVATION_MESSAGE in text:
        return 'conservation_violation'
    if FUTURE_MESSAGE in text:
        return 'causal_violation'
    return 'error'


def retr_enabled() -> bool:
    """retR の列 (src/dl1k_retr.py) を足すか。既定 OFF (EX の 70 列・指紋は変えない)。"""
    return os.environ.get(RETR_ENV) == '1'


def add_retr(engine: ExEngine, value: dict, result: dict) -> None:
    """射影の直後 (死亡証明が value を書き換える前) に retR を計算する。失敗は retR だけ欠測にして EX は続ける。"""
    from src.phase_j._live_ex_frozen.src.dl1k_retr import blank, compute_retr
    started = perf_counter()
    try:
        result['retr'], result['retr_why'] = compute_retr(engine, value)
    except Exception as exc:  # noqa: BLE001 - retR の失敗で EX の計算を止めない
        result['retr'], result['retr_why'] = blank(), [f'error:{exc!r}']*2
    result['retr_ms'] = (perf_counter()-started)*MS


def compute(engine: ExEngine) -> dict:
    """現在の engine の状態で 1 要求を計算する。B3 は呼ばない。"""
    started = perf_counter()
    result: dict[str, Any] = dict(output=None, reason=None, observed_scores=None, proof=None, failure=None)
    try:
        value = engine.projection()
        if retr_enabled():
            add_retr(engine, value, result)
        result['proof'] = proof_of(engine, value)
        result['output'], result['reason'], result['observed_scores'] = search_of(engine, value)
    except Exception as exc:  # noqa: BLE001 - 1 要求の失敗は欠測 (不成立) として種類別に数え、動画全体は止めない
        result.update(failure=classify_error(exc), error=repr(exc))
    result['compute_ms'] = (perf_counter()-started)*MS
    return result
