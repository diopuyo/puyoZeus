"""発火・絶対終了・OCR確定を束ねる状態付きラッパー。評価器は純粋なまま使う。"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path

import numpy as np

from src.chain_id_resolver import ChainIdResolver, ChainObservation, ObservationKind
from src.exchange_event_evaluator import (
    ExchangeEndInput, ExchangeModels, FiringInput, StaticInput, evaluate_exchange_event,
)

SIDE_LABELS = ("1P", "2P")
S3_SCORE_STABLE_FRAMES = 10


@dataclass
class ExchangeChainRecord:
    """時刻は動画絶対秒。得点はOCR差分のみで、推定生成量を含めない。"""

    side: str
    chain_id: int
    trigger_sec: float
    observed_sec: float
    end_signal_sec: float | None = None
    score_finalize_sec: float | None = None
    score_delta: float | None = None
    end_reason: str | None = None
    end_signals: list[dict] = field(default_factory=list)
    score_ready_sec: float | None = None
    score_ready_reason: str | None = None
    score_before: float | None = None
    formula_total: float | None = None
    display_score: float | None = None
    stable_frames: int = 0


@dataclass
class ExchangeRecord:
    """E3が得点確定と着地の前後関係をそのまま比較できる単位。"""

    exchange_id: int
    game_idx: int
    trigger_sec: float
    chains: list[ExchangeChainRecord] = field(default_factory=list)
    landings: list[dict] = field(default_factory=list)
    values: list[dict] = field(default_factory=list)
    closed_sec: float | None = None
    close_reason: str | None = None


class ExchangeEventTracker:
    """同一フレームの両側発火を先に登録し、全参加連鎖の確定を待つ。"""

    def __init__(self, models: ExchangeModels) -> None:
        self.models = models
        self.records: list[ExchangeRecord] = []
        self.current: ExchangeRecord | None = None
        self.resolver = ChainIdResolver()
        self.firing: FiringInput | None = None
        self.source = "waiting_confirmed"
        self.probability: float | None = None
        self._seen: set[tuple[int, str, float]] = set()
        self._game_idx: int | None = None
        self._score_elapsed = 0.0
        self._s3_sec: float | None = None
        self._contexts: dict[int, tuple[FiringInput, float]] = {}

    def boundary(self, game_idx: int, t_sec: float) -> None:
        """試合をまたぐ未完イベントを確定扱いにせず切り離す。"""
        if game_idx == self._game_idx:
            return
        if self.current is not None:
            self.current.closed_sec = t_sec
            self.current.close_reason = "match_boundary"
        self.current, self.firing = None, None
        self.resolver = ChainIdResolver()
        self._seen.clear()
        self.source, self.probability = "waiting_confirmed", None
        self._game_idx, self._s3_sec = game_idx, None

    def fire(self, *, t_sec: float, triggers: tuple[float | None, float | None],
             static: StaticInput, prefire_sides: np.ndarray,
             score_elapsed_sec: float,
             observations: tuple[ChainObservation, ...] = ()) -> None:
        """既存trigger_secを同定子にし、成長中の再通知は同じ連鎖へ束ねる。"""
        fresh = [(side, trigger) for side, trigger in zip(SIDE_LABELS, triggers)
                 if trigger is not None and (self._game_idx, side, trigger) not in self._seen]
        if not fresh and not observations:
            return
        for side, trigger in fresh:
            self._seen.add((self._game_idx, side, trigger))
        for observation in observations:
            self.resolver.push(observation)
        if self.current is None and observations and self._resume_existing(observations):
            if self.current is None:
                return
        if observations and self.current is None and not self.resolver.active():
            return  # 得点確定後のbaseline残響から空の撃ち合いを作らない。
        if self.current is None:
            first = min(trigger for trigger in triggers if trigger is not None)
            self.current = ExchangeRecord(len(self.records) + 1, self._game_idx, first)
            self.records.append(self.current)
            firing = tuple(any(s == side and ts == first for s, ts in fresh) for side in SIDE_LABELS)
            self.firing = FiringInput(static, prefire_sides, firing)
            self._score_elapsed, self._s3_sec = score_elapsed_sec, None
            self._contexts[self.current.exchange_id] = (self.firing, score_elapsed_sec)
            self._evaluate(self.firing, "S1", t_sec)
        if observations:
            for observation in observations:
                idx = SIDE_LABELS.index(observation.side)
                self._record_chain(observation, triggers[idx])
            return
        for side, trigger in fresh:
            active = self.latest_chain(side)
            if active is not None and active.end_signal_sec is None:
                continue
            if active is not None and trigger <= active.end_signal_sec:
                continue
            self.resolver.push(ChainObservation(side, t_sec, ObservationKind.FORMULA_STEP,
                                                chain_count=1))
            chain = next(c for c in self.resolver.active() if c.side == side)
            self.current.chains.append(ExchangeChainRecord(side, chain.chain_id, trigger, t_sec))

    def _resume_existing(self, observations: tuple[ChainObservation, ...]) -> bool:
        """早期S3後のbaseline残響は無視し、実段継続なら元の撃ち合いへ戻す。"""
        sides = {o.side for o in observations}
        active = {c.chain_id: c for c in self.resolver.active() if c.side in sides}
        for record in reversed(self.records):
            if record.game_idx != self._game_idx:
                continue
            known = [c for c in record.chains if c.chain_id in active]
            if not known:
                continue
            if set(active) != {c.chain_id for c in known}:
                return False
            if all(active[c.chain_id].awaiting_finalize for c in known):
                return True
            self.current = record
            self.firing, self._score_elapsed = self._contexts[record.exchange_id]
            record.closed_sec, record.close_reason = None, None
            values = [v for v in record.values if v["source"] == "S3"]
            self._s3_sec = values[-1]["t_sec"] if values else None
            if values:
                self.source, self.probability = "S3", values[-1]["p1"]
            return True
        return False

    def _record_chain(self, observation: ChainObservation, trigger_sec: float) -> None:
        """段別の実観測をresolverへ渡し、段継続・baseline残響の同一性を再用する。"""
        chain = next((c for c in self.resolver.active() if c.side == observation.side), None)
        if chain is None:
            return
        existing = next((c for c in self.current.chains if c.chain_id == chain.chain_id), None)
        if existing is None:
            self.current.chains.append(ExchangeChainRecord(
                observation.side, chain.chain_id, trigger_sec, observation.t_sec))
        elif not chain.awaiting_finalize and existing.end_signal_sec is not None:
            existing.end_signals[-1]["revoked_sec"] = observation.t_sec
            existing.end_signal_sec, existing.end_reason = None, None
            existing.score_ready_sec, existing.score_ready_reason = None, None
            existing.score_delta, existing.stable_frames = None, 0

    def latest_chain(self, side: str) -> ExchangeChainRecord | None:
        """現撃ち合いの該当側の末尾連鎖を返す。"""
        if self.current is None:
            return None
        return next((c for c in reversed(self.current.chains) if c.side == side), None)

    def end(self, side: str, t_sec: float, reason: str) -> None:
        """終了合図だけではS3に進めず、resolverを確定待ちにする。"""
        chain = self.latest_chain(side)
        if chain is None or chain.end_signal_sec is not None:
            return
        chain.end_signal_sec, chain.end_reason = t_sec, reason
        chain.end_signals.append(dict(t_sec=t_sec, reason=reason))
        self.resolver.push(ChainObservation(side, t_sec, ObservationKind.CHAIN_END_SIGNAL))

    def finalize(self, side: str, t_sec: float, score_delta: float) -> None:
        """既存OCR確定通知を同じchain_idへ供給する。ledger closeは参照しない。"""
        if not np.isfinite(score_delta) or score_delta < 0:
            raise ValueError("得点確定通知にはOCRの非負差分が必要")
        pending = [c for r in self.records if r.game_idx == self._game_idx
                   for c in r.chains if c.side == side and c.score_finalize_sec is None]
        if not pending:
            return
        chain = pending[-1]
        chain.score_finalize_sec = t_sec
        if any(c.chain_id == chain.chain_id for c in self.resolver.active()):
            self.resolver.push(ChainObservation(side, t_sec, ObservationKind.SCORE_FINALIZE,
                                                total_score=int(score_delta)))
        if chain.score_ready_sec is None:
            self._score_ready(chain, t_sec, score_delta, "score_finalize")

    def observe_score(self, side: str, t_sec: float, score: float | None,
                      score_before: float | None = None,
                      formula_total: float | None = None) -> None:
        """終了合図以降だけ連続表示を数え、式の一致なら即時採用する。"""
        chain = self.latest_chain(side)
        if chain is None:
            return
        if chain.score_before is None:
            chain.score_before = score_before
        if formula_total is not None and formula_total > 0:
            chain.formula_total = formula_total
        if chain.end_signal_sec is None or chain.score_ready_sec is not None:
            return
        if score is None or not np.isfinite(score) or chain.score_before is None:
            chain.display_score, chain.stable_frames = None, 0
            return
        chain.stable_frames = chain.stable_frames + 1 if score == chain.display_score else 1
        chain.display_score = score
        delta = score - chain.score_before
        if delta < 0:
            chain.stable_frames = 0
        elif chain.formula_total is not None and delta == chain.formula_total:
            self._score_ready(chain, t_sec, chain.formula_total, "formula_match")
        elif chain.stable_frames >= S3_SCORE_STABLE_FRAMES:
            self._score_ready(chain, t_sec, delta, "display_stable")

    @staticmethod
    def _score_ready(chain: ExchangeChainRecord, t_sec: float,
                     delta: float, reason: str) -> None:
        """評価側だけを確定し、会計resolverへ早期確定を送らない。"""
        chain.score_ready_sec, chain.score_ready_reason = t_sec, reason
        chain.score_delta = delta

    def fall_start(self, side: str, t_sec: float) -> None:
        """落下開始を着地完了と混同せず、別の時刻で保持する。"""
        record = self._landing_record()
        if record is not None:
            record.landings.append(dict(side=side, fall_start_sec=t_sec, t_sec=None))

    def landing(self, side: str, t_sec: float) -> None:
        """既存の物理着弾検出を記録する。配送会計の差分で代用しない。"""
        record = self._landing_record()
        if record is not None:
            pending = next((r for r in reversed(record.landings)
                            if r["side"] == side and r["t_sec"] is None), None)
            if pending is not None:
                pending["t_sec"] = t_sec
            else:
                record.landings.append(dict(side=side, fall_start_sec=None, t_sec=t_sec))

    def _landing_record(self) -> ExchangeRecord | None:
        """G_fe復帰後も次の撃ち合いまでの着地を同じ時刻表へ記録する。"""
        if self.current is not None:
            return self.current
        return next((r for r in reversed(self.records) if r.game_idx == self._game_idx), None)

    def finish_frame(self, t_sec: float) -> None:
        """左右の通知を全て処理してからS3を一度だけ更新する。"""
        if self.current is None or not self.current.chains:
            return
        chains = self.current.chains
        if any(c.end_signal_sec is None or c.score_ready_sec is None for c in chains):
            return
        finalized = max(c.score_ready_sec for c in chains)
        if self._s3_sec is not None and finalized <= self._s3_sec:
            return
        totals = np.array([sum(c.score_delta for c in chains if c.side == side)
                           for side in SIDE_LABELS])
        event = ExchangeEndInput(self.firing, np.zeros(2), totals, self._score_elapsed)
        self._evaluate(event, "S3", t_sec)
        self._s3_sec = t_sec

    def ready_for_static(self, t_sec: float, confirmed_times: tuple[float, float]) -> bool:
        """S3を最低1フレーム表示し、着地後の両側新規確定を待つ。"""
        if self.current is None:
            return True
        if self._s3_sec is None or t_sec <= self._s3_sec:
            return False
        if any(c.end_signal_sec is None or c.score_ready_sec is None for c in self.current.chains):
            return False
        totals = [sum(c.score_delta for c in self.current.chains if c.side == side)
                  for side in SIDE_LABELS]
        landings = [r["t_sec"] for r in self.current.landings if r["t_sec"] is not None]
        if len(landings) != len(self.current.landings):
            return False
        if not landings and totals[0] != totals[1]:
            return False
        cutoff = max([self._s3_sec] + landings)
        return min(confirmed_times) > cutoff

    def static(self, event: StaticInput, t_sec: float,
               confirmed_times: tuple[float, float]) -> bool:
        """撃ち合い優先を保ったまま、着地後のG_feへ戻す。"""
        if not self.ready_for_static(t_sec, confirmed_times):
            return False
        self._evaluate(event, "G_fe", t_sec)
        self.close_confirmed(t_sec, confirmed_times)
        return True

    def close_confirmed(self, t_sec: float, confirmed_times: tuple[float, float]) -> bool:
        """モデルの供給状態と独立に、着地後の確定盤面で物理区間を閉じる。"""
        if not self.ready_for_static(t_sec, confirmed_times):
            return False
        if self.current is not None:
            self.current.closed_sec, self.current.close_reason = t_sec, "confirmed_after_landing"
        self.current, self.firing, self._s3_sec = None, None, None
        return True

    def _evaluate(self, event: StaticInput | FiringInput | ExchangeEndInput,
                  source: str, t_sec: float) -> None:
        """イベントの勝率を保持し、表示値と計装値を一致させる。"""
        self.probability = evaluate_exchange_event(event, self.models)
        self.source = source
        if self.current is not None:
            self.current.values.append(dict(source=source, t_sec=t_sec, p1=self.probability))

    def save(self, path: Path) -> None:
        """未完の撃ち合いも欠落させずJSONLへ保存する。"""
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as stream:
            for record in self.records:
                stream.write(json.dumps(asdict(record), ensure_ascii=False, allow_nan=False) + "\n")
