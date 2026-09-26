"""発火・絶対終了・OCR確定を束ねる状態付きラッパー。評価器は純粋なまま使う。"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path

import numpy as np

from src.chain_id_resolver import ChainIdResolver, ChainObservation, ObservationKind
from src.ojama_accounting import CHAIN_TOTAL_MIN_SCORE
from src.exchange_event_evaluator import (
    ExchangeEndInput, ExchangeModels, FiringInput, StaticInput, evaluate_exchange_event,
)

SIDE_LABELS = ("1P", "2P")
S3_SCORE_STABLE_FRAMES = 10
S3_END_QUIET_FRAMES = 10
OBSERVATION_FPS = 30
S3_END_QUIET_SEC = S3_END_QUIET_FRAMES / OBSERVATION_FPS
EXCHANGE_IDLE_TIMEOUT_SEC = 3.0
TIME_EPSILON_SEC = 1e-9


def valid_nonnegative(value: object) -> bool:
    """欠測・非数・無限値を、評価器へ渡す前に弾く。"""
    return isinstance(value, (int, float, np.number)) and bool(np.isfinite(value) and value >= 0)


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
    last_activity_sec: float | None = None
    drop_bonus_score: float = 0.0
    post_end_drop_sec: float | None = None


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
        self.diagnostics: list[dict] = []
        self._diagnostic_keys: set[tuple] = set()
        self._static_probability: float | None = None
        self._last_activity_sec = 0.0
        self._chain_aliases: dict[int, int] = {}

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
        self._static_probability = None
        self._last_activity_sec = t_sec
        self._chain_aliases.clear()

    def missing_input(self, reason: str, t_sec: float, stage: str,
                      key: object = None) -> None:
        """同じ観測の再通知は一件とし、利用可能な下位評価へ戻す。"""
        identity = (self._game_idx, reason, stage, repr(t_sec if key is None else key))
        if identity not in self._diagnostic_keys:
            self._diagnostic_keys.add(identity)
            self.diagnostics.append(dict(game_idx=self._game_idx, t_sec=t_sec,
                                         reason=reason, requested_stage=stage))
        if stage == "S3" and self.current is not None:
            values = [v for v in self.current.values if v["source"] == "S1"]
            if values:
                self.source, self.probability = "S1", values[-1]["p1"]
                return
        self.probability = self._static_probability
        self.source = "G_fe" if self.probability is not None else "waiting_confirmed"

    def fire(self, *, t_sec: float, triggers: tuple[float | None, float | None],
             static: StaticInput, prefire_sides: np.ndarray,
             score_elapsed_sec: float,
             observations: tuple[ChainObservation, ...] = ()) -> None:
        """既存trigger_secを同定子にし、成長中の再通知は同じ連鎖へ束ねる。"""
        if len(triggers) != len(SIDE_LABELS) or any(
                ts is not None and not valid_nonnegative(ts) for ts in triggers):
            self.missing_input("invalid_trigger", t_sec, "S1", triggers)
            return
        fresh = [(side, trigger) for side, trigger in zip(SIDE_LABELS, triggers)
                 if trigger is not None and (self._game_idx, side, trigger) not in self._seen]
        if not fresh and not observations:
            return
        for side, trigger in fresh:
            self._seen.add((self._game_idx, side, trigger))
        self._last_activity_sec = t_sec
        for observation in observations:
            self._push_chain_observation(observation)
        if self.current is None and observations and self._resume_existing(observations):
            if self.current is None:
                return
        if observations and self.current is None and not self.resolver.active():
            return  # 得点確定後のbaseline残響から空の撃ち合いを作らない。
        if self.current is None:
            if not self._start_exchange(t_sec, triggers, fresh, static,
                                        prefire_sides, score_elapsed_sec):
                return
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

    def _start_exchange(self, t_sec: float, triggers: tuple, fresh: list,
                        static: StaticInput, prefire_sides: np.ndarray,
                        score_elapsed_sec: float) -> bool:
        """入力検証に通るまでrecordやS1を公開しない。"""
        first = min((ts for ts in triggers if ts is not None), default=None)
        firing = tuple(any(s == side and ts == first for s, ts in fresh) for side in SIDE_LABELS)
        if first is None or not any(firing):
            self.missing_input("unknown_firing_side", t_sec, "S1", triggers)
            return False
        try:
            if not valid_nonnegative(score_elapsed_sec):
                raise ValueError("発火経過秒が欠測")
            event = FiringInput(static, prefire_sides, firing)
        except (ValueError, TypeError) as error:
            self.missing_input("firing_input: " + str(error), t_sec, "S1", triggers)
            return False
        if not self._evaluate(event, "S1", t_sec):
            return False
        self.current = ExchangeRecord(len(self.records) + 1, self._game_idx, first)
        self.records.append(self.current)
        self.current.values.append(dict(source="S1", t_sec=t_sec, p1=self.probability))
        self.firing = event
        self._score_elapsed, self._s3_sec = score_elapsed_sec, None
        self._contexts[self.current.exchange_id] = (event, score_elapsed_sec)
        return True

    def _resume_existing(self, observations: tuple[ChainObservation, ...]) -> bool:
        """早期S3後のbaseline残響は無視し、実段継続なら元の撃ち合いへ戻す。"""
        sides = {o.side for o in observations}
        active = {self._chain_aliases.get(c.chain_id, c.chain_id): c
                  for c in self.resolver.active() if c.side in sides}
        for record in reversed(self.records):
            if record.game_idx != self._game_idx:
                continue
            known = [c for c in record.chains if c.chain_id in active]
            if not known:
                continue
            if record.close_reason == "activity_timeout":
                return True  # 安全弁で閉じた区間を古いresolverの残響で再開しない。
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
        identity = self._chain_aliases.get(chain.chain_id, chain.chain_id)
        existing = next((c for c in self.current.chains if c.chain_id == identity), None)
        if existing is None:
            previous = self.latest_chain(observation.side)
            if previous is not None and previous.end_signal_sec is None:
                # 同側の物理終了が未観測なら別の参加連鎖を増やさない。
                # resolverの段数訂正・baseline断片で旧IDを永久待ちにしない。
                self._chain_aliases[chain.chain_id] = previous.chain_id
                return
            self.current.chains.append(ExchangeChainRecord(
                observation.side, chain.chain_id, trigger_sec, observation.t_sec))
            self._s3_sec = None
            values = [v for v in self.current.values if v["source"] == "S1"]
            self.source, self.probability = "S1", values[-1]["p1"]
        elif not chain.awaiting_finalize and existing.end_signal_sec is not None:
            self._revoke_end(existing, observation.t_sec)

    def note_placement(self, side: str, t_sec: float) -> None:
        """終了後の操作加点は、次のツモを操作できた物理証拠として保存する。"""
        chain = next((c for r in reversed(self.records) if r.game_idx == self._game_idx
                      for c in reversed(r.chains) if c.side == side), None)
        if chain is not None and chain.end_signal_sec is not None and t_sec > chain.end_signal_sec:
            chain.post_end_drop_sec = t_sec

    def _push_chain_observation(self, observation: ChainObservation) -> None:
        """終了→次ツモ操作を挟んだ発火を、古い式セッションへ再結合しない。"""
        active = next((c for c in self.resolver.active() if c.side == observation.side), None)
        identity = self._chain_aliases.get(active.chain_id, active.chain_id) if active else None
        previous = next((c for r in reversed(self.records) if r.game_idx == self._game_idx
                         for c in reversed(r.chains) if c.chain_id == identity), None)
        if (observation.kind == ObservationKind.FORMULA_STEP and previous is not None
                and previous.end_signal_sec is not None and previous.post_end_drop_sec is not None
                and previous.score_delta is not None):
            self.resolver.push(ChainObservation(observation.side, observation.t_sec,
                ObservationKind.SCORE_FINALIZE, total_score=int(previous.score_delta)))
        self.resolver.push(observation)

    def _revoke_end(self, chain: ExchangeChainRecord, t_sec: float) -> None:
        """同じ側の活動再開を記録し、終了候補と早期得点を取り消す。"""
        chain.end_signals[-1]["revoked_sec"] = t_sec
        chain.end_signal_sec, chain.end_reason = None, None
        chain.score_ready_sec, chain.score_ready_reason = None, None
        chain.score_delta, chain.stable_frames = None, 0
        chain.post_end_drop_sec = None
        self._s3_sec = None
        chain.last_activity_sec = t_sec
        if self.current is not None:
            values = [v for v in self.current.values if v["source"] == "S1"]
            self.source, self.probability = "S1", values[-1]["p1"]

    def activity(self, side: str, t_sec: float) -> None:
        """掛け算式の実表示・得点変化は同じ側の終了合図を撤回する。"""
        if self.current is None:
            return
        chain = self.latest_chain(side)
        if chain is None or (chain.end_signal_sec is not None and chain.post_end_drop_sec is not None):
            return
        self._last_activity_sec = t_sec
        if chain is not None:
            chain.last_activity_sec = t_sec
            if chain.end_signal_sec is not None and t_sec > chain.end_signal_sec:
                self._revoke_end(chain, t_sec)

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
        self._last_activity_sec = t_sec
        chain.end_signals.append(dict(t_sec=t_sec, reason=reason))
        self.resolver.push(ChainObservation(side, t_sec, ObservationKind.CHAIN_END_SIGNAL))

    def finalize(self, side: str, t_sec: float, score_delta: float) -> None:
        """既存OCR確定通知を同じchain_idへ供給する。ledger closeは参照しない。"""
        if not valid_nonnegative(score_delta):
            self.missing_input("invalid_score_finalize", t_sec, "S3", (side, t_sec))
            return
        pending = [c for r in self.records if r.game_idx == self._game_idx
                   for c in r.chains if c.side == side and c.score_finalize_sec is None]
        if not pending:
            return
        chain = pending[-1]
        chain.score_finalize_sec = t_sec
        if self.current is not None and chain in self.current.chains:
            self._last_activity_sec = t_sec
        if any(self._chain_aliases.get(c.chain_id, c.chain_id) == chain.chain_id
               for c in self.resolver.active()):
            self.resolver.push(ChainObservation(side, t_sec, ObservationKind.SCORE_FINALIZE,
                                                total_score=int(score_delta)))
        if chain.score_ready_sec is None:
            self._score_ready(chain, t_sec, score_delta, "score_finalize")

    def observe_score(self, side: str, t_sec: float, score: float | None,
                      score_before: float | None = None,
                      formula_total: float | None = None) -> None:
        """消去得点の再開だけを活動とし、落下加点で終了を撤回しない。"""
        chain = self.latest_chain(side)
        if chain is None:
            return
        if chain.score_before is None and valid_nonnegative(score_before):
            chain.score_before = score_before + chain.drop_bonus_score
        previous_score = chain.display_score
        if (chain.end_signal_sec is not None and chain.post_end_drop_sec is not None
                and valid_nonnegative(score) and previous_score is not None
                and score - previous_score >= CHAIN_TOTAL_MIN_SCORE):
            return  # 操作可能になった後の消去得点を、終了済み連鎖へ足さない。
        unchanged = score == previous_score
        if valid_nonnegative(score):
            if previous_score is not None and score != previous_score:
                unchanged = self._score_changed(chain, t_sec, score - previous_score)
            chain.display_score = score
        if valid_nonnegative(formula_total) and formula_total > 0:
            chain.formula_total = formula_total
        if chain.end_signal_sec is None or chain.score_ready_sec is not None:
            return
        if not valid_nonnegative(score) or chain.score_before is None:
            chain.display_score, chain.stable_frames = None, 0
            self.missing_input("missing_display_score_or_baseline", t_sec, "S3", chain.chain_id)
            return
        chain.stable_frames = chain.stable_frames + 1 if unchanged else 1
        chain.display_score = score
        delta = score - chain.score_before
        if delta < 0:
            chain.stable_frames = 0
        elif chain.formula_total is not None and delta == chain.formula_total:
            self._score_ready(chain, t_sec, chain.formula_total, "formula_match")
        elif chain.stable_frames >= S3_SCORE_STABLE_FRAMES:
            self._score_ready(chain, t_sec, delta, "display_stable")

    def _score_changed(self, chain: ExchangeChainRecord, t_sec: float, delta: float) -> bool:
        """終了後の落下加点を基準へ吸収し、連鎖得点が不変ならTrueを返す。"""
        if 0 < delta < CHAIN_TOTAL_MIN_SCORE:
            if chain.end_signal_sec is not None:
                chain.drop_bonus_score += delta
                if chain.score_before is not None:
                    chain.score_before += delta
                return True
            return False  # 終了前でも落下加点は連鎖活動ではない。
        # 4個消去×10点が最小連鎖得点。通常落下の端数加点は連鎖再開ではない。
        if delta >= CHAIN_TOTAL_MIN_SCORE:
            self.activity(chain.side, t_sec)
            return False
        # 得点の減少はOCR異常として再確認するが、活動タイマーは延長しない。
        chain.score_ready_sec, chain.score_ready_reason = None, None
        chain.score_delta, chain.stable_frames = None, 0
        if chain.end_signal_sec is not None and t_sec > chain.end_signal_sec:
            values = [v for v in self.current.values if v["source"] == "S1"]
            self.source, self.probability = "S1", values[-1]["p1"]
        return False

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
        if t_sec - self._last_activity_sec >= EXCHANGE_IDLE_TIMEOUT_SEC:
            self._close(t_sec, "activity_timeout")
            return
        chains = self.current.chains
        if any(c.end_signal_sec is None or c.score_ready_sec is None for c in chains):
            return
        if any(t_sec - c.end_signal_sec + TIME_EPSILON_SEC < S3_END_QUIET_SEC for c in chains):
            return
        finalized = max(c.score_ready_sec for c in chains)
        if self._s3_sec is not None and finalized <= self._s3_sec:
            return
        if any(not valid_nonnegative(c.score_delta) for c in chains):
            self.missing_input("missing_chain_score", t_sec, "S3", self.current.exchange_id)
            return
        totals = np.array([sum(c.score_delta for c in chains if c.side == side)
                           for side in SIDE_LABELS])
        try:
            event = ExchangeEndInput(self.firing, np.zeros(2), totals, self._score_elapsed)
        except (ValueError, TypeError) as error:
            self.missing_input("end_input: " + str(error), t_sec, "S3", self.current.exchange_id)
            return
        if self._evaluate(event, "S3", t_sec):
            self._s3_sec = t_sec

    def ready_for_static(self, t_sec: float, confirmed_times: tuple[float, float]) -> bool:
        """最後の参加連鎖の得点確定以降、両側の確定盤面が揃えば閉じる。"""
        if self.current is None:
            return True
        if not self.current.chains or any(c.score_ready_sec is None or c.end_signal_sec is None
                                          for c in self.current.chains):
            return False
        # 終了確認の猶予を飛び越えてS1→G_feに直行させず、S3を実表示してから戻す。
        if self._s3_sec is None or t_sec <= self._s3_sec:
            return False
        cutoff = max(c.score_ready_sec for c in self.current.chains)
        return self._s3_sec >= cutoff and min(confirmed_times) > cutoff

    def static(self, event: StaticInput, t_sec: float,
               confirmed_times: tuple[float, float]) -> bool:
        """撃ち合い優先を保ったまま、着地後のG_feへ戻す。"""
        if not self.ready_for_static(t_sec, confirmed_times):
            return False
        if not self._evaluate(event, "G_fe", t_sec):
            return False
        self.close_confirmed(t_sec, confirmed_times)
        return True

    def close_confirmed(self, t_sec: float, confirmed_times: tuple[float, float]) -> bool:
        """モデルの供給状態と独立に、着地後の確定盤面で物理区間を閉じる。"""
        if not self.ready_for_static(t_sec, confirmed_times):
            return False
        if self.current is not None:
            self._close(t_sec, "confirmed_after_score")
        return True

    def _close(self, t_sec: float, reason: str) -> None:
        """閉じたS1/S3は表示に残さず、直近の静止評価へ戻す。"""
        self.current.closed_sec, self.current.close_reason = t_sec, reason
        self.current, self.firing, self._s3_sec = None, None, None
        self.probability = self._static_probability
        self.source = "G_fe" if self.probability is not None else "waiting_confirmed"

    def _evaluate(self, event: StaticInput | FiringInput | ExchangeEndInput,
                  source: str, t_sec: float) -> bool:
        """イベントの勝率を保持し、表示値と計装値を一致させる。"""
        try:
            probability = evaluate_exchange_event(event, self.models)
        except (ValueError, TypeError, FloatingPointError) as error:
            self.missing_input("evaluation: " + str(error), t_sec, source)
            return False
        self.probability = probability
        self.source = source
        if source == "G_fe":
            self._static_probability = probability
        if self.current is not None:
            self.current.values.append(dict(source=source, t_sec=t_sec, p1=self.probability))
        return True

    def save(self, path: Path) -> None:
        """未完の撃ち合いも欠落させずJSONLへ保存する。"""
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as stream:
            for record in self.records:
                stream.write(json.dumps(asdict(record), ensure_ascii=False, allow_nan=False) + "\n")
        counts = {reason: sum(row["reason"] == reason for row in self.diagnostics)
                  for reason in {row["reason"] for row in self.diagnostics}}
        path.with_suffix(".diagnostics.json").write_text(json.dumps(
            dict(counts=counts, events=self.diagnostics), ensure_ascii=False,
            allow_nan=False, indent=2), encoding="utf-8")
