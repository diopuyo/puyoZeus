"""交換episodeを動画処理中に増分更新する既定OFFアダプタ。"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable

from src.chain_id_resolver import (
    FINALIZED_SOURCE_SCORE_OCR_DIFF,
    FINALIZED_SOURCE_SIMULATE_FALLBACK,
    ActiveChainSnapshot,
    ChainObservation,
    ObservationKind,
    ResolvedChain,
)
from src.exchange_episode_tracker import (
    ChainEventObservation,
    ExchangeEpisodeTracker,
    GenerationObservation,
    SettlementObservation,
)
from src.exchange_ledger import (
    ClosedEpisodeSummary,
    EventKind,
    ExchangeEvent,
    LedgerSnapshot,
    PhysicalContext,
    Side,
)

_SIDE_MAP: dict[str, Side] = {"1P": Side.P1, "2P": Side.P2}
_EPS: float = 1e-9


@dataclass(frozen=True)
class LiveEpisodeSnapshot:
    """1フレーム時点の台帳とchain帰属をdump可能な形で返す。"""

    ledger: LedgerSnapshot
    active_chains: tuple[ActiveChainSnapshot, ...]
    latest_chain_id_p1: int | None
    latest_chain_id_p2: int | None
    latest_generation_p1: float
    latest_generation_p2: float
    resolved_chain_count: int
    closed_episode_count: int
    closed_unreconciled_total: float
    closed_normal_unreconciled_count: int
    last_close_reason: str
    last_closed_status: str
    last_closed_generated: float
    last_closed_canceled: float
    last_closed_landed: float
    last_closed_unreconciled: float
    last_closed_has_settlement: bool
    last_closed_oversettled: float
    last_closed_oversettled_chain_count: int
    unattributed_settlement_total: float
    open_episode_outstanding: float
    ledger_residual_all: float
    simulate_excluded_chain_count: int
    simulate_excluded_amount: float
    formula_step_observation_count: int
    provisional_score_decrease_ignored_count: int
    boundary_count: int
    boundary_settlement_excluded_count: int
    boundary_settlement_excluded_amount: float


class LiveExchangeEpisodeTracker(ExchangeEpisodeTracker):
    """既存resolver/ledgerを再利用し、毎フレームの未解決状態を公開する。"""

    def __init__(
        self, *, enabled: bool = False, allow_simulate_fallback: bool = False,
    ) -> None:
        super().__init__(
            enabled=enabled, allow_simulate_fallback=allow_simulate_fallback,
            defer_max_sec_close=True)
        self._live_amount: dict[int, float] = {}
        self._start_pending: dict[int, float] = {}
        self._resolved_cursor = 0
        self._last_context: PhysicalContext | None = None
        self._latest_chain_id: dict[Side, int | None] = {
            Side.P1: None, Side.P2: None}
        self._latest_generation: dict[Side, float] = {Side.P1: 0.0, Side.P2: 0.0}
        self._simulate_excluded_chain_count = 0
        self._simulate_excluded_amount = 0.0
        self._boundary_count = 0
        self._boundary_settlement_excluded_count = 0
        self._boundary_settlement_excluded_amount = 0.0

    def observe_frame(
        self, *, t_sec: float, context: PhysicalContext,
        chain_observations: Iterable[ChainEventObservation] = (),
        generation_observations: Iterable[GenerationObservation] = (),
        settlement: SettlementObservation | None = None,
        wiped_sides: Iterable[Side] = (),
    ) -> LiveEpisodeSnapshot:
        """同一frameの事実をchain→wipe→settlementの物理順で反映する。"""
        if not self._enabled:
            return self._snapshot(context)
        boundary_advanced = self._advance_game_boundary(t_sec, context)
        for obs in chain_observations:
            super().observe(obs)
        # state machine の CHAIN/OJAMA_FALL 瞬断より、chain resolver が持つ
        # 物理連鎖の寿命を優先する。生成量がまだ 0 の開始直後も、交換は既に
        # 始まっているため「解決済み」として扱わない。
        effective_context = self._context_with_active_chains(context)
        self._sync_active(effective_context)
        for obs in generation_observations:
            super().observe_generation(obs)
        # SCORE_FINALIZE により片側だけ解決する場合があるため、active集合を
        # 更新後にもう一度取り直す。これにより反撃側だけが生きているフレームも
        # hard override の victim_chaining 判定へ確実に届く。
        effective_context = self._context_with_active_chains(context)
        self_cancel = self._self_cancel_by_side(settlement)
        self._sync_resolved(effective_context, self_cancel)
        for side in wiped_sides:
            # side wipe は resolver の残存activeより強い物理終端。ここへ
            # chainingラッチを渡すと正常closeを不必要に遅らせるため、生の
            # contextで明示終端を処理する。
            self._ledger.retire_side_chains(side, t_sec, context)
        if settlement is not None and not boundary_advanced:
            self._process_settlement(settlement)
        elif settlement is not None:
            self._record_boundary_settlement_exclusion(settlement)
        # FINALIZE/CANCEL/LANDを同一frameの原子的入力として全反映した後に
        # max_secを判定し、close直後の同frame着弾を取りこぼさない。
        self._ledger.close_expired(t_sec)
        self._last_context = effective_context
        snapshot = self._snapshot(effective_context)
        self._track_net_divergence(snapshot.ledger)
        return snapshot

    def finish_live(self, context: PhysicalContext | None = None) -> LiveEpisodeSnapshot:
        """resolverだけをflushし、区間端OPEN episodeは強制closeせず残す。"""
        ctx = context or self._last_context or PhysicalContext()
        if self._enabled:
            self._resolver.flush()
            self._sync_resolved(ctx, {Side.P1: 0.0, Side.P2: 0.0})
        return self._snapshot(ctx)

    def finish(self) -> None:
        """親のoffline再生を使わず、増分台帳をそのままflushする。"""
        self.finish_live()

    def closed_episodes_live(self) -> list[ClosedEpisodeSummary]:
        """正常close/強制closeの全履歴をコピーで返す。"""
        return self._ledger.closed_episodes()

    def resolved_chains_live(self) -> list[ResolvedChain]:
        """resolverが解決済みにした全chainをコピーで返す。"""
        return self._resolver.resolved()

    def _advance_game_boundary(self, t_sec: float, context: PhysicalContext) -> bool:
        previous = self._last_context
        if previous is None or previous.game_idx == context.game_idx:
            return False
        self._resolver.push(ChainObservation(
            side="1P", t_sec=t_sec, kind=ObservationKind.MATCH_BOUNDARY))
        self._sync_resolved(previous, {Side.P1: 0.0, Side.P2: 0.0})
        self._ledger.push(ExchangeEvent(
            kind=EventKind.TSUMO_PLACED, side=Side.P1, t_sec=t_sec,
            source="formal_match_boundary"), context)
        self._last_game_idx = context.game_idx
        self._boundary_count += 1
        return True

    def _record_boundary_settlement_exclusion(
        self, settlement: SettlementObservation,
    ) -> None:
        """境界ワイプを着弾・相殺として二重適用せず、量を監査に残す。"""
        amount = sum((
            settlement.canceled_by_1p, settlement.canceled_by_2p,
            settlement.landed_on_1p, settlement.landed_on_2p))
        if amount <= 0.0:
            return
        self._boundary_settlement_excluded_count += 1
        self._boundary_settlement_excluded_amount += float(amount)

    def _sync_active(self, context: PhysicalContext) -> None:
        for active in self._resolver.active():
            if not active.growth_observed:
                continue
            side = self._side(active.side)
            amount = self._active_net_amount(active, side, context)
            previous = self._live_amount.get(active.chain_id)
            if previous is None:
                self._push_fire(active, side, amount, context)
            elif amount + _EPS < previous:
                raise ValueError(
                    f"active provisional decreased: chain={active.chain_id} "
                    f"{previous} -> {amount}")
            elif amount > previous + _EPS:
                self._push_step(active, side, amount - previous, context)
            self._live_amount[active.chain_id] = amount
            self._remember_chain(side, active.chain_id, amount)

    def _active_net_amount(
        self, active: ActiveChainSnapshot, side: Side, context: PhysicalContext,
    ) -> float:
        raw = float(self._to_ojama(active.provisional_score, self._elapsed_at_open(active)))
        pending_now = (
            context.p1_pending_uncapped if side is Side.P1
            else context.p2_pending_uncapped)
        pending_at_start = self._start_pending.setdefault(
            active.chain_id, float(pending_now))
        return max(0.0, raw - pending_at_start)

    def _elapsed_at_open(self, active: ActiveChainSnapshot) -> float:
        ctx = self._context_by_t.get(active.opened_at_sec)
        return ctx.elapsed_sec if ctx is not None else 0.0

    def _push_fire(
        self, active: ActiveChainSnapshot, side: Side, amount: float,
        context: PhysicalContext,
    ) -> None:
        event_context = self._context_for_open(active, context)
        self._ledger.push(ExchangeEvent(
            kind=EventKind.FIRE, side=side, t_sec=active.opened_at_sec,
            amount=amount, chain_id=active.chain_id,
            chain_count=active.step_count, source="live_formula"), event_context)

    def _push_step(
        self, active: ActiveChainSnapshot, side: Side, delta: float,
        context: PhysicalContext,
    ) -> None:
        self._ledger.push(ExchangeEvent(
            kind=EventKind.STEP, side=side, t_sec=active.last_t_sec,
            amount=delta, chain_id=active.chain_id,
            chain_count=active.step_count, source="live_formula"), context)

    def _context_for_open(
        self, active: ActiveChainSnapshot, context: PhysicalContext,
    ) -> PhysicalContext:
        observed = self._context_by_t.get(active.opened_at_sec)
        return replace(context, game_idx=(
            observed.game_idx if observed is not None else context.game_idx))

    def _sync_resolved(
        self, context: PhysicalContext, self_cancel: dict[Side, float],
    ) -> None:
        resolved = self._resolver.resolved()
        for chain in resolved[self._resolved_cursor:]:
            self._sync_one_resolved(chain, context, self_cancel)
        self._resolved_cursor = len(resolved)

    def _sync_one_resolved(
        self, chain: ResolvedChain, context: PhysicalContext,
        self_cancel: dict[Side, float],
    ) -> None:
        side = self._side(chain.side)
        if self._exclude_simulate(chain):
            self._record_simulate_exclusion(chain)
            self._live_amount.pop(chain.chain_id, None)
            self._start_pending.pop(chain.chain_id, None)
            return
        provisional = max(
            0.0, float(self._provisional_ojama_for(chain)) - self_cancel[side])
        if chain.chain_id not in self._live_amount:
            self._push_late_fire(chain, side, provisional, context)
        if chain.was_finalized:
            finalized = max(
                0.0, float(self._finalized_ojama_for(chain)) - self_cancel[side])
            self._push_finalize(chain, side, finalized, context)
            self._remember_chain(side, chain.chain_id, finalized)
        else:
            self._remember_chain(side, chain.chain_id, provisional)
        self._live_amount.pop(chain.chain_id, None)
        self._start_pending.pop(chain.chain_id, None)

    def _exclude_simulate(self, chain: ResolvedChain) -> bool:
        return (
            not self._allow_simulate_fallback
            and chain.finalized_source == FINALIZED_SOURCE_SIMULATE_FALLBACK)

    def _record_simulate_exclusion(self, chain: ResolvedChain) -> None:
        """simulate-only chainを会計外へ出した件数と量を黙って捨てない。"""
        amount = (
            self._finalized_ojama_for(chain)
            if chain.was_finalized else self._provisional_ojama_for(chain))
        self._simulate_excluded_chain_count += 1
        self._simulate_excluded_amount += float(amount)

    def _push_late_fire(
        self, chain: ResolvedChain, side: Side, amount: float,
        context: PhysicalContext,
    ) -> None:
        self._ledger.push(ExchangeEvent(
            kind=EventKind.FIRE, side=side, t_sec=chain.closed_at_sec,
            amount=amount, chain_id=chain.chain_id, chain_count=chain.step_count,
            source="live_authoritative_late_fire"), context)

    def _push_finalize(
        self, chain: ResolvedChain, side: Side, amount: float,
        context: PhysicalContext,
    ) -> None:
        source = chain.finalized_source or FINALIZED_SOURCE_SCORE_OCR_DIFF
        self._ledger.push(ExchangeEvent(
            kind=EventKind.FINALIZE, side=side, t_sec=chain.closed_at_sec,
            amount=amount, chain_id=chain.chain_id, source=source), context)

    def _remember_chain(self, side: Side, chain_id: int, amount: float) -> None:
        self._latest_chain_id[side] = chain_id
        self._latest_generation[side] = amount

    def _side(self, label: str) -> Side:
        side = _SIDE_MAP.get(label)
        if side is None:
            raise ValueError(f"unknown side: {label}")
        return side

    def _self_cancel_by_side(
        self, settlement: SettlementObservation | None,
    ) -> dict[Side, float]:
        if settlement is None:
            return {Side.P1: 0.0, Side.P2: 0.0}
        return {
            Side.P1: float(settlement.canceled_by_1p),
            Side.P2: float(settlement.canceled_by_2p),
        }

    def _context_with_active_chains(
        self, context: PhysicalContext,
    ) -> PhysicalContext:
        """resolver が保持する物理連鎖を context の chaining 状態へラッチする。"""
        active_sides = {item.side for item in self._resolver.active()}
        p1_chaining = context.p1_chaining or "1P" in active_sides
        p2_chaining = context.p2_chaining or "2P" in active_sides
        if (p1_chaining, p2_chaining) == (
            context.p1_chaining, context.p2_chaining,
        ):
            return context
        return replace(
            context, p1_chaining=p1_chaining, p2_chaining=p2_chaining)

    @staticmethod
    def _active_only_death_target(context: PhysicalContext) -> float | None:
        """台帳量が未到着でも、片側だけの物理死亡は確定方向として残す。"""
        if context.p1_dead == context.p2_dead:
            return None
        return -100.0 if context.p1_dead else 100.0

    def _effective_ledger_snapshot(
        self, context: PhysicalContext,
        active: tuple[ActiveChainSnapshot, ...],
    ) -> LedgerSnapshot:
        """FIRE 前の物理連鎖も未解決として公開する表示用スナップショット。"""
        ledger = self._ledger.snapshot(context)
        if not active or ledger.is_unresolved:
            return ledger
        target = self._active_only_death_target(context)
        return replace(
            ledger, is_unresolved=True,
            allows_hard_override=target is not None,
            hard_override_target=target)

    def _snapshot(self, context: PhysicalContext) -> LiveEpisodeSnapshot:
        closed = self._ledger.closed_episodes()
        last = closed[-1] if closed else None
        resolver_stats = self._resolver.stats()
        active = tuple(self._resolver.active())
        return LiveEpisodeSnapshot(
            ledger=self._effective_ledger_snapshot(context, active),
            active_chains=active,
            latest_chain_id_p1=self._latest_chain_id[Side.P1],
            latest_chain_id_p2=self._latest_chain_id[Side.P2],
            latest_generation_p1=self._latest_generation[Side.P1],
            latest_generation_p2=self._latest_generation[Side.P2],
            resolved_chain_count=len(self._resolver.resolved()),
            closed_episode_count=len(closed),
            closed_unreconciled_total=sum(item.unreconciled for item in closed),
            closed_normal_unreconciled_count=(
                self._ledger.closed_normal_unreconciled_count()),
            last_close_reason=(last.close_reason if last else ""),
            last_closed_status=(last.status.name if last else ""),
            last_closed_generated=(last.total_generated if last else 0.0),
            last_closed_canceled=(last.total_canceled if last else 0.0),
            last_closed_landed=(last.total_landed if last else 0.0),
            last_closed_unreconciled=(last.unreconciled if last else 0.0),
            last_closed_has_settlement=(last.has_settlement_input if last else False),
            last_closed_oversettled=(last.oversettled if last else 0.0),
            last_closed_oversettled_chain_count=(
                last.oversettled_chain_count if last else 0),
            unattributed_settlement_total=self._unattributed_settlement_total,
            open_episode_outstanding=self._ledger.open_episode_outstanding(),
            ledger_residual_all=self._ledger.total_outstanding_all_chains(),
            simulate_excluded_chain_count=self._simulate_excluded_chain_count,
            simulate_excluded_amount=self._simulate_excluded_amount,
            formula_step_observation_count=(
                resolver_stats.formula_step_observation_count),
            provisional_score_decrease_ignored_count=(
                resolver_stats.provisional_score_decrease_ignored_count),
            boundary_count=self._boundary_count,
            boundary_settlement_excluded_count=(
                self._boundary_settlement_excluded_count),
            boundary_settlement_excluded_amount=(
                self._boundary_settlement_excluded_amount),
        )


__all__ = ["LiveEpisodeSnapshot", "LiveExchangeEpisodeTracker"]
