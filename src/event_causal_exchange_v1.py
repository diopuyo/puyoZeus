"""公開時点の情報だけで攻撃残量を再構成する因果会計。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


SIDES = ("p1", "p2")
SIGN = {"p1": 1, "p2": -1}
RECIPIENT_SIGN = {"p1": -1, "p2": 1}


@dataclass(slots=True)
class CausalExchangeReplay:
    """推定落下を使わず、確定攻撃と盤面差から未解決量を再生する。"""

    finalized_balance: int = 0
    provisional_by_attack: dict[str, tuple[str, int]] = field(default_factory=dict)
    provisional_event_attack: dict[str, str] = field(default_factory=dict)
    unallocated_landing: dict[str, int] = field(
        default_factory=lambda: {side: 0 for side in SIDES}
    )
    effective_rate: dict[str, int | None] = field(
        default_factory=lambda: {side: None for side in SIDES}
    )
    quarantined_finalized_count: int = 0
    quarantined_finalized_amount: int = 0
    quarantined_landing_count: int = 0
    quarantined_landing_amount: int = 0

    def reset(self) -> None:
        """正式な試合境界で旧試合の状態を破棄する。"""
        self.finalized_balance = 0
        self.provisional_by_attack.clear()
        self.provisional_event_attack.clear()
        self.unallocated_landing = {side: 0 for side in SIDES}
        self.effective_rate = {side: None for side in SIDES}
        self.quarantined_finalized_count = 0
        self.quarantined_finalized_amount = 0
        self.quarantined_landing_count = 0
        self.quarantined_landing_amount = 0

    def observe(self, event: Mapping[str, Any]) -> None:
        """一つの出来事を、その公開順のまま反映する。"""
        event_type = str(event.get("event_type", ""))
        if event_type == "attack_provisional_updated":
            self._observe_provisional(event)
        elif event_type == "attack_finalized":
            self._observe_finalized(event)
        elif event_type == "attack_provisional_unfinalized":
            side = _side(event)
            self._remove_attack(_relation_attack_id(event, side))
        elif event_type == "garbage_landing_board_compared":
            self._observe_physical_landing(event)

    @property
    def usable(self) -> bool:
        """一つの物理連鎖へ帰属できない確定攻撃が混じっていないか。"""
        return not (self.quarantined_finalized_count or self.quarantined_landing_count)

    @property
    def reason_codes(self) -> tuple[str, ...]:
        """現在残量を信用できない理由を、表示と監査へ公開する。"""
        reasons = []
        if self.quarantined_finalized_count:
            reasons.append("causal_exchange_ambiguous_attack_finalization")
        if self.quarantined_landing_count:
            reasons.append("causal_exchange_ambiguous_landing_observation")
        return tuple(reasons)

    def pending(self, recipient: str) -> int | None:
        """確定攻撃だけから求めた受け手別の未解決量を返す。"""
        if recipient not in SIDES or not self.usable:
            return None
        if recipient == "p2":
            return max(0, self.finalized_balance)
        return max(0, -self.finalized_balance)

    def observed_balance(self) -> int | None:
        """確定残量に現在観測できた進行中攻撃を加えた符号付き差を返す。"""
        if not self.usable:
            return None
        provisional = sum(
            SIGN[side] * amount
            for side, amount in self.provisional_by_attack.values()
        )
        return self._after_visible_landing_credit(
            self.finalized_balance + provisional,
        )

    def rate(self, side: str) -> int | None:
        """その側について最後に直接観測した1個当たり得点を返す。"""
        return self.effective_rate.get(side)

    def _observe_provisional(self, event: Mapping[str, Any]) -> None:
        side = _side(event)
        attack_id = _relation_attack_id(event, side)
        amount = _nonnegative_int(event, "provisional_generated_amount")
        self._observe_rate(event, side)
        self.provisional_by_attack[attack_id] = (side, amount)
        self.provisional_event_attack[str(event["event_id"])] = attack_id

    def _observe_finalized(self, event: Mapping[str, Any]) -> None:
        targets = event.get("relations", {}).get("revision", {}).get(
            "target_event_ids", (),
        )
        if isinstance(targets, list):
            for event_id in targets:
                self._remove_attack(self.provisional_event_attack.get(str(event_id)))
        side = _side(event)
        self._remove_attack(_relation_attack_id(event, side))
        self._remove_attack(f"legacy-active:{side}")
        amount = _nonnegative_int(event, "generated_amount")
        self._observe_rate(event, side)
        if amount and not _single_chain_finalization(event):
            self.quarantined_finalized_count += 1
            self.quarantined_finalized_amount += amount
            return
        self.finalized_balance += SIGN[side] * amount
        self._allocate_visible_landings()

    def _remove_attack(self, attack_id: str | None) -> None:
        if attack_id is not None:
            self.provisional_by_attack.pop(attack_id, None)

    def _observe_rate(self, event: Mapping[str, Any], side: str) -> None:
        value = event.get("payload", {}).get("effective_rate")
        if value is None:
            return
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError("effective_rateが正整数ではありません")
        self.effective_rate[side] = value

    def _observe_physical_landing(self, event: Mapping[str, Any]) -> None:
        amount = _positive_visible_landing_amount(event)
        if amount is None:
            return
        if not _is_clean_visible_landing(event, amount):
            self.quarantined_landing_count += 1
            self.quarantined_landing_amount += amount
            return
        side = _side(event)
        self.unallocated_landing[side] += amount
        self._allocate_visible_landings()

    def _allocate_visible_landings(self) -> None:
        """符号が一致する確定残量へ、盤面で見えた落下だけを割り当てる。"""
        if self.finalized_balance > 0:
            used = min(self.finalized_balance, self.unallocated_landing["p2"])
            self.finalized_balance -= used
            self.unallocated_landing["p2"] -= used
        elif self.finalized_balance < 0:
            used = min(-self.finalized_balance, self.unallocated_landing["p1"])
            self.finalized_balance += used
            self.unallocated_landing["p1"] -= used

    def _after_visible_landing_credit(self, balance: int) -> int:
        """確定より先に見えた落下を、符号を反転させない範囲で相殺する。"""
        if balance > 0:
            return max(0, balance - self.unallocated_landing["p2"])
        if balance < 0:
            return min(0, balance + self.unallocated_landing["p1"])
        return 0


def settled_pending_disagreement(
    physical_pending: Mapping[str, int],
    causal_pending: Mapping[str, int | None],
    chain_active: Mapping[str, bool],
    provisional_generated: Mapping[str, int | None],
    *,
    causal_usable: bool,
) -> int:
    """物理会計と因果会計の確定済み未処理量の差を返す。"""

    if not causal_usable:
        return 0
    values = {
        side: (
            _strict_nonnegative(physical_pending.get(side), "physical pending"),
            _strict_nonnegative(causal_pending.get(side), "causal pending"),
            _strict_optional_nonnegative(
                provisional_generated.get(side), "provisional generated",
            ),
        )
        for side in SIDES
    }
    if any(type(chain_active.get(side)) is not bool for side in SIDES):
        raise ValueError("chain activeがboolではありません")
    # 両pendingは確定攻撃だけを含む。active/provisionalは値域だけ検証し、
    # 無関係な進行中連鎖で確定済み会計の不一致を隠さない。
    return sum(abs(values[side][0] - values[side][1]) for side in SIDES)


def _strict_nonnegative(value: Any, label: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{label}が非負整数ではありません")
    return value


def _strict_optional_nonnegative(value: Any, label: str) -> int | None:
    if value is None:
        return None
    return _strict_nonnegative(value, label)


def _relation_attack_id(event: Mapping[str, Any], side: str | None = None) -> str:
    value = event.get("relations", {}).get("attack_id")
    if isinstance(value, str) and value:
        return value
    if side in SIDES:
        return f"legacy-active:{side}"
    raise ValueError("攻撃IDがありません")


def _side(event: Mapping[str, Any]) -> str:
    value = event.get("side")
    if value not in SIDES:
        raise ValueError("攻撃または落下のsideが不正です")
    return str(value)


def _nonnegative_int(event: Mapping[str, Any], key: str) -> int:
    value = event.get("payload", {}).get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{key}が非負整数ではありません")
    return value


def _positive_visible_landing_amount(event: Mapping[str, Any]) -> int | None:
    payload = event.get("payload", {})
    differences = payload.get("differences")
    if not isinstance(differences, Mapping):
        return None
    garbage = differences.get("garbage")
    valid_amount = isinstance(garbage, int) and not isinstance(garbage, bool) and garbage > 0
    return int(garbage) if valid_amount else None


def _is_clean_visible_landing(event: Mapping[str, Any], amount: int) -> bool:
    payload = event.get("payload", {})
    differences = payload.get("differences", {})
    return bool(
        payload.get("before_is_immediate_previous_observation") is True
        and isinstance(differences, Mapping)
        and differences.get("color") == 0
        and differences.get("unknown") == 0
        and differences.get("occupied") == amount
    )


def _single_chain_finalization(event: Mapping[str, Any]) -> bool:
    """明示的に複数・未観測とされた確定攻撃だけを信用対象から外す。"""
    state = event.get("payload", {}).get("chain_relation_state")
    return state in {None, "single_observed_chain"}


__all__ = ["CausalExchangeReplay", "settled_pending_disagreement"]
