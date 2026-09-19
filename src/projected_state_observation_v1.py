"""正式projected-stateモデルへ渡す不変な観測契約。"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, replace
from enum import StrEnum
from itertools import combinations
from math import comb
from typing import Any

from src.board import BOARD_COLS, BOARD_ROWS, COLOR_UNKNOWN, VALID_COLORS
from src.scoring import OJAMA_MAX_DROP_PER_TURN


PROJECTED_STATE_SCHEMA_VERSION = "projected-state-observation-v1"
GATE_STATUSES = frozenset({"guaranteed", "not_applicable", "untrusted"})
SIDES = ("p1", "p2")
BOARD_PROVENANCES = frozenset({
    "confirmed", "observed", "physics_projected", "unknown",
})
MAX_LANDING_BRANCHES = comb(BOARD_COLS, BOARD_COLS // 2)
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")

ImmutableGrid = tuple[tuple[int, ...], ...]
ImmutableMask = tuple[tuple[bool, ...], ...]


class ProjectedStateContractError(ValueError):
    """projected-state入力が正式契約を満たさない。"""


class ProjectedStateReason(StrEnum):
    """v1で意味を固定したside非依存reason code。"""

    LANDING_COLUMNS_AMBIGUOUS = "landing_columns_ambiguous"
    PREFIRE_INCOMING_ZERO = "prefire_incoming_zero"
    PROJECTED_BALANCE_ZERO = "projected_balance_zero"
    RECIPIENT_CHANGED_FROM_PREFIRE_SIDE = "recipient_changed_from_prefire_side"
    RECIPIENT_NOT_ACTIVE_CHAIN_SIDE = "recipient_not_active_chain_side"
    CHAIN_STATE_MISSING = "chain_state_missing"
    ACTIVE_FAMILY_MISSING = "active_family_missing"
    ACTIVE_FAMILY_UNTRUSTED = "active_family_untrusted"
    PREFIRE_ANCHOR_MISSING = "prefire_anchor_missing"
    PREFIRE_ANCHOR_AMBIGUOUS = "prefire_anchor_ambiguous"
    EFFECTIVE_RATE_MISSING = "effective_rate_missing"
    LEFTOVER_SEQUENCE_DISCONTINUOUS = "leftover_sequence_discontinuous"
    CAUSAL_EXCHANGE_UNTRUSTED = "causal_exchange_untrusted"
    SOURCE_IDENTITY_MISMATCH = "source_identity_mismatch"
    BOUNDARY_MISMATCH = "boundary_mismatch"
    CUTOFF_MISMATCH = "cutoff_mismatch"
    ASSET_HASH_MISMATCH = "asset_hash_mismatch"
    CURRENT_BOARD_MISSING = "current_board_missing"
    POST_CHAIN_BOARD_MISSING = "post_chain_board_missing"
    FIRST_DROP_TIMING_UNTRUSTED = "first_drop_timing_untrusted"


NOT_APPLICABLE_REASONS = frozenset({
    ProjectedStateReason.PREFIRE_INCOMING_ZERO,
    ProjectedStateReason.PROJECTED_BALANCE_ZERO,
    ProjectedStateReason.RECIPIENT_CHANGED_FROM_PREFIRE_SIDE,
    ProjectedStateReason.RECIPIENT_NOT_ACTIVE_CHAIN_SIDE,
})
UNTRUSTED_REASONS = frozenset(ProjectedStateReason) - NOT_APPLICABLE_REASONS - {
    ProjectedStateReason.LANDING_COLUMNS_AMBIGUOUS,
}


@dataclass(frozen=True, slots=True)
class IntegerQuantity:
    """0と未観測を混同しない整数値。"""

    value: int | None
    present: bool

    def __post_init__(self) -> None:
        if type(self.present) is not bool:
            raise ProjectedStateContractError("quantity.presentはbool必須です")
        if self.present and type(self.value) is not int:
            raise ProjectedStateContractError("presentなquantityに整数値がありません")
        if not self.present and self.value is not None:
            raise ProjectedStateContractError("missing quantityはvalue=None必須です")

    @classmethod
    def known(cls, value: int) -> IntegerQuantity:
        """観測済み整数を作る。"""
        return cls(value=value, present=True)

    @classmethod
    def missing(cls) -> IntegerQuantity:
        """未観測値を作る。"""
        return cls(value=None, present=False)

    def negated(self) -> IntegerQuantity:
        """符号付き既知値だけを反転する。"""
        return self if not self.present else IntegerQuantity.known(-int(self.value))


@dataclass(frozen=True, slots=True)
class ProjectedBoardState:
    """出所を伴う13行6列の不変盤面。"""

    grid: ImmutableGrid | None
    unknown_mask: ImmutableMask | None
    provenance: str
    source_event_id: str | None
    present: bool = True

    def __post_init__(self) -> None:
        if type(self.present) is not bool:
            raise ProjectedStateContractError("盤面presentはbool必須です")
        if (not isinstance(self.provenance, str)
                or self.provenance not in BOARD_PROVENANCES):
            raise ProjectedStateContractError("盤面provenanceが正式値ではありません")
        if not self.present:
            if (self.grid is not None or self.unknown_mask is not None
                    or self.source_event_id is not None or self.provenance != "unknown"):
                raise ProjectedStateContractError("missing盤面の値または由来が不正です")
            return
        if self.provenance == "unknown":
            raise ProjectedStateContractError(
                "present盤面のprovenanceにunknownは使えません"
            )
        _validate_grid(self.grid)
        _validate_unknown_mask(self.grid, self.unknown_mask)
        _require_text(self.source_event_id, "盤面source event ID")

    @classmethod
    def missing(cls) -> ProjectedBoardState:
        """盤面未取得を全cell UNKNOWNとは別表現で作る。"""
        return cls(None, None, "unknown", None, present=False)


@dataclass(frozen=True, slots=True)
class DualBoardState:
    """同じ因果cutoffにおける1P/2P盤面。"""

    p1: ProjectedBoardState
    p2: ProjectedBoardState

    def __post_init__(self) -> None:
        if not isinstance(self.p1, ProjectedBoardState):
            raise ProjectedStateContractError("p1盤面の型が不正です")
        if not isinstance(self.p2, ProjectedBoardState):
            raise ProjectedStateContractError("p2盤面の型が不正です")

    def swap_sides(self) -> DualBoardState:
        """盤面の1P/2Pスロットを交換する。"""
        return DualBoardState(p1=self.p2, p2=self.p1)


@dataclass(frozen=True, slots=True)
class LandingBranch:
    """端数列の一つの可能性と、その初回着弾後の両盤面。"""

    remainder_columns: tuple[int, ...]
    boards: DualBoardState

    def __post_init__(self) -> None:
        _validate_remainder_columns(self.remainder_columns)
        if not isinstance(self.boards, DualBoardState):
            raise ProjectedStateContractError("着弾候補の両盤面が不正です")

    def swap_sides(self) -> LandingBranch:
        """端数列は維持し、盤面スロットだけを交換する。"""
        return LandingBranch(self.remainder_columns, self.boards.swap_sides())


@dataclass(frozen=True, slots=True)
class RootFamilyReference:
    """投影根拠となったactive root-familyの不変参照。"""

    side: str
    game_idx: int
    root_anchor: str
    prefire_anchor_event_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.side not in SIDES:
            raise ProjectedStateContractError("root-family sideがp1/p2ではありません")
        _require_nonnegative_int(self.game_idx, "root-family game index")
        _require_text(self.root_anchor, "root-family root anchor")
        if (type(self.prefire_anchor_event_ids) is not tuple
                or not self.prefire_anchor_event_ids):
            raise ProjectedStateContractError("root-family anchor IDsが空です")
        for event_id in self.prefire_anchor_event_ids:
            _require_text(event_id, "root-family prefire anchor event ID")

    def swap_side(self) -> RootFamilyReference:
        """familyの物理同一性を保ちsideだけ交換する。"""
        return replace(self, side=_opponent(self.side))


@dataclass(frozen=True, slots=True)
class ProjectedStateObservationV1:
    """学習・推論で共用する正式projected-state入力。"""

    observation_id: str
    source_video_id: str
    build_id: str
    attempt_id: str
    game_idx: int
    boundary_segment: int
    through_event_seq: int
    causal_cutoff_digest: str
    schema_hash: str
    adapter_hash: str
    physics_hash: str
    recognition_hash: str
    input_manifest_hash: str
    gate_status: str
    active_chain_side: str | None
    recipient: str | None
    reason_codes: tuple[ProjectedStateReason, ...]
    landing_columns_ambiguous: bool
    prefire_incoming_amount: IntegerQuantity
    signed_projected_balance: IntegerQuantity
    first_drop_amount: IntegerQuantity
    leftover_after_first_drop: IntegerQuantity
    current_state: DualBoardState
    post_chain_state: DualBoardState
    p1_root_family: RootFamilyReference | None = None
    p2_root_family: RootFamilyReference | None = None
    landing_branches: tuple[LandingBranch, ...] = ()

    def __post_init__(self) -> None:
        _validate_identity_and_hashes(self)
        _validate_gate(self)
        _validate_quantities(self)
        _validate_board_states(self)
        _validate_landing_branches(self)

    def to_dict(self) -> dict[str, Any]:
        """input_digestの対象となるJSON互換値を返す。"""
        return {"contract_version": PROJECTED_STATE_SCHEMA_VERSION, **asdict(self)}

    def canonical_json_bytes(self) -> bytes:
        """キー順・空白・改行を固定した正規JSONを返す。"""
        return canonical_json_bytes(self)

    @property
    def input_digest(self) -> str:
        """正式入力全体のSHA-256を返す。"""
        return hashlib.sha256(self.canonical_json_bytes()).hexdigest()

    def swap_sides(self) -> ProjectedStateObservationV1:
        """1P/2P交換を全side依存列へ一貫して適用する。"""
        recipient = None if self.recipient is None else _opponent(self.recipient)
        active_side = (
            None if self.active_chain_side is None
            else _opponent(self.active_chain_side)
        )
        return replace(
            self,
            active_chain_side=active_side,
            recipient=recipient,
            signed_projected_balance=self.signed_projected_balance.negated(),
            current_state=self.current_state.swap_sides(),
            post_chain_state=self.post_chain_state.swap_sides(),
            p1_root_family=_swap_family(self.p2_root_family),
            p2_root_family=_swap_family(self.p1_root_family),
            landing_branches=tuple(
                branch.swap_sides() for branch in self.landing_branches
            ),
        )


def canonical_json_bytes(observation: ProjectedStateObservationV1) -> bytes:
    """正式DTOだけを決定論的JSONバイト列へ変換する。"""
    if not isinstance(observation, ProjectedStateObservationV1):
        raise ProjectedStateContractError("正規JSONの入力DTOが不正です")
    text = json.dumps(
        observation.to_dict(), ensure_ascii=False, allow_nan=False,
        sort_keys=True, separators=(",", ":"),
    )
    return (text + "\n").encode("utf-8")


def _validate_identity_and_hashes(value: ProjectedStateObservationV1) -> None:
    for field, label in (
        (value.observation_id, "observation ID"),
        (value.source_video_id, "source video ID"),
        (value.build_id, "build ID"),
        (value.attempt_id, "attempt ID"),
    ):
        _require_text(field, label)
    for field, label in (
        (value.game_idx, "game index"),
        (value.boundary_segment, "boundary segment"),
        (value.through_event_seq, "through event sequence"),
    ):
        _require_nonnegative_int(field, label)
    for field, label in (
        (value.causal_cutoff_digest, "causal cutoff digest"),
        (value.schema_hash, "schema hash"),
        (value.adapter_hash, "adapter hash"),
        (value.physics_hash, "physics hash"),
        (value.recognition_hash, "recognition hash"),
        (value.input_manifest_hash, "input manifest hash"),
    ):
        _require_sha256(field, label)


def _validate_gate(value: ProjectedStateObservationV1) -> None:
    if not isinstance(value.gate_status, str) or value.gate_status not in GATE_STATUSES:
        raise ProjectedStateContractError("gate statusが正式値ではありません")
    if value.recipient is not None and value.recipient not in SIDES:
        raise ProjectedStateContractError("recipientがp1/p2ではありません")
    if value.active_chain_side is not None and value.active_chain_side not in SIDES:
        raise ProjectedStateContractError("active chain sideがp1/p2ではありません")
    _validate_reason_codes(value.reason_codes)
    _validate_status_reasons(value.gate_status, value.reason_codes)
    if type(value.landing_columns_ambiguous) is not bool:
        raise ProjectedStateContractError("landing ambiguity maskはbool必須です")
    ambiguity_reason = ProjectedStateReason.LANDING_COLUMNS_AMBIGUOUS in value.reason_codes
    if ambiguity_reason != value.landing_columns_ambiguous:
        raise ProjectedStateContractError("landing ambiguity maskとreasonが一致しません")
    if value.gate_status == "guaranteed":
        if value.recipient not in SIDES or value.active_chain_side not in SIDES:
            raise ProjectedStateContractError("guaranteedにside情報がありません")
        if value.recipient != value.active_chain_side:
            raise ProjectedStateContractError("guaranteedのrecipientがactive chain側ではありません")
    elif not value.reason_codes:
        raise ProjectedStateContractError("非guaranteedにはreason codeが必要です")


def _validate_quantities(value: ProjectedStateObservationV1) -> None:
    quantities = (
        value.prefire_incoming_amount, value.signed_projected_balance,
        value.first_drop_amount, value.leftover_after_first_drop,
    )
    if any(not isinstance(item, IntegerQuantity) for item in quantities):
        raise ProjectedStateContractError("quantityの型が不正です")
    _require_nonnegative_quantity(value.prefire_incoming_amount, "prefire incoming")
    _require_nonnegative_quantity(value.first_drop_amount, "first drop")
    _require_nonnegative_quantity(value.leftover_after_first_drop, "leftover")
    _validate_balance_breakdown(value)
    if value.gate_status == "guaranteed":
        incoming = value.prefire_incoming_amount
        if not incoming.present or int(incoming.value) <= 0:
            raise ProjectedStateContractError("guaranteedには正のprefire incomingが必要です")
    elif value.gate_status == "not_applicable":
        _validate_not_applicable_facts(value)


def _validate_not_applicable_facts(value: ProjectedStateObservationV1) -> None:
    if value.active_chain_side not in SIDES:
        raise ProjectedStateContractError("active chain側欠測はuntrustedで記録します")
    checks = {
        ProjectedStateReason.PREFIRE_INCOMING_ZERO: (
            value.prefire_incoming_amount.present
            and value.prefire_incoming_amount.value == 0
        ),
        ProjectedStateReason.PROJECTED_BALANCE_ZERO: (
            value.signed_projected_balance.present
            and value.signed_projected_balance.value == 0
        ),
        ProjectedStateReason.RECIPIENT_CHANGED_FROM_PREFIRE_SIDE: (
            value.prefire_incoming_amount.present
            and int(value.prefire_incoming_amount.value) > 0
            and value.recipient in SIDES
            and value.recipient != value.active_chain_side
        ),
        ProjectedStateReason.RECIPIENT_NOT_ACTIVE_CHAIN_SIDE: (
            value.prefire_incoming_amount.present
            and value.prefire_incoming_amount.value == 0
            and value.recipient in SIDES
            and value.recipient != value.active_chain_side
        ),
    }
    if any(not checks[reason] for reason in value.reason_codes):
        raise ProjectedStateContractError("not-applicable reasonと物理量・方向が一致しません")


def _validate_balance_breakdown(value: ProjectedStateObservationV1) -> None:
    balance = value.signed_projected_balance
    first_drop = value.first_drop_amount
    leftover = value.leftover_after_first_drop
    if not balance.present:
        if first_drop.present or leftover.present or value.recipient is not None:
            raise ProjectedStateContractError("欠損balanceから着弾内訳は確定できません")
        return
    signed = int(balance.value)
    expected_recipient = None if signed == 0 else ("p2" if signed > 0 else "p1")
    if value.recipient != expected_recipient:
        raise ProjectedStateContractError("balance符号とrecipientが一致しません")
    if not first_drop.present or not leftover.present:
        raise ProjectedStateContractError("既知balanceの着弾内訳が欠損しています")
    expected_first = min(abs(signed), OJAMA_MAX_DROP_PER_TURN)
    if first_drop.value != expected_first or leftover.value != abs(signed) - expected_first:
        raise ProjectedStateContractError("balanceと初回着弾内訳が一致しません")
    if value.gate_status == "guaranteed" and signed == 0:
        raise ProjectedStateContractError("guaranteedのprojected residualは非0必須です")


def _validate_landing_branches(value: ProjectedStateObservationV1) -> None:
    branches = value.landing_branches
    if type(branches) is not tuple or len(branches) > MAX_LANDING_BRANCHES:
        raise ProjectedStateContractError("着弾候補数または格納型が不正です")
    if value.gate_status != "guaranteed":
        if branches or value.landing_columns_ambiguous:
            raise ProjectedStateContractError("非guaranteedは着弾候補を公開できません")
        return
    first_drop = int(value.first_drop_amount.value)
    remainder = first_drop % BOARD_COLS
    expected_columns = tuple(combinations(range(BOARD_COLS), remainder))
    expected_ambiguity = remainder != 0
    if value.landing_columns_ambiguous != expected_ambiguity:
        raise ProjectedStateContractError("端数とlanding ambiguity maskが一致しません")
    if any(not isinstance(branch, LandingBranch) for branch in branches):
        raise ProjectedStateContractError("着弾候補の型が不正です")
    actual_columns = tuple(branch.remainder_columns for branch in branches)
    if actual_columns != expected_columns:
        raise ProjectedStateContractError("端数列候補が全組合せの決定論順序ではありません")
    expected_count = 1 if remainder == 0 else comb(BOARD_COLS, remainder)
    if len(branches) != expected_count:
        raise ProjectedStateContractError("端数列候補数がC(6,r)と一致しません")
    _validate_unchanged_opponent(value)


def _validate_unchanged_opponent(value: ProjectedStateObservationV1) -> None:
    """初回着弾は受け手だけを変えるため、相手盤面の混入を拒否する。"""
    if value.recipient not in SIDES:
        raise ProjectedStateContractError("guaranteedのrecipientが不正です")
    other = _opponent(value.recipient)
    expected = getattr(value.post_chain_state, other)
    if any(getattr(branch.boards, other) != expected for branch in value.landing_branches):
        raise ProjectedStateContractError("着弾候補が非recipient盤面を変更しています")


def _validate_board_states(value: ProjectedStateObservationV1) -> None:
    if not isinstance(value.current_state, DualBoardState):
        raise ProjectedStateContractError("current stateの型が不正です")
    if not isinstance(value.post_chain_state, DualBoardState):
        raise ProjectedStateContractError("post-chain stateの型が不正です")
    current_missing = not value.current_state.p1.present or not value.current_state.p2.present
    post_missing = not value.post_chain_state.p1.present or not value.post_chain_state.p2.present
    _validate_missing_board_reason(
        value, current_missing, ProjectedStateReason.CURRENT_BOARD_MISSING,
        "current state",
    )
    _validate_missing_board_reason(
        value, post_missing, ProjectedStateReason.POST_CHAIN_BOARD_MISSING,
        "post-chain state",
    )
    if value.gate_status != "guaranteed":
        _validate_root_families(value)
        return
    if any(not branch.boards.p1.present or not branch.boards.p2.present
           for branch in value.landing_branches if isinstance(branch, LandingBranch)):
        raise ProjectedStateContractError("guaranteed着弾候補は両側盤面必須です")
    _validate_root_families(value)


def _validate_root_families(value: ProjectedStateObservationV1) -> None:
    """family参照のslot・試合・guaranteed根拠を検証する。"""
    for side, family in (("p1", value.p1_root_family), ("p2", value.p2_root_family)):
        if family is None:
            continue
        if not isinstance(family, RootFamilyReference):
            raise ProjectedStateContractError("root-family参照の型が不正です")
        if family.side != side or family.game_idx != value.game_idx:
            raise ProjectedStateContractError("root-family参照のslotまたは試合が不一致です")
    if value.gate_status == "guaranteed":
        active = getattr(value, f"{value.active_chain_side}_root_family")
        if active is None:
            raise ProjectedStateContractError("guaranteedのactive root-familyがありません")


def _validate_missing_board_reason(
    value: ProjectedStateObservationV1, missing: bool,
    reason: ProjectedStateReason, label: str,
) -> None:
    has_reason = reason in value.reason_codes
    if missing != has_reason:
        raise ProjectedStateContractError(f"{label}欠測とreason codeが一致しません")
    if missing and value.gate_status != "untrusted":
        raise ProjectedStateContractError(f"{label}欠測はuntrusted必須です")


def _validate_grid(grid: ImmutableGrid | None) -> None:
    if type(grid) is not tuple or len(grid) != BOARD_ROWS:
        raise ProjectedStateContractError("盤面はtupleの13行必須です")
    for row in grid:
        if type(row) is not tuple or len(row) != BOARD_COLS:
            raise ProjectedStateContractError("盤面は各行tupleの6列必須です")
        if any(type(cell) is not int or cell not in VALID_COLORS for cell in row):
            raise ProjectedStateContractError("盤面cellが正式色値ではありません")


def _validate_unknown_mask(
    grid: ImmutableGrid | None, mask: ImmutableMask | None,
) -> None:
    if type(mask) is not tuple or len(mask) != BOARD_ROWS:
        raise ProjectedStateContractError("unknown maskはtupleの13行必須です")
    for grid_row, mask_row in zip(grid, mask):
        if type(mask_row) is not tuple or len(mask_row) != BOARD_COLS:
            raise ProjectedStateContractError("unknown maskは各行tupleの6列必須です")
        if any(type(cell) is not bool for cell in mask_row):
            raise ProjectedStateContractError("unknown mask cellはbool必須です")
        if any(flag != (cell == COLOR_UNKNOWN) for cell, flag in zip(grid_row, mask_row)):
            raise ProjectedStateContractError("gridとunknown maskが一致しません")


def _validate_remainder_columns(columns: tuple[int, ...]) -> None:
    if type(columns) is not tuple:
        raise ProjectedStateContractError("端数列はtuple必須です")
    if any(type(column) is not int or not 0 <= column < BOARD_COLS for column in columns):
        raise ProjectedStateContractError("端数列が0..5の範囲外です")
    if tuple(sorted(set(columns))) != columns:
        raise ProjectedStateContractError("端数列は昇順かつ重複なし必須です")


def _validate_reason_codes(reasons: tuple[ProjectedStateReason, ...]) -> None:
    if type(reasons) is not tuple:
        raise ProjectedStateContractError("reason codesはtuple必須です")
    if any(not isinstance(reason, ProjectedStateReason) for reason in reasons):
        raise ProjectedStateContractError("reason codeがv1固定Enumではありません")
    if tuple(sorted(set(reasons), key=str)) != reasons:
        raise ProjectedStateContractError("reason codesは昇順かつ重複なし必須です")


def _validate_status_reasons(
    status: str, reasons: tuple[ProjectedStateReason, ...],
) -> None:
    allowed = {
        "guaranteed": frozenset({ProjectedStateReason.LANDING_COLUMNS_AMBIGUOUS}),
        "not_applicable": NOT_APPLICABLE_REASONS,
        "untrusted": UNTRUSTED_REASONS,
    }[status]
    if any(reason not in allowed for reason in reasons):
        raise ProjectedStateContractError("gate statusとreason code分類が一致しません")


def _require_nonnegative_quantity(value: IntegerQuantity, label: str) -> None:
    if value.present and int(value.value) < 0:
        raise ProjectedStateContractError(f"{label}は非負整数必須です")


def _require_nonnegative_int(value: object, label: str) -> None:
    if type(value) is not int or value < 0:
        raise ProjectedStateContractError(f"{label}は非負整数必須です")


def _require_sha256(value: object, label: str) -> None:
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise ProjectedStateContractError(f"{label}は小文字64桁SHA-256必須です")


def _require_text(value: object, label: str) -> None:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ProjectedStateContractError(f"{label}は空白なしの非空文字列必須です")
    if any(ord(char) < 32 or ord(char) == 127 for char in value):
        raise ProjectedStateContractError(f"{label}に制御文字は使えません")


def _opponent(side: str) -> str:
    return "p2" if side == "p1" else "p1"


def _swap_family(
    family: RootFamilyReference | None,
) -> RootFamilyReference | None:
    return None if family is None else family.swap_side()


__all__ = [
    "BOARD_PROVENANCES", "DualBoardState", "ImmutableGrid", "ImmutableMask",
    "IntegerQuantity", "LandingBranch", "MAX_LANDING_BRANCHES",
    "PROJECTED_STATE_SCHEMA_VERSION", "ProjectedBoardState",
    "ProjectedStateContractError", "ProjectedStateObservationV1",
    "ProjectedStateReason", "RootFamilyReference",
    "canonical_json_bytes",
]
