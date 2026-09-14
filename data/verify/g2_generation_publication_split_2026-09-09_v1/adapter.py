"""候補破棄と公開解除を分離する負例閉鎖部品。正常復帰policyは接続しない。"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass, replace
import functools
import hashlib
import inspect
import json
from pathlib import Path
import sys
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[2]
MODULE = "scripts.diagnose_video38_c6_pending_commit_shadow_v1"
REASON = "discarded_candidate_current_unverified"
SIDES = ("1P", "2P")
FIXED = {
    "scripts/diagnose_video38_c6_pending_commit_shadow_v1.py": "563fb383e9c0ceb804c3950522a67c77460a69f5e483a0c09c9becdd2f6cbab0",
    "scripts/diagnose_video38_prediction_ledger_shadow_v1.py": "8b77732c66021677da216c2727088cb6f6bcc74496bc9978e8504ff5fa4659a2",
    "src/chain_prediction_ledger_v1.py": "2dcf09dcdad08c62fe0384a34e4d1d6ed85adf5a1ee4f3884c76dde49f5319d7",
    "src/chain_commit_candidate_v1.py": "443b79a896c8e46a268fab8e9f54b0a5238940bf4495706041423ee7de9596c5",
    "tests/test_diagnose_video38_c6_pending_commit_shadow_v1.py": "c6d4b6480f434764ad91626b406c1aea093b04da3cc912d44927ac7fcfbe8320",
}
_INSTALLED: set[type] = set()


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def guards() -> dict[str, str]:
    values = {str(PROJECT / path): digest for path, digest in FIXED.items()}
    if any(sha(Path(path)) != digest for path, digest in values.items()):
        raise RuntimeError("fixed_pending_source_changed")
    return values | {str(ROOT / name): sha(ROOT / name)
        for name in ("adapter.py", "test_adapter.py", "runner.py")}


def encoded(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


@dataclass(frozen=True)
class PublicationHold:
    """既存software候補/旧originの不変copy。Debtや物理game証明ではない。"""
    side: str
    instance_id: int
    candidate_id: str
    origin_reset_epoch: int
    origin_action_revision: int
    discard_reason: str
    candidate_json: str
    origin_snapshot_json: str
    generation_observation_json: str


class State:
    def __init__(self, module: Any) -> None:
        self.module, self.closed = module, False
        self._owners: dict[int, Any] = {}
        self._holds: dict[tuple[int, str], tuple[PublicationHold, ...]] = {}
        self._failures: list[dict[str, Any]] = []
        self._rows: list[dict[str, Any]] = []

    def failure(self, rec: Any, stage: str, exc: BaseException) -> None:
        self._owners[id(rec)] = rec
        self._failures.append({"owner_index": self.owner_index(rec), "stage": stage,
            "exception": type(exc).__name__, "message": str(exc)})

    def owner_index(self, rec: Any) -> int:
        self._owners.setdefault(id(rec), rec)
        return list(self._owners).index(id(rec))

    def snapshot(self, rec: Any, side: str) -> tuple[PublicationHold, ...]:
        if side not in SIDES:
            raise ValueError("publication_side")
        return self._holds.get((id(rec), side), ())

    def preserve(self, rec: Any, side: str, reason: str, observation: Any) -> PublicationHold:
        record = rec.active_candidates[side]
        candidate = record["candidate"]
        if side not in SIDES or candidate.identity.side != side:
            raise ValueError("publication_candidate_side")
        handle = next(h for h in rec.all_handles if h.instance_id == record["instance_id"] and h.side == side)
        snapshot = rec.ledger.snapshot(handle)
        payload = self.module.prediction._json_value
        identity = candidate.identity
        value = PublicationHold(side, handle.instance_id, candidate.candidate_id,
            identity.reset_epoch, identity.action_revision, reason, encoded(payload(candidate)),
            encoded(payload(snapshot)), encoded(observation))
        previous = self.snapshot(rec, side)
        same = next((old for old in previous if old.candidate_id == value.candidate_id), None)
        if same is not None and same != value:
            raise ValueError("publication_hold_identity_collision")
        self.owner_index(rec)
        if same is None:
            self._holds[(id(rec), side)] = previous + (value,)
        return value

    def publication(self, rec: Any, side: str, result: Any) -> Any:
        if any(row["owner_index"] == self.owner_index(rec) for row in self._failures):
            raise RuntimeError("publication_hold_observer_failed")
        held = self.snapshot(rec, side)
        if not held:
            return result
        current = rec.generation_recorder.generation(side)
        epochs = [item.origin_reset_epoch for item in held]
        self._rows.append({"kind": "generation_split_publication_hold", "side": side,
            "owner_index": self.owner_index(rec), "frame_idx": rec.frame, "time_sec": rec.time_sec,
            "origin_reset_epochs": epochs, "current_reset_epoch": current.reset_epoch,
            "current_action_revision": current.action_revision,
            "reset_differs": any(epoch != current.reset_epoch for epoch in epochs),
            "candidate_ids": [item.candidate_id for item in held], "hold_reason": REASON,
            "current_certified": False, "accounting_settled": False, "game_certified": False})
        return replace(result, confirmed_board=None, prob_board=None, board_none_reason=REASON)

    def report(self) -> dict[str, Any]:
        from dataclasses import asdict
        holds = [{"owner_index": list(self._owners).index(owner), "side": side,
                  "origins": [asdict(item) for item in values]}
                 for (owner, side), values in self._holds.items()]
        return json.loads(encoded({"closed": self.closed, "holds": holds, "rows": self._rows,
            "failures": self._failures, "normal_recovery_policy_connected": False,
            "accounting_permission": False, "quality_gate_clear": False}))


def discard_wrapper(original: Any, state: State) -> Any:
    @functools.wraps(original)
    def discard(rec: Any, side: str, reason: str, observation: Any = None) -> Any:
        try:
            state.preserve(rec, side, reason, observation)
            return original(rec, side, reason, observation)
        except BaseException as exc:
            state.failure(rec, "discard", exc)
            raise
    return discard


def publication_wrapper(original: Any, state: State) -> Any:
    @functools.wraps(original)
    def publication(rec: Any, side: str, result: Any) -> Any:
        try:
            returned = original(rec, side, result)
            return state.publication(rec, side, returned)
        except BaseException as exc:
            state.failure(rec, "publication", exc)
            raise
    return publication


@contextlib.contextmanager
def installed(recorder_class: type) -> Iterator[State]:
    guards()
    module = sys.modules.get(MODULE)
    if module is None or recorder_class is not module.PendingCommitRecorder:
        raise ValueError("pending_runtime_class")
    if Path(inspect.getfile(recorder_class)).resolve() != PROJECT / "scripts/diagnose_video38_c6_pending_commit_shadow_v1.py":
        raise ValueError("pending_runtime_path")
    if recorder_class in _INSTALLED:
        raise RuntimeError("publication_split_already_installed")
    state = State(module)
    methods = {name: getattr(recorder_class, name) for name in ("_discard_candidate", "isolate_side_result")}
    _INSTALLED.add(recorder_class)
    try:
        recorder_class._discard_candidate = discard_wrapper(methods["_discard_candidate"], state)
        recorder_class.isolate_side_result = publication_wrapper(methods["isolate_side_result"], state)
        yield state
    finally:
        for name, method in methods.items():
            setattr(recorder_class, name, method)
        _INSTALLED.remove(recorder_class)
        state.closed = True
