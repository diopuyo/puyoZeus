"""元CPU fixtureの明示Noneだけを原trackerへ替える。新人工条件であり同値を主張しない。"""
from __future__ import annotations
import hashlib
from pathlib import Path
import sys
from typing import Any, Iterator

PROJECT = Path(__file__).resolve().parents[3]
SNAPSHOT = PROJECT / '.runtime_snapshots/event_first30_observed_context_v5_2026-08-30/src'
PINS = {'recognition_pipeline.py': '6e945d7584025ae9803e14c9ac079b1a468f483d0beac681a1f0e580cdd10e02',
        'chain_detector.py': '2daf05c9a81165958b07a63ced9fe8a298808e637eae52143388c897fca81cfb'}
NAMES = ('chain_tracker_1p', 'chain_tracker_2p')


def tracker_type(cls: Any) -> Any:
    module = sys.modules[cls.__module__]
    assert module.RecognitionPipeline is cls and Path(module.__file__).resolve() == SNAPSHOT / 'recognition_pipeline.py'
    tracker = module.VideoChainTracker
    assert Path(sys.modules[tracker.__module__].__file__).resolve() == SNAPSHOT / 'chain_detector.py'
    for name, sha in PINS.items():
        assert hashlib.sha256((SNAPSHOT / name).read_bytes()).hexdigest() == sha, 'tracker_source'
    return tracker


def transport(original: Any, evidence: dict[str, Any]) -> Any:
    def real(frozen: Any, monkeypatch: Any) -> Iterator[Any]:
        cls, calls = frozen.RecognitionPipeline, []
        constructor, tracker = cls.__init__, tracker_type(cls)
        def init(pipe: Any, *args: Any, **supplied: Any) -> None:
            assert not calls and all(name in supplied and supplied[name] is None for name in NAMES), 'tracker_explicit_none_required'
            values = {name: tracker() for name in NAMES}
            calls.append(dict(source_sha=PINS, original_none=True))
            constructor(pipe, *args, **(supplied | values))
            assert all(type(getattr(pipe, '_' + name)) is tracker for name in NAMES)
        cls.__init__ = init
        try:
            yield from original(frozen, monkeypatch)
        finally:
            cls.__init__ = constructor
            evidence.update(tracker_constructor_calls=len(calls), tracker_sources=calls,
                tracker_constructor_restored=cls.__init__ is constructor, prefix_equivalence_claimed=False)
    return real


def observe(pipe: Any) -> dict[str, Any]:
    tracker = tracker_type(type(pipe))
    assert all(type(getattr(pipe, '_' + name)) is tracker for name in NAMES)
    return dict(tracker_class_verified=True,
        active_origin_1p=pipe._active_chain_1p is not None, active_origin_2p=pipe._active_chain_2p is not None,
        match_active_started_time=pipe._match_active_started_time,
        original_tracker_may_be_recreated_on_reset=True, quality_gate_clear=False)
