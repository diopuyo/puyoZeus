"""既存評価器の値だけを公開するC案の読み取り専用射影。"""
from __future__ import annotations

from dataclasses import asdict
from typing import Any

from .contracts import DisplayLayers

PROJECTED_SOURCES = frozenset({'S3_landing', 'unavoidable_death'})


def evaluation_layers(overlay: Any, displayed: float, mc_included: bool = False) -> dict[str, Any]:
    """破棄理由は評価器が明示したものだけ載せ、表示側で推測・判定しない。"""
    tracker = overlay.tracker
    projected = tracker.source in PROJECTED_SOURCES or mc_included
    current = getattr(tracker, '_static_probability', None)
    predicted = displayed if mc_included else tracker.probability if projected else None
    reason = getattr(tracker, 'prediction_discard_reason', None)
    layers = DisplayLayers(current, predicted,
        displayed if tracker.probability is not None or mc_included else None, projected, reason)
    return asdict(layers)
