"""決着・試合外判定の既存観測をspoolへ複写する（再検出しない）。"""
from __future__ import annotations

from typing import Any


def archive_terminal_observation(pipe: Any, notice: Any, ready: bool) -> None:
    """任意の重い盤面監査OFFでも、配布の常設spoolへ小さい診断行を残す。"""
    owner = getattr(pipe, 'pipe', pipe)
    sink = getattr(owner, '_live_terminal_observations', None)
    if sink is not None:
        b3 = getattr(notice, 'b3', None)
        extra = ({'warning_async': b3[0]['warning_async']}
                 if b3 is not None and 'warning_async' in b3[0] else {})
        sink.append(dict(frame=notice.frame, t_sec=notice.t_sec,
            captured_at=notice.captured_at, recognized_at=notice.recognized_at,
            ready=ready, **recognition_diagnostics(pipe), **extra))


def recognition_diagnostics(pipe: Any) -> dict[str, Any]:
    """高速NCCは近似値・下界の場合もあるので、経路と回数を併記する。"""
    owner = getattr(pipe, 'pipe', pipe)
    inputs = getattr(owner, '_live_snapshot_inputs', None)
    death = getattr(inputs, 'terminal', None)
    detector = getattr(owner, '_match_end_detector', None)
    result = getattr(detector, 'last_detection', None)
    return dict(
        observed_death=None if death is None else dict(
            scores=list(death.scores), streak=list(death.streak), frames=death.frames,
            fast=death.fast, full_matches=list(death.full_matches), local_hits=list(death.local_hits)),
        match_end=None if result is None else dict(
            score=result.score, detected=result.detected, template_name=result.template_name,
            observed_t_sec=detector.last_detection_t,
            **(dict(ncc=dict(result.ncc), winner=result.winner,
                    ncc_observed_t_sec=dict(result.ncc_observed_t_sec),
                    scanned_roi=result.scanned_roi)
               if getattr(owner, '_enable_match_end_pair', False) else {})),
        match_off=dict(getattr(owner, '_live_match_off_diagnostics', {})))
