"""既存の実台帳所有fixtureで終了後clock不一致を再現。動画状態復元ではない。"""
from contextlib import ExitStack
from pathlib import Path
from typing import Any
import sys
import pytest

VERIFY = Path(__file__).resolve().parent.parent
for name in ('g2_async_projected_evaluation_2026-09-13_v1',
             'g2_prefix_lane_integration_2026-09-13_v1',
             'g2_arrival_backlog_2026-09-13_v1'):
    sys.path.insert(0, str(VERIFY / name))
import test_prefix_live_reader as F

source = F.source
STRIDE = 2


def test_frozen_clock_rejects_next_completed_frame(source: Any, tmp_path: Path) -> None:
    with ExitStack() as stack:
        live, mode, session, step, lane = F.fixture(source, tmp_path, stack)
        assert F.R.owner(live, F.A, mode, session, step) is lane
        frozen = mode.arrival_ledger
        following = step | dict(frame_idx=step['frame_idx'] + STRIDE)
        with pytest.raises(ValueError, match='projection_live_finished_clock'):
            F.R.owner(live, F.A, mode, session, following)
        assert mode.arrival_ledger is frozen
