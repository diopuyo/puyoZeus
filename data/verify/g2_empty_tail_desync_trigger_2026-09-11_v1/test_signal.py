"""不整合2観測の正常/非quiet/別scope/単発/重複対照。信号から会計認可を作らない。"""
from __future__ import annotations
from dataclasses import replace
from typing import Any
import pytest
import desync_signal as S
import saved_signal as R


def pair() -> Any:
    first = S.Fact(108,('source','run',1,'1P',0,'segment'),100,(3,4),(4,5),(1,1),
        (1,104,104,'actual-candidate'),(106,108),True)
    return first,replace(first,frame=110,quiet_frames=(108,110))


def test_saved_signal_before_actual_failure() -> None:
    result = R.evaluate()
    assert [[r['frame'] for r in s] for s in result['signals']]==[[35156,35158]]
    assert result['saved_updates']==57 and not result['reset_permission']


def test_two_observations_once() -> None:
    first,second = pair()
    matcher = S.Matcher()
    assert matcher.push(first) is None and matcher.push(second)==(first,second)
    assert matcher.push(replace(second,frame=112,quiet_frames=(110,112))) is None


@pytest.mark.parametrize('change',[dict(native_quiet=False),dict(quiet_frames=None),
    dict(quiet_frames=[108,110]),dict(reason='await_motion'),dict(native_appended=True),
    dict(pair=(3,4)),dict(pair=(4,1)),dict(dnext=(2,2)),dict(baseline_frame=102),
    dict(scope=('source','run',1,'1P',1,'segment')),dict(scope=('source','run',2,'1P',0,'segment')),
    dict(candidate=(2,104,104,'actual-candidate')),dict(candidate=None),
    dict(frame=112,quiet_frames=(110,112)),dict(pair=(True,5))])
def test_no_signal(change: Any) -> None:
    first,second = pair()
    assert not S.joined(first,replace(second,**change))


def test_gap_clears_previous() -> None:
    first,second = pair()
    matcher = S.Matcher()
    assert matcher.push(first) is None and matcher.push(None) is None and matcher.push(second) is None
