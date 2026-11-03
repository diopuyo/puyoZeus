"""助走を重複計上せず、異なる速度の台へ区間を割り当てる。"""
import pytest

from scripts.image_free_segment_plan import assign, partition, plan


def test_partition_covers_video_with_warmup() -> None:
    video = dict(id='v', path='video', interval_evidence='receipt')
    segments = partition(video, [30., 60., 90.], 30., 3600)
    assert sum(s['end_frame']-s['owned_start'] for s in segments) == 3600
    assert segments[0]['owned_start'] == 0 and segments[-1]['end_frame'] == 3600
    assert all(s['start_frame'] == s['owned_start']-150 for s in segments[1:])


def test_faster_host_receives_proportional_work() -> None:
    segments = [dict(id=str(i), start_frame=0, end_frame=100) for i in range(12)]
    assigned = assign(segments, dict(pc1=1., pc2=3.))
    assert sum(s['host'] == 'pc1' for s in assigned) == 3
    assert sum(s['host'] == 'pc2' for s in assigned) == 9


@pytest.mark.parametrize('rates', [dict(pc1=0., pc2=1.), dict(pc1=1.)])
def test_missing_or_zero_rate_fails(rates: dict) -> None:
    with pytest.raises(ValueError):
        assign([], rates)


def test_repeated_nearby_boundaries_are_coalesced() -> None:
    segments = partition(dict(id='v', interval_evidence='receipt'), [30, 30, 31], 30, 1800)
    assert len(segments) == 2
