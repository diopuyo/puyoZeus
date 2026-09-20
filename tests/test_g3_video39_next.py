"""元実pipeline/元NEXT CPU fixtureを再用し、30fpsのcold履歴を検査する。"""
from __future__ import annotations
from copy import deepcopy
from typing import Any
import pytest
from scripts import g3_video39_next as G
from tests import test_next_enqueue_live_shadow_v1 as T

frozen = T.frozen


def item() -> dict:
    """事前登録sourceを参照。モデル検収済routeの再実行はしない。"""
    prereg = G.P.M.document(G.P.M.PREREG, G.P.M.PREREG_SHA)
    source = G.P.M.one(prereg['sources'], 'source_video_id', G.SOURCE)['source']
    return dict(source=G.SOURCE, source_id='sha256:' + source['source_video_sha256'],
                time_base=[1, 30], stride=1, end_frame_exclusive=11878)


@pytest.fixture
def real39(frozen: Any, monkeypatch: Any) -> Any:
    """元fixtureの人工detectorだけを使い、実constructor/update/元bodyを通す。"""
    def install(stack: Any, collector: Any, rec: Any) -> Any:
        return G.install(stack, collector, rec, item(), 'artificial-video39-cpu')
    monkeypatch.setattr(T.subject, 'install', install)
    yield from T.real.__wrapped__(frozen, monkeypatch)


def test_cold_actual_update(real39: Any) -> None:
    pipe, controller, rec, image, source = real39
    for frame, pair, diff in ((0, (5, 5), 63.0), (1, (2, 3), 1.0), (2, (3, 5), 1.0)):
        source.update(pair=pair, diff=diff)
        pipe.update(frame, frame / 30, image)
        if frame < 2:
            assert not pipe._pending_tsumo_1p and not pipe._pending_tsumo_2p
    assert source['calls'] == T.Counter(next=3, slide=6)
    assert len(rec.rows) == 6 and controller.active is None
    assert all(row['source_id'] == item()['source_id'] and row['run_id'] == 'artificial-video39-cpu'
               and not row['physical_placement_verified'] for row in rec.rows)
    assert controller.instances[id(pipe)].histories['1P'].accepted == (3, 5)
    assert T.subject.BASE.FPS == 60 and T.subject.FIRST_FRAME == 32494 and T.subject.STRIDE == 2


@pytest.mark.parametrize('fault', ['clock', 'gap', 'foreign_fifo'])
def test_reject_bad_input(real39: Any, fault: str) -> None:
    pipe, controller, rec, image, source = real39
    if fault == 'foreign_fifo':
        pipe._pending_tsumo_1p.append((1, 2))
        with pytest.raises(ValueError, match='非空FIFO'):
            pipe.update(0, 0.0, image)
        assert list(pipe._pending_tsumo_1p) == [(1, 2)]
    else:
        pipe.update(0, 0.0, image)
        with pytest.raises(ValueError):
            pipe.update(1 if fault == 'clock' else 2, 1 / 60 if fault == 'clock' else 2 / 30, image)
    assert controller.active is None


def test_contract_and_namespace() -> None:
    original = (T.subject.BASE, T.subject.FIRST_FRAME, T.subject.STRIDE)
    with G.namespace(item()) as module:
        assert module.BASE is not original[0] and module.BASE.FPS == 30
        assert module.BASE.PIPELINE == original[0].PIPELINE
        module._clock(5940, 198.0)
        with pytest.raises(ValueError):
            module._clock(G.END, G.END / 30)
    assert original == (T.subject.BASE, T.subject.FIRST_FRAME, T.subject.STRIDE)
    wrong = deepcopy(item())
    wrong['source_id'] = 'sha256:' + '0' * 64
    with pytest.raises(ValueError):
        with G.namespace(wrong):
            pass


@pytest.mark.parametrize('fails', [False, True])
def test_actual_software_reset(real39: Any, monkeypatch: Any, fails: bool) -> None:
    """元resetの成功/失敗を保存履歴へ反映するが、ゲーム開始証明にはしない。"""
    pipe, controller, rec, image, source = real39
    pipe.update(0, 0.0, image)
    runtime = controller.instances[id(pipe)]
    epoch = runtime.histories['1P'].epoch
    if fails:
        def fail(*args: Any, **kwargs: Any) -> None:
            raise RuntimeError('original_reset_failed')
        monkeypatch.setattr(pipe._sm_1p, 'reset', fail)
        with pytest.raises(RuntimeError, match='original_reset_failed'):
            pipe.reset()
        assert runtime.reset_status == 'failed' and runtime.histories['1P'].epoch == epoch
        with pytest.raises(ValueError, match='失敗reset'):
            pipe.update(1, 1 / 30, image)
    else:
        pipe.reset()
        assert runtime.reset_status == 'software_reset_observed_not_game_proof'
        assert runtime.histories['1P'].epoch == epoch + 1
        pipe.update(1, 1 / 30, image)
        assert runtime.histories['1P'].clock == 1
    assert controller.g3_source_contract['software_reset_is_game_proof'] is False
