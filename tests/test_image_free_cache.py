"""旧キャッシュの同等性・失効・再開契約。"""
from pathlib import Path

from joblib.externals import cloudpickle
import pytest

from scripts.image_free_cache import CachedOld, CHUNK_FRAMES, cache_key, file_sha


class Counter:
    def __init__(self) -> None:
        self.total = 0

    def step(self, frame: int, index: int, fps: float) -> dict:
        self.total += frame
        return dict(total=self.total, index=index, fps=fps)


def collect(cache: CachedOld, start: int = 0, count: int = CHUNK_FRAMES) -> list[dict]:
    return [cache.step(1, i, 60.) for i in range(start, start+count)]


def test_hit_verifies_prefix_and_restores_state(tmp_path: Path) -> None:
    first = CachedOld(Counter(), tmp_path, 'key')
    expected = collect(first, count=2*CHUNK_FRAMES)
    second = CachedOld(Counter(), tmp_path, 'key')
    assert collect(second, count=2*CHUNK_FRAMES) == expected
    assert second.hits == 2
    assert second.computed == 2 * second.verify_frames
    assert second.runtime.total == 2 * CHUNK_FRAMES


def test_prefix_mismatch_recomputes_all_without_resetting_live_state(tmp_path: Path) -> None:
    collect(CachedOld(Counter(), tmp_path, 'key'))
    changed = Counter()
    changed.total = 9
    second = CachedOld(changed, tmp_path, 'key')
    actual = collect(second)
    assert second.rejected == 1 and second.hits == 0
    assert second.computed == CHUNK_FRAMES
    assert actual[-1]['total'] == CHUNK_FRAMES + 9


def test_key_change_never_reuses(tmp_path: Path) -> None:
    collect(CachedOld(Counter(), tmp_path, 'key'))
    second = CachedOld(Counter(), tmp_path, 'changed')
    collect(second)
    assert second.rejected == 1 and second.computed == CHUNK_FRAMES


@pytest.mark.parametrize('hit', [False, True])
def test_mid_chunk_pickle_resume(tmp_path: Path, hit: bool) -> None:
    if hit:
        collect(CachedOld(Counter(), tmp_path, 'key'))
    current = CachedOld(Counter(), tmp_path, 'key')
    count = 13
    values = collect(current, count=count)
    resumed = cloudpickle.loads(cloudpickle.dumps(current))
    values += collect(resumed, start=count, count=CHUNK_FRAMES-count)
    assert values[-1]['total'] == CHUNK_FRAMES
    assert resumed.runtime.total == CHUNK_FRAMES


def test_corrupt_chunk_recomputes(tmp_path: Path) -> None:
    (tmp_path / '000000000.pkl').write_bytes(b'')
    cache = CachedOld(Counter(), tmp_path, 'key')
    collect(cache)
    assert cache.rejected == 1


def test_content_hash_and_identity(tmp_path: Path) -> None:
    path = tmp_path / 'video'
    path.write_bytes(b'a')
    first = file_sha(path)
    path.write_bytes(b'b')
    assert first != file_sha(path)
    assert cache_key(dict(a=1, b=2)) == cache_key(dict(b=2, a=1))


def test_nonconsecutive_frame_fails(tmp_path: Path) -> None:
    cache = CachedOld(Counter(), tmp_path, 'key')
    cache.step(1, 0, 60.)
    with pytest.raises(ValueError, match='不連続'):
        cache.step(1, 2, 60.)


def test_cache_preserves_scene_counts_across_chunks(tmp_path: Path) -> None:
    from scripts.compare_image_free_all import compare
    reports = []
    for _ in range(2):
        old = CachedOld(Counter(), tmp_path, 'key')
        state = dict(counts={}, display_scenes=0, previous_display_difference=False)
        differences = []
        for index in range(2*CHUNK_FRAMES):
            value = old.step(1, index, 60.)
            changed = CHUNK_FRAMES-2 <= index <= CHUNK_FRAMES+2
            values = [dict(**value, displayed_probability=0., display_text=0),
                      dict(**value, displayed_probability=float(changed), display_text=int(changed))]
            diff = compare(values, state, index, 60.)
            if diff is not None:
                differences.append(diff)
        reports.append((state, differences))
    assert reports[0] == reports[1]
    assert reports[1][0]['display_scenes'] == 1
    assert reports[1][0]['display_text_scenes'] == 1


def test_nonzero_checkpoint_start(tmp_path: Path) -> None:
    original = Counter()
    original.total = 7138
    first = CachedOld(original, tmp_path, 'key')
    expected = collect(first, start=7138)
    resumed = Counter()
    resumed.total = 7138
    second = CachedOld(resumed, tmp_path, 'key')
    assert collect(second, start=7138) == expected
    assert second.hits == 1


@pytest.mark.parametrize('change', ['code', 'model', 'video', 'environment'])
def test_real_identity_invalidates_content(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                         change: str) -> None:
    from scripts import image_free_cache as cache
    source = tmp_path/'source'
    (source/'src').mkdir(parents=True)
    code = source/'src/a.py'
    code.write_text('initial')
    asset = tmp_path/'assets'
    (asset/'models').mkdir(parents=True)
    model = asset/'models/model'
    model.write_bytes(b'initial')
    video = tmp_path/'video'
    video.write_bytes(b'initial')
    monkeypatch.setattr(cache.importlib.util, 'find_spec', lambda _: None)
    monkeypatch.setattr(cache, 'package_versions', lambda: {})
    config = dict(asset_root=str(asset), b3_model=str(model), b3_table=str(model))
    first = cache.identity(video, source, config)
    if change == 'environment':
        monkeypatch.setenv('PUYO_TEST_FLAG', 'changed')
    else:
        {'code': code, 'model': model, 'video': video}[change].write_bytes(b'changed')
    assert cache.cache_key(first) != cache.cache_key(cache.identity(video, source, config))
