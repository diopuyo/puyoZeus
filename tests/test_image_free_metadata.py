"""番号補正が特徴・欠測・時刻を変えないことを確かめる。"""
from scripts.image_free_metadata import MetadataOffset, leaf_differences, metadata_only


def test_common_lean_offset_keeps_missing_and_original() -> None:
    source = dict(b3_features=dict(lean_row_p1=-1, lean_row_p2=4, game_idx=1,
                  row_key='live|g1|f80|o0', video_id='live', frame_idx=80, orientation=0,
                  score_p1=100, t_sec=1.5))
    actual = MetadataOffset(20, 3).apply(source)
    row = actual['b3_features']
    assert row['lean_row_p1'] == -1 and row['lean_row_p2'] == 24
    assert row['row_key'] == 'live|g4|f80|o0'
    assert source['b3_features']['game_idx'] == 1
    assert metadata_only(leaf_differences(source, actual))


def test_feature_or_timestamp_difference_is_not_metadata() -> None:
    for key in ('score_p1', 'queue_age_p1', 't_sec'):
        assert not metadata_only(leaf_differences({'b3_features': {key: 1}},
                                                 {'b3_features': {key: 2}}))


def test_missing_key_is_recorded_even_for_none() -> None:
    assert leaf_differences({}, {'x': None}) == [dict(column='x', old=None, split=None, missing='old')]


def test_empty_features_stay_empty() -> None:
    assert MetadataOffset(100, 2).apply({'b3_features': None}) == {'b3_features': None}


def test_fix1_game_metadata_and_null_adoption() -> None:
    value = dict(game=dict(index=0, adopted=None, active=False), b3_features=None)
    offset = MetadataOffset(100, 6, 7, 6)
    assert offset.apply(value)['game'] == dict(index=6, adopted=6, active=False)
    value['game']['adopted'] = 1
    assert offset.apply(value)['game']['adopted'] == 8
