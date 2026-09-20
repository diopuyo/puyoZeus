"""元checkpointとheldout契約の接続・拒否・namespace非干渉を検査。"""
from __future__ import annotations

import json
import sys
from typing import Any

import pytest
from scripts import g3_source_models as M


def source_sha(source: str) -> str:
    return M.one(json.loads(M.PREREG.read_text())['sources'], 'source_video_id', source)[
        'source_video_sha_expected_from_manifest']


@pytest.mark.parametrize('source', list(M.SOURCE_FOLDS))
def test_actual_source_membership(source: str) -> None:
    assert M.source_contract(source, source_sha(source))['fold'] == M.SOURCE_FOLDS[source]


def test_unknown_source_and_wrong_sha() -> None:
    with pytest.raises(ValueError, match='unknown_source'):
        M.source_contract('video_40', '0' * 64)
    with pytest.raises(ValueError, match='source_sha'):
        M.source_contract('video_38', '0' * 64)


def test_two_routes_and_exception_cleanup() -> None:
    before = set(sys.modules)
    with M.source_loader('video_38', source_sha('video_38')) as first:
        original = dict(first.SPECS)
        with pytest.raises(LookupError, match='planned'):
            with M.source_loader('video_39', source_sha('video_39')) as second:
                assert first.FOLD == 6 and second.FOLD == 1
                assert first.TrainedMember is not second.TrainedMember
                assert first.SPECS == original
                raise LookupError('planned')
        assert first.SPECS == original and sys.modules[first.__name__] is first
    assert not [key for key in set(sys.modules) - before if key.startswith('_g3_source_loader_')]


@pytest.mark.parametrize('source', ['video_39', 'video_c74', 'video_c80'])
def test_actual_model_load_and_foreign_source_rejection(source: str) -> None:
    with M.source_loader(source, source_sha(source)) as loader:
        members = loader.load_members(loader.SOURCE_ID)
        assert len(members) == 3 and all(member.fold == M.SOURCE_FOLDS[source] for member in members)
        with pytest.raises(ValueError, match='source_id'):
            loader.verify_members('sha256:' + source_sha('video_38'), members)
        assert not loader.backend().torch.cuda.is_initialized()
