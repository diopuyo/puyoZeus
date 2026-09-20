"""v5の原期限を保ち、NEXT欠測の同call観測票だけを私有経路へ接続する。"""
from __future__ import annotations
import importlib.util
import inspect
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent/'g2_basis_observation_runtime_2026-09-12_v5'
POLICY = ROOT.parent/'g2_exit_next_missing_context_2026-09-12_v1'
SPEC = importlib.util.spec_from_file_location('_g2_exit_next_base_adapter',BASE/'owned_adapter.py')
A = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(A)
PRIOR,START,SIDE = A.PRIOR,A.START,A.SIDE
protect = A.protect
CORE_SUFFIX = 'g2_live_probability_context_2026-09-12_v1/context.py'


def sources() -> tuple[Path,...]:
    names = ('next_vote_policy.py','prior_adapter.py','exit_next_evidence.py','live_connection.py')
    return tuple(sorted(set(A.sources())|{ROOT/'owned_adapter.py',BASE/'owned_adapter.py'}
                        |{POLICY/name for name in names}))


def vote_module(parent: type) -> Any:
    functions = [cls.__dict__['perform'] for cls in parent.__mro__ if 'perform' in cls.__dict__
                 and cls.__dict__['perform'].__code__.co_filename.endswith(CORE_SUFFIX)]
    A.C.require(len(functions) == 1,'original_core_count')
    return inspect.getclosurevars(functions[0]).nonlocals['deps'].votes()


def modules(owner: Any, votes: Any) -> Any:
    load = owner.bootstrap().load
    policy = load('_g2_exit_next_policy',POLICY/'next_vote_policy.py',{'prior_votes_v2':votes.V})
    adapter = load('_g2_exit_next_prior',POLICY/'prior_adapter.py',{'next_vote_policy':policy})
    evidence = load('_g2_exit_next_evidence',POLICY/'exit_next_evidence.py',{'next_vote_policy':policy})
    return load('_g2_exit_next_live',POLICY/'live_connection.py',
                {'prior_adapter':adapter,'exit_next_evidence':evidence})


def configured(stack: Any) -> Any:
    selected = A.configured(stack)
    owner = sys.modules[A.A.OWNED_ALIAS]
    original = owner.build
    owned: dict[Any,Any] = {}
    def build(parent: type, *, end_frame: int) -> type:
        A.A.require_owned(owner)
        A.C.require(owner.build is build,'build_owner')
        base = original(parent,end_frame=end_frame)
        votes = vote_module(base)
        pipeline = sys.modules['src.recognition_pipeline']
        connection = modules(owner,votes)
        connection.install(stack,votes,pipeline.RecognitionPipeline.update,
                           pipeline.NextSlideDetector,A.A.replace_owned,owned)
        return base
    A.A.replace_owned(stack,owner,'build',build)
    return selected
