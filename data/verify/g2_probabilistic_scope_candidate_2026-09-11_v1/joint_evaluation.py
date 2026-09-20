"""連鎖後の候補相関を壊さず両盤面を抽選する。実モデル/採録への接続は別。"""
from __future__ import annotations
from dataclasses import dataclass,field
import math
from typing import Any,Callable
import numpy as np
import belief as B

DEFAULT_SAMPLES=256


@dataclass(frozen=True)
class Result:
    frame: int
    probability_p1: float
    score_variance: float|None
    mc_standard_error: float|None
    used_samples: int
    world_counts: tuple[int,int]
    seed: int
    within_side_joint_support_preserved: bool=field(default=True,init=False)
    between_sides_independence_assumed: bool=field(default=True,init=False)
    sampling_error_only: bool=field(default=True,init=False)
    provisional: bool=field(default=True,init=False)
    source_producer_connected: bool=field(default=False,init=False)
    integer_current_permission: bool=field(default=False,init=False)
    accounting_permission: bool=field(default=False,init=False)
    quality_gate_clear: bool=field(default=False,init=False)


def inputs(values: tuple[B.Belief,B.Belief],frame: int,states: tuple[str,str],
           observed: tuple[B.Board,B.Board]) -> None:
    B.require(type(values) is tuple and len(values)==2,'joint_side_count')
    B.require(type(states) is tuple and states==('STABLE','STABLE'),'joint_both_STABLE')
    B.require(type(observed) is tuple and len(observed)==2,'joint_observations')
    for side,value,board in zip(('1P','2P'),values,observed,strict=True):
        B.validate(value)
        B.require(value.scope[-1]==side,'joint_side_identity')
        B.require(type(frame) is int and value.frame<=frame<=value.deadline,'joint_clock')
        visible=B.grid(board)[B.HIDDEN_ROWS:]
        B.require(all(w.grid[B.HIDDEN_ROWS:]==visible for w in value.worlds),'joint_current_visible_mismatch')
    B.require(all(values[0].scope[i]==values[1].scope[i] for i in (0,1,3)),'joint_source_run_pipe')


def draw(values: tuple[B.Belief,B.Belief],count: int,seed: int) -> np.ndarray:
    rng=np.random.default_rng(seed)
    choices=[rng.choice(len(value.worlds),size=count,p=[w.weight for w in value.worlds]) for value in values]
    batch=np.asarray([[values[s].worlds[choices[s][i]].grid for s in range(2)] for i in range(count)],dtype=np.uint8)
    batch.setflags(write=False)
    return batch


def evaluate(values: tuple[B.Belief,B.Belief],frame: int,states: tuple[str,str],
             observed: tuple[B.Board,B.Board],scorer: Callable[[np.ndarray],np.ndarray],*,
             sample_count: int=DEFAULT_SAMPLES,seed: int=0) -> Result:
    inputs(values,frame,states,observed)
    B.require(type(sample_count) is int and sample_count>0 and type(seed) is int and seed>=0,'joint_sampling')
    B.require(callable(scorer),'joint_scorer')
    counts=tuple(len(v.worlds) for v in values)
    certain=counts==(1,1)
    used=1 if certain else sample_count
    scores=scorer(draw(values,used,seed))
    B.require(type(scores) is np.ndarray and scores.shape==(used,) and scores.dtype.kind in 'fiu','joint_scores_shape')
    B.require(bool(np.isfinite(scores).all()) and bool(((scores>=0)&(scores<=1)).all()),'joint_scores_range')
    variance=0.0 if certain else (float(np.var(scores,ddof=1)) if used>1 else None)
    error=math.sqrt(variance/used) if variance is not None else None
    return Result(frame,float(np.mean(scores)),variance,error,used,counts,seed)
