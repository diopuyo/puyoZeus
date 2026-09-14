"""整数Sとは別の有限確率状態。初期分布以後は盤面候補間の相関を保つ。"""
from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass,field,replace
from itertools import product
import math
import numpy as np
from typing import Any,Callable
from src.board import Board,BOARD_ROWS,BOARD_COLS,HIDDEN_ROWS,COLOR_UNKNOWN
from src.probabilistic_board import ProbabilisticBoard,PROB_COLORS
from frozen_physics import ChainSimulator

MAX_WORLDS = 4096  # 私有CPU診断の資源上限。超過時に候補や確率を捨てない。
SUM_TOLERANCE = 1e-12
PIECE_COLORS = frozenset((1,2,3,4,5))
Grid = tuple[tuple[int,...],...]


def require(ok: bool,reason: str) -> None:
    if not ok: raise ValueError('probabilistic_scope:'+reason)


def grid(board: Board) -> Grid:
    require(type(board) is Board,'board_type')
    require(type(board._grid) is np.ndarray and board._grid.shape==(BOARD_ROWS,BOARD_COLS)
        and board._grid.dtype.kind in 'iu','board_array_type')
    value=tuple(tuple(int(board.get(r,c)) for c in range(BOARD_COLS)) for r in range(BOARD_ROWS))
    require(all(v in (*PROB_COLORS,COLOR_UNKNOWN) for row in value for v in row),'board_color')
    require(all(v in PROB_COLORS for row in value[HIDDEN_ROWS:] for v in row),'visible_unknown')
    return value


def scope_valid(scope: tuple[Any,...]) -> None:
    require(type(scope) is tuple and len(scope)==7,'scope_type')
    require(all(type(v) is str and bool(v) for v in scope[:2])
        and all(type(v) is int and v>=0 for v in scope[2:6])
        and type(scope[6]) is str and scope[6] in ('1P','2P'),'scope_values')


def supported(cells: Grid) -> bool:
    return all(not (cells[r][c]!=0 and cells[r+1][c]==0)
        for r in range(BOARD_ROWS-1) for c in range(BOARD_COLS))


@dataclass(frozen=True)
class World:
    grid: Grid
    weight: float


@dataclass(frozen=True)
class Belief:
    scope: tuple[Any,...]
    frame: int
    deadline: int
    worlds: tuple[World,...]
    tokens: tuple[str,...] = ()
    independent_initial_cells: bool = field(default=True,init=False)
    subsequent_joint_support_preserved: bool = field(default=True,init=False)
    source_producer_connected: bool = field(default=False,init=False)
    landing_reachability_checked: bool = field(default=False,init=False)
    physical_certified: bool = field(default=False,init=False)
    integer_current_permission: bool = field(default=False,init=False)
    accounting_permission: bool = field(default=False,init=False)
    production_permission: bool = field(default=False,init=False)
    quality_gate_clear: bool = field(default=False,init=False)


def validate(value: Belief) -> None:
    require(type(value) is Belief,'belief_type')
    scope_valid(value.scope)
    require(type(value.frame) is int and type(value.deadline) is int and 0<=value.frame<=value.deadline,'state_clock')
    require(type(value.worlds) is tuple and 0<len(value.worlds)<=MAX_WORLDS,'state_support')
    require(type(value.tokens) is tuple and all(type(t) is str and bool(t) for t in value.tokens)
        and len(set(value.tokens))==len(value.tokens),'state_tokens')
    for world in value.worlds:
        require(type(world) is World and type(world.weight) is float and math.isfinite(world.weight)
            and world.weight>0,'world_weight')
        require(type(world.grid) is tuple and len(world.grid)==BOARD_ROWS
            and all(type(row) is tuple and len(row)==BOARD_COLS for row in world.grid),'world_shape')
        require(all(type(c) is int and c in PROB_COLORS for row in world.grid for c in row),'world_color')
        require(supported(world.grid),'world_gravity')
    require(abs(math.fsum(w.weight for w in value.worlds)-1.0)<=SUM_TOLERANCE,'state_mass')


def distribution(cell: Any) -> tuple[tuple[int,float],...]:
    value=cell.probs
    require(type(value) is dict and bool(value),'distribution_missing')
    require(all(type(k) is int and k in PROB_COLORS and type(v) in (int,float)
        and math.isfinite(v) and v>=0 for k,v in value.items()),'distribution_values')
    require(abs(math.fsum(value.values())-1.0)<=SUM_TOLERANCE,'distribution_sum')
    return tuple((k,float(v)) for k,v in sorted(value.items()) if v>0)


def establish(scope: tuple[Any,...],frame: int,deadline: int,confirmed: Board,
              probability: ProbabilisticBoard) -> Belief:
    scope_valid(scope)
    require(type(frame) is int and type(deadline) is int and 0<=frame<deadline,'clock')
    require(type(probability) is ProbabilisticBoard,'probability_type')
    fixed=grid(confirmed)
    hidden=[]
    for r in range(BOARD_ROWS):
        for c in range(BOARD_COLS):
            dist=distribution(probability.cell(r,c))
            if r<HIDDEN_ROWS: hidden.append(dist)
            else: require(dist==((fixed[r][c],1.0),),'visible_probability_mismatch')
    require(math.prod(map(len,hidden))<=MAX_WORLDS,'support_limit')
    worlds=[]
    for values in product(*hidden):
        cells=[list(row) for row in fixed]
        for index,(color,_) in enumerate(values): cells[index//BOARD_COLS][index%BOARD_COLS]=color
        weight=math.prod(p for _,p in values)
        worlds.append(World(tuple(map(tuple,cells)),weight))
    result=Belief(scope,frame,deadline,tuple(worlds))
    validate(result)
    return result


def advance(value: Belief,scope: tuple[Any,...],frame: int,token: str,observed: Board,
            transform: Callable[[Grid],Grid|None]) -> tuple[Belief,float]:
    validate(value)
    require(type(value) is Belief and scope==value.scope,'scope_changed')
    require(type(frame) is int and value.frame<frame<=value.deadline,'deadline_or_clock')
    require(type(token) is str and bool(token) and token not in value.tokens,'token_reused_or_missing')
    expected=grid(observed)
    merged: dict[Grid,list[float]]=defaultdict(list)
    for world in value.worlds:
        result=transform(world.grid)
        if result is not None and result[HIDDEN_ROWS:]==expected[HIDDEN_ROWS:]:
            merged[result].append(world.weight)
    require(bool(merged),'observation_has_zero_support')
    require(len(merged)<=MAX_WORLDS,'support_limit')
    weights={key:math.fsum(parts) for key,parts in merged.items()}
    mass=math.fsum(weights.values())
    require(math.isfinite(mass) and mass>0,'posterior_mass')
    worlds=tuple(World(key,weight/mass) for key,weight in sorted(weights.items()))
    result=replace(value,frame=frame,worlds=worlds,tokens=value.tokens+(token,))
    validate(result)
    return result,mass


def put(before: Grid,placements: tuple[tuple[int,int,int],...]) -> Grid|None:
    cells=[list(row) for row in before]
    for r,c,color in sorted(placements,reverse=True):
        if cells[r][c]!=0 or (r<BOARD_ROWS-1 and cells[r+1][c]==0): return None
        cells[r][c]=color
    return tuple(map(tuple,cells))


def land(value: Belief,scope: tuple[Any,...],frame: int,token: str,pair: tuple[int,int],
         placements: tuple[tuple[int,int,int],...],observed: Board) -> tuple[Belief,float]:
    require(type(pair) is tuple and len(pair)==2 and all(type(c) is int and c in PIECE_COLORS for c in pair),'pair')
    require(type(placements) is tuple and len(placements)==2,'placements')
    require(all(type(p) is tuple and len(p)==3 and all(type(v) is int for v in p)
        and 0<=p[0]<BOARD_ROWS and 0<=p[1]<BOARD_COLS and p[2] in PIECE_COLORS for p in placements),'placement_cell')
    require(all(p[0]>=HIDDEN_ROWS for p in placements),'hidden_placement_requires_independent_support')
    require(len({p[:2] for p in placements})==2 and sorted(p[2] for p in placements)==sorted(pair),'pair_mismatch')
    (r1,c1,_),(r2,c2,_)=placements
    require((c1==c2 and abs(r1-r2)==1) or abs(c1-c2)==1,'pair_geometry')
    return advance(value,scope,frame,token,observed,lambda before:put(before,placements))


def settle(value: Belief,scope: tuple[Any,...],frame: int,token: str,chain_count: int,
           observed: Board) -> tuple[Belief,float]:
    require(type(chain_count) is int and chain_count>0,'chain_count')
    simulator=ChainSimulator(exclude_hidden_row_from_pop=True)
    def transform(before: Grid) -> Grid|None:
        result=simulator.simulate(Board.from_dict({'grid':before}))
        return grid(result.final_board) if result.chain_count==chain_count else None
    return advance(value,scope,frame,token,observed,transform)


def marginals(value: Belief) -> ProbabilisticBoard:
    validate(value)
    result=ProbabilisticBoard.from_board(Board())
    for r in range(BOARD_ROWS):
        for c in range(BOARD_COLS):
            parts: dict[int,list[float]]=defaultdict(list)
            for world in value.worlds: parts[world.grid[r][c]].append(world.weight)
            result.set_distribution(r,c,{color:math.fsum(weights) for color,weights in parts.items()})
    return result
