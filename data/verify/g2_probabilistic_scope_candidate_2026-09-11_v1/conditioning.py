"""初期独立分布を可視支持条件で条件付ける。消えた確率質量を明示する。"""
from __future__ import annotations
from itertools import product
import math
from typing import Any
import belief as B


def establish_conditioned(scope: tuple[Any,...],frame: int,deadline: int,confirmed: B.Board,
                          probability: B.ProbabilisticBoard) -> tuple[B.Belief,float,int]:
    B.scope_valid(scope)
    B.require(type(frame) is int and type(deadline) is int and 0<=frame<deadline,'clock')
    B.require(type(probability) is B.ProbabilisticBoard,'probability_type')
    fixed=B.grid(confirmed)
    hidden=[]
    for r in range(B.BOARD_ROWS):
        for c in range(B.BOARD_COLS):
            dist=B.distribution(probability.cell(r,c))
            if r<B.HIDDEN_ROWS: hidden.append(dist)
            else: B.require(dist==((fixed[r][c],1.0),),'visible_probability_mismatch')
    count=math.prod(map(len,hidden))
    B.require(count<=B.MAX_WORLDS,'support_limit')
    supported=[]
    for values in product(*hidden):
        cells=[list(row) for row in fixed]
        for index,(color,_) in enumerate(values): cells[index//B.BOARD_COLS][index%B.BOARD_COLS]=color
        cells=tuple(map(tuple,cells))
        if B.supported(cells): supported.append(B.World(cells,math.prod(p for _,p in values)))
    mass=math.fsum(w.weight for w in supported)
    B.require(math.isfinite(mass) and mass>0,'gravity_has_zero_support')
    state=B.Belief(scope,frame,deadline,tuple(B.World(w.grid,w.weight/mass) for w in supported))
    B.validate(state)
    return state,mass,count-len(supported)
