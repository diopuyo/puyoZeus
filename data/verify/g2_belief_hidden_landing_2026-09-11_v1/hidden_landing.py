"""隠し段を含む着地候補の数学コア。V2レビュー N-2 の数学的解消だけを担う。

原候補root (g2_probabilistic_scope_candidate_2026-09-11_v1) の joint 世界表現
Belief/World をそのまま使い、その belief.land は一切変更しない
(= 点決め placement の隠し段は今後も hidden_placement_requires_independent_support で拒否)。
本 module は別経路 land_candidates を足すだけで、
実source / 実NEXT / J資格 は未接続、physical/current/accounting/production/G2 の許可は全て False 固定。

数学:
    事前 belief は world (= hidden 段まで具体化した盤面) 上の有限分布。
    world ごとに原 enumerate_landing_patterns で「その world の物理」から着地候補を全列挙し、
    原 enumerate_color_assignments で資格付き pair の色配分を全列挙する。
    各 (world, 仮説) の重み = world.weight * 配置prior(仮説 | world)。
    可視 row のみを観測とし、一致した枝だけ残して正規化する。
    隠し段は「観測が識別した分だけ」動き、識別力が無ければ質量は分かれたまま残る。
    配置prior は較正されていない外部入力であり、学習済み確率とは呼ばない (PlacementPrior.calibrated は常に False)。
"""
from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass,field,replace
import math
from pathlib import Path
import sys
from typing import Any,Callable

ROOT=Path(__file__).resolve().parent
CANDIDATE_ROOT=ROOT.parent/'g2_probabilistic_scope_candidate_2026-09-11_v1'
if str(CANDIDATE_ROOT) not in sys.path: sys.path.insert(0,str(CANDIDATE_ROOT))

import belief as B  # 原候補rootの joint 世界。編集しない。
import frozen_placement as P

HIDDEN_ROWS=B.HIDDEN_ROWS
BOARD_ROWS=B.BOARD_ROWS
BOARD_COLS=B.BOARD_COLS


def require(ok: bool,reason: str) -> None:
    if not ok: raise ValueError('hidden_landing:'+reason)


@dataclass(frozen=True)
class PairQualification:
    """「資格付き pair」の仮定を明示で持たせる入れ物。実NEXT/J資格には未接続。"""
    pair: tuple[int,int]
    assumption: str
    real_next_source_connected: bool = field(default=False,init=False)
    j_qualification_connected: bool = field(default=False,init=False)


@dataclass(frozen=True)
class PlacementPrior:
    """配置仮説の重み。未較正の外部入力であることを型で固定する。"""
    weight: Callable[[Any],float]
    assumption: str
    calibrated: bool = field(default=False,init=False)


@dataclass(frozen=True)
class Hypothesis:
    """原 LandingPattern 1 件 + 原 enumerate_color_assignments の色配分 1 件。"""
    cells: tuple[tuple[int,int],tuple[int,int]]
    orientation: str
    colors: tuple[int,int]

    @property
    def hidden_cells(self) -> int:
        return sum(1 for r,_ in self.cells if r<HIDDEN_ROWS)

    @property
    def touches_hidden(self) -> bool:
        return self.hidden_cells>0

    @property
    def fully_hidden(self) -> bool:
        return self.hidden_cells==len(self.cells)


@dataclass(frozen=True)
class Report:
    """採択前の質量と仮定を開示する。可否判定や許可の付与はしない。"""
    prior_mass_total: float
    posterior_mass: float
    worlds_before: int
    worlds_after: int
    hypotheses_total: int
    hypotheses_touching_hidden: int
    hypotheses_fully_hidden: int
    surviving_branches: int
    surviving_mass_visible_only: float
    surviving_mass_touching_hidden: float
    surviving_mass_fully_hidden: float
    unobservable_branch_present: bool
    pair_assumption: str
    prior_assumption: str
    placement_prior_calibrated: bool = field(default=False,init=False)
    real_source_connected: bool = field(default=False,init=False)
    next_pair_source_connected: bool = field(default=False,init=False)
    j_qualification_connected: bool = field(default=False,init=False)
    physical_certified: bool = field(default=False,init=False)
    integer_current_permission: bool = field(default=False,init=False)
    accounting_permission: bool = field(default=False,init=False)
    production_permission: bool = field(default=False,init=False)
    g2_permission: bool = field(default=False,init=False)


def uncalibrated_uniform(assumption: str) -> PlacementPrior:
    """「world ごとに列挙結果を一様」という明示の仮定。実頻度の推定ではない。"""
    require(type(assumption) is str and bool(assumption),'prior_assumption')
    return PlacementPrior(lambda _hypothesis: 1.0,assumption)


def _board(cells: B.Grid) -> B.Board:
    return B.Board.from_dict({'grid':[list(row) for row in cells]})


def enumerate_hypotheses(world: B.Grid,pair: tuple[int,int]) -> tuple[Hypothesis,...]:
    """その world の物理でありうる着地候補を原APIだけで全列挙する。

    隠し段を含む候補も除外しない。ここで「どれが正しいか」は決めない。
    """
    board=_board(world)
    out: list[Hypothesis]=[]
    for pattern in P.enumerate_landing_patterns(board):
        require(type(pattern) is P.LandingPattern,'pattern_type')
        for colors in P.enumerate_color_assignments(pattern,pair):
            out.append(Hypothesis(tuple(map(tuple,pattern.cells)),pattern.orientation,
                (int(colors[0]),int(colors[1]))))
    require(len({(h.cells,h.colors) for h in out})==len(out),'duplicate_hypothesis')
    return tuple(out)


def apply_hypothesis(world: B.Grid,hypothesis: Hypothesis) -> B.Grid:
    """原 materialize_pattern で書き込む。独自の落下計算はしない。"""
    pattern=P.LandingPattern(cells=hypothesis.cells,orientation=hypothesis.orientation)
    after=B.grid(P.materialize_pattern(_board(world),pattern,hypothesis.colors[0],hypothesis.colors[1]))
    require(B.supported(after),'hypothesis_broke_gravity')
    require(sum(1 for r in range(BOARD_ROWS) for c in range(BOARD_COLS)
        if after[r][c]!=0)==sum(1 for r in range(BOARD_ROWS) for c in range(BOARD_COLS)
        if world[r][c]!=0)+2,'hypothesis_overwrote_existing')
    return after


def _check_pair(qualification: PairQualification) -> tuple[int,int]:
    require(type(qualification) is PairQualification,'qualification_type')
    require(type(qualification.assumption) is str and bool(qualification.assumption),'pair_assumption')
    require(not qualification.real_next_source_connected
        and not qualification.j_qualification_connected,'qualification_must_stay_unconnected')
    pair=qualification.pair
    require(type(pair) is tuple and len(pair)==2
        and all(type(c) is int and c in B.PIECE_COLORS for c in pair),'pair')
    return pair


def land_candidates(value: B.Belief,scope: tuple[Any,...],frame: int,token: str,
                    qualification: PairQualification,observed: B.Board,
                    prior: PlacementPrior) -> tuple[B.Belief,Report]:
    """隠し段を含む着地候補を joint のまま前進させる。placement の点指定は受け取らない。

    呼び出し側が cell を名指しできないので、「情報のない隠し段の点決め」は構造的に不可能。
    観測は可視 row のみ。矛盾すれば zero-support で拒否し、入力 belief は不変のまま返らない。
    """
    B.validate(value)
    require(scope==value.scope,'scope_changed')
    require(type(frame) is int and type(value.frame) is int
        and value.frame<frame<=value.deadline,'deadline_or_clock')
    require(type(token) is str and bool(token) and token not in value.tokens,'token_reused_or_missing')
    require(type(prior) is PlacementPrior and type(prior.assumption) is str
        and bool(prior.assumption) and prior.calibrated is False,'prior_type')
    pair=_check_pair(qualification)
    expected=B.grid(observed)

    merged: dict[B.Grid,list[float]]=defaultdict(list)
    prior_parts: list[float]=[]
    kept: list[tuple[Hypothesis,float]]=[]
    hypotheses_total=touching=fully=0
    for world in value.worlds:
        hypotheses=enumerate_hypotheses(world.grid,pair)
        require(bool(hypotheses),'world_without_physical_landing')
        weights=[]
        for hypothesis in hypotheses:
            w=prior.weight(hypothesis)
            require(type(w) is float and math.isfinite(w)
                and w>0,'prior_must_support_every_valid_candidate')
            weights.append(w)
        total=math.fsum(weights)
        require(math.isfinite(total) and total>0,'prior_mass')
        hypotheses_total+=len(hypotheses)
        touching+=sum(1 for h in hypotheses if h.touches_hidden)
        fully+=sum(1 for h in hypotheses if h.fully_hidden)
        for hypothesis,w in zip(hypotheses,weights):
            share=world.weight*(w/total)
            prior_parts.append(share)
            after=apply_hypothesis(world.grid,hypothesis)
            if after[HIDDEN_ROWS:]==expected[HIDDEN_ROWS:]:
                merged[after].append(share)
                kept.append((hypothesis,share))
    require(bool(merged),'observation_has_zero_support')
    require(len(merged)<=B.MAX_WORLDS,'support_limit')
    weights_by_grid={key:math.fsum(parts) for key,parts in merged.items()}
    mass=math.fsum(weights_by_grid.values())
    require(math.isfinite(mass) and mass>0,'posterior_mass')
    worlds=tuple(B.World(key,w/mass) for key,w in sorted(weights_by_grid.items()))
    following=replace(value,frame=frame,worlds=worlds,tokens=value.tokens+(token,))
    B.validate(following)
    report=Report(
        prior_mass_total=math.fsum(prior_parts),
        posterior_mass=mass,
        worlds_before=len(value.worlds),
        worlds_after=len(worlds),
        hypotheses_total=hypotheses_total,
        hypotheses_touching_hidden=touching,
        hypotheses_fully_hidden=fully,
        surviving_branches=len(kept),
        surviving_mass_visible_only=math.fsum(s for h,s in kept if not h.touches_hidden),
        surviving_mass_touching_hidden=math.fsum(s for h,s in kept if h.touches_hidden),
        surviving_mass_fully_hidden=math.fsum(s for h,s in kept if h.fully_hidden),
        unobservable_branch_present=any(h.fully_hidden for h,_ in kept),
        pair_assumption=qualification.assumption,
        prior_assumption=prior.assumption,
    )
    return following,report


def hidden_marginal(value: B.Belief,col: int) -> dict[int,float]:
    """隠し段 1 cell の周辺分布。点決めしていないことの確認用。"""
    require(type(col) is int and 0<=col<BOARD_COLS,'col')
    parts: dict[int,list[float]]=defaultdict(list)
    for world in value.worlds: parts[world.grid[0][col]].append(world.weight)
    return {color:math.fsum(w) for color,w in sorted(parts.items())}
