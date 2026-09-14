"""隠し段を含む着地候補の数学コアの検査。実source接続・G2合格の試験ではない。"""
from __future__ import annotations
from dataclasses import replace
from pathlib import Path
import sys
import pytest

ROOT=Path(__file__).resolve().parent
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))

import hidden_landing as H
import belief as B

SCOPE=('hidden-landing-source','hidden-landing-run',3,123,456,3,'1P')
PAIR=H.PairQualification((4,4),'人手仮定のpair。実NEXT未接続。')
PRIOR=H.uncalibrated_uniform('world内の列挙結果を一様と仮定。実頻度の推定ではない。')
OJAMA=9


def board_from(rows: dict[tuple[int,int],int]) -> B.Board:
    board=B.Board()
    for (r,c),v in rows.items(): board.set(r,c,v)
    return board


def establish(board: B.Board,hidden: dict[int,dict[int,float]]) -> B.Belief:
    pb=B.ProbabilisticBoard.from_board(board)
    for c in range(B.BOARD_COLS):
        pb.set_distribution(0,c,hidden.get(c,{0:1.0}))
    return B.establish(SCOPE,10,30,board,pb)


def two_stacked_columns() -> B.Belief:
    """col0/col1 は row2 まで積み上がり (= 縦置きが row0 に跨る)。col5 の隠し段だけ未確定。"""
    rows={(r,c):OJAMA for c in (0,1) for r in range(2,B.BOARD_ROWS)}
    rows.update({(r,5):OJAMA for r in range(1,B.BOARD_ROWS)})
    rows[(0,5)]=B.COLOR_UNKNOWN
    return establish(board_from(rows),{5:{4:0.3,5:0.7}})


def observe(value: B.Belief,writes: dict[tuple[int,int],int]) -> B.Board:
    grid=[list(row) for row in value.worlds[0].grid]
    for (r,c),v in writes.items(): grid[r][c]=v
    for c in range(B.BOARD_COLS): grid[0][c]=0  # 可視のみ観測する。隠し段は観測に入れない。
    return B.Board.from_dict({'grid':grid})


# --- 1. 原 land は据え置き ---------------------------------------------------

def test_original_land_still_rejects_hidden_placement() -> None:
    value=two_stacked_columns()
    with pytest.raises(ValueError,match='hidden_placement_requires_independent_support'):
        B.land(value,SCOPE,12,'old-path',(4,4),((0,0,4),(1,0,4)),observe(value,{(1,0):4}))
    assert value.tokens==() and len(value.worlds)==2


def test_new_path_does_not_touch_original_module() -> None:
    import inspect
    assert 'hidden_placement_requires_independent_support' in inspect.getsource(B.land)
    assert 'land_candidates' not in inspect.getsource(B)


# --- 2. row0 跨ぎ (可視1cell + 隠し1cell、同色) -------------------------------

def test_row0_crossing_vertical_is_enumerated_by_original_physics() -> None:
    value=two_stacked_columns()
    hypotheses=H.enumerate_hypotheses(value.worlds[0].grid,(4,4))
    crossing=[h for h in hypotheses if h.cells==((0,0),(1,0))]
    assert len(crossing)==1 and crossing[0].orientation=='vertical'
    assert crossing[0].touches_hidden and not crossing[0].fully_hidden
    assert len(hypotheses)==8  # col0/col1 縦2 + 横1 + 空列 縦3 + 横2


def test_visible_and_hidden_same_color_landing_is_accepted() -> None:
    value=two_stacked_columns()
    observed=observe(value,{(1,0):4})
    following,report=H.land_candidates(value,SCOPE,12,'cross',PAIR,observed,PRIOR)
    assert report.worlds_after==2 and report.surviving_branches==2
    assert report.hypotheses_total==16 and report.hypotheses_touching_hidden==4
    assert report.posterior_mass==pytest.approx(0.125)
    assert report.surviving_mass_touching_hidden==pytest.approx(0.125)
    assert report.surviving_mass_visible_only==pytest.approx(0.0)
    assert all(w.grid[0][0]==4 and w.grid[1][0]==4 for w in following.worlds)


def test_untouched_hidden_cell_is_not_point_estimated() -> None:
    value=two_stacked_columns()
    assert H.hidden_marginal(value,5)=={4:pytest.approx(0.3),5:pytest.approx(0.7)}
    following,_=H.land_candidates(value,SCOPE,12,'cross',PAIR,observe(value,{(1,0):4}),PRIOR)
    assert H.hidden_marginal(following,5)=={4:pytest.approx(0.3),5:pytest.approx(0.7)}


def test_correlation_between_hidden_cells_is_preserved() -> None:
    value=two_stacked_columns()
    following,_=H.land_candidates(value,SCOPE,12,'cross',PAIR,observe(value,{(1,0):4}),PRIOR)
    pairs={(w.grid[0][0],w.grid[0][5]):w.weight for w in following.worlds}
    assert pairs=={(4,4):pytest.approx(0.3),(4,5):pytest.approx(0.7)}


def test_probability_mass_stays_normalized() -> None:
    value=two_stacked_columns()
    following,report=H.land_candidates(value,SCOPE,12,'cross',PAIR,observe(value,{(1,0):4}),PRIOR)
    assert sum(w.weight for w in following.worlds)==pytest.approx(1.0)
    assert report.prior_mass_total==pytest.approx(1.0)
    assert 0.0<report.posterior_mass<=1.0


# --- 3. 原 2 個可視の正常系は新旧一致 ----------------------------------------

def test_two_visible_cells_match_original_land() -> None:
    value=two_stacked_columns()
    observed=observe(value,{(11,2):4,(12,2):4})
    following,report=H.land_candidates(value,SCOPE,12,'visible',PAIR,observed,PRIOR)
    reference,mass=B.land(value,SCOPE,12,'visible',(4,4),((11,2,4),(12,2,4)),observed)
    assert following.worlds==reference.worlds and mass==pytest.approx(1.0)
    assert report.surviving_mass_visible_only==pytest.approx(report.posterior_mass)
    assert report.surviving_mass_touching_hidden==pytest.approx(0.0)


# --- 4. 観測矛盾 / 旧入力不変 ------------------------------------------------

def test_contradictory_observation_is_zero_support() -> None:
    value=two_stacked_columns()
    with pytest.raises(ValueError,match='observation_has_zero_support'):
        H.land_candidates(value,SCOPE,12,'bad',PAIR,B.Board(),PRIOR)
    assert value.tokens==() and len(value.worlds)==2 and value.frame==10


def test_input_belief_is_unchanged_after_success() -> None:
    value=two_stacked_columns()
    before=(value.frame,value.tokens,value.worlds)
    following,_=H.land_candidates(value,SCOPE,12,'cross',PAIR,observe(value,{(1,0):4}),PRIOR)
    assert (value.frame,value.tokens,value.worlds)==before
    assert following.tokens==('cross',) and following.frame==12


# --- 5. 完全隠し段の枝は落とさず開示する ------------------------------------

def test_fully_hidden_branches_are_kept_and_reported() -> None:
    rows={(r,c):OJAMA for c in range(B.BOARD_COLS) for r in range(1,B.BOARD_ROWS)}
    value=establish(board_from(rows),{})
    following,report=H.land_candidates(value,SCOPE,12,'blind',PAIR,
        B.Board.from_dict({'grid':[list(r) for r in value.worlds[0].grid]}),PRIOR)
    assert report.hypotheses_total==5 and report.hypotheses_fully_hidden==5
    assert report.unobservable_branch_present and report.worlds_after==5
    assert report.posterior_mass==pytest.approx(1.0)
    assert H.hidden_marginal(following,0)=={0:pytest.approx(0.8),4:pytest.approx(0.2)}


# --- 6. 拒否条件 -------------------------------------------------------------

def test_prior_must_support_every_valid_candidate() -> None:
    value=two_stacked_columns()
    pinned=H.PlacementPrior(lambda h: 1.0 if h.cells==((0,0),(1,0)) else 0.0,'点決めprior')
    with pytest.raises(ValueError,match='prior_must_support_every_valid_candidate'):
        H.land_candidates(value,SCOPE,12,'pinned',PAIR,observe(value,{(1,0):4}),pinned)


def test_world_without_physical_landing_is_refused() -> None:
    rows={(r,c):OJAMA for c in range(B.BOARD_COLS) for r in range(B.BOARD_ROWS)}
    value=establish(board_from(rows),{c:{OJAMA:1.0} for c in range(B.BOARD_COLS)})
    with pytest.raises(ValueError,match='world_without_physical_landing'):
        H.land_candidates(value,SCOPE,12,'full',PAIR,
            B.Board.from_dict({'grid':[list(r) for r in value.worlds[0].grid]}),PRIOR)


@pytest.mark.parametrize('case',('scope','expired','same_frame','duplicate','pair','prior','qualified'))
def test_finite_clock_token_and_assumption_rejections(case: str) -> None:
    value=two_stacked_columns()
    scope,frame,token,pair,prior=SCOPE,12,'t',PAIR,PRIOR
    if case=='scope': scope=(*SCOPE[:2],4,*SCOPE[3:])
    elif case=='expired': frame=31
    elif case=='same_frame': frame=10
    elif case=='duplicate': value=replace(value,tokens=('t',))
    elif case=='pair': pair=H.PairQualification((4,0),'空色は資格pairではない')
    elif case=='prior': prior=H.PlacementPrior(lambda _h: 1.0,'')
    elif case=='qualified':
        pair=H.PairQualification((4,4),'偽装')
        object.__setattr__(pair,'real_next_source_connected',True)
    with pytest.raises(ValueError):
        H.land_candidates(value,scope,frame,token,pair,observe(value,{(1,0):4}),prior)


# --- 7. 未接続と凍結の固定 ---------------------------------------------------

def test_all_permissions_stay_false() -> None:
    value=two_stacked_columns()
    following,report=H.land_candidates(value,SCOPE,12,'cross',PAIR,observe(value,{(1,0):4}),PRIOR)
    assert not any((report.placement_prior_calibrated,report.real_source_connected,
        report.next_pair_source_connected,report.j_qualification_connected,
        report.physical_certified,report.integer_current_permission,
        report.accounting_permission,report.production_permission,report.g2_permission))
    assert not any((following.source_producer_connected,following.physical_certified,
        following.integer_current_permission,following.accounting_permission,
        following.production_permission,following.quality_gate_clear))
    assert not PAIR.real_next_source_connected and not PRIOR.calibrated


def test_frozen_sources_are_pinned() -> None:
    import inspect
    import frozen_placement as P
    assert '.runtime_snapshots' in inspect.getfile(P.enumerate_landing_patterns)
    assert '.runtime_snapshots' in inspect.getfile(B.ChainSimulator)
    assert P.SOURCE.is_file() and P.LIVE.is_file()
