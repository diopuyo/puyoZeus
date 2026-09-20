"""実保存channelの拒否と人工正常対照を分離。未保存flagsは明示して有利に仮定。"""
from dataclasses import replace
import pytest
import gate_v2 as G
from saved_channels import records
from src.board import Board
from src.probabilistic_board import ProbabilisticBoard


def observations() -> tuple:
    clock,waits=records()
    rows=[]
    for w in waits:
        c=w['channels']
        grid=lambda b:None if b is None else tuple(map(tuple,b['grid']))
        cells=c['probability'].get('cells') if c['probability'] is not None else None
        dist=lambda indices:None if cells is None else tuple(tuple((k,float(v)) for k,v in cells[r][col] if v>0) for r,col in indices)
        scope=tuple(w['scope'])
        # 資格flagsは人工的に有利に設定。盤面・PB・scope・tokenは実保存値のまま。
        rows.append(G.V1.CallObservation(w['journal_token'],w['frame'],scope,scope[2],scope[5],
            c['state_value'],None,True,False,False,True,w['observation_stage'],c['confirmed'] is not None,
            tuple(map(tuple,w['raw'])),grid(c['cnn']),grid(c['confirmed']),grid(c['confirmed']),
            dist([(r,col) for r in range(1,13) for col in range(6)]),dist([(0,col) for col in range(6)])))
    return clock,rows


def test_actual_channels_rejected_without_inventing_missing_boards() -> None:
    clock,rows=observations()
    gate=G.SettledBasisGate(rows[0].scope,clock['reset_frame'],clock['deadline'])
    for row in rows: assert gate.observe(row) is None
    reasons={r['frame']:r['reason'] for r in gate.rows}
    assert reasons[35164]=='awaiting_tsumo_fall_after_reset'
    assert reasons[35172]=='raw_SM_mismatch'
    assert reasons[35174]=='baseline_deadline:raw_SM_mismatch' and gate.state=='BROKEN'


def test_artificial_preserved_raw_basis_at_landing_completion() -> None:
    clock,rows=observations()
    gate=G.SettledBasisGate(rows[0].scope,clock['reset_frame'],clock['deadline'])
    for row in rows:
        if row.frame==35172:
            # 人工正常対照：実rawを原SMが保持し、原PB fallbackがUNKNOWNを分布化した場合。
            pb=ProbabilisticBoard.from_board(Board.from_dict({'grid':row.raw_grid}))
            hidden=tuple(tuple(sorted(pb.cell(0,c).probs.items())) for c in range(6))
            row=replace(row,sm_confirmed_grid=row.raw_grid,returned_grid=row.raw_grid,hidden_probability=hidden)
            candidate=gate.observe(row)
            assert candidate is not None and candidate.frame==35172
            assert candidate.tsumo_fall_frames==(35168,35170) and candidate.source_call_token==row.call_token
            assert sum(len(d)>1 for d in candidate.hidden_probability)==4
            assert not candidate.current_permission and not candidate.display_update_permission
        else: assert gate.observe(row) is None
    assert gate.rows[-1]['reason']=='candidate_already_issued'


@pytest.mark.parametrize('field,value',(('frame',True),('epoch',3.0),('generation',True),
                                      ('call_token',''),('match_active',1)))
def test_input_type_rejected(field: str,value: object) -> None:
    clock,rows=observations()
    gate=G.SettledBasisGate(rows[0].scope,clock['reset_frame'],clock['deadline'])
    with pytest.raises(ValueError): gate.observe(replace(rows[0],**{field:value}))
