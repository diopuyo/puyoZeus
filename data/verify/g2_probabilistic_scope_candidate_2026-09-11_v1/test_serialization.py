"""隠し支持の相関と質量を保存。整数collector入力へ変換しない。"""
import json
import pytest
from serialization import encode,decode
from test_belief import actual
from test_joint_evaluation import states


def test_actual_343_roundtrip() -> None:
    value,_,_=actual()
    packet=encode(value)
    restored=decode(json.loads(json.dumps(packet)))
    assert restored==value and len(packet['visible'])==12 and len(packet['hidden_worlds'])==343
    assert all(len(w['cells'])==1 for w in packet['hidden_worlds'])


def test_correlated_worlds_roundtrip() -> None:
    values,_=states()
    result=decode(json.loads(json.dumps(encode(values[0]))))
    assert result==values[0]
    assert all(w.grid[0][0]!=w.grid[0][1] for w in result.worlds)


@pytest.mark.parametrize('case',('permission','weight','visible_unknown','float_color','hidden_shape'))
def test_invalid_payload_rejected(case: str) -> None:
    values,_=states()
    packet=encode(values[0])
    if case=='permission': packet['accounting_permission']=True
    elif case=='weight': packet['hidden_worlds'][0]['weight']=float('nan')
    elif case=='visible_unknown': packet['visible'][0][0]=10
    elif case=='float_color': packet['hidden_worlds'][0]['cells'][0][0]=4.0
    elif case=='hidden_shape': packet['hidden_worlds'][0]['cells'][0].append(0)
    with pytest.raises(ValueError): decode(packet)
