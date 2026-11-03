"""6個の反対称特徴と非負係数の単調変換。相手情報を使うのは変換だけ。"""
from __future__ import annotations
import math
from src.ama_plain_reference import DEPTH

FEATURES = ('tempo', 'pressure', 'pending', 'consumption', 'ama_value', 'survival')
SEND_SCALE, SCORE_SCALE, CAPACITY_SCALE = 180., 600000., 78.


def features(value: dict) -> list:
    a, b = value['plans']
    pa, pb = value['pending']
    ha, hb = [p['h'] if p['h'] is not None else DEPTH+1 for p in (a, b)]
    pressure_a = max(0, a['sent']-pa)/(1+b['capacity'])
    pressure_b = max(0, b['sent']-pb)/(1+a['capacity'])
    # 差は[-1,1]に正規化。入替で符号反転するため、切片0ならpも1-pに反転する。
    return [(hb-ha)/(DEPTH+1), math.tanh((pressure_a-pressure_b)/6),
            math.tanh((pb-pa)/SEND_SCALE), a['consumption']-b['consumption'],
            math.tanh((a['ama_score']-b['ama_score'])/SCORE_SCALE),
            (a['capacity']-b['capacity'])/CAPACITY_SCALE]


def probability(value: dict, model: dict) -> float:
    if model['features'] != list(FEATURES):
        raise ValueError('特徴順序が不一致')
    dead = [p.get('terminal', False) for p in value['plans']]
    if any(dead):
        return .5 if all(dead) else float(dead[1])
    score = sum(x*w for x, w in zip(features(value), model['coefficients']))
    return 1/(1+math.exp(-max(-700., min(700., score))))
