"""可視盤面を共有し、隠し段のjoint支持を別列へ損失なく保存する私有schema。"""
from __future__ import annotations
from typing import Any
import belief as B

SCHEMA='probabilistic-scope-state/v1'
PERMISSIONS=('physical_certified','integer_current_permission','accounting_permission','production_permission','quality_gate_clear')


def encode(value: B.Belief) -> dict[str,Any]:
    B.validate(value)
    visible=value.worlds[0].grid[B.HIDDEN_ROWS:]
    B.require(all(w.grid[B.HIDDEN_ROWS:]==visible for w in value.worlds),'serialized_visible_not_shared')
    return dict(schema=SCHEMA,scope=list(value.scope),frame=value.frame,deadline=value.deadline,
        tokens=list(value.tokens),visible=[list(row) for row in visible],
        hidden_worlds=[dict(cells=[list(row) for row in w.grid[:B.HIDDEN_ROWS]],weight=w.weight) for w in value.worlds],
        provisional=True,within_side_joint_preserved=True,**dict.fromkeys(PERMISSIONS,False))


def decode(packet: dict[str,Any]) -> B.Belief:
    B.require(type(packet) is dict and packet.get('schema')==SCHEMA,'serialized_schema')
    B.require(packet.get('provisional') is True and packet.get('within_side_joint_preserved') is True
              and all(packet.get(k) is False for k in PERMISSIONS),'serialized_permissions')
    B.require(type(packet.get('scope')) is list and type(packet.get('tokens')) is list,'serialized_scope_tokens')
    visible=packet.get('visible')
    hidden=packet.get('hidden_worlds')
    B.require(type(visible) is list and len(visible)==B.BOARD_ROWS-B.HIDDEN_ROWS
              and all(type(row) is list and len(row)==B.BOARD_COLS for row in visible),'serialized_visible')
    B.require(type(hidden) is list and 0<len(hidden)<=B.MAX_WORLDS,'serialized_support')
    worlds=[]
    for item in hidden:
        B.require(type(item) is dict and type(item.get('cells')) is list
                  and len(item['cells'])==B.HIDDEN_ROWS,'serialized_hidden')
        B.require(all(type(row) is list and len(row)==B.BOARD_COLS for row in item['cells']),'serialized_hidden_shape')
        worlds.append(B.World(tuple(map(tuple,item['cells']+visible)),item.get('weight')))
    value=B.Belief(tuple(packet['scope']),packet.get('frame'),packet.get('deadline'),tuple(worlds),tuple(packet['tokens']))
    B.validate(value)
    return value
