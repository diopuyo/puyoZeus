"""不変Beliefの前進だけを一所有者内で直列化する。実factory接続は別責務。"""
from __future__ import annotations
from threading import Lock
from typing import Any,Callable
import belief as B


class Owner:
    def __init__(self,value: B.Belief) -> None:
        B.validate(value)
        self._value=value
        self._lock=Lock()

    @property
    def state(self) -> B.Belief:
        with self._lock: return self._value

    def _apply(self,expected: B.Belief,operation: Callable[...,Any],*args: Any) -> tuple[B.Belief,float]:
        with self._lock:
            B.require(expected is self._value,'owner_stale_reference')
            following,mass=operation(self._value,*args)
            self._value=following
            return following,mass

    def land(self,expected: B.Belief,scope: tuple[Any,...],frame: int,token: str,pair: tuple[int,int],
             placements: tuple[tuple[int,int,int],...],observed: B.Board) -> tuple[B.Belief,float]:
        return self._apply(expected,B.land,scope,frame,token,pair,placements,observed)

    def settle(self,expected: B.Belief,scope: tuple[Any,...],frame: int,token: str,chain_count: int,
               observed: B.Board) -> tuple[B.Belief,float]:
        return self._apply(expected,B.settle,scope,frame,token,chain_count,observed)
