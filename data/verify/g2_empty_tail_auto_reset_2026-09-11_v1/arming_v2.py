"""別side通知で残っている同一Advisoryは既存予約の再確認として扱う。"""
from __future__ import annotations
from types import FunctionType
import arming as OLD

E,VERIFY = OLD.E,OLD.VERIFY


class Arming(OLD.Arming):
    def observe(self) -> None:
        if self.error is not None: raise self.error
        if self.armed is not None and self.observer.pending is self.armed:
            return
        super().observe()


install = FunctionType(OLD.install.__code__,dict(vars(OLD),Arming=Arming))
