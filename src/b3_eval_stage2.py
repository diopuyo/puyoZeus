"""B3 段2 (zenchi 115 試合) と封印セットの合否の機械判定 (事前登録 §4, §6.2)。

入力は採点器 (既存 M_time/M_agree/誤確定 + 本モジュールの確率採点) の出力を構造化した辞書。
判定は PASS/FAIL/HOLD。未提供 (None) の項目は HOLD で、合格に数えない。
数値門は事前登録の不等号をそのまま定数化 (結果を見て動かさない)。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

PASS, FAIL, HOLD = "PASS", "FAIL", "HOLD"


@dataclass(frozen=True)
class Stage2Gates:
    """§4 の数値門。"""
    first_confirm_sec_max: float = 2766.0      # 3:00 場面の正しい確定時刻
    delay_p95_over_r_max: float = 0.5          # 共通確定事例の遅れ p95 の R 比 (秒)
    new_delay_over_sec: float = 5.0            # 5 秒超の新規遅れは 0
    m_time_point_max: float = 0.002
    m_time_upper_lt: float = 0.020
    m_agree_point_min: float = -0.005
    m_agree_lower_gt: float = -0.030
    dll_point_max: float = 0.0
    dll_upper_max: float = 0.010
    brier_diff_max: float = 0.002


@dataclass(frozen=True)
class SealedGates:
    """§6.2 の数値門。"""
    per_set_dll_point_max: float = 0.010
    per_set_brier_max: float = 0.005
    total_dll_point_max: float = 0.0
    total_dll_upper_max: float = 0.010
    total_brier_max: float = 0.002
    total_ece_max: float = 0.010
    delay_p95_over_r_max: float = 0.5
    new_delay_over_sec: float = 5.0
    min_correct_confirms_per_set: int = 5
    min_correct_confirms_total: int = 15
    min_games_exchange: int = 20
    min_games_build: int = 20
    min_games_per_set: int = 20


STAGE2_GATES = Stage2Gates()
SEALED_GATES = SealedGates()


def _it(name: str, value, thr: str, ok: Optional[bool], note: str = "") -> dict:
    return {"name": name, "value": value, "threshold": thr, "ok": ok, "note": note}


def _fin(*xs) -> bool:
    return all(isinstance(x, (int, float)) and math.isfinite(x) for x in xs)


def _verdict(items: Sequence[Mapping]) -> str:
    if any(i["ok"] is False for i in items):
        return FAIL
    if any(i["ok"] is None for i in items):
        return HOLD
    return PASS


def _confirm_items(prefix: str, c: Optional[Mapping], g) -> list[dict]:
    """正しい確定の消失・遅れ (R 比)。c = {lost, delay_p95_diff, new_delay_over_5s, ...}。"""
    if c is None:
        return [_it(f"{prefix}confirm_lost", None, "<=0 (R 以下)", None, "未提供")]
    return [
        _it(f"{prefix}confirm_lost", c.get("lost"), "正しい確定の消失数 <= R (差<=0)",
            None if c.get("lost") is None else c["lost"] <= 0),
        _it(f"{prefix}delay_p95_over_r", c.get("delay_p95_diff"), f"<=+{g.delay_p95_over_r_max}s",
            None if c.get("delay_p95_diff") is None else c["delay_p95_diff"] <= g.delay_p95_over_r_max),
        _it(f"{prefix}new_delay_over_5s", c.get("new_delay_over_5s"), "0",
            None if c.get("new_delay_over_5s") is None else c["new_delay_over_5s"] == 0),
    ]


def evaluate_stage2(inp: Mapping, g: Stage2Gates = STAGE2_GATES) -> dict:
    """段2。inp のキー (全て任意、無ければ HOLD):
    false_confirms: {records18, short5, game14, all115} 各件数 / first_confirm_sec_3min / confirms (R 比)
    m_time, m_agree: {point, lo, hi} (R 比の対差、既存採点器) / dll: {point, hi} / brier_diff
    review_windows_done: bool / time_gate_ok: bool / forecast72: {sole_confirms, false_confirms}
    """
    items: list[dict] = []
    fc = inp.get("false_confirms")
    for k in ("records18", "short5", "game14", "all115"):
        v = None if fc is None else fc.get(k)
        items.append(_it(f"false_confirm_{k}", v, "0", None if v is None else v == 0, "母数は別欄で併記"))
    t = inp.get("first_confirm_sec_3min")
    items.append(_it("first_confirm_3min", t, f"<={g.first_confirm_sec_max}s", None if t is None else t <= g.first_confirm_sec_max))
    items += _confirm_items("", inp.get("confirms"), g)
    mt, ma = inp.get("m_time"), inp.get("m_agree")
    items.append(_it("m_time_vs_R", mt, f"点推定<=+{g.m_time_point_max} かつ 上端<{g.m_time_upper_lt}",
                     None if mt is None else bool(_fin(mt["point"], mt["hi"]) and mt["point"] <= g.m_time_point_max and mt["hi"] < g.m_time_upper_lt)))
    items.append(_it("m_agree_vs_R", ma, f"点推定>={g.m_agree_point_min} かつ 下端>{g.m_agree_lower_gt}",
                     None if ma is None else bool(_fin(ma["point"], ma["lo"]) and ma["point"] >= g.m_agree_point_min and ma["lo"] > g.m_agree_lower_gt)))
    d = inp.get("dll")
    items.append(_it("delta_ll", d, f"点推定<={g.dll_point_max} かつ 上端<={g.dll_upper_max}",
                     None if d is None else bool(_fin(d["point"], d["hi"]) and d["point"] <= g.dll_point_max and d["hi"] <= g.dll_upper_max)))
    b = inp.get("brier_diff")
    items.append(_it("brier_diff", b, f"<={g.brier_diff_max}", None if b is None else bool(_fin(b) and b <= g.brier_diff_max)))
    for key, label in (("review_windows_done", "review_7_windows"), ("time_gate_ok", "time_gate_115_and_30min")):
        v = inp.get(key)
        items.append(_it(label, v, "True", None if v is None else bool(v)))
    f72 = inp.get("forecast72")
    items.append(_it("forecast72_false_confirms", f72, "0 (単独根拠の誤確定1件でも不合格)",
                     None if f72 is None else f72.get("false_confirms") == 0))
    return {"verdict": _verdict(items), "items": items, "unmet": [i["name"] for i in items if i["ok"] is False],
            "not_evaluated": [i["name"] for i in items if i["ok"] is None]}


def evaluate_sealed(sets: Mapping[str, Mapping], total: Optional[Mapping], g: SealedGates = SEALED_GATES,
                    *, time_gate_ok: Optional[bool] = None, forecast72: Optional[Mapping] = None,
                    opening_complete: bool = True) -> dict:
    """封印 (一括開封後)。sets[name] = {games, false_confirms, undetermined_confirms, correct_confirms_R,
    confirms(R比), dll:{point}, brier_diff} / total = {dll:{point,hi}, brier_diff, ece_diff, games_exchange, games_build}。

    開封で破損・欠落が出たセットは opening_complete=False にして HOLD (判定不能)。
    開封前に分からない条件 (R の正しい確定 5/15 件、撃ち合い/両者構築 20 試合) が不足なら HOLD (閾値は緩めない)。
    """
    items: list[dict] = []
    if not opening_complete:
        items.append(_it("opening_complete", False, "全セット欠落なし", None, "破損・欠落 → 判定不能 (保留)"))
    cc_total = 0
    for name, s in sets.items():
        p = f"{name}:"
        gm = s.get("games")
        items.append(_it(p + "games", gm, f">={g.min_games_per_set}", None if gm is None else gm >= g.min_games_per_set))
        for k, label in (("false_confirms", "false_confirms"), ("undetermined_confirms", "undetermined_confirms")):
            v = s.get(k)
            items.append(_it(p + label, v, "0", None if v is None else v == 0))
        cc = s.get("correct_confirms_R")
        cc_total += cc or 0
        items.append(_it(p + "correct_confirms_R", cc, f">={g.min_correct_confirms_per_set} (不足は保留)",
                         None if cc is None or cc < g.min_correct_confirms_per_set else True,
                         "不足 → 最終合格は保留" if cc is not None and cc < g.min_correct_confirms_per_set else ""))
        items += _confirm_items(p, s.get("confirms"), g)
        d, b = s.get("dll"), s.get("brier_diff")
        items.append(_it(p + "delta_ll", d, f"点推定<=+{g.per_set_dll_point_max}",
                         None if d is None else bool(_fin(d["point"]) and d["point"] <= g.per_set_dll_point_max)))
        items.append(_it(p + "brier_diff", b, f"<=+{g.per_set_brier_max}",
                         None if b is None else bool(_fin(b) and b <= g.per_set_brier_max)))
    items.append(_it("correct_confirms_R_total", cc_total, f">={g.min_correct_confirms_total} (不足は保留)",
                     True if cc_total >= g.min_correct_confirms_total else None,
                     "" if cc_total >= g.min_correct_confirms_total else "不足 → 保留"))
    if total is None:
        items.append(_it("total", None, "全体 ΔLL/Brier/ECE", None, "未提供"))
    else:
        d = total.get("dll")
        items.append(_it("total_delta_ll", d, f"点推定<={g.total_dll_point_max} かつ 上端<={g.total_dll_upper_max}",
                         None if d is None else bool(_fin(d["point"], d["hi"]) and d["point"] <= g.total_dll_point_max and d["hi"] <= g.total_dll_upper_max)))
        for key, thr in (("brier_diff", g.total_brier_max), ("ece_diff", g.total_ece_max)):
            v = total.get(key)
            items.append(_it(f"total_{key}", v, f"<={thr}", None if v is None else bool(_fin(v) and v <= thr)))
        for key, mn, label in (("games_exchange", g.min_games_exchange, "coverage_exchange"),
                               ("games_build", g.min_games_build, "coverage_build")):
            v = total.get(key)
            items.append(_it(label, v, f">={mn} 試合 (不足は保留)", None if v is None or v < mn else True,
                             "不足 → 保留" if v is not None and v < mn else ""))
    items.append(_it("time_gate", time_gate_ok, "全セット再生 + 指定30分", time_gate_ok, "未実施" if time_gate_ok is None else ""))
    items.append(_it("forecast72", forecast72, "単独根拠の誤確定 0", None if forecast72 is None else forecast72.get("false_confirms") == 0))
    return {"verdict": _verdict(items), "items": items, "unmet": [i["name"] for i in items if i["ok"] is False],
            "not_evaluated": [i["name"] for i in items if i["ok"] is None]}
