"""B3 段1 の合否を事前登録の数値門どおりに機械判定し、受領票 (母数つき) を返す (§3.2〜§3.5)。

判定は 3 値 + 停止: PASS / FAIL / HOLD / STOP_ORACLE。
未実施の検査 (None) は合格に数えない (HOLD)。測っていないことを「0件=合格」と読み替えない。
"""
from __future__ import annotations

import math
from typing import Mapping, Optional, Sequence

from src.b3_eval_metrics import oracle_exceeded
from src.b3_train_config import (MONO_HOLD_GAMES, MONO_HOLD_WIDTH, MONO_MINOR_SHARE, MONO_MINOR_WIDTH,
                                 OTHER_STRATA, PAIR_STRATA, SECONDARY_HOLD_MEAN, SECONDARY_HOLD_PLAYER,
                                 SECONDARY_HOLD_PLAYER_MIN_GAMES, STAGE1_GATES, Stage1Gates)

PASS, FAIL, HOLD, STOP = "PASS", "FAIL", "HOLD", "STOP_ORACLE"


def _item(name: str, value, threshold: str, ok: Optional[bool], note: str = "", population: Optional[dict] = None) -> dict:
    return {"name": name, "value": value, "threshold": threshold, "ok": ok, "note": note,
            "population": population}


def _finite(*xs: float) -> bool:
    return all(isinstance(x, (int, float)) and math.isfinite(x) for x in xs)


def check_leak_prechecks(prechecks: Optional[Mapping[str, Optional[int]]]) -> dict:
    """§3.1 前提検査 (違反件数)。None/未提供の項目は未実施=HOLD。全項目 0 で OK。"""
    required = ("future_suffix_prefix_hash", "train_replay_live_hash", "symmetry", "schema_mask",
               "candidate_correspondence", "row_key_unique", "time_fire_id", "sealed_overlap",
               "available_at", "group_separation")
    if prechecks is None:
        return _item("leak_prechecks", None, "全項目の違反0", None, "未提供")
    missing = [k for k in required if prechecks.get(k) is None]
    bad = {k: v for k, v in prechecks.items() if v not in (None, 0)}
    ok = False if bad else (None if missing else True)
    return _item("leak_prechecks", {"violations": bad, "not_run": missing}, "全項目の違反0", ok)


def evaluate_stage1(scores: Mapping, gates: Stage1Gates = STAGE1_GATES, *, oracle_ll: Optional[float] = None,
                    prechecks: Optional[Mapping[str, Optional[int]]] = None,
                    time_gate_ok: Optional[bool] = None, secondary: Optional[Mapping] = None,
                    monotonic: Optional[Mapping] = None) -> dict:
    """段1 の受領票。scores は b3_eval_cv.stage1_scores の出力。"""
    items: list[dict] = []
    ov = scores["overall"]
    d = ov["delta_ll"]
    cand_ll = ov.get("ll_cand", float("nan"))
    stop = oracle_exceeded(cand_ll, oracle_ll)
    items.append(_item("oracle_not_exceeded", {"cand_ll": cand_ll, "oracle_ll": oracle_ll, "exceeded": stop},
                       "候補LL >= オラクルLL (下回ればリーク疑いで停止)", None if stop is None else (not stop),
                       "oracle 未指定" if stop is None else ""))
    items.append(check_leak_prechecks(prechecks))
    # 1. 全体
    ok_pt = _finite(d["point"], d["hi"]) and d["point"] < gates.overall_point_max
    ok_hi = _finite(d["hi"]) and d["hi"] <= gates.overall_upper_max
    items.append(_item("overall_delta_ll", {"point": d["point"], "hi": d["hi"], "lo": d["lo"]},
                       f"点推定<{gates.overall_point_max} かつ 上端<={gates.overall_upper_max}", bool(ok_pt and ok_hi),
                       "上端<0 (改善の確証)" if _finite(d["hi"]) and d["hi"] < 0 else "", ov["population"]))
    # 2. 層
    for name in PAIR_STRATA:
        items.append(_stratum_item(scores, name, gates, pair=True))
    for name in OTHER_STRATA:
        items.append(_stratum_item(scores, name, gates, pair=False))
    # 3. Brier / ECE (全体)
    items.append(_item("overall_delta_brier", ov["delta_brier"], f"<={gates.brier_diff_max}",
                       bool(_finite(ov["delta_brier"]) and ov["delta_brier"] <= gates.brier_diff_max)))
    items.append(_item("overall_delta_ece", ov["delta_ece"], f"<={gates.ece_diff_max}",
                       bool(_finite(ov["delta_ece"]) and ov["delta_ece"] <= gates.ece_diff_max)))
    # 5. 計算時間門 / 6. 副次 CV / 単調性
    items.append(_item("time_gate", time_gate_ok, "§8 の計算時間・適用範囲の門", time_gate_ok,
                       "未実施" if time_gate_ok is None else ""))
    items.append(_secondary_item(secondary))
    items.append(_monotonic_item(monotonic))
    verdict = _verdict(items, stop)
    return {"verdict": verdict, "items": items,
            "unmet": [i["name"] for i in items if i["ok"] is False],
            "not_evaluated": [i["name"] for i in items if i["ok"] is None],
            "gates": gates.__dict__}


def _stratum_item(scores: Mapping, name: str, gates: Stage1Gates, pair: bool) -> dict:
    s = scores["strata"].get(name)
    label = f"stratum_{name}"
    if s is None or s.get("empty"):
        return _item(label, None, "最低母数 各20試合・5 group", None,
                     "層が空 (母数不足 → 保留)", (s or {}).get("population"))
    pop, d = s["population"], s["delta_ll"]
    if not s["meets_min_population"]:
        return _item(label, {"point": d["point"], "hi": d["hi"]}, "最低母数 各20試合・5 group", None,
                     "母数不足 → 保留 (合格扱いにしない)", pop)
    if pair:
        ok = _finite(d["point"]) and d["point"] < gates.pair_strata_point_max
        thr = f"点推定<{gates.pair_strata_point_max}"
    else:
        ok = _finite(d["point"], d["hi"]) and d["point"] <= gates.other_strata_point_max and \
            d["hi"] <= gates.other_strata_upper_max
        thr = f"点推定<={gates.other_strata_point_max} かつ 上端<={gates.other_strata_upper_max}"
    return _item(label, {"point": d["point"], "hi": d["hi"], "lo": d["lo"]}, thr, bool(ok), "", pop)


def secondary_hold(d_mean: float, per_player: Mapping[str, Mapping]) -> dict:
    """§3.3 の保留条件: d̄ > +0.02、または d_P > +0.05 の選手が (test 20試合以上で) 1 名以上。"""
    hits = [p for p, v in per_player.items() if v["d"] > SECONDARY_HOLD_PLAYER
            and v["test_games"] >= SECONDARY_HOLD_PLAYER_MIN_GAMES]
    mean_hold = d_mean > SECONDARY_HOLD_MEAN
    return {"hold": bool(mean_hold or hits), "mean_hold": bool(mean_hold), "players_over": hits}


def _secondary_item(sec: Optional[Mapping]) -> dict:
    if sec is None:
        return _item("secondary_player_cv", None, f"d̄<={SECONDARY_HOLD_MEAN} かつ 選手別 d_P<={SECONDARY_HOLD_PLAYER}",
                     None, "未実施")
    h = secondary_hold(sec["d_mean"], sec["per_player"])
    return _item("secondary_player_cv", {"d_mean": sec["d_mean"], **h}, "保留条件に当たらない", not h["hold"],
                 "報告項目 (乖離時は保留)")


def classify_monotonic(layers: Sequence[Mapping], static_total_sec: float) -> dict:
    """§3.5: 色ぷよ差単調性の違反層の扱い。

    layers: [{name, held_sec, max_violation, games_over_005}]。違反なし (max_violation<=0) の層は無視。
    軽微: 違反層の保持秒計 <=5% かつ全違反幅 <=0.03。保留: 違反層保持秒 >5% または 幅>0.05 が 20 試合以上。
    それ以外 (中間) は統括判断 (ok=None)。
    """
    viol = [l for l in layers if l["max_violation"] > 0]
    share = sum(l["held_sec"] for l in viol) / static_total_sec if static_total_sec > 0 else float("nan")
    games_over = sum(l.get("games_over_005", 0) for l in viol)
    width = max((l["max_violation"] for l in viol), default=0.0)
    if not viol:
        return {"class": "none", "ok": True, "share": 0.0, "max_width": 0.0, "games_over_005": 0}
    if share > MONO_MINOR_SHARE or games_over >= MONO_HOLD_GAMES:
        cls, ok = "hold", False
    elif share <= MONO_MINOR_SHARE and width <= MONO_MINOR_WIDTH:
        cls, ok = "minor", True
    else:
        cls, ok = "intermediate", None
    return {"class": cls, "ok": ok, "share": share, "max_width": width, "games_over_005": games_over}


def _monotonic_item(mono: Optional[Mapping]) -> dict:
    if mono is None:
        return _item("monotonic_post_check", None, "軽微/なし (違反層5%以下・幅0.03以下)", None, "未実施")
    return _item("monotonic_post_check", dict(mono), "軽微/なし", mono.get("ok"),
                 "中間は統括判断" if mono.get("class") == "intermediate" else "")


def _verdict(items: Sequence[Mapping], stop: Optional[bool]) -> str:
    if stop:
        return STOP
    if any(i["ok"] is False for i in items):
        return FAIL
    if any(i["ok"] is None for i in items):
        return HOLD
    return PASS
