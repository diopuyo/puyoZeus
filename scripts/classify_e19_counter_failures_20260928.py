"""E19: 評価器を変更せず、勝者一致の反転を撃ち合いと実観測へ対応付ける。"""
from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any
import numpy as np
from scripts.run_e3_exchange_eval_20260926 import save_json
from scripts.replay_exchange_event_20260926 import read_records
from src.board_state_machine import BoardState
from src.indicators_v2 import SEC_PER_HAND

OUT = Path("logs/e19")
BASE = Path("logs/e17/combined/zenchi")
ON = Path("logs/e18/on/zenchi")
LABELS = ("実際に着弾前に応手", "可能予測だが着弾前の新発火なし", "応手後に相手の次撃・敗北", "その他・時刻根拠不足")


def classify(event: dict, receiver: str, games: list, all_events: list) -> dict:
    """当該側で最初に採用した応手予測を起点に、同試合の実測発火と落下開始を見る。"""
    idx = int(receiver == "2P")
    selected = [v for v in event["values"] if v.get("response_selected") and v["incoming"][idx] > 0]
    if not selected:
        return dict(category=LABELS[3], reason="採用された応手仮定なし")
    decision = selected[0]
    stamp = decision["t_sec"]
    official = next((g for g in games if g["start"] <= stamp < g["end"]), None)
    same = [e for e in all_events if e["game_idx"] == event["game_idx"]]
    landings = [l for e in same for l in e["landings"] if l["side"] == receiver
                and l.get("fall_start_sec", l.get("t_sec", -1)) is not None
                and l.get("fall_start_sec", l.get("t_sec", -1)) > stamp]
    landing = min((l.get("fall_start_sec", l["t_sec"]) for l in landings), default=None)
    chains = sorted([c for e in same for c in e["chains"]], key=lambda c: c["trigger_sec"])
    cutoff = landing if landing is not None else (official["end"] if official else stamp)
    replies = [c for c in chains if c["side"] == receiver and stamp < c["trigger_sec"] < cutoff]
    reply = replies[0] if replies else None
    second = next((c for c in chains if reply and c["side"] != receiver
                   and c["trigger_sec"] > reply["trigger_sec"]), None)
    category = LABELS[3]
    if reply is not None and landing is not None:
        category = LABELS[2] if second and official and official["winner"] != receiver else LABELS[0]
    elif reply is None and landing is not None:
        category = LABELS[1]
    return dict(category=category, decision=decision, receiver=receiver, landing_sec=landing,
        reply=reply, next_attack=second, winner=official["winner"] if official else None,
        reason="着弾未観測は負例にしない。次撃後の敗北は時間順の分類で因果確定ではない")


def changed_rows(base: Any, on: Any, games: list, events: list) -> list[dict]:
    """終盤1/3・表示符号の既存一致率と完全に同じフレームを抽出する。"""
    np.testing.assert_array_equal(base["t_sec"], on["t_sec"])
    output = []
    for game in games:
        begin = game["start"] + (game["end"]-game["start"])*2/3
        sign = 1 if game["winner"] == "1P" else -1
        eligible = (on["t_sec"] >= begin) & (on["t_sec"] < game["end"])
        before, after = base["display_adv"]*sign > 0, on["display_adv"]*sign > 0
        for i in np.flatnonzero(eligible & (before != after)):
            stamp, local_game = float(on["t_sec"][i]), int(on["game_idx"][i])
            prior = [e for e in events if e["game_idx"] == local_game and e["trigger_sec"] <= stamp]
            event = prior[-1] if prior else None
            values = [v for v in event["values"] if v["t_sec"] <= stamp and v.get("response_selected")] if event else []
            value = values[-1] if values else None
            receiver = ("1P" if value["incoming"][0] > 0 else "2P") if value else "不明"
            output.append(dict(t_sec=stamp, game=game["game"], game_idx=local_game,
                exchange_id=event["exchange_id"] if event else None, receiver=receiver,
                direction="正→誤" if before[i] else "誤→正", baseline_p1=float(base["display_p1"][i]),
                counter_p1=float(on["display_p1"][i]), baseline_adv=float(base["display_adv"][i]),
                counter_adv=float(on["display_adv"][i]), projection=value))
    return output


def attach_observations(groups: list[dict]) -> None:
    """意思決定時点の両側確定盤面と、実際の落下までの時間を記録する。"""
    targets = defaultdict(list)
    for group in groups:
        if "decision" in group["evidence"]:
            targets[group["evidence"]["decision"]["t_sec"]].append(group)
    latest = [None, None]
    for item in read_records(Path("logs/e16/records/zenchi.jsonl.gz")):
        if item["kind"] != "update":
            continue
        result, _, _, stamp, *_ = item["args"]
        for idx, side in enumerate((result.p1, result.p2)):
            if side.state == BoardState.STABLE and side.confirmed_board is not None:
                latest[idx] = side.confirmed_board
        for group in targets.get(stamp, ()):
            evidence = group["evidence"]
            evidence["max_height"] = [max(b.height_of(c) for c in range(6)) if b else None for b in latest]
            evidence["confirmed_boards"] = [b._grid.tolist() if b else None for b in latest]
            evidence["states"] = [s.state.name for s in (result.p1, result.p2)]
            delta = None if evidence["landing_sec"] is None else evidence["landing_sec"]-stamp
            evidence["actual_seconds_to_fall"] = delta
            evidence["observed_time_budget_hands"] = None if delta is None else int(delta/SEC_PER_HAND)+1
            idx = int(evidence["receiver"] == "2P")
            evidence["time_budget_overestimate"] = delta is not None and (
                evidence["decision"]["hands"][idx] > evidence["observed_time_budget_hands"])
            evidence["no_response_cause"] = "手数過大の時間的証拠あり" if evidence["time_budget_overestimate"] else (
                "未選択・NF過大・置けないの識別は記録のみでは不能")


def run() -> None:
    """全反転行と分類根拠を残し、最多フレームの撃ち合いから代表を選ぶ。"""
    games = json.loads(Path("logs/review_zenchi_part3/official_games.json").read_text())
    events = [json.loads(s) for s in (ON/"events.jsonl").read_text().splitlines()]
    with np.load(BASE/"display.npz") as base, np.load(ON/"display.npz") as on:
        rows = changed_rows(base, on, games, events)
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["direction"], row["game_idx"], row["exchange_id"], row["receiver"])].append(row)
    groups = []
    for (direction, game, identity, receiver), frames in grouped.items():
        event = next((e for e in events if (e["game_idx"], e["exchange_id"]) == (game, identity)), None)
        evidence = classify(event, receiver, games, events) if event and receiver != "不明" else dict(category=LABELS[3])
        groups.append(dict(direction=direction, game_idx=game, exchange_id=identity, receiver=receiver,
            frames=len(frames), first_sec=frames[0]["t_sec"], last_sec=frames[-1]["t_sec"],
            representative=frames[len(frames)//2], evidence=evidence))
    attach_observations(groups)
    table = []
    for direction in ("正→誤", "誤→正"):
        for label in LABELS:
            selected = [g for g in groups if g["direction"] == direction and g["evidence"]["category"] == label]
            table.append(dict(direction=direction, category=label, frames=sum(g["frames"] for g in selected),
                exchanges=len({(g["game_idx"], g["exchange_id"]) for g in selected})))
    totals = Counter(r["direction"] for r in rows)
    assert totals["正→誤"]-totals["誤→正"] == 580
    representatives = {}
    for direction in totals:
        ordered = sorted([g for g in groups if g["direction"] == direction], key=lambda g: -g["frames"])
        diverse = {g["evidence"]["category"]: g for g in reversed(ordered)}
        representatives[direction] = sorted(diverse.values(), key=lambda g: -g["frames"])[:3]
    save_json(OUT/"CLASSIFICATION.json", dict(table=table, totals=dict(totals), groups=groups,
        representatives=representatives, note="同一撃ち合いで受け側・分類が変わる場合は件数が重複する"))
    save_json(OUT/"CHANGED_FRAMES.json", rows)
    print(json.dumps(dict(table=table, totals=dict(totals)), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    run()
