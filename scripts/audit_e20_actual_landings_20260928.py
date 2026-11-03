"""E19の着弾通知と、後続STABLE盤面のおじゃま増加を独立に照合する。"""
from __future__ import annotations
import json
from pathlib import Path
from scripts.replay_exchange_event_20260926 import read_records
from scripts.run_e3_exchange_eval_20260926 import save_json
from src.board_state_machine import BoardState

AUDIT_WINDOW_SEC = 30.


def main() -> None:
    """認識された落下開始を真値とせず、映像確認すべき時刻を抽出する。"""
    groups = json.loads(Path("logs/e19/CLASSIFICATION.json").read_text())["groups"]
    targets = [dict(game=g["game_idx"], receiver=g["receiver"],
        decision_sec=g["evidence"]["decision"]["t_sec"], notification_sec=g["evidence"]["landing_sec"],
        changes=[], last_count=None) for g in groups if g["direction"] == "正→誤" and
        g["evidence"]["category"] == "可能予測だが着弾前の新発火なし"]
    for item in read_records(Path("logs/e16/records/zenchi.jsonl.gz")):
        if item["kind"] != "update":
            continue
        result, _, _, stamp, game, *_ = item["args"]
        for row in targets:
            if game != row["game"] or not row["decision_sec"] <= stamp <= row["decision_sec"]+AUDIT_WINDOW_SEC:
                continue
            side = result.p1 if row["receiver"] == "1P" else result.p2
            if side.state != BoardState.STABLE or side.confirmed_board is None:
                continue
            grid = side.confirmed_board._grid
            count = int((grid == 9).sum())
            if count != row["last_count"]:
                row["changes"].append(dict(t_sec=stamp, count=count))
                row["last_count"] = count
    save_json(Path("logs/e20/LANDING_OBSERVATIONS.json"), targets)
    print(json.dumps(targets, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
