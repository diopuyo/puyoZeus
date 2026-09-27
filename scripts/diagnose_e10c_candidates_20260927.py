"""実手数へ縮めた探索が新たに断定した候補の盤面と応手を確認する。"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from src.chain import ChainSimulator
from src.exchange_event_record import read_records
from src.exchange_event_landing import future_send, RESPONSE_BEAM_WIDTH
from src.indicators_v2 import near_future_fire_power
from src.production_config import GHOST_CHAIN_RULE_ENABLED
from src.scoring import calculate_chain_score

OUT = Path("logs/e10c/iteration1")
ROOT = Path("/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer")


def examine(source: str, cases: list[dict]) -> None:
    """受け側確定盤面の実手数と、既存指標の楽観的候補探索を比較する。"""
    record = (Path("logs/review_zenchi_part3/on_e9/inputs.jsonl.gz") if source == "zenchi"
              else Path(f"logs/e8/renders/{source}/on/inputs.jsonl.gz"))
    latest, done = [None, None], set()
    sim = ChainSimulator(exclude_hidden_row_from_pop=GHOST_CHAIN_RULE_ENABLED)
    for item in read_records(record):
        if item["kind"] != "update":
            continue
        result, snap, _, t, *_ = item["args"]
        for i, side in enumerate((result.p1, result.p2)):
            if side.state.name == "STABLE" and side.confirmed_board is not None:
                latest[i] = (t, side)
        for case in cases:
            if t < case["first_sec"]-.001 or case["exchange_id"] in done:
                continue
            done.add(case["exchange_id"])
            i = ("1P", "2P").index(case["side"])
            timestamp, side = latest[i]
            board = side.confirmed_board
            resolved = sim.simulate(board)
            grid = resolved.final_board._grid
            queue = tuple(side.next_pair or (0, 0)) + tuple(side.dnext_pair or (0, 0))
            hands = case["first"]["hands"][i]
            broad = near_future_fire_power(resolved.final_board, queue[:2], queue[2:],
                k_levels=(hands,), beam_width=RESPONSE_BEAM_WIDTH, resolve_before_death=True)
            print(json.dumps(dict(source=source, t=t, board_t=timestamp,
                state=(result.p1, result.p2)[i].state.name, board=board._grid.tolist(),
                resolved=grid.tolist(), queue=queue, hands=hands,
                actual=future_send(grid.tobytes(), grid.shape, grid.dtype.str, queue, hands, 0),
                optimistic=broad.values[hands].raw, snapshot=vars(snap),
                simulated_score=calculate_chain_score(resolved).total_score,
                chain=None if (result.p1, result.p2)[i].chain_event is None else
                    vars((result.p1, result.p2)[i].chain_event))), flush=True)


def frames(cases: list[dict]) -> None:
    """発火と2秒後を動画別の実画面で照合する。"""
    tiles = []
    for case in cases:
        source = case["source"]
        video = (ROOT / "data/frames/video_zenchi_c0BQoMJwwQU.mp4" if source == "zenchi"
                 else Path(f"/mnt/d/puyo_analyzer/videos/source/{source}_first_0_900_20260925_v1.mp4"))
        cap = cv2.VideoCapture(str(video))
        for offset in (0, 2):
            t = case["first_sec"] + offset
            cap.set(cv2.CAP_PROP_POS_MSEC, t*1000)
            ok, frame = cap.read()
            if not ok:
                raise ValueError(f"実画面取得失敗: {source} {t}")
            cv2.imwrite(str(OUT / f"{source}_{t:.3f}.jpg"), frame)
            tile = cv2.resize(frame, (640, 360))
            cv2.putText(tile, f"{source} {t:.3f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 0, 255), 2)
            tiles.append(tile)
        cap.release()
    cv2.imwrite(str(OUT / "candidates.jpg"), np.vstack([
        np.hstack(tiles[i:i+2]) for i in range(0, len(tiles), 2)]))


def main() -> None:
    """初回全区間検証で新たに出た誤発火を全件保存する。"""
    cases = [r for r in json.loads((OUT / "report.json").read_text())["deaths"] if r["false_positive"]]
    frames(cases)
    for source in dict.fromkeys(r["source"] for r in cases):
        examine(source, [r for r in cases if r["source"] == source])


if __name__ == "__main__":
    main()
