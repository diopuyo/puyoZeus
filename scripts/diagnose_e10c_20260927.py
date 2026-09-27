"""E10bの全誤発火を確定盤面・会計・実画像で時系列照合する。"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from src.exchange_event_record import read_records

OUT = Path("logs/e10c")
SOURCE = "mia8KCjr52g"
VIDEO = Path("/mnt/d/puyo_analyzer/videos/source") / f"{SOURCE}_first_0_900_20260925_v1.mp4"


def extract_frames(cases: list[dict]) -> None:
    """各発火時と2秒後の実画面を保存する。"""
    cap = cv2.VideoCapture(str(VIDEO))
    tiles = []
    for case in cases:
        for offset in (0, 2):
            t = case["first_sec"] + offset
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ok, frame = cap.read()
            if not ok:
                raise ValueError(f"フレーム取得失敗: {t}")
            name = f"case_{case['exchange_id']}_{t:.3f}.jpg"
            cv2.imwrite(str(OUT / "frames" / name), frame)
            tile = cv2.resize(frame, (640, 360))
            cv2.putText(tile, f"E{case['exchange_id']} {t:.3f}s", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 0, 255), 2)
            tiles.append(tile)
    cv2.imwrite(str(OUT / "frames/contact.jpg"), np.vstack([
        np.hstack(tiles[i:i+2]) for i in range(0, len(tiles), 2)]))
    cap.release()


def describe(side: object, t: float) -> dict:
    """STABLE盤面の実配列を保存し、時刻ずれを再現可能にする。"""
    board = side.confirmed_board
    return dict(t=t, grid=board._grid.tolist(), dead=board.is_dead(),
                queue=(side.next_pair, side.dnext_pair))


def main() -> None:
    """6件の前後を、変化点と毎秒・発火瞬間で保存する。"""
    (OUT / "frames").mkdir(parents=True, exist_ok=True)
    report = json.loads(Path("logs/e10b/report.json").read_text())
    cases = [r for r in report["deaths"] if r["false_positive"]]
    extract_frames(cases)
    latest, previous, rows = [None, None], {}, []
    record = Path(f"logs/e8/renders/{SOURCE}/on/inputs.jsonl.gz")
    for row in read_records(record):
        if row["kind"] != "update":
            continue
        result, snap, fin, t, game, *rest = row["args"]
        for i, side in enumerate((result.p1, result.p2)):
            if side.state.name == "STABLE" and side.confirmed_board is not None:
                latest[i] = describe(side, t)
        for case in cases:
            if not case["first_sec"] - 3 <= t <= case["first_sec"] + 4:
                continue
            states = [s.state.name for s in (result.p1, result.p2)]
            key = (states, [s["grid"] if s else None for s in latest], vars(snap), int(t))
            if previous.get(case["exchange_id"]) == key and abs(t-case["first_sec"]) > .001:
                continue
            previous[case["exchange_id"]] = key
            chains = [None if s.chain_event is None else vars(s.chain_event)
                      for s in (result.p1, result.p2)]
            rows.append(dict(case=case["exchange_id"], t=t, game=game, states=states,
                             latest=list(latest), snapshot=vars(snap), rest=rest,
                             finalization=vars(fin), chains=chains))
    (OUT / "diagnosis.json").write_text(json.dumps(rows, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
