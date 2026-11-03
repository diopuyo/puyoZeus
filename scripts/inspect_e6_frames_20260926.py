"""撤回分類の代表フレームと全件の直前入力を保存する。"""
from __future__ import annotations

import json
from pathlib import Path

import cv2

from scripts.run_e3_exchange_eval_20260926 import SOURCES, VIDEO_ROOT, save_json
from src.exchange_event_record import read_records

OUT = Path("logs/e6/diagnosis")
SAMPLE_EVENTS = {SOURCES[0]: (6,), SOURCES[1]: (5, 9), SOURCES[2]: (3, 10, 11, 31)}


def enrich(source: str, cases: list[dict]) -> None:
    """直前の非欠測表示得点とNEXT変化時刻を全件に追加する。"""
    previous, last_next, next_change = [None, None], [None, None], [None, None]
    targets = {row["signal"]["revoked_sec"] for row in cases}
    for item in read_records(Path("logs/e5/renders") / source / "on/inputs.jsonl.gz"):
        if item["kind"] != "update":
            continue
        result, _, _, t, _, _, scores, _ = item["args"]
        for idx, side in enumerate((result.p1, result.p2)):
            if side.next_pair is not None and side.next_pair != last_next[idx]:
                next_change[idx], last_next[idx] = t, side.next_pair
            for row in cases:
                if row["side"] != ("1P", "2P")[idx]:
                    continue
                if t == row["signal"]["t_sec"]:
                    row["last_next_change_sec"] = next_change[idx]
                if t in targets and t == row["signal"]["revoked_sec"]:
                    row["previous_display_score"] = previous[idx]
                    row["score_step"] = (scores[idx] - previous[idx]
                        if scores[idx] is not None and previous[idx] is not None else None)
            if scores[idx] is not None:
                previous[idx] = scores[idx]


def frames(source: str, cases: list[dict]) -> None:
    """信号前・信号時・撤回時を元動画から抽出する。"""
    cap = cv2.VideoCapture(str(VIDEO_ROOT / f"{source}_first_0_900_20260925_v1.mp4"))
    directory = OUT / "frames"
    directory.mkdir(exist_ok=True)
    selected = {}
    for row in cases:
        if row["exchange_id"] in SAMPLE_EVENTS[source]:
            selected.setdefault(row["exchange_id"], row)
    for event, row in selected.items():
        for t in (row["signal"]["t_sec"] - .2, row["signal"]["t_sec"], row["signal"]["revoked_sec"]):
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ok, frame = cap.read()
            if not ok:
                raise ValueError(f"フレーム欠測: {source} {t}")
            cv2.imwrite(str(directory / f"{source}_e{event}_{t:.3f}.jpg"), frame)
    cap.release()


def main() -> None:
    """24信号・15撃ち合いを同じ分類台帳へ保持する。"""
    rows = json.loads((OUT / "cases.json").read_text())
    for source in SOURCES:
        cases = [r for r in rows if r["source"] == source]
        enrich(source, cases)
        frames(source, cases)
    save_json(OUT / "cases_enriched.json", rows)


if __name__ == "__main__":
    main()
