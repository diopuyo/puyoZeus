"""B3 テスト用の合成 B2 記録 (observations.jsonl.gz / events.jsonl / lean.npz) を作る。"""
from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np

FPS_STRIDE = 2
FRAME_RATE = 60.0
N_FRAMES = 700          # 認識フレーム数 (約23秒@30fps)
PLACE_EVERY = 40        # 何フレームごとに1手置くか
FIRE_FRAME = 400
CHAIN_FRAMES = 60


def _empty() -> np.ndarray:
    return np.zeros((13, 6), dtype=np.int8)


def _chain_board() -> np.ndarray:
    """縦4個赤 + 上に青2個が重なる1段消し (2連鎖)。"""
    g = _empty()
    g[9:13, 0] = 1
    g[8, 0] = 2
    g[7, 0] = 2
    g[12, 1] = 2
    g[11, 1] = 2
    return g


def _icon(total: int) -> dict:
    return {"side": "x", "icons": [{"icon_type": "small", "count_value": 1}] * total, "total_count": total}


def build_record(root: Path, swap: bool = False, winner_p1: int = 1) -> Path:
    """合成記録を root/record に書く。swap=True なら 1P と 2P の役割を入れ替えて書く。"""
    rec = root / "record"
    rec.mkdir(parents=True, exist_ok=True)
    names = ("1P", "2P")
    if swap:
        names = ("2P", "1P")      # 論理側0を '2P' として書く
    rng = np.random.default_rng(7)
    boards = [_empty(), _empty()]
    lean_rows = []
    frames, events = [], []
    for i in range(N_FRAMES):
        frame = (i + 1) * FPS_STRIDE
        t = frame / FRAME_RATE
        sides_raw = []
        for k in range(2):
            changed = False
            state = "STABLE"
            if i > 0 and i % PLACE_EVERY == (3 if k == 0 else 11) and i < FIRE_FRAME // FPS_STRIDE + 10 or (
                    i > FIRE_FRAME // FPS_STRIDE + CHAIN_FRAMES and i % PLACE_EVERY == (3 if k == 0 else 11)):
                col = int(rng.integers(0, 6))
                row = 12 - int((boards[k][:, col] != 0).sum())
                if row >= 2:
                    boards[k][row, col] = int(rng.integers(1, 5))
                    boards[k][row - 1, col] = int(rng.integers(1, 5))
                    changed = True
            if k == 0 and i == FIRE_FRAME // FPS_STRIDE:
                boards[0] = _chain_board()
                changed = True
            chain_phase = FIRE_FRAME // FPS_STRIDE <= i < FIRE_FRAME // FPS_STRIDE + CHAIN_FRAMES
            if k == 0 and chain_phase:
                state = "CHAIN"
            shown = None if state != "STABLE" else boards[k].tolist()
            sides_raw.append({"state": state, "confirmed_board": shown, "board_provenance": "observed",
                              "score": 1000 * k + i, "tsumo_count": i // PLACE_EVERY, "next_slide_motion": changed,
                              "icon": _icon(0 if k == 0 else (i // 100) % 3), "formula_step": None})
            if changed and state == "STABLE":
                lean_rows.append((k, boards[k].copy(), frame, t))
        for k in range(2):
            name = names[k]
            sides_raw[k]["side"] = name
        ordered = sorted(sides_raw, key=lambda s: s["side"])
        frames.append({"kind": "frame", "frame": frame, "t_sec": t, "available_at_sec": t, "game_idx": 1,
                       "match_active": True, "match_end_locked": False, "sides": ordered,
                       "ledger": {"pending_p1": 0, "pending_p2": 0, "forecast_p1": 0, "forecast_p2": 0,
                                  "net_balance_capped": 0, "pending_p1_uncapped": 0, "pending_p2_uncapped": 0}})
        if i % PLACE_EVERY == (3 if True else 0) and i > 0 and i < 380:
            events.append({"kind": "placement_signal", "side": names[0], "frame": frame, "game_idx": 1, "observed_at_sec": t})
        if i % PLACE_EVERY == 11 and i > 0 and i < 380:
            events.append({"kind": "placement_signal", "side": names[1], "frame": frame, "game_idx": 1, "observed_at_sec": t})
        if i == FIRE_FRAME // FPS_STRIDE:
            events.append({"kind": "fire_start", "side": names[0], "frame": frame, "game_idx": 1, "observed_at_sec": t,
                           "fire_id": "1:A:1", "trigger_sec": t, "pre": None, "steps": [],
                           "chain_before": _chain_board().tolist()})
            events.append({"kind": "formula_step", "side": names[0], "frame": frame, "game_idx": 1, "observed_at_sec": t,
                           "t_sec": t, "left": 40, "right": 1, "erased": 4, "multiplier": 1, "product": 40,
                           "observed_ordinal": 1})
        if i == FIRE_FRAME // FPS_STRIDE + CHAIN_FRAMES:
            events.append({"kind": "fire_complete", "side": names[0], "frame": frame, "game_idx": 1, "observed_at_sec": t,
                           "fire_id": "1:A:1", "trigger_sec": FIRE_FRAME / FRAME_RATE, "pre": None, "steps": [],
                           "consumed_board_delta": None, "consumed_formula": None})
    # frames の sides は常に ('1P','2P') 順
    frames = [dict(f, sides=sorted(f["sides"], key=lambda s: s["side"])) for f in frames]
    with gzip.open(rec / "observations.jsonl.gz", "wt", encoding="utf-8") as f:
        f.write(json.dumps({"kind": "header"}) + "\n")
        for fr in frames:
            f.write(json.dumps(fr) + "\n")
    with (rec / "events.jsonl").open("w", encoding="utf-8") as f:
        for e in events:
            f.write(json.dumps(e) + "\n")
    _write_lean(rec, lean_rows, names, winner_p1, swap)
    return rec


def _write_lean(rec: Path, rows: list, names: tuple[str, str], winner_p1: int, swap: bool) -> None:
    n = len(rows)
    grids = np.stack([r[1] for r in rows]) if n else np.zeros((0, 13, 6), dtype=np.int8)
    side = np.array([names[r[0]] for r in rows])
    # 論理側0が勝者かどうか: swap=False なら 1P(=論理0) が winner_p1
    logical_win = np.array([1.0 if (r[0] == 0) == bool(winner_p1) else 0.0 for r in rows], dtype=np.float32)
    q = np.tile(np.array([1, 2, 3, 4], dtype=np.int8), (n, 1))
    np.savez(rec / "lean.npz", grids=grids, side=side, game_idx=np.ones(n, dtype=np.int32),
             frame_idx=np.array([r[2] for r in rows], dtype=np.int32), t_sec=np.array([r[3] for r in rows], dtype=np.float32),
             won=logical_win, next1_a=q[:, 0], next1_b=q[:, 1], dnext_a=q[:, 2], dnext_b=q[:, 3])
