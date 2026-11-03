"""B3 学習行の生成器。

更新契機 (各側の設置確定・式の新段・発火系・着弾・アイコン補正値の変化・境界・固定 heartbeat) ごとに、
その時点で available_at <= t の情報だけで1行を作る。教師は最終勝敗 (won_p1) で、特徴生成には渡さない。
保持秒で時間加重し試合内重み合計1、1P/2P 交換複製は重みを二分する。
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

from src.b3_column_spec import COLUMN_NAMES, COLUMN_SPECS, SPEC_BY_NAME, swap_sides
from src.b3_features import board_is_dead, build_features
from src.b3_source import iter_frames, load_events, load_lean, obs_game_ranges, winners_by_overlap
from src.b3_state import GameState
from src.b3_timing import TimingTable

HEARTBEAT_SEC = 0.5
HELD_CAP_SEC = 2.0
FRAME_DT_DEFAULT = 1.0 / 30.0
ICON_CHANGE_EPS = 0.5
TRIGGER_BITS = {"placement": 1, "formula": 2, "fire": 4, "landing": 8, "icon": 16, "boundary": 32,
                "heartbeat": 64, "board": 128}
AV_EPS = 1e-4   # lean.npz の t_sec は float32 (約1e-5秒の丸め)。フレーム周期16msより十分小さい


class LeakError(RuntimeError):
    """行の元データ available_at が評価時点を超えた。"""


def _av_max(row: dict) -> float:
    vals = [v for k, v in row.items() if k.startswith("av_") and k != "av_max" and v is not None and np.isfinite(v)]
    return max(vals) if vals else float("nan")


def generate_rows(record: Path, video_id: str, table: TimingTable, max_frame: Optional[int] = None,
                  winners: Optional[dict[int, Optional[int]]] = None, require_winner: bool = True,
                  heartbeat: float = HEARTBEAT_SEC,
                  progress: Optional[Callable[[int, int], None]] = None,
                  match_mapping: Optional[dict[int, dict]] = None) -> tuple[list[dict], dict]:
    """原向きの行 (重み未確定) と集計情報を返す。max_frame を指定すると未来を切り落とした再生になる。"""
    lean = load_lean(record, max_frame)
    events = load_events(record, max_frame)
    winners, mapping = _match_metadata(record, winners, match_mapping)
    gs = GameState(lean, match_mapping=mapping)
    rows: list[dict] = []
    info = {"frames": 0, "frames_in_match": 0, "rows": 0, "skipped_no_winner_frames": 0, "leak_violations": 0}
    info.update(excluded_outside_match_rows=0, excluded_outside_match_rows_by_game={},
                excluded_outside_match_frames=0, stable_death={})
    placed = _placement_frames(events)
    ev_i, last_t, boundary, prev_icon, prev_land = 0, -1e18, False, [None, None], [0, 0]
    for d in iter_frames(record, max_frame):
        info["frames"] += 1
        if not _inside_match(float(d["t_sec"]), mapping.get(int(d["game_idx"]), {})):
            info["excluded_outside_match_frames"] += 1
        boundary |= gs.update_frame(d, placed.get(d["frame"], ()))
        if boundary and gs.in_match:
            last_t, prev_icon, prev_land = -1e18, [None, None], [0, 0]
        trig = 0
        while ev_i < len(events) and events[ev_i]["frame"] <= d["frame"]:
            trig |= gs.apply_event(events[ev_i])
            ev_i += 1
        if not gs.in_match:
            continue
        info["frames_in_match"] += 1
        if require_winner and winners.get(gs.game_idx) is None:
            info["skipped_no_winner_frames"] += 1
            continue
        _track_stable_death(info["stable_death"], d)
        trig |= _frame_triggers(gs, boundary, last_t, heartbeat, prev_icon, prev_land)
        boundary = False
        if trig == 0 or all(s.board is None for s in gs.sides):
            continue
        last_t = gs.t
        row = _make_row(gs, table, video_id, trig, winners.get(gs.game_idx))
        if row["av_max"] > gs.t + AV_EPS:
            info["leak_violations"] += 1
        rows.append(row)
    info["rows"] = len(rows)
    return rows, info


def _match_metadata(record: Path, winners: Optional[dict], mapping: Optional[dict]) -> tuple[dict, dict]:
    """選択範囲は全記録から決め、max_frame による特徴のprefix検査と分離する。"""
    if winners is None or mapping is None:
        mapped_winners, mapped_ranges = winners_by_overlap(load_lean(record), obs_game_ranges(record))
        winners = mapped_winners if winners is None else winners
        mapping = mapped_ranges if mapping is None else mapping
    return winners, mapping


def _placement_frames(events: list[dict]) -> dict[int, tuple[str, ...]]:
    placed: dict[int, tuple[str, ...]] = {}
    for e in events:
        if e["kind"] == "placement_signal":
            placed[e["frame"]] = placed.get(e["frame"], ()) + (e["side"],)
    return placed


def _inside_match(t: float, span: dict) -> bool:
    """状態更新と行出力の対象になる lean 試合範囲。"""
    return span.get("start_sec") is not None and span["start_sec"] - AV_EPS <= t <= span["end_sec"] + AV_EPS


def _count_excluded(info: dict, game: int) -> None:
    """従来なら出力した原向き行を動画・試合別に数える。"""
    info["excluded_outside_match_rows"] += 1
    counts = info["excluded_outside_match_rows_by_game"]
    counts[str(game)] = counts.get(str(game), 0) + 1


def _track_stable_death(last: dict, frame: dict) -> None:
    """側別の最後のSTABLE確定盤面を保持し、連鎖開始後の古い盤面は失効させる。"""
    game = last.setdefault(str(frame["game_idx"]), {})
    for side, raw in zip(("p1", "p2"), frame["sides"]):
        if raw["state"] in ("CHAIN", "GRAVITY_SETTLE"):
            game[side] = None
        elif raw["state"] == "STABLE" and raw.get("confirmed_board") is not None:
            board = np.asarray(raw["confirmed_board"], dtype=np.int8)
            game[side] = {"dead": int(board_is_dead(board.tobytes())), "frame": frame["frame"], "t_sec": frame["t_sec"]}


def _frame_triggers(gs: GameState, boundary: bool, last_t: float, hb: float, prev_icon: list, prev_land: list) -> int:
    trig = TRIGGER_BITS["boundary"] if boundary else 0
    for i, st in enumerate(gs.sides):
        if st.board_changed:
            trig |= TRIGGER_BITS["board"]
        if st.land_count != prev_land[i]:
            trig |= TRIGGER_BITS["landing"]
            prev_land[i] = st.land_count
        c = st.icon.corrected_total()
        if np.isfinite(c) and (prev_icon[i] is None or abs(c - prev_icon[i]) >= ICON_CHANGE_EPS):
            if prev_icon[i] is not None:
                trig |= TRIGGER_BITS["icon"]
            prev_icon[i] = c
    if gs.t - last_t >= hb:
        trig |= TRIGGER_BITS["heartbeat"]
    return trig


def _make_row(gs: GameState, table: TimingTable, video_id: str, trig: int, winner: Optional[int]) -> dict:
    row = build_features(gs, table)
    row.update(video_id=video_id, game_idx=gs.game_idx, frame_idx=gs.frame, t_sec=gs.t, orientation=0,
               trigger_mask=trig, av_frame=gs.av_frame, held_sec=float("nan"), weight=float("nan"),
               won_p1=float("nan") if winner is None else float(winner))
    row["row_key"] = f"{video_id}|g{gs.game_idx}|f{gs.frame}|o0"
    row["av_max"] = _av_max(row)
    return row


def finalize_weights(rows: list[dict], frame_dt: float = FRAME_DT_DEFAULT) -> None:
    """保持秒 held_sec と行重み (試合内合計1, 原向き0.5) を付ける。未来の行に依存するため特徴ではない。"""
    by_game: dict[int, list[dict]] = {}
    for r in rows:
        by_game.setdefault(r["game_idx"], []).append(r)
    for g_rows in by_game.values():
        g_rows.sort(key=lambda r: r["frame_idx"])
        for a, b in zip(g_rows, g_rows[1:]):
            a["held_sec"] = min(b["t_sec"] - a["t_sec"], HELD_CAP_SEC)
        g_rows[-1]["held_sec"] = frame_dt
        total = sum(r["held_sec"] for r in g_rows)
        for r in g_rows:
            r["weight"] = r["held_sec"] / total * 0.5


def mirror_row(row: dict) -> dict:
    """1P/2P 交換複製 (台帳の kind 規則)。重みは原向きと同じ (各0.5)。"""
    out = swap_sides(row)
    out["orientation"] = 1 - int(row["orientation"])
    head, _o = row["row_key"].rsplit("|o", 1)
    out["row_key"] = f"{head}|o{out['orientation']}"
    return out


def with_mirrors(rows: list[dict]) -> list[dict]:
    return rows + [mirror_row(r) for r in rows]


_DTYPES = {"f8": "float64", "f4": "float32", "i4": "Int32", "i1": "Int8", "str": "object"}


def to_frame(rows: list[dict]) -> pd.DataFrame:
    """台帳の列順・dtype の DataFrame。台帳に無い/不足の列は例外 (黙って0埋めしない)。"""
    if not rows:
        return pd.DataFrame({c.name: pd.Series(dtype=_DTYPES[c.dtype]) for c in COLUMN_SPECS})
    missing = set(COLUMN_NAMES) - set(rows[0])
    extra = set(rows[0]) - set(COLUMN_NAMES)
    if missing or extra:
        raise ValueError(f"台帳と行の列が不一致: 不足{sorted(missing)[:5]} 余分{sorted(extra)[:5]}")
    df = pd.DataFrame(rows)[COLUMN_NAMES]
    for c in COLUMN_SPECS:
        dt = _DTYPES[c.dtype]
        df[c.name] = df[c.name].astype(dt) if c.dtype != "i1" and c.dtype != "i4" else pd.array(
            df[c.name].where(df[c.name].notna(), pd.NA), dtype=dt)
    return df


def label_vs_death_audit(rows: list[dict], stable_death: Optional[dict] = None) -> dict:
    """最後のSTABLE確定盤面で教師を監査。両側窒息・連鎖前凍結・観測無しは判定不能。

    試合番号照合では検出できない1試合ずれを捕まえる (独立検収 P0)。母数=判定可能試合数。
    """
    last: dict[int, dict] = {}
    for r in rows:
        if r["game_idx"] not in last or r["frame_idx"] > last[r["game_idx"]]["frame_idx"]:
            last[r["game_idx"]] = r
    judged = bad = 0
    for g, r in last.items():
        pair = (stable_death or {}).get(str(g), {})
        if not pair.get("p1") or not pair.get("p2"):
            continue
        d1, d2, w = pair["p1"]["dead"], pair["p2"]["dead"], r["won_p1"]
        if not (np.isfinite(d1) and np.isfinite(d2)) or d1 == d2 or not np.isfinite(w):
            continue
        judged += 1
        bad += int((d1 == 1.0) == (w == 1.0))     # 1Pが窒息なのに1P勝ち (または逆)
    return {"games": len(last), "judged": judged, "mismatch": bad, "undetermined": len(last) - judged}
