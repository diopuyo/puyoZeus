"""B2 取り直し記録の読込 (完了票・attempt 解決・フレーム/イベント/lean)。

読むのは完了票のある attempt の record/ だけ。進行中 attempt・status.json・manager 系には触らない。
"""
from __future__ import annotations

from dataclasses import dataclass
import gzip
import hashlib
import json
import os
from pathlib import Path
from typing import Iterator, Optional

import numpy as np

_B2_REL = "puyo_analyzer/wt_b2/data/verify/recollect_2026-10-03"
B2_ROOT_DEFAULT = Path("D:/" + _B2_REL) if os.name == "nt" else Path("/mnt/d/" + _B2_REL)
EXCLUDED_VIDEOS: frozenset[str] = frozenset()
MATCH_END_GRACE_SEC = 3.0  # lean 最終確定盤面から決着演出までの余裕


@dataclass(frozen=True)
class RecordPaths:
    """1動画の採用 attempt。"""
    video_id: str
    attempt: str
    record: Path
    receipt: Path


def list_complete_videos(root: Path = B2_ROOT_DEFAULT) -> list[str]:
    """完了票があり status=complete の動画ID (除外動画を除く)。"""
    out = []
    for f in sorted((root / "receipts").glob("*.json")):
        if f.stem in EXCLUDED_VIDEOS:
            continue
        if json.loads(f.read_text(encoding="utf-8")).get("status") == "complete":
            out.append(f.stem)
    return out


def resolve_attempt(video_id: str, root: Path = B2_ROOT_DEFAULT) -> RecordPaths:
    """完了票の attempt を採用し、superseded に載る旧試行は拒否する。"""
    receipt = root / "receipts" / f"{video_id}.json"
    data = json.loads(receipt.read_text(encoding="utf-8"))
    if data.get("status") != "complete":
        raise ValueError(f"{video_id}: 完了票が complete でない")
    attempt = Path(data["attempt"]).name
    sup_file = root / "superseded.json"
    sup = json.loads(sup_file.read_text(encoding="utf-8")) if sup_file.exists() else {}
    if attempt in sup.get(video_id, {}).get("attempts", []):
        raise ValueError(f"{video_id}: 採用 attempt {attempt} は superseded")
    record = root / "attempts" / video_id / attempt / "record"
    for need in ("observations.jsonl.gz", "events.jsonl", "lean.npz"):
        if not (record / need).exists():
            raise FileNotFoundError(f"{record / need}")
    return RecordPaths(video_id, attempt, record, receipt)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def iter_frames(record: Path, max_frame: Optional[int] = None) -> Iterator[dict]:
    """観測フレームを時刻順に返す (kind=frame のみ)。max_frame 超過で打切り。"""
    with gzip.open(record / "observations.jsonl.gz", "rt", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if d.get("kind") != "frame":
                continue
            if max_frame is not None and d["frame"] > max_frame:
                return
            yield d


def load_events(record: Path, max_frame: Optional[int] = None) -> list[dict]:
    """events.jsonl を frame 昇順 (同 frame は出現順) で返す。"""
    events = []
    with (record / "events.jsonl").open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            e = json.loads(line)
            if max_frame is not None and e["frame"] > max_frame:
                continue
            e["_seq"] = i
            events.append(e)
    events.sort(key=lambda e: (e["frame"], e["_seq"]))
    return events


@dataclass(frozen=True)
class LeanTable:
    """lean.npz (STABLE 確定盤面の側別行)。frame_idx<=max_frame に制限できる。"""
    grids: np.ndarray
    side: np.ndarray
    game: np.ndarray
    frame: np.ndarray
    t_sec: np.ndarray
    won: np.ndarray
    queue: np.ndarray   # (N,4) next1_a,next1_b,dnext_a,dnext_b

    def rows_of(self, side: str) -> np.ndarray:
        """側の行 index を frame 昇順で返す。"""
        idx = np.where(self.side == side)[0]
        return idx[np.argsort(self.frame[idx], kind="stable")]


def load_lean(record: Path, max_frame: Optional[int] = None) -> LeanTable:
    z = np.load(record / "lean.npz", allow_pickle=False)
    keep = np.ones(len(z["frame_idx"]), dtype=bool) if max_frame is None else z["frame_idx"] <= max_frame
    queue = np.stack([z["next1_a"], z["next1_b"], z["dnext_a"], z["dnext_b"]], axis=1)[keep]
    return LeanTable(z["grids"][keep], z["side"][keep], z["game_idx"][keep], z["frame_idx"][keep],
                     z["t_sec"][keep].astype(np.float64), z["won"][keep], queue)


def game_winners(lean: LeanTable) -> dict[int, Optional[int]]:
    """試合ごとの最終勝者 (1=1P, 0=2P, None=不明/矛盾)。lean の won 列のみ。"""
    out: dict[int, Optional[int]] = {}
    for g in np.unique(lean.game):
        votes = set()
        for side, sign in (("1P", 1), ("2P", 0)):
            m = (lean.game == g) & (lean.side == side) & ~np.isnan(lean.won)
            for v in np.unique(lean.won[m]):
                votes.add(int(v) if sign else 1 - int(v))
        out[int(g)] = votes.pop() if len(votes) == 1 else None
    return out


def obs_game_ranges(record: Path, max_frame: Optional[int] = None) -> dict[int, tuple[float, float]]:
    """観測側の試合番号ごとの (最初, 最後) 時刻。教師ラベルの対応づけ専用 (特徴には使わない)。"""
    out: dict[int, list[float]] = {}
    for d in iter_frames(record, max_frame):
        r = out.setdefault(int(d["game_idx"]), [d["t_sec"], d["t_sec"]])
        r[1] = d["t_sec"]
    return {g: (a, b) for g, (a, b) in out.items()}


def winners_by_overlap(lean: LeanTable, ranges: dict[int, tuple[float, float]]) -> tuple[dict[int, Optional[int]], dict[int, dict]]:
    """観測試合 g の勝者 = 時間帯の重なりが最大の lean 試合の勝者 (番号は使わない)。

    lean と観測で試合番号が1ずれる (独立検収 P0)。重なり率 (観測試合長に対する) が0.5未満なら None。
    """
    by_lean = game_winners(lean)
    spans = {int(g): (float(lean.t_sec[lean.game == g].min()), float(lean.t_sec[lean.game == g].max()))
             for g in np.unique(lean.game)}
    out: dict[int, Optional[int]] = {}
    info: dict[int, dict] = {}
    for g, (a, b) in ranges.items():
        best, best_ov = None, 0.0
        for lg, (c, d) in spans.items():
            ov = max(0.0, min(b, d) - max(a, c))
            if ov > best_ov:
                best, best_ov = lg, ov
        lspan = spans[best][1] - spans[best][0] if best is not None else 0.0
        frac = best_ov / max(min(b - a, lspan), 1e-9)   # lean は STABLE 行だけで短いので短い方を分母にする
        ok = best is not None and frac >= 0.5
        out[g] = by_lean.get(best) if ok else None
        info[g] = {"lean_game": best, "overlap_frac": round(frac, 3),
                   "start_sec": spans[best][0] if ok else None,
                   "end_sec": spans[best][1] + MATCH_END_GRACE_SEC if ok else None}
    return out, info
