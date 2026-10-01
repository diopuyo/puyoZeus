"""148動画の学習原票の各行へ、補正 queue (因果規則C / 真値選択T) を作って保存する (2026-10-01)。

出力: logs/next_shift_train/queues/<stem>.npz (原票の行順と一対一)
  queue_orig / queue_C / queue_T : (行,4) int8
  method_C : 因果規則の方式 (0=そのまま, 1=繰り上がり補正, 2=無効色)
  chosen_T : (行,2) 真値選択できた枠 (1=表示枠から真値を選んだ)
  entry    : 入口遷移コード (ENTRY_CODES の添字)
  truth_known : (行,2) P_k / P_{k+1} が盤面差分で確定した行
  ok_orig/ok_C/ok_T : (行,2) 真値と一致した枠 (truth_known の行のみ意味あり)
監査: logs/next_shift_train/BUILD.json (状況別の母数・被覆・一致率、打ち切り再生の不一致数)
"""
from __future__ import annotations

from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import numpy as np

from scripts.next_shift_common_20261001 import LEAN_ROOT, raw_queues, read_raw, side_game_indices
from src.next_queue_alignment import (
    KIND_CHAIN, KIND_MULTI, KIND_OJAMA, KIND_PLACE, KIND_SAME, KIND_START, PAIR_CELLS,
    entry_kinds, pair_key, placed_pairs, truth_row_queue, truth_sequence, correct_sequence, valid_pair,
)

OUT = Path("logs/next_shift_train")
ENTRY_CODES = (KIND_START, KIND_PLACE, KIND_CHAIN, KIND_OJAMA, KIND_MULTI, KIND_SAME)
WORKERS = 8
TRUNCATION_SAMPLES = 64
SEED = 0


def truth_flags(exits: list, queue: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(真値既知, 一致) を P_k / P_{k+1} の2枠で返す。"""
    n = len(exits)
    known, ok = np.zeros((n, PAIR_CELLS), bool), np.zeros((n, PAIR_CELLS), bool)
    for r in range(n):
        targets = (exits[r], exits[r + 1] if exits[r] is not None and r + 1 < n else None)
        for slot, truth in enumerate(targets):
            if truth is None:
                continue
            pair = tuple(int(v) for v in queue[r, 2 * slot:2 * slot + 2])
            known[r, slot] = True
            ok[r, slot] = valid_pair(pair) and pair_key(*pair) == truth
    return known, ok


def truncation_mismatches(grids: np.ndarray, queues: np.ndarray, full: np.ndarray,
                          rng: np.random.Generator) -> tuple[int, int]:
    """行 r の T 出力を、表示の読みを r まで・盤面を r+2 までに打ち切って再計算し、全体計算と比べる。"""
    rows = rng.choice(len(grids), size=min(TRUNCATION_SAMPLES, len(grids)), replace=False)
    bad = 0
    for r in rows:
        causal, _ = correct_sequence(grids[:r + 1], queues[:r + 1])
        value, _ = truth_row_queue(int(r), grids[:r + 3], queues[:r + 1], causal[r])
        bad += int(not np.array_equal(value, full[r]))
    return bad, len(rows)


def build_video(path_text: str) -> dict:
    """1動画分の補正 queue と監査カウンタ。"""
    raw = read_raw(path_text)
    queues = raw_queues(raw)
    n = len(queues)
    arrays = dict(queue_orig=queues, queue_C=queues.copy(), queue_T=queues.copy(),
                  method_C=np.zeros(n, np.int8), chosen_T=np.zeros((n, 2), np.int8),
                  entry=np.zeros(n, np.int8), truth_known=np.zeros((n, 2), bool),
                  ok_orig=np.zeros((n, 2), bool), ok_C=np.zeros((n, 2), bool), ok_T=np.zeros((n, 2), bool))
    rng = np.random.default_rng(SEED)
    trunc = [0, 0]
    for ids in side_game_indices(raw):
        grids, q = raw["grids"][ids], queues[ids]
        fixed_t, methods, chosen = truth_sequence(grids, q)
        fixed_c, _ = correct_sequence(grids, q)
        exits = placed_pairs(grids)
        arrays["queue_C"][ids], arrays["queue_T"][ids] = fixed_c, fixed_t
        arrays["method_C"][ids], arrays["chosen_T"][ids] = methods, chosen
        arrays["entry"][ids] = [ENTRY_CODES.index(k) for k in entry_kinds(grids)]
        for name, value in (("orig", q), ("C", fixed_c), ("T", fixed_t)):
            known, ok = truth_flags(exits, value)
            arrays["truth_known"][ids], arrays[f"ok_{name}"][ids] = known, ok
        bad, total = truncation_mismatches(grids, q, fixed_t, rng)
        trunc[0] += bad
        trunc[1] += total
    target = OUT / "queues" / (Path(path_text).stem + ".npz")
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target, **arrays)
    return dict(summary(arrays), truncation_bad=trunc[0], truncation_n=trunc[1])


def summary(a: dict[str, np.ndarray]) -> dict:
    """状況別の母数・被覆・一致数 (合算用に件数で返す)。"""
    c: Counter = Counter()
    for code, kind in enumerate(ENTRY_CODES):
        rows = a["entry"] == code
        for tag in ("all", kind):
            sel = rows if tag == kind else np.ones_like(rows)
            if tag == "all" and code:
                continue
            c[f"{tag}|rows"] += int(sel.sum())
            c[f"{tag}|changed_C"] += int((a["queue_C"][sel] != a["queue_orig"][sel]).any(1).sum())
            c[f"{tag}|changed_T"] += int((a["queue_T"][sel] != a["queue_orig"][sel]).any(1).sum())
            for slot, name in enumerate(("Pk", "Pk1")):
                known = sel & a["truth_known"][:, slot]
                c[f"{tag}|{name}|known"] += int(known.sum())
                c[f"{tag}|{name}|chosen_T"] += int((sel & (a["chosen_T"][:, slot] > 0)).sum())
                for v in ("orig", "C", "T"):
                    c[f"{tag}|{name}|ok_{v}"] += int((known & a[f"ok_{v}"][:, slot]).sum())
    return dict(c)


def main() -> None:
    """全動画を並列処理し、合算した監査票を保存する。"""
    paths = sorted(str(p) for p in LEAN_ROOT.glob("*.npz"))
    total: Counter = Counter()
    with ProcessPoolExecutor(WORKERS) as pool:
        for part in pool.map(build_video, paths):
            total.update(part)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "BUILD.json").write_text(json.dumps(dict(videos=len(paths), counts=dict(sorted(total.items()))),
                                               ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
