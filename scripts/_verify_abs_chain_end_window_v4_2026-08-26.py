"""Codex v4 決着ホールドの密な実表示ベース検収。"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path("data/verify/abs_chain_end_2026-08-26")
FRAME_GAP_MAX_SEC = 0.05
# 連鎖1段の実測最大1.4秒 (`src/chain_count_ocr.py`) に量子化余裕0.1秒。
EARLY_SCORE_WINDOW_SEC = 1.5
STEP_SCORE_MIN = 40


def _eq(a: np.ndarray, b: np.ndarray) -> bool:
    if a.dtype.kind in "fc":
        return np.array_equal(np.nan_to_num(a, nan=-9e99), np.nan_to_num(b, nan=-9e99))
    return np.array_equal(a, b)


def _active_runs(d) -> list[tuple[float, float]]:
    t = d["t_sec"]
    active = d["resolved_active"].astype(bool)
    runs: list[tuple[float, float]] = []
    start: int | None = None
    for i, on in enumerate(active):
        if on and start is None:
            start = i
        if start is not None and (not on or i == len(active) - 1):
            end = i if on else i - 1
            runs.append((float(t[start]), float(t[end])))
            start = None
    return runs


def _score_suspects(d, runs: list[tuple[float, float]]) -> list[tuple[float, int]]:
    t = d["t_sec"]
    scores = (d["score1"].astype(np.int64), d["score2"].astype(np.int64))
    out: list[tuple[float, int]] = []
    for _start, end in runs:
        m = (t >= end) & (t <= end + EARLY_SCORE_WINDOW_SEC)
        max_step = max((int(np.diff(s[m]).max()) if int(m.sum()) >= 2 else 0)
                       for s in scores)
        if max_step >= STEP_SCORE_MIN:
            out.append((end, max_step))
    return out


def _score_tail_delays(d, runs: list[tuple[float, float]]) -> list[tuple[float, float | None]]:
    """各ホールドの最終40点段から解除までの秒数を返す。"""
    t = d["t_sec"]
    scores = (d["score1"].astype(np.int64), d["score2"].astype(np.int64))
    out: list[tuple[float, float | None]] = []
    for start, end in runs:
        step_times: list[float] = []
        for score in scores:
            ids = np.flatnonzero((t >= start) & (t <= end))
            if len(ids) < 2:
                continue
            delta = np.diff(score[ids])
            step_times.extend(float(t[ids[j + 1]])
                              for j in np.flatnonzero(delta >= STEP_SCORE_MIN))
        last = max(step_times, default=None)
        out.append((end, None if last is None else end - last))
    return out


def _print_longest_run_transitions(d, runs: list[tuple[float, float]]) -> None:
    """最長ホールド内の state/score 変化を診断表示する。"""
    if not runs:
        return
    start, end = max(runs, key=lambda run: run[1] - run[0])
    t = d["t_sec"]
    ids = np.flatnonzero((t >= start) & (t <= end + 1.0))
    print(f"最長hold遷移 {start:.3f}-{end:.3f}:")
    prev: tuple | None = None
    changes: list[tuple[float, tuple]] = []
    for i in ids:
        cur = (str(d["state1"][i]), str(d["state2"][i]),
               int(d["score1"][i]), int(d["score2"][i]),
               bool(d["resolved_active"][i]))
        if cur == prev:
            continue
        changes.append((float(t[i]), cur))
        prev = cur
    selected = changes if len(changes) <= 100 else changes[:50] + changes[-50:]
    for when, cur in selected:
        print(f"  t={when:.3f} state={cur[0]}/{cur[1]} "
              f"score={cur[2]}/{cur[3]} active={int(cur[4])}")
    if len(changes) > 100:
        print(f"  ... 中央 {len(changes) - 100}遷移省略")


def main() -> int:
    tag = sys.argv[1] if len(sys.argv) > 1 else "v4"
    log_path = Path(f"logs/abs_chain_end_2026-08-26/{tag}_on.log")
    off_path = ROOT / f"window_{tag}_off_display.npz"
    on_path = ROOT / f"window_{tag}_on_display.npz"
    if not off_path.exists() or not on_path.exists():
        print("v4のOFF/ON密表示dumpが未完了")
        return 1
    off = np.load(off_path, allow_pickle=False)
    on = np.load(on_path, allow_pickle=False)
    bad = [key for key in ("t_sec", "game_idx", "state1", "state2",
                            "score1", "score2") if not _eq(off[key], on[key])]
    print(f"OFF/ON 認識共通6キー: 不一致 {len(bad)}/6 {bad}")
    off_sparse_bad: list[str] = []
    ref_off_path = ROOT / "window_v4_off.npz"
    new_off_path = ROOT / f"window_{tag}_off.npz"
    if tag != "v4" and ref_off_path.exists() and new_off_path.exists():
        ref = np.load(ref_off_path, allow_pickle=True)
        new = np.load(new_off_path, allow_pickle=True)
        shared = sorted(set(ref.files) & set(new.files))
        off_sparse_bad = [key for key in shared if not _eq(ref[key], new[key])]
        print(f"既定OFF sparse v4→{tag}: 不一致 {len(off_sparse_bad)}/{len(shared)} "
              f"{off_sparse_bad}")
    dt = np.diff(on["t_sec"])
    gap_bad = int(np.count_nonzero(dt > FRAME_GAP_MAX_SEC))
    print(f"密表示: {len(on['t_sec'])}行 / >{FRAME_GAP_MAX_SEC:.2f}秒 gap "
          f"{gap_bad}/{len(dt)} / 最大gap {float(dt.max()):.4f}秒")
    runs = _active_runs(on)
    durations = [b - a for a, b in runs]
    print(f"ON hold runs {len(runs)}本: "
          + ", ".join(f"{a:.3f}-{b:.3f} ({b-a:.2f}s)" for a, b in runs))
    print(f"最大hold {max(durations, default=0.0):.2f}秒")
    tails = _score_tail_delays(on, runs)
    print("最終40点段→解除: " + ", ".join(
        f"end={end:.3f} delay={'段なし' if delay is None else f'{delay:.3f}s'}"
        for end, delay in tails))
    _print_longest_run_transitions(on, runs)
    near_old = [(a, b) for a, b in runs if abs(b - 1713.033) <= 0.5]
    print(f"書出窓内の旧早期解除 t=1713.033±0.5: {len(near_old)}/{len(runs)}")
    suspects = _score_suspects(on, runs)
    print(f"解除後{EARLY_SCORE_WINDOW_SEC:.1f}秒以内の40点以上増分: "
          f"{len(suspects)}/{len(runs)} {suspects}")
    close_rearms = sum(b0 - a1 <= 0.5 for (_a0, a1), (b0, _b1) in zip(runs, runs[1:]))
    print(f"解除後0.5秒以内の再武装: {close_rearms}/{max(len(runs)-1, 0)}")
    old_release_in_log = False
    chain_state_releases = 0
    if log_path.exists():
        text = log_path.read_text(encoding="utf-8", errors="replace")
        m = re.search(r"保持セッション\s+(\d+).*?絶対律で解除\s+(\d+)/(\d+)", text)
        print("ログ母数: " + (m.group(0) if m else "取得不能"))
        early = re.search(r"解除時にCHAIN継続\s+(\d+)/(\d+)", text)
        if early:
            chain_state_releases = int(early.group(1))
            print(f"解除時にCHAIN継続: {early.group(1)}/{early.group(2)}")
        old_release_in_log = "(1713.033," in text
        print(f"warmupを含むログの旧早期解除 t=1713.033: "
              f"{'再発' if old_release_in_log else '0件'}")
    # state=CHAIN は認識表示の残留があり物理終端とは一致しない。合否は
    # 実得点段の前後関係 (`suspects`) で判定し、state件数は診断値に留める。
    return int(bool(bad or off_sparse_bad or gap_bad or near_old or suspects
                    or close_rearms or old_release_in_log))


if __name__ == "__main__":
    raise SystemExit(main())
