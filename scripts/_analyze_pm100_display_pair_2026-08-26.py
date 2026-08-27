"""Gate 4用: settled行ではなく密な実表示dumpのOFF/ONを比較する。"""
from __future__ import annotations

import sys
import csv
from pathlib import Path

import numpy as np

STICK_TH = 99.5
EVEN_TH = 3.0
SWING_PT = 150.0
SWING_SEC = 1.0
MAX_DENSE_GAP_SEC = 0.05
TRUTH_TAIL_SEC = 2.0
FINAL_DISPLAY_TAIL_SEC = 1.0
FINAL_WEAK_TH = 30.0

Segment = tuple[str, dict[str, np.ndarray]]
GameKey = tuple[str, int]


def _load_dir(path: Path) -> list[Segment]:
    rows: list[Segment] = []
    for p in sorted(path.glob("seg*_display.npz")):
        with np.load(p, allow_pickle=False) as data:
            if "display_adv" not in data.files:
                raise ValueError(f"密な実表示dumpではない: {p}")
            rows.append((p.stem[:5], {key: data[key] for key in data.files}))
    if not rows:
        raise ValueError(f"seg*_display.npz が無い: {path}")
    return rows


def _count_swings(t: np.ndarray, adv: np.ndarray, games: np.ndarray) -> int:
    count = 0
    i = 0
    while i < len(t) - 1:
        j = i + 1
        while j < len(t) and games[j] == games[i] and t[j] - t[i] <= SWING_SEC:
            if abs(float(adv[j]) - float(adv[i])) >= SWING_PT:
                count += 1
                i = j
                break
            j += 1
        else:
            i += 1
    return count


def _sticky_flips(
    seg: str, adv: np.ndarray, games: np.ndarray,
) -> tuple[set[GameKey], int]:
    last_sign: dict[GameKey, int] = {}
    flip_games: set[GameKey] = set()
    events = 0
    for value, game in zip(adv, games):
        if abs(float(value)) < STICK_TH:
            continue
        key = (seg, int(game))
        sign = 1 if float(value) > 0.0 else -1
        if key in last_sign and last_sign[key] != sign:
            flip_games.add(key)
            events += 1
        last_sign[key] = sign
    return flip_games, events


def _metrics(segs: list[Segment]) -> dict[str, object]:
    total = stick = wrong = 0.0
    swings = gap_bad = flip_events = 0
    flip_games: set[GameKey] = set()
    for seg, d in segs:
        t = d["t_sec"].astype(float)
        adv = d["display_adv"].astype(float)
        raw = d["adv_raw_last"].astype(float)
        games = d["game_idx"].astype(int)
        dt = np.diff(t, append=t[-1])
        valid_dt = (dt >= 0.0) & (dt <= MAX_DENSE_GAP_SEC)
        gap_bad += int(np.count_nonzero(~valid_dt[:-1]))
        total += float(dt[valid_dt].sum())
        on = np.abs(adv) >= STICK_TH
        stick += float(dt[valid_dt & on].sum())
        reverse = on & (np.abs(raw) >= EVEN_TH) & (np.sign(adv) != np.sign(raw))
        wrong += float(dt[valid_dt & reverse].sum())
        swings += _count_swings(t, adv, games)
        games_with_flip, n_events = _sticky_flips(seg, adv, games)
        flip_games.update(games_with_flip)
        flip_events += n_events
    return {"total": total, "stick": stick, "wrong": wrong,
            "swings": swings, "gap_bad": gap_bad,
            "flip_games": flip_games, "flip_events": flip_events}


def _winner_from_flags(dead1: bool, dead2: bool) -> str | None:
    if dead1 == dead2:
        return None
    return "2P" if dead1 else "1P"


def _merge_truth(truths: dict[GameKey, str], key: GameKey, winner: str) -> None:
    if key in truths and truths[key] != winner:
        raise ValueError(f"勝者根拠が矛盾: {key} {truths[key]} vs {winner}")
    truths[key] = winner


def _truth_from_confirmed(
    seg: str, d: dict[str, np.ndarray],
) -> dict[GameKey, str]:
    if "is_dead1_confirmed" not in d or "is_dead2_confirmed" not in d:
        return {}
    truths: dict[GameKey, str] = {}
    games = d["game_idx"].astype(int)
    for game in np.unique(games):
        mask = games == game
        winner = _winner_from_flags(
            bool(np.any(d["is_dead1_confirmed"][mask])),
            bool(np.any(d["is_dead2_confirmed"][mask])),
        )
        if winner is not None and int(game) > 0:
            truths[(seg, int(game) - 1)] = winner
    return truths


def _truth_from_dead_tail(
    seg: str, d: dict[str, np.ndarray],
) -> dict[GameKey, str]:
    if "is_dead1" not in d or "is_dead2" not in d:
        return {}
    truths: dict[GameKey, str] = {}
    t = d["t_sec"].astype(float)
    games = d["game_idx"].astype(int)
    for game in np.unique(games):
        mask = games == game
        tail = mask & (t >= float(t[mask][-1]) - TRUTH_TAIL_SEC)
        winner = _winner_from_flags(
            bool(np.any(d["is_dead1"][tail])), bool(np.any(d["is_dead2"][tail])))
        if winner is not None:
            truths[(seg, int(game))] = winner
    return truths


def _load_truths(path: Path) -> dict[GameKey, str]:
    truths: dict[GameKey, str] = {}
    required = {
        "t_sec", "game_idx", "is_dead1", "is_dead2",
        "is_dead1_confirmed", "is_dead2_confirmed",
    }
    for p in sorted(path.glob("seg*_timeline.npz")):
        with np.load(p, allow_pickle=False) as data:
            # drivers_top3_names等のobject列は勝者根拠に不要。全列ロードすると
            # allow_pickle=Falseで停止するため、必要なプリミティブ列だけ読む。
            d = {key: data[key] for key in required if key in data.files}
        seg = p.stem[:5]
        sources = (_truth_from_confirmed(seg, d), _truth_from_dead_tail(seg, d))
        for source in sources:
            for key, winner in source.items():
                _merge_truth(truths, key, winner)
    return truths


def _load_panel_truth(path: Path) -> dict[GameKey, str]:
    """WIN★パネル差分のTSVを、Gate 4の正解データとして読む。"""
    truths: dict[GameKey, str] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            winner = row["winner"]
            if winner not in ("1P", "2P"):
                continue
            key = (row["segment"], int(row["game_idx"]))
            _merge_truth(truths, key, winner)
    if not truths:
        raise ValueError(f"WIN★勝者根拠が0件: {path}")
    return truths


def _final_displays(segs: list[Segment]) -> dict[GameKey, float]:
    out: dict[GameKey, float] = {}
    for seg, d in segs:
        t = d["t_sec"].astype(float)
        adv = d["display_adv"].astype(float)
        games = d["game_idx"].astype(int)
        for game in np.unique(games):
            mask = games == game
            end = float(t[mask][-1])
            tail = mask & (t >= end - FINAL_DISPLAY_TAIL_SEC)
            out[(seg, int(game))] = float(np.mean(adv[tail]))
    return out


def _assert_paired(
    off: list[Segment],
    on: list[Segment],
) -> None:
    if [x[0] for x in off] != [x[0] for x in on]:
        raise ValueError("OFF/ONの区間集合が一致しない")
    for (seg, a), (_, b) in zip(off, on):
        for key in ("t_sec", "game_idx", "state1", "state2", "score1", "score2"):
            if not np.array_equal(a[key], b[key]):
                raise ValueError(f"{seg}: OFF/ON共通キー {key} が不一致")


def _print(label: str, result: dict[str, object]) -> None:
    total = float(result["total"])
    print(f"{label}: 実表示時間 {total:.1f}s / ±100張り付き "
          f"{float(result['stick']):.1f}s ({100*float(result['stick'])/total:.2f}%) / "
          f"生評価と逆符号 {float(result['wrong']):.1f}s "
          f"({100*float(result['wrong'])/total:.2f}%) / "
          f"反転試合 {len(result['flip_games'])} (延べ{result['flip_events']}) / "
          f"急変 {result['swings']} / "
          f"密表示gap異常 {result['gap_bad']}")


def _print_final_verdicts(
    off: list[Segment], on: list[Segment], truths: dict[GameKey, str],
) -> None:
    final_off, final_on = _final_displays(off), _final_displays(on)
    keys = sorted(set(final_off) & set(final_on))
    known = [key for key in keys if key in truths]
    wrong_off = wrong_on = weakened = 0
    for key in known:
        want = 1.0 if truths[key] == "1P" else -1.0
        wrong_off += int(np.sign(final_off[key]) != want)
        wrong_on += int(np.sign(final_on[key]) != want)
        weakened += int(
            abs(final_off[key]) >= FINAL_WEAK_TH
            and abs(final_on[key]) < FINAL_WEAK_TH)
    changed = sum(
        np.sign(final_off[key]) != np.sign(final_on[key]) for key in keys)
    print(f"決着方向: 全試合{len(keys)} / 勝者根拠あり{len(known)} / "
          f"逆方向 OFF {wrong_off} / ON {wrong_on} / 真の致死弱化 {weakened} / "
          f"OFF-ON符号変化 {changed}")


def main() -> int:
    off_dir, on_dir = Path(sys.argv[1]), Path(sys.argv[2])
    off = _load_dir(off_dir)
    on = _load_dir(on_dir)
    _assert_paired(off, on)
    if len(sys.argv) >= 4:
        truths_on = _load_panel_truth(Path(sys.argv[3]))
    else:
        truths_off, truths_on = _load_truths(off_dir), _load_truths(on_dir)
        if truths_off != truths_on:
            raise ValueError("OFF/ONで勝者根拠が一致しない")
    m_off, m_on = _metrics(off), _metrics(on)
    _print("OFF", m_off)
    _print("ON ", m_on)
    _print_final_verdicts(off, on, truths_on)
    return int(bool(m_off["gap_bad"] or m_on["gap_bad"]))


if __name__ == "__main__":
    raise SystemExit(main())
