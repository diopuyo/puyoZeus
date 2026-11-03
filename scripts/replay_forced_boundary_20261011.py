"""強制境界のデバウンス枠分離 (2026-10-11) の再生台と検証。

評価ループの試合境界は、得点リセット (R1) と入力世代切替の強制境界 (R2) の2経路で進む。
R2が R1 用の5秒デバウンス枠を消費して本物のR1を拒否した欠陥 (soak game11) を、保存済みの
評価行 (spool/evaluations.pickle) から再生して確かめる。

台の本体 (run_mini_generate) は、本番と同じ adapt_loop で注入文を挿した最小generate を実行する。
本物の accept_formal_boundary / _detect_score_reset / update_score_reset_latch をそのまま使うので、
境界の通り方は本番の評価ループと同じ論理になる (周辺の重い処理だけを省いている)。

使い方 (リポジトリ直下から):
    python -m scripts.replay_forced_boundary_20261011 <記録ディレクトリ>...
各記録ディレクトリは output/spool/evaluations.pickle を持つこと。
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import pickle
from types import SimpleNamespace
from typing import Any

# 通知の間隔がこれを超えて空いた直後の通知は、入力世代切替 (強制境界) を消費した通知とみなす。
# 検証中は評価が公開されないため、通常の通知間隔 (30Hz/2Hz) より十分大きい値を置く。
FORCED_GAP_SEC = 1.5
LONG_DEATH_RUN_SEC = 5.0  # 前試合から持ち越したと疑う死亡確定の連続長
EPOCH_MATCH_MARGIN_SEC = 1.0  # 入力世代の変化時刻と強制境界通知の照合に許す余裕

# 本物の generate の境界ブロック (visualize_advantage_overlay.py の `if _formal_boundary:`) と同形の最小generate。
# adapt_loop が探す文 (r = pipe.update( / snap = _drive_ojama / if fi < write_frame: / waiting =) を本物と同じ順に置く。
MINI_GENERATE = '''
def generate(packets):
    game_idx = 0
    _last_formal_boundary_t = None
    _score_reset_latched = False
    prev_score1 = prev_score2 = None
    write_frame = 0
    pipe = cap = fps = start_frame = n = stride = None
    counter_tracker = event_overlay = resolved_active = disp_p1 = disp_adv = None
    resolved_tracker = _RESOLVED
    trace = []
    for fi in packets:
        r = pipe.update(fi)
        _reset_now = _detect_score_reset(r.p1.score, r.p2.score, prev_score1, prev_score2)
        _formal_boundary = accept_formal_boundary(
            reset_now=_reset_now, latched=_score_reset_latched,
            t_sec=t, last_formal_t=_last_formal_boundary_t)
        if _formal_boundary:
            _ending_game_idx = game_idx
            _last_formal_boundary_t = t
            game_idx += 1
        _score_reset_latched = update_score_reset_latch(
            _score_reset_latched, _reset_now, r.p1.score, r.p2.score)
        prev_score1, prev_score2 = r.p1.score, r.p2.score
        trace.append((fi, game_idx, _last_formal_boundary_t))
        snap = _drive_ojama()
        if fi < write_frame:
            continue
        waiting = False
    return trace
'''

Event = tuple[float, tuple[int | None, int | None], bool]   # (t_sec, (score1, score2), 強制境界か)
Trace = list[tuple[int, int, Any]]                           # (frame, game_idx, デバウンス枠の時刻)


class FakeBridge:
    """通知を順に流し、強制境界は指定通知で1回だけ立てる (consume_input_boundary は一度きり)。"""

    def __init__(self, packets: list[SimpleNamespace], forced_frames: set[int]) -> None:
        self.packets, self.forced = packets, set(forced_frames)
        self.current: SimpleNamespace | None = None
        self.consumed: list[int] = []
        from src.phase_j.live_boundary import RecoveryBoundary
        self.recovery_boundary = RecoveryBoundary()

    def boundary_increment(self, forced: bool, scored: bool, now: float) -> int:
        return self.recovery_boundary.increment(forced, scored, now)

    def observations(self, *args: Any) -> Any:
        for packet in self.packets:
            self.current = packet
            yield packet

    def consume_input_boundary(self) -> bool:
        assert self.current is not None
        self.consumed.append(self.current.frame)
        return self.current.frame in self.forced

    def counter_context(self, *args: Any) -> None:
        pass

    def mc_in_display(self, *args: Any) -> None:
        pass

    def observe(self, *args: Any) -> None:
        pass


def make_packet(index: int, t_sec: float, scores: tuple[int | None, int | None]) -> SimpleNamespace:
    result = SimpleNamespace(p1=SimpleNamespace(score=scores[0]), p2=SimpleNamespace(score=scores[1]))
    return SimpleNamespace(frame=index, t_sec=t_sec, captured_at=0.0, pipeline=None, result=lambda: result)


def run_mini_generate(events: list[Event], isolate: bool, coalesce: bool = False) -> Trace:
    """events を最小generateへ流し、各通知の (番号, game_idx, デバウンス枠の時刻) を返す。"""
    from scripts.visualize_advantage_overlay import (
        _detect_score_reset, accept_formal_boundary, update_score_reset_latch)
    from src.phase_j.live_bridge import adapt_loop
    packets = [make_packet(i, t, scores) for i, (t, scores, _) in enumerate(events)]
    forced = {i for i, (_, _, flag) in enumerate(events) if flag}
    tree = ast.parse(MINI_GENERATE)
    loop = next(node for node in tree.body[0].body if isinstance(node, ast.For))
    adapt_loop(loop, isolate, coalesce_recovery=coalesce)
    bridge = FakeBridge(packets, forced)
    namespace = dict(_live_bridge=bridge, _detect_score_reset=_detect_score_reset,
                     accept_formal_boundary=accept_formal_boundary,
                     update_score_reset_latch=update_score_reset_latch,
                     _drive_ojama=lambda *args, **kwargs: None,
                     _RESOLVED=SimpleNamespace(_counter_tracker=None))
    exec(compile(ast.fix_missing_locations(tree), '<mini_generate>', 'exec'), namespace)
    trace = namespace['generate'](None)
    assert bridge.consumed == list(range(len(events))), '強制境界は毎通知ちょうど1回だけ読む'
    return trace


def load_pickle_rows(path: Path) -> list[Any]:
    rows: list[Any] = []
    if not path.exists():
        return rows
    with path.open('rb') as stream:
        while True:
            try:
                rows.append(pickle.load(stream))
            except EOFError:
                return rows


def unique_rows(rows: list[Any]) -> list[dict]:
    """評価行をframeで一意化する (再送行は最初の1行だけ)。"""
    seen: dict[int, dict] = {}
    for row in rows:
        if isinstance(row, dict):
            seen.setdefault(row['frame'], row)
    return [seen[key] for key in sorted(seen)]


def to_events(rows: list[dict]) -> list[Event]:
    """通知の間隔が FORCED_GAP_SEC を超えて空いた直後の通知を、強制境界の消費通知とみなす。"""
    events: list[Event] = []
    previous: float | None = None
    for row in rows:
        t = row['t_sec']
        events.append((t, (row['score1'], row['score2']), previous is not None and t - previous > FORCED_GAP_SEC))
        previous = t
    return events


def boundary_times(rows: list[dict], trace: Trace) -> list[float]:
    """game_idx が増えた通知の時刻 (境界の受理時刻)。"""
    times, previous = [], 0
    for row, (_, game, _) in zip(rows, trace):
        times.extend([row['t_sec']] * (game - previous))
        previous = game
    return times


def death_runs(rows: list[dict]) -> list[dict]:
    """frame一意化後の評価行で confirmed_death が連続した区間 (開始・終了・試合番号)。

    再送行 (同frameの別内容) を混ぜると区間が人工的に分断されるため、必ず一意化後の行を渡す。
    """
    runs: list[dict] = []
    current: dict | None = None
    for row in rows:
        if row.get('source') == 'confirmed_death':
            current = current or dict(start=row['t_sec'], game=row['game'])
            current['end'] = row['t_sec']
        elif current is not None:
            runs.append(current)
            current = None
    if current is not None:
        runs.append(current)
    return runs


def death_runs_ending_at_game_change(rows: list[dict], runs: list[dict]) -> int:
    """死亡確定の区間の直後の行で、記録された game_idx が変わっている区間の数 (行が尽きた区間は数えない)。"""
    times = [row['t_sec'] for row in rows]
    count = 0
    for run in runs:
        following = times.index(run['end']) + 1
        count += following < len(rows) and rows[following]['game'] != run['game']
    return count


def epoch_change_times(directory: Path) -> list[float]:
    """入力世代が1以上へ変わった時刻 (単調時計)。強制境界の通知との照合用。"""
    times, previous = [], None
    for event in load_pickle_rows(directory / 'input_events.pickle'):
        epoch = event.get('epoch') if isinstance(event, dict) else None
        if epoch is not None and epoch != previous:
            if epoch > 0:
                times.append(event['at'])
            previous = epoch
    return times


@dataclass
class RunReport:
    name: str
    rows: int
    forced: int
    legacy_mismatch_rows: int
    boundaries_off: int
    boundaries_on: int
    added: list[float]
    removed: list[float]
    offset_ok: bool
    changed_rows: int
    rejected_off: list[tuple[float, float | None]]
    rejected_on: list[tuple[float, float | None]]
    death_runs: int
    death_end_is_boundary: int
    long_runs: list[dict]
    epoch_changes: int
    forced_matches_epoch: int


def rising_rejections(rows: list[dict], trace: Trace) -> list[tuple[float, float | None]]:
    """得点リセットの立ち上がりなのに受理されなかった通知 (時刻, 直前の受理境界からの経過秒)。"""
    from scripts.visualize_advantage_overlay import _detect_score_reset, update_score_reset_latch
    out: list[tuple[float, float | None]] = []
    latched, previous, accepted_t = False, (None, None), None
    previous_game = 0
    for row, (_, game, _) in zip(rows, trace):
        scores = (row['score1'], row['score2'])
        reset = _detect_score_reset(*scores, *previous)
        accepted = game > previous_game
        if accepted:
            accepted_t = row['t_sec']
        elif reset and not latched:
            since = None if accepted_t is None else round(row['t_sec'] - accepted_t, 2)
            out.append((round(row['t_sec'], 2), since))
        latched = update_score_reset_latch(latched, reset, *scores)
        previous, previous_game = scores, game
    return out


def replay(directory: Path) -> RunReport:
    spool = directory / 'output' / 'spool'
    raw = load_pickle_rows(spool / 'evaluations.pickle')
    rows = unique_rows(raw)
    events = to_events(rows)
    off, on = run_mini_generate(events, False), run_mini_generate(events, True)
    mismatch = sum(1 for row, (_, game, _) in zip(rows, off) if game != row['game'])
    times_off, times_on = boundary_times(rows, off), boundary_times(rows, on)
    added = sorted((Counter(times_on) - Counter(times_off)).elements())
    removed = sorted((Counter(times_off) - Counter(times_on)).elements())
    # 増えた境界より前は一致、後は一定の差 (増えた境界数) だけずれる、が成り立つか。
    offset_ok = all(g_on - g_off == sum(1 for a in added if a <= row['t_sec']) - sum(1 for a in removed if a <= row['t_sec'])
                    for row, (_, g_off, _), (_, g_on, _) in zip(rows, off, on))
    changed = sum(1 for (_, g_off, _), (_, g_on, _) in zip(off, on) if g_off != g_on)
    runs = death_runs(rows)
    ends = death_runs_ending_at_game_change(rows, runs)
    forced_rows = [(i, rows[i]) for i, (_, _, flag) in enumerate(events) if flag]
    epochs = epoch_change_times(spool)
    matched = sum(1 for i, row in forced_rows
                  if any(rows[i - 1]['captured_at'] - EPOCH_MATCH_MARGIN_SEC <= at <= row['captured_at'] + EPOCH_MATCH_MARGIN_SEC
                         for at in epochs))
    return RunReport(directory.name, len(rows), len(forced_rows), mismatch, len(times_off), len(times_on),
                     added, removed, offset_ok, changed, rising_rejections(rows, off),
                     rising_rejections(rows, on), len(runs), ends,
                     [r for r in runs if r['end'] - r['start'] > LONG_DEATH_RUN_SEC], len(epochs), matched)


def format_report(report: RunReport) -> str:
    lines = [f'== {report.name}: 評価行 {report.rows} / 強制境界 {report.forced}']
    lines.append(f'  [台の妥当性] 修正前 (OFF) の再生と実game_idxの不一致行: {report.legacy_mismatch_rows}/{report.rows}')
    lines.append(f'  [境界数] OFF {report.boundaries_off} -> ON {report.boundaries_on} '
                 f'(増 {[round(t, 2) for t in report.added]} / 減 {[round(t, 2) for t in report.removed]})')
    lines.append(f'  [game_idx] ON/OFFで値が変わる行: {report.changed_rows}/{report.rows}'
                 f' (増えた境界以後が一定に+1ずれる整合: {report.offset_ok})')
    lines.append(f'  [得点リセット立ち上がりの拒否 (時刻, 直前境界からの秒)] OFF {report.rejected_off} / ON {report.rejected_on}')
    lines.append(f'  [死亡確定の区間] {report.death_runs}件, 区間の直後の行でgame_idxが変わる {report.death_end_is_boundary}/{report.death_runs}'
                 f', {LONG_DEATH_RUN_SEC:.0f}秒超 {report.long_runs}')
    lines.append(f'  [強制境界の妥当性] 入力世代の変化 {report.epoch_changes}回, '
                 f'強制境界とみなした通知のうち世代変化と時刻が合う {report.forced_matches_epoch}/{report.forced}')
    return '\n'.join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directories', nargs='+', type=Path)
    for directory in parser.parse_args().directories:
        print(format_report(replay(directory)))


if __name__ == '__main__':
    main()
