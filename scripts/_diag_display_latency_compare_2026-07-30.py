"""表示反映速度の定量比較スクリプト (2026-07-30)

scripts/_diag_settle_freeze_2026-07-29.py が出力する診断ログ (warmup有無A/B比較用)
から、盤面確定(settled)状態の遷移・SETTLED行の発生間隔・大きな発火イベント(例:
12連鎖)への応答遅延を、推測を交えず実測値のみで報告する。

厳守事項 (userの明示指示):
- ログから読めない指標は "算出不能" と明記し、数値を作らない。
- 2本のログを比較する際は、共通して観測できている時刻範囲(RESET行=試合開始
  以降 〜 両ログの末尾のうち早い方)に切り詰めて比較する。
- 代表値は中央値だけでなく最大値・p90 も必ず出す (「代表値を出す前に層別せよ」
  「プールした中央値は相殺で見かけ倒し」というプロジェクト確定教訓のため)。

実行例 (WSL, nice経由。ログのテキスト処理のみで軽量、他の並列処理を邪魔しない):
    nice -n 19 ./venv/bin/python -m scripts._diag_display_latency_compare_2026-07-30 \
        --log-a logs/_diag_c56_g3_warmup0_2026-07-29.log \
        --log-b logs/_diag_c56_g3_warmup30_2026-07-29.log
"""
from __future__ import annotations

import argparse
import re
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# --- ログ行パターン (_diag_settle_freeze_2026-07-29.py の print 文と1対1対応) ---
_LINE_PREFIX_RE = re.compile(r"^\[(?P<label>[^\]]+)\] t=(?P<t>[0-9]+\.[0-9]+)s (?P<rest>.*)$")
_SETTLED_TRANSITION_RE = re.compile(r"^settled (?P<prev>True|False)->(?P<cur>True|False)\b")
_RESET_RE = re.compile(r"^RESET$")
_SETTLED_LINE_RE = re.compile(
    r"^SETTLED score1=(?P<score1>-?\d+) score2=(?P<score2>-?\d+) .*?"
    r"adv_ema=(?P<adv_ema>[+-]?\d+\.\d+) p1=(?P<p1>[0-9.]+)$"
)
_EARLYFIRE_RE = re.compile(
    r"^EARLYFIRE disp_adv=(?P<disp_adv>[+-]?\d+\.\d+) "
    r"\(adv_ema=(?P<adv_ema>[+-]?\d+\.\d+) bias=(?P<bias>[+-]?\d+\.\d+)\) "
    r"settled=(?P<settled>True|False)$"
)

# --- 分析パラメータ (マジックナンバー禁止規約によりすべて定数化) ---
IGNITION_SCORE_JUMP_THRESHOLD = 2000  # 1SETTLED行間でこれ以上スコアが増えたら「発火」とみなす閾値(点)
IGNITION_SEARCH_CENTER_SEC = 339.0  # user報告「動画51秒付近」を絶対時刻に換算した探索中心(試合開始t≈288s基準)
IGNITION_SEARCH_HALF_WIDTH_SEC = 20.0  # 発火探索窓の半幅(中心±この秒数を捜索)
ADV_EMA_MOVE_THRESHOLD = 10.0  # 発火後、確定値 adv_ema がこれ以上動いたら「反応した」とみなす閾値
DISP_ADV_MOVE_THRESHOLD = 10.0  # 早期発火(disp_adv)の即時反応判定しきい値(参考値・確定値ではない)


@dataclass
class SettledLineEvent:
    """SETTLED行(確定計算が走った瞬間)1件分。"""

    t: float
    score1: int
    score2: int
    adv_ema: float


@dataclass
class EarlyFireEvent:
    """EARLYFIRE行(早期発火バイアス込みの表示値)1件分。"""

    t: float
    disp_adv: float
    adv_ema: float
    bias: float


@dataclass
class ParsedLog:
    """1本の診断ログから抽出した時系列イベント一式。"""

    path: Path
    label: str = ""
    reset_times: list = field(default_factory=list)
    settled_transitions: list = field(default_factory=list)  # (t, new_settled)
    settled_lines: list = field(default_factory=list)
    earlyfire_events: list = field(default_factory=list)
    all_times: list = field(default_factory=list)  # ログ末尾検出用(prefix付き行すべて)


def parse_log(path: Path) -> ParsedLog:
    """診断ログ1本を読み、行種別ごとにタイムスタンプ付きで振り分ける。

    `[label] t=X.XXs ...` の prefix を持たない行 (numpy警告等) は無視する。
    """
    log = ParsedLog(path=path)
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            m = _LINE_PREFIX_RE.match(raw.rstrip("\n"))
            if m is None:
                continue
            log.label = m.group("label")
            t = float(m.group("t"))
            rest = m.group("rest")
            log.all_times.append(t)
            if _RESET_RE.match(rest):
                log.reset_times.append(t)
                continue
            tm = _SETTLED_TRANSITION_RE.match(rest)
            if tm is not None:
                log.settled_transitions.append((t, tm.group("cur") == "True"))
                continue
            sm = _SETTLED_LINE_RE.match(rest)
            if sm is not None:
                log.settled_lines.append(SettledLineEvent(
                    t=t, score1=int(sm.group("score1")), score2=int(sm.group("score2")),
                    adv_ema=float(sm.group("adv_ema"))))
                continue
            em = _EARLYFIRE_RE.match(rest)
            if em is not None:
                log.earlyfire_events.append(EarlyFireEvent(
                    t=t, disp_adv=float(em.group("disp_adv")),
                    adv_ema=float(em.group("adv_ema")), bias=float(em.group("bias"))))
    return log


def match_start_time(log: ParsedLog) -> Optional[float]:
    """試合開始時刻 = 最初の RESET 行の時刻(userの明示指示)。"""
    return log.reset_times[0] if log.reset_times else None


def log_end_time(log: ParsedLog) -> Optional[float]:
    """このログで観測できている最後の時刻(タイムスタンプ付き行の最大値)。"""
    return max(log.all_times) if log.all_times else None


@dataclass
class SettledSegment:
    """settled=True/False が連続していた区間1件分。"""

    start: float
    end: float
    settled: bool


def build_settled_segments(log: ParsedLog, range_start: float,
                            range_end: float) -> list:
    """settled True/False の遷移行から [range_start, range_end] 内の区間列を復元する。

    range_start 時点の状態は、それ以前の最後の遷移行から引き継ぐ。
    range_start より前に遷移行が1件も無い場合は復元不能として空リストを返す。
    """
    transitions = sorted(log.settled_transitions, key=lambda x: x[0])
    prior = [tr for tr in transitions if tr[0] <= range_start]
    if not prior:
        return []
    state = prior[-1][1]
    boundary = range_start
    segments: list = []
    for t, new_state in transitions:
        if t <= range_start:
            continue
        if t > range_end:
            break
        segments.append(SettledSegment(start=boundary, end=t, settled=state))
        state = new_state
        boundary = t
    segments.append(SettledSegment(start=boundary, end=range_end, settled=state))
    return segments


def settled_ratio(segments: list) -> float:
    """settled=True だった時間の割合(0.0〜1.0)。"""
    total = sum(s.end - s.start for s in segments)
    settled_total = sum(s.end - s.start for s in segments if s.settled)
    return settled_total / total if total > 0 else float("nan")


def freeze_durations(segments: list) -> tuple:
    """settled=False の連続区間長リストを返す。

    範囲末尾で凍結継続中のまま終わる(=真の解消時刻が範囲外で不明)区間は
    右打ち切りデータとして分布から除外し、2つ目の戻り値(経過秒数)で報告する。
    """
    closed: list = []
    censored: Optional[float] = None
    for i, seg in enumerate(segments):
        if not seg.settled:
            dur = seg.end - seg.start
            if i == len(segments) - 1:
                censored = dur
            else:
                closed.append(dur)
    return closed, censored


def settled_line_intervals(log: ParsedLog, range_start: float,
                            range_end: float) -> list:
    """[range_start, range_end] 内の SETTLED 行タイムスタンプの連続差分リスト。"""
    ts = sorted(e.t for e in log.settled_lines if range_start <= e.t <= range_end)
    return [b - a for a, b in zip(ts, ts[1:])]


def summarize(values: list) -> dict:
    """中央値・p90・最大値・件数をまとめて返す(単一代表値を避けるプロジェクト方針)。"""
    if not values:
        return {"n": 0}
    srt = sorted(values)
    n = len(srt)
    p90_idx = min(n - 1, int(round(0.9 * (n - 1))))
    return {
        "n": n, "median": statistics.median(srt), "p90": srt[p90_idx],
        "max": srt[-1], "mean": statistics.fmean(srt),
    }


@dataclass
class IgnitionResult:
    """発火イベントとその応答遅延の計測結果。note が非空なら算出不能。"""

    ignition_t: Optional[float]
    ignition_desc: str
    response_t: Optional[float]
    response_delay_sec: Optional[float]
    unresolved_lower_bound_sec: Optional[float]
    earlyfire_reaction_t: Optional[float]
    earlyfire_reaction_delay_sec: Optional[float]
    note: str


def _find_ignition_event(log: ParsedLog, win_start: float,
                          win_end: float) -> Optional[SettledLineEvent]:
    """探索窓内で SETTLED 行間のスコア急増(発火)を検出し、その行を返す。"""
    lines = sorted(log.settled_lines, key=lambda e: e.t)
    prev: Optional[SettledLineEvent] = None
    for cur in lines:
        if prev is not None and win_start <= cur.t <= win_end:
            d1, d2 = cur.score1 - prev.score1, cur.score2 - prev.score2
            if max(d1, d2) >= IGNITION_SCORE_JUMP_THRESHOLD:
                return cur
        prev = cur
    return None


def _describe_ignition(log: ParsedLog, ignition: SettledLineEvent) -> str:
    """発火の同定根拠(どのSETTLED行の何がどう跳ねたか)を文字列化する。"""
    lines = sorted(log.settled_lines, key=lambda e: e.t)
    idx = lines.index(ignition)
    prev = lines[idx - 1]
    d1, d2 = ignition.score1 - prev.score1, ignition.score2 - prev.score2
    return (f"SETTLED t={ignition.t:.2f}s score1 {prev.score1}->{ignition.score1} "
            f"(Delta{d1:+d}), score2 {prev.score2}->{ignition.score2} (Delta{d2:+d})")


def find_ignition_response(log: ParsedLog, log_end: float,
                            center_sec: float = IGNITION_SEARCH_CENTER_SEC,
                            half_width_sec: float = IGNITION_SEARCH_HALF_WIDTH_SEC
                            ) -> IgnitionResult:
    """探索窓内の発火イベントを検出し、adv_ema/disp_adv の反応遅延を計測する。"""
    win_start, win_end = center_sec - half_width_sec, center_sec + half_width_sec
    if log_end < win_start:
        note = (f"ログが探索窓開始({win_start:.1f}s)まで到達していないため算出不能"
                f"(ログ末尾={log_end:.2f}s)")
        return IgnitionResult(None, "", None, None, None, None, None, note)
    ignition = _find_ignition_event(log, win_start, win_end)
    if ignition is None:
        if log_end < win_end:
            note = (f"ログが探索窓終了({win_end:.1f}s)まで未到達のため窓を完全に"
                     f"走査できていない(ログ末尾={log_end:.2f}s)。この時点までに"
                     f"閾値{IGNITION_SCORE_JUMP_THRESHOLD}点以上のスコア急増は無いが、"
                     f"窓の残り部分は未確認であり「発火なし」と断定できない")
        else:
            note = (f"探索窓[{win_start:.1f}s, {win_end:.1f}s]を完全に走査したが閾値"
                     f"{IGNITION_SCORE_JUMP_THRESHOLD}点以上のスコア急増は見つからず")
        return IgnitionResult(None, "", None, None, None, None, None, note)
    desc = _describe_ignition(log, ignition)
    response_t, response_delay = _find_adv_ema_response(log, ignition)
    unresolved_lb = None if response_t is not None else (log_end - ignition.t)
    ef_t, ef_delay = _find_earlyfire_reaction(log, ignition)
    return IgnitionResult(ignition.t, desc, response_t, response_delay,
                          unresolved_lb, ef_t, ef_delay, note="")


def _find_adv_ema_response(log: ParsedLog, ignition: SettledLineEvent) -> tuple:
    """発火後、確定値 adv_ema が閾値以上動いた最初の時刻を探す(未解消ならNone)。"""
    baseline = ignition.adv_ema
    for cur in sorted(log.settled_lines, key=lambda e: e.t):
        if cur.t <= ignition.t:
            continue
        if abs(cur.adv_ema - baseline) >= ADV_EMA_MOVE_THRESHOLD:
            return cur.t, cur.t - ignition.t
    return None, None


def _find_earlyfire_reaction(log: ParsedLog, ignition: SettledLineEvent) -> tuple:
    """発火後、早期発火の表示値(disp_adv)が閾値以上動いた最初の時刻を探す(参考値)。"""
    baseline = ignition.adv_ema  # settled中はbias=0のためdisp_adv=adv_emaが基準値
    for ef in sorted(log.earlyfire_events, key=lambda e: e.t):
        if ef.t <= ignition.t:
            continue
        if abs(ef.disp_adv - baseline) >= DISP_ADV_MOVE_THRESHOLD:
            return ef.t, ef.t - ignition.t
    return None, None


def _print_summary_block(title: str, stats: dict) -> None:
    if stats.get("n", 0) == 0:
        print(f"  {title}: 該当データなし(算出不能)")
        return
    print(f"  {title}: n={stats['n']} 中央値={stats['median']:.2f}s "
          f"p90={stats['p90']:.2f}s 最大={stats['max']:.2f}s 平均={stats['mean']:.2f}s")


def report_range(log: ParsedLog, range_start: float, range_end: float, title: str) -> None:
    """指定範囲について settled比率/凍結区間分布/SETTLED間隔分布を出力する。"""
    print(f"--- {title}: [{range_start:.2f}s, {range_end:.2f}s] "
          f"(幅{range_end - range_start:.2f}s) ---")
    segments = build_settled_segments(log, range_start, range_end)
    if not segments:
        print("  settled状態の復元不能(範囲開始以前の遷移行が見つからない)")
        return
    ratio = settled_ratio(segments)
    print(f"  settled=True 時間率: {ratio * 100:.1f}%")
    closed, censored = freeze_durations(segments)
    _print_summary_block("凍結区間長(完結分のみ)", summarize(closed))
    if censored is not None:
        print(f"  (注) 範囲末尾時点で凍結継続中(打ち切り、経過{censored:.2f}秒以上、"
              f"上記分布には含めていない)")
    intervals = settled_line_intervals(log, range_start, range_end)
    _print_summary_block("SETTLED行発生間隔", summarize(intervals))


def _print_ignition(res: IgnitionResult) -> None:
    print("  [発火応答遅延]")
    if res.note:
        print(f"    算出不能: {res.note}")
        return
    print(f"    発火根拠: {res.ignition_desc}")
    if res.response_delay_sec is not None:
        print(f"    adv_ema(確定値)反応: t={res.response_t:.2f}s "
              f"(遅延{res.response_delay_sec:.2f}秒)")
    else:
        print(f"    adv_ema(確定値)反応: ログ範囲内で未解消"
              f"(打ち切り、経過{res.unresolved_lower_bound_sec:.2f}秒以上は下限値、"
              f"真の反応遅延は算出不能)")
    if res.earlyfire_reaction_delay_sec is not None:
        print(f"    早期発火(disp_adv)反応: t={res.earlyfire_reaction_t:.2f}s "
              f"(遅延{res.earlyfire_reaction_delay_sec:.2f}秒、※確定値ではない参考値)")
    else:
        print("    早期発火(disp_adv)反応: 該当イベント見つからず算出不能")


def common_range(log_a: ParsedLog, log_b: ParsedLog) -> tuple:
    """両ログで共通して観測できている時刻範囲(試合開始〜早い方の末尾)を求める。"""
    starts = [t for t in (match_start_time(log_a), match_start_time(log_b)) if t is not None]
    ends = [t for t in (log_end_time(log_a), log_end_time(log_b)) if t is not None]
    if not starts or not ends:
        return None, None
    return max(starts), min(ends)


def _analyze_single(path: Path, center_sec: float, half_width_sec: float) -> ParsedLog:
    """1本のログを解析し、単体レポート(全区間+発火応答遅延)を出力する。"""
    log = parse_log(path)
    start, end = match_start_time(log), log_end_time(log)
    print(f"=== {path.name} (label={log.label}) ===")
    if start is None or end is None:
        print("  RESET行またはタイムスタンプ付き行が見つからず解析不能")
        return log
    print(f"試合開始(最初のRESET行): t={start:.2f}s / ログ末尾: t={end:.2f}s")
    report_range(log, start, end, "単体(全区間)")
    _print_ignition(find_ignition_response(log, end, center_sec, half_width_sec))
    return log


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log-a", required=True, type=Path)
    ap.add_argument("--log-b", type=Path, default=None,
                     help="比較対象の2本目(省略時は--log-aのみ単体解析)")
    ap.add_argument("--ignition-center-sec", type=float, default=IGNITION_SEARCH_CENTER_SEC)
    ap.add_argument("--ignition-half-width-sec", type=float,
                    default=IGNITION_SEARCH_HALF_WIDTH_SEC)
    args = ap.parse_args()

    log_a = _analyze_single(args.log_a, args.ignition_center_sec, args.ignition_half_width_sec)
    if args.log_b is None:
        return
    print()
    log_b = _analyze_single(args.log_b, args.ignition_center_sec, args.ignition_half_width_sec)

    print("\n=== 共通区間での比較 ===")
    common_start, common_end = common_range(log_a, log_b)
    if common_start is None or common_end is None or common_end <= common_start:
        print("  共通区間が確保できず比較不能")
        return
    print(f"共通区間: [{common_start:.2f}s, {common_end:.2f}s] "
          f"(幅{common_end - common_start:.2f}s)")
    report_range(log_a, common_start, common_end, args.log_a.name)
    report_range(log_b, common_start, common_end, args.log_b.name)


if __name__ == "__main__":
    main()
