"""E7の事前登録済みv2指標を保存済みOFFと再生ONへ適用する。"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np

from scripts import aggregate_e3_exchange_eval_20260926 as e3
from scripts.aggregate_e4_exchange_eval_20260926 import m1_events
from scripts.e6b_exchange_reach_20260926 import DWELL_TOLERANCE_SEC
from scripts.run_e3_exchange_eval_20260926 import SOURCES, save_json
from scripts.visualize_advantage_overlay import _winprob_to_adv
from src.display_freshness import evaluation_freshness
from src.old.scorer import EVEN_THRESHOLD, SCORE_RANGE_MIN, SCORE_RANGE_MAX

OUT = Path("logs/e7")
OFF = Path("logs/e3/renders")
MIN_CHANGE = 5.0
MIDPOINT_FRACTION = .5
M1_LIMIT = .20
LOSS_TOLERANCE = .005
FLIP_RATIO = 1.1
FRESHNESS_TOLERANCE = 1.0
MODES = ("off", "on")


def load_display(path: Path) -> dict:
    """圧縮列を読み、ファイルを閉じた後も独立して参照する。"""
    with np.load(path) as data:
        return {key: data[key].copy() for key in data.files}


def m4(display: dict) -> dict:
    """中立域を挟んだ両側到達だけを数え、試合境界では状態を捨てる。"""
    values, games = display["display_adv"], display["game_idx"]
    if not np.isfinite(values).all():
        raise ValueError("M4表示値が非有限")
    boundaries = np.flatnonzero(np.r_[True, games[1:] != games[:-1], True])
    flips = 0
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        outside = values[start:end][np.abs(values[start:end]) >= EVEN_THRESHOLD]
        signs = np.sign(outside)
        flips += int(np.count_nonzero(signs[1:] != signs[:-1]))
    frames = len(values)
    minutes = frames / e3.FPS / e3.SECONDS_PER_MINUTE
    return dict(frames=frames, minutes=minutes, flips=flips, flips_per_minute=flips / minutes)


def delay(display: dict, game: int, trigger: float, end: float,
          cutoff: float, target: float) -> dict:
    """発火前の実表示を基準とし、未到達を分母から取り除かない。"""
    times, values = display["t_sec"], display["display_adv"]
    same_game = display["game_idx"] == game
    before = np.flatnonzero(same_game & (times < trigger))
    if not len(before):
        return dict(excluded="missing_prefire")
    base = float(values[before[-1]])
    change = target - base
    result = dict(base=base, change=change, base_sec=float(times[before[-1]]))
    if abs(change) < MIN_CHANGE:
        return dict(result, excluded="small_change")
    midpoint = base + MIDPOINT_FRACTION * change
    eligible = same_game & (times >= end) & (times < cutoff)
    reached = np.flatnonzero(eligible & ((values - midpoint) * np.sign(change) >= 0))
    seconds = float(times[reached[0]] - end) if len(reached) else float("inf")
    return dict(result, midpoint=midpoint, seconds=seconds)


def event_target(event: dict, events: list[dict], on: dict) -> dict:
    """終了合図・閉鎖後目標・次区間の打切りを両モード共通に固定する。"""
    ends = [chain["end_signal_sec"] for chain in event["chains"]]
    if not ends or any(t is None for t in ends):
        return dict(excluded="missing_end")
    if event["closed_sec"] is None or event["close_reason"] == "match_boundary":
        return dict(excluded="not_completed")
    game = event["game_idx"]
    stamps = on["t_sec"][on["game_idx"] == game]
    following = [e["trigger_sec"] for e in events
                 if e["game_idx"] == game and e["exchange_id"] > event["exchange_id"]]
    cutoff = min(following + [float(stamps[-1] + 1 / e3.FPS)])
    eligible = np.flatnonzero((on["game_idx"] == game)
        & (on["t_sec"] >= event["closed_sec"]) & (on["t_sec"] < cutoff)
        & (on["source"] == "G_fe"))
    if not len(eligible):
        return dict(excluded="missing_postclose_static")
    idx = eligible[0]
    target = float(np.clip(_winprob_to_adv(float(on["display_p1"][idx])),
                           SCORE_RANGE_MIN, SCORE_RANGE_MAX))
    return dict(end_sec=max(ends), cutoff_sec=cutoff, target=target,
                target_sec=float(on["t_sec"][idx]))


def m2_rows(events: list[dict], displays: dict) -> list[dict]:
    """除外も含む全撃ち合い台帳を作る。"""
    rows = []
    for event in events:
        row = dict(exchange_id=event["exchange_id"], game_idx=event["game_idx"],
                   trigger_sec=event["trigger_sec"], closed_sec=event["closed_sec"])
        target = event_target(event, events, displays["on"])
        row.update(target)
        if "excluded" not in target:
            row["modes"] = {mode: delay(display, event["game_idx"], event["trigger_sec"],
                target["end_sec"], target["cutoff_sec"], target["target"])
                for mode, display in displays.items()}
        rows.append(row)
    return rows


def m2_summary(rows: list[dict]) -> dict:
    """モード別の変化量除外と未到達をそれぞれ明記する。"""
    result = dict(exchanges=len(rows), common_excluded=dict(Counter(
        r["excluded"] for r in rows if "excluded" in r)))
    for mode in MODES:
        measured = [r["modes"][mode] for r in rows if "modes" in r]
        values = [r["seconds"] for r in measured if "seconds" in r]
        result[mode] = dict(eligible=len(values), unreached=sum(np.isinf(v) for v in values),
            excluded=dict(Counter(r["excluded"] for r in measured if "excluded" in r)),
            median_seconds=float(np.median(values)) if values else None)
    return result


def summarize(source: str) -> tuple[dict, list[dict], list[dict]]:
    """E6bと同じ入力・ラベル集合を照合して動画単位で保存する。"""
    root = OUT / "renders" / source / "on"
    displays = dict(off=load_display(OFF / source / "off/display.npz"),
                    on=load_display(root / "display.npz"))
    for field in ("t_sec", "game_idx"):
        np.testing.assert_array_equal(displays["off"][field], displays["on"][field])
    baseline = load_display(Path("logs/e6b/current/renders") / source / "on/display.npz")
    for field in ("display_p1", "source", "adv_raw_last"):
        np.testing.assert_array_equal(baseline[field], displays["on"][field])
    import json
    events = [json.loads(line) for line in (root / "events.jsonl").read_text().splitlines()]
    assert (root / "events.jsonl").read_bytes() == (
        Path("logs/e6b/current/renders") / source / "on/events.jsonl").read_bytes()
    rows = m2_rows(events, displays)
    windows, provenance = e3.outcomes(source)
    first, _ = m1_events(events)
    result = dict(source=source, M1=first, M2=m2_summary(rows),
        M3={mode: e3.m3_scores(display, windows) for mode, display in displays.items()},
        M4={mode: m4(display) for mode, display in displays.items()},
        freshness={mode: evaluation_freshness(display, mode, e3.FPS)
                   for mode, display in displays.items()}, outcome_provenance=provenance)
    save_json(OUT / "v2" / source / "events.json", e3.finite_json(rows))
    save_json(OUT / "v2" / source / "summary.json", e3.finite_json(result))
    return result, rows, events


def pooled(summaries: list[dict], rows: list[dict], events: list[dict]) -> dict:
    """動画ごとの中央値を平均せず、全対象を合算する。"""
    first, _ = m1_events(events)
    result = dict(M1=first, M2=m2_summary(rows), M3_q=summaries[0]["M3"], M4={}, freshness={})
    for mode in MODES:
        flips = sum(s["M4"][mode]["flips"] for s in summaries)
        minutes = sum(s["M4"][mode]["minutes"] for s in summaries)
        result["M4"][mode] = dict(flips=flips, minutes=minutes, flips_per_minute=flips / minutes)
        fields = ("frames", "adjacent_pairs", "equal_pairs", "updates", "missing_frames")
        fresh = {k: sum(s["freshness"][mode][k] for s in summaries) for k in fields}
        fresh.update(longest_equal_seconds=max(s["freshness"][mode]["longest_equal_seconds"]
            for s in summaries), equal_fraction=fresh["equal_pairs"] / fresh["adjacent_pairs"],
            updates_per_minute=fresh["updates"] / minutes, minutes=minutes)
        result["freshness"][mode] = fresh
    result["reach"] = e3.read_json(OUT / "reach_summary.json")["pooled"]
    return result


def gates(data: dict, summaries: list[dict]) -> dict:
    """事前登録した比較方向と許容幅のみを適用する。"""
    reach = data["reach"]
    off, on = (data["M2"][m]["median_seconds"] for m in MODES)
    checks = dict(M1=data["M1"]["revoked_fraction"] <= M1_LIMIT,
        M2=on is not None and off is not None and on <= off,
        M3=data["M3_q"]["on"]["groups"]["all"]["log_loss"] <=
            data["M3_q"]["off"]["groups"]["all"]["log_loss"] + LOSS_TOLERANCE,
        M4=data["M4"]["on"]["flips_per_minute"] <=
            data["M4"]["off"]["flips_per_minute"] * FLIP_RATIO,
        reach=reach["reach_pass"], display_reach=reach["display_reach_pass"],
        dwell=reach["dwell_pass"], freshness=all(
            s["freshness"]["on"]["longest_equal_seconds"] <=
            s["freshness"]["off"]["longest_equal_seconds"] + FRESHNESS_TOLERANCE for s in summaries))
    return dict(checks, all_pass=all(checks.values()))


def main() -> None:
    """全指標と全件の分母をJSONへ残す。"""
    summaries, rows, events = [], [], []
    for source in SOURCES:
        summary, video_rows, video_events = summarize(source)
        summaries.append(summary)
        rows.extend(video_rows)
        events.extend(video_events)
    data = pooled(summaries, rows, events)
    data["gates"] = gates(data, summaries)
    save_json(OUT / "v2/summary.json", e3.finite_json(summaries))
    save_json(OUT / "v2/pooled.json", e3.finite_json(data))
    save_json(OUT / "gates.json", dict(metric_version=2, **data["gates"]))
    report(data, summaries)
    print(e3.finite_json(data), flush=True)


def report(data: dict, summaries: list[dict]) -> None:
    """数値と分母を同じJSONから読める検収表にする。"""
    lines = ["# E7 計測定義v2の再生結果", "", "対象は3動画×900秒=45分、各27000フレーム。",
        "ONのdisplay_p1・source・adv_raw_last全列とevents全バイトはE6bと一致。", "",
        "|指標|OFF|ON|ゲート|", "|---|---:|---:|---|"]
    m2 = data["M2"]
    fmt = lambda mode: (f'{m2[mode]["median_seconds"]:.3f}秒 '
        f'({m2[mode]["eligible"]}/{m2["exchanges"]}件、未到達{m2[mode]["unreached"]}件)')
    lines.append(f'|M2中央値|{fmt("off")}|{fmt("on")}|{data["gates"]["M2"]}|')
    for key, label, field, denominator in (
        ("M4", "M4反転/分", "flips_per_minute", "minutes"),
        ("freshness", "最長同値秒", "longest_equal_seconds", "frames"),
        ("freshness", "同値割合", "equal_fraction", "adjacent_pairs"),
        ("freshness", "更新回数/分", "updates_per_minute", "minutes")):
        values = [f'{data[key][m][field]:.6f} (分母{data[key][m][denominator]})' for m in MODES]
        verdict = "参考" if field in ("equal_fraction", "updates_per_minute") else data["gates"][key]
        lines.append(f'|{label}|{values[0]}|{values[1]}|{verdict}|')
    losses = [data["M3_q"][m]["groups"]["all"] for m in MODES]
    lines.append('|M3 q log loss|' + '|'.join(
        f'{r["log_loss"]:.6f} ({r["frames"]}フレーム/{r["matches"]}試合)' for r in losses)
        + f'|{data["gates"]["M3"]}|')
    r, first = data["reach"], data["M1"]
    lines += [f'|M1(i)|対象外|{first["revoked"]}/{first["s3_exchanges"]}|{data["gates"]["M1"]}|',
        f'|S3内部/実表示到達|対象外|各{r["reached"]}/{r["completed"]} '
        f'({r["reach_fraction"]:.2%})|{data["gates"]["reach"]}|',
        f'|S1滞在中央値|対象外|{r["s1_median_seconds"]:.3f}秒 ({r["completed"]}件)|'
        f'{data["gates"]["dwell"]} (上限{r["firing_to_last_end_median"] + DWELL_TOLERANCE_SEC:.3f}秒)|', "",
        f'M2共通除外: {m2["common_excluded"]}。5点未満: '
        f'OFF={m2["off"]["excluded"]}、ON={m2["on"]["excluded"]}。',
        'S1上限の発火→最終終了中央値は終了合図の揃う完了100件から算出。', "",
        "|動画|M2秒 OFF/ON (対象件数)|M4回数 OFF/ON (各15分)|最長同値秒 OFF/ON|",
        "|---|---|---|---|"]
    for s in summaries:
        delays = '/'.join(f'{s["M2"][m]["median_seconds"]:.3f} ({s["M2"][m]["eligible"]})' for m in MODES)
        flips = '/'.join(str(s["M4"][m]["flips"]) for m in MODES)
        fresh = '/'.join(f'{s["freshness"][m]["longest_equal_seconds"]:.3f}' for m in MODES)
        lines.append(f'|{s["source"]}|{delays}|{flips}|{fresh}|')
    lines += ["", f'全ゲート: {data["gates"]}',
        'M2未達のため不合格。閾値・定義は変更しない。合格後条件の追加テスト・OFF25秒SHA再実行は未実施。',
        '既存関連テスト結果: logs/e7/tests.log。全件台帳: logs/e7/v2/各動画/events.json。']
    (OUT / "v2/REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
