"""E9の全指標比較とq撃ち合い10の更新・実表示時系列を保存する。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.report_e7_exchange_20260927 import load_display
from scripts.run_e3_exchange_eval_20260926 import SOURCES, save_json

OUT = Path("logs/e9")
PREVIOUS = Path("logs/e8")
TARGET_EXCHANGE = 10
CHECK_TIMES = (267.0, 268.9, 270.5, 272.5, 273.0, 275.5, 280.0)


def read(path: Path) -> dict | list:
    """固定成果物を読む。"""
    return json.loads(path.read_text(encoding="utf-8"))


def timeline() -> list[dict]:
    """発火・各段評価・終了・確定S3の全時刻を両版の実表示と照合する。"""
    displays, events = {}, {}
    stamps: dict[float, list[str]] = {t: ["表示確認"] for t in CHECK_TIMES}
    for version, root in (("E8", PREVIOUS), ("E9", OUT)):
        directory = root / "renders" / SOURCES[0] / "on"
        displays[version] = load_display(directory / "display.npz")
        events[version] = next(json.loads(line) for line in
            (directory / "events.jsonl").read_text().splitlines()
            if json.loads(line)["exchange_id"] == TARGET_EXCHANGE)
        for value in events[version]["values"]:
            detail = f'{version} {value["source"]} p1={value["p1"]:.6f}'
            if "score_totals" in value:
                detail += f' 得点={value["score_totals"]}'
            stamps.setdefault(value["t_sec"], []).append(detail)
    for chain in events["E9"]["chains"]:
        stamps.setdefault(chain["observed_sec"], []).append(f'{chain["side"]} 発火')
        if chain["end_signal_sec"] is not None:
            stamps.setdefault(chain["end_signal_sec"], []).append(f'{chain["side"]} 終了')
    rows = []
    for t, descriptions in sorted(stamps.items()):
        row = dict(t_sec=t, events=descriptions)
        for version, display in displays.items():
            idx = int(np.searchsorted(display["t_sec"], t))
            row[version] = dict(p1=float(display["display_p1"][idx]),
                display_adv=float(display["display_adv"][idx]), source=str(display["source"][idx]))
        rows.append(row)
    current = displays["E9"]
    response = (current["t_sec"] >= CHECK_TIMES[3]) & (current["t_sec"] < events["E9"]["closed_sec"])
    before_s3 = (current["t_sec"] >= 265.8333333333333) & (current["t_sec"] < 271.9)
    checks = dict(disadvantage_before_final=bool(np.any(current["display_adv"][before_s3] < -3)),
                  no_return_after_response=bool(np.all(current["display_adv"][response] < -3)))
    save_json(OUT / "exchange10_timeline.json", dict(rows=rows, checks=checks))
    assert all(checks.values()), checks
    return rows


def metric_rows(data: dict) -> list[tuple[str, str]]:
    """v2の分母を省略せず、一つの比較表へ整形する。"""
    reach, m1, m2 = data["reach"], data["M1"], data["M2"]["on"]
    fresh, loss = data["freshness"]["on"], data["M3_q"]["on"]["groups"]["all"]
    return [
        ("M1(i) 早すぎS3（終了撤回）", f'{m1["revoked"]}/{m1["s3_exchanges"]} ({m1["revoked_fraction"]:.2%})'),
        ("M1(ii) S3後の追加参加", f'{m1["continued"]}/{m1["s3_exchanges"]}'),
        ("M1 初回着地→S3中央値", f'{m1["delta_median_sec"]:.3f}秒'),
        ("M1 着地前S3", f'{m1["before_landing"]}/{m1["paired"]}'),
        ("M2 半分到達中央値", f'{m2["median_seconds"]:.3f}秒（対象{m2["eligible"]}、未到達{m2["unreached"]}）'),
        ("確定S3 内部／実表示到達", f'{reach["reached"]}/{reach["completed"]} ／ {reach["display_reached"]}/{reach["completed"]}'),
        ("S1滞在中央値", f'{reach["s1_median_seconds"]:.3f}秒'),
        ("発火→最終終了中央値", f'{reach["firing_to_last_end_median"]:.3f}秒'),
        ("評価最長同値", f'{fresh["longest_equal_seconds"]:.3f}秒'),
        ("評価同値割合", f'{fresh["equal_fraction"]:.6f}（{fresh["equal_pairs"]}/{fresh["adjacent_pairs"]}）'),
        ("評価更新／分", f'{fresh["updates_per_minute"]:.3f}'),
        ("q log loss", f'{loss["log_loss"]:.6f}（{loss["frames"]}フレーム／{loss["matches"]}試合）'),
        ("q AUC", f'{loss["auc"]:.6f}'),
        ("±3帯反転", f'{data["M4"]["on"]["flips"]}回／45分（{data["M4"]["on"]["flips_per_minute"]:.3f}/分）'),
    ]


def main() -> None:
    """集計JSONから比較表と全段時系列を生成する。"""
    before, after = (read(root / "v2/pooled.json") for root in (PREVIOUS, OUT))
    rows = timeline()
    lines = ["# E9 検証結果", "", "3動画×900秒、計81,000フレーム。暫定S3は確定S3の分子に含めない。", "",
             "|v2指標|E8|E9|", "|---|---:|---:|"]
    for (label, left), (_, right) in zip(metric_rows(before), metric_rows(after)):
        lines.append(f"|{label}|{left}|{right}|")
    lines += ["", f'悪化ゲート: {after["regression_gates"]}',
              f'OFF比較の既存ゲート: {after["gates_off"]}', "", "## q 撃ち合い10", "",
              "display_advはEMA後、p1は評価器の保持値。", "",
              "|秒|観測・更新|E8 p1 / 表示|E9 p1 / 表示|", "|---:|---|---:|---:|"]
    for row in rows:
        values = [f'{row[v]["p1"]:.6f} / {row[v]["display_adv"]:+.3f}' for v in ("E8", "E9")]
        lines.append(f'|{row["t_sec"]:.3f}|{"; ".join(row["events"])}|{values[0]}|{values[1]}|')
    lines += ["", "## 動画別の全指標", "", "詳細: v2/summary.json（qの進行度別log loss・AUCを含む）。"]
    for summary in read(OUT / "v2/summary.json"):
        old = next(s for s in read(PREVIOUS / "v2/summary.json") if s["source"] == summary["source"])
        lines.append(f'\n### {summary["source"]}\n')
        for key in ("M1", "M2", "M3", "M4", "freshness"):
            lines.append(f'- {key}: E8 {old[key]} → E9 {summary[key]}')
    sha = OUT / "off_sha256.json"
    if sha.exists():
        lines += ["", f'OFF 25秒 SHA: {read(sha)}']
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
