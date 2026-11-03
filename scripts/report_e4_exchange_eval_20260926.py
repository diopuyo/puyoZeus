"""E4の固定閾値の合否と分母を検収表へ出す。"""
from __future__ import annotations

from scripts.aggregate_e4_exchange_eval_20260926 import OUT
from scripts.report_e3_exchange_eval_20260926 import ALIASES, correctness_rows, number, verdict


def freshness_rows(summaries: list[dict]) -> list[str]:
    """平滑化前の評価値の更新頻度と最長停止時間を比較する。"""
    rows = ["|動画|同値割合 ON/OFF|最長同値秒 ON/OFF|更新回数/分 ON/OFF|鮮度判定|",
            "|---|---|---|---|---|"]
    for data in summaries:
        on, off = (data["F4"][mode] for mode in ("on", "off"))
        rows.append(f"|{ALIASES[data['source']]}|{on['equal_fraction']:.2%}/{off['equal_fraction']:.2%}|"
                    f"{on['longest_equal_seconds']:.3f}/{off['longest_equal_seconds']:.3f}|"
                    f"{on['updates_per_minute']:.2f}/{off['updates_per_minute']:.2f}|"
                    f"{verdict(data['thresholds']['F4'])}|")
    return rows


def timing_rows(summaries: list[dict]) -> list[str]:
    """撤回と追加参加の件数を、S3発生撃ち合い数で割る。"""
    rows = ["|動画（撃ち合い/試合）|M1(i)撤回・判定|M1(ii)追加参加・参考|M2秒 OFF→ON（対象/全体）・判定|",
            "|---|---|---|---|"]
    for data in summaries:
        first, second = data["M1"], data["M2"]
        count, median = first["s3_exchanges"], second["median_seconds"]
        rates = [f"{first[key]}/{count} ({first[key + '_fraction']:.1%})" if count else "—"
                 for key in ("revoked", "continued")]
        rows.append(f"|{ALIASES[data['source']]} ({first['exchanges']}/{data['event_matches']})|"
            f"{rates[0]}・{verdict(data['thresholds']['M1'])}|{rates[1]}|"
            f"{number(median['off'])}→{number(median['on'])} ({second['eligible']}/{second['exchanges']})・"
            f"{verdict(data['thresholds']['M2'])}|")
    return rows


def stability_rows(summaries: list[dict]) -> list[str]:
    """M3/M4は鮮度ゲート未達のとき測定値だけを示す。"""
    rows = ["|動画|M3 log loss OFF→ON・判定|M4張り付き OFF→ON|M4反転/分 OFF→ON・判定|",
            "|---|---|---|---|"]
    for data in summaries:
        off, on = (data["M4"][mode] for mode in ("off", "on"))
        loss = "qを本判定"
        if "M3" in data:
            loss = "→".join(number(data["M3"][mode]["groups"]["all"]["log_loss"]) for mode in ("off", "on"))
        status3 = "判定保留（鮮度未達）" if not data["thresholds"]["F4"] else verdict(data["thresholds"]["M3"])
        status4 = "判定保留（鮮度未達）" if not data["thresholds"]["F4"] else verdict(data["thresholds"]["M4"])
        rows.append(f"|{ALIASES[data['source']]}|{loss}・{status3}|"
            f"{off['saturated_fraction']:.2%}→{on['saturated_fraction']:.2%}|"
            f"{off['flips_per_minute']:.2f}→{on['flips_per_minute']:.2f}・{status4}|")
    return rows


def report(summaries: list[dict], pooled: dict) -> None:
    """詳細JSONと同じ値から再現可能な表を生成する。"""
    lines = ["E4: ON 3本を再レンダ、OFF 3本はE3を再用。各27000フレーム・試合ID一致。", ""]
    lines += ["鮮度: ON=display_p1、OFF=adv_raw_last（平滑化前）。未評価NaN同士は未更新として全フレーム分母に含める。", ""]
    lines += freshness_rows(summaries + [pooled]) + [""]
    lines += timing_rows(summaries + [pooled]) + [""]
    lines += stability_rows(summaries + [pooled]) + [""] + correctness_rows(summaries) + [""]
    for data in summaries:
        m1, m2 = data["M1"], data["M2"]
        lines.append(f"{ALIASES[data['source']]}: M1着地差中央値 {number(m1['delta_median_sec'])}秒、"
            f"着地前 {m1['before_landing']}/{m1['paired']}、S3欠測 {m1['missing_s3']}、"
            f"M2得点確定欠測 {m2['missing_finalize']}/G_fe欠測 {m2['missing_target']}、"
            f"未到達OFF/ON {m2['unreached']['off']}/{m2['unreached']['on']}（∞）、"
            f"M3無ラベル {data['M3']['off']['unlabeled_frames']}フレーム、"
            f"発火側不明 {data['unknown_firing_side']}。")
    lines.append("M1(i)(ii)は重複可。M3のfcX/miaは認識由来ラベルの参考値。鮮度合算合格には全動画の合格が必要。")
    (OUT / "metrics/REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
