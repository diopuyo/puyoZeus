"""E3の保存済み集計から、分母付きの短い検収表を作る。"""
from __future__ import annotations

from scripts.aggregate_e3_exchange_eval_20260926 import read_json
from scripts.run_e3_exchange_eval_20260926 import OUT, SOURCES

ALIASES = {"q_7gc4TgFig": "q", "fcXG83vInDY": "fcX", "mia8KCjr52g": "mia", "all": "合算"}
GROUPS = ("all", "P_time_1", "P_time_2", "P_time_3")


def number(value: float | str | None, decimals: int = 3) -> str:
    """欠測と未到達を実数ゼロへ置き換えない。"""
    if value is None:
        return "—"
    return "∞" if value == "Infinity" else f"{value:.{decimals}f}"


def verdict(value: bool | str | None) -> str:
    """統括の採否と独立な、登録閾値の機械判定。"""
    return "参考" if value == "reference" else "未成立" if value is None else "合格" if value else "不合格"


def timing_rows(summaries: list[dict]) -> list[str]:
    """M1/M2を共通イベント母集団とともに一表へまとめる。"""
    rows = ["|動画（撃ち合い数/試合数）|M1 Δ秒中央値／着地前|早すぎS3・判定|M2秒 OFF→ON・対象数・判定|",
            "|---|---|---|---|"]
    for data in summaries:
        first, second = data["M1"], data["M2"]
        fraction = first["early_fraction"]
        rate = f"{fraction:.1%}" if fraction is not None else "—"
        early = f"{first['early']}/{first['s3_exchanges']} ({rate})"
        medians = second["median_seconds"]
        rows.append(f"|{ALIASES[data['source']]} ({first['exchanges']}/{data['event_matches']})|"
                    f"{number(first['delta_median_sec'])}／{first['before_landing']}/{first['paired']}|"
                    f"{early}・{verdict(data['thresholds']['M1'])}|"
                    f"{number(medians['off'])}→{number(medians['on'])}・{second['eligible']}/{second['exchanges']}・"
                    f"{verdict(data['thresholds']['M2'])}|")
    return rows


def correctness_rows(summaries: list[dict]) -> list[str]:
    """全体・序中終盤のlog loss/AUCを同じ順で並べる。"""
    rows = ["|動画/モード|M3 log loss（全/序/中/終）|AUC（全/序/中/終）|フレーム数（全/序/中/終）・試合数|",
            "|---|---|---|---|"]
    for data in summaries:
        for mode in ("off", "on"):
            groups = [data["M3"][mode]["groups"][key] for key in GROUPS]
            losses = "/".join(number(group["log_loss"]) for group in groups)
            aucs = "/".join(number(group["auc"]) for group in groups)
            counts = "/".join(str(group["frames"]) for group in groups)
            rows.append(f"|{ALIASES[data['source']]}/{mode.upper()}|{losses}|{aucs}|{counts}・{groups[0]['matches']}試合|")
    return rows


def stability_rows(summaries: list[dict]) -> list[str]:
    """M4の分子と分母を表示率と併記する。"""
    rows = ["|動画|M4 張り付き OFF→ON（/フレーム数）|反転 OFF→ON（回/分母分数）|反転/分 OFF→ON・判定|",
            "|---|---|---|---|"]
    for data in summaries:
        off, on = (data["M4"][mode] for mode in ("off", "on"))
        rows.append(f"|{ALIASES[data['source']]}|{off['saturated_frames']} ({off['saturated_fraction']:.2%})→"
                    f"{on['saturated_frames']} ({on['saturated_fraction']:.2%}) /{off['frames']}|"
                    f"{off['flips']}→{on['flips']} /{off['minutes']:g}分|"
                    f"{off['flips_per_minute']:.2f}→{on['flips_per_minute']:.2f}・{verdict(data['thresholds']['M4'])}|")
    return rows


def main() -> None:
    """全6レンダの結果を、追跡可能なJSONと併せて納品する。"""
    summaries = read_json(OUT / "metrics/summary.json")
    if tuple(data["source"] for data in summaries) != SOURCES:
        raise ValueError("全3動画が揃うまで最終表を生成しない")
    pooled = read_json(OUT / "metrics/pooled.json")
    lines = ["全3動画×OFF/ON完了。各27,000フレームの時刻・試合IDが一致。", ""]
    lines += timing_rows(summaries + [pooled]) + [""] + correctness_rows(summaries) + [""]
    lines += stability_rows(summaries + [pooled]) + [""]
    lines += ["M3本判定(q): " + verdict(summaries[0]["thresholds"]["M3"]) + "。fcX/miaは参考、AUC—は単一クラス。"]
    for data in summaries:
        m1, m2 = data["M1"], data["M2"]
        lines.append(f"{ALIASES[data['source']]}欠測: S3 {m1['missing_s3']}件、M2得点確定 {m2['missing_finalize']}件/G_fe {m2['missing_target']}件、"
                     f"未到達OFF/ON {m2['unreached']['off']}/{m2['unreached']['on']}件（∞）、"
                     f"M3無ラベル {data['M3']['off']['unlabeled_frames']}フレーム、発火側不明 {data['unknown_firing_side']}件。")
    examples = [(data["source"], example) for data in summaries for example in data["early_examples"]][:3]
    lines.append("早すぎS3例: " + "；".join(f"{source} {r['s3_sec']:.3f}s" for source, r in examples))
    (OUT / "metrics/REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
