"""段2の母数・全門・時間を統括の判断用に整形する。"""
from __future__ import annotations

import json
from typing import Any

from scripts.gfe_phase_stage2_train_20261002 import OUT


def pair(value: dict) -> str:
    """比較値・対差・95%区間を同じ精度で記載する。"""
    return (f"{value['baseline']:.9f}|{value['candidate']:.9f}|{value['difference']:+.9f}|"
            f"[{value['ci95'][0]:+.9f}, {value['ci95'][1]:+.9f}]|{value['games']}")


def comparison_lines(result: dict[str, Any]) -> list[str]:
    """全115試合・各セットのLLと終盤一致を全て記載する。"""
    lines = ["|構成|区分|指標|本番|候補|差|95%区間|試合数|",
             "|---|---|---|---:|---:|---:|---|---:|"]
    for mode in ("V3", "V1b"):
        lines.append(f"|{mode}|全体|M_time|{pair(result['decisions'][mode]['overall'])}|")
        for name in ("set1", "set2"):
            for metric in ("m_time", "agree"):
                lines.append(f"|{mode}|{name}|{metric}|{pair(result[name]['comparisons'][mode][metric])}|")
    return lines


def regression_lines(result: dict[str, Any]) -> list[str]:
    """反転と誤確定を母数付きで表示し、指定場面も省略しない。"""
    lines = ["", "|区分|試合数|全記録表示行|M_time評価行|終盤一致評価行|",
             "|---|---:|---:|---:|---:|"]
    for name in ("set1", "set2"):
        data = result[name]
        lines.append(f"|{name}|{data['games']}|{data['display_frames']['baseline']}|"
                     f"{data['evaluated_frames']['baseline']}|{data['late_frames']['baseline']}|")
    lines += ["", "|区分|指標|本番|V3|V1b|母数|", "|---|---|---:|---:|---:|---|"]
    for name in ("set1", "set2"):
        data = result[name]
        for metric in ("flips", "false_deaths"):
            values = "|".join(str(data[metric][m]) for m in ("baseline", "V3", "V1b"))
            lines.append(f"|{name}|{metric}|{values}|{data['games']}試合|")
    for key in ("cases", "false", "scene_first_sec", "game14"):
        values = "|".join(str(result["regression"][m][key]) for m in ("baseline", "V3", "V1b"))
        lines.append(f"|既存記録|{key}|{values}|5記録|")
    lines += ["", "|構成|通知数|G_fe全体P95(ms)|推論のみP95(ms)|最大(ms)|",
              "|---|---:|---:|---:|---:|"]
    for mode, data in result["latency"].items():
        lines.append(f"|{mode}|{data['notifications']}|{data['p95_ms']:.6f}|"
                     f"{data['inference_p95_ms']:.6f}|{data['maximum_ms']:.6f}|")
    return lines


def write_summary(result: dict[str, Any]) -> None:
    """本番登録をせず機械判定の全表を保存する。"""
    lines = ["# G_fe user局面 段2", "", "状態: COMPLETE。採否は統括（Claude）。本番登録なし。", "",
        "事前登録の案C条件1〜4・6をそのまま使用。セット1=57試合、セット2=58試合、既存5記録。",
        "参照文書に後から追加された門v2は適用していない。今回の依頼の「条件・構成を変えない」に従う。",
        "V3=HistGBC、V1b=LR。どちらも54列。全1,035,639行 / 7,686試合 / 144動画で最終学習。",
        "試合単位の対bootstrap 10,000回・seed 0。G_fe時間は特徴生成から推論完了まで。",
        "通知はG_fe計算のあるupdate。1通知内の静止・現在・仮想着弾のG_fe呼出し時間を合計する。"
        "呼出し単位のP95もMETRICS.jsonに併記。", ""]
    lines += comparison_lines(result) + regression_lines(result)
    lines += ["", "|構成|条件1|条件2|条件3|条件4|条件6|全条件|", "|---|---|---|---|---|---|---|"]
    for mode, data in result["decisions"].items():
        values = [data["gates"][f"condition{i}"] for i in (1, 2, 3, 4, 6)] + [data["passed"]]
        lines.append(f"|{mode}|" + "|".join("成立" if v else "不成立" for v in values) + "|")
    lines += ["", "既定OFFの表示NPZ・イベントJSONLは各記録で本番保存値とのバイト一致を検査。",
        "M0・S′・本番設定は不変。大きいモデル・再生原票はコミット対象外。",
        "固定0f28a04版証明器にはearly_exit引数が無いため、Falseのみ受け付ける実験内アダプタを使用。"
        "証明メソッドは無変更で継承し、Trueは拒否する（INITIAL_API_STOP.jsonに起動停止履歴）。"]
    if "timing" in result:
        lines += ["", f"全行学習: {result['timing']['training_seconds']:.2f}秒。"
                  f"再生・採点の経過時間: {result['timing']['replay_and_score_seconds']:.2f}秒。"]
    (OUT / "SUMMARY.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
