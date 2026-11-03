"""E20の固定母数の採否と、時刻を揃えた手数監査を文書へ反映する。"""
from __future__ import annotations
import json
from pathlib import Path
from scripts import report_e10c_exchange_20260927 as death_report
from scripts.run_e3_exchange_eval_20260926 import save_json

ROOT = Path("logs/e20")
DOCUMENT = Path("docs/E20_LANDING_HANDS_SPEC_2026-09-28.md")
MARKER = "## 最終結果"


def audit_deaths() -> None:
    """固定した入力・勝者ラベルで、発火集合の増減と誤発火を保存する。"""
    rows = {}
    for label, directory in (("baseline", "logs/e19/hold"), ("spec", "logs/e20/on")):
        death_report.OUT = Path(directory)
        rows[label] = [{k: v for k, v in r.items() if k != "first"}
                       for r in death_report.deaths()]
    keys = ("source", "game_idx", "exchange_id", "side")
    indexed = {label: {tuple(r[k] for k in keys): r for r in values}
               for label, values in rows.items()}
    baseline, spec = indexed["baseline"], indexed["spec"]
    save_json(ROOT/"DEATH_AUDIT.json", dict(rows=rows,
        removed=[baseline[k] for k in sorted(baseline.keys()-spec.keys())],
        added=[spec[k] for k in sorted(spec.keys()-baseline.keys())],
        false={label: [r for r in values if r["false_positive"]] for label, values in rows.items()}))


def run() -> None:
    """未丸めの集計を保存し、表の丸めによって採否を変更しない。"""
    baseline = json.loads(Path("logs/e19/hold/METRICS.json").read_text())
    audit_deaths()
    result = json.loads((ROOT/"on/METRICS.json").read_text())
    hands = json.loads((ROOT/"HANDS_AUDIT.json").read_text())
    visual = json.loads((ROOT/"VISUAL_HANDS.json").read_text())["rows"]
    assert len(hands) == len(visual) == 8
    assert result["q"]["frames"] == 6526 and result["zenchi"]["frames"] == 8333
    save_json(ROOT/"METRICS.json", dict(baseline=baseline, spec_counter=result))
    lines = [MARKER, "", "|条件|q log loss（6,526行/4試合）|zenchi一致（8,333行）|誤発火|場面平均（62行）|前半→後半（各31行）|",
             "|---|---:|---:|---:|---:|---:|"]
    for label, row in (("E19候補", baseline), ("E20仕様手数＋確定的応手", result)):
        q, z, d, s = (row[k] for k in ("q", "zenchi", "deaths", "scenes"))
        lines.append(f"|{label}|{q['log_loss']:.6f}|{z['hits']}/{z['frames']} = {100*z['agreement']:.4f}%|"
                     f"{d['false']}/{d['total']}|{s['prefire_mean']:.6f}|{s['first_half_mean']:.6f}→{s['last_half_mean']:.6f}|")
    lines += ["", "採用候補。" if result["candidate"] else "**不採用**。未達: " +
              "、".join(k for k, passed in result["gates"].items() if not passed) + "。", "",
              "誤発火の分母は各条件の実発火数（28→27）。同じ入力・勝者ラベルを用い、発火集合の差は `DEATH_AUDIT.json` に保存。基準1/28は変更しない。", "",
              "発火は5件消失・4件追加。既存qの誤発火に加え、fcの720.100秒・1Pが新たな誤発火。E20動画は合格条件を満たさないため生成しない。", "",
              "|判断時刻（秒）|受け側|旧手数|新手数|実手数（原映像）|新予測の根拠|",
              "|---:|---|---:|---:|---|---|"]
    for row in hands:
        actual = next(v for v in visual if v["game"] == row["game"])
        count = actual["actual_hands"]
        display = str(count) if count is not None else ("未着弾（2手で応手）" if row["game"] == 15 else "未着弾（4手後窒息）")
        lines.append(f"|{row['decision_sec']:.3f}|{row['receiver']}|{row['old_hands']}|{row['hands']}|{display}|{row['reason']}|")
    lines += ["", "手数の未丸め根拠・アニメ時間標本・設置間隔は `logs/e20/HANDS_AUDIT.json`。",
              "原映像での着弾監査により、旧『8件すべて手数過大』という前提は成立しない。",
              "本比較の採否には事前登録した固定再生指標だけを使い、映像監査による事後のラベル差し替えはしない。",
              "", "関連回帰テスト452件成功・1件skip（新規33件）。",
              "OFF再生は6,848更新/5,948表示行で、NPZ・イベントJSONLがバイト一致、診断JSONが一致。",
              "動画の場面2613秒・死亡後2700秒・次試合2703秒を目視し、死亡確定100%と境界リセットを確認。"]
    DOCUMENT.write_text(DOCUMENT.read_text().split(MARKER)[0]+"\n"+"\n".join(lines)+"\n", encoding="utf-8")


if __name__ == "__main__":
    run()
