"""E6の根因時系列とE5比較を再生出力から報告する。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REASONS = {"display_score_change": "表示得点変化", "chain_observation": "連鎖観測復帰",
           "formula_visible": "掛け算式再表示"}
SHORT = {"q_7gc4TgFig": "q", "fcXG83vInDY": "fcX", "mia8KCjr52g": "mia"}


def read(path: Path) -> dict | list:
    return json.loads(path.read_text(encoding="utf-8"))


def classification(rows: list[dict]) -> list[str]:
    """信号単位の表と、重複を含む撃ち合い数を並記する。"""
    lines = ["# E6 終了合図撤回の根因と記録再生", "",
        "統括のE6b差戻しにより合格扱いは撤回。以下は到達・滞在ゲート追加前の測定記録。", "",
        "細部決定: 消去最小40点で落下加点と再開を区別し、得点安定は独立に再確認する。"
        "NEXTは通常色のみ、式表示中は終了不可、slideは既存の式セッション猶予2秒を共用する。", "",
        "E5無改変の診断再生は3本ともNPZ・JSONLの全バイト一致。"
        "15撃ち合いにはS3後に撤回された24信号が含まれる。", "",
        "|発生元|撤回理由|信号数|撃ち合い数（重複可）|", "|---|---|---:|---:|"]
    for reason, label in REASONS.items():
        chosen = [r for r in rows if r["stop"]["reason"] == reason]
        count = len({(r["source"], r["exchange_id"]) for r in chosen})
        lines.append(f"|同側NEXT色変化|{label}|{len(chosen)}|{count}|")
    lines += ["|物理slide・新ツモ落下・累積落下量・相手側信号|全理由|0|0|", "",
        "表示得点変化18信号は+1点17件・+2点1件。通常落下加点を連鎖再開としていた。",
        "実連鎖継続の3信号（q E6/2P、fcX E5/2P・E9/2P）はNEXTに9（おじゃま）が"
        "混入。旧判定の『非Noneなら有効色』という実装と、式表示抑止の後置が原因。",
        "連鎖観測復帰4信号のうちmia E10・E11・E31は10.967/22.933/14.733秒後の"
        "別連鎖。実画面で間に通常操作があることを確認。元の終了が早かったとは限らない。", "",
        "代表フレーム: `logs/e6/diagnosis/frames/`（元動画から21枚）。"
        "入力全件: `logs/e6/diagnosis/cases_enriched.json`。", ""]
    return lines


def timeline(rows: list[dict]) -> list[str]:
    """15撃ち合いの24信号を動画内時系列で漏れなく列挙する。"""
    lines = ["## 全件時系列", "", "|動画/撃ち合い/側|終了秒|撤回秒|間隔秒|撤回理由|",
             "|---|---:|---:|---:|---|"]
    ordered = sorted(rows, key=lambda r: (list(SHORT).index(r["source"]), r["signal"]["t_sec"]))
    for row in ordered:
        label = f"{SHORT[row['source']]} E{row['exchange_id']} {row['side']}"
        reason = REASONS[row["stop"]["reason"]]
        if row["score_step"] is not None:
            reason += f"（+{row['score_step']}点）"
        lines.append(f"|{label}|{row['signal']['t_sec']:.3f}|{row['signal']['revoked_sec']:.3f}"
                     f"|{row['duration']:.3f}|{reason}|")
    return lines + [""]


def number(value: object) -> str:
    return "∞" if value == "Infinity" else f"{value:.3f}"


def metrics(out: Path) -> list[str]:
    """測定器を変更せず、E5と今回の実出力を比較する。"""
    old_root, new_root = Path("logs/e5/replay/metrics"), out / "metrics"
    old, new = [read(root / "summary.json") for root in (old_root, new_root)]
    old.append(read(old_root / "pooled.json"))
    new.append(read(new_root / "pooled.json"))
    lines = ["## 再生比較（E5 → E6）", "", "|動画|M1(i)|M2中央値秒|M3 log loss|M4反転/分|最長同値秒|",
             "|---|---|---|---|---|---|"]
    for before, after in zip(old, new):
        name = SHORT.get(before["source"], "合算")
        values = []
        for item in (before, after):
            m1 = item["M1"]
            values.append([f"{m1['revoked']}/{m1['s3_exchanges']}",
                number(item["M2"]["median_seconds"]["on"]),
                f"{item['M3']['on']['groups']['all']['log_loss']:.6f}" if "M3" in item else "qで判定",
                number(item["M4"]["on"]["flips_per_minute"]),
                number(item["F4"]["on"]["longest_equal_seconds"])])
        lines.append("|" + name + "|" + "|".join(f"{a} → {b}" for a, b in zip(*values)) + "|")
    gates = read(out / "gates.json")
    lines += ["", "E6固定ゲート: " + json.dumps(gates, ensure_ascii=False), "",
        "合算M1≤20%、M2はE5+0.2秒以内、q log lossはE5+0.005以内、反転/分はE5×1.1以内、"
        "最長同値は各動画OFF+1.0秒以内。既存E4 REPORTの鮮度判定（許容0秒）とは別に適用。", ""]
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("logs/e6/final"))
    options = parser.parse_args()
    rows = read(Path("logs/e6/diagnosis/cases_enriched.json"))
    lines = classification(rows) + timeline(rows) + metrics(options.out)
    test_log = Path("logs/e6/tests_final.log").read_text(encoding="utf-8").splitlines()[-1]
    off = read(Path("logs/e6/off_sha256.json"))
    lines += ["## 検証・再実行", "", test_log,
        f"OFF q 140〜165秒: 動画・display.npz・settled.npzのSHA-256一致 = {off['identical']}。",
        "全編レンダ・commit・pushなし。修正はONのtracker/overlay境界に限定。", "",
        "再生: `python -m scripts.run_e6_exchange_eval_20260926`（出力 `logs/e6/final`）。",
        "根因診断: `python -m scripts.diagnose_e6_exchange_20260926`"
        "（E5リビジョン8e48558をメモリ内で読み込み、作業ツリーは変更しない）。",
        "報告再生成: `python -m scripts.report_e6_exchange_20260926`。", ""]
    target = Path("docs/agent_coordination/E6_DECISIONS_2026-09-26.md")
    target.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
