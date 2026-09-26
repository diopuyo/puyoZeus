"""固定ゲートを変更せず、E6bの比較と残った不合格の内訳を記録する。"""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

import numpy as np

from scripts.run_e3_exchange_eval_20260926 import SOURCES, save_json

OUT = Path("logs/e6b")
STAGES = (("E5", OUT / "e5", Path("logs/e5/replay")),
          ("E6", OUT / "e6", Path("logs/e6/final")),
          ("E6b", OUT / "current", OUT / "current"))
DOCUMENT = Path("docs/agent_coordination/E6B_DECISIONS_2026-09-26.md")


def read(path: Path) -> dict | list:
    """保存済みの全動画集計だけを読む。"""
    return json.loads(path.read_text(encoding="utf-8"))


def metrics(name: str, reach_path: Path, metric_path: Path) -> dict:
    """内部到達と実表示到達を混ぜず、既存指標も転記する。"""
    reach = read(reach_path / "reach_summary.json")
    pooled = read(metric_path / "metrics/pooled.json")
    summaries = read(metric_path / "metrics/summary.json")
    return dict(stage=name, reach=reach, M1=pooled["M1"], M2=pooled["M2"],
        q_log_loss=summaries[0]["M3"]["on"]["groups"]["all"]["log_loss"],
        flips_per_minute=pooled["M4"]["on"]["flips_per_minute"],
        longest_equal_seconds=pooled["F4"]["on"]["longest_equal_seconds"])


def flip_sources(root: Path) -> dict:
    """M4と同じゼロ除去・試合分離を使い、反転の前後の評価元を数える。"""
    counts: Counter = Counter()
    for source in SOURCES:
        with np.load(root / "renders" / source / "on/display.npz") as data:
            for game in np.unique(data["game_idx"]):
                idx = np.flatnonzero((data["game_idx"] == game) & (data["display_adv"] != 0))
                changed = np.sign(data["display_adv"][idx[1:]]) != np.sign(data["display_adv"][idx[:-1]])
                for previous, current in zip(idx[:-1][changed], idx[1:][changed]):
                    counts[f'{data["source"][previous]}→{data["source"][current]}'] += 1
    return dict(counts)


def delay_provenance(root: Path) -> dict:
    """時刻の帰属不整合も除外せず残し、測定値を都合よく改善しない。"""
    rows = [r for source in SOURCES
            for r in read(root / "metrics" / source / "event_metrics.json")["M2"]
            if "delays" in r]
    return dict(eligible=len(rows), end_after_cutoff=sum(r["end_sec"] >= r["cutoff_sec"] for r in rows),
                target_before_end=sum(r["target_sec"] < r["end_sec"] for r in rows))


def comparison_lines(rows: list[dict]) -> list[str]:
    """完了母数と全閉鎖理由を同じ表に残す。"""
    lines = ["|段階|完了数|S3内部／実表示到達率|S1中央値(s)|発火→最後の終了(s)|正常／安全弁／境界／末尾|",
             "|---|---:|---:|---:|---:|---|"]
    for row in rows:
        r = row["reach"]["pooled"]
        c = r["close_reasons"]
        reasons = "/".join(str(c.get(k, 0)) for k in (
            "confirmed_after_score", "activity_timeout", "match_boundary", "stream_end"))
        lines.append(f'|{row["stage"]}|{r["completed"]}|{r["reach_fraction"]:.1%}／'
            f'{r["display_reach_fraction"]:.1%}|{r["s1_median_seconds"]:.3f}|'
            f'{r["firing_to_last_end_median"]:.3f}|{reasons}|')
    lines += ["", "|段階|M1(i)|M2中央値(s)|q log loss|反転/分|最長同値(s)|",
              "|---|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f'|{r["stage"]}|{r["M1"]["revoked"]}/{r["M1"]["s3_exchanges"]}'
            f' ({r["M1"]["revoked_fraction"]:.1%})|{r["M2"]["median_seconds"]["on"]:.3f}|'
            f'{r["q_log_loss"]:.6f}|{r["flips_per_minute"]:.3f}|{r["longest_equal_seconds"]:.3f}|')
    return lines


def main() -> None:
    """比較表・阻害条件・既存ゲートの残課題を一つの台帳へ保存する。"""
    rows = [metrics(*stage) for stage in STAGES]
    checks = read(OUT / "current/gates.json")
    details = {name: dict(flips=flip_sources(root), delay=delay_provenance(root))
               for name, _, root in STAGES}
    save_json(OUT / "comparison.json", dict(stages=rows, gates=checks, details=details))
    lines = ["# E6b 終了後の落下加点とS3到達", "",
        "細部決定: 完了母数は正常終了＋3秒安全弁終了、試合境界・区間末は打切りとして別記。S1は稼働中IDへ実表示フレームを割り当て、再開区間を二重計上しない。",
        "E6の合格扱いは撤回済み。E6bも全固定ゲートが通るまでは不合格。", ""]
    lines += comparison_lines(rows)
    lines += ["", "## 確定した根因と修正", "",
        "- E6では小得点変化6952回が活動時刻代入を通り、得点確定解除1043回・S3→S1復帰14回を発生。終了後の40点未満の正増分を落下加点として基準へ吸収し、連鎖得点不変フレームを継続する。40点は4個消去×10点の最小消去得点。",
        "- E6終了未確認130連鎖中90連鎖に同側の後続IDが存在。最新IDだけ観測する構造で古い参加連鎖が永久待ちになった。物理終了前のresolver断片を同じ参加連鎖へ結び付ける。",
        "- 得点確定後にS3猶予を飛び越えてG_feへ直行した。全終了・得点確定・S3実表示の後にのみ静止へ復帰する。追加参加や撤回では古いS3を無効化する。",
        "- 実画面でNEXT誤読と、NEXT移動後の別の単発を確認。色変化にDNEXT→NEXTの送りを要求し、終了後の落下加点を挟んだ発火を古い式セッションへ戻さない。",
        "- queue_history案の未終了49連鎖中22連鎖で同側OJAMA_FALLを記録。落下開始に着地会計増加を要求していたため、実落下の終了信号と会計追随を分離。相手側・会計増加だけでは終了しない。", "",
        "## 検証と全件台帳", "",
        f'固定ゲート: `{json.dumps(checks, ensure_ascii=False)}`。閾値・既存のM2/M4採点式は変更していない。',
        "各撃ち合いの発火・S3到達・S1滞在・閉鎖理由・不足条件: `logs/e6b/{e5,e6,current}/reach_events.json`。",
        "動画別集計・実表示source内訳: 同ディレクトリの `reach_summary.json`。",
        "E5/E6は保存出力とNPZ全列・ファイルバイト・events全バイト一致を確認。凍結E6コードは `logs/e6b/baseline/`。",
        "保存E5の実表示S3はq=26/fcX=9/mia=620フレーム。4000〜6000という別集計とは一致せず、本表は指定記録と8e48558の再生一致を基準とする。",
        "完了・S3未到達の阻害条件: E5=終了未確認24/猶予中4、E6=終了未確認26/猶予中3/得点未確定1、E6b=終了未確認17/得点未確定4。E6bの落下加点4455回についてタイマー・得点確定・S3のリセットはいずれも0回。",
        "中間案: bonus → reach_order → physical_identity → queue_motion → queue_history → placement_close → fall_start → current。placed_identityは正常閉鎖順序の実装不備があり不採用。",
        "実画面: `logs/e6b/frames/`。435.233→436.366秒の2PはNEXT送り後の新しい40×1、492.866→493.233秒の2Pは継続中の4連鎖、499.200秒の1Pは別の40×1。",
        "既存指標の残課題内訳: `logs/e6b/comparison.json` の details（M2時刻不整合、M4の評価元別反転）。",
        "E6bはM2=1.683秒>上限0.983秒、M4=21.156回/分>上限18.700で不合格。M2対象46件のうち9件はOCR確定が次区間開始以後、41件は目標G_feがOCR確定より前。これらを採点から除外していない。G_fe同士の反転はE5の693回から834回へ増加。",
        "関連テスト354 passed/1 skipped（E2_SHORT_ARTIFACTS未指定）: `logs/e6b/tests.log`。全ゲート不成立のためOFF25秒SHA再実行は未着手。全編レンダ・commit・pushなし。"]
    DOCUMENT.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
