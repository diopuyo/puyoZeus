"""148本再収集の進捗断面 (30/60/100/144本) ごとに新旧構成を A/B 比較する (2026-08-18)。

## 背景・目的
148本の再収集 (2026-08-18 新本番構成、8フラグ全群ON、
`docs/RECOLLECT148_2026-08-18.md` 参照) が完了するのは 8/19 夕方見込み。
それを待たず「新構成 (2026-08-18) は旧構成 (2026-08-11) より学習結果を
改善するか」を早期に知りたい。60本ちょうどを待つのではなく、
**収集済み本数が節目 (30/60/100/144本) に達するたびに学習を回し、
本数が増えるにつれて改善しているのか頭打ちなのかを推移で見る**
(user指示、2026-08-18 追加変更)。

## 設計 (対応のある比較、本数差の交絡を排除。各断面で不変)
1. 新構成npz (`data/indicators_v2/boards_lean_phase_l_2026-08-18/`) から
   `data/verify/regen_2026-08-18/status.tsv` の `collect_status=OK` を
   ポーリングし、断面本数Nに達したら「最初にOKになったN本」を選ぶ
   (`finished_at` 昇順、先着順で決定的。かつ既に完了済み断面のN本は
   常により大きい断面のN'本の部分集合になる = 断面間で入れ替わらない)。
2. 選ばれたN本のうち、旧CSV (`labeled_win_full148_2026-08-14.csv`) に
   実在する video_id を持つものだけを採用する。旧CSVは144本
   (`BROKEN_VIDEOS`=c26/c30/c58/c69 を除外済み) しか無いため、この4本は
   候補から事前除外し、旧CSV側の実在確認も行う。したがって理論上の
   最大本数は144本 (最後の断面がこれに相当、それ以上は要求しない)。
3. 旧構成側は **CSVビルドを再実行しない** (2026-08-14実測で4.4時間かかる
   重い処理。旧npz は既にCSV化済みで、該当N本の行を旧CSVから抽出する
   だけでよい)。
4. 新構成側は該当N本のnpzだけをスナップショット (ハードリンク、フォール
   バックはコピー) したディレクトリを作り、そこに対して
   `build_labeled_win_from_npz` を実行してCSVを作る。

## `--exclude-match-end-locked` の非対称性の扱い (判断根拠)
旧CSV (2026-08-14) はこのフラグ無しでビルドされている。新CSVをこのフラグ
付きでビルドすると「認識構成の差」と「学習データビルドオプションの差」が
両方混ざり、どちらが効いたのか切り分けられなくなる。
**判断: 新構成側は2種類ビルドし、二軸で見る**
  (a) `no_lock` (フラグ無し): 旧CSVとビルドオプションを完全一致させ、
      **認識構成の差だけ**を測る主比較。
  (b) `with_lock` (フラグ有り): 2026-08-18 に標準採用された実際の本番
      ビルド設定 (`src/production_config.py` の
      `LEARNING_DATA_BUILD_ADOPTED`) を反映した、**実際に148本で使う
      予定の設定**での参考値。認識構成差とビルドオプション差が混ざる
      ことを承知の上で、実運用に近い見立てとして併記する。
どちらか一方を「答え」とせず、(a) を主判定・(b) を参考値として
レポートに明記する。

なお `no_lock` 側にも新規列 (W12系など、2026-08-14以降にCSVビルダーへ
追加された列) が旧CSVには無い形で入り込む非対称は残る。これは本比較の
主目的 (認識構成8フラグの効果) とは別の変化であり、実際の148本移行でも
同時に起こる変化のため、ここで無理に列を揃えない。レポートにその旨を
明記するに留める。

## 断面設計 (2026-08-18 追加変更)
STAGE_SIZES=(30, 60, 100, 144) の各断面で、揃い次第すぐ学習する
(60本到達を待たずに30本時点で先行実行)。最終断面144は旧CSVの
最大本数と一致するため「全数」を意味する。各断面が終わるたびに、
それまでに完了した断面を並べた推移レポート (本数→AUC) を書き直す。

## 実行制御・優先度 (重要)
- CSVビルド・学習は `nice -n 19` で実行する。148再収集が14並列で
  CPUを飽和させており、**さらに別の動画解析ジョブ (1時間57分・
  1920x1080@60fps) も並行して走る予定** (user補足、2026-08-18)。
  CPUの取り合いが起きる前提で、本スクリプトは「終わらなくても困らない」
  姿勢で組む: 各断面の待機はタイムアウトしても例外で落とさず、そこまで
  出来た断面だけで推移レポートを更新して終了する
- 断面ごとに前段の出力 (selected_targets.tsv / npz_snapshot / CSV /
  retrain結果) が既にあれば再利用し、再計算しない (中断・再実行に強い)
- **このファイルを書いた時点ではまだ重い処理を実行しない**
  (2026-08-18 コーダタスクの指示)

## 使い方 (WSL detach、長時間放置前提)
    wsl -d Ubuntu -- bash -c "cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer && \\
      setsid -f bash scripts/_run_ab_stage_compare_2026-08-18.sh < /dev/null"

軽量部分だけ試す場合 (小さいテスト用tsv/npzディレクトリで自己診断):
    python scripts/_ab_stage_compare_2026-08-18.py --dry-run --stage-sizes 2,3 \\
      --status-tsv <テスト用tsv> --new-npz-dir <テスト用npzディレクトリ>
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

_PROJ_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))

DEFAULT_STATUS_TSV = "data/verify/regen_2026-08-18/status.tsv"
DEFAULT_NEW_NPZ_DIR = "data/indicators_v2/boards_lean_phase_l_2026-08-18"
DEFAULT_OLD_CSV = "data/verify/labeled_win_full148_2026-08-14/labeled_win_full148.csv"
DEFAULT_STAGE_OUT_PREFIX = "data/verify/retrain_stage_N"
DEFAULT_TREND_OUT_DIR = "data/verify/ab_stage_trend_2026-08-18"
DEFAULT_RETRAIN_SCRIPT = "scripts/_retrain148_2026-08-14.py"

# user指示 (2026-08-18): 60本ちょうどではなく、この4断面で逐次実行する。
DEFAULT_STAGE_SIZES: tuple[int, ...] = (30, 60, 100, 144)

# status.tsv は148収集プロセスが書くファイルなので軽くしか触らない
# (10分間隔、148収集+並行する動画解析ジョブのCPUを邪魔しないため)。
DEFAULT_POLL_INTERVAL_SEC: float = 600.0
# 「終わらなくても困らない」(user、2026-08-18) 姿勢のため長めに待つ。
# タイムアウトしても例外で落とさず、そこまでの断面で推移レポートを出す。
DEFAULT_MAX_WAIT_HOURS: float = 48.0
# 旧CSVの巨大さ (1,035,644行) をメモリに一度に載せないためのチャンクサイズ。
OLD_CSV_CHUNK_SIZE: int = 200_000

VARIANT_CHOICES = ("both", "no_lock", "with_lock")


class StageWaitTimeout(Exception):
    """断面待機がタイムアウトしたことを示す (致命的エラーではない)。"""


def _load_broken_videos() -> tuple[str, ...]:
    """`scripts.build_labeled_win_from_npz.BROKEN_VIDEOS` を遅延importで取得する。"""
    from scripts.build_labeled_win_from_npz import BROKEN_VIDEOS
    return BROKEN_VIDEOS


def _read_ok_rows(status_tsv: Path) -> list[dict]:
    """status.tsv から collect_status=OK の行を読む (target_id重複時は最初の行)。"""
    if not status_tsv.exists():
        return []
    seen: dict[str, dict] = {}
    with status_tsv.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if row.get("collect_status") != "OK":
                continue
            tid = row["target_id"]
            if tid not in seen:
                seen[tid] = row
    return list(seen.values())


def _load_known_old_video_ids(old_csv: Path) -> set:
    """旧CSVの video_id 列のユニーク値集合を、1列だけ読んで軽量に取得する。"""
    col = pd.read_csv(old_csv, usecols=["video_id"])["video_id"]
    return set(col.unique())


def select_paired_target_ids(
    ok_rows: list[dict], npz_dir: Path, known_old_video_ids: set,
    broken_videos: tuple[str, ...], min_ok: int,
) -> list[str] | None:
    """新構成OK行から、旧CSVにも実在しnpzファイルもある候補を先着順にmin_ok本選ぶ。

    3条件 (BROKEN_VIDEOS除外/旧CSV実在/npz実在) を満たす候補が min_ok に
    満たない場合は None を返す (呼び出し側でポーリング続行の合図とする)。
    """
    candidates = []
    for row in ok_rows:
        tid = row["target_id"]
        if tid in broken_videos:
            continue
        if f"video_{tid}" not in known_old_video_ids:
            continue
        if not (npz_dir / f"{tid}.npz").exists():
            continue
        candidates.append(row)
    if len(candidates) < min_ok:
        return None
    candidates.sort(key=lambda r: r["finished_at"])
    return [r["target_id"] for r in candidates[:min_ok]]


def count_paired_candidates(
    ok_rows: list[dict], npz_dir: Path, known_old_video_ids: set,
    broken_videos: tuple[str, ...],
) -> int:
    """現時点で対応が取れている候補本数を数える (待機ログ表示用)。"""
    return sum(
        1 for r in ok_rows
        if r["target_id"] not in broken_videos
        and f"video_{r['target_id']}" in known_old_video_ids
        and (npz_dir / f"{r['target_id']}.npz").exists()
    )


def wait_for_paired_target_ids(
    status_tsv: Path, npz_dir: Path, known_old_video_ids: set,
    broken_videos: tuple[str, ...], min_ok: int,
    poll_interval_sec: float, max_wait_hours: float,
) -> list[str]:
    """断面本数min_okが揃うまで軽くポーリングする (揃わなければ StageWaitTimeout)。"""
    deadline = time.time() + max_wait_hours * 3600.0
    while True:
        ok_rows = _read_ok_rows(status_tsv)
        selected = select_paired_target_ids(
            ok_rows, npz_dir, known_old_video_ids, broken_videos, min_ok,
        )
        if selected is not None:
            print(f"[wait] {min_ok}本の対応が揃いました (全OK={len(ok_rows)}本)",
                  flush=True)
            return selected
        n_valid = count_paired_candidates(ok_rows, npz_dir, known_old_video_ids, broken_videos)
        print(f"[wait] 対応可能={n_valid}/{min_ok} (全OK={len(ok_rows)}本)"
              f" ... {poll_interval_sec:.0f}秒後に再確認", flush=True)
        if time.time() > deadline:
            raise StageWaitTimeout(
                f"{max_wait_hours}時間待っても{min_ok}本揃いませんでした"
                f" (対応可能={n_valid}本)",
            )
        time.sleep(poll_interval_sec)


def write_selected_targets_tsv(
    ok_rows: list[dict], target_ids: list[str], out_path: Path,
) -> None:
    """選定したN本の対応表を監査用に書き出す。"""
    by_id = {r["target_id"]: r for r in ok_rows}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="\n") as f:
        f.write("target_id\tvideo_id\ttier\tfinished_at\trows\n")
        for tid in target_ids:
            r = by_id[tid]
            f.write(f"{tid}\t{r['video_id']}\t{r['tier']}\t"
                    f"{r['finished_at']}\t{r['rows']}\n")


def snapshot_npz_subset(src_dir: Path, dst_dir: Path, target_ids: list[str]) -> None:
    """新構成npzのうち target_ids 分だけを dst_dir にハードリンク (不可ならコピー) する。"""
    dst_dir.mkdir(parents=True, exist_ok=True)
    for tid in target_ids:
        src = src_dir / f"{tid}.npz"
        dst = dst_dir / f"{tid}.npz"
        if dst.exists():
            continue
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)
    print(f"[snapshot] {len(target_ids)}本を {dst_dir} へ配置", flush=True)


def extract_old_csv_subset(old_csv: Path, target_ids: list[str], out_csv: Path) -> int:
    """旧CSVから target_ids 該当行だけをチャンク読み込みで抽出する (CSVビルド不要)。"""
    wanted = {f"video_{tid}" for tid in target_ids}
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    n_rows = 0
    first = True
    for chunk in pd.read_csv(old_csv, chunksize=OLD_CSV_CHUNK_SIZE):
        sub = chunk[chunk["video_id"].isin(wanted)]
        if len(sub) == 0:
            continue
        sub.to_csv(out_csv, mode="w" if first else "a", header=first, index=False)
        first = False
        n_rows += len(sub)
    found: set = set()
    if not first:
        found = set(pd.read_csv(out_csv, usecols=["video_id"])["video_id"].unique())
    missing = wanted - found
    if missing:
        raise RuntimeError(f"旧CSVに見つからない video_id があります: {sorted(missing)}")
    print(f"[extract_old] {out_csv} に {n_rows}行 ({len(target_ids)}本) を書き出し", flush=True)
    return n_rows


def _nice_prefix(use_nice: bool) -> list[str]:
    if use_nice and shutil.which("nice") is not None:
        return ["nice", "-n", "19"]
    return []


def build_new_csv(
    npz_dir: Path, out_csv: Path, exclude_match_end_locked: bool, use_nice: bool = True,
) -> None:
    """スナップショットnpzから新CSVをビルドする (`build_labeled_win_from_npz` 呼び出し)。"""
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    cmd = _nice_prefix(use_nice) + [
        sys.executable, "-m", "scripts.build_labeled_win_from_npz",
        "--npz-dir", str(npz_dir), "--out", str(out_csv), "--profile", "full",
    ]
    if exclude_match_end_locked:
        cmd.append("--exclude-match-end-locked")
    print(f"[build_new_csv] 実行: {' '.join(cmd)}", flush=True)
    env = {**os.environ, "PYTHONPATH": str(_PROJ_ROOT)}
    subprocess.run(cmd, check=True, cwd=str(_PROJ_ROOT), env=env)


def run_retrain(
    csv_path: Path, out_dir: Path, skip_perm: bool = True, use_nice: bool = True,
) -> None:
    """`_retrain148_2026-08-14.py` を --csv/--out-dir 明示指定で呼ぶ (既定パス上書き事故防止)。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = _nice_prefix(use_nice) + [
        sys.executable, str(_PROJ_ROOT / DEFAULT_RETRAIN_SCRIPT),
        "--csv", str(csv_path), "--out-dir", str(out_dir),
    ]
    if skip_perm:
        cmd.append("--skip-perm")
    print(f"[run_retrain] 実行: {' '.join(cmd)}", flush=True)
    env = {**os.environ, "PYTHONPATH": str(_PROJ_ROOT)}
    subprocess.run(cmd, check=True, cwd=str(_PROJ_ROOT), env=env)


def _summary_or_none(out_dir: Path) -> dict | None:
    p = out_dir / "summary.json"
    if not p.exists():
        return None
    with p.open(encoding="utf-8") as f:
        return json.load(f)


def stage_out_dir(stage_prefix: str, n: int) -> Path:
    """断面Nの出力ルート (例 data/verify/retrain_stage_N30_2026-08-18)。"""
    return Path(f"{stage_prefix}{n}_2026-08-18")


def stage_is_done(stage_dir: Path, variants: str) -> bool:
    """断面がold + 要求variantsぶん完了済みかを summary.json の有無で判定する。"""
    needed = ["old"]
    if variants in ("both", "no_lock"):
        needed.append("new_no_lock")
    if variants in ("both", "with_lock"):
        needed.append("new_with_lock")
    return all((stage_dir / name / "summary.json").exists() for name in needed)


def run_one_stage(
    stage_n: int, stage_dir: Path, args, known_old_video_ids: set,
    broken_videos: tuple[str, ...],
) -> None:
    """断面1つ分 (選定→抽出/スナップショット→CSVビルド→学習) を実行する。"""
    manifest = stage_dir / "selected_targets.tsv"
    if manifest.exists() and not args.force_reselect:
        with manifest.open(encoding="utf-8") as f:
            rows = list(csv.DictReader(f, delimiter="\t"))
        target_ids = [r["target_id"] for r in rows]
        print(f"[stage {stage_n}] 既存manifestを再利用 ({len(target_ids)}本)", flush=True)
    else:
        target_ids = wait_for_paired_target_ids(
            Path(args.status_tsv), Path(args.new_npz_dir), known_old_video_ids,
            broken_videos, stage_n, args.poll_interval_sec, args.max_wait_hours,
        )
        ok_rows = _read_ok_rows(Path(args.status_tsv))
        write_selected_targets_tsv(ok_rows, target_ids, manifest)

    old_csv_subset = stage_dir / "labeled_win_old.csv"
    if not old_csv_subset.exists() or args.force_reselect:
        extract_old_csv_subset(Path(args.old_csv), target_ids, old_csv_subset)

    npz_snap_dir = stage_dir / "npz_snapshot"
    snapshot_npz_subset(Path(args.new_npz_dir), npz_snap_dir, target_ids)

    if args.dry_run:
        print(f"[stage {stage_n}] dry-run: CSVビルド/学習はスキップ", flush=True)
        return

    run_retrain(old_csv_subset, stage_dir / "old", args.skip_perm, args.use_nice)

    if args.variants in ("both", "no_lock"):
        new_csv_no_lock = stage_dir / "labeled_win_new_no_lock.csv"
        build_new_csv(npz_snap_dir, new_csv_no_lock, False, args.use_nice)
        run_retrain(new_csv_no_lock, stage_dir / "new_no_lock", args.skip_perm, args.use_nice)
    if args.variants in ("both", "with_lock"):
        new_csv_with_lock = stage_dir / "labeled_win_new_with_lock.csv"
        build_new_csv(npz_snap_dir, new_csv_with_lock, True, args.use_nice)
        run_retrain(new_csv_with_lock, stage_dir / "new_with_lock", args.skip_perm, args.use_nice)


def build_trend_report(
    stage_sizes: tuple[int, ...], stage_prefix: str, trend_out_dir: Path,
) -> None:
    """完了済み断面を並べ、本数→AUCの推移表を書き出す (未完了断面は都度スキップ)。"""
    trend_out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for n in stage_sizes:
        stage_dir = stage_prefix_dir = stage_out_dir(stage_prefix, n)
        for variant, subdir in (
            ("old", "old"), ("new_no_lock", "new_no_lock"), ("new_with_lock", "new_with_lock"),
        ):
            s = _summary_or_none(stage_prefix_dir / subdir)
            if s is None:
                continue
            ff = s.get("full_features", {})
            rows.append({
                "n_videos_target": n, "n_videos_actual": s.get("n_videos"),
                "variant": variant, "pooled_auc": ff.get("auc"),
                "video_median_auc": ff.get("video_median_auc"),
            })
    lines = [
        "# 段階的A/B比較 推移レポート (30/60/100/144本、2026-08-18)",
        "",
        "本数が増えるにつれて改善しているか頭打ちかを見るための推移表。"
        "位相別AUCの絶対値は信用しないこと (W18)、ここは相対比較用。",
        "",
        "| 断面(目標本数) | 実本数 | 構成 | プールAUC | 動画別中央値AUC |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        pooled = r["pooled_auc"]
        med = r["video_median_auc"]
        pooled_s = f"{pooled:.4f}" if pooled is not None else "-"
        med_s = f"{med:.4f}" if med is not None else "-"
        lines.append(
            f"| {r['n_videos_target']} | {r['n_videos_actual']} | {r['variant']} |"
            f" {pooled_s} | {med_s} |",
        )
    if not rows:
        lines.append("| (まだ完了した断面がありません) | | | | |")
    (trend_out_dir / "trend_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with (trend_out_dir / "trend_summary.json").open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2, default=str)
    print(f"[trend] {trend_out_dir / 'trend_report.md'} を更新 ({len(rows)}行)", flush=True)


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--status-tsv", default=DEFAULT_STATUS_TSV)
    p.add_argument("--new-npz-dir", default=DEFAULT_NEW_NPZ_DIR)
    p.add_argument("--old-csv", default=DEFAULT_OLD_CSV)
    p.add_argument("--stage-out-prefix", default=DEFAULT_STAGE_OUT_PREFIX)
    p.add_argument("--trend-out-dir", default=DEFAULT_TREND_OUT_DIR)
    p.add_argument(
        "--stage-sizes", default=",".join(str(n) for n in DEFAULT_STAGE_SIZES),
        help="カンマ区切りの断面本数リスト (例 30,60,100,144)",
    )
    p.add_argument("--poll-interval-sec", type=float, default=DEFAULT_POLL_INTERVAL_SEC)
    p.add_argument("--max-wait-hours", type=float, default=DEFAULT_MAX_WAIT_HOURS)
    p.add_argument("--variants", choices=VARIANT_CHOICES, default="both")
    p.add_argument("--force-reselect", action="store_true", default=False)
    p.add_argument(
        "--dry-run", action="store_true", default=False,
        help="各断面の選定・スナップショット・旧CSV抽出までで停止し、"
             "CSVビルド/学習(重い処理)を実行しない",
    )
    p.add_argument(
        "--with-perm-importance", dest="skip_perm", action="store_false", default=True,
        help="既定はpermutation importanceをスキップ(早期判定・低負荷を優先)。指定で有効化。",
    )
    p.add_argument("--no-nice", dest="use_nice", action="store_false", default=True)
    p.add_argument(
        "--trend-only", action="store_true", default=False,
        help="既存の断面出力から推移レポートだけ再生成する",
    )
    return p


def main() -> int:
    args = _build_arg_parser().parse_args()
    stage_sizes = tuple(int(x) for x in args.stage_sizes.split(","))
    trend_out_dir = Path(args.trend_out_dir)

    if args.trend_only:
        build_trend_report(stage_sizes, args.stage_out_prefix, trend_out_dir)
        return 0

    broken_videos = _load_broken_videos()
    known_old_video_ids = _load_known_old_video_ids(Path(args.old_csv))
    max_achievable = len(known_old_video_ids)
    print(f"[main] 旧CSV側の対応可能上限={max_achievable}本"
          f" (BROKEN_VIDEOS除外済み)", flush=True)

    for stage_n in stage_sizes:
        effective_n = min(stage_n, max_achievable)
        if effective_n < stage_n:
            print(f"[main] 断面{stage_n}は旧CSV上限{max_achievable}に"
                  f"クランプします", flush=True)
        stage_dir = stage_out_dir(args.stage_out_prefix, stage_n)
        if stage_is_done(stage_dir, args.variants) and not args.force_reselect:
            print(f"[main] 断面{stage_n}は完了済み、スキップ", flush=True)
            build_trend_report(stage_sizes, args.stage_out_prefix, trend_out_dir)
            continue
        try:
            run_one_stage(effective_n, stage_dir, args, known_old_video_ids, broken_videos)
        except StageWaitTimeout as e:
            print(f"[main] 断面{stage_n}はタイムアウト: {e}"
                  f" (「終わらなくても困らない」方針により、ここで打ち切り"
                  f"推移レポートのみ更新して終了)", flush=True)
            build_trend_report(stage_sizes, args.stage_out_prefix, trend_out_dir)
            return 0
        build_trend_report(stage_sizes, args.stage_out_prefix, trend_out_dir)

    print("[main] 全断面完了", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
