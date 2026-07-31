"""画面「N れんさ!」OCR を 23動画の全 FireEvent に適用し、simulate() 由来の
chain_count とのズレを集計する (全動画版・重い処理)。

## 実行タイミングに関する重要な注意 (2026-07-29 userタスク指定)

**このスクリプトは now 実行しないこと。** 追加収集14ジョブが完走する
23:53頃以降に実行すること (CPU競合回避のため)。今は
scripts/_verify_chain_count_screen_read_c54_2026-07-29.py (c54 単体、軽量)
のみ実行済み。

実行時は以下を厳守:
    - nice -n 19 で実行する。
    - 並列は使わない (本スクリプトは単一プロセス・逐次処理の設計)。
    - 動画ファイルが CLAUDE.md のストレージ管理ルールにより削除済みの場合が
      ある (処理後削除の運用)。存在しない動画は SKIP して継続する
      (存在チェックは _iter_target_videos 内で実施済み)。

## 処理内容

23動画 (data/indicators_v2/boards_lean_fixed_regen_2026-07-28 に npz がある
もの) の de-frag 後 FireEvent 全件について:
    1. 旧npz (boards_lean_fixed) での simulate() chain_count / 整合性
    2. 新npz (boards_lean_fixed_regen_2026-07-28) での simulate() chain_count
       / 整合性
    3. 画面OCR (ChainCountOcr) の window内最大連鎖数
を突き合わせ、CSV に出力する。

## 想定コスト (実行前に把握しておくこと)

- 対象動画: 23本 (data/indicators_v2/boards_lean_fixed_regen_2026-07-28/*.npz)
- 対象イベント: 514件 (score有効イベント、_verify_score_consistency_2026-07-29.py
  の実測値)。ただし動画ファイル自体が削除済みの動画は画面OCRをSKIPする
  (削除済みなら simulate()側の比較のみ出力)。
- 1イベントあたりの動画window探索: 経験値で 5〜11秒程度の区間を 0.05秒間隔で
  サンプリング (約100〜230フレーム)。c54の1イベントの実測で数秒程度で完了。
  514件全体では動画shrink待ち時間・シーク コストが支配的になるため、
  実行前に対象動画本数×イベント数から所要時間を見積もること
  (本スクリプトは概算表示のみ行い、実行はしない)。

## 既知の制約 (実行前に必ず確認すること)

src/chain_count_ocr.py の docstring 「既知の制約」を参照。特に:
    - digit_5〜digit_9 のテンプレが未整備 (5連鎖以上は検出不能)。
    - 誤検出 (無関係な連鎖ステップ中に別の桁を弱く誤検出) が観測されており、
      max 集計方式は「真の最大値より小さい誤検出」には強いが「真の最大値より
      大きい誤検出」が起きると結果を汚染しうる (video_c54 では偶然発生せず)。
      全動画結果は鵜呑みにせず、screen_chain_count が simulate() 由来の値と
      大きく乖離するケースは個別に実フレームで目視確認すること。

使い方 (23:53頃以降、収集ジョブ完走を確認してから):
    nice -n 19 PYTHONPATH=. ./venv/bin/python \
        scripts/_verify_chain_count_ocr_full_corpus_2026-07-29.py \
        --out data/verify/chain_count_ocr_full_corpus_2026-07-29.csv
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import pandas as pd

PROJ_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ_ROOT))

from src.board import Board  # noqa: E402
from src.chain import ChainSimulator  # noqa: E402
from src.chain_count_ocr import ChainCountOcr, _approx_min_chain_score  # noqa: E402
from src.scoring import calculate_chain_score, is_score_consistent  # noqa: E402
from scripts.measure_exchange_dynamics import (  # noqa: E402
    NPZ_DIR, SCORE_MISSING_SENTINEL, TIER_MAP, FireEvent, _load_npz,
    _process_video, _subset,
)

NPZ_DIR_REGEN: Path = PROJ_ROOT / "data" / "indicators_v2" / "boards_lean_fixed_regen_2026-07-28"
VIDEO_DIR: Path = PROJ_ROOT / "data" / "frames"

# 画面OCR window の前後バッファ (秒)。c54実測 (t_fire - t_chain_start ≈ 10秒、
# ポップアップ演出は t_fire 手前で終わる) を踏まえた保守的な値。
WINDOW_START_LEAD_SEC: float = 0.2
WINDOW_END_BUFFER_SEC: float = 1.0

DEFAULT_OUT_CSV: Path = PROJ_ROOT / "data" / "verify" / "chain_count_ocr_full_corpus_2026-07-29.csv"


def _video_path_for_stem(stem: str) -> Path | None:
    """video_{stem}.mp4 を data/frames/ から探す (削除済みなら None)。"""
    p = VIDEO_DIR / f"video_{stem}.mp4"
    return p if p.exists() else None


def _events_for_stem(npz_dir: Path, stem: str) -> dict[tuple[str, int, float], FireEvent]:
    """stem の de-frag 後 FireEvent を (side, game_idx, t_chain_start概算) キーで返す。"""
    sim = ChainSimulator()
    npz_path = npz_dir / f"{stem}.npz"
    if not npz_path.exists():
        return {}
    _, defrag, _ = _process_video(npz_path, sim, 0)
    return {(e.fire_side, e.game_idx, round(e.t_chain_start, 1)): e for e in defrag}


def _reconstruct_before_board(npz_path: Path, ev: FireEvent) -> Board | None:
    """FireEvent から before_board (before_idx 時点の grid) を復元する。

    scripts/_verify_score_consistency_2026-07-29.py と同一手順 (再利用のため
    複製、既存ファイルは変更しない)。
    """
    records = _load_npz(npz_path)
    by_side = {r.side: r for r in records}
    if ev.fire_side not in by_side:
        return None
    rec = by_side[ev.fire_side]
    mask = rec.game_idx == ev.game_idx
    g = _subset(rec, mask)
    if ev.before_idx < 0 or ev.before_idx >= len(g.t_sec):
        return None
    return Board.from_list(g.grids[ev.before_idx].tolist())


def _exact_consistent(
    sim: ChainSimulator, npz_path: Path, ev: FireEvent | None,
) -> tuple[int | None, bool | None]:
    """simulate(before_board) による厳密な期待得点と整合可否を返す。

    (49.3% 不整合の元となった _verify_score_consistency_2026-07-29.py の
    判定方式と完全に同一 (ChainStep 内訳込みの厳密計算)。screen_consistent
    (近似式ベース) と方式が異なる点に注意 — 本ファイル docstring 参照。
    """
    if ev is None:
        return None, None
    before = _reconstruct_before_board(npz_path, ev)
    if before is None:
        return None, None
    expected = calculate_chain_score(sim.simulate(before)).total_score
    return expected, is_score_consistent(expected, ev.delta_score)


def _process_one_video(
    stem: str, ocr: ChainCountOcr, progress_every: int = 5,
) -> list[dict]:
    """1動画分の全 FireEvent を処理し、比較行のリストを返す。

    progress_every: この件数ごとにイベント内進捗を print する (実行時間の
        内訳把握用、コスト見積もりが目的。0以下で無効化、backwards compat)。
    """
    new_events = _events_for_stem(NPZ_DIR_REGEN, stem)
    old_events = _events_for_stem(NPZ_DIR, stem)
    video_path = _video_path_for_stem(stem)
    cap = cv2.VideoCapture(str(video_path)) if video_path is not None else None
    sim = ChainSimulator()
    new_npz_path = NPZ_DIR_REGEN / f"{stem}.npz"
    old_npz_path = NPZ_DIR / f"{stem}.npz"
    rows: list[dict] = []
    n_events = len(new_events)
    t_video0 = time.time()
    for ev_i, (key, ev_new) in enumerate(new_events.items(), 1):
        if progress_every > 0 and ev_i % progress_every == 0:
            print(f"    ...{stem}: event {ev_i}/{n_events} "
                  f"({time.time() - t_video0:.1f}秒経過)", flush=True)
        if ev_new.delta_score == SCORE_MISSING_SENTINEL:
            continue
        ev_old = old_events.get(key)
        # 厳密 simulate() ベースの整合性 (49.3% 不整合の判定と同一方式)。
        new_expected, new_consistent = _exact_consistent(sim, new_npz_path, ev_new)
        old_expected, old_consistent = _exact_consistent(sim, old_npz_path, ev_old)
        screen_max: int | None = None
        n_hits = 0
        method = None
        score_ratio = None
        if cap is not None:
            t_start = ev_new.t_chain_start - WINDOW_START_LEAD_SEC
            t_end = ev_new.t_fire + WINDOW_END_BUFFER_SEC
            # 得点裏取り方式 (2026-07-29 追加) を使う。delta_score を渡さないと
            # 旧・連続列方式 (video_c54 2P側で 9→3 に過小評価する既知の弱点) の
            # ままになるため、ここで必ず渡す (userタスク指定の確認事項1)。
            result = ocr.read_max_in_window(
                cap, ev_new.fire_side, t_start, t_end, delta_score=ev_new.delta_score,
            )
            screen_max = result.max_chain_count
            n_hits = result.n_hits
            method = result.method
            score_ratio = result.score_ratio
        # screen (OCR) 側は ChainStep 内訳を持たないため、得点裏取り方式と
        # 同じ下限近似 (_approx_min_chain_score) で is_score_consistent を
        # 判定する (new/old の厳密判定とは近似精度が異なる点に注意、
        # 本ファイル docstring 参照)。
        screen_consistent = (
            is_score_consistent(_approx_min_chain_score(screen_max), ev_new.delta_score)
            if screen_max is not None else None
        )
        rows.append({
            "video_stem": stem, "tier": TIER_MAP.get(stem, "不明"),
            "side": ev_new.fire_side, "game_idx": ev_new.game_idx,
            "t_chain_start": ev_new.t_chain_start, "t_fire": ev_new.t_fire,
            "delta_score": ev_new.delta_score,
            "old_chain_count": ev_old.chain_count if ev_old else None,
            "old_expected_score": old_expected,
            "old_consistent": old_consistent,
            "new_chain_count": ev_new.chain_count,
            "new_expected_score": new_expected,
            "new_consistent": new_consistent,
            "screen_chain_count": screen_max,
            "screen_n_hits": n_hits,
            "screen_method": method,
            "screen_score_ratio": score_ratio,
            "screen_consistent": screen_consistent,
            "video_available": video_path is not None,
        })
    if cap is not None:
        cap.release()
    return rows


def _print_cost_estimate() -> None:
    """実行前の概算コスト表示 (npz本数・イベント数のみ、軽量なメタデータ走査)。"""
    stems = sorted(p.stem for p in NPZ_DIR_REGEN.glob("*.npz"))
    n_video_present = sum(1 for s in stems if _video_path_for_stem(s) is not None)
    print(f"[概算] 対象動画 (npz): {len(stems)}本、うち動画ファイル現存: "
          f"{n_video_present}本 (削除済みは画面OCRをSKIP)", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_CSV)
    parser.add_argument(
        "--dry-run", action="store_true",
        help="コスト概算のみ表示して終了 (実際のOCR処理はしない)",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="先頭N本の動画のみ処理する (backwards compat、既定は全動画)。"
             "実行時間の事前見積もりに使う。",
    )
    args = parser.parse_args()

    _print_cost_estimate()
    if args.dry_run:
        return

    stems = sorted(p.stem for p in NPZ_DIR_REGEN.glob("*.npz"))
    if args.limit is not None:
        stems = stems[: args.limit]
    ocr = ChainCountOcr.load_default()
    all_rows: list[dict] = []
    args.out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for i, stem in enumerate(stems, 1):
        rows = _process_one_video(stem, ocr)
        all_rows.extend(rows)
        elapsed = time.time() - t0
        print(f"[{i}/{len(stems)}] {stem}: {len(rows)}件処理 (累計 {elapsed:.1f}秒)", flush=True)
        # 途中経過チェックポイント保存 (数時間規模のジョブのため、中断・クラッシュ
        # 時にもそれまでの結果を失わないようにする。最終書き込みと同じパスに
        # 上書きするため、正常完了時の挙動は変わらない (backwards compat)。
        pd.DataFrame(all_rows).to_csv(args.out, index=False)

    df = pd.DataFrame(all_rows)
    df.to_csv(args.out, index=False)
    print(f"\n保存: {args.out} ({len(df)}行)")

    if len(df):
        n_total = len(df)
        n_video_avail = int(df["video_available"].sum())
        n_read = int(df["screen_chain_count"].notna().sum())
        print(f"\n[読み取り率] 全{n_total}件中、動画現存={n_video_avail}件、"
              f"OCR読取成功={n_read}件 ({n_read / max(1, n_video_avail):.1%} of 動画現存)")

        both_new = df[df["screen_chain_count"].notna() & df["new_chain_count"].notna()]
        if len(both_new):
            agree_new = (both_new["screen_chain_count"] == both_new["new_chain_count"]).mean()
            print(f"[一致率] OCR vs 新npz simulate chain_count: {agree_new:.1%} (n={len(both_new)})")

        for label, col in [("旧npz simulate", "old_consistent"), ("新npz simulate", "new_consistent"),
                            ("画面OCR(近似式)", "screen_consistent")]:
            sub = df[df[col].notna()]
            if len(sub):
                incons = float((~sub[col].astype(bool)).mean())
                print(f"[得点不整合率] {label}: {incons:.1%} (n={len(sub)})")

        print("\n[連鎖数帯別 OCR読取成功率] (new_chain_count 基準、催促域1-4 / 本線域5+)")
        for lo, hi, name in [(1, 4, "1-4連鎖(催促域)"), (5, 19, "5連鎖以上(本線域)")]:
            sub = df[(df["new_chain_count"] >= lo) & (df["new_chain_count"] <= hi)
                     & (df["video_available"])]
            if len(sub):
                rate = sub["screen_chain_count"].notna().mean()
                print(f"  {name}: 読取率 {rate:.1%} (n={len(sub)})")

        print("\n[1P/2P別 OCR読取成功率]")
        for side in ("1P", "2P"):
            sub = df[(df["side"] == side) & (df["video_available"])]
            if len(sub):
                rate = sub["screen_chain_count"].notna().mean()
                print(f"  {side}: 読取率 {rate:.1%} (n={len(sub)})")


if __name__ == "__main__":
    main()
