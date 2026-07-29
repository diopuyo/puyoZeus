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

from src.chain import ChainSimulator  # noqa: E402
from src.chain_count_ocr import ChainCountOcr  # noqa: E402
from scripts.measure_exchange_dynamics import (  # noqa: E402
    NPZ_DIR, SCORE_MISSING_SENTINEL, TIER_MAP, FireEvent, _process_video,
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


def _process_one_video(stem: str, ocr: ChainCountOcr) -> list[dict]:
    """1動画分の全 FireEvent を処理し、比較行のリストを返す。"""
    new_events = _events_for_stem(NPZ_DIR_REGEN, stem)
    old_events = _events_for_stem(NPZ_DIR, stem)
    video_path = _video_path_for_stem(stem)
    cap = cv2.VideoCapture(str(video_path)) if video_path is not None else None
    rows: list[dict] = []
    for key, ev_new in new_events.items():
        if ev_new.delta_score == SCORE_MISSING_SENTINEL:
            continue
        ev_old = old_events.get(key)
        screen_max: int | None = None
        n_hits = 0
        if cap is not None:
            t_start = ev_new.t_chain_start - WINDOW_START_LEAD_SEC
            t_end = ev_new.t_fire + WINDOW_END_BUFFER_SEC
            result = ocr.read_max_in_window(cap, ev_new.fire_side, t_start, t_end)
            screen_max = result.max_chain_count
            n_hits = result.n_hits
        rows.append({
            "video_stem": stem, "tier": TIER_MAP.get(stem, "不明"),
            "side": ev_new.fire_side, "game_idx": ev_new.game_idx,
            "t_chain_start": ev_new.t_chain_start, "t_fire": ev_new.t_fire,
            "delta_score": ev_new.delta_score,
            "old_chain_count": ev_old.chain_count if ev_old else None,
            "new_chain_count": ev_new.chain_count,
            "screen_chain_count": screen_max,
            "screen_n_hits": n_hits,
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
          f"{n_video_present}本 (削除済みは画面OCRをSKIP)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_CSV)
    parser.add_argument(
        "--dry-run", action="store_true",
        help="コスト概算のみ表示して終了 (実際のOCR処理はしない)",
    )
    args = parser.parse_args()

    _print_cost_estimate()
    if args.dry_run:
        return

    stems = sorted(p.stem for p in NPZ_DIR_REGEN.glob("*.npz"))
    ocr = ChainCountOcr.load_default()
    all_rows: list[dict] = []
    t0 = time.time()
    for i, stem in enumerate(stems, 1):
        rows = _process_one_video(stem, ocr)
        all_rows.extend(rows)
        elapsed = time.time() - t0
        print(f"[{i}/{len(stems)}] {stem}: {len(rows)}件処理 (累計 {elapsed:.1f}秒)")

    df = pd.DataFrame(all_rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"\n保存: {args.out} ({len(df)}行)")

    if len(df):
        both = df[df["screen_chain_count"].notna() & df["old_chain_count"].notna()]
        if len(both):
            agree_old = (both["screen_chain_count"] == both["old_chain_count"]).mean()
            print(f"画面OCR vs 旧npz(整合実績あり) 一致率: {agree_old:.1%} (n={len(both)})")
        diverge = df[df["screen_chain_count"] != df["new_chain_count"]]
        print(f"画面OCR が新npz chain_count と異なる件数: {len(diverge)}/{len(df)}")


if __name__ == "__main__":
    main()
