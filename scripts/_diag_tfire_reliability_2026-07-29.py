"""読み取り専用の集計診断: t_fire(連鎖完了時刻)検出の信頼性調査 (2026-07-29)。

user指摘: measure_ojama_landing_delay.py の anim_dur = t_fire - t_chain_start が、
recognition_diag_chain_anim_duration_multi の視覚実測 (母集団 n=418) の
同一連鎖数ビン min〜max を超過するケースが多発 (14件中6件、うち2連鎖4件中3件)。
本スクリプトは t_fire の実体 (_find_chain_windows の post_idx が何を指すか) を
該当イベントの生データ (npz の t_sec/score/game_idx) から特定する。

制約: src/, scripts/measure_exchange_dynamics.py, scripts/measure_ojama_landing_delay.py は
一切変更しない (import のみ)。動画I/Oは行わない (npz読み込み + ChainSimulator.simulate()
のみ、既存 _process_video の再利用)。CNN再推論・npz再生成はしない。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJ_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ_ROOT))

from src.chain import ChainSimulator  # noqa: E402
from scripts.measure_exchange_dynamics import (  # noqa: E402
    NPZ_DIR, TIER_MAP, FireEvent, _load_npz, _process_side_game, _defrag_events,
    _process_video, _subset,
)

# 重要 (2026-07-29発見): exchange_landing_delay_regen_2026-07-28.csv は
# scripts/_tmp_measure_landing_regen_2026-07-28.py が生成したものであり、
# NPZ_DIR (旧 boards_lean_fixed) ではなく "認識強化後に再収集した" 別ディレクトリ
# boards_lean_fixed_regen_2026-07-28 を使っている。一方 母集団実測
# (recognition_diag_chain_anim_duration_multi) は旧 NPZ_DIR (boards_lean_fixed)
# を使っている (scripts/_diag_chain_anim_duration_multi.py の import 元と同一)。
# 比較対象が異なる npz 世代である可能性を検証するため、両方で再実行する。
NPZ_DIR_REGEN: Path = PROJ_ROOT / "data" / "indicators_v2" / "boards_lean_fixed_regen_2026-07-28"

# user が報告した外れ値ケース (video_stem, fire_side, game_idx, t_chain_start概算)
TARGET_CASES: list[tuple[str, str, int, float]] = [
    ("c62", "2P", 6, 700.4),   # 21.0秒 (2連鎖、母集団max約2倍)
    ("c44", "2P", 2, 257.8),   # 16.4秒 (2連鎖)
    ("c59", "1P", 1, 261.2),   # 11.4秒 (2連鎖)
    ("c21", "2P", 2, 279.8),   # 0.20秒 (1連鎖、母集団min未満)
    ("c54", "1P", 1, 252.6),   # 8.40秒 (1連鎖、母集団maxをわずかに超過)
]


def dump_case(video_stem: str, fire_side: str, game_idx: int, t_chain_start_approx: float) -> None:
    npz_path = NPZ_DIR_REGEN / f"{video_stem}.npz"
    if not npz_path.exists():
        print(f"[WARN] npz不在: {npz_path}")
        return
    records = _load_npz(npz_path)
    by_side = {r.side: r for r in records}
    sim = ChainSimulator()

    print(f"\n{'=' * 70}\n{video_stem} / {fire_side} / game_idx={game_idx} (target t_chain_start~{t_chain_start_approx})\n{'=' * 70}")

    # --- _process_video で raw(defrag前)/defrag後 の両方を再現する ---
    # (t_chain_start近傍で全side/全game_idx突合せ: game_idxフィルタ誤りを避けるため
    #  fire_side一致のみで抽出し、時刻の近さで対象を特定する)
    raw, defrag, _ = _process_video(npz_path, sim, 0)
    raw_side_all = [e for e in raw if e.fire_side == fire_side]
    defrag_side_all = [e for e in defrag if e.fire_side == fire_side]

    if not defrag_side_all:
        print("  (該当 side の defrag イベントなし)")
        return
    target = min(defrag_side_all, key=lambda e: abs(e.t_chain_start - t_chain_start_approx))
    print(f"[近傍で特定した対象イベント] t_chain_start={target.t_chain_start:.2f} t_fire={target.t_fire:.2f} "
          f"game_idx(FireEvent属性)={target.game_idx} chain_count={target.chain_count} "
          f"frag_count={target.frag_count} before_idx(game内ローカル)={target.before_idx} "
          f"fi_idx(game内ローカル)={target.fi_idx}")

    # 同じ game_idx の raw 断片一覧 (defragの元になった生断片)
    raw_same_game = [e for e in raw_side_all if e.game_idx == target.game_idx]
    print(f"[この game_idx の raw イベント一覧 ({len(raw_same_game)}件)]")
    for e in sorted(raw_same_game, key=lambda x: x.t_fire):
        print(f"    raw: t_chain_start={e.t_chain_start:.2f} t_fire={e.t_fire:.2f} "
              f"chain_count={e.chain_count} delta_score={e.delta_score} label={e.label}")

    # --- 生の t_sec/score 列を、正しい game_idx サブセット (g_target) でダンプ ---
    # before_idx/fi_idx は _process_video 内部で game_idx サブセット後のローカル
    # インデックスのため、同じサブセットを再現してから使う (rec全体に直接使うと
    # 別の試合の行を指してしまうバグを避ける)。
    rec_full = by_side[fire_side]
    game_mask = rec_full.game_idx == target.game_idx
    g_target = _subset(rec_full, game_mask)
    lo = max(0, target.before_idx - 2)
    hi = min(len(g_target.t_sec), target.fi_idx + 5)
    print(f"\n[生データ] game_idx={target.game_idx} サブセット内 t_sec/score (idx {lo}..{hi - 1})")
    for i in range(lo, hi):
        marker = ""
        if i == target.before_idx:
            marker = "  <- before_idx (pre-chain静止盤面)"
        if i == target.fi_idx:
            marker = "  <- fi_idx (post_idx, t_fire)"
        print(f"    idx={i:4d} t_sec={g_target.t_sec[i]:9.3f} score={int(g_target.score[i]):7d}{marker}")


def summarize_pipeline_vs_visual() -> None:
    """recognition_diag_chain_anim_duration_multi の events_raw.csv 内で、
    visual_duration_sec と pipeline_duration_sec (どちらも既存資産に同梱済み)
    の chain_count別の乖離を集計する (新規計算不要、既存CSVの読み取りのみ)。
    """
    import pandas as pd
    csv_path = PROJ_ROOT / "data/verify/recognition_diag_chain_anim_duration_multi/events_raw.csv"
    df = pd.read_csv(csv_path)
    ok = df[df["status"] == "ok"].copy()
    ok["gap"] = ok["pipeline_duration_sec"] - ok["visual_duration_sec"]
    print(f"\n{'=' * 70}\n母集団内 pipeline(t_fire-t_chain_start) vs visual(ピクセルdiff実測) 乖離\n{'=' * 70}")
    print(f"n={len(ok)} (status==ok のみ)")
    g = ok.groupby(ok["chain_count"].clip(upper=8))["gap"].agg(["median", "mean", "std", "min", "max", "count"])
    print(g.to_string())
    n_pipeline_longer = int((ok["gap"] > 0).sum())
    print(f"\npipelineの方がvisualより長いケース: {n_pipeline_longer}/{len(ok)} = {n_pipeline_longer / len(ok):.3f}")


def main() -> None:
    for video_stem, fire_side, game_idx, t_cs in TARGET_CASES:
        dump_case(video_stem, fire_side, game_idx, t_cs)
    summarize_pipeline_vs_visual()


if __name__ == "__main__":
    main()
