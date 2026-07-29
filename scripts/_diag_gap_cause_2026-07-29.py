"""#24 打ち合いゲート「相手を見失った」原因分類 (2026-07-29)。

背景 (userタスク明記):
    「相手を何秒見失ったか」という一律の時間閾値だけで欠測を測るのは誤り。
    相手が連鎖中なら盤面が確定しないのは物理的に正常な現象であり、認識の
    欠陥ではない。一方、相手が普通に設置しているだけなのに見失っているなら
    それは認識の取りこぼし。この2つを同じ「欠測」として扱っている現状の
    測定 (gap_in_window 346件) を、得点差から間接推定で分類し直す。

方式:
    scripts/measure_ojama_landing_delay.py の _find_landing と全く同じ
    ロジック (同一関数を import してそのまま呼ぶ) で detection_status を
    再現し、既知の内訳 (gap_in_window 346 / no_future_frames 100 /
    landed 42 / no_landing_in_window 18 / no_baseline 0、閾値2.0秒) と
    一致することを確認した上で、gap_in_window / no_future_frames の
    各イベントについてのみ、ギャップ区間の前後にある相手側フレームの
    score を読み、得点増分から A(連鎖由来と推定)/B(非連鎖)/C(判別不能) に
    分類する追加処理を行う。

⚠️ 得点差による連鎖推定は「あくまで推定」であり、相手が実際に連鎖したか
   どうかを直接観測したものではない (盤面上に連鎖中フラグ等の列は npz に
   存在しないため)。

⚠️ 認識の再実行は一切行わない。既存 npz
   (data/indicators_v2/boards_lean_fixed_regen_2026-07-28/、m30収集とは
   無関係の完了済みデータ) の読み取り集計のみで完結する。
   scripts/measure_exchange_dynamics.py・scripts/measure_ojama_landing_delay.py
   は import のみで一切変更しない。

使い方:
    nice -n 19 venv/bin/python -m scripts._diag_gap_cause_2026-07-29
"""
from __future__ import annotations

import os
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

for _env_key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_env_key, "1")

PROJ_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ_ROOT))

from src.chain import ChainSimulator  # noqa: E402
from scripts.measure_exchange_dynamics import (  # noqa: E402
    ANOMALY_DELTA_SCORE_MIN, SCORE_DELTA_FIRE, TIER_MAP, FireEvent,
    _process_video, _subset,
)
from scripts.measure_ojama_landing_delay import (  # noqa: E402
    MAX_LANDING_SEARCH_SEC, OPP_GAP_THRESHOLD_SEC, _find_landing, _load_npz,
    _opponent_frame_mask, _visible_ojama_counts,
)

# 既存資産 (m30収集とは別プロセス、完了済み・変更なし)
NPZ_DIR_REGEN: Path = PROJ_ROOT / "data" / "indicators_v2" / "boards_lean_fixed_regen_2026-07-28"
OUTPUT_CSV: Path = PROJ_ROOT / "data" / "indicators_v2" / "gap_cause_diag_2026-07-29.csv"

# ゲート判定 (#24 project_exchange_meter_design_b 準拠)
COVERAGE_GATE_PCT: float = 20.0
TOTAL_EVENTS_KNOWN: int = 506  # 既知の全イベント数 (既存測定との整合確認用)

# ============================
# A/B/C 分類の閾値 (新規発明せず、既存プロジェクト定数を流用)
# ============================
# 連鎖由来とみなす得点増分の下限。SCORE_DELTA_FIRE (measure_exchange_dynamics.py:76)
# は「1連鎖の理論最小得点相当 (4個消し=40点)」としてプロジェクトで既に採用済みの
# 値であり、本スクリプトはこれをそのまま流用する (新規閾値を発明しない)。
CHAIN_DELTA_MIN: int = SCORE_DELTA_FIRE

# 異常値とみなす得点増分の上限。ANOMALY_DELTA_SCORE_MIN (同ファイル:155) も既存
# 定数の流用 (過去に最大73995個相当の異常値が88件確認された既知問題域)。
ANOMALY_DELTA_MIN: int = ANOMALY_DELTA_SCORE_MIN


@dataclass
class GapCauseRow:
    """1 gap_in_window / no_future_frames イベント分の分類結果。"""
    video_stem: str
    tier: str
    game_idx: int
    fire_side: str
    t_fire: float
    detection_status: str
    sub_status: str
    gap_len_sec: float
    score_before: float
    score_after: float
    delta_score: float
    cause: str  # "A"/"B"/"C"
    cause_reason: str


# ============================
# ギャップ境界特定 (_find_landing と同一走査順序、判定ロジックは不変更)
# ============================


def _locate_gap_boundary(
    t_sec: np.ndarray, t_fire: float, gap_threshold_sec: float, search_max_sec: float,
) -> tuple[int, int] | None:
    """gap_in_window 判定の原因になった実際の (idx_a, idx_b) を返す。

    _find_landing (measure_ojama_landing_delay.py:103-139) と全く同じ走査
    順序で「連続観測間隔が gap_threshold_sec を初めて超えた地点」を特定し、
    その直前/直後のインデックスを返す (該当なしなら None)。判定ロジック
    自体は既存関数を呼ぶだけで変更せず、原因特定用の付随計算のみ行う
    (scripts/_gap_threshold_sweep_2026-07-29.py の _first_violating_gap と
    同じ設計方針、既存関数は不変更)。
    """
    idx_before = int(np.searchsorted(t_sec, t_fire, side="right")) - 1
    if idx_before < 0:
        return None
    window_end = t_fire + search_max_sec
    n = len(t_sec)
    prev_idx = idx_before
    i = idx_before + 1
    while i < n and float(t_sec[i]) <= window_end:
        if (float(t_sec[i]) - float(t_sec[prev_idx])) > gap_threshold_sec:
            return prev_idx, i
        prev_idx = i
        i += 1
    return None


def _locate_no_future_boundary(t_sec: np.ndarray, t_fire: float) -> tuple[int, int] | None:
    """no_future_frames イベントの「窓を超えて存在する次フレーム」を探す。

    detection_status=no_future_frames は「t_fire+45秒以内に相手フレームが
    1件も無い」ことを意味する。マスク後の系列全体 (探索窓の制限なし) で
    idx_before の次に存在するフレームを探し、それが実在すれば「45秒を
    超える長時間ブラックアウト」、存在しなければ「そこで観測系列自体が
    終わっている (試合終了等)」と区別する。
    """
    idx_before = int(np.searchsorted(t_sec, t_fire, side="right")) - 1
    if idx_before < 0 or idx_before + 1 >= len(t_sec):
        return None
    return idx_before, idx_before + 1


# ============================
# 得点差 -> A/B/C 分類 (推定、捏造禁止のため理由も残す)
# ============================


def _classify_score_delta(score_before: float, score_after: float) -> tuple[str, str]:
    """得点差からギャップ原因を A(連鎖由来と推定)/B(非連鎖)/C(判別不能) に推定分類する。

    ⚠️ これは推定である。相手が実際に連鎖したかどうかを直接観測したもの
    ではなく、npz に連鎖中フラグ等の列が存在しないため得点変化からの
    間接推定に留まる。

    閾値の根拠 (userタスク指定通り、新規閾値は発明せず既存定数を流用):
      - C (判別不能): score<0 は score OCR 欠損マーカー (measure_exchange_
        dynamics._find_chain_windows の valid_idx フィルタと同じ定義)。
        delta<0 はスコアが試合中に減ることは仕様上あり得ないため OCR誤読
        とみなす。delta>=ANOMALY_DELTA_MIN(30000, 既存定数) は既知の異常値
        問題域 (過去に88件、最大73995個相当を実測済み) として区別する。
      - A (連鎖由来と推定): delta>=CHAIN_DELTA_MIN(=SCORE_DELTA_FIRE=40、
        「4個消し(1連鎖の理論最小得点)」としてプロジェクトが既に採用済みの
        値をそのまま流用)。
      - B (非連鎖): 0<=delta<40 (落下ボーナス等の微増、または無変化)。
    """
    if score_before < 0 or score_after < 0:
        return "C", "score_missing"
    delta = score_after - score_before
    if delta < 0:
        return "C", "negative_delta_anomaly"
    if delta >= ANOMALY_DELTA_MIN:
        return "C", "delta_over_anomaly_threshold"
    if delta >= CHAIN_DELTA_MIN:
        return "A", "chain_likely"
    return "B", "non_chain"


# ============================
# 1 動画分の処理
# ============================


def _process_one_video(npz_path: Path, events: list[FireEvent]) -> list[GapCauseRow]:
    """1動画分の発火イベントについて detection_status を再現し、
    gap_in_window / no_future_frames のみ得点差分類を追加する。
    """
    records = _load_npz(npz_path)
    by_side = {r.side: r for r in records}
    if "1P" not in by_side or "2P" not in by_side:
        return []
    # ⚠️ 重要: _find_landing は「gapチェック」と「着弾(counts>baseline)チェック」を
    # 同じ走査ループ内で行っており、着弾が先に見つかれば以降のgapは評価されない
    # (早期return)。そのため実測 (346/100/42/18/0) と同じ detection_status を
    # 再現するには、本物の可視おじゃま数 (_visible_ojama_counts) を使う必要が
    # ある (ダミー値を渡すと着弾判定が常に不成立になり、本来 landed だった
    # イベントの一部が誤って gap_in_window に化ける事故が実測で確認された→
    # 修正済み)。
    counts_by_side = {side: _visible_ojama_counts(rec.grids) for side, rec in by_side.items()}

    rows: list[GapCauseRow] = []
    for ev in events:
        opp_side = "2P" if ev.fire_side == "1P" else "1P"
        opp_rec = by_side[opp_side]
        own_rec = by_side[ev.fire_side]
        mask = _opponent_frame_mask(opp_rec, own_rec, ev.game_idx, use_game_idx_mask=False)
        if not mask.any():
            rows.append(GapCauseRow(
                ev.video_stem, ev.tier, ev.game_idx, ev.fire_side, ev.t_fire,
                "no_opp_game_data", "no_opp_game_data", float("nan"),
                float("nan"), float("nan"), float("nan"), "N/A", "no_mask",
            ))
            continue
        opp_game = _subset(opp_rec, mask)
        opp_counts = counts_by_side[opp_side][mask]

        result = _find_landing(
            opp_game.t_sec, opp_counts, ev.t_fire, OPP_GAP_THRESHOLD_SEC, MAX_LANDING_SEARCH_SEC,
        )
        status = result.status
        if status not in ("gap_in_window", "no_future_frames"):
            continue

        if status == "gap_in_window":
            boundary = _locate_gap_boundary(
                opp_game.t_sec, ev.t_fire, OPP_GAP_THRESHOLD_SEC, MAX_LANDING_SEARCH_SEC,
            )
            sub_status = "gap_in_window"
        else:
            boundary = _locate_no_future_boundary(opp_game.t_sec, ev.t_fire)
            sub_status = "long_blackout_gt45s" if boundary is not None else "match_end_no_more_opp_frames"

        if boundary is None:
            rows.append(GapCauseRow(
                ev.video_stem, ev.tier, ev.game_idx, ev.fire_side, ev.t_fire,
                status, sub_status, float("nan"),
                float("nan"), float("nan"), float("nan"), "C", "no_boundary_found",
            ))
            continue

        idx_a, idx_b = boundary
        gap_len = float(opp_game.t_sec[idx_b]) - float(opp_game.t_sec[idx_a])
        score_before = float(opp_game.score[idx_a])
        score_after = float(opp_game.score[idx_b])
        delta = score_after - score_before if (score_before >= 0 and score_after >= 0) else float("nan")
        cause, reason = _classify_score_delta(score_before, score_after)
        rows.append(GapCauseRow(
            ev.video_stem, ev.tier, ev.game_idx, ev.fire_side, ev.t_fire,
            status, sub_status, gap_len, score_before, score_after, delta, cause, reason,
        ))
    return rows


# ============================
# レポート集計
# ============================


def _print_cause_table(df: pd.DataFrame, status: str) -> None:
    """指定 detection_status の A/B/C 内訳を件数+比率で出力する。"""
    sub = df[df["detection_status"] == status]
    if sub.empty:
        print(f"\n[{status}] 該当0件")
        return
    n = len(sub)
    print(f"\n[{status}] n={n}")
    counts = sub["cause"].value_counts()
    for cause in ["A", "B", "C", "N/A"]:
        c = int(counts.get(cause, 0))
        if c == 0 and cause not in counts.index:
            continue
        print(f"  {cause}: {c:>4}件 ({c / n * 100:5.1f}%)")
    print("  [cause_reason 内訳]")
    print(sub["cause_reason"].value_counts().to_string())


def _print_gap_len_comparison(df: pd.DataFrame) -> None:
    """A と B でギャップ長分布 (中央値・90%ile) を比較する。"""
    gw = df[df["detection_status"] == "gap_in_window"]
    print("\n[gap_in_window: A(連鎖由来推定) vs B(非連鎖) のギャップ長分布]")
    for cause in ["A", "B"]:
        g = gw[gw["cause"] == cause]["gap_len_sec"].dropna()
        if g.empty:
            print(f"  {cause}: 該当0件")
            continue
        print(
            f"  {cause}: n={len(g)} 中央値={g.median():.2f}秒 "
            f"90%ile={g.quantile(0.90):.2f}秒 最大={g.max():.2f}秒",
        )


def _print_coverage_recalc(df_status_counts: dict[str, int]) -> None:
    """A件数を欠測でなく「連鎖中で応手不能」として扱った場合の実質被覆率。"""
    landed = df_status_counts.get("landed", 0)
    no_landing = df_status_counts.get("no_landing_in_window", 0)
    a_count = df_status_counts.get("gap_in_window_A", 0)
    total = TOTAL_EVENTS_KNOWN
    effective = landed + no_landing + a_count
    pct = effective / total * 100.0
    print(
        f"\n[実質被覆率] (landed={landed} + no_landing_in_window={no_landing} "
        f"+ gap_in_window内A={a_count}) / {total} = {pct:.1f}% "
        f"({'20%基準超え' if pct > COVERAGE_GATE_PCT else '20%基準未達'})",
    )


def main() -> None:
    warnings.filterwarnings("ignore")
    npz_paths = [NPZ_DIR_REGEN / f"{stem}.npz" for stem in TIER_MAP]
    present = [p for p in npz_paths if p.exists()]
    print(f"[INFO] 対象 {len(present)}/23 動画 (gap_in_window/no_future_frames の原因分類)")

    sim = ChainSimulator()
    all_rows: list[GapCauseRow] = []
    seq_id = 0
    n_events_total = 0
    for npz_path in sorted(present, key=lambda p: p.stem):
        _, defrag_events, seq_id = _process_video(npz_path, sim, seq_id)
        n_events_total += len(defrag_events)
        rows = _process_one_video(npz_path, defrag_events)
        all_rows.extend(rows)
        print(f"  {npz_path.stem}: 発火{len(defrag_events)}件 -> 対象(gap系)抽出{len(rows)}件")

    print(f"\n[整合確認] 全イベント数 n_events_total={n_events_total} (既知値 {TOTAL_EVENTS_KNOWN} と一致すべき)")
    if n_events_total != TOTAL_EVENTS_KNOWN:
        print(
            f"  ⚠️ 不一致 (差={n_events_total - TOTAL_EVENTS_KNOWN})。npz構成や"
            f"de-fragロジックのバージョン差の可能性。以降の集計は実測値ベースで継続。",
        )

    df = pd.DataFrame([vars(r) for r in all_rows])
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"[DONE] {len(df)} 行を {OUTPUT_CSV} に保存しました")

    _print_cause_table(df, "gap_in_window")
    _print_gap_len_comparison(df)
    _print_cause_table(df, "no_future_frames")

    print("\n[no_future_frames 内訳 (match_end vs long_blackout)]")
    nf = df[df["detection_status"] == "no_future_frames"]
    print(nf["sub_status"].value_counts().to_string())

    gw = df[df["detection_status"] == "gap_in_window"]
    a_in_gw = int((gw["cause"] == "A").sum())
    status_counts = {
        "landed": 42, "no_landing_in_window": 18, "gap_in_window_A": a_in_gw,
    }
    _print_coverage_recalc(status_counts)


if __name__ == "__main__":
    main()
