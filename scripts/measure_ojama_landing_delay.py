"""打ち合い計測器 ステップ0: お邪魔着弾遅延の実測。

目的:
    連鎖を発火してから、相手盤面に実際にお邪魔が降り始める(着弾する)までの
    遅延秒数を実データから実測し、連鎖数との関係を出す。
    src/indicators_v2.py の chain_to_time (TIME_PER_CHAIN_SEC=0.30、
    FRAMES_PER_CHAIN=84) は未検証の仮値であり、本スクリプトはその検証も兼ねる。

    memory `reference_saisoku_exchange_model_2026-07-22` (催促の有効性判定
    条件1/2) の時間窓の根拠データを作る、打ち合い計測器の最優先ステップ。

方式:
    1. 発火イベントは scripts/measure_exchange_dynamics.py の既存資産
       (de-frag 済み発火検出、_process_video) をそのまま再利用する
       (本ファイルからは import のみ、書き換えはしない)。
    2. 各発火イベント (攻撃側) の t_fire 以降、相手側 (opp) 盤面の
       可視お邪魔ぷよ数 ((grid==COLOR_OJAMA).sum() 相当) が発火直前の
       基準値を超えた最初のフレーム時刻を着弾 t_landed とする。
    3. 相手側フレームが無い/連続観測が途切れる (gap > OPP_GAP_THRESHOLD_SEC)
       区間は検出不能として delay_sec=NaN + detection_status で残す
       (推定で埋めない)。

⚠️ 本スクリプトはデータ分析専用であり、src/ 配下は一切変更しない。
   scripts/measure_exchange_dynamics.py・src/indicators_v2.py・src/chain.py も
   import のみで書き換えない (別コーダが飽和連鎖量を並行作業中のため)。

使い方:
    PYTHONPATH=. python -m scripts.measure_ojama_landing_delay
"""
from __future__ import annotations

import os
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# スレッド制限 (熱暴走防止、feedback_thermal_safety_mandatory 準拠)
for _env_key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_env_key, "2")

PROJ_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ_ROOT))

from src.board import COLOR_OJAMA, HIDDEN_ROWS  # noqa: E402
from src.chain import ChainSimulator  # noqa: E402
from src.indicators_v2 import TIME_PER_CHAIN_SEC, chain_to_time  # noqa: E402
from scripts.measure_exchange_dynamics import (  # noqa: E402
    CHAIN_BIN_CAP, NPZ_DIR, TIER_MAP, FireEvent, NpzRecord, _load_npz,
    _process_video, _subset,
)

# ============================
# 定数定義
# ============================
OUTPUT_CSV: Path = PROJ_ROOT / "data" / "indicators_v2" / "exchange_landing_delay.csv"

# 相手側フレームの連続観測が途切れたとみなすギャップ秒数 (仮閾値、userタスク指定)。
# これを超えるギャップに遭遇したら、着弾を見逃した可能性を排除できないため
# 検出不能 (opp_available=False) として扱う (推定で埋めない)。
OPP_GAP_THRESHOLD_SEC: float = 2.0

# 着弾検出の探索窓秒数 (数十秒、userタスク指定の考え方に基づく)。
# 根拠: 連鎖実行時間 (大連鎖で ~10秒級) + マージンタイム送出サイクル
# (reference_puyo_rules_confirmed_2026-07-22: 16秒毎×0.75 送出) + 着弾アニメ
# 数秒を見込み、余裕を持って 45 秒とする。
MAX_LANDING_SEARCH_SEC: float = 45.0

# 参考値扱いの n 閾値 (これ未満のビンは "参考値" と明示する、userタスク指定)。
MIN_N_FOR_STABLE_BIN: int = 10

# 実測対象から除外する「送りお邪魔0個」判定の下限 (個)。0個の発火は相殺等で
# 着弾しない可能性が高く、delay_sec の分布に混ぜると「検出失敗」と「そもそも
# 送っていない」が区別できなくなるため、集計時のみ分けて扱う。
MIN_OJAMA_SENT_FOR_DELAY_STATS: float = 0.0


@dataclass
class LandingResult:
    """1 発火イベントの着弾検出結果。"""
    status: str
    available: bool
    t_landed: float
    landed_idx: int


# ============================
# 可視お邪魔カウント (src/indicators_v2.py の _count_visible_ojama 相当を
# 生 grid 配列向けにベクトル化したもの。ロジックは同一: 隠し段を除いた
# 可視領域で COLOR_OJAMA を数える)
# ============================


def _visible_ojama_counts(grids: np.ndarray) -> np.ndarray:
    """(n_frames, ROWS, COLS) の grids から、フレームごとの可視お邪魔数を返す。"""
    visible = grids[:, HIDDEN_ROWS:, :]
    return (visible == COLOR_OJAMA).sum(axis=(1, 2)).astype(np.int32)


def _find_landing(
    t_sec: np.ndarray,
    counts: np.ndarray,
    t_fire: float,
    gap_threshold_sec: float,
    search_max_sec: float,
) -> LandingResult:
    """t_fire 以降、counts が基準値を超える最初のフレームを探す。

    基準値は t_fire 直前 (以下) の最後のフレームの可視お邪魔数。
    連続観測が gap_threshold_sec を超えて途切れたら検出不能として打ち切る
    (見逃しの可能性を排除できないため、推定で埋めない)。
    """
    idx_before = int(np.searchsorted(t_sec, t_fire, side="right")) - 1
    if idx_before < 0:
        return LandingResult("no_baseline", False, float("nan"), -1)

    baseline = int(counts[idx_before])
    window_end = t_fire + search_max_sec
    n = len(t_sec)
    prev_t = float(t_sec[idx_before])
    found_future_frame = False

    i = idx_before + 1
    while i < n and float(t_sec[i]) <= window_end:
        found_future_frame = True
        cur_t = float(t_sec[i])
        if (cur_t - prev_t) > gap_threshold_sec:
            return LandingResult("gap_in_window", False, float("nan"), -1)
        if int(counts[i]) > baseline:
            return LandingResult("landed", True, cur_t, i)
        prev_t = cur_t
        i += 1

    if not found_future_frame:
        return LandingResult("no_future_frames", False, float("nan"), -1)
    return LandingResult("no_landing_in_window", True, float("nan"), -1)


# ============================
# 1 動画分の着弾遅延測定
# ============================


def _opponent_frame_mask(
    opp_rec: NpzRecord,
    own_rec: NpzRecord,
    game_idx: int,
    use_game_idx_mask: bool = False,
) -> np.ndarray:
    """相手側フレームを「同じ実試合」に絞り込むマスクを返す。

    バグ修正 (2026-07-29): 旧実装 (use_game_idx_mask=True) は 1P/2P 独立
    カウンタである game_idx ラベルの一致で絞っていたが、片側がスコア
    リセット検知を1回でも見逃すと動画残り全体で固定オフセット分ズレる
    (実測: 両側データありのペア361件中208件=57.6%が5秒超ズレ、c11は
    一貫して+2)。このズレにより実在する相手観測を誤って「別試合」として
    捨てていた (no_baseline 118件中109件=92.4%がこのラベル不一致由来)。

    修正後 (既定、use_game_idx_mask=False): game_idx ラベルを信用せず、
    実時刻を真実として使う。攻撃側 (own_rec) 自身の game_idx 境界時刻
    [開始, 終了] で相手フレームを絞る。攻撃側自身の game_idx は自身の
    スコアリセットで区切られているため信用できる一方、相手側の game_idx
    ラベルはズレうるため使わない。この時間区間内の相手フレームは定義上
    「同じ実試合の同じ時間帯」になる。

    Args:
        use_game_idx_mask: True で旧挙動 (相手側 game_idx ラベル一致) を
            再現する (backwards compat、A/B比較用)。既定 False = 新挙動。
    """
    if use_game_idx_mask:
        return opp_rec.game_idx == game_idx
    own_mask = own_rec.game_idx == game_idx
    if not own_mask.any():
        return np.zeros(len(opp_rec.t_sec), dtype=bool)
    t_start = float(own_rec.t_sec[own_mask].min())
    t_end = float(own_rec.t_sec[own_mask].max())
    return (opp_rec.t_sec >= t_start) & (opp_rec.t_sec <= t_end)


def _measure_landing_for_video(
    npz_path: Path, events: list[FireEvent], use_game_idx_mask: bool = False,
) -> list[dict]:
    """1 動画分の発火イベント一覧に対し、相手盤面の着弾遅延を測定する。

    Args:
        use_game_idx_mask: True で旧挙動 (相手側 game_idx ラベル一致) を
            再現する (backwards compat、A/B比較用)。既定 False =
            時刻ベースマスク (_opponent_frame_mask 参照、2026-07-29 修正)。
    """
    records = _load_npz(npz_path)
    by_side = {r.side: r for r in records}
    if "1P" not in by_side or "2P" not in by_side:
        return []
    counts_by_side = {side: _visible_ojama_counts(rec.grids) for side, rec in by_side.items()}

    rows: list[dict] = []
    for ev in events:
        opp_side = "2P" if ev.fire_side == "1P" else "1P"
        opp_rec = by_side[opp_side]
        own_rec = by_side[ev.fire_side]
        mask = _opponent_frame_mask(opp_rec, own_rec, ev.game_idx, use_game_idx_mask)
        if not mask.any():
            rows.append(_build_row(ev, LandingResult("no_opp_game_data", False, float("nan"), -1)))
            continue
        opp_game = _subset(opp_rec, mask)
        opp_counts = counts_by_side[opp_side][mask]
        result = _find_landing(
            opp_game.t_sec, opp_counts, ev.t_fire,
            OPP_GAP_THRESHOLD_SEC, MAX_LANDING_SEARCH_SEC,
        )
        rows.append(_build_row(ev, result))
    return rows


def _build_row(ev: FireEvent, result: LandingResult) -> dict:
    """FireEvent + LandingResult から出力 1 行分の dict を組み立てる。

    ⚠️ 重要な区別 (実測で判明): FireEvent.t_fire は「連鎖アニメーションが
    完全に終わった後の確定盤面時刻」(post-chain) であり、連鎖の実行時間
    そのものは含まない。一方 FireEvent.t_chain_start は「連鎖発火直前の
    静止盤面時刻」(pre-chain、真のトリガー時刻)。
    ユーザー仕様の delay_sec (= t_landed - t_fire) は「連鎖終了後 〜 着弾」
    のみを測る値であり、chain_to_time (連鎖実行時間モデル) と直接比較する
    には delay_from_chain_start_sec (= t_landed - t_chain_start、連鎖実行
    時間+マージン待ちの合算) の方が適切なため、追加列として両方残す。
    """
    has_landed = result.available and not np.isnan(result.t_landed)
    delay_sec = result.t_landed - ev.t_fire if has_landed else float("nan")
    delay_from_start = result.t_landed - ev.t_chain_start if has_landed else float("nan")
    return {
        "video_stem": ev.video_stem,
        "tier": ev.tier,
        "game_idx": ev.game_idx,
        "fire_side": ev.fire_side,
        "t_chain_start": ev.t_chain_start,
        "t_fire": ev.t_fire,
        "chain_count": ev.chain_count,
        "ojama_sent_count": ev.ojama_sent_count,
        "t_landed": result.t_landed,
        "delay_sec": delay_sec,
        "delay_from_chain_start_sec": delay_from_start,
        "opp_available": result.available,
        "detection_status": result.status,
        "landed_idx": result.landed_idx,
    }


# ============================
# レポート集計
# ============================


def _delay_subset(df: pd.DataFrame) -> pd.DataFrame:
    """delay_sec 分布集計の対象を絞り込む (送りお邪魔0個/欠損は除く)。

    送りお邪魔が0個の発火は相殺等で物理的に着弾しない可能性が高く、
    「検出失敗」と区別せず delay_sec 分布に混ぜると解釈を誤るため、
    ojama_sent_count > 0 の行のみを対象にする (NaN=スコア欠損は許容、
    着弾検出自体はスコアと独立に行えているため)。
    """
    sent = df["ojama_sent_count"]
    keep = sent.isna() | (sent > MIN_OJAMA_SENT_FOR_DELAY_STATS)
    return df[keep]


def _report_delay_by_chain_bin(sub: pd.DataFrame, delay_col: str) -> pd.DataFrame:
    """連鎖数ビン別の delay 統計 (median/mean/std/n) + 仮値予測との比較。

    Args:
        delay_col: "delay_sec" (t_fire=連鎖終了後基準) または
            "delay_from_chain_start_sec" (t_chain_start=真のトリガー基準)。
    """
    valid = sub.dropna(subset=[delay_col]).copy()
    if valid.empty:
        return pd.DataFrame()
    valid["chain_bin"] = valid["chain_count"].clip(upper=CHAIN_BIN_CAP)
    g = valid.groupby("chain_bin")[delay_col].agg(["median", "mean", "std", "count"])
    g["predicted_sec_TIME_PER_CHAIN_0.30"] = [
        chain_to_time(float(b)) for b in g.index
    ]
    g["gap_median_minus_predicted"] = g["median"] - g["predicted_sec_TIME_PER_CHAIN_0.30"]
    g["note"] = g["count"].apply(
        lambda n: "参考値(n<10)" if n < MIN_N_FOR_STABLE_BIN else "",
    )
    return g


def _report_availability(df: pd.DataFrame) -> pd.Series:
    """detection_status の内訳比率 (opp_available の実態を隠さず出す)。"""
    return df["detection_status"].value_counts(normalize=True)


def _report_examples(df: pd.DataFrame, n: int = 8) -> pd.DataFrame:
    """人の目レビュー用の代表例 (landed かつ chain_count が散らばるよう抽出)。"""
    landed = df[df["detection_status"] == "landed"].copy()
    if landed.empty:
        return landed
    landed = landed.sort_values("chain_count")
    idx = np.linspace(0, len(landed) - 1, num=min(n, len(landed))).astype(int)
    cols = [
        "video_stem", "tier", "fire_side", "chain_count", "ojama_sent_count",
        "t_chain_start", "t_fire", "t_landed", "delay_sec", "delay_from_chain_start_sec",
    ]
    return landed.iloc[idx][cols]


def _print_tier_block(tier: str, sub: pd.DataFrame) -> None:
    """1 ティア分 (または全体) の検出状況 + 2 基準の delay 統計を出力する。"""
    print(f"\n=== {tier} (n_fire={len(sub)}) ===")
    print("[検出状況の内訳 (detection_status)]")
    print(_report_availability(sub))
    n_opp_avail = int(sub["opp_available"].sum())
    print(f"[opp_available 率] {n_opp_avail}/{len(sub)} = {n_opp_avail / len(sub):.3f}")

    delay_sub = _delay_subset(sub)
    print(f"[delay統計対象 (送り0個/相殺除外後)] n={len(delay_sub)}")

    print("[(A) t_chain_start基準 (真のトリガー~着弾、chain_to_time比較の本命)]")
    report_a = _report_delay_by_chain_bin(delay_sub, "delay_from_chain_start_sec")
    print(report_a if not report_a.empty else "  (実測0件)")

    print("[(B) t_fire基準 (連鎖アニメ終了後~着弾、旧定義・参考)]")
    report_b = _report_delay_by_chain_bin(delay_sub, "delay_sec")
    print(report_b if not report_b.empty else "  (実測0件)")


def _print_report(df: pd.DataFrame) -> None:
    """ティア別 + 全体で delay 分布・検出状況・代表例を出力する。"""
    print(f"\n[全体イベント数] n={len(df)}")
    print(f"[TIME_PER_CHAIN_SEC 仮値] {TIME_PER_CHAIN_SEC} 秒/連鎖 (検証対象)")

    for tier in ("チャレンジャー", "マスター", "S級", "全体"):
        sub = df if tier == "全体" else df[df["tier"] == tier]
        if sub.empty:
            continue
        _print_tier_block(tier, sub)

    print("\n[代表例 (landed、連鎖数順に抜粋)]")
    print(_report_examples(df).to_string(index=False))


# ============================
# メイン
# ============================


def main() -> None:
    """メイン処理: 23動画を処理し着弾遅延 CSV 保存 + 集計レポート出力する。"""
    warnings.filterwarnings("ignore")
    npz_paths = [NPZ_DIR / f"{stem}.npz" for stem in TIER_MAP]
    missing = [p for p in npz_paths if not p.exists()]
    if missing:
        print(f"[ERROR] npz不足: {missing}", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] 対象 {len(npz_paths)} 動画 (お邪魔着弾遅延を実測)")
    sim = ChainSimulator()
    all_rows: list[dict] = []
    seq_id = 0
    for npz_path in sorted(npz_paths, key=lambda p: p.stem):
        _, defrag_events, seq_id = _process_video(npz_path, sim, seq_id)
        rows = _measure_landing_for_video(npz_path, defrag_events)
        all_rows.extend(rows)
        print(f"  {npz_path.stem} ({TIER_MAP[npz_path.stem]}): 発火{len(defrag_events)}件 -> 着弾測定{len(rows)}件")

    if not all_rows:
        print("[ERROR] 発火イベントが0件でした。", file=sys.stderr)
        sys.exit(1)

    df = pd.DataFrame(all_rows)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n[DONE] {len(df)} 行を {OUTPUT_CSV} に保存しました")

    _print_report(df)


if __name__ == "__main__":
    main()
