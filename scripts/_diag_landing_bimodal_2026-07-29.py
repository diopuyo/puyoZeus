"""着弾遅延の定量化 + 「連鎖完了→相手ツモ着地」仮説の再検証 (2026-07-29)。

読み取り専用の診断スクリプト。src/ 配下・measure_exchange_effectiveness.py は一切変更しない。

user確定の挙動:
    おじゃまが降るには (1) 攻撃側の連鎖が完了する (2) その後、受け側のツモが
    着地する (受け側の手が確定する) の2条件が順に必要。「最後の一手を迷えば
    その分だけ着弾が遅れる」。

⚠️ 経緯 (訂正記録として残す): 本ファイルの最初のバージョンは「核心検証」を
    t_chain_start (連鎖発火直前=連鎖アニメ開始前) 基準で行っていたが、これは
    誤り。連鎖アニメには数秒〜十数秒かかるため、その基準では「相手が複数手
    打ち終えていた」のは当然の結果に過ぎず、仮説の検証になっていなかった。
    正しい基準は t_chain_end (連鎖アニメ終了=連鎖完了時刻)。本バージョンは
    これを修正し、t_fire (post-chain 確定盤面時刻、= t_chain_end) を基準に
    再検証する。

データ:
- data/indicators_v2/exchange_landing_delay_regen_2026-07-28.csv の
  detection_status=="landed" 行 (25件)。
  列定義 (scripts/measure_ojama_landing_delay.py:_build_row のdocstring):
    t_chain_start = 連鎖発火直前の静止盤面時刻 (連鎖アニメ開始前、pre-chain)
    t_fire        = 連鎖アニメーションが完全に終わった後の確定盤面時刻
                    (post-chain) = 本タスクでいう t_chain_end そのもの
    delay_sec     = t_landed - t_fire  (= 連鎖完了→着弾、まさに知りたい値)
    delay_from_chain_start_sec = t_landed - t_chain_start (旧検証で使用、
                    連鎖実行時間も含んでしまうため今回の核心検証には使わない)
- data/verify/labeled_win_{c20_2026-07-26,m20_2026-07-28,m30_2026-07-28}/study/*.csv
  (video_stem 単位、 base/_gap/_mid の3ファイル、 t_sec 絶対時刻・game_idx不使用)
  side列+tsumo列で相手側ツモカウンタの増分イベントを検出する
  (tsumo_1p/tsumo_2p列は存在しない)。
- data/verify/recognition_diag_chain_anim_duration_multi/summary.json
  連鎖数別の連鎖アニメ時間 visual_median (n=418、t_fire実測が使えない場合の
  代用チェック用)。
"""
from __future__ import annotations

import csv
import json
import os
from typing import Optional

EXCHANGE_CSV = "data/indicators_v2/exchange_landing_delay_regen_2026-07-28.csv"
STUDY_DIRS = [
    "data/verify/labeled_win_c20_2026-07-26/study",
    "data/verify/labeled_win_m20_2026-07-28/study",
    "data/verify/labeled_win_m30_2026-07-28/study",
]
SEGMENT_SUFFIXES = ["", "_gap", "_mid"]
CHAIN_ANIM_DURATION_JSON = "data/verify/recognition_diag_chain_anim_duration_multi/summary.json"
# 既存 SEC_PER_HAND (src/indicators_v2.py、読み取りのみ・変更しない) の実測中央値。
# 「連鎖アニメ中に相手が実際に何手打っているか」の期待値との比較用。
SEC_PER_HAND_EXISTING: float = 0.733

# study CSV は 0-300 / 300-900 / 1200-1560 秒の3ジョブ分割で、900-1200秒帯は
# そもそも収集されていない。この帯に近い区間で「直近サンプルが遠い」場合は
# 収集断絶による偽の gap なので、この閾値を超えたら信頼不可としてマークする。
COVERAGE_GAP_THRESHOLD_SEC: float = 15.0

# 「0付近」判定における測定系由来の遅れの許容閾値。根拠:
# data/verify/c34_reflection_lag_2026-07-25/summary.txt の実測 (クリーン着地
# のみ) では反映遅延は中央値0.0秒・p90 0.0〜0.09秒・最大0.13秒 (60fps側)。
# study CSV はさらに粗い間引き収集 (STABLE確定時のみ行が出る) なので、
# 安全側に見て 1.0 秒までは「測定系の遅れ」として許容し、それを超える残差の
# みを「モデルと食い違う」とみなす。
MEASUREMENT_LAG_TOLERANCE_SEC: float = 1.0


def load_landed_rows(path: str) -> list[dict]:
    """detection_status=="landed" の行のみ読み込む。"""
    rows = []
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["detection_status"] == "landed":
                rows.append(row)
    return rows


def load_chain_anim_median_by_count() -> dict[int, float]:
    """連鎖数別の連鎖アニメ時間 visual_median (n=418実測) を読み込む。"""
    with open(CHAIN_ANIM_DURATION_JSON, encoding="utf-8") as f:
        data = json.load(f)
    return {b["chain_bin"]: b["visual_median"] for b in data["chain_count_bins"]}


def find_study_files(video_stem: str) -> list[str]:
    """video_stem に対応する study CSV (base/_gap/_mid) を全 study dir から探す。"""
    found = []
    for d in STUDY_DIRS:
        for suf in SEGMENT_SUFFIXES:
            p = os.path.join(d, f"{video_stem}{suf}.csv")
            if os.path.exists(p):
                found.append(p)
    return found


def load_defender_series(paths: list[str], defender_side: str) -> list[tuple[float, int]]:
    """defender_side ("1P"/"2P") の (t_sec, tsumo) 系列を全セグメントから結合して返す。"""
    series: list[tuple[float, int]] = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row["side"] != defender_side:
                    continue
                try:
                    t = float(row["t_sec"])
                    v = int(row["tsumo"])
                except (ValueError, KeyError):
                    continue
                series.append((t, v))
    series.sort(key=lambda x: x[0])
    return series


def all_increment_times(series: list[tuple[float, int]]) -> list[float]:
    """defender 側の tsumo 増分が起きた全時刻を返す。"""
    times = []
    prev_val: Optional[int] = None
    for t, v in series:
        if prev_val is not None and v != prev_val:
            times.append(t)
        prev_val = v
    return times


def max_gap_in_window(series: list[tuple[float, int]], t_start: float, t_end: float) -> float:
    """[t_start, t_end] 内 (前後の直近サンプルを含む) の最大サンプル間隔を返す。

    900-1200秒帯のような収集断絶帯が区間内に丸ごと/一部でも入っていると
    ここが極端に大きくなるため、「実測0手」がデータ欠損由来かどうかの
    判定に使う。 データが1件も無ければ inf を返す。
    """
    ts = [t for t, _v in series if t_start - 1.0 <= t <= t_end + 1.0]
    if not ts:
        return float("inf")
    ts = sorted(ts)
    gaps = [b - a for a, b in zip(ts, ts[1:])]
    max_internal_gap = max(gaps) if gaps else 0.0
    # 区間の両端がサンプルで覆われているかも見る (端に接していないと
    # 「区間の最初/最後が欠損」を見逃す)
    edge_gap = max(ts[0] - t_start, t_end - ts[-1], 0.0)
    return max(max_internal_gap, edge_gap)


def nearest_sample_gap(series: list[tuple[float, int]], t_ref: float) -> float:
    """t_ref 直前の実サンプルとの時間差 (収集断絶検知用)。"""
    prev_t: Optional[float] = None
    for t, _v in series:
        if t > t_ref:
            break
        prev_t = t
    if prev_t is None:
        return float("inf")
    return t_ref - prev_t


def defender_side_of(fire_side: str) -> str:
    return "1P" if fire_side == "2P" else "2P"


def summarize(vals: list[float], label: str) -> None:
    if not vals:
        print(f"{label}: データなし")
        return
    vs = sorted(vals)
    n = len(vs)

    def pct(p: float) -> float:
        idx = min(n - 1, max(0, int(round(p * (n - 1)))))
        return vs[idx]

    median = pct(0.5)
    print(
        f"{label}: n={n} min={vs[0]:.2f} q25={pct(0.25):.2f} median={median:.2f} "
        f"q75={pct(0.75):.2f} max={vs[-1]:.2f} mean={sum(vs)/n:.2f}"
    )


def main() -> None:
    landed = load_landed_rows(EXCHANGE_CSV)
    anim_median = load_chain_anim_median_by_count()
    print(f"landed行数: {len(landed)}")

    # ------------------------------------------------------------------
    # 項目1+2: t_chain_end (=t_fire) → t_landed の分布 (全25件、study CSV不要)
    # ------------------------------------------------------------------
    print("\n=== 項目1/2: delay_sec (= t_landed - t_fire = 連鎖完了→着弾) の分布 ===")
    print("(t_fire の定義は measure_ojama_landing_delay.py:_build_row のdocstring通り")
    print(" 「連鎖アニメーションが完全に終わった後の確定盤面時刻」= t_chain_end)")

    all_delay_sec = []
    by_chain: dict[int, list[float]] = {}
    negative_count = 0
    cross_check_rows = []
    for row in landed:
        d = float(row["delay_sec"])
        cc = int(float(row["chain_count"]))
        all_delay_sec.append(d)
        by_chain.setdefault(cc, []).append(d)
        if d < 0:
            negative_count += 1
        # 交差チェック: t_chain_start + 連鎖数別中央値アニメ時間 で t_chain_end を代用
        t_chain_start = float(row["t_chain_start"])
        t_landed = float(row["t_landed"])
        model_dur = anim_median.get(min(cc, 8))
        if model_dur is not None:
            model_chain_end = t_chain_start + model_dur
            cross_check_rows.append(
                (row["video_stem"], cc, t_landed - model_chain_end, d)
            )

    summarize(all_delay_sec, "全体 delay_sec (連鎖完了→着弾)")
    print("\n[連鎖数別]")
    for cc in sorted(by_chain):
        summarize(by_chain[cc], f"  chain_count={cc}")

    print(f"\nt_landed < t_fire (連鎖完了より前に着弾) の件数: {negative_count} / {len(landed)}")

    print("\n[交差チェック: 実測t_fireベース delay_sec と 母集団中央値アニメ時間ベースの残差]")
    print("(母集団中央値で t_chain_end を代用した場合の t_landed-model_chain_end と、")
    print(" 実測delay_secの差。 大きくずれれば「この動画のこの発火は中央値と違う")
    print(" 連鎖アニメ時間だった」ことを意味し、モデル代用の誤差の目安になる)")
    diffs = [abs(model_val - real_val) for (_v, _c, model_val, real_val) in cross_check_rows]
    summarize(diffs, "  |model_delay - real_delay_sec|")

    # ------------------------------------------------------------------
    # 項目3+4: t_chain_end (=t_fire) 以降で最初に来る相手ツモ増分と t_landed の関係
    # ------------------------------------------------------------------
    print("\n=== 項目3/4: 核心検証 (t_fire=連鎖完了 基準、修正版) ===")
    results = []
    missing = []
    for row in landed:
        vs = row["video_stem"]
        paths = find_study_files(vs)
        if not paths:
            missing.append(row)
            continue
        defender_side = defender_side_of(row["fire_side"])
        series = load_defender_series(paths, defender_side)
        t_fire = float(row["t_fire"])
        t_landed = float(row["t_landed"])
        inc_times = all_increment_times(series)
        first_inc_after_chain_end = next((t for t in inc_times if t > t_fire), None)
        if first_inc_after_chain_end is None:
            missing.append(row)
            continue
        gap = t_landed - first_inc_after_chain_end
        coverage_gap = nearest_sample_gap(series, t_landed)
        reliable = coverage_gap <= COVERAGE_GAP_THRESHOLD_SEC
        # 「連鎖完了後、最初の増分」と「着地直前の最後の増分」が同一かどうか
        # (途中に別の増分を挟んでいないか = 本当に最初の1手で降ちたか)
        last_inc_before_landed = max((t for t in inc_times if t <= t_landed), default=None)
        matches_first = (
            last_inc_before_landed is not None
            and abs(last_inc_before_landed - first_inc_after_chain_end) < 1e-6
        )
        results.append(
            {
                "video_stem": vs,
                "fire_side": row["fire_side"],
                "chain_count": row["chain_count"],
                "delay_sec": float(row["delay_sec"]),
                "t_fire": t_fire,
                "t_landed": t_landed,
                "first_inc_after_chain_end": first_inc_after_chain_end,
                "gap_landed_minus_first_inc": gap,
                "coverage_gap_sec": coverage_gap,
                "reliable": reliable,
                "matches_first": matches_first,
            }
        )

    print(f"\n突き合わせ成功: {len(results)} / 不能: {len(missing)}")
    if missing:
        print("突き合わせ不能 video_stem 一覧:", sorted({r['video_stem'] for r in missing}))

    header = (
        f"{'video':6} {'fire':4} {'chain':5} {'delay_sec':9} {'t_fire':9} {'t_landed':9} "
        f"{'first_inc_after_end':19} {'gap(landed-inc)':16} {'cover_gap':9} {'reliable':8} {'match1st':8}"
    )
    print(header)
    for r in sorted(results, key=lambda x: x["delay_sec"]):
        print(
            f"{r['video_stem']:6} {r['fire_side']:4} {r['chain_count']:>5} "
            f"{r['delay_sec']:9.2f} {r['t_fire']:9.2f} {r['t_landed']:9.2f} "
            f"{r['first_inc_after_chain_end']:19.2f} {r['gap_landed_minus_first_inc']:16.2f} "
            f"{r['coverage_gap_sec']:9.2f} {str(r['reliable']):8} {str(r['matches_first']):8}"
        )

    n_unreliable = sum(1 for r in results if not r["reliable"])
    if n_unreliable:
        print(
            f"\n注意: {n_unreliable} 件は t_landed 付近が study CSV の収集断絶帯"
            f" (900-1200秒帯など) に該当し信頼できません。以下はこれらを除外して集計します。"
        )
    reliable_results = [r for r in results if r["reliable"]]
    print(f"\n[信頼できる行のみで再集計: n={len(reliable_results)} / {len(results)}]")

    gaps = [r["gap_landed_minus_first_inc"] for r in reliable_results]
    summarize(gaps, "gap = t_landed - (連鎖完了後最初の相手ツモ増分時刻)")

    n_within_tol = sum(1 for g in gaps if abs(g) <= MEASUREMENT_LAG_TOLERANCE_SEC)
    n_match_first = sum(1 for r in reliable_results if r["matches_first"])
    print(
        f"\n許容閾値 {MEASUREMENT_LAG_TOLERANCE_SEC:.1f} 秒以内 (測定系の遅れとみなせる範囲) "
        f"に収まった件数: {n_within_tol} / {len(reliable_results)}"
    )
    print(
        f"「連鎖完了後最初の増分」= 「着地直前の最後の増分」の一致件数"
        f" (途中に別の増分を挟んでいない): {n_match_first} / {len(reliable_results)}"
    )

    print("\n[個別内訳 (gapが許容閾値を超える行を明示)]")
    for r in sorted(reliable_results, key=lambda x: abs(x["gap_landed_minus_first_inc"])):
        g = r["gap_landed_minus_first_inc"]
        flag = "" if abs(g) <= MEASUREMENT_LAG_TOLERANCE_SEC else "  <-- 閾値超過"
        print(f"  {r['video_stem']:6} delay_sec={r['delay_sec']:6.2f} gap={g:7.2f}{flag}")

    # ------------------------------------------------------------------
    # 追加項目: 連鎖アニメ中 [t_chain_start, t_fire] に相手が実際に何手打ったか
    # (「遅延秒 / SEC_PER_HAND」の過大評価疑惑を直接検証)
    # ------------------------------------------------------------------
    print("\n=== 追加項目: 連鎖アニメ中の相手側実測手数 vs 0.733秒/手モデル ===")
    hand_rows = []
    hand_missing = []
    for row in landed:
        vs = row["video_stem"]
        paths = find_study_files(vs)
        if not paths:
            hand_missing.append(row)
            continue
        defender_side = defender_side_of(row["fire_side"])
        series = load_defender_series(paths, defender_side)
        t_chain_start = float(row["t_chain_start"])
        t_fire = float(row["t_fire"])
        anim_dur = t_fire - t_chain_start
        inc_times = all_increment_times(series)
        n_hands_actual = sum(1 for t in inc_times if t_chain_start < t <= t_fire)
        n_hands_model = int(anim_dur // SEC_PER_HAND_EXISTING)
        window_gap = max_gap_in_window(series, t_chain_start, t_fire)
        hand_rows.append(
            {
                "video_stem": vs,
                "chain_count": row["chain_count"],
                "anim_dur_sec": anim_dur,
                "n_hands_actual": n_hands_actual,
                "n_hands_model": n_hands_model,
                "window_gap": window_gap,
                "reliable": window_gap <= COVERAGE_GAP_THRESHOLD_SEC,
            }
        )
    print(f"{'video':6} {'chain':5} {'anim_dur':9} {'実測手数':8} {'モデル手数(÷0.733)':18} {'window_gap':10} {'reliable':8}")
    for r in sorted(hand_rows, key=lambda x: x["anim_dur_sec"]):
        print(
            f"{r['video_stem']:6} {r['chain_count']:>5} {r['anim_dur_sec']:9.2f} "
            f"{r['n_hands_actual']:8} {r['n_hands_model']:18} {r['window_gap']:10.2f} {str(r['reliable']):8}"
        )
    n_unreliable_window = sum(1 for r in hand_rows if not r["reliable"])
    if n_unreliable_window:
        print(
            f"\n注意: {n_unreliable_window} 件は連鎖アニメ区間 [t_chain_start, t_fire] 自体が"
            f" 900-1200秒帯などの収集断絶帯にかかっており、「実測手数」がデータ欠損による"
            f" 見かけ上の値の可能性があります。以下はこれらを除外して集計します。"
        )
    hand_rows_reliable = [r for r in hand_rows if r["reliable"]]
    print(f"\n[信頼できる区間のみで再集計: n={len(hand_rows_reliable)} / {len(hand_rows)}]")
    actual_hands = [r["n_hands_actual"] for r in hand_rows_reliable]
    model_hands = [r["n_hands_model"] for r in hand_rows_reliable]
    summarize([float(v) for v in actual_hands], "実測手数")
    summarize([float(v) for v in model_hands], "モデル手数(÷0.733)")
    overestimate = sum(1 for r in hand_rows_reliable if r["n_hands_model"] > r["n_hands_actual"])
    underestimate = sum(1 for r in hand_rows_reliable if r["n_hands_model"] < r["n_hands_actual"])
    exact = sum(1 for r in hand_rows_reliable if r["n_hands_model"] == r["n_hands_actual"])
    print(
        f"\nモデル > 実測 (過大評価): {overestimate} / {len(hand_rows_reliable)}, "
        f"モデル < 実測 (過小評価): {underestimate} / {len(hand_rows_reliable)}, "
        f"一致: {exact} / {len(hand_rows_reliable)}"
    )


if __name__ == "__main__":
    main()
