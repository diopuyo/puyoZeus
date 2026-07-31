# -*- coding: utf-8 -*-
"""勝利数パネル境界 x 盤面ぷよ総量で位相ラベルを再定義し win-AUC を再評価する (2026-07-30)。

背景 (コーディネーター経由でuser確定定義を受領、複数回の指示更新を反映した最終版):

1. 試合の切り出し
   手数(tsumo)リセット検出でなく勝利数パネル境界
   (data/verify/winners_panel_diff_2026-07-26/video_*.json の start_sec/end_sec)
   を使う。tsumoはフレーム間引きで壊れる量で複数試合が融合していた実測あり
   (feedback_indicator_sampling_10frames_2026-07-29)。

2. 試合内の3分割軸 = 盤面のぷよ総量 (おじゃま含む、空・UNKNOWN除く)
   user確定: 「盤面の、お邪魔を含めたぷよ量で判断します」。
   時間進行率や手数は「本線を撃ち合って盤面が空くと序盤に戻る」現象を表現
   できない(定義上単調増加のため)ので不採用。ぷよ総量は各時点の盤面から
   直接計算されるため後戻りも自然に表現される。
   既存列 board_puyo_total_raw (= Board.count_puyos(), おじゃま含む・空/UNKNOWN
   除く) をそのまま使う。この値は盤面内部表現の全13行 (隠し段row0含む) を
   対象に計算されている (src/board.py Board.count_puyos() 実装確認済み、
   combined66実測で72超の行が317/73416=0.43%存在し隠し段を含む証拠)。
   「隠し段を含めるか未指定」との指示に対し、本スクリプトは「含めた版」
   (=既存列そのまま) を採用する。

3. 閾値 (user指定の実数、三分位計算はしない)
   初回指示: 序盤 raw<=20 / 中盤 21<=raw<=56 / 終盤 raw>=57
   2026-07-30 改定 (これが正・デフォルト): 序盤 raw<=18 / 中盤 19<=raw<=47 /
   終盤 raw>=48 (reference_phase_split_by_color_puyo_count_2026-07-30)。
   将来の再改定に備え、閾値は環境変数 PUYO_PHASE_EARLY_MAX /
   PUYO_PHASE_LATE_MIN で上書き可能にパラメータ化した(マジックナンバー禁止規約
   準拠)。出力先 OUT_DIR は閾値の組ごとに自動で分かれるため、異なる閾値で
   実行しても過去の結果ファイルを上書きしない。

4. 1P/2P統合 = 遅い方(終盤寄り)を採用
   user確定:「序盤＜中盤＜後半で遅い方を採用する」。合計でも平均でもない。

5. 「切り出し方法」と「軸」の効果を分離するため以下を並べる:
   A. 旧(公表値): 手数リセット切り出し x 手数の相対位置軸
      (data/verify/win_eval_combined66_2026-07-29/relphase_combined66/
       relphase_auc_summary.csv を読むのみ、再計算しない)
   D. 旧切り出し(手数リセット) x 新軸(ぷよ総量20/57、遅い方)
      base.build_phase_map/merge_phase (2026-07-26作成、変更しない) を
      read-only importして再利用し、旧方式のseg_max_1p(contamination判定)
      のみ借用する。
   E. 新切り出し(勝利数パネル) x 新軸(ぷよ総量20/57、遅い方) ← 本命

既存ファイルは一切変更しない。scripts/_tmp_relphase_win_auc_2026-07-26.py の
build_phase_map/merge_phase/_oof_auc_for_mask を read-only import して
再利用する。

データ出自:
  labeled_win: data/verify/win_eval_combined66_2026-07-29/labeled_win_combined66.csv
    (m30+m20+c20 単純concat、73416ペア行、2026-07-29生成)
  試合境界(新軸の切り出し元): data/verify/winners_panel_diff_2026-07-26/video_c*.json
    (93動画分、2026-07-26生成)
  study CSV(旧切り出しのseg_max取得用): data/verify/labeled_win_{m30_2026-07-28,
    m20_2026-07-28,c20_2026-07-26}/study/ (combined66公式評価と同一ソース)
  旧結果: data/verify/win_eval_combined66_2026-07-29/relphase_combined66/
    relphase_auc_summary.csv (2026-07-29生成、読むだけ)

使い方:
    PYTHONPATH=. ./venv/bin/python -m scripts._diag_relphase_by_winpanel_2026-07-30
"""
from __future__ import annotations

import importlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import scripts.model_indicator_win as miw  # noqa: E402

# ファイル名がハイフンを含み import 文の識別子構文では読めないため
# importlib.import_module (文字列引数) 経由で read-only 参照する。
base = importlib.import_module("scripts._tmp_relphase_win_auc_2026-07-26")

LABELED_WIN_CSV = "data/verify/win_eval_combined66_2026-07-29/labeled_win_combined66.csv"
WINPANEL_DIR = Path("data/verify/winners_panel_diff_2026-07-26")
OLD_RELPHASE_SUMMARY_CSV = Path(
    "data/verify/win_eval_combined66_2026-07-29/relphase_combined66/relphase_auc_summary.csv"
)
def _out_dir_for_thresholds(early: float, late: float) -> Path:
    """閾値の組に応じて出力先ディレクトリを分ける(過去結果を上書きしないため)。"""
    if early == 20.0 and late == 57.0:
        return Path("data/verify/relphase_winpanel_2026-07-30")
    if early == 18.0 and late == 48.0:
        return Path("data/verify/relphase_winpanel_th18_48_2026-07-30")
    return Path(f"data/verify/relphase_winpanel_th{early:.0f}_{late:.0f}_2026-07-30")

# 旧切り出し(手数リセット)のstudyディレクトリ (combined66公式評価と同一ソース)
OLD_STUDY_DIRS = [
    Path("data/verify/labeled_win_m30_2026-07-28/study"),
    Path("data/verify/labeled_win_m20_2026-07-28/study"),
    Path("data/verify/labeled_win_c20_2026-07-26/study"),
]

# 極端な試合長を除外する境界 (勝利数パネル側 clean フィルタ用)。combined66対象
# 66動画・3627試合の実測分布より (p1=4秒/p5=18秒/p50=54秒/p99=122秒/p99.5=134秒、
# 最大2件が722秒・1122秒の明らかな検出破綻)。正常分布の裾を広めに残しつつ
# 破綻2件のみ除く形で設定 (実際にlabeled_winが触れる試合は20-158秒に収まり
# 本フィルタでの除外は0件だった。理由は下記診断出力を参照)。
EXCLUDE_MIN_DURATION_SEC: float = 15.0
EXCLUDE_MAX_DURATION_SEC: float = 200.0

def _resolve_threshold(env_name: str, default: float) -> float:
    """環境変数があれば優先、無ければデフォルト値を使う(閾値パラメータ化)。"""
    raw = os.environ.get(env_name)
    return float(raw) if raw is not None else default


# 盤面ぷよ総量(おじゃま含む)の位相境界。
# デフォルトは2026-07-30改定の確定値(18/48)。初回指示の20/57は環境変数で
# 明示指定した場合のみ再現する(reference_phase_split_by_color_puyo_count_2026-07-30)。
PUYO_TOTAL_EARLY_MAX: float = _resolve_threshold("PUYO_PHASE_EARLY_MAX", 18.0)
PUYO_TOTAL_LATE_MIN: float = _resolve_threshold("PUYO_PHASE_LATE_MIN", 48.0)

OUT_DIR = _out_dir_for_thresholds(PUYO_TOTAL_EARLY_MAX, PUYO_TOTAL_LATE_MIN)

# 旧方式のcontamination除外しきい値 (base モジュールと同一値を明示的に踏襲)
CONTAMINATION_SEG_MAX: float = base.CONTAMINATION_SEG_MAX

# 位相の順序 (1P/2P統合で「遅い方」を採る際の比較に使う)
_PHASE_ORDER: dict[str, int] = {"序盤": 0, "中盤": 1, "終盤": 2}
_PHASE_ORDER_INV: dict[int, str] = {v: k for k, v in _PHASE_ORDER.items()}

def phase_from_puyo_total(raw: pd.Series) -> pd.Series:
    """盤面ぷよ総量(おじゃま含む)から序盤/中盤/終盤を割り当てる(閾値はモジュール定数)。"""
    return pd.Series(
        np.select(
            [raw <= PUYO_TOTAL_EARLY_MAX, raw >= PUYO_TOTAL_LATE_MIN],
            ["序盤", "終盤"], default="中盤",
        ),
        index=raw.index,
    )


def combine_phase_slower_side(phase_1p: pd.Series, phase_2p: pd.Series) -> pd.Series:
    """1P/2Pの位相のうち、より終盤寄り(序盤<中盤<終盤)を試合全体の位相とする。"""
    order_1p = phase_1p.map(_PHASE_ORDER)
    order_2p = phase_2p.map(_PHASE_ORDER)
    combined_order = np.maximum(order_1p.values, order_2p.values)
    return pd.Series([_PHASE_ORDER_INV[int(v)] for v in combined_order], index=phase_1p.index)


def build_puyo_total_phase(paired: pd.DataFrame) -> pd.Series:
    """paired (1P/2P列を持つ) から遅い方基準のぷよ総量位相を計算する。"""
    phase_1p = phase_from_puyo_total(paired["board_puyo_total_raw_1p"].astype(float))
    phase_2p = phase_from_puyo_total(paired["board_puyo_total_raw_2p"].astype(float))
    return combine_phase_slower_side(phase_1p, phase_2p)


def report_hidden_row_impact(paired: pd.DataFrame) -> None:
    """隠し段(row0)を含むことによる影響件数を参考値として報告する(72超の行数)。"""
    over_1p = int((paired["board_puyo_total_raw_1p"] > 72).sum())
    over_2p = int((paired["board_puyo_total_raw_2p"] > 72).sum())
    n = len(paired)
    print(f"[隠し段影響] board_puyo_total_raw>72の行: 1P {over_1p}/{n}"
          f" ({over_1p / n:.2%})  2P {over_2p}/{n} ({over_2p / n:.2%})"
          " (この値は既存列が全13行(隠し段含む)を対象に計算されている証拠)")


def load_winpanel_games(video_ids: list[str]) -> pd.DataFrame:
    """指定 video_id 群の勝利数パネル試合境界 JSON を読み込み結合する。"""
    rows: list[dict] = []
    missing: list[str] = []
    for vid in video_ids:
        path = WINPANEL_DIR / f"{vid}.json"
        if not path.exists():
            missing.append(vid)
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        for g in data["games"]:
            rows.append({
                "video_id": vid,
                "game_abs_idx": g["game_abs_idx"],
                "start_sec": float(g["start_sec"]),
                "end_sec": float(g["end_sec"]),
                "confidence": g.get("confidence"),
            })
    if missing:
        print(f"[warn] winners_panel JSON 不在 ({len(missing)}件): {missing}")
    games = pd.DataFrame(rows)
    games["duration_sec"] = games["end_sec"] - games["start_sec"]
    print(f"[winpanel] 読み込み: {len(games)} 試合 ({games['video_id'].nunique()} 動画)")
    return games


def _match_one_video(grp: pd.DataFrame, g_games: pd.DataFrame) -> pd.DataFrame:
    """1動画分の paired 行に対し、絶対時刻 t_sec_1p が属する試合区間を特定する。"""
    grp_sorted = grp.sort_values("t_sec_1p")
    if len(g_games) == 0:
        out = grp_sorted.copy()
        for col in ("game_abs_idx_1p", "game_duration_sec_1p"):
            out[col] = np.nan
        out["game_confidence_1p"] = None
        out["_in_game_1p"] = False
        return out
    g_sorted = g_games.sort_values("start_sec").rename(columns={
        "game_abs_idx": "game_abs_idx_1p", "confidence": "game_confidence_1p",
        "duration_sec": "game_duration_sec_1p",
    })
    merged = pd.merge_asof(
        grp_sorted, g_sorted, left_on="t_sec_1p", right_on="start_sec", direction="backward",
    )
    in_range = merged["t_sec_1p"] < merged["end_sec"]
    merged["_in_game_1p"] = in_range.fillna(False)
    meta_cols = ["game_abs_idx_1p", "game_duration_sec_1p", "game_confidence_1p"]
    merged.loc[~merged["_in_game_1p"], meta_cols] = np.nan
    return merged.drop(columns=["video_id", "start_sec", "end_sec"])


def assign_game_boundary(paired: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """paired 全行に対し動画ごとに勝利数パネル試合区間をマッチングする(新切り出し)。"""
    parts = [
        _match_one_video(grp, games[games["video_id"] == vid])
        for vid, grp in paired.groupby("video_id_1p")
    ]
    result = pd.concat(parts, ignore_index=True)
    n_unmatched = int((~result["_in_game_1p"]).sum())
    n_total = len(result)
    print(f"[assign_game_boundary] 非マッチ行(試合区間外): {n_unmatched} / {n_total}"
          f" ({n_unmatched / n_total:.1%})")
    return result


def build_winpanel_clean_mask(merged: pd.DataFrame) -> np.ndarray:
    """新切り出しの clean マスク: confidence=strict かつ試合長が正常範囲内。"""
    strict = (merged["game_confidence_1p"] == "strict").values
    dur = merged["game_duration_sec_1p"].values
    dur_ok = (dur >= EXCLUDE_MIN_DURATION_SEC) & (dur <= EXCLUDE_MAX_DURATION_SEC)
    return strict & dur_ok


def load_old_cut_seg_max(paired: pd.DataFrame) -> pd.DataFrame:
    """旧方式(手数リセット)のseg_max_1p(contamination判定用)をpairedに結合する。

    base.build_phase_map/merge_phase (2026-07-26作成、変更しない) を
    read-only importして再利用する。combined66公式評価と同一のstudyソース。
    """
    phase_map_parts = [base.build_phase_map(d) for d in OLD_STUDY_DIRS]
    phase_map = pd.concat(phase_map_parts, ignore_index=True)
    return base.merge_phase(paired, phase_map)


def evaluate_by_phase(paired: pd.DataFrame, phase: pd.Series, label: str) -> dict[str, float]:
    """phase (序盤/中盤/終盤カテゴリ) 別に OOF AUC を計算する。base._oof_auc_for_mask を再利用。"""
    print(f"\n=== 位相評価 ({label}) ===  対象行数: {len(paired)}")
    y = paired["won_1p"].astype(int).values
    groups = paired["video_id_1p"].values
    leak_cols = frozenset([
        "game_abs_idx", "game_duration_sec", "seg_max", "rel_phase",
    ])
    indicator_cols = [c for c in miw._get_indicator_cols(paired) if c not in leak_cols]
    feat_df = miw.build_features(paired, indicator_cols)
    X = feat_df.fillna(0.0).values.astype(float)
    masks = {
        "全体": np.ones(len(paired), dtype=bool),
        "序盤": (phase == "序盤").values,
        "中盤": (phase == "中盤").values,
        "終盤": (phase == "終盤").values,
    }
    result: dict[str, float] = {}
    for name, mask in masks.items():
        result[name] = base._oof_auc_for_mask(paired, X, y, groups, mask, f"{label}:{name}")
    return result


def load_old_relphase_summary(path: Path) -> pd.DataFrame:
    """既存(手数リセット x 手数相対位置)combined66 AUC結果をそのまま読み込む(再計算しない)。"""
    if not path.exists():
        print(f"[warn] 旧結果ファイルが見つからない: {path}")
        return pd.DataFrame()
    old = pd.read_csv(path)
    print(f"[old_relphase] 読込: {path} ({len(old)} 行)")
    return old


def report_touched_games_confidence(merged: pd.DataFrame) -> None:
    """新切り出しでlabeled_winが実際に触れた試合(ユニーク)のconfidence内訳を報告する。

    全3627試合中のconfidence分布(strict約93%)と food行単位の分布が一致するとは
    限らない(サンプリング窓が試合を選ばないため)。診断用に両方報告する。
    """
    touched = merged[merged["_in_game_1p"]][
        ["video_id_1p", "game_abs_idx_1p", "game_confidence_1p", "game_duration_sec_1p"]
    ].drop_duplicates()
    print(f"[診断] labeled_winが触れたユニーク試合数: {len(touched)}")
    print(f"[診断] そのconfidence内訳:\n{touched['game_confidence_1p'].value_counts()}")
    print(f"[診断] その試合長(game単位)分布:\n{touched['game_duration_sec_1p'].describe()}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== 1. labeled_win.csv 読み込み・ペアリング ===")
    df = miw.load_labeled_csv(LABELED_WIN_CSV)
    paired = miw.pair_sides_for_win(df, miw.DEFAULT_MAX_TDIFF)
    video_ids = sorted(paired["video_id_1p"].unique())
    print(f"  対象動画数: {len(video_ids)}")
    report_hidden_row_impact(paired)

    print("\n=== 2. 新切り出し(勝利数パネル)マッチング ===")
    games = load_winpanel_games(video_ids)
    new_merged = assign_game_boundary(paired, games)
    report_touched_games_confidence(new_merged)
    new_merged.to_csv(OUT_DIR / "relphase_winpanel_map.csv", index=False)

    print("\n=== 3. 旧切り出し(手数リセット) seg_max 結合 ===")
    old_merged = load_old_cut_seg_max(paired)

    results, row_counts = run_all_variants(new_merged, old_merged)

    print("\n=== 4. 旧(公表値)結果 読み込み ===")
    old_summary = load_old_relphase_summary(OLD_RELPHASE_SUMMARY_CSV)

    n_unmatched_new = int((~new_merged["_in_game_1p"]).sum())
    _print_and_save_summary(results, row_counts, old_summary, n_unmatched_new, len(new_merged))


def run_all_variants(
    new_merged: pd.DataFrame, old_merged: pd.DataFrame,
) -> tuple[dict[str, dict[str, float]], dict[str, int]]:
    """D(旧切り出しx新軸)・E(新切り出しx新軸) の all/clean 版をまとめて評価する。"""
    results: dict[str, dict[str, float]] = {}
    counts: dict[str, int] = {}

    old_matched = old_merged[old_merged["rel_phase_1p"].notna()].copy()
    old_clean = old_matched[old_matched["seg_max_1p"] <= CONTAMINATION_SEG_MAX].copy()
    old_matched["_puyo_phase"] = build_puyo_total_phase(old_matched)
    old_clean["_puyo_phase"] = build_puyo_total_phase(old_clean)
    results["D_oldcut_puyoaxis_all"] = evaluate_by_phase(
        old_matched, old_matched["_puyo_phase"], "D_oldcut_puyoaxis_all")
    counts["D_oldcut_puyoaxis_all"] = len(old_matched)
    results["D_oldcut_puyoaxis_clean"] = evaluate_by_phase(
        old_clean, old_clean["_puyo_phase"], "D_oldcut_puyoaxis_clean")
    counts["D_oldcut_puyoaxis_clean"] = len(old_clean)

    new_matched = new_merged[new_merged["_in_game_1p"]].copy()
    new_clean_mask = build_winpanel_clean_mask(new_matched)
    new_clean = new_matched[new_clean_mask].copy()
    new_matched["_puyo_phase"] = build_puyo_total_phase(new_matched)
    new_clean["_puyo_phase"] = build_puyo_total_phase(new_clean)
    results["E_newcut_puyoaxis_all"] = evaluate_by_phase(
        new_matched, new_matched["_puyo_phase"], "E_newcut_puyoaxis_all")
    counts["E_newcut_puyoaxis_all"] = len(new_matched)
    results["E_newcut_puyoaxis_clean"] = evaluate_by_phase(
        new_clean, new_clean["_puyo_phase"], "E_newcut_puyoaxis_clean")
    counts["E_newcut_puyoaxis_clean"] = len(new_clean)

    return results, counts


def _format_row(label: str, n_rows: object, res: dict[str, float]) -> str:
    """比較表の1行を整形する。"""
    n_str = "-" if n_rows is None else str(n_rows)
    vals = "  ".join(f"{res.get(p, float('nan')):>8.4f}" for p in ["全体", "序盤", "中盤", "終盤"])
    return f"  {label:<32}  {n_str:>8}  {vals}"


def _print_and_save_summary(
    results: dict[str, dict[str, float]],
    row_counts: dict[str, int],
    old_summary: pd.DataFrame,
    n_unmatched_new: int,
    n_total_new: int,
) -> None:
    """新旧結果を整形して表示・CSV保存する。"""
    print("\n" + "=" * 86)
    print("  相対位相 win-AUC 比較 (A:旧軸/旧切り出し, D:新軸/旧切り出し, E:新軸/新切り出し)")
    print("=" * 86)
    print(f"  {'条件':<32}  {'行数':>8}  {'全体':>8}  {'序盤':>8}  {'中盤':>8}  {'終盤':>8}")
    rows_out: list[dict] = []
    for _, r in old_summary.iterrows():
        cond = "A_old_" + str(r["condition"])
        res = {p: r.get("auc_" + p) for p in ["全体", "序盤", "中盤", "終盤"]}
        print(_format_row(cond, None, res))
        rows_out.append({"condition": cond, "n_rows": None, **{f"auc_{k}": v for k, v in res.items()}})
    for label, res in results.items():
        print(_format_row(label, row_counts[label], res))
        rows_out.append({"condition": label, "n_rows": row_counts[label],
                          **{f"auc_{k}": v for k, v in res.items()}})
    out_csv = OUT_DIR / "relphase_winpanel_auc_summary.csv"
    pd.DataFrame(rows_out).to_csv(out_csv, index=False)
    print(f"\n  新切り出しで試合区間に紐付かなかった行: {n_unmatched_new} / {n_total_new}"
          f" ({n_unmatched_new / n_total_new:.1%})")
    print(f"  採用したぷよ総量閾値: 序盤<= {PUYO_TOTAL_EARLY_MAX:.0f}個 "
          f" 中盤(その間)  終盤>= {PUYO_TOTAL_LATE_MIN:.0f}個 (1P/2Pは遅い方を採用)")
    print(f"\n[save] {out_csv}")


if __name__ == "__main__":
    main()
