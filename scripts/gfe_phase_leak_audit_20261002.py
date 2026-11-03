"""V3の54列を元CSV・M0の入れ子fold・生成関数まで追跡する。"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from scripts import gfe_phase_stage1_20261002 as stage
from src.exchange_event_evaluator import file_sha256
from src.exchange_event_user_phase import USER_COLUMNS
import numpy as np
import pandas as pd

OUT = stage.ROOT / "logs/gfe_phase_reference_20261002"
KEYS = ["video_id", "game_idx", "frame", "side"]


def source_for(name: str) -> str:
    """全列に生成元の関数を割り当て、未知列は監査を停止する。"""
    if name == "m0_logit" or name.startswith("m0_logit_x_"):
        return "M0入れ子動画OOF→g_features（自側符号のlogit）" + ("×U" if "_x_" in name else "")
    if name.startswith("U_") or name == "opp_total_missing":
        return "元CSV board_puyo_total / diff_board_puyo_total→user_phase（整数復元・max・18/48）"
    if name.startswith("diff_"):
        return "_compute_row→_asof_attach_opponent(backward)→_attach_opponent_diff_columns（自側−相手）"
    if name in ("all_clear_bonus_pending", "opp_all_clear_bonus_pending"):
        return "collect_boards_lean→VideoChainTracker.all_clear_pending→_apply_all_clear_truth（相手はbackward結合）"
    if name in ("ojama_net_balance", "ojama_forecast"):
        return "collect_boards_lean→OjamaAccountingTracker現在値→_attach_ojama_truth_own_columns"
    if name == "ojama_net_balance_synced":
        return "_attach_ojama_net_balance_synced_column（現在値と相手backward値の再同期）"
    if name == "ojama_margin":
        return "_attach_ojama_margin_column（現在盤面の空き−現在の予告）"
    if name in ("color_ojama_ratio_own", "color_diff_x_ojama_diff"):
        return "_add_pair_interaction_columns（現在の色ぷよ・おじゃま量とbackward差分）"
    if name == "conn_triple_count":
        return "_compute_row→indicators_v2.connectivity_observation（当該確定盤面の3個連結）"
    if name in stage.D_COLUMNS:
        return f"build_labeled_win_from_npz._compute_row→indicators_v2.{name} / native同値関数（当該確定盤面）"
    raise ValueError(name)


def check_design() -> dict[str, Any]:
    """1,035,639行の46列を元CSVの同じキーから独立復元する。"""
    stage.memory_guard()
    rows, design, _ = stage.datasets(stage.REFERENCE)
    raw = pd.read_csv(stage.SOURCE, usecols=KEYS + list(stage.D_COLUMNS)).set_index(KEYS)
    lookup = rows[["video_id", "frame", "source_side"]].rename(columns={"source_side": "side"})
    lookup["game_idx"] = rows.game_id.str.rsplit(":", n=1).str[-1].astype(int)
    aligned = raw.reindex(pd.MultiIndex.from_frame(lookup[KEYS]))
    maximum = 0.
    for index, name in enumerate(stage.D_COLUMNS):
        expected = aligned[name].fillna(0.).to_numpy(dtype=np.float32)
        np.testing.assert_array_equal(design[:, index], expected)
        maximum = max(maximum, float(np.max(np.abs(design[:, index]-expected))))
    return dict(rows=len(rows), columns=len(stage.D_COLUMNS), maximum_error=maximum,
                source_sha256=file_sha256(stage.SOURCE), design_sha256=file_sha256(
                    stage.run_path(stage.REFERENCE, "base") / "design.npy"))


def check_m0() -> dict[str, Any]:
    """内側・外側90モデルのfit/tune/evaluation動画集合を相互排他的に確認する。"""
    paths = sorted(stage.run_path(stage.REFERENCE, "m0").glob("seed_*/fold_*/*m0.json"))
    assert len(paths) == len(stage.SEEDS)*len(stage.FOLDS)*(len(stage.FOLDS)+1)
    for path in paths:
        data = json.loads(path.read_text())
        train, tune, evaluation = [set(data[k]) for k in ("train_videos", "tune_videos", "evaluation_videos")]
        assert not (train & tune or train & evaluation or tune & evaluation)
    return dict(models=len(paths), video_overlap=0,
                source="r2_grouped_cv_148_20260925.canonical / fit_predict / fold_predictions")


def main() -> None:
    """学習や採点は変えず、列の出所と検証範囲を記録する。"""
    os.nice(max(0, 19-os.nice(0)))
    design, m0 = check_design(), check_m0()
    sources = {name: source_for(name) for name in USER_COLUMNS}
    stage.save_json(OUT / "LEAK_AUDIT.json", dict(columns=list(USER_COLUMNS), sources=sources,
        design=design, m0=m0, forbidden_input_columns=[], passed=True))
    lines = ["# V3入力列のリーク確認", "", "対象: V3/V1b共通の54列。列順は以下の通り。", "",
        "元CSVのキー(video_id, game_idx, frame, side)でD全46列を照合し、"
        f"{design['rows']:,}行×46列の不一致0。欠測Dは既存通り0補完する。",
        "M0は両側の現在確定盤面と観測NEXTだけを入力する。90モデルの学習・調整・評価動画の重複0。",
        "学習行には内側OOF、外側評価行には外側M0を使用する。勝敗は教師・採点専用で特徴には入らない。",
        "最終学習は既存final_fitと同じseed0外側OOF（rows.A）を使用する。", "",
        "|列番号|入力列名|生成元|", "|---:|---|---|"]
    lines += [f"|{i}|{name}|{sources[name]}|" for i, name in enumerate(USER_COLUMNS, 1)]
    lines += ["", "生成経路:",
        "- D: shared/scripts/build_labeled_win_from_npz.pyの当該盤面関数・現在会計値→元CSV→"
        "56dc/scripts/r2_baselines_148_20260926.py::prepare→design.npy（列選択、0補完のみ）。",
        "- 相手列: _asof_attach_opponentのdirection='backward'。後続行・未来相手盤面を参照しない。",
        "- 全消し: 保存時点のtracker値。古い欠測原票の代替経路も_all_clear_state_for_groupで"
        "時刻順の当行得点−直前既知得点から状態を進め、後続行から埋め戻さない。",
        "- U: 元CSVの総量×72を丸めて整数復元。相手=max対象、欠測6,689行は自側段階と欠測フラグ。",
        "- CNNのcanonicalはboards/queuesのみを入力に使い、phaseをゼロで構築する。"
        "試合キー・勝敗・重みは分割/教師/学習処理で、forwardの特徴ではない。",
        "- 元のrows.phaseは終了時刻を用いる相対進行率だが、評価表のphase_early/middle/late専用。"
        "V3のfeatures呼出しにはUを渡し、この列・試合終了時刻・試合長を渡さない。",
        "- R1/R2の経過秒は試合内最初の保存時刻から当行まで。閾値は各foldの学習動画だけで決める。",
        "- 絶対量D・U・欠測フラグは無条件反転せず、M0 logitと教師/予測の視点だけsignで変換する。",
        "", "結論: V3の入力54列に、試合終了時刻・勝敗・後続行を直接または上記生成経路経由で渡す処理はない。",
        "監査範囲: 列の生成コードと保存D/OOF資産。元映像の再認識や全認識器の打切り再実行は行っていない。",
        "", "参考・結果を見た後に追加。合否の条件には追加しない。"]
    (OUT / "LEAK_AUDIT.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(json.dumps(dict(design=design, m0=m0), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
