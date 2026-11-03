"""B3 段1: 元動画 group 5fold × 3seed の外側 CV と層別・対応付き採点 (事前登録 §3)。

学習器 (trainer) を差し替え可能にする。候補 = 内側 3fold で設定選択した F_static/F_exchange、
比較相手 R_cv = src/b3_train_rcv の本番レシピ再学習。どちらも同じ fold・同じ行で走らせる。
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from src.b3_eval_metrics import (average_over_seeds, ece, game_values, match_equal_mean, paired_bootstrap,
                                 paired_scalar_diff, population, row_brier, row_log_loss, score_predictions)
from src.b3_eval_strata import build_strata
from src.b3_train_config import (BOOTSTRAP_N, BOOTSTRAP_SEED, N_OUTER_FOLDS, STRATUM_MIN_GAMES,
                                 STRATUM_MIN_GROUPS)
from src.b3_train_folds import (assert_no_group_leak, games_per_group_from_rows, inner_assignment,
                                outer_assignment, save_json)
from src.b3_train_model import (DEFAULT_GRID, HgbConfig, InnerChoice, TwoStateModel, feature_columns, fit_two_state, hgb_threads,
                                select_config, monotonic_constraints, model_monotonic_constraints)

INNER_SELECTION_SEED = 1003
INNER_CLOSE_LL = 0.001


@dataclass
class TrainContext:
    """trainer に渡す fold 文脈。"""
    outer_fold: int
    seed: int
    group_of_video: Mapping[str, str]
    inner_of_group: Mapping[str, int]
    columns: Sequence[str]


Predictor = Callable[[pd.DataFrame], np.ndarray]
Trainer = Callable[[pd.DataFrame, TrainContext], tuple[Predictor, dict]]


def selection_margin(choice: InnerChoice) -> dict:
    """最良と2位の LL 差を、費用による選択規則とは独立に記録する。"""
    losses = sorted(row["ll"] for row in choice.table)
    gap = losses[1] - losses[0] if len(losses) > 1 else None
    return {"inner_ll_gap": gap, "inner_margin": "僅差" if gap is not None and gap < INNER_CLOSE_LL else ""}


def _shared_choice(df: pd.DataFrame, ctx: TrainContext, grid: Sequence[HgbConfig], parts: Path) -> InnerChoice:
    """seed 1003 の選択を原子的に公開し、後続ジョブは同じ fold の記録を読む。"""
    path = parts / f"inner_selection_fold{ctx.outer_fold}.json"
    context = {"outer_fold": ctx.outer_fold, "selection_seed": INNER_SELECTION_SEED,
               "monotonic_cst": monotonic_constraints(ctx.columns),
               "columns": list(ctx.columns), "grid": [asdict(cfg) for cfg in grid],
               "inner_of_group": dict(ctx.inner_of_group), "group_of_video": dict(ctx.group_of_video),
               "train_rows_sha256": hashlib.sha256("\n".join(df["row_key"].astype(str)).encode()).hexdigest()}
    if ctx.seed != INNER_SELECTION_SEED:
        saved = json.loads(path.read_text(encoding="utf-8"))
        if saved["context"] != context:
            raise ValueError(f"共有する内側選択の条件が一致しない: {path}")
        return InnerChoice(HgbConfig(**saved["chosen"]), saved["table"], saved["tie_rule_applied"])
    choice = select_config(df, ctx.columns, grid, ctx.seed, ctx.inner_of_group, ctx.group_of_video)
    payload = {**asdict(choice), "context": context, "hgb_threads": hgb_threads(), "blas_threads": 1,
               **selection_margin(choice)}
    temporary = path.with_suffix(".tmp")
    save_json(temporary, payload)
    temporary.replace(path)
    return choice


def candidate_trainer(grid: Sequence[HgbConfig] = DEFAULT_GRID,
                      shared_selection_parts: Optional[Path] = None) -> Trainer:
    """候補 (B-full) の学習手順: 内側 CV で設定選択 → train 全体で 2 評価器を学習。"""
    def train(df: pd.DataFrame, ctx: TrainContext) -> tuple[Predictor, dict]:
        choice = (select_config(df, ctx.columns, grid, ctx.seed, ctx.inner_of_group, ctx.group_of_video)
                  if shared_selection_parts is None else _shared_choice(df, ctx, grid, shared_selection_parts))
        model: TwoStateModel = fit_two_state(df, ctx.columns, choice.chosen, ctx.seed)
        info = {"chosen": choice.chosen.name, "inner_table": choice.table, "tie_rule": choice.tie_rule_applied,
                "monotonic_cst": model_monotonic_constraints(model),
                "dropped_all_nan": {n: e.dropped_all_nan for n, e in
                                    (("static", model.static), ("exchange", model.exchange)) if e is not None},
                "n_train_rows": int(len(df)), "hgb_threads": hgb_threads(), "blas_threads": 1,
                "share_inner_selection": shared_selection_parts is not None,
                "selection_seed": ctx.seed if shared_selection_parts is None else INNER_SELECTION_SEED,
                **selection_margin(choice)}
        return model.predict, info
    return train


@dataclass
class CvRun:
    """1 trainer の外側 CV 結果。"""
    name: str
    preds: dict[int, np.ndarray] = field(default_factory=dict)         # seed -> 行ごとの p (df 行順)
    infos: list[dict] = field(default_factory=list)
    fold_of_row: Optional[np.ndarray] = None


def run_cv(name: str, df: pd.DataFrame, trainer: Trainer, group_of_video: Mapping[str, str],
           seeds: Sequence[int], columns: Sequence[str], n_folds: int = N_OUTER_FOLDS,
           progress: Optional[Callable[[str], None]] = None, folds: Optional[Sequence[int]] = None) -> CvRun:
    """全 seed × 外側 fold で学習し、OOF 予測を df の行順で返す。fold 割当は seed によらず 1 つ。

    folds を指定すると一部の fold だけ学習する (並列ジョブ用。他の fold の p は NaN)。割当は常に全 group で決める。"""
    gpg = games_per_group_from_rows(df, group_of_video)
    outer = outer_assignment(gpg)
    groups = df["video_id"].map(group_of_video).to_numpy()
    fold_row = np.array([outer[g] for g in groups])
    run = CvRun(name, fold_of_row=fold_row)
    for seed in seeds:
        p = np.full(len(df), np.nan)
        for k in (range(n_folds) if folds is None else folds):
            test = fold_row == k
            if not test.any():
                raise ValueError(f"outer fold {k} が空")
            assert_no_group_leak(set(groups[~test]), set(groups[test]))
            train_df = df[~test]
            inner = inner_assignment(gpg, k, outer)
            ctx = TrainContext(k, seed, group_of_video, inner, columns)
            predict, info = trainer(train_df, ctx)
            p[test] = predict(df[test])
            run.infos.append({"seed": seed, "outer_fold": k, "hgb_threads": hgb_threads(),
                              "blas_threads": 1, **info})
            if progress:
                progress(f"{name} seed={seed} fold={k} done")
        if folds is None and np.isnan(p).any():
            raise AssertionError("OOF に未予測行がある")
        run.preds[seed] = p
    return run


def _stratum_stats(df: pd.DataFrame, mask: np.ndarray, p_c: dict[int, np.ndarray], p_r: dict[int, np.ndarray],
                   group_of_video: Mapping[str, str], status: Optional[np.ndarray]) -> dict:
    """1 層の候補−比較相手の ΔLL (bootstrap)・ΔBrier・ΔECE・母数。"""
    from src.b3_eval_metrics import apply_null
    y = df["won_p1"].to_numpy(dtype=np.float64)
    lc, lr_, bc, br_ = [], [], [], []
    dece = []
    for seed in p_c:
        pc, pr = apply_null(p_c[seed], status), apply_null(p_r[seed], status)
        lc.append(game_values(df, row_log_loss(y, pc), group_of_video, mask))
        lr_.append(game_values(df, row_log_loss(y, pr), group_of_video, mask))
        bc.append(game_values(df, row_brier(y, pc), group_of_video, mask))
        br_.append(game_values(df, row_brier(y, pr), group_of_video, mask))
        dece.append(paired_scalar_diff(df, "won_p1", pc, pr, ece, mask))
    glc, glr = average_over_seeds(lc), average_over_seeds(lr_)
    gbc, gbr = average_over_seeds(bc), average_over_seeds(br_)
    boot = paired_bootstrap(glc, glr, BOOTSTRAP_N, BOOTSTRAP_SEED)
    pop = population(glc)
    return {"delta_ll": boot, "ll_cand": match_equal_mean(glc), "ll_ref": match_equal_mean(glr),
            "delta_brier": match_equal_mean(gbc) - match_equal_mean(gbr),
            "delta_ece": float(np.mean(dece)), "population": pop,
            "meets_min_population": bool(pop["games"] >= STRATUM_MIN_GAMES and pop["groups"] >= STRATUM_MIN_GROUPS)}


def stage1_scores(df: pd.DataFrame, cand: CvRun, ref: CvRun, group_of_video: Mapping[str, str],
                  status: Optional[np.ndarray] = None, strata: Optional[Mapping[str, np.ndarray]] = None) -> dict:
    """全体 + 全層の対応付き採点。Brier/ECE は全体の差を Stage1 門の相手にする。"""
    if set(cand.preds) != set(ref.preds):
        raise ValueError("候補と R_cv の seed が一致しない")
    masks = dict(strata) if strata is not None else build_strata(df)
    out = {"strata": {}, "seeds": sorted(cand.preds)}
    for name, m in masks.items():
        if not m.any():
            out["strata"][name] = {"population": {"games": 0, "groups": 0, "videos": 0, "rows": 0, "held_sec": 0.0},
                                   "meets_min_population": False, "empty": True}
            continue
        out["strata"][name] = _stratum_stats(df, m, cand.preds, ref.preds, group_of_video, status)
    out["overall"] = out["strata"]["overall"]
    # seed 平均前の単独成績も併記 (報告用)
    s0 = sorted(cand.preds)[0]
    out["cand_seed0"] = score_predictions(df, cand.preds[s0], group_of_video, status)
    out["ref_seed0"] = score_predictions(df, ref.preds[s0], group_of_video, status)
    return out
