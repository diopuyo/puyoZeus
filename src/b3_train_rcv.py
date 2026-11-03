"""R_cv: 比較基準 R (本番 ccc9623) の学習済み勝率部品を、外側 fold の train 側だけで学習し直したもの (事前登録 §1.2)。

R の既存モデルは旧 144 動画で学習済みで、B3 の 58 動画を含む。そのまま段1で採点すると R だけが test 動画を
学習済みになる。R_cv は「同じ本番レシピ (列・ハイパーパラメータ・学習コード)」を train 側だけで再学習する。

本モジュールは **レシピ水準の再現** (B3 学習行の列を R の列名へ対応させて同じ推定器で学習) であり、
本番評価器の再生 (EventInput 駆動) ではない。対応できない列は coverage で明示する (黙って落とさない)。
再学習する部品:
  1. G_fe 静止 (R1s)   : HistGBC 300iter/lr0.05/leaf31/minleaf200、59列 (出典 wt_phase gfe_phase_stage2b_train)
  2. S1'/S3' 撃ち合い : HistGBC 既定系 (出典 exev models/exchange_event_v3, 98/105列) — 部分対応
  3. 仮想着弾 G_fe LR  : LogisticRegression C=1 (G_fe 同59列) — 本実装では未使用 (欠測非対応、§3.4 の中央値+mask 版が要る)
  4. M0 (盤面 CNN の logit): b3_m0_per_foldのnested OOFを任意指定。既定OFF。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from src.b3_train_config import HGB_REFERENCE
from src.b3_train_model import HgbConfig, MODE_COLUMN, fit_two_state
from src.b3_train_matrix import (MATRIX_ATTR, ROW_POSITION, SharedMatrix, available_columns)

CELLS = 13 * 6                       # exev exchange_event_features.CELLS
PHASE_BOUNDS = (1 / 3, 2 / 3)        # 充填率の三分位 (exev PHASE_BOUNDS)
ELAPSED_QUANTILES = PHASE_BOUNDS     # 経過秒閾値は train の分位 (R と同じ作り)

# R の D_COLUMNS 46 列 -> B3 列 (own = p1)。値は列名 or None (対応なし)。
D_OWN = ("board_color_puyo_total", "board_puyo_total", "center_bulge_color", "center_bulge_ojama",
         "board_ojama_count", "color_diversity_evenness", "buried_hole_count", "current_max_chain",
         "dig_resistance", "ukeyasusa", "sub_chain_count", "chain_efficiency", "min_puyos_to_ignite",
         "second_chain_potential", "main_linked_pair_count", "isolated_pair_count", "main_linked_ratio",
         "ignition_point_count", "multi_color_ignition", "simultaneous_pop_richness",
         "chain_articulation_point_count", "saturation_chain_upper", "conn_triple_count")
D_DIFF = ("max_column_height", "column_bumpiness", "death_margin", "death_margin_neighbor", "conn_pair_count",
          "conn_max_group_size", "board_color_puyo_total", "board_puyo_total", "board_ojama_count",
          "center_bulge_color", "center_bulge_ojama", "current_max_chain", "dig_resistance", "ukeyasusa",
          "sub_chain_count")
# 決定木は単調変換に不変なので、diff_centered (0.5+0.5*(p1-p2)) は R の生差分と同じ分割を作れる。
D_UNMAPPED = ("all_clear_bonus_pending", "ojama_net_balance_synced", "ojama_margin", "opp_all_clear_bonus_pending",
              "color_ojama_ratio_own", "color_diff_x_ojama_diff")
LEDGER_MAP = {"ojama_net_balance": "r_ledger_net_balance", "ojama_forecast": "r_ledger_forecast_p1"}
PHASE_NAMES = ("early", "middle", "late")
M0_COLUMNS = ("m0_logit", *(f"m0_logit_x_{k}_{p}" for k in ("fill", "elapsed") for p in PHASE_NAMES))

R1S_CONFIG = HgbConfig("R1s", HGB_REFERENCE["max_iter"], HGB_REFERENCE["learning_rate"],
                       HGB_REFERENCE["max_leaf_nodes"], HGB_REFERENCE["min_samples_leaf"])
# S1'/S3' は v3 manifest の推定器 (max_iter100/lr0.1/leaf31/minleaf20)。early_stopping 'auto' は決定性のため off 化:
# 学習済み本番とは「停止規則」だけ異なる (coverage に記録)。
S1P_CONFIG = HgbConfig("S1prime", 100, 0.1, 31, 20)


def d_column_map() -> dict[str, Optional[str]]:
    """R の列名 -> B3 列名 (None=対応なし)。"""
    m: dict[str, Optional[str]] = {}
    for n in D_OWN:
        m[n] = f"ind_{n}_p1"
    for n in D_UNMAPPED:
        m[n] = None
    m.update(LEDGER_MAP)
    for n in D_DIFF:
        m[f"diff_{n}"] = f"df_ind_{n}"
    return m


@dataclass
class RcvCoverage:
    """R の学習列のうち B3 列へ対応できたもの / できなかったものの一覧。"""
    mapped: dict[str, str]
    unmapped: list[str]
    derived: list[str]
    notes: list[str]


def coverage(df_columns: Sequence[str], with_m0: bool = False) -> RcvCoverage:
    have = set(df_columns)
    mp, un = {}, []
    for r, b in d_column_map().items():
        if b is not None and b in have:
            mp[r] = b
        else:
            un.append(r)
    derived = [f"{k}_{p}" for k in ("fill", "elapsed") for p in PHASE_NAMES]
    if not with_m0:
        un += ["m0_logit"] + [f"m0_logit_x_{k}_{p}" for k in ("fill", "elapsed") for p in PHASE_NAMES]
    notes = ["レシピ水準の再現 (本番評価器の再生ではない)", "S3' の得点列・S1' の own_/opp_ 側別列・first_trigger は未対応",
             "m0_logit 無しは R を弱める方向 (候補に有利) — 段1 の判定時に注記必須" if not with_m0 else "m0 は aux 注入",
             "R1s は本番と同一 HGB 設定。S1' は early_stopping 'auto' を off にした近似"]
    return RcvCoverage(mp, un, derived, notes)


def phase_inputs(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """R の局面入力: 充填率 = 両側平均の総量/78 (own - diff/2 と同値)、経過秒 = 最初の1手から。"""
    t1 = df["cells_total_p1"].to_numpy(dtype=np.float64, na_value=np.nan)
    t2 = df["cells_total_p2"].to_numpy(dtype=np.float64, na_value=np.nan)
    fill = np.clip((t1 + t2) / 2.0 / CELLS, 0, 1)      # 欠測は NaN のまま (R は 0 補完だが B3 は mask を保つ)
    elapsed = df["elapsed_first_move"].to_numpy(dtype=np.float64, na_value=np.nan)
    return fill, elapsed


def _bucket(x: np.ndarray, bounds: Sequence[float]) -> np.ndarray:
    out = np.searchsorted(np.asarray(bounds), x, side="left").astype(np.float64)
    out[~np.isfinite(x)] = np.nan
    return out


def r_design(df: pd.DataFrame, elapsed_bounds: Sequence[float], cov: RcvCoverage,
             m0: Optional[np.ndarray] = None) -> pd.DataFrame:
    """R の列名の学習行列。経過秒の閾値は train の分位 (呼出側が train から決める)。"""
    store = df.attrs.get(MATRIX_ATTR)
    cols = {} if store else {r: df[b].to_numpy(dtype=np.float64, na_value=np.nan) for r, b in cov.mapped.items()}
    fill, elapsed = phase_inputs(df)
    fb, eb = _bucket(fill, PHASE_BOUNDS), _bucket(elapsed, elapsed_bounds)
    for kind, b in (("fill", fb), ("elapsed", eb)):
        for i, p in enumerate(PHASE_NAMES):
            cols[f"{kind}_{p}"] = np.where(np.isfinite(b), (b == i).astype(np.float64), np.nan)
    if m0 is not None:
        cols["m0_logit"] = np.asarray(m0, dtype=np.float64)
        for kind in ("fill", "elapsed"):
            for p in PHASE_NAMES:
                cols[f"m0_logit_x_{kind}_{p}"] = cols["m0_logit"] * cols[f"{kind}_{p}"]
    out = pd.DataFrame(cols, index=df.index)
    for c in ("won_p1", "weight", "video_id", "game_idx", MODE_COLUMN):
        out[c] = df[c].to_numpy()
    if store:
        out[ROW_POSITION] = df[ROW_POSITION].to_numpy()
        out.attrs[MATRIX_ATTR] = SharedMatrix(store.values, store.columns, dict(cov.mapped))
    return out


def _m0_values(df: pd.DataFrame, fold: int, column: str | None, role: str) -> Optional[np.ndarray]:
    """fold専用の内側OOF/train・外側OOF/testだけを受け付ける。"""
    if column is None:
        return None
    if not (df["m0_outer_fold"] == fold).all() or not (df["m0_role"] == role).all():
        raise ValueError("M0の外側foldまたはOOF役割が不一致")
    from src.b3_m0_inputs import fold_maps
    _, folds = fold_maps()
    assigned = df.video_id.map(folds)
    if assigned.isna().any() or not ((assigned == fold) == (role == "outer_test")).all():
        raise ValueError("M0の事前登録foldとR_cvの分離が不一致")
    values = df[column].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("M0 logitに欠測または非有限値")
    return values


def _m0_attach(df: pd.DataFrame, table: Optional[pd.DataFrame]) -> pd.DataFrame:
    """行順に依存せずrow_keyで結合する。共有特徴行列の属性を保持する。"""
    if table is None:
        return df
    if table.index.duplicated().any() or df.row_key.duplicated().any():
        raise ValueError("M0 row_keyが重複")
    if not df.row_key.isin(table.index).all():
        raise ValueError("M0成果物に必要なrow_keyが無い")
    out = df.copy()
    for name in ("m0_logit", "m0_outer_fold", "m0_role"):
        out[name] = table.loc[df.row_key, name].to_numpy()
    return out


def rcv_trainer(m0_fn=None, m0_column: Optional[str] = None, m0_oof_dir: Optional[Path] = None):
    """R_cv の学習手順 (b3_eval_cv.Trainer 互換)。

    m0_columnはfold専用OOF列、m0_oof_dirはb3_m0_per_fold成果物。既定は従来の近似。
    m0_fnの任意コールバックは隔離を確認できないため引き続き拒否する。
    M0は静止G_feと局面交互作用だけへ追加し、撃ち合いS1'の列は変えない。
    """
    if m0_fn is not None:
        raise NotImplementedError("任意m0_fnは不可。fold専用のm0_columnまたはm0_oof_dirを使用してください")

    if m0_column is not None and m0_oof_dir is not None:
        raise ValueError("M0列指定と成果物ディレクトリ指定は排他")

    def train(df: pd.DataFrame, ctx):
        table = _load_m0_table(m0_oof_dir, ctx.outer_fold)
        column = "m0_logit" if table is not None else m0_column
        df = _m0_attach(df, table)
        cov = coverage(available_columns(df), with_m0=column is not None)
        fill, elapsed = phase_inputs(df)
        bounds = np.nanquantile(elapsed, ELAPSED_QUANTILES)          # train 側だけで推定 (fold 内)
        m0_tr = _m0_values(df, ctx.outer_fold, column, "inner_train_oof")
        rdf = r_design(df, bounds, cov, m0_tr)
        cols = [*cov.mapped, *cov.derived]
        # static は R1s、exchange は S1' 設定。fit_two_state は 1 設定なので 2 回に分けて合成する。
        static_cols = cols + (list(M0_COLUMNS) if column is not None else [])
        s_model = fit_two_state(rdf[rdf[MODE_COLUMN] == 0], static_cols, R1S_CONFIG, ctx.seed, monotonic=False)
        x_model = fit_two_state(rdf[rdf[MODE_COLUMN] == 1], cols, S1P_CONFIG, ctx.seed, monotonic=False)

        def predict(test: pd.DataFrame) -> np.ndarray:
            test = _m0_attach(test, table)
            m0_te = _m0_values(test, ctx.outer_fold, column, "outer_test")
            rt = r_design(test, bounds, cov, m0_te)
            mode = rt[MODE_COLUMN].to_numpy(dtype=np.float64)
            p = np.full(len(rt), np.nan)
            for flag, mdl in ((0.0, s_model.static), (1.0, x_model.exchange)):
                sel = mode == flag
                if sel.any():
                    if mdl is None:
                        raise ValueError(f"mode={int(flag)} の行があるが学習行が無く評価器が作れていない")
                    sub = rt[sel]
                    from src.b3_train_model import predict_hgb, to_matrix
                    p[sel] = predict_hgb(mdl.model, to_matrix(sub, mdl.columns))
            return p
        info = {"recipe": "R_cv", "elapsed_bounds": [float(b) for b in bounds], "n_train_rows": int(len(df)),
                "coverage_unmapped": cov.unmapped, "coverage_mapped_n": len(cov.mapped), "notes": cov.notes,
                "m0_column": column, "m0_scope": "static_only" if column is not None else "off"}
        return predict, info
    return train


def _load_m0_table(directory: Optional[Path], fold: int) -> Optional[pd.DataFrame]:
    """完成した通常実行の成果物だけを段1へ接続する。"""
    if directory is None:
        return None
    import json
    from src.b3_m0_inputs import FOLD_SHA256
    from src.b3_source import sha256_file
    directory = Path(directory)
    manifest = json.loads((directory / f"fold{fold}.json").read_text(encoding="utf-8"))
    path = directory / f"fold{fold}.parquet"
    if manifest["smoke"] or manifest["fold_sha256"] != FOLD_SHA256:
        raise ValueError("M0が試走成果物またはfold正本不一致")
    if sha256_file(path) != manifest["logits_sha256"]:
        raise ValueError("M0 logit成果物ハッシュ不一致")
    return pd.read_parquet(path).set_index("row_key")


def validate_m0_sources(directory: Optional[Path], paths: Mapping[str, Path],
                        folds: Optional[Sequence[int]] = None) -> None:
    """段1の行ファイルとM0学習入力の一致を、学習開始前に確認する。"""
    if directory is None:
        return
    import json
    from src.b3_source import sha256_file
    hashes = {video: sha256_file(path) for video, path in paths.items()}
    for fold in (range(5) if folds is None else folds):
        manifest = json.loads((Path(directory) / f"fold{fold}.json").read_text(encoding="utf-8"))
        if {video: data["rows_sha256"] for video, data in manifest["sources"].items()} != hashes:
            raise ValueError("段1の学習行とM0の学習行ハッシュが不一致")
        _load_m0_table(directory, fold)
