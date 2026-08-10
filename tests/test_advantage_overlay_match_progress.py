"""進行度の文脈列 (match_progress / color_puyo_x_earliness) の検証。

Phase1-1 B-2 (2026-08-10、user承認済み・アーキ設計): 有利不利モデルに
「試合進行度」を示す特徴列が一切無かった穴を埋める。 確定事実
(data/verify/j1_color_lead_clean_noinflight_2026-08-10.txt) より色ぷよ差は
序盤ほど強く効く (序盤79.7% / 中盤65.2%)。

観点:
  1. match_progress の値域が [0,1] に収まること
  2. match_progress が 1P/2P 入替に対して不変であること (side非依存の絶対量)
  3. color_puyo_x_earliness (可変×不変の交互作用) が side 入替で符号反転すること
  4. paired 省略時 (既存呼出元との後方互換) は進行度列をスキップし例外にならないこと
  5. SIDE_INVARIANT_COLS / _mirror_sign が進行度列を正しく分類すること
  6. _train_model エンドツーエンド: 学習データに board_puyo_total があれば
     model._puyo_uses_progress が True になり、 対称化ミラー標本で
     match_progress は反転せず・color_puyo_x_earliness は反転すること
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import scripts.visualize_advantage_overlay as vao  # noqa: E402


# =============================================================================
# 1. 値域 [0,1]
# =============================================================================

def test_match_progress_value_range_via_add_interaction_columns() -> None:
    """_add_interaction_columns が付与する match_progress_diff は [0,1] に収まる。"""
    feat = pd.DataFrame({"board_color_puyo_total_diff": [0.1, -0.2, 0.05]})
    paired = pd.DataFrame({
        "board_puyo_total_1p": [0.0, 0.5, 1.0],
        "board_puyo_total_2p": [0.0, 0.5, 1.0],
    })
    out_feat, cols = vao._add_interaction_columns(
        feat, ["board_color_puyo_total"], paired,
    )
    col = f"{vao.MATCH_PROGRESS_COL}_diff"
    assert col in cols
    assert col in out_feat.columns
    assert out_feat[col].between(0.0, 1.0).all()
    np.testing.assert_allclose(out_feat[col].values, [0.0, 0.5, 1.0])


def test_match_progress_from_totals_clips_out_of_range_inputs() -> None:
    """浮動小数の丸め等で範囲外が来ても clip で [0,1] に収める。"""
    out = vao._match_progress_from_totals(
        np.array([1.05, -0.02]), np.array([1.05, -0.02]),
    )
    assert np.all(out >= 0.0)
    assert np.all(out <= 1.0)


# =============================================================================
# 2. 1P/2P 入替に対する不変性 (side非依存の絶対量)
# =============================================================================

def test_match_progress_invariant_under_side_swap() -> None:
    """(1P+2P)/2 は 1P/2P を入れ替えても同じ値になる。"""
    total_1p = np.array([0.2, 0.9, 0.0])
    total_2p = np.array([0.8, 0.1, 0.0])
    p_orig = vao._match_progress_from_totals(total_1p, total_2p)
    p_swapped = vao._match_progress_from_totals(total_2p, total_1p)
    np.testing.assert_allclose(p_orig, p_swapped)


# =============================================================================
# 3. color_puyo_x_earliness の符号反転 (可変×不変の交互作用)
# =============================================================================

def test_color_puyo_x_earliness_flips_sign_on_side_swap() -> None:
    """色ぷよ差×早さは「符号可変×符号不変」なので side 入替で符号反転する。

    進行度 (match_progress) は不変のまま維持される。
    """
    feat_cols = ["board_color_puyo_total"]
    feat_orig = pd.DataFrame({"board_color_puyo_total_diff": [0.3, -0.1]})
    paired_orig = pd.DataFrame({
        "board_puyo_total_1p": [0.2, 0.6],
        "board_puyo_total_2p": [0.4, 0.6],
    })
    out_orig, _ = vao._add_interaction_columns(feat_orig, feat_cols, paired_orig)

    # 1P/2P 入替: diff の符号反転 + paired の 1p/2p 列も入替
    feat_mirror = pd.DataFrame({"board_color_puyo_total_diff": [-0.3, 0.1]})
    paired_mirror = pd.DataFrame({
        "board_puyo_total_1p": [0.4, 0.6],
        "board_puyo_total_2p": [0.2, 0.6],
    })
    out_mirror, _ = vao._add_interaction_columns(feat_mirror, feat_cols, paired_mirror)

    earliness_col = f"{vao.COLOR_EARLINESS_INTERACTION_COL}_diff"
    progress_col = f"{vao.MATCH_PROGRESS_COL}_diff"
    np.testing.assert_allclose(
        out_mirror[earliness_col].values, -out_orig[earliness_col].values,
    )
    np.testing.assert_allclose(
        out_mirror[progress_col].values, out_orig[progress_col].values,
    )


# =============================================================================
# 4. paired 省略時の後方互換 (列存在ガード)
# =============================================================================

def test_add_interaction_columns_without_paired_skips_progress() -> None:
    """paired 未指定 (既存呼出元) では進行度列を追加せず例外も出さない。"""
    feat = pd.DataFrame({
        "board_color_puyo_total_diff": [0.1],
        "board_ojama_count_diff": [0.0],
        "ojama_forecast_diff": [0.0],
    })
    feat_cols = ["board_color_puyo_total", "board_ojama_count", "ojama_forecast"]
    out_feat, cols = vao._add_interaction_columns(feat, feat_cols)  # paired 省略
    assert f"{vao.MATCH_PROGRESS_COL}_diff" not in cols
    assert f"{vao.COLOR_EARLINESS_INTERACTION_COL}_diff" not in cols
    # 既存のおじゃまフラット交互作用は引き続き有効 (回帰防止)
    assert f"{vao.COLOR_OJAMA_INTERACTION_COL}_diff" in cols
    assert f"{vao.OJAMA_FLAT_COL}_diff" in cols


def test_add_interaction_columns_paired_missing_columns_skips_progress() -> None:
    """paired はあるが board_puyo_total_{1p,2p} が無い場合も安全にスキップする。"""
    feat = pd.DataFrame({"board_color_puyo_total_diff": [0.1]})
    paired_without_progress_cols = pd.DataFrame({"t_sec": [1.0]})
    out_feat, cols = vao._add_interaction_columns(
        feat, ["board_color_puyo_total"], paired_without_progress_cols,
    )
    assert f"{vao.MATCH_PROGRESS_COL}_diff" not in cols


# =============================================================================
# 5. SIDE_INVARIANT_COLS / _mirror_sign の分類
# =============================================================================

def test_mirror_sign_progress_col_invariant_and_earliness_variant() -> None:
    """match_progress は +1 (不変)、color_puyo_x_earliness は -1 (可変)。"""
    cols = [f"{vao.MATCH_PROGRESS_COL}_diff", f"{vao.COLOR_EARLINESS_INTERACTION_COL}_diff"]
    sign = vao._mirror_sign(cols)
    assert sign[0] == pytest.approx(1.0)
    assert sign[1] == pytest.approx(-1.0)


def test_match_progress_registered_in_side_invariant_cols() -> None:
    """登録漏れ防止の直接確認。"""
    assert f"{vao.MATCH_PROGRESS_COL}_diff" in vao.SIDE_INVARIANT_COLS
    assert f"{vao.COLOR_EARLINESS_INTERACTION_COL}_diff" not in vao.SIDE_INVARIANT_COLS


# =============================================================================
# 6. _train_model エンドツーエンド (合成 CSV)
# =============================================================================

_PROGRESS_SYNTH_SCENARIOS: tuple[tuple, ...] = (
    # t, color1, color2, ojama1, ojama2, forecast1, forecast2, bp1, bp2, won1
    (1.0, 0.60, 0.40, 0.10, 0.10, 0.05, 0.05, 0.20, 0.20, 1),  # 序盤・色ぷよ差で1P勝ち
    (2.0, 0.30, 0.55, 0.05, 0.05, 0.02, 0.02, 0.80, 0.80, 0),  # 終盤・色ぷよ差で2P勝ち
    (3.0, 0.50, 0.45, 0.40, 0.05, 0.10, 0.02, 0.50, 0.50, 0),  # 中盤・おじゃま差大
    (4.0, 0.45, 0.50, 0.02, 0.35, 0.01, 0.20, 0.10, 0.10, 1),  # 序盤・おじゃま差大
)


def _make_synth_csv_with_progress(tmp_path: Path) -> Path:
    """board_puyo_total 付きの小さい合成 labeled_win.csv を作る。"""
    rows: list[dict] = []
    for t, c1, c2, o1, o2, f1, f2, bp1, bp2, won1 in _PROGRESS_SYNTH_SCENARIOS:
        for side, color, ojama, forecast, bp, won in (
            ("1P", c1, o1, f1, bp1, won1), ("2P", c2, o2, f2, bp2, 1 - won1),
        ):
            rows.append({
                "video_id": "video_synth", "side": side, "t_sec": t, "won": won,
                "board_color_puyo_total": color,
                "max_column_height": 0.3, "column_bumpiness": 0.1,
                "death_margin": 0.5, "death_margin_neighbor": 0.5,
                "current_max_chain": 0.2, "conn_pair_count": 0.2,
                "conn_triple_count": 0.1, "ojama_net_balance": 0.0,
                "ojama_forecast": forecast, "board_ojama_count": ojama,
                "dig_resistance": 0.4,
                "board_puyo_total": bp,
            })
    csv_path = tmp_path / "synth_labeled_win_progress.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return csv_path


class _CapturingGBCProgress:
    """HistGradientBoostingClassifier の代わりに fit 引数を記録するだけの二重
    (test_advantage_overlay_mirror_symmetrize.py の _CapturingGBC と同じ役割)。"""

    last_X: np.ndarray | None = None
    last_y: np.ndarray | None = None

    def __init__(self, **_kwargs: object) -> None:
        pass

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_CapturingGBCProgress":
        _CapturingGBCProgress.last_X = np.asarray(X, dtype=float)
        _CapturingGBCProgress.last_y = np.asarray(y, dtype=float)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        n = len(X)
        return np.tile([0.5, 0.5], (n, 1))


def test_train_model_uses_progress_columns_when_available(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """board_puyo_total 列がある学習データでは _puyo_uses_progress=True になり、

    対称化ミラー標本で match_progress は反転せず・color_puyo_x_earliness は
    反転することを end-to-end で確認する。
    """
    csv_path = _make_synth_csv_with_progress(tmp_path)
    monkeypatch.setattr(vao, "TRAIN_CSV_PATH", str(csv_path))
    monkeypatch.setattr(
        "sklearn.ensemble.HistGradientBoostingClassifier", _CapturingGBCProgress,
    )

    model = vao._train_model()

    assert model._puyo_uses_progress is True
    feat_cols = list(model._puyo_feature_cols)
    cols = [f"{c}_diff" for c in feat_cols]
    cols.append(f"{vao.COLOR_OJAMA_INTERACTION_COL}_diff")
    cols.append(f"{vao.OJAMA_FLAT_COL}_diff")
    cols.append(f"{vao.MATCH_PROGRESS_COL}_diff")
    cols.append(f"{vao.COLOR_EARLINESS_INTERACTION_COL}_diff")

    X_sym = _CapturingGBCProgress.last_X
    assert X_sym is not None, "model.fit が呼ばれていない"
    n = X_sym.shape[0] // 2
    first_half, second_half = X_sym[:n], X_sym[n:]

    progress_idx = cols.index(f"{vao.MATCH_PROGRESS_COL}_diff")
    earliness_idx = cols.index(f"{vao.COLOR_EARLINESS_INTERACTION_COL}_diff")

    # 不変列: 進行度はミラーで反転しない (常に非負のまま)
    np.testing.assert_allclose(second_half[:, progress_idx], first_half[:, progress_idx])
    assert np.all(second_half[:, progress_idx] >= 0.0)

    # 可変列: 色ぷよ差×早さはミラーでちょうど符号反転
    np.testing.assert_allclose(second_half[:, earliness_idx], -first_half[:, earliness_idx])


def test_train_model_without_progress_column_still_works(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """board_puyo_total 列が無い旧CSVでも例外にならず _puyo_uses_progress=False。

    後方互換の直接確認 (旧 labeled_win.csv を使う既存呼出元が壊れないこと)。
    """
    rows: list[dict] = []
    for t, c1, c2, o1, o2, f1, f2, _bp1, _bp2, won1 in _PROGRESS_SYNTH_SCENARIOS:
        for side, color, ojama, forecast, won in (
            ("1P", c1, o1, f1, won1), ("2P", c2, o2, f2, 1 - won1),
        ):
            rows.append({
                "video_id": "video_synth", "side": side, "t_sec": t, "won": won,
                "board_color_puyo_total": color,
                "max_column_height": 0.3, "column_bumpiness": 0.1,
                "death_margin": 0.5, "death_margin_neighbor": 0.5,
                "current_max_chain": 0.2, "conn_pair_count": 0.2,
                "conn_triple_count": 0.1, "ojama_net_balance": 0.0,
                "ojama_forecast": forecast, "board_ojama_count": ojama,
                "dig_resistance": 0.4,
                # board_puyo_total 列を意図的に含めない (旧CSV相当)
            })
    csv_path = tmp_path / "synth_labeled_win_no_progress.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    monkeypatch.setattr(vao, "TRAIN_CSV_PATH", str(csv_path))
    monkeypatch.setattr(
        "sklearn.ensemble.HistGradientBoostingClassifier", _CapturingGBCProgress,
    )

    model = vao._train_model()

    assert model._puyo_uses_progress is False
