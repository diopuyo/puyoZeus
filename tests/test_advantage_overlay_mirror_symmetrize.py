"""_train_model の対称化 (side入れ替えミラー標本) 符号反転バグ修正の検証。

バグ内容 (2026-08-10 発見・アーキ案B-1):
  _train_model は side 対称性のため `X_sym = np.vstack([X, -X])` で
  全列を無条件反転していたが、`ojama_flat_score_diff` (np.abs ベースの
  side非依存な絶対量) まで反転すると「負のフラット度」というあり得ない
  値がミラー標本に混入する。 本テストは:
    1. `_mirror_sign()` を小さい合成列名リストで直接検証
       (不変列は+1、可変列は-1になること)
    2. `_train_model()` を小さい合成 CSV で走らせ、実際に HistGBC.fit へ
       渡される X_sym で不変列が反転されず・可変列が反転されていることを
       確認する (スモーク: 学習自体が例外なく完走することも兼ねて検証)
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
# 1. _mirror_sign() 単体テスト (小さい合成列名リスト)
# =============================================================================

def test_mirror_sign_invariant_col_is_not_flipped() -> None:
    """SIDE_INVARIANT_COLS 登録列は符号 +1 (そのまま複製)。"""
    cols = ["a_diff", f"{vao.OJAMA_FLAT_COL}_diff", "b_diff"]
    sign = vao._mirror_sign(cols)
    idx = cols.index(f"{vao.OJAMA_FLAT_COL}_diff")
    assert sign[idx] == pytest.approx(1.0)


def test_mirror_sign_variant_cols_are_flipped() -> None:
    """通常の「自−相手」差分列は符号 -1 (反転)。"""
    cols = ["board_color_puyo_total_diff", f"{vao.COLOR_OJAMA_INTERACTION_COL}_diff"]
    sign = vao._mirror_sign(cols)
    assert list(sign) == [-1.0, -1.0]


def test_mirror_sign_matches_side_invariant_cols_exactly() -> None:
    """SIDE_INVARIANT_COLS に無い列は必ず -1、ある列は必ず +1 (漏れ検出)。"""
    cols = ["x_diff", "y_diff", f"{vao.OJAMA_FLAT_COL}_diff", "z_diff"]
    sign = vao._mirror_sign(cols)
    for c, s in zip(cols, sign):
        expect = 1.0 if c in vao.SIDE_INVARIANT_COLS else -1.0
        assert s == pytest.approx(expect), f"{c} の符号が期待と不一致"


# =============================================================================
# 2. _train_model() 統合スモーク (小さい合成 CSV)
# =============================================================================

# _train_model が使う FEATURES のうち、合成 CSV に用意する最小列。
# 交互作用生成に必要な board_color_puyo_total / board_ojama_count /
# ojama_forecast も FEATURES に含まれるため、これらだけで交互作用列も
# 自動的に有効化される。
_SYNTH_FEATURE_COLS: tuple[str, ...] = vao.FEATURES


def _make_synth_csv(tmp_path: Path) -> Path:
    """1P/2P が対になった小さい合成 labeled_win.csv を作る (4 時刻 x 1動画)。

    ojama 差・予告差の組み合わせを変え、 フラット (差ゼロ) な行と
    そうでない行の両方を含める (フラット度が定数に潰れないように)。
    """
    rows: list[dict] = []
    # (t_sec, 1P値, 2P値) のセットを複数用意。 ojama_count/forecast は
    # 意図的に「フラットな行」と「差が大きい行」を混在させる。
    scenarios = [
        # t, color1, color2, ojama1, ojama2, forecast1, forecast2, won1
        (1.0, 0.60, 0.40, 0.10, 0.10, 0.05, 0.05, 1),   # フラット・色ぷよ差で1P勝ち
        (2.0, 0.30, 0.55, 0.05, 0.05, 0.02, 0.02, 0),   # フラット・色ぷよ差で2P勝ち
        (3.0, 0.50, 0.45, 0.40, 0.05, 0.10, 0.02, 0),   # おじゃま差大・2P勝ち
        (4.0, 0.45, 0.50, 0.02, 0.35, 0.01, 0.20, 1),   # おじゃま差大・1P勝ち
    ]
    for t, c1, c2, o1, o2, f1, f2, won1 in scenarios:
        for side, color, ojama, forecast, won in (
            ("1P", c1, o1, f1, won1), ("2P", c2, o2, f2, 1 - won1),
        ):
            row = {
                "video_id": "video_synth", "side": side, "t_sec": t, "won": won,
                "board_color_puyo_total": color,
                "max_column_height": 0.3, "column_bumpiness": 0.1,
                "death_margin": 0.5, "death_margin_neighbor": 0.5,
                "current_max_chain": 0.2, "conn_pair_count": 0.2,
                "conn_triple_count": 0.1, "ojama_net_balance": 0.0,
                "ojama_forecast": forecast, "board_ojama_count": ojama,
                "dig_resistance": 0.4,
            }
            rows.append(row)
    csv_path = tmp_path / "synth_labeled_win.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return csv_path


class _CapturingGBC:
    """HistGradientBoostingClassifier の代わりに fit 引数を記録するだけの二重。

    重い実学習を避けてテストを高速化しつつ、_train_model が実際に
    model.fit へ渡す X_sym / y_sym をそのまま検証できるようにする。
    """

    last_X: np.ndarray | None = None
    last_y: np.ndarray | None = None

    def __init__(self, **_kwargs: object) -> None:
        pass

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_CapturingGBC":
        _CapturingGBC.last_X = np.asarray(X, dtype=float)
        _CapturingGBC.last_y = np.asarray(y, dtype=float)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        n = len(X)
        return np.tile([0.5, 0.5], (n, 1))


def test_train_model_mirror_preserves_invariant_flips_variant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_train_model が実際に HistGBC.fit へ渡す X_sym で:
    - 可変列 (board_color_puyo_total_diff 等) は後半がちょうど符号反転
    - 不変列 (ojama_flat_score_diff) は後半が前半と同一 (符号反転されない)
    であることを end-to-end で確認する (学習自体も例外なく完走することの
    スモークテストを兼ねる)。
    """
    csv_path = _make_synth_csv(tmp_path)
    monkeypatch.setattr(vao, "TRAIN_CSV_PATH", str(csv_path))
    monkeypatch.setattr(
        "sklearn.ensemble.HistGradientBoostingClassifier", _CapturingGBC,
    )

    model = vao._train_model()

    assert model._puyo_uses_interaction is True, "合成CSVは交互作用が有効になる列構成のはず"
    feat_cols = list(model._puyo_feature_cols)
    cols = [f"{c}_diff" for c in feat_cols]
    cols.append(f"{vao.COLOR_OJAMA_INTERACTION_COL}_diff")
    cols.append(f"{vao.OJAMA_FLAT_COL}_diff")

    X_sym = _CapturingGBC.last_X
    assert X_sym is not None, "model.fit が呼ばれていない"
    n_total = X_sym.shape[0]
    assert n_total % 2 == 0
    n = n_total // 2
    first_half = X_sym[:n]
    second_half = X_sym[n:]

    flat_idx = cols.index(f"{vao.OJAMA_FLAT_COL}_diff")
    interaction_idx = cols.index(f"{vao.COLOR_OJAMA_INTERACTION_COL}_diff")
    color_idx = cols.index("board_color_puyo_total_diff")

    # 不変列: フラット度は exp(-x) > 0 なので反転していれば必ず負値になる。
    # 反転されていなければ前半と完全一致 かつ 全て正。
    np.testing.assert_allclose(second_half[:, flat_idx], first_half[:, flat_idx])
    assert np.all(second_half[:, flat_idx] > 0.0), (
        "ojama_flat_score_diff が反転され負値になっている (バグ再発)"
    )

    # 可変列 (真の差分・交互作用) はちょうど符号反転。
    np.testing.assert_allclose(second_half[:, color_idx], -first_half[:, color_idx])
    np.testing.assert_allclose(
        second_half[:, interaction_idx], -first_half[:, interaction_idx],
    )
