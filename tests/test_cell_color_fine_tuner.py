"""CellColorFineTuner のテスト."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.board import (
    COLOR_BLUE,
    COLOR_EMPTY,
    COLOR_GREEN,
    COLOR_PURPLE,
    COLOR_RED,
    COLOR_YELLOW,
)
from src.self_supervised.cell_color_fine_tuner import (
    MIN_TOTAL_SAMPLES,
    CellColorFineTuner,
)
from src.self_supervised.pseudo_label import (
    COMPONENT_CELL,
    PseudoLabelSample,
)


# torch 必須なので未インストール環境では skip
torch_spec = pytest.importorskip("torch")


def _make_synthetic_patch(color: int, size: int = 16) -> np.ndarray:
    """色ごとに異なる BGR 値を持つ合成 patch (uint8)."""
    rng = np.random.default_rng(seed=color * 13 + 7)
    bgr_table = {
        COLOR_EMPTY: (10, 10, 10),
        COLOR_RED: (40, 40, 220),
        COLOR_BLUE: (220, 80, 40),
        COLOR_GREEN: (60, 200, 60),
        COLOR_YELLOW: (40, 230, 230),
        COLOR_PURPLE: (180, 60, 180),
    }
    base = bgr_table.get(color, (128, 128, 128))
    patch = np.zeros((size, size, 3), dtype=np.float32)
    for c in range(3):
        patch[:, :, c] = base[c]
    # ノイズ追加
    noise = rng.normal(0, 6.0, patch.shape)
    patch = np.clip(patch + noise, 0, 255).astype(np.uint8)
    return patch


def _make_sample(color: int, frame_idx: int = 0) -> PseudoLabelSample:
    """合成 patch + label の 1 件 サンプル."""
    return PseudoLabelSample(
        component=COMPONENT_CELL,
        timestamp=float(frame_idx) * 0.2,
        input_data={
            "patch": _make_synthetic_patch(color),
            "side": "1P",
            "row": 11,
            "col": 0,
        },
        label=int(color),
        confidence=0.9,
        metadata={"frame_idx": int(frame_idx)},
    )


# ============================
# 初期化バリデーション
# ============================


def test_init_invalid_lr():
    """lr <= 0 で ValueError."""
    with pytest.raises(ValueError):
        CellColorFineTuner(lr=0.0)


def test_init_invalid_epochs():
    """epochs <= 0 で ValueError."""
    with pytest.raises(ValueError):
        CellColorFineTuner(epochs=0)


def test_init_invalid_validation_ratio():
    """validation_ratio が範囲外で ValueError."""
    with pytest.raises(ValueError):
        CellColorFineTuner(validation_ratio=1.5)


# ============================
# fine_tune 動作
# ============================


def test_fine_tune_zero_samples():
    """サンプル 0 件 → no-op で metrics 返却."""
    tuner = CellColorFineTuner(
        base_model_path="nonexistent.pt",
        output_path="/tmp/cell_color_finetune_test_zero.pt",
    )
    metrics = tuner.fine_tune([])
    assert metrics["n_samples"] == 0
    assert metrics["n_epochs"] == 0


def test_fine_tune_below_min_samples():
    """MIN_TOTAL_SAMPLES 未満 → no-op."""
    samples = [_make_sample(COLOR_RED, i) for i in range(MIN_TOTAL_SAMPLES - 1)]
    tuner = CellColorFineTuner(
        base_model_path="nonexistent.pt",
        output_path="/tmp/cell_color_finetune_test_below.pt",
    )
    metrics = tuner.fine_tune(samples)
    assert metrics["n_epochs"] == 0


def test_fine_tune_synthetic_dataset(tmp_path: Path):
    """合成 dataset で fine-tune が走り、保存される."""
    # 各色 6 サンプル × 6 色 = 36 件 (MIN_TOTAL_SAMPLES=10 を超える)
    samples: list[PseudoLabelSample] = []
    colors = [
        COLOR_EMPTY, COLOR_RED, COLOR_BLUE, COLOR_GREEN,
        COLOR_YELLOW, COLOR_PURPLE,
    ]
    fi = 0
    for color in colors:
        for _ in range(6):
            samples.append(_make_sample(color, fi))
            fi += 1
    out = tmp_path / "cell_finetuned.pt"
    tuner = CellColorFineTuner(
        base_model_path="nonexistent.pt",  # ベースなしでも CnnPatchClassifier は init
        output_path=out,
        epochs=2, batch_size=8, validation_ratio=0.1,
    )
    metrics = tuner.fine_tune(samples)
    assert metrics["n_samples"] == 36
    assert metrics["n_epochs"] == 2
    assert metrics["n_train"] > 0
    assert metrics["saved_to"] == str(out)
    assert out.exists()


def test_fine_tune_filters_other_components():
    """component != "cell" のサンプルは除外される."""
    samples: list[PseudoLabelSample] = []
    # 8 件 cell (= MIN_TOTAL_SAMPLES 未満)
    for i in range(8):
        samples.append(_make_sample(COLOR_RED, i))
    # 別 component を 5 件入れて合計 13 件にしても、cell は 8 件のまま
    for i in range(5):
        samples.append(PseudoLabelSample(
            component="next",
            timestamp=0.0,
            input_data={"patch_top": _make_synthetic_patch(COLOR_RED)},
            label={"top_color": 1, "bot_color": 2},
            confidence=0.9,
        ))
    tuner = CellColorFineTuner(
        base_model_path="nonexistent.pt",
        output_path="/tmp/cell_color_finetune_test_filt.pt",
    )
    metrics = tuner.fine_tune(samples)
    # cell サンプルだけカウントされ、min 未満で fine_tune skip
    assert metrics["n_samples"] == 8
    assert metrics["n_epochs"] == 0


def test_rollback(tmp_path: Path):
    """fine_tune 後 rollback で重みが復元される."""
    samples: list[PseudoLabelSample] = []
    colors = [
        COLOR_EMPTY, COLOR_RED, COLOR_BLUE, COLOR_GREEN,
        COLOR_YELLOW, COLOR_PURPLE,
    ]
    fi = 0
    for color in colors:
        for _ in range(4):
            samples.append(_make_sample(color, fi))
            fi += 1
    out = tmp_path / "cell_rollback.pt"
    tuner = CellColorFineTuner(
        base_model_path="nonexistent.pt",
        output_path=out, epochs=1, batch_size=8,
    )
    tuner.fine_tune(samples)
    # backup から重みを復元 (rollback 後 _backup_state は None になる)
    assert tuner._backup_state is not None
    tuner.rollback()
    assert tuner._backup_state is None


def test_metrics_have_expected_keys(tmp_path: Path):
    """metrics dict に必須キーが揃っている."""
    samples = [_make_sample(COLOR_RED, i) for i in range(20)]
    out = tmp_path / "cell_keys.pt"
    tuner = CellColorFineTuner(
        base_model_path="nonexistent.pt",
        output_path=out, epochs=1, batch_size=8,
    )
    metrics = tuner.fine_tune(samples)
    expected_keys = {
        "n_samples", "n_train", "n_val", "n_epochs",
        "loss_before", "loss_after",
        "accuracy_before", "accuracy_after",
        "samples_per_label", "saved_to",
    }
    assert expected_keys.issubset(set(metrics.keys()))


def test_external_cnn_injection(tmp_path: Path):
    """外部から CnnPatchClassifier を渡すと、それが使われる."""
    from src.patch_classifier import CnnPatchClassifier
    cnn = CnnPatchClassifier()
    samples = [_make_sample(COLOR_RED, i) for i in range(20)]
    out = tmp_path / "cell_external.pt"
    tuner = CellColorFineTuner(
        base_model_path="nonexistent.pt",
        output_path=out, epochs=1, batch_size=8, cnn=cnn,
    )
    metrics = tuner.fine_tune(samples)
    assert metrics["n_epochs"] == 1
    # 注入した cnn が tuner 内で使われていることを確認
    assert tuner._cnn is cnn


# ============================
# augment=True 統合テスト
# ============================


def test_augment_default_off_for_backward_compat():
    """augment 既定値は False (後方互換)."""
    tuner = CellColorFineTuner(
        base_model_path="nonexistent.pt",
        output_path="/tmp/cell_augment_default.pt",
    )
    assert tuner._augment is False


def test_fine_tune_with_augment_runs(tmp_path: Path):
    """augment=True でも fine_tune が完走し、metrics が返る."""
    samples: list[PseudoLabelSample] = []
    colors = [
        COLOR_EMPTY, COLOR_RED, COLOR_BLUE, COLOR_GREEN,
        COLOR_YELLOW, COLOR_PURPLE,
    ]
    fi = 0
    for color in colors:
        for _ in range(6):
            samples.append(_make_sample(color, fi))
            fi += 1
    out = tmp_path / "cell_aug.pt"
    tuner = CellColorFineTuner(
        base_model_path="nonexistent.pt",
        output_path=out,
        epochs=2, batch_size=8, validation_ratio=0.1,
        augment=True,
    )
    metrics = tuner.fine_tune(samples)
    assert metrics["n_samples"] == 36
    assert metrics["n_epochs"] == 2
    assert out.exists()


def test_augment_batch_label_consistency():
    """_augment_batch が patch と label を同じ permutation で変換する."""
    from src.cell_augment import REPRESENTATIVE_HUE
    from src.patch_classifier import (
        CLASS_INDEX_TO_COLOR,
        COLOR_TO_CLASS_INDEX,
    )
    # RED の class index を入力に
    rng = np.random.default_rng(seed=0)
    bx = [_make_synthetic_patch(COLOR_RED) for _ in range(4)]
    by = [COLOR_TO_CLASS_INDEX[COLOR_RED]] * 4
    new_bx, new_by = CellColorFineTuner._augment_batch(bx, by, rng)
    assert len(new_bx) == 4
    assert len(new_by) == 4
    # 同一 batch には同一 permutation が適用される
    assert len(set(new_by)) == 1
    new_color = CLASS_INDEX_TO_COLOR[int(new_by[0])]
    assert new_color in (
        COLOR_RED, COLOR_BLUE, COLOR_GREEN, COLOR_YELLOW,
    )
