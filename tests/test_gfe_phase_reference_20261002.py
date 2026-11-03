"""参考比較が指定列だけを使用することを確認する。"""
import numpy as np

from scripts import gfe_phase_reference_20261002 as reference
from scripts import gfe_phase_stage1_20261002 as stage


def test_reference_columns_and_values() -> None:
    """R1はV0と完全一致し、R2はV1の列順だけを変えたもの。"""
    design = np.arange(138).reshape(3, 46) / 138
    p, sign = np.array([.2, .5, .8]), np.array([1, -1, 1])
    phase, missing = np.array([[0, 2], [1, 1], [2, 0]]), np.array([0, 1, 0])
    args = design, p, sign, phase, missing
    np.testing.assert_array_equal(reference.features("R1", *args), stage.features("V0", *args))
    assert reference.columns("R1") == stage.column_names("V0")
    actual = reference.features("R2", *args)
    assert actual.shape[1] == len(reference.columns("R2")) == 60
    np.testing.assert_array_equal(actual[:, :54], stage.features("V3", *args))
    names = stage.column_names("V1")
    order = [names.index(n) for n in reference.columns("R2")]
    np.testing.assert_array_equal(actual, stage.features("V1", *args)[:, order])
