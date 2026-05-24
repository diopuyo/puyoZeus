"""cycle 55 学習後 val acc H1 ゲート確認.

val acc >= 99.0% が学習採用の最低条件 (= H1 ゲート、 ユーザー合意済)。
val report JSON が存在しない場合は学習 log から parse、 どちらもなければ skip。
"""
import json
import re
from pathlib import Path

import pytest

CYCLE55_LOG = Path("logs/cycle55_train.log")
CYCLE55_VAL_REPORT = Path("data/verify/cycle55_val_report.json")
MIN_VAL_ACC = 0.990


def _parse_val_acc_from_log() -> float | None:
    if not CYCLE55_LOG.exists():
        return None
    text = CYCLE55_LOG.read_text(encoding="utf-8")
    m = re.search(r"'accuracy_after':\s*([\d.]+)", text)
    return float(m.group(1)) if m else None


def _resolve_val_acc() -> float | None:
    if CYCLE55_VAL_REPORT.exists():
        data = json.loads(CYCLE55_VAL_REPORT.read_text(encoding="utf-8"))
        for key in ("val_accuracy", "accuracy_after"):
            if key in data and data[key] is not None:
                return float(data[key])
    return _parse_val_acc_from_log()


def test_val_acc_above_99_0() -> None:
    """val acc が H1 ゲート (= 99.0%) を超えていること."""
    val_acc = _resolve_val_acc()
    if val_acc is None:
        pytest.skip("val report / log どちらにも val acc が見つからない")
    assert val_acc >= MIN_VAL_ACC, \
        f"val acc {val_acc:.4f} < {MIN_VAL_ACC} (H1 ゲート失敗)"


def test_dramatic_improvement_detector_flag() -> None:
    """劇的改善 detector: val acc が cycle 32d (= 99.41%) を +0.5pt 超えた場合は
    強制 viz レビュー 必須 (= アーキ案 H5 ルール、 cycle 32g 再演防止)。

    このテストは「フラグが立つかどうか」 を report に warn 出力するのみ。
    fail はさせない (= 採用判定そのものは別途 viz 目視で決定)。
    """
    val_acc = _resolve_val_acc()
    if val_acc is None:
        pytest.skip("val acc 取得不可")
    cycle_32d_threshold = 0.9941 + 0.005
    if val_acc > cycle_32d_threshold:
        # 警告として stderr に出すが fail はさせない (= 採用前提 viz 必須を促す)
        import warnings
        warnings.warn(
            f"DRAMATIC IMPROVEMENT DETECTED: val_acc {val_acc:.4f} > "
            f"{cycle_32d_threshold:.4f} = 強制 viz レビュー 必須",
        )
