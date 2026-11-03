"""C2+EX ref3の列・SHA検査。監査列は推論へ渡さない。"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from ._live_ex_frozen.src.dl1k_columns import AUDIT_COLUMNS, LEARNING_COLUMNS

from ._live_ex_frozen.src.dl1k_retr import COLUMNS as RETR_COLUMNS, SECOND_COLUMNS

from ._live_ex_frozen.src.dl1k_xc import COLUMNS as XC_COLUMNS
from .live_xc_fill_frac import XC_FILL_FRAC_COLUMNS

MODEL_MEMBERS = 3


def validate_model(model: Any) -> None:
    from .live_obs_trial import check_c2_model
    from src.b3_column_spec import build_column_specs, NET_LEDGER_FEATURE_NAMES
    allowed = {column.name for column in build_column_specs(True)}
    allowed.update((*NET_LEDGER_FEATURE_NAMES, *LEARNING_COLUMNS, *RETR_COLUMNS, *SECOND_COLUMNS,
                    *XC_COLUMNS, *XC_FILL_FRAC_COLUMNS))
    if len(model.members) != MODEL_MEMBERS:
        raise ValueError('24hのEXモデルはref3本が必要です')
    check_c2_model(model)
    for evaluator in (model.static, model.exchange):
        columns = tuple(evaluator.columns)
        if tuple(name for name in columns if name in LEARNING_COLUMNS) != LEARNING_COLUMNS:
            raise ValueError('EX学習40列の名前・順序が違います')
        if set(columns) & set(AUDIT_COLUMNS):
            raise ValueError('EX監査列をモデルへ渡せません')
        if set(columns) - allowed:
            raise ValueError('配信の行に存在しないモデル特徴列があります')


def load_model(path: Path) -> Any:
    from src.b3_ensemble import load_ensemble_model
    model = load_ensemble_model(path)
    validate_model(model)
    return model
