"""半教師あり学習モジュール (Phase I).

ユーザー指摘:
    「置いた後一瞬誤認、少し待つと正しい色に固定されるパターンがほとんど」
    Score / Next / ChainEvent も同様に、後続の盤面状態と整合する値を
    自己整合性で擬似ラベル化できる。

本モジュールは pipeline 結果から擬似ラベルを抽出し、ディスクに永続化、
さらに既存 model (CNN / score template / next CNN) を fine-tune する
基盤を提供する。

主要コンポーネント:
    PseudoLabelSample: 擬似ラベル 1 件のデータクラス
    CrossValidator (ABC): 自己整合性による擬似ラベル抽出 base
    LabelStore: ディスク永続化 (JSONL / numpy 画像)
    OnlineFineTuner (ABC): model fine-tune base
    ScoreValidator: score OCR の単調性 / 連鎖整合性チェック
    NextValidator: next 配置 trace チェック
    ChainValidator: chain detect の score delta 整合性チェック

すべて backwards compat: enable_pseudo_label=False (default) で既存挙動維持。
"""
from __future__ import annotations

from src.self_supervised.pseudo_label import PseudoLabelSample
from src.self_supervised.cross_validator import CrossValidator
from src.self_supervised.label_store import LabelStore
from src.self_supervised.online_fine_tuner import OnlineFineTuner
from src.self_supervised.score_validator import ScoreValidator
from src.self_supervised.next_validator import NextValidator
from src.self_supervised.chain_validator import ChainValidator
from src.self_supervised.cell_color_validator import CellColorValidator
from src.self_supervised.cell_color_fine_tuner import CellColorFineTuner

__all__ = [
    "PseudoLabelSample",
    "CrossValidator",
    "LabelStore",
    "OnlineFineTuner",
    "ScoreValidator",
    "NextValidator",
    "ChainValidator",
    "CellColorValidator",
    "CellColorFineTuner",
]
