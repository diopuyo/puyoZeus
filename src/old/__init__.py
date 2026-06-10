"""過去の指標系モジュールのアーカイブ置き場。

ぷよぷよ有利不利判定の「指標 (indicator) / scoring (win 予測) 系」を
2026-06-10 に指標再構築の準備として隔離したもの。

含まれるもの (認識系コアからは独立):
- indicators.py            : 45 指標計算 (IndicatorCalculator など)
- scorer.py                : 指標 → 有利不利スコア (Scorer / PhaseAwareScorer)
- model.py                 : 指標ベクトル → 勝率 MLP
- win_predictor.py         : 盤面状態特徴 → 勝率 MLP
- state_features.py        : 状態特徴エンコード
- form_templates.py        : GTR/サブマリン等の形テンプレート
- timeline_analyzer.py     : 時系列指標分析
- timeseries_indicator_wrapper.py : 時系列 indicator wrapper
- rotation_tracker.py      : indicators 定数依存の回転追跡

注意: 認識系コア (recognition_pipeline / state_pipeline /
board_state_machine / image_reader / ojama_accounting / scoring) は
本パッケージを一切 import しない (クリーン分離済)。
"""
