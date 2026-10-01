# ライブ評価 (通知単位の表示EMA) への切替平滑の移植 (2026-10-01)
- 実装: src/phase_j/live_notification_eval.py (`switch_smoothing` 引数、既定OFF)、live_eval_worker.py (状態の受け渡しを smoothing_state() へ)、
  scripts/run_live_pipeline_20260928.py (`--switch-smoothing` 既定OFF → 評価器生成へ注入)、config/live_defaults.json (`switch_smoothing: true` = 配布ランチャーでON)。
  (a)(b)(c) は exev と同じ SwitchAwareDisplayEMA。通知単位の評価は旧評価器の値を持たない (None) ため (a) は該当せず、(b) 試合境界で前試合の古いEMAから再開しない・(c) 非イベント切替のみ0.5秒ブレンド・確定死亡は即時。
  状態は export_state/restore_state で親子プロセス・journal再実行へ受け渡す (OFFは従来の3要素のまま)。
- テスト: tests/test_live_switch_smoothing.py 6件 (途中書出し→復元が通し実行と完全一致を4分割位置で確認 ほか)。tests/test_live_b1*.py (b10〜b19) 306件通過。
- 保存済み評価入力 5記録 (114,146通知) をライブ評価器で再生 (scripts/verify_live_switch_smoothing.py):
  | | OFF | ON |
  |---|---|---|
  | 全飛び | 349 | 305 |
  | 試合境界の飛び | 34 | 7 |
  | 物理イベントなし・由来切替に帰属 | 34 | 18 |
  | 発火 / 死亡の飛び (正当、保存されるべき) | 112 / 47 | 112 / 48 |
  由来切替 656 回 (両者同じ)。評価値・events は不変、表示のみ。
