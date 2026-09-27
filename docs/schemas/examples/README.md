# Phase J DTO canonical examples

- `initial_hidden.json`: HTTP開始前に発行する安全な初期snapshot。
- `live_practical.json`: 両盤面確定後の通常実戦予測。
- `both_lanes.json`: 同一generation/digestの実戦予測と範囲付き最善行動評価。
- `hold_visible.json`: 現generationが不確実な間に直前generationの確定値を保持する状態。
- `integrity_fault.json`: sequence不整合で数値を即時非表示にしたfail-closed状態。
- `physical_prediction.json`: 片側連鎖中に一意な物理解決後盤面を使った実戦予測。
- `terminal_p1.json`: allowlist済み左右結果ロゴで1P勝利を確定した瞬間。
- `result_p2.json`: 2P勝利確定後、通常評価を消して結果表示へ移った状態。

全例を`puyo_overlay_snapshot_v1.schema.json`で検証する。確率和、較正関数、generation、
terminal方向などJSON Schemaを跨ぐ制約はruntime validatorと独立監査でも検証する。
