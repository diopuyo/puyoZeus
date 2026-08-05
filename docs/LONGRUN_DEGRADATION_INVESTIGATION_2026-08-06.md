# 長時間連続実行の認識劣化 調査計画 (2026-08-06 アーキ確定版)

(全文はアーキ出力より転記 — 要点)

## 機構仮説 (コード読解で特定済み、要検証)
1. **OnlineHsvCalibrator 初回inject後完全凍結** (recognition_pipeline.py:3959 ガードの内側に update() が同居、_online_hsv_injected=True で収集自体が永久停止。コメントの「段階的inject」意図と矛盾)
2. **reset() が較正状態を未クリア** (2835-2904、_online_hsv/_online_hsv_injected/_injected_colors に不触 → 動画内の全後続試合に凍結状態が持続)
3. **drift resync 安全弁の永久停止** (DRIFT_RESYNC_MIN_CALIBRATED_COLORS=3 ガードが同じ未クリア状態に依存 → 較正凍結の瞬間から自己修復も死ぬ = 無防備化が本体)
4. 較正range自体は append=True で広げるのみ (直接害でなく無防備化)
5. load_default は短尺向け緩和 (high_conf 0.85/min 50) → 長尺では「試合1の数十秒で凍結」の逆効果。本番89動画全てがこの経路

## 検証・規模測定・修正 (要約)
- 実験A (今日中): enable_drift_resync_hsv_gate=False で c22 cross-boundary 再走行 → 解消すれば③主因確定 (30分-1h)
- 実験B (Aで未解消時): reset() に較正クリアの診断パッチ (30分-1h)
- 規模Lv0 (並行・追加収集ゼロ): 89動画npzで「盤面進行中に列まるごと長時間空」異常検出
- 規模Lv1: game_idx (実行時間代理) と異常率の相関
- 規模Lv2: 3-5動画の短窓再構築で検出器の精度校正 (動画一時再取得、処理後削除)
- 修正推奨: A (試合毎reset) + B (凍結ガード撤廃・継続更新)。C (gateの試合スコープ化) はA+Bに包含見込み
- **Phase L regen は修正採用まで凍結** (欠陥込みデータで学習する手戻り防止)
- 工数: Step1-7で3-4日

詳細な行番号・対応表・リスク表はアーキ出力 (セッション記録) 参照。
