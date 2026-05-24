# 次セッションキックオフプロンプト (cycle 71b → 72)

以下を Claude にそのまま貼り付けて開始してください。

---

puyo_analyzer プロジェクト続行。 直近の引継ぎは
`docs/HANDOFF_2026-05-11_cycle71b.md` に記載。

## 最新状態
- **cycle 71 → 71b 完了**:
  - Phase 1a (= 物理推論主軸化、 `src/placement_inferrer.py` 新規)
  - cell 認識 CNN メイン化 (= `cnn_override_prob` 0.90 → 0.70)
  - cycle 71b: 案 A (連鎖整合性) + 案 B (縦/横幾何判定) で候補絞り込み強化
  - 旧 `_compute_landing_inferred` / `_inject_pseudo_chain_event` 削除
- **テスト 102/102 pass、 広範 1149/1150**
- **ユーザーレビュー (Phase 1a viz)**: 物理推論効いてる、 ただし「**置いた後の位置誤り + 後から修正が目立つ**」 が残課題
- **走行中ジョブ** (= setsid -f で detach、 セッション切替後も継続):
  - v91 cycle 71b の viz: `data/test_unknown/v91_match1_75s_viz_phase1b.mp4`
  - v91 cycle 71b の diag: `data/diagnostics/v91_match1_75s_diag_phase1b.jsonl`

## 即着手タスク

### 1. cycle 71b 完走確認 + AB 数値比較
```bash
wsl -d Ubuntu -- bash -c 'pgrep -af "python.*scripts" 2>&1'
# プロセスがいなくなれば完走、 jsonl 行数 4500+ 確認後:
PYTHONPATH=. ./venv/bin/python -m scripts.analyze_chain_diag \
    --input data/diagnostics/v91_match1_75s_diag_phase1b.jsonl \
    --output data/diagnostics/v91_match1_75s_diag_phase1b_summary.md
```
A=hit 件数が cycle 70 baseline (17 件) / Phase 1a (17 件) と比べてどう変化したか確認。

### 2. cycle 71b viz レビュー依頼
ユーザーに `data/test_unknown/v91_match1_75s_viz_phase1b.mp4` のレビュー依頼。
特に **15s, 21s, 23s, 25-30s** 付近 (= 連鎖発火時間帯) で:
- 着地直後の位置判定が cycle 71b で改善されているか
- 縦/横の取り違いが減ったか
- 連鎖発生候補の選別が改善されたか

## 次の方向 (= AB 結果次第)

### A=hit が十分減ったら
- **案 C 実装** (= 修正速度向上): STABLE→STABLE で連続 N frame の認識ずれ検出 + 即時修正
- **Phase 1b 着手**: 連鎖開始の正確検出 = score エリアの掛け算式 OCR、 終了 = 12 段目 col=2 出現検出
  - 仕様: memory `reference_chain_phase_detection_spec.md` 参照

### A=hit が減っていない場合
- 残課題の真因を viz + diag JSONL の per-frame log で特定
- 案 A/B の閾値・優先度調整 (= chain_count ボーナス 3 → 5 等)
- 別アプローチ (= 落下中の partial 観察、 sub_region 投票等) 検討

## 注意点
- **self-hit による永久ループ**: `pgrep -f X` で bash 自身が match して止まらない事故あり。 `pgrep -f "python.*X"` 等で工夫すること。
- **シェル escape**: `$(...)` を wsl 経由で渡す時は single quote で wsl コマンド全体を囲む。
- **自律運転前提**: 確認なし実行 OK (memory `feedback_autonomous_operation.md`)。 ただし大きい変更前は設計確認を取る。
- **学習データはユーザーレビュー**: Phase 2 (= 新規 CNN) でユーザーに依頼予定。

## メモリ参照
- `project_handoff_2026-05-11.md` — cycle 70 までの引継ぎ
- `reference_chain_phase_detection_spec.md` — Phase 1b 仕様 (新規)
- `feedback_autonomous_operation.md`
- `feedback_chain_phase_physics_only.md`
- `feedback_msys_pipe_escape.md`

詳細は `docs/HANDOFF_2026-05-11_cycle71b.md` 参照。
