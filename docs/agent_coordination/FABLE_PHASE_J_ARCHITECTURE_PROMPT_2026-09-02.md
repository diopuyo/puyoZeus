# Fable依頼: Phase Jリアルタイム配信設計の独立レビュー

## 役割

あなたはPhase Jの独立アーキテクト兼反対査読者です。Codex案へ同意することではなく、
配信中に壊れる境界、要件の穴、既存資産との衝突を先に見つけてください。

## 安全制約

- 読み取り専用。コード、設定、既存成果物、連携台帳を変更しない。
- 動画解析、ダウンロード、学習、GPU処理を起動しない。
- 実行中の `scripts.run_event_reserve_background_v1` を停止、再起動、重複起動しない。
- 2026-09-02 02:04以降を正とし、攻撃差10%補正は不採用、本番設定は未変更として扱う。

## 最初に全文を読む資料

1. `docs/agent_coordination/CURRENT.md`
2. `docs/agent_coordination/DECISIONS.md`
3. `docs/ADVANTAGE_MODEL_LESSONS_2026-09-01.md`
4. `docs/agent_coordination/PLAN.md`
5. `docs/ADVANTAGE_MODEL_DESIGN_AGREEMENT_2026-08-28.md`
6. `src/stream_overlay.py`
7. `tests/test_stream_overlay.py`
8. `tests/test_stream_overlay_obs.py`
9. `src/analyzer.py`
10. `docs/PHASE_J_REALTIME_OVERLAY_SPEC_2026-09-02.md`
11. `docs/PHASE_J_IMPLEMENTATION_PLAN_2026-09-02.md`
12. `docs/PHASE_J_TEST_MATRIX_2026-09-02.md`
13. `docs/PHASE_J_REQUIREMENT_TRACEABILITY_2026-09-02.md`
14. `docs/schemas/puyo_overlay_snapshot_v1.schema.json`
15. `docs/schemas/puyo_overlay_snapshot_v1_semantic_rules.md`
16. `docs/schemas/puyo_overlay_enums_v1.json`
17. `docs/schemas/puyo_overlay_health_v1.schema.json`
18. `docs/schemas/examples/`と`docs/schemas/health_examples/`の全canonical例

必要に応じて `scripts/visualize_advantage_overlay.py`、`src/recognition_pipeline.py`、
`src/event_source_v1.py`、関連テストを追加で参照してください。

## 固定済み要件

- 2026-09-02のuser回答により、初版は解析とOBSを同じPCで動かす。localhost限定とし、
  別配信PC/LAN公開、LAN認証、TLSは初版スコープ外。
- 配信表示は「実戦予測のみ」「最善行動時のみ」「両方」の3モード。既定は実戦予測。
- 実戦予測、最善行動時評価、対戦者実力補正を混ぜず、公開既定は選手情報なしの盤面基準。
- 信頼できる設置ごとに更新。片側連鎖中は物理計算後盤面を使い、操作側の設置で更新。
  両側連鎖中は新しい物理事実がない限り原則保留。
- 認識不能時は最終確定値、経過時間、物理予測である旨を表示。試合前と終了演出中は非表示。
  物理的勝敗確定時に100対0を一度示し、結果表示へ移る。
- 被覆率、stale時間、最長保留を難所も含む分母で監査する。
- 品質段階は軽量、標準、高精度、分析。配信中の黙った段階変更は禁止し、試合境界でのみ変更。
- 古い計算を破棄し最新状態を優先。品質段階別に較正し、入力品質、遅延、探索量を記録。
- 主入力は盤面+next、未解決物理状態、直近出来事。操作速度は勝率入力にしない。
- 認識は30fps、通常の数値表示は最大2Hzとして別clockにし、boundary/hold/fault/terminal/healthは
  2Hz制限を受けず即時publishする。
- 未来評価は現在状態→物理解決後盤面→両者1手→見えるnext→未知ツモ/長手数近似の順に
  独立採否し、高度探索を初期betaのblocking条件にしない。
- Phase J実装前にW36/W37是正後の認識再測定と、リアルタイム死亡確定・境界状態遷移を閉じる。
- OBSブラウザソース対応。既存基盤はHTTP+SSEだが、ロードマップにはWebSocketと記載がある。
- スコアは -100〜+100、`EVEN_THRESHOLD=3.0`。本番採用フラグの正本は
  `src/production_config.py`で、ユーザー承認前に変更しない。

## Codex v0.5の責務分割（反論対象）

1. `CaptureSource`: 直接取得/OBS再取得を共通`CapturedFrame`契約へ変換し、欠落・反復を通知。
2. `ObservationAdapter`: 認識結果と出来事原本を、時点利用可能な入力へ変換。
3. `RealtimeStateReducer`: match/state/更新/hold/hide/terminalを単一状態機械で確定。
4. `PredictionScheduler`: placement/event単位のjobを世代番号付きで投入し、latest-winsで古い結果を破棄。
5. `PredictionEngine`: 実戦予測と最善行動評価を別出力。品質tierを引数で固定。
6. `DisplayProjector`: モデル出力を表示契約へ変換し、stale、予測、保留理由、終局を明示。
7. `OverlayTransport`: 初期snapshot + 単調sequenceの更新配信。OBS再接続後はsnapshotから復元。
8. `TelemetrySink`: 入力時刻、計算開始/終了、破棄理由、キュー深度、被覆、hold、資産版を追記保存。

`src/stream_overlay.py`の`StreamState`を表示配信の薄い層として残し、旧`AnalysisResult`へ
Phase Jの全意味を詰め込まず、版付きの新しい配信DTOを設ける案を第一候補とします。

## 必ず検討する論点

1. SSE維持、WebSocket移行、両対応のどれが要件に最小か。双方向操作を本当に配信経路へ持たせるべきか。
2. 配信DTOの必須項目と版管理。少なくともmatch ID、state sequence、input/event時刻、
   calculation age、表示状態、hold理由、由来、品質tier、モデル/認識資産版、2種評価を検討。
3. 認識スレッドを探索負荷から隔離し、計算キャンセル不能でも古い結果を絶対に表示しない仕組み。
4. 試合境界を越えた結果、同一sequenceの順序逆転、再接続、遅い購読者、キュー飽和、時計ずれへの対処。
5. 更新・保留・非表示・決着の状態遷移と、各遷移の根拠・最大時間・fail-safe。
6. 軽量tierでも意味を変えないための共通最低契約と、tier別予算・較正・推奨判定。
7. OBSを同一PCで使う場合と、別配信PC/LANで使う場合のbind、認証、CORS、キャッシュ、TLS方針。
8. UIで誤解を生まない表示。実戦予測/最善行動、盤面基準/選手補正、LIVE/予測/HOLD、
   最終更新時刻を小さい画面でも区別する方法。
9. オフラインreplayでリアルタイム意味論を再現し、未来参照を検出するテスト戦略。
10. beta、productionの入口/出口条件。W36/W37、死亡演出、c36/c60境界修正、100本超学習との依存関係。
11. v0.5 Schema、22 Enum、相関制約、48 test、12 requirementの間に矛盾・不足がないか。
12. CaptureSourceのsession/profile/cutoff伝播、画像lease、欠落時の連続判定reset、OBS clean feed、
   A/B事前登録閾値が、旧frame・旧job復活と後付け採否を十分に防げるか。

## 出力してほしいもの

ファイルは変更せず、応答本文として次を返してください。

1. P0/P1/P2の指摘（出典を `file:line` で示す）
2. 推奨アーキテクチャと責務境界
3. 配信DTO v1 Schemaへの具体的な修正案
4. 状態遷移表または状態遷移の列挙
5. SSE/WebSocketのADR案
6. latest-winsスケジューラとバックプレッシャーの失敗条件
7. テスト・計測・採用ゲート
8. 既存資産から再利用するもの、切り離すもの
9. ユーザーへ確認しないと決められない要件だけを、優先度順に最大7問
10. Codex案へ反対する点と、代案

確定事実、設計提案、未検証仮説を明確に分けてください。
