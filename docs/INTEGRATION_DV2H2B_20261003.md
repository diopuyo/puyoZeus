# A: D-v2h2b 統合記録

対象: `wt_integ` / `claude/integrate-r1s-dv2d-20261002`、基点 `85bf6bf`。
開始: 2026-10-03 03:21:43 JST。統合元: `f34d3be`。
依頼された構成の登録・配線を行い、最終判断は統括 Claude に留保する。pushしない。

## 履歴と衝突解消

画像除去後の D-v2d コミットは統合側 `570a921`、evalset側 `f9e6576`。
両方の tree は `752b01af57ee01aa8c17fa40b5a8a8457b9758ca` で完全一致する。
この共通内容を二重実装せず、evalset の後続変更と統合側の既存変更を組み合わせた。

競合13ファイル:

- `falsedeath_replay_20261001.py`、`falsedeath_score_20261001.py`、
  `falsedeath_v2_ladder_20261002.py`: v2e〜v2h2b のモードと検証経路を採用。
- `replay_exchange_event_20260926.py`、`visualize_advantage_overlay.py`:
  R1s・表示平滑化を維持し、新フラグを argparse → 実overlayへ追加。
- `exchange_event_cli.py`: 候補上限検査を維持し、新しい依存・排他検査を追加。
- `exchange_event_landing.py`: tracker更新後にresult・時刻を照合器へ渡す。
- `exchange_event_overlay.py`: 既存初期化・E35依存エラー文を維持し、新機能を追加。
- `exchange_event_tracker.py`: recordに開始証拠を保持し、試合辞書破棄後も保存できる従来修正を維持。
- `exchange_history_integrity.py`: H1′の境界証拠を取得してからgames/auditを破棄する。
  公開dead取消と計算証拠保持を維持。retain_auditを含めkeyword-only。
- `exchange_post_counter_bound.py`: 新しい段観測と元予測・鮮度照合を追加し、
  retain_auditと試合境界の監査破棄を維持。early_exit以外はkeyword-only。
- `test_falsedeath_v2.py`、`test_falsedeath_v2d.py`: 既定OFF検査と通常運転の保持契約を維持。
  旧本番互換検査は新採用の8フラグも除去して旧構成を検査する。

保存用のcoverageスクリプト3本は `retain_audit=True` を明示し、全試合の原票を維持する。
仕様判断を必要とする不一致は検出していない。

## production_config の差分

EXCHANGE_EVENT_ADOPTEDへ、採用日2026-10-03・依頼文そのままの共通根拠
`DV2H2B_ADOPTION_REASON` で次の4項目を追加した。

- `--death-history-boundary-start`
- `--post-counter-completion-recent-integrity`
- `--death-integrity-v2h2`
- `--death-freshness-cap`

D-v2dの既存4項目の採用履歴、R1s、認識側D4の登録は維持する。
全8フラグは `enabled_flags('v2h2b')` と集合一致を実描画CLIテストで検査する。

## 検証

初期の統合・厳格化・鮮度回帰: 66 passed、30.04秒。
関連回帰: 1,594 passed / 8 failed / 1 skipped / 5 deselected、143.58秒。
失敗8件はすべて本番フラグ追加に伴う既存テストの期待値未更新で、修正した。
修正後の本番構成・依存契約: 206 passed、12.97秒（失敗8件を含む）。
本番コードに起因する未解決の失敗は0。件数には重複があるため合算しない。
原票: `logs/integration_dv2h2b_20261003/pytest.log`、`production_final.log`。
skipはE2_SHORT_ARTIFACTS未指定、除外5件は既定OFFの全長保存記録再生。
再実行: `bash scripts/run_integration_dv2h2b_checks.sh`。
起動時nice 10、BLAS各1、起動前空きメモリ20,252,528,640 bytes（6 GiB以上）。
後続のB資源監査で、既存の `test_e14b_exact_score.py` が学習初期化を呼び、
pytest自身をnice19へ下げる副作用を発見した。初回のA/B広域回帰も同じ試験を含むため、
全期間nice10だったとは扱わない。この試験では指標関数の不変性だけを検査し、
優先度変更はmonkeypatchで抑止した。本番の学習コードは変更していない。
起動・終了時のnice/BLAS/空きメモリを記録するpytestプラグインを追加し、
最終回帰を `logs/integration_dv2h2b_20261003/final/` へ別保存する。
最終回帰: **1,602 passed / 1 skipped / 5 deselected / 0 failed、190.36秒**。
資源プラグインで開始・終了のnice10、BLAS各1、空きメモリ6GiB以上を確認した。
今回、全リポジトリpytestや18記録の採点再実行は行わない。

ゲーム画像の追加・変更は0。基点から引き継いだ画像はそのまま保持し、
`git ls-files` と基点との差分で増加がないことを検査する。
logs配下の追跡画像は基点155件→統合後155件、集合一致。画像差分0。

A初期検証完了: 2026-10-03 03:36:40 JST、開始から14分57秒（コミット前）。
資源条件を含むA最終検証完了: 2026-10-03 03:59:02 JST、開始から37分19秒。
