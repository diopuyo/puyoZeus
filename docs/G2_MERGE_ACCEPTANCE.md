# PR #26：G2限定修復のマージ受入

2026-09-14、ユーザーは「G2限定のマージ準備→マージ→G3検証→本番採用判断」の順序を承認した。今回実施するのはマージまでで、G3実走・学習・本番採用は開始しない。

## 何を統合するか

G2の診断・保存検証に使う修復ソース、回帰試験、必要性台帳、ノウハウをmainへ保存する。既存tracked runtime・production_config・通常入口・依存パッケージ設定は変更しない。これはA40の終了コードを正常化する処置ではなく、認識や統合勝率の本番切替でもない。

新しいsrc3件は`chain_commit_candidate_v1.py`、`chain_prediction_ledger_v1.py`、`event_source_v1.py`。既存認識へ自動接続する変更はない。commit candidateはpolicy未接続なら拒否し、他2件も明示呼出を要する補助APIである。

マージ準備で既存の純粋CPU試験`tests/test_chain_prediction_ledger_v1.py`と`tests/test_event_source_v1.py`を収録へ追加する。2件は人工盤面・一時ファイルで動作し、私有動画/モデルを必要としない。旧707件の台帳とは別の追加回帰である。

## 残制約の受入

ユーザーが受入済みのA40動画末尾到達と修復箇所通過を、G2診断資材の統合根拠とする。一方、以下は未完のまま残し、マージで合格へ読み替えない。

- 原entry1、終了時I/O障害、POSTCOMMIT_CONSUMER_ROWSの破損、未生成4票。
- 全保存監査、旧M1数値採録、統合勝率の較正、複数動画での一般性。
- D保存runnerの実接続・容量監視。保存方針の更新と実装済みを混同しない。
- 私有原票・モデル・凍結snapshotに依存する診断fixtureの完全な可搬化と、一括pytest時の同名module衝突対策。

この受入は、失われた比較票を再生成した、ストレージ障害を修復した、全G2品質条件を無条件PASSにしたという宣言ではない。次の実走・本番採用は別の開始条件で判断する。

## 最終レビューで確認した境界

Opusは新規src3件の明示APIとpolicy拒否、snapshot/hash列挙経路、文書をread-only確認した。Gitの新旧区分とテスト結果は親が別に確認した。

レビュー中の「既存15本がevent_sourceをimportしている」という指摘は、作業場の未追跡ソースとPR/mainを混同していた。親の`git ls-tree`/`git grep`により、指摘の`event_snapshot_export_v1.py`と`run_first30_provisional_pipeline_v1.py`はmainにもPRにも含まれないことを確認した。既存通常経路の自動採用という阻害理由にはしない。

ただし、将来これらのローカルexportツールを使うときは、srcファイル集合がcode hash・build_id・event_idに影響し得る。新しいbuildのIDを過去runと同一視しない。ファイル列挙をコードの自動実行とも混同しない。

旧文書の「Draftのまま/マージしない」は初回提出時点の停止境界だった。今回の明示承認を修復レビュー・必要性説明・保存ルールへ追記した。未完の品質制約は削除しない。

## 検証記録

最終全件比較はmainが6,115 PASS・124 skip、PR候補が6,226 PASS・124 skip。両版失敗0・実exit0、slow指定1件は各対象外。共通6,239件の結果差0、追加の純粋CPU試験111件すべてPASS。pytest表示時間はmain543.61秒、候補565.24秒。skip/対象外をPASSへ換算せず、G2限定のソース統合を受入可とする。

原ログ・JUnit・終了コードは`D:/puyo_analyzer/verify/g2_a40_recovery_2026-09-14_v1/merge_final_base/`と`merge_final_candidate/`、比較票は同rootの`MERGE_FINAL_SUMMARY.json`へ保持。比較元mainは`bf0f0811600c4ca08e78bd89fc7973648b2ac5b1`、候補のruntimeは親`77f6f139234dee1534cc103e4a644970da173185`と同一で、最終差分は回帰2件と文書/ルールのみ。追加試験の単独実行も111 PASS（4.22秒）。

最初の隔離環境では両版同じ5件、その後の未実行分では両版同じ2件が資材不足で失敗した。設定JSON、終了/X印等のテンプレート、既定CNNモデルを同じSHAの検証用コピーで補った。元モデル・原ログは変更せず、資材をPRへ混入させない。モデル存在有無が初期化に影響するため、最終判定には資材を揃えた環境の全件結果を用い、以前の部分PASSの合算だけでは済ませない。

再現環境はPython 3.12の既存venv、数値library各2thread、CUDAをこのCPU試験processで無効化。資材は`models/calibration_video01.json`、`models/cnn_phase_b_large_v2.pt`、`models/ui_templates/*.png`の既存12ファイル。main/候補で同一のコピーを使用し、Junit・ログ・一時ファイルをDの専用rootへ出力する。資材なしのfresh checkoutで同じ全件結果が出るとは主張しない。

GitHubにはCI workflow・必須status check・branch protectionが設定されておらず、外部の未解決レビューコメントも確認時点で0件。チェックが無いことをCI成功とは呼ばない。ローカル試験の結果と独立レビューを明示して判断する。

## 統合方法と停止

最新mainに基づく専用PR branchをsquash mergeする。直前のhead SHAを指定し、別変更へすり替わっていないことを確認する。元checkout・通常index・既存未コミット変更・原動画・検証原票は保持する。

マージ後は実結果とmerge commitを報告し、所有検証処理を終了確認、heartbeatはPAUSEDを維持する。G3以降を自動開始しない。
