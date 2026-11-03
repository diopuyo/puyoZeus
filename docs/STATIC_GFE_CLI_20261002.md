# 静止専用G_feのCLI配線（2026-10-02）

作業ブランチ: `claude/gfe-user-phase-20261002`。基点: `bac5df2`。
採否・本番登録は統括担当。`src/production_config.py` と他worktreeは変更していない。

## 実装

- 共通argparseへ `--exchange-event-static-gfe-dir <dir>` を追加（既定 `None`）。描画 `main → generate → ExchangeEventOverlay → ExchangeEventTracker` および保存記録再生CLIへ接続。
- 指定時だけ通常ロード後に `with_static_gfe` を適用。通常静止は候補、仮想着弾は保持した元のモデルを使用。S1/S3は変更しない。
- `exchange_event_flags()` の返す文字列に `--exchange-event-static-gfe-dir models/exchange_event_v1_r1_hgb` を追加すれば、`--production-exchange-event` 経由で到達する。登録内容そのものは統括が決める。
- 推論入口でG_fe全59列（M0由来の列を含む）の有限性を検査。非有限値は `ValueError(missing_input)` とし、trackerの欠測経路へ戻す。従来の特徴生成にあるD欠測補完は維持。
- `landing_models` 保持済みの二重適用を拒否。manifest必須キー欠落を `ValueError` に統一。

## 検証

WSL、nice 10、OMP/OpenBLAS/MKL各1スレッド。起動時・再生完了時の利用可能メモリを確認し、6GiB以上を維持。

関連13ファイル: **453 passed, 1 skipped**（60.91秒）。skipは既存の `E2_SHORT_ARTIFACTS` 未指定テスト。
新規テストは実main/argparse/generateを通し、実overlay/tracker構築後に動画I/O直前で止める。小さな実LR/HistGBCで静止、`weighted_s3`、`weighted_landing`、`_landing_gfe`、キャッシュヒットを確認。将来の本番1行登録もメモリ内で模擬。既定OFFは予測値のバイト一致を確認。

保存記録reviewの先頭90秒、2,701更新（暖機30秒を含む）を再生。表示は **2579.05～2639.05秒、1,801行**。

- 引数なし: 保存本番（段2b baseline）の全表示列・dtype・形状・配列バイトが一致。
- 引数あり: 保存段2b R1sの全表示列・dtype・形状・配列バイトが一致。
- 引数ありのG_fe実呼出し: 通常静止HistGBC **1,533回**、仮想着弾LR **297回**。S3のHistGBC **539回**は従来モデルのまま。
- 描画CLI構築済みoverlayと、独立起動した再生CLIの間でも、あり/なし両方で `display.npz` / `events.jsonl` のファイル全バイトが一致し、診断JSONも一致。
- 動画認識・エンコードは行わず、保存済み認識結果と死亡信号を使用。実描画CLIで構築したoverlayに再生入力を渡す。

再現: `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 nice -n 10 <venv>/bin/python -m scripts.verify_static_gfe_cli_20261002`
結果原票: `logs/static_gfe_cli_20261002/RESULT.json`。再生スクリプトは本番の `exchange_event_flags()` をそのまま利用する。

## rtへの移植計画（読み取り調査のみ）

以下は `<home>/.codex/worktrees/rt/puyo_analyzer` の調査時点の行番号。

1. `scripts/run_live_pipeline_20260928.py:45` の `build_command` は既に `--production-exchange-event` を追加。`:267` で `build_live_generate` に差し替え、`:268` でargvを設定、`:269` で描画mainを実行している。ここに別のモデルロードは不要。
2. `src/exchange_event_cli.py:14` の共通パーサーへ今回の引数追加を移植。描画側 `scripts/visualize_advantage_overlay.py:6460` 付近のgenerate引数、`:9115` の解析から `:9140` のgenerate呼出し、`:7094` の `FileExchangeModels.load` 後の `with_static_gfe` 適用を移植する。
3. `src/phase_j/live_bridge.py:322` の `build_live_generate` はgenerate全体のASTを再利用し、`:359` でコンパイルする。新しい任意引数とモデル初期化はそのまま受け継がれる。別のlive専用引数追加は不要だが、AST変換後の起動テストは必要。
4. rtには段2bモデル境界がまだ無い。`src/exchange_event_evaluator.py:179` のモデルクラスへ `landing_models` / `with_static_gfe` と今回の検証修正を移植し、`src/exchange_event_landing.py:377` の着弾評価を保持した元モデルへ切り替える。CLIだけの移植では不十分。
5. `src/phase_j/live_evaluation.py:61` の `SplitExchangeOverlay.__init__` は受け取ったモデルを親へ渡す構造。分離trackerでも通常静止HistGBC・着弾LRの振り分けとキャッシュを検証する。
6. `scripts/run_live_pipeline_20260928.py:60` で本番フラグを取得し、`:83` の `prediction_model_hash` は現状v3/v1/M0のmanifestだけを対象にする。候補引数がある場合は候補manifest（可能なら実モデルファイルも）を追加し、R1s採用後のDTO資産IDが従来LRと区別できるようにする。

候補モデル資産の配置と本番登録は統括が実施する。rtのコード・資産は本作業では一切変更していない。
