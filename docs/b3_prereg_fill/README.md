# B3事前登録F1〜F14の記入根拠 (2026-10-05)

結論・未記入理由は [事前登録の追記B](../B3_PREREG_2026-10-05.md) に記載した。本ディレクトリは結果閲覧前に固定したメタデータであり、学習・推論・採点の出力ではない。

- F1: `r_config.json` と `r_flags.json`。exev ccc9623のgit blobから設定関数の出力を取得。
- F2: `r_models.sha256.tsv` は実在モデルとmanifest・特徴列のhash。`r_source.sha256.tsv` はccc9623の平滑化・モデル解決・既存採点器のソースhash。モデルは復元していない。
- F4: `training_manifest.json` と `receipts.sha256.tsv`。B2完了票60本の採用attemptを使用し、旧attemptは選ばない。observationsは完了票記載hash、leanは実ファイルhash。
- F5: `c110_c112_comparison.json`。指定2本のleanの既存ラベルのみを照合。試合数・勝者列の不一致は、再編集された同一対戦の可能性を否定しない。
- F6: `outer_folds.tsv` と `training_summary.json`。既存の決定的手順による外側5fold。c110/c112は同じgroup、試合数は両記録の和。
- F8: `players.tsv` と `players_normalized.json`。60タイトルから抽出した表記と正規化キー。追加の別名統合はなし。
- F11: `sealed.sha256.tsv`。封印領域をバイナリとしてhashしただけで、JSON/NPZ/ラベル/成績の解釈はしていない。封印側3本の記録を含む236ファイル。開封承認・開封記録ではない。
- F12: `regression18.json`。回帰18記録の完了票の動画引数を出典とする。評価データ本体・評価ラベルを読まずに元動画IDを確認。
- F13: `environment_pc1.json` と `pip_freeze_pc1.txt` は今回の1台目WSLの値、`pip_freeze_pc2.txt` は提供済みPC2セットアップ記録の写し。ソフトウェアの過去時点の版を保証するものではない。
- `artifact_hashes.tsv`: 本ディレクトリの固定成果物のSHA256（一覧自身は除外）。

`fill_inventory.py` は上記収集に用いた補助スクリプト。Windowsのgit worktree参照をWSLのgitが直接解決できないため、共通gitオブジェクトストアを `--git-dir` で明示し、固定commitのblobを取得する。出力の改行はLFに固定している。

実行に用いたPythonは `<home>/.gemini/antigravity/scratch/puyo_analyzer/venv/bin/python`。`-B` を付けて補助スクリプトを起動した。既存のfold手順と設定出力以外の本番処理は実行しない。再実行は現存ファイルを再採取して出力を更新するため、固定後の再採取は事前登録の改訂として扱う。`artifact_hashes.tsv` は全ファイルが揃った後に別途生成した。

F3・F9・F10・F14は事前登録の追記Bで統括確定。F9の抽出規則・6列の一覧・除外理由は `F9_columns.txt`。F10の72個根拠は第1版対象外。F13の正式ベンチ条件などの残欄は段2・時間門の実行時に記録する。本作業では学習・採点を実施していない。
