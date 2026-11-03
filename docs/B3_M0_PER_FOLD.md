# B3 M0 fold別学習（10/6 決定C）

M0部分を本番レシピで再学習する。R_cv全体には既存の未対応列・仮想着弾LR等の近似が残り、coverageで引き続き明示する。成績の判断は今回行わない。

## 特定した本番資産

- exev = `<home>/.codex/worktrees/exev/puyo_analyzer`。
- 正体: `models/exchange_event_v1/M0/model.pt`、`AdvantageM0CurrentCNNV2`。SHA256 `f2192f695ad40d34f6919dcb31f3cfed0875925ded0c48069f0ab33f09b7b85f` を実ファイルとF2 `docs/b3_prereg_fill/r_models.sha256.tsv`で照合済み。manifest SHA `7954f90679b04fcc892df190c0f155532a9507dd3f8f968df383b16e98d1e5f3`。
- `src/exchange_event_m0.py:FileM0Predictor` → `src/exchange_event_features.py:g_features` が静止G_feへの経路。認識用の `cnn_global_best.pt` / `cnn_phase_b_large_v2.pt` とは別。
- 学習入口: `scripts/train_exchange_event_m0.py` → `scripts/train_advantage_m1_causal_ledger_v1.py:_fit_fold`。wt_b3の両スクリプトとモデル定義はexevとファイルSHA一致。
- 元学習入力: manifest記載の `56dc/puyo_analyzer/logs/r2_grouped_cv_148_20260925/inputs/video_*.npz`、144動画1,035,639行。boards/queues/labels/source_groups/game_keysから、最終勝敗と試合均等重み（試合内1/N）で学習。
- 本体 `<home>/.gemini/antigravity/scratch/puyo_analyzer` に汎用M0 v1/v2学習スクリプトはあるが、F2の `models/exchange_event_v1/M0` は存在しない。出典はexevの最終学習manifestを採用。部品表 `D:/puyo_analyzer/b3_prep_2026-10-05/ASSET_INVENTORY.md` の動画CV・内側M0 OOF・R1s記述も参照。

## 入力と学習契約

入力は `(N,2,13,6)` の位置を保持した盤面（row0込み）、`(N,2,4)` のNEXT軸/子・DNEXT軸/子。rawの0/1..5/9/10を0/1..5/6/7へ、本番関数で変換する。queue不明は0。盤面左右反転はせず、orientation=1では両者の側だけ交換する。モデル自体が色置換不変・側交換反対称。

既存B3 parquetには盤面セル・queueそのものが無い。`av_board_p1/p2`・`game_idx`から観測記録のconfirmed_boardを完全一致で取得し、`lean_row_p1/p2`からqueueを取得する。盤面をlean盤面で代用しない。生成manifestのattempt・観測/leanハッシュ、盤面総量、取得時刻、側を検査する。欠測盤面はエラーで停止し、空盤面補完しない。元B2記録もPC2で参照可能にすること。

AdamW、lr=3e-4、weight_decay=1e-4、batch=512、最大12epoch、patience=3、seed=0。左右交換augmentation、本番の試合均等BCE、train内tuneのLLでepoch選択、切片0の対称Platt slopeを同じtuneだけで学習。評価foldの成績は算出しない。保存済みM0の重み/slopeは初期化に使わず、毎回scratch学習。

外側割当は `outer_folds_rows_v2.tsv` のSHA256 `22a058395eea9a315c27f39d8b7d7cf781d201c24fa74549eb240cf065a59213` を必須検査。正本は60動画・59group。各外側foldではtrainをさらに3分割してG_fe学習用OOFを作り、別に外側trainで学習したM0でtestを予測（通常4モデル/fold、計20モデル）。各モデルのfit/tuneはtrain側のGroupKFold(5)先頭split。全データで作ったOOF列を単純にR_cvへ流用すると、R_cv学習側のM0が外側testを見てしまうため、**fold別の内側OOF列を使用する**。

3動画試走のみ `--smoke` で内側2分割、1groupしか残らないモデルのfit/tuneを試合単位で分離する（試合の左右複製は同一側）。これは試走専用で、通常の段1接続はsmoke成果物を拒否する。

## 実行（PC2、GPU学習は今回未実施）

リポジトリ直下でPC2の `venv_gpu` のpythonを使用する。torch 2.7.1+cu128が対象。PC1は2.5.1+cu121。共通のtorch APIを使い、CUDA不可時にCPUへ自動退避しない。異なるGPU/torch間のビット一致は保証しない。

```bash
python -m scripts.b3_m0_per_fold --rows-dir <B3行ディレクトリ> --b2-root <B2再収集ルート> --out <新規出力ディレクトリ> --device cuda
```

`--folds 0` で1foldだけ実行可（全動画の入力は維持）。出力先は新規必須。`foldK.parquet` は全行キー付きの `m0_logit` / `m0_outer_fold` / `m0_role`、`oof_foldK.parquet` は外側test行だけ。`foldK.json` に学習/較正/予測group、入力/checkpoint/logitハッシュ、GPUメモリ、時間、レシピを記録。`COMPLETE.json` は指定fold群の完了票。

既存の段1コマンドに `--m0-oof-dir <出力ディレクトリ>` を追加する。既定OFF。静止G_feにlogit+局面交互作用7列だけを追加し、撃ち合いS1'には追加しない。OOF列は符号付きlogitをそのまま用い、追加の正規化・全列反転はしない。各orientationの盤面から計算するため、側交換時の符号はモデルの反対称性で決まる。

APIでは `rcv_trainer(m0_column="列名")` も使用可。同じDataFrameにfoldとrole列が必須で、学習側は`inner_train_oof`、予測側は`outer_test`。通常はハッシュ検査付きの `rcv_trainer(m0_oof_dir=Path(...))` を使用する。共有行列でもlogitは別列として保持し、row_keyで照合する。

試走コマンド: `python -m scripts.b3_m0_per_fold --rows-dir <rows_v2> --out <新規出力> --videos video_29 video_30 video_33 --folds 0 --device cuda --smoke --check-rcv`。`--check-rcv` は予測完了と有限値だけ確認し、スコアを出さない。

## PC1試走とPC2の所要見積もり

`D:/puyo_analyzer/b3_data/codex_tasks/m0_smoke3/`：3動画87,850行、外側fold0のtest 27,308行。入力復元16.81秒、M0内側2本+外側1本52.12秒、R_cv接続2.53秒、実行器合計72.19秒（Python起動/importは含まない）。M0は各4epoch実行。成績の表示・保存・判断はしていない。

RTX 4060 Laptop / torch 2.5.1+cu121：torch最大allocated **376,224,768 bytes = 358.80 MiB**、最大reserved **677,380,096 bytes = 646 MiB**。nvidia-smiの途中観測834 MiBはドライバ等込みの参考値で、ピーク測定値ではない。

全60動画は1,922,710行。通常20モデルの学習対象合計は約12N行、fit/tune約80/20%。初回CUDA初期化を除く2モデルの実測6.01/9.74秒から、学習1行を推論3行相当とする概算で5.5〜6万推論相当行/秒。全量では4epoch時約2.54億、12epoch時約7.34億相当行となり、PC1換算のM0計算は約71〜77分/204〜223分。入力復元は試走の行数比例で約6分。PC2の加速を未測定のため見込まず、**PC2の計画枠は約80分（4epoch）〜約4時間（12epoch）**とする。CPU/ストレージ・実際のearly stoppingで変動し、全量R_cvの木学習時間は別。

batch=512固定なので全量でもGPUメモリは概ね同規模（ドライバ等を含め1〜2 GiBを計画枠とする）。ホストRAMは全行配列と内側subsetの複製に比例する。PC2実機でのBlackwell動作・速度は未検証。正本60動画の盤面参照列には欠測なしを確認したが、全記録の復元はPC2実行時に検査する。

R_cvの木学習・予測は試走2.53秒を行数×5foldで線形換算すると約4.6分/seed。CPU差・木構造による変動を含め5〜10分/seed（段1の3seedでは15〜30分）を別枠とする。M0はseed0固定の成果物を各R_cv seedで共有できる。この見積もりには候補モデルの探索・学習・採点は含めない。
