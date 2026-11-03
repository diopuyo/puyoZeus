# D2: 再収集で盤面が変わった原因の診断（2026-09-29）

原因はE34bの収集ラッパーによる**認識既定値7個の意図しない無効化**。
`collect_e34b.install_capture()` が `RecognitionPipeline.load_default` を
`configured(**kwargs)` に置き換え、元の関数署名を引き継がなかった。
描画側の `_pipeline_default()` は `inspect.signature()` に引数がなければFalseを返す。
その結果、CLIに `--production-recognition` があってもライブラリ既定Trueの7個がFalseになる。
評価コード・本番設定・既存収集スクリプトは変更していない。

## 原票と設定の全項目照合

全CLI項目（同値・相違・出力先も含む）、ヘッダー、出所、SHA256は
[`logs/d2/inventory.json`](../logs/d2/inventory.json) に保存。
診断時点のHEADは `c5efef9`。E34b収集実装の導入は `dd859c2`。
旧原票には収集時HEAD・CNNハッシュの直接記載がないため、収集時コミットを断定しない。
履歴上の対応実装はE8 `1dcdf20`、E10c `be85fe7`、E14 `b6a543b`。
E8から診断HEADまで `production_config.py` は同一、`recognition_pipeline.py` の差は
診断・予測用dataclassフィールドの追加のみ（`recognition_code_history.diff`）。

旧入力の実収集元は次の通り。E31の補完処理は盤面を再認識していない。

- q / fc / mia: `logs/e8/renders/<source>/on/inputs.jsonl.gz`。
- zenchi: `logs/review_zenchi_part3/on_e10c/inputs.jsonl.gz`。
- review: **`logs/review_zenchi_g41_43_e14/inputs.jsonl.gz`**。
  E34bがコマンドを借りたE27は、旧盤面の実収集元ではない。
- 以上 → E16（死亡観測追記）→ E26（途中盤面追記）→ E31（発火前画像窓追記）。
  原収集とE31の確定盤面・状態は全114,146行で一致。

### 実効認識設定の差

`config.json` は `load_default` の全引数を既定値まで展開したもの。
旧認識設定→E34b新認識設定で、次の7個が **True→False**。

- `enable_landing_observed_color`
- `enable_drift_resync_match_start_guard`
- `enable_drift_resync_hsv_gate`
- `enable_match_start_full_clear`
- `enable_recovery_counter_carryover`
- `enable_cnn_flicker_hsv_fallback`
- `enable_initial_confirm_vote`

別途、保持印取得のため `enable_landing_chain_record_hold` と
`enable_chain_active_record_hold` がFalse→True。その他の実効引数は同値。
保持フラグを維持して署名だけ保存すると旧盤面に戻るため、保持フラグ自体が初回差の原因ではない。

### CLI・モデル・入力・補完経路の差

- 両方とも `advantage_overlay_flags()` の本番14項目を保持。手書きによるCLI採用フラグの欠落はない。
- 新収集には `--exchange-event-live-count`、`--exchange-event-death-guard`、
  `--confirmed-death-hold`、`--death-formula-guard`、`--color-score-safety`、
  `--multi-landing-death`、`--death-pending-ledger`、`--hidden-row-death`、
  `--midchain-completion`、`--prefire-snapshot`、`--hidden-row-belief` が追加。
- 評価モデルは旧q/fc/mia/zenchiのヘッダーで `models/exchange_event_v1`、
  reviewは `models/exchange_event_v2`、新は全て `models/exchange_event_v3`。
  これは撃ち合い評価モデルであり認識CNNではない。新CLI・v3を維持した診断でも旧盤面を再現した。
- 新は `--no-render` を追加し、reviewのデータパネル・CSVを除去。出力先はE34bへ移動。
  保存コマンドの `--worker` は収集器が実行時に再挿入する。
- 盤面CNNは `models/cnn_phase_b_large_v2.pt`、NEXTは `models/cnn_global_best.pt`。
  E3保存のモデル・較正・評価資産6ファイルのSHA256は現在の現物と全て一致。
  CNN採用閾値は両方0.70。認識モデルの差し替えではない。
- 動画パス、1920×1080化、30fps正規化、サンプル間隔0、seek、ウォームアップは同じ。
  q/fc/miaは0～900秒・ウォームアップ0。
  zenchiは表示2580.566～3427.166秒・ウォームアップ30秒、
  reviewは表示2579.066～2777.333秒・ウォームアップ30秒。
- Online HSVは両方とも新規生成（high_conf=0.85、min_samples=50、require_cnn_proba=False）。
  qはframe 14でおじゃま色9のみ較正済み。初期化方法の違いではなく、較正不足時のガードが落ちた。
- 新ヘッダーには `prefire_match_ranges`（範囲・出所・ハッシュ）を追加。
  既存v4/OCR境界から取得し、認識フレームの除外には使用しない。
  新の `prefire_origin_hold` はその場で取得。旧の発火前画像窓はE31で後付け、
  新は `SnapshotCapture` から同じ `capture_window()` を呼ぶ。
  これらは初回の確定盤面差を説明しない。
- 乱数初期化は旧新とも既存 `worker()` のseed=20260926。診断は同じWSL venv、
  OMP/OpenBLAS/MKL各1スレッド、nice=10、最大3プロセスで実施。

## 最初に効いた差分

qの **frame 54、1.800秒** で、確定盤面が初めて分岐する。
`cnn_board` / `raw_cnn_board`（画像読取器のCNN・HSV融合出力）とHSV-only盤面は旧新で完全一致。
両側とも状態名はSTABLEだが、確定盤面だけが旧「20セル / 1セル」から新「両側None」へ分岐した。
20セルは隠し段UNKNOWNを含む。得点・NEXTにも差はない。

旧では開始15秒以内ガードと、較正色が3未満のHSVガードが再同期を抑制する。
直前の **frame 53、1.766667秒** では旧新とも `needs_resync=True`、
不一致は1P 17セル・2P 29セル、連続52回。新だけが
`recognition_pipeline.py:8305` の `sm.reset(keep_match_state=True)` を両側で実行した。
同フレームの戻り値にはreset前のコンテキストが残り、次のframe 54で両側Noneが表面化する。
frame 54の新のdriftは既に0/False、旧は連続53回/Trueのまま。
較正色は `[9]` の1個。reset呼出箇所・時刻は `reset_events.json` に保存。
**観測画像の非決定性ではなく、同じ観測に対する再同期の許否が最初の分岐。**

中間値の全配列は各 `trace.jsonl.gz`、差分初回の旧新配列は `comparisons.json`、
内部カウンタとセル数は `summary.json` に保存。
全5記録の最初の確定盤面差・直前フレームは `first_frames.json`。

### zenchi・reviewの最初の差は着地色

全5記録とも最初の確定盤面差でCNN/HSV観測と状態名は旧新一致。
ただしzenchi/reviewの最初の差は、再同期によるNone化ではなく着地色の相違だった。
zenchiは2550.850秒の1P着地で `(row8,col0)` が旧緑→新青、
`(row9,col0)` が旧緑→新赤。reviewは2549.716667秒の2P着地で
`(row7,col0)` が旧黄→新赤。座標は0始まり。
両方とも `enable_landing_observed_color` がFalseになったことによる分岐。
初期確定投票だけの復元では初回差は変わらない。

原票の処理開始時刻からの経過はq/fc/miaが1.800秒、zenchiが0.300秒、reviewが0.666667秒。
したがって5記録全てを同じ経過時間・同じ最初の機構として扱うことはできない。
根本の「署名欠落で7設定が変わる」問題は共通。

## 0～10秒の切り分け

qは全条件300行。盤面不一致数 / 状態不一致数を旧E31原票に対して数えた。

- 新設定そのまま: **78 / 80**。新E34b原票とは **0 / 0**。
- 初期確定投票だけ旧値へ復元: **78 / 80**。
- 試合開始時全クリアだけ復元: **78 / 80**。
- 初期確定投票＋全クリアを累積復元: **78 / 80**。
- **HSV再同期ガードだけ復元: 0 / 0**。
- **開始直後再同期ガードだけ復元: 0 / 0**。
- 関数署名を保存して7個を旧値へ戻す（保持2個はONのまま）: **0 / 0**。
- 旧認識設定（保持2個もOFF）: **0 / 0**。
- 保持2個だけをOFF（署名欠落は維持）: **78 / 80**。

2つの再同期ガードはOR条件なので、この短区間ではどちらか1つの復元で旧と一致する。
片方だけが唯一の原因とはしない。初回分岐に効く差を特定できたため、他の閾値探索は行わない。

全5記録の処理開始から10秒（各300行）で、新設定はE34b原票と完全一致、
署名維持設定はE31原票と完全一致（確定盤面・状態・得点・NEXT/DNEXT）。
新設定の旧原票に対する盤面不一致はq 78、fc 84、mia 78、zenchi 197、review 198行。
zenchi/reviewはウォームアップ開始からの認識300行を比較しており、表示出力0行でも欠測ではない。

zenchi/reviewも1個ずつ復元して完全一致まで切り分けた（盤面不一致 / 状態不一致）。

- 着地色補正のみ: zenchi **8 / 1**、review **9 / 1**。初回の着地色差は解消。
- そこへHSV再同期ガードを追加: **8 / 1、9 / 1**のまま。
- さらに `enable_recovery_counter_carryover` を追加: **両記録とも0 / 0**。

残差はSTABLE復旧の遅れで、zenchiの最初の残差は2555.050秒の1P `(5,0)`（旧紫、新空）、
reviewは2551.283333秒の1P `(9,3)`（旧空、新赤）。
本番値に戻した復旧カウンタ持ち越しで、両方とも冒頭300行の全比較項目が一致した。

## 非決定性と本番設定

同一設定・独立プロセスで新設定を2回、署名維持設定を2回収集。
各300行のCNN・HSV・確定盤面・状態・得点・NEXTは一致。
gzipの作成時刻を除いた収集原票本文もSHA256が完全一致。
新は `bf41fbb40a1f4d27c0801d5e4ed15959d9d1c678e3fa5098b9578aacb0f16a2f`、
署名維持は `63147a29bbbc80d3873fc071545e77f82b46d35cf2d3be53db8be68984a91062`。
**この短区間で真の非決定性は検出されない。**

現行の本番**認識設定に対応するのは旧入力側**。
`production_config.py` の採用設定と、描画側がその上で読むライブラリ既定値を保つ必要がある。
E34b新入力はCLIの採用項目を持っていても、ラッパーの署名欠落で実効値が非本番へ変わっている。
旧新ともE系列の評価用追加スイッチを含むため、評価器まで含めた完全な本番コマンドとは区別する。

## 再現

既存WSL venvで、worktreeをcwdとして実行する。

```sh
python -B -m scripts._d2_run --jobs q_new=new q_old=old q_hsv=restore:enable_drift_resync_hsv_gate q_start=restore:enable_drift_resync_match_start_guard q_sig=signature q_sig2=signature q_new2=new
python -B -m scripts._d2_inventory --revision c5efef9
python -B -m scripts._d2_compare
python -B -m scripts._d2_evidence
```

ランナーはWSLでdetachし、条件ごとに独立プロセスを順次実行する。
`--jobs <name>@<source>=<variant>` で他の記録も指定可能。
各条件の `complete.json` が完成マーカー。評価コードの挙動修正は本タスクの対象外。

検証は `python -B -m scripts._d2_verify`。全5記録の旧新再現、同設定反復、
旧原票114,146行の盤面・状態保持、単独/累積復元による一致、診断スクリプトの構文・型注釈・
1関数50行制限を確認。結果は `logs/d2/verification.json`。
