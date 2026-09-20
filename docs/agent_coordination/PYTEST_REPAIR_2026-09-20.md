# 2026-09-20 pytest修復と保存済み診断の確認

対象は添付指示の依頼1〜6。Claudeの予測入力観測（g3_native_scope.py、Dのs3系）には触れない。
終了条件は31件の単独失敗修復、同居失敗の走行契約決定、保存済み診断の母数と限界提示、限定コミット。
G4開始、学習、本番採用、動画の認識再走は対象外。

独立検収はFableで設計・製造・型抜け修復・未読補完を実施。最終は「差分は妥当・条件付き了承」。
原JUnitをパケット抜粋で渡したこと等による原本実読の制約が残り、独立合格とはしていない。
親は原JUnit197件と元失敗31/31対応を実読・照合済み。無変更のレビュー反復はここで止める。
使用量はDの`review_usage.json`へ保存。CLI推定費用と実請求は区別する。
Git差分検査は既存未追跡資産`src/projected_set_bounded_shrink_cnn_v2.py:113`の末尾空行1件のみ。
凍結SHAを変えないため整形しなかった。関係する未追跡依存コードを含めて保存し、無関係な差分は残す。

## 初回CPU反例と修復

- 最終CPU統合: 15ファイル197件、失敗0/error0/skip0。開始票にコマンドとコードSHA、
  終了票にJUnit集計と実行前後の版不変を保存。`final_cpu_v1/{START,RESULT}.json`。
  元の単独失敗31件は原JUnitへ31/31対応（`regression_mapping.json`）。
- その後の独立指摘で、元ソースが整数1なのに票がTrueを申告する型すり抜けを修復。
  最終差分の依存契約と受理側3ファイル20件PASS（`contract_final_junit.xml`）。197件と重複するので足さない。
- ランチャー末尾のPID表示だけCRによってexit1だったが、親PID570・pytest子573の起動とSTART保存を
  確認し、重複起動しなかった。本体exit0、RESULT保存、親子の終了まで確認済み。

- 原票: `D:/puyo_analyzer/verify/agent_output_budget/run_79x_rbpx.log`。
  4ファイル26件中9失敗・17成功。描画5件は`enable_pseudo_chain_score_fill`の二重供給で
  load_default呼出し前にTypeError。物差し1件と可視化2件は採用値の転送漏れ。M2監査1件は旧SHA。
- 描画・物差し・可視化・ワーカ引数順の6ファイル91件は修復後PASS。
  `run_y8etp8gm.log`、32.44秒。これは動画品質の再合格ではない。
- 旧SHA関連9ファイル99件はPASS。`run_e94ugnqh.log`、50.82秒。
- 物差しの不足は初期案の2個ではなく3個:
  `enable_ojama_entry_gravity_settle_guard`、`enable_pseudo_chain_score_fill`、
  `enable_gravity_settle_reset_on_exit`。main→collect→serial/parallel→worker→process→load_defaultへ転送。
  spawnの位置引数は`_collect_parallel`末尾へ同順で追加。既存署名の引数は削除しない。
- overlayのpseudoはNoneを未指定とし、明示Falseも尊重。本番OFF・未指定はキーを省略。
  可視化の追加採用値はboolだけOR解決し、数値閾値の明示値優先を維持。

## 旧SHAの契約

旧SHA直書き7本に加え、参照側4本（M2監査、M1重み診断、M1増分診断、M2増分起動）が該当。
旧SHA定数は過去receiptの来歴として残す。現在SHAへ一括置換しない。

保存盤面のM0/M1/M2演算と保存予測のレビューが物理演算で参照する台帳値は
`GHOST_CHAIN_RULE_ENABLED=True`。src配下のproduction_config importを検索した結果、
indicators_v2、chain_detector、inference_board、exchange_virtual_board、
review_causal_outcome、review_signed_physical_projection_v1、projected_state_observation_adapter_v1等は
この定数のみを参照する。学習入力は保存されたdataset/NPZ、モデルと計算codeのSHAで固定する。
認識器を再実行するcollect_flags/RECOGNITION_ADOPTEDの可変値とは契約を分ける。
新たな依存値を使う変更ではこの契約自体と試験を更新する必要がある。
静的抽出は現在の単一リテラル定義を対象にする。AugAssignやglobals操作など、
任意の将来Pythonコードの動的効果を完全に検証するものではない。

`production_dependency_contract.py`は実ファイルの依存値を読み、欠落・型変更・複数代入を拒否する。
現物SHA、旧SHAとの同一性、依存互換を別項目で記録。`unchanged=False`を互換PASSに書き換えない。
旧学習票は既知旧SHA、新規票は元ソースも票に保存し、受理側でSHAと依存値を再計算する。
自己申告の`compatible=True`だけで通す案は独立検収指摘を受けて修正した。
dataset/model/実験codeの照合は維持する。変更された学習codeで古い学習票をそのまま再認定しない。
Formal100凍結manifestは今回変更しない。既存二段ピンは保持し、将来の版更新候補として記録する。

## 同居失敗の扱い

c案を採る。通常のpytest収集から恒久除外せず、同居条件を要求する22ファイルだけ独立プロセスで
走らせ、残りの全テストと合算して判定する。削除・skip追加・門の緩和は行わない。
根拠は`logs/isolate_failfiles_2026-09-20.txt`の29ファイル単独結果。
G3のsys.modules非占有確認は入口の契約であり、他テストのcollectionがその前提を壊す。
既存`_run_all_g3_tests.sh`は末尾文字列だけで成功を判定するため、今回の全体判定には使わない。
各プロセスの終了コードとJUnit件数を保持し、失敗・error・未走行があれば全体PASSにしない。
単独では通ることを、一般的な同居安全性があるという意味には使わない。
対象22本はDの`isolated_partition.json`に固定した。新たな全体再走は行わず、
元の全pytest結果と無変更の単独試験結果を保持する。今回のコード修復は15ファイルのCPU統合で検収する。
この決定だけでGate 4やG3全体の合格を宣言しない。

## 保存済み診断（品質合格ではない）

原票 `D:/puyo_analyzer/verify/pytest_repair_2026-09-20_v1/episode_audit.json`。
5動画全長の上部行食い違い3,659件を受領票で再現。video38全長は508件、他4動画は3,151件。
video38全長にはnpz_detail.jsonlが存在せず、508件の時点別結合は未測定。
人手判定用video38先頭500秒は別run（74件）で、全長の代用にはしない。

他4動画では食い違い5,119 / 1,634,324基準成立セル。
うち上部行3,151件は全件、記録時の状態ラベルがSTABLE。
区間履歴にCHAIN/GRAVITY_SETTLEを含むもの264件、含まないもの2,887件。
このラベルだけでは画面の連鎖終了を保証できない。全消しテロップ専用ラベルも無いので、
「連鎖アニメ中／全消しテロップ／その他」の確定分類は保存項目だけでは不可能。
全消し0件と報告せず未測定とする。既知c80 f20917実画面には2Pの6連鎖表示がある。

episodesは補正前confirmed_boardを記録し、npz_detailは記録時補正後の盤面を記録する。
したがってn_recorded>0だけでは実保存誤りを意味しない。同一runのside/row/col/値/区間で突合した。
statesとwritersは区間内の別々の集計で、混在区間の記録時点における書き手は復元できない。

他4動画の書き手が単一で確定する食い違いは4,121 / 5,119件。
重力適用が1,838件、浮き除去が890件（合計2,728 / 5,119 = 約53.3%）。
複数書き手を含む候補数はそれぞれ2,295件、1,182件で、重複しうるため足さない。
これらは代理指標の食い違いであって、すべて本物の誤りという意味ではない。
縦2セル以上の連続空欠落形は2,285 / 5,119件。

c80既知8セルは同run明細へ8/8結合、すべて縦連続欠落形。
c80内では8 / 1,098食い違い（基準成立441,229セル）、4動画では8 / 5,119。
人手確定した誤り13件中の8件という母数は別に維持する。
8件中5件は重力適用単独、残り3件は複数書き手。後者を浮き除去の単独責任とは断定できない。
実画面は既存切出器で2フレームを読み、Dのframesへ保存して確認した（元動画1280×720）。

根治候補は「連鎖完了前のシミュレーション終盤面を確定・記録しない」接続の是正。
ただし保存票の状態ラベル自体がSTABLEなので、STABLEだけの記録ガードでは既知8件を防げない。
代案は記録前の縦連続空欠落を疑わしい盤面として保留する既定OFFガード。
既知8/8を候補として拾うが、他4動画では2,285セルへ作用しうるため、改善率や誤検出率は未確定。
列を観測色で即復元すると、正当に消えた連鎖中セルまで戻す恐れがある。
今回はガードを追加せず、採否には同時点の物理状態・画面と正常対照の判定が必要とする。
