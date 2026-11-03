# rt: R1s / D-v2d の統合

対象は `D:/puyo_analyzer/wt_integ_rt`、ブランチは
`claude/realtime-integrate-r1s-dv2d-20261002`。
基点 `8fe251c` に `560f8716ec0d96865ae7d28b869968759aca4bc5` をマージした。
最終判断は統括 Claude に留保する。push は行わない。
作業開始: 2026-10-02 20:03:31 JST。
検証・集計完了: 21:00:33 JST（57分02秒、コミット前）。

## 衝突と接続修正

衝突は `src/exchange_event_tracker.py` の発火時レコード生成1か所。
rtの `_next_exchange_id` による試合越しの連番を保持し、Aの
`history_integrity` をレコード自身へ結び付ける処理を併存させた。
仕様判断を要する衝突はなかった。

- A納品の `docs/patches/rt_live_memory_dv2d_20261002.patch` を適用。
- `SplitExchangeOverlay._observe_landing` で E35 の `observe` と
  `refresh_current_observation` を呼び、NEXT移動・状態・手持ち観測を反映。
- `asset_hashes()` は本番設定の静止専用モデルディレクトリを読み、R1sの
  `manifest.json` と `G_fe.joblib` を予測モデルのハッシュ対象へ追加。
- 実評価子プロセスにも `MemoryProbe('event-evaluation')` を接続。
  親の代理オブジェクトだけでは読めないE35/H1の保持量を測定できる。
- 通常保存・試合境界での封印・ディスク退避のレコード変換を
  `ExchangeEventTracker.record_mapping()` に統一。H1 OFF時の不要なnull項目を
  除外し、D2の動的な得点証拠も保持する。
- 実動画で、基点にもある設置監査ログの接続不具合を検出した。
  `generate` の `pipe._placement_reconcile.log_to(...)` が、AST差替後の
  認識設定辞書へアクセスして停止していた。出力先をbridgeから認識workerへ
  渡し、実認識器の生成後にログを開くよう修正した。
  設置フレーム観測は `RecognitionPipeline.update()` が既に実行するため、
  重複した観測呼び出しは追加していない。
- 公開間引きで `continue` した通知の入力記録に未確定fallbackが残り、
  次の全通知が `前フレームのfallbackが未記録` で失敗する既存不具合を修正。
  通知ごとのfinallyで、計算しなかった旧評価値を捏造せず、fallbackフィールドの
  ない従来形式の入力行として確定する。例外経路でも次通知を止めない。
- 終了時の `PipelineView.next_recolor_guard_counts` 参照も既存の接続漏れ。
  認識終了監査をsummary通知で親へ配送して保存する。H1のsummaryも既存の
  worker監査配送に追加し、supervisor越しの保存を可能にした。

本番フラグの単一情報源はAから取り込んだ `src/production_config.py`。
`build_command` → 描画main → AST差替generate → supervisor → workerの
既存引数配送を使用し、rt固有の採用フラグの複製は作っていない。
D4は `PLACEMENT_RECONCILE_ADOPTED` から認識worker設定へ届く。

## 検証方法

実動画は既存の `video_zenchi_c0BQoMJwwQU.mp4` を読み取り使用。
対象は2579〜2650秒、ウォームアップ1秒。試合開始を含む。
`--source video --realtime --cpu-threads 1 --evaluation-nice 10 --cnn-device cpu`
に認識・ランタイム監査を付け、描画なしの既存live経路を実行する。

比較基準は `8fe251c` のコードを本worktreeの
`logs/rt_integration_20261002/baseline_8fe251c` へ展開したもの。
無修正基準は上記 `AttributeError: 'dict' object has no attribute '_placement_reconcile'`
で停止した（`baseline_unmodified_failure.log`）。そのため数値比較では
両版に同一の設置監査接続修正（`live_bridge.py` / `live_process.py` のみ）を適用する。
この共通修正には、続けて検出した入力記録の通知単位確定と認識終了監査の配送も含む。
**数値の基準は「8fe251c＋共通接続修正」であり、無修正8fe251cの実測値ではない。**
他worktreeのコードは変更していない。認識・既存評価モデルは同じ資産を使用。
UIテンプレート画像は複製せず、既存ディレクトリを読み取り参照するリンクを使用。
ゲーム画像の生成・追加コミットは行っていない。マージ元および基点から継承した
画像は変更していない。

全実行はnice 10、BLAS各1、起動前の空きメモリ6GiB以上を確認。
他担当のWSL検証プロセスと同居しており、専有環境での性能測定ではない。

`scripts/verify_rt_integration_live_20261002.py` は両構成に同じ計装を適用する。
通知時間は実 `NotificationExchangeOverlay.update()` の入口から戻りまで
（評価・表示平滑を含み、計装のJSON書き込みは除く）。RSSは同じ評価子プロセスを
30通知ごとに標本化する。親側の通知処理時間も別途保存する。
この比較はリアルタイム運転であり、負荷による認識欠落フレームは構成間で異なり得る。

長時間保持はAの設計を継承し、通常運転ではE35.auditとH1.audit/gamesを
試合境界で解放する。さらにliveでは両auditを直近2,048件に制限し、
単一試合が長い場合も固定上限を守る。評価ロジックやrecordの証拠は変えない。
保存再生の `retain_audit=True` は全件保持する。

## 再実行と原票

関連pytestの実行器は `scripts/run_rt_integration_checks_20261002.sh`。
動画計装器には環境変数 `RT_VERIFY_OUTPUT` で出力先を指定する。
`PUYO_MEMORY_PROFILE`、`PUYO_MEMORY_TRACE=0`、`PUYO_MEMORY_INTERVAL=10`
で各プロセスの保持量も取得する。

原票は `logs/rt_integration_20261002/` に保存する。
`scripts/summarize_rt_integration_20261002.py` が両構成の監査を
`video_runs/RESULT.json` に集計する。正常完走の動画原票は
`video_runs/baseline/` と `video_runs/integrated/`。

初回の局所回帰は60成功・1失敗。失敗はH1 OFF時のlive封印だけにnull項目が
増える差分で、保存変換の共通化で修正した。

## テスト結果

| 範囲 | 結果 | 時間 |
| --- | --- | --- |
| 既存live一式・関連exchange/E系/D-v2d・A/B統合回帰 | 2,103 passed / 1 skipped / 5 deselected / 0 failed | 1,091.62秒 |
| 監査上限追加後の最終コード: B7/B11/B16・rt統合9ケース | 81 passed / 0 failed | 285.24秒 |
| 実動画で見つけた接続不具合の修正後: bridge/process/B6/B20・rt統合14ケース | 79 passed / 0 failed | 43.46秒 |

広い回帰の実行中に監査上限を追加したため、その変更に関係する分離評価・
ディスク退避・worker封印と新規試験は後者で改めて確認した。件数は重複するので
足し合わせない。skipは既存の `E2_SHORT_ARTIFACTS` 未指定検査。
除外5件はAと同じ `production_off_is_byte_identical` の全長記録再生。
全リポジトリのpytestを実施したという意味ではない。

原票: `pytest.log` / `pytest.exit`、`final_targeted.log` / `final_targeted.exit`。
変更対象の全関数は50行以内で、構文検査と `git diff --check` を通過した。

## 実動画の確認結果

両版とも終了コード0。親側処理エラー・評価側エラー・ワーカー再起動は全て0。
統合版の実認識設定 `recognition_config.json` はD4=True。
実評価workerの監査ではD1、D2、E35″、H1の全フラグがTrue。
最終の30通知周期標本（2649.833秒）までに静止G_feのHistGBCを378回、
仮想着弾G_feのLRを210回呼んでいた。S3のHistGBCは203回で、静止G_feとは区別する。
R1s資産は実ロード時にmanifestのSHA-256検証を通過した。

試合境界とその後の対戦開始を観測した。ただしこの実走のH1は2試合とも
`history_complete=False` だった。開始直後の空STABLE盤面が取れておらず、
完全履歴が観測できた場合の陽性動作を実動画で検証したとは扱わない。
不完全履歴に対する抑止と完全履歴の条件は関連pytestで検証する。

保持量の実測最大はE35監査29件、H1監査179件、H1.gamesは1件。
通常liveの監査上限2,048件は、各6,144行を追加する合成試験で別に検証した。
試合境界100回の試験でも通常1試合分・保存再生100試合分の契約を確認した。

## 遅延・RSSの比較

基準列は上記のとおり **8fe251c＋両版共通の接続修正**。
無修正8fe251cは起動失敗のため有効な比較値を取得できなかった。

| 指標 | 修正基準 | 統合版 |
| --- | ---: | ---: |
| 評価通知数 | 887 | 871 |
| 評価worker 1通知 P50（ms） | 3.197 | 4.446 |
| 評価worker 1通知 P95（ms） | 31.022 | 37.136 |
| 評価worker 1通知 最大（ms） | 285.313 | 627.923 |
| 親側1通知 P50 / P95 / 最大（ms） | 6.959 / 54.747 / 287.531 | 8.228 / 50.090 / 630.946 |
| 評価worker RSS標本最大（MiB、各30標本） | 572.852 | 541.672 |
| 評価worker RSS最初→最後（MiB） | 532.477→572.852 | 533.797→541.672 |
| 動画ソースの欠落フレーム数（warmup込み） | 1,112 | 1,104 |

workerのP50は約1.25ms、P95は約6.11ms増加。最大値は2612.6秒の
627.9msで、平滑なリアルタイム評価を満たしたとは断定しない。
同時刻のE35証明の記録時間は約4µs（キャッシュ命中）なので、この監査だけから
スパイクの主因をE35探索と断定することもできない。

同じ動画区間・CPU設定だが、認識の取りこぼしと入力確認の完了時刻は異なる。
基準は既存検証キャッシュあり、統合版は初回入力確認で、最初の通知はそれぞれ
2585.3秒 / 2587.1667秒。上記は運転時の参考比較であり、同一通知列に対する
評価器単体の因果比較ではない。RSSも短区間の標本値で、長時間リークの否定には使わない。

保存した集計は `docs/verification/rt_r1s_dv2d_20261002.json`。
採用・性能許容の最終判断は統括 Claude に留保する。
