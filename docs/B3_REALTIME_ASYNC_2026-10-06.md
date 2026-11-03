# B3 非同期接続（既定OFF）

基点は `9e8a9b1`。探索手数・幅・列・モデル・採用設定を変更せず、
`--b3-judgment --b3-async` を指定したときだけ非同期化する。
launcherの利用者JSONでも `"b3_judgment": true, "b3_async": true` を指定できる。
`--no-b3-async` または指定省略は従来の同期経路。B3無効で非同期だけの指定は起動エラー。

## 状態と計算の分離

- 評価スレッドは観測を固定して無制限FIFOへ投入し、到着済み結果だけを読む。
  認識からの容量30 FIFOをB3の探索・台帳・特徴・推論の完了待ちで塞がない。
- 状態processは全フレーム・イベント・NEXTを入力順に処理する。
  台帳、発火エピソード、更新契機、heartbeatの時刻は計算を省いた更新でも進む。
  発火起点の連鎖予測も状態に必要なので、このprocessで実行する。
- 計算processは固定snapshotから同期版と同じ `_make_row` と `predict_row` を呼ぶ。
  実行中1件と待機中最新1件だけを保持し、待機要求は新しい更新契機で置換する。
  可変台帳本体・NEXT先読み口はsnapshotに入れず、確定した台帳snapshotを渡す。
  両processはspawnで独立し、状態やLRUキャッシュを共有しない。
- 入力epoch・試合ID・元フレーム・元時刻で結果を照合し、旧入力、旧試合、逆順、
  未来、元時刻から2秒を超えた結果は採用しない。入力切替と境界で待機要求を失効させる。
  表示保持・失敗時0.5秒上限・R由来の死亡証明は毎通知の評価側で継続する。
- 表示は最後に採用した結果を既存EMAへ渡し、`B3 基準 n.nnn秒` を併記する。
  公開DTOの `evaluations.practical.request_id` は `b3-frame-<元frame>`、
  `input_generation` は元frame。古い結果の保持はDTO上もholdにする。
  B3表示時の `calculation_age_ms` は現在の映像時刻と元時刻の差（主laneとtimingで共通）。
  死亡証明の表示にはB3の基準時刻を付けない。

状態process自体が入力速度を下回る場合、全観測FIFOのメモリと待ち時間は増える。
有限容量による認識への背圧へ戻したり、状態入力を黙って捨てたりはしない。
この条件での追従性・メモリ上限・capture drop改善は静かな時の測定対象であり、
今回の実装だけで時間門合格とはしない。
終了時は最大10秒を目安に末尾をdrainし、残る子processは回収する。
`b3_async_status.json` に投入数、末尾集計、強制終了の有無を残す。
強制終了やworkerエラーのあるrunは「全観測処理済み」と扱わない。

## 一致確認

```text
python -m scripts.b3_async_check --root D:/puyo_analyzer/b3_eval_records --video b3_zenchi_s1 --max-frame 36000 --model D:/puyo_analyzer/share/results/pc2/final_v3/b3_final_v3_4e7b4da.joblib --timing-table D:/puyo_analyzer/b3_data/timing_v1/timing_table.json --out D:/puyo_analyzer/b3_data/codex_tasks/b3_realtime_async/parity_600s.json
```

WSLではパスを `/mnt/d/...` に読み替える。同じ同期入力列に対して、
`AsyncB3(drop=False)` の間引きなしモード、続いて `drop=True` を実行する。
間引きなしは試合境界を含む全要求を保存する検証専用モードで、ライブには使わない。
浮動小数点値はfloat64のビット列（NaNを含む）、他の値は型別直列化で行全体をハッシュ化し、
元frameで照合する。予測は単行出力のfloat64ビットを直接比較する。
差分・例外・重複、間引きなしの欠落は失敗終了。探索等の計時は無効。

実測結果は18,001観測、同期3,042更新。間引きなし3,042件すべて特徴・予測ビット一致、
欠落・例外0。間引きありは302件計算・2,740要求破棄、計算した302件すべてビット一致。
両モードとも18,001観測を全件状態更新した。間引き数はスケジュール依存で性能値ではない。
以前の保存学習行照合3,020件は事後の試合範囲選択あり。今回はlive同様にその選択を入れない。
モデルSHA256は `eb270921f164c39b53bac27c121d707cfcd76d3502d3e721e22a21b9882afde8`。
証跡は `D:/puyo_analyzer/b3_data/codex_tasks/b3_realtime_async/`。
Windows配布Pythonでも合成700観測を通し、実モデル予測の同期一致・表示時刻・子process回収を確認。
画像の再認識一致、OS間の確率ビット一致、OBS実描画の試験ではない。

テストは `tests/test_b3*.py`、`test_live_b3_measurements.py`、
`phase_j/test_live_bridge.py`、`phase_j/test_live_publish.py`、`test_live_b7.py`、
`test_packaging_closure.py`、`test_run_live_pipeline_20260928.py` の488件成功。
追加した集計2件も成功し、終了処理と計時の最終調整後に対象25件を再確認した。
失敗・skipなし。全体実行には既存DataFrame構築のPerformanceWarningが12件ある。
`tests.xml / tests.log` が全体実行の証跡、`windows_smoke.json` がWindows接続の証跡。
長時間の性能測定、配布物の組立・置換、pushは行っていない。

## 原票と計時の意味

`b3_config.json` に非同期設定と従来どおりモデル・表のSHA、探索仕様を保存する。
`b3_state_timing.jsonl` は**全観測**、`b3_updates.jsonl` は**計算された更新**の原票。
後者には採用可否 `accepted` と元epoch・試合・frame・時刻も残す。
計時はwallと `thread_time()` によるスレッドCPUを記録し、入れ子を排他的に差し引く。

- `search_ms / search_cpu_ms`: B3の `ideal_fire_search` と発火起点 `origin_candidates`。
  呼出し全体を測るためキャッシュhitの経路も含む。
- `ledger_ms / ledger_cpu_ms`: 得点台帳のフレーム変換・更新・snapshot取得。
- `state_ms / state_cpu_ms`: 上記を除いた状態・更新契機処理。非同期ではsnapshot固定も含む。
- `feature_only_ms / feature_only_cpu_ms`: B3探索を除いた特徴・行組立。
  非同期ではsnapshot復元も含む。既存静的指標内部の処理はこの区分。
- `inference_ms / inference_cpu_ms`: 単行予測。
- `state_wait_ms`: 評価側投入開始から状態processの処理開始まで。
  `compute_wait_ms`: snapshot要求作成から計算開始まで。
  `result_wait_ms`: 計算完了から評価側受取まで。`wait_ms` はこの3つの和。
  同期経路はこれらのIPC待ちが0。OS待ちやGCの専用計測ではない。

`feature_ms` は従来互換の状態＋特徴の総時間。非同期では対応観測の状態と計算を合算する。
`python -m scripts.b3_realtime_report <out>` の `breakdown` で各項目の件数・合計・分位点を得る。
`all_observations` は更新契機がない/計算を省いた観測も含む。`computed` の状態時間と重複するため、
両方を足さない。全観測の状態/台帳/発火起点探索と、計算更新の特徴/探索/推論を分けて読む。
新原票にない項目は件数0とし、旧原票から後付けで内訳を復元しない。

## 静かな時の3条件測定（今回は実施しない）

[既存30分測定コマンド](B3_REALTIME_2026-10-06.md#時間測定の再実行)を共通ベースにする。
他の収集・学習・テスト・ビルドが止まった同じホストで、同一動画SHA・区間300〜2100秒・
warmup60秒・stride・CPU/GPU・スレッド数・認識設定・SSE/OBS条件を固定する。
3本を同時実行せず、以下の順に別ディレクトリへ保存する。

1. `quiet_off`: `--no-b3-judgment --no-b3-async`。B3モデル引数は不要。
2. `quiet_sync`: `--b3-judgment --no-b3-async` と固定のモデル・表を指定。
3. `quiet_async`: 同じモデル・表で `--b3-judgment --b3-async`。

全runのコマンド、設定JSON、資産SHA、開始終了時のCPU負荷・実行processを保存する。
途中で他の重い処理が入ったrunは比較に採用しない。
`metrics.json` の同じ54,000予定フレームを母数にcapture drop件数/率と各ON−OFF差を比較し、
認識件数・取得→SSE・評価待ち・キュー最大・末尾滞留・drainも併記する。
ON2条件では上記内訳、全観測数、更新要求数、計算件数、採用/棄却、状態待ちと元状態の鮮度を見る。
非同期終了の `submitted == finished.inputs`、workerエラーなし、強制終了なしを確認する。

**非同期の設置→SSEに旧reportの「設置以降の最初のSSE」を使わない。**
それは計算前の古い結果を数え得るため、非同期の同項目はreportで未集計とする。
`runtime-audit` のSSE原票にある `evaluations.practical.request_id` の元frameを、
`b3_updates.jsonl` のframeへ結び付け、同じ元frameの最初のSSE受信/送信を使う。
結果の到着前のSSE、破棄要求、未公開結果、死亡証明を別々に数える。
SSE受信原票しかない場合は受信時間と明記し、flush時間の測定とは呼ばない。
探索手数の削減や再学習の判断はこの3条件の測定後に行う。
