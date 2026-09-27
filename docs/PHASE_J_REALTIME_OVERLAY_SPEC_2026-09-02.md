# Phase J リアルタイム配信オーバーレイ仕様

作成日: 2026-09-02 / 確定: 2026-09-04  
状態: 仕様 v1.0（Codex検収・Claude/Fable独立レビューGO、P0/P1残存なし）  
対象: ぷよぷよeスポーツ上級者対戦、OBSブラウザソース、同一PC構成

## 1. 目的

認識済みの現在状態から、配信中に利用可能な情報だけを使って有利不利を評価し、OBSへ安全に表示する。
Phase Jで最も重要な保証は、計算速度や画面装飾ではなく、次の四つである。

1. 前の試合、古い盤面、古いモデル計算を現在値として表示しない。
2. 認識不能、物理状態未解決、計算待ちを、確定した新評価として表示しない。
3. 実戦予測、最善行動時評価、対戦者補正を混ぜない。
4. 難しい区間を分母から除外せず、被覆率、保留時間、非表示時間、遅延を監査できる。

## 2. 初版の範囲

### 2.1 含むもの

- 解析とOBSを同じWindows PCで実行する。
- `127.0.0.1`にだけHTTPサーバを公開する。
- OBSブラウザソースへHTTP+Server-Sent Events（SSE）で最新snapshotを配信する。
- 実戦予測、最善行動時評価、または両方を表示できる契約を持つ。
- 軽量、標準、高精度の配信tierと、録画後専用の分析tierを定義する。
- LIVE、物理予測、HOLD、非表示、決着確定、結果の意味を明示する。
- 認識、予測、表示、配信の遅延と欠落を追記監査する。

### 2.2 初版に含めないもの

- 別配信PC、LAN公開、外部公開、認証、TLS。
- OBSブラウザ画面からの設定変更。
- 配信中の分析tier利用。
- 旧8指標scorerをPhase Jの評価器として使うこと。
- レビュー動画でのみ可能な未来参照、遡及訂正。
- 攻撃差だけによる100%対0%の強制確定。
- 対戦者の試合中行動傾向への適応。

## 3. 既決事項

- 表示モードは「実戦予測のみ」「最善行動時のみ」「両方」。既定は実戦予測。
  意味の違う値を人手重みで混合しない（`docs/agent_coordination/DECISIONS.md`
  「2026-08-28 — user決定: Phase Jでは実戦予測と最善行動時の評価を切替表示可能にする」）。
- 信頼できる設置ごとに更新する。片側連鎖中は連鎖終了後盤面を物理計算し、操作可能側の
  設置に応じて更新する。両側連鎖中は新しい事実がない限り値を動かさない（同
  「2026-08-28 — user決定: 評価を更新・保留・終了する場面の基本方針」）。
- 認識不能時は最後の確定値と経過時間を表示し、物理計算値は予測と明記する（同
  「2026-08-28 — user決定: 評価を更新・保留・終了する場面の基本方針」）。
- 試合前と終了演出中は勝率を表示しない。現行v1で100%対0%を許す根拠は、
  左右の勝敗ロゴを2フレーム間隔で2回同方向に確認した場合だけとする。
  攻撃量、死亡候補、次試合開始からの確定は禁止する（同
  「2026-09-02 — AI安全判断: 攻撃量だけによる0%／100%確定を停止する」、
  「2026-09-02 — AI品質判断: 左右一致の勝敗ロゴは2フレーム間隔・2回一致で検出する」）。
- 不確実時の2秒holdは確定値でなく初期候補である
  （`docs/ADVANTAGE_MODEL_DESIGN_AGREEMENT_2026-08-28.md:246`）。
- 品質tierを配信中の負荷で黙って変更しない。変更は試合境界で行う
  （`docs/agent_coordination/DECISIONS.md`
  「2026-08-28 — user決定: Phase Jでは端末性能に応じた未来予測の品質段階を選択可能にする」）。
- 公開既定は選手情報を使わない盤面基準の実戦予測。対戦者補正は別出力にする（同
  「2026-08-28 — user決定: 盤面の価値・対戦者の実力・最善行動時の評価を分離する」）。
- 操作時間、操作速度、操作途中の列・向きを勝率入力にしない（同
  「2026-08-28 — user決定: 操作速度を勝率入力から外し、連鎖・おじゃまを出来事と物理規則で扱う」）。
- スコアは-100〜+100、絶対値3以下はEVEN。
- 本番採用の単一情報源は`src/production_config.py`。ユーザー承認前に登録しない。

## 4. 実装入口条件

設計、DTO、純粋な状態遷移試験の準備は先行できるが、本番入力への接続は次が揃うまで行わない。

1. reserveキューが完走し、全対象の状態を再確認する。
2. c36/c60で再現した試合境界resetをobserver/resolverで修正し、回帰を通す。
3. 停止9件を保存済みNPZ/sidecarから別名再exportし、独立監査する。
4. W36/W37是正後の物差しで認識を再測定し、99.5%以上を確認する
   （`docs/agent_coordination/DECISIONS.md`
   「2026-09-04 — AI品質判断: W36/W37再基準化をPhase J入口条件PASSとして確定する」）。
5. 未来参照なしの試合境界と勝敗ロゴ検出を独立監査する。下沈み演出検出は候補事実として
   閉じるが、別の採用決定なしに100%対0%の根拠へ追加しない（同
   「【Claude の設計ミス + user 指示 2026-08-25】死亡確定は『12段目設置 + ネクスト1.5秒不動』の簡易検出で一旦進める」、
   「2026-09-02 — AI品質判断: 左右一致の勝敗ロゴは2フレーム間隔・2回一致で検出する」）。
6. 100本超の正式学習、未知選手・未知動画評価、tier別較正を完了する。
7. 攻撃差10%補正は使わない。2026-09-02 02:04時点の安全側モデルを基準に再評価する。

## 5. 全体構成

```text
CaptureSource
  -> RecognitionPipeline
  -> ObservationAdapter
  -> immutable ObservationEnvelope
  -> RealtimeReducer
       -> 即時のHOLD/HIDE/物理予測状態
       -> input_generation発行
  -> PredictionScheduler
       -> practical lane（最優先）
       -> best-action lane（独立、余剰予算）
  -> PredictionEngine worker process
  -> PredictionCommitGate
  -> SingleWriterEventLoop
       -> RealtimeReducer.apply_result()
       -> DisplayProjector
       -> stream_seq採番
       -> SnapshotHubへ原子的publish
  -> HTTP + SSE
  -> OBS Browser Source
```

全段から`TelemetrySink`へ非同期で監査情報を送る。監査書込みの遅延や故障で認識を止めない。
reducerの状態変更からHub格納までは単一writerだけが実行し、commit済み旧結果が境界後に
publishされる隙間を作らない。

STABLE原則との境界を次のように固定する。`confirmed_board`と既存45指標は両者STABLE時だけ更新し、
NON-STABLE中は最後の確定値を凍結する。物理的に一意な解決後盤面は`confirmed_board`へ書き戻さず、
別のimmutable `projected_state`としてfuture modelへ渡す。projected入力を学習・較正していない
モデル、または既存指標をSTABLE相当として流用する経路ではprojected値を出さない。現在状態の
baselineが利用可能ならその値を維持し、candidate都合の新規holdは作らない。現在状態自体が既存の
安全条件を満たさない場合だけholdする。projected値を公開する場合は必ず`physical_prediction`と
provenanceを示す。

## 6. 責務

### 6.1 CaptureSource

動画ファイル専用の`src/frame_reader.py`とは分離し、ライブ入力を時刻付きframeとして供給する。
具体的な入力方式は実装前の小規模測定で決める。OBS出力画面の再キャプチャは候補だが、
compositor待ち、texture copy、GPU→CPU readback、色変換、resize、encode/decodeの実経路と遅延を
測らず既定にしない。

CaptureSourceから先は入力方式へ依存させず、次のimmutable契約で受け渡す。

```text
CapturedFrame
  capture_session_id / source_kind / source_id / capture_profile_id
  progress_signal_kind
  capture_seq / source_frame_seq|null / source_timestamp|null
  captured_monotonic_ms / received_monotonic_ms
  width / height / pixel_format / semantic_frame_digest
  source_dropped_before / buffer_replaced_before / backend_duplicate
  image_ownership / lease_id|null / image
```

- `capture_seq`はcapture session内で単調増加させ、逆順と同一seq異内容を隠さない。backendが報告する
  source欠落と、1件bufferが正常動作として捨てた中間frameを別counterで保持する。
- 認識producerを待たせないlatest-winsの1件bufferとし、置換数を次に渡すframeとtelemetryへ残す。
- 入力解像度は認識直前に既存規約の1920×1080へ正規化する。元解像度と変換条件は資産IDへ含める。
- `semantic_frame_digest`は実際に認識・terminal判定へ渡す正規化済み画素範囲から一度だけ作る。
  digest方式と対象ROIを版付きにし、raw frame全体の毎frame hashをhot pathの必須条件にしない。
- `image_ownership=owned_immutable`ならconsumerへ渡した後に書き換えない。`read_only_lease`ならconsumerが
  releaseするまでbackendがbufferを再利用せず、満杯時は既存bufferを書き換えず新frameを捨てる。
  lease保持者は認識+terminal前処理を行う単一consumerだけとし、画像自体をfan-outまたはprocess境界へ
  渡さない。派生したimmutable観測だけを後段へ渡し、consumerは`try/finally`でlease IDを厳密に1回
  releaseする。早期再利用、二重release、未releaseを黙って許さず、理由付きfaultとcounterへ残す。
- source、backend、crop、pixel format、color space、HDR/SDR、resize profileは版付き
  `capture_profile_id`としてsession開始時に固定し、試合中の黙った切替と自動failoverを禁止する。
  profileは`source_frame_seq`、`source_timestamp`、`backend_duplicate_flag`のうち少なくとも一つを
  信頼できる`progress_signal_kind`として宣言し、対応値を供給できないbackendは起動前に拒否する。
- `capture_session_id`の唯一の発行者はCaptureSourceとし、起動・再接続ごとに未使用IDを発行して再利用しない。
  reducerは受理・記録・照合だけを行い、IDを生成または書き換えない。切断eventが欠落しても、新しい
  capture session IDを初めて受理した時点でgenerationを進め、旧jobを全て無効化する。
- 切断、source欠落、backend duplicateを`capture_status`としてreducerへ通知し、旧画像を新観測として
  再利用しない。同一digestの連続だけでは正当な静止盤面と区別できないためhold理由にしない。
  stallはsource timestamp/sequenceが進まない状態またはbackend duplicate印が、profileで測定固定した
  時間を超えた場合だけ成立させる。source/consumer欠落時はterminal等の連続frame成立回数をリセットする。
- `CaptureStatus`はframeと別eventにし、切断、再接続、profile変更要求、欠落、stallを通知する。
- OBS経路を選ぶ場合はPhase J browser sourceを含まないclean scene、合成前source、clean Virtual Camera、
  またはclean shared textureだけを許す。Program出力しか取得できず自己再帰を除けない場合は不採用とする。
- 直接ゲーム/ウィンドウ取得とOBS再キャプチャは同じcontractのadapterとして比較する。まず5分の同時取得で
  source frame IDまたは認識済みplacement/event列を整列し、ROI画素差と意味結果を比較する。性能は二重取得の
  競合を避け、OBS配信負荷を固定したA→B/B→A各30分の単独runでcapture→認識/表示P95/P99、source drop、
  buffer置換、重複率、CPU/GPU/VRAM/RAM、OBS render/encoder lagを測る。曖昧pairを分母から隠さず、
  整列後のSTABLE盤面・物理event・境界が非劣化なら低遅延側を採り、採用backend単独で長時間soakする。

A/B本測定前に次を版付きmanifestへ固定し、結果確認後に変更しない。

- backendごとの`capture_profile_id`、入力source SHA、HSV較正資産IDを固定する。正規化後ROIの
  画素同値を別監査で証明できない限り、HSV較正資産を異なるbackend/profile間で共用しない。

- 各backendは30分以上、整列可能frame 1,000以上、accepted placement 100以上、formal boundary 10以上を
  満たす。不足時は同一設定の追試を加え、既存runを捨てない。
- STABLE cell精度は各backendで99.5%以上、候補と基準backendの差の片側95%区間下限は-0.10 percentage
  point以上、coverage低下は0.5 percentage point以内とする。
- formal boundaryとterminal方向の不一致は0、未来参照・旧capture session採用・自己再帰も0とする。
- capture→OBS表示はP95 500ms、P99 1秒以内。認識30fps実効率、source drop、buffer置換、OBS render/
  encoder lag、CPU/GPU/VRAM/RAMは全期間を分母にし、難所だけ除外しない。
- 両者合格時はcapture→認識P95が低い方、差が10ms未満ならOBS render/encoder lagの低い方、なお同等なら
  外部依存の少ない方を採る。閾値変更は新しいmanifestと別runでのみ行う。

精度の正解源は基準backendの出力にせず、既存Phase I物差し手順と同じ独立確定盤面、または同手順で
blind作成した追加正解盤面とする。同一source frame/eventへ両backendを対応付けるpaired設計を用い、
未整列・片側欠落もcoverage分母へ残す。cellを独立標本として水増しせず、matchを第一cluster、formal
boundaryを補助clusterとしたseed固定10,000回のpaired cluster bootstrapで差の片側95%区間を求める。
正解asset hash、抽出規則、cluster ID、seed、反復数をA/B manifestへ実行前に記録する。

### 6.2 ObservationAdapter

`PipelineResult`、時点利用可能な出来事、認識・設定資産hashをimmutableな
`ObservationEnvelope`へ変換する。試合境界の採否、hold、表示内容は決めない。

未来情報を型境界で遮断するため、内部契約を次のように固定する。

```text
CausalCutoff
  source_id / capture_profile_id / capture_session_id / capture_seq / semantic_frame_digest
  available_through_frame
  available_through_event_seq
  accepted_boundary_event_id

ObservationEnvelope
  session_id / observation_seq / capture_profile_id / capture_session_id / capture_seq
  semantic_frame_digest / available_frame / available_ms
  event_batch_id / max_available_event_seq / boundary_event_id
  observations / provenance / asset_bundle_id / input_digest

PredictionRequest
  request_id / commit_token / causal_cutoff / input_digest
  practical_or_best_action_input / tier_profile_id / hard_deadline

PredictionResult
  request_id / echoed_commit_token / input_digest / result / runtime_metrics
```

`PredictionRequest`へ公式勝者、事後確定view、cutoffより後の訂正を表現できるfieldを置かない。
訂正・取消は、それ自身が利用可能になった新しいevent sequenceとしてreducerへ渡す。
`input_digest`はcapture profile/session/seq/semantic frame digest、cutoff、採用event ID、盤面、未解決物理状態、
資産IDから作る。CaptureSource再接続で新しいcapture session IDを発行し、旧sessionから遅れて届いた
frame/envelopeは新sessionへ投入せず拒否する。

reducerが受け付けるevent型は、`observation_available`、`formal_boundary`、
`terminal_evidence`、`prediction_completed`、`worker_fault`、`capture_status`、
`config_change_requested`、`timer_fired`に限定する。eventの欠番、逆順、重複、同一ID異内容は
`integrity_fault`へ遷移させる。

再利用候補:

- `src/recognition_pipeline.py`のstate、confirmed/estimated board、next、provenance。
- `src/event_snapshot_adapter_v1.py`のSTABLE盤面、利用可能時刻、出来事由来。
- `src/event_source_v1.py`の単調通番と利用可能時点の契約。

### 6.3 RealtimeReducer

外部I/Oを持たない純粋な状態遷移器とする。単一の巨大enumではなく、次の直積状態を管理する。

- `match_phase`: pre_match / active / terminal / result / intermission / integrity_fault
- `display_state`: hidden / awaiting / live / physical_prediction / hold
- `job_state`: idle / queued / running / result_ready / rejected

`match_phase=intermission`は内部状態を`display_state=hidden`へ射影し、公開DTOを
`visibility=hidden`、`status=waiting`、両lane `unavailable`とする。前試合値を保持しない。

正式な境界通知だけが`match_id`を更新できる。境界受理と同じ原子的更新で旧jobを無効化する。
途中起動時は正式な試合開始と初回の信頼入力が揃うまで非表示とする。次試合も新しい確定入力に
基づく予測がcommitされるまで非表示とし、前試合値を仮表示しない。

### 6.4 PredictionScheduler

実戦予測と最善行動時評価を別laneにする。各laneは実行中最大1件、待機中は最新1件だけを保持する。
新入力時は待機jobを置換し、実行中jobへ協調cancelを通知する。cancel成功は安全条件にしない。
laneごとにhard deadlineと独立workerを持つ。deadlineと短い終了猶予を超えたworkerは破棄して
再生成し、最新の待機jobを新workerへ渡す。連続timeout時は現在tierを黙って変えず、
`runtime.status=degraded`と理由を表示し、次の試合境界でのtier変更を提案する。
Windowsではworkerが入力を受け付ける前にモデル・較正器・必要GPU資産をpre-warmする。
再生成中は`prediction_worker_fault`のholdを継続し、`worker_health=restarting`のlaneから届いた
新resultはcommitしない。pre-warm完了後の新しいworker tokenだけを受付可能にする。
GPUを共有する場合はpractical専用の予約予算を持ち、practicalがqueued/runningの間は
best-actionの新しいGPU区間を開始しない。best-actionは探索節点間でpreemption checkpointを持ち、
practicalのdeadlineを一度でも悪化させた構成を標準tierに採用しない。

### 6.5 PredictionCommitGate

結果を表示状態へ採用する唯一の入口。次のtokenが現在値と完全一致し、deadline内の場合だけ、
同一ロック内で採用する。

```text
session_id
match_id
capture_session_id
capture_seq
causal_cutoff_digest
input_generation
evaluation_kind
tier_profile_id
asset_bundle_id
deadline_monotonic_ns
```

一致しない結果は表示へ触れず、理由付きで破棄する。

`capture_seq`と`causal_cutoff_digest`の比較対象はcapture backendの最新headではなく、現在の
`input_generation`を発行したimmutable入力snapshotとする。後続frameが到着しても、STABLE盤面、
物理事実、terminal/capture安全状態が変わらなければgenerationとtokenを更新せず、進行中結果を
失効させない。意味入力が変わった場合だけ新generationを発行する。

commit gate自身はHubへ直接publishしない。受理候補を`SingleWriterEventLoop`へ戻し、同loopが
現在tokenを再検査したうえで、reducer更新、projection、`stream_seq`採番、Hub格納を一つの
直列処理として完了する。各snapshotは`reducer_revision`を持ち、Hubも現在revision未満または
現在sessionと異なるsnapshotを拒否する。

### 6.6 PredictionEngine

認識とは別worker processで動かす。表示状態やOBSを知らず、型付きの予測結果だけを返す。
GPUを共有する場合も認識のframe結果が変化しないことを併走A/Bで証明する。

### 6.7 DisplayProjector

reducer状態を配信DTOへ写す純粋変換。独自のhold timer、試合判定、勝率補正を持たない。

### 6.8 SnapshotHubとOverlayTransport

`SnapshotHub`は常に最新snapshot一件だけを保持する。subscriber別FIFOを持たない。
`OverlayTransport`はHTTP、SSE、再接続、cache制御だけを担当し、domain sequenceを生成しない。
subscribeはHubのpublishと同じlock内でsubscriber登録と現在snapshotの1件投入を原子的に行い、
「latestを読んだ後・登録前」の更新を取りこぼす窓を作らない。subscriber bufferが満杯なら同じ
lock内で旧値を除去し、最新値を一件だけ入れる。

Hubへの書込み主体は`SingleWriterEventLoop`一つだけとする。process再起動時は新しい`session_id`を
発行し、順序キーを`session_id + stream_seq`とする。`match_state_seq`はsession内で単調増加し、
試合ごとに0へ戻さない。

既存`src/stream_overlay.py`のHTTPライフサイクル、透過HTML、localhost既定を再利用する。
`StreamState.update(AnalysisResult)`はlegacy endpoint専用に残せるが、Phase Jは新しい
`publish_snapshot(OverlaySnapshotV1)`を使う。旧`/events`とPhase J payloadを混在させず、
Phase Jは`/v1/overlay/`（OBS用HTML）、`/v1/overlay/events`、`/v1/overlay/latest`、
`/v1/overlay/health`へ分離する。Phase J HTMLは同じoriginのv1 eventsだけを購読し、legacy `/`を
書き換えない。

## 7. 配信DTO v1

schema名は`puyo-overlay-snapshot/v1`とする。OBS表示に不要な全盤面・全指標は含めず、
監査用詳細endpointと分離する。

機械可読な正本は
[`docs/schemas/puyo_overlay_snapshot_v1.schema.json`](schemas/puyo_overlay_snapshot_v1.schema.json)
とする。JSON Schema Draft 2020-12で検証し、下記の相関制約は同じmanifestを読む
runtime validatorと独立監査で追加検証する。rule IDと責務分担は
[`docs/schemas/puyo_overlay_snapshot_v1_semantic_rules.md`](schemas/puyo_overlay_snapshot_v1_semantic_rules.md)
を正本とする。canonical payloadは`docs/schemas/examples/`に置く。
producer validatorはDraft 2020-12のformat assertionを有効にし、`format: date-time`を
RFC 3339として検査する。Schemaへformat注釈を書くだけで検証済みと扱わない。

```yaml
schema_version: puyo-overlay-snapshot/v1
identity:
  session_id: string
  stream_seq: integer
  reducer_revision: integer
  match_id: string|null
  match_state_seq: integer
  input_generation: integer
timing:
  source_available_frame: integer|null
  source_available_ms: integer|null
  published_at_utc: string
  capture_to_publish_latency_ms: number|null
  calculation_age_ms: number|null
  last_confirmed_age_ms: number|null
display:
  visibility: visible|hidden
  status: live|physical_prediction|hold|terminal_fact|result|waiting|integrity_fault
  update_reason: typed-enum
  primary_hold_reason: typed-enum|null
  all_hold_reasons: [typed-enum]
  hold_started_ms: integer|null
  hold_elapsed_ms: integer|null
integrity:
  status: ok|degraded|fault
  fault_codes: [typed-enum]
mode:
  evaluation_mode: practical|best_action|both
  practical_basis: board_baseline
evaluations:
  practical:
    availability: available|pending|unavailable
    request_id: string|null
    input_generation: integer|null
    input_digest: string|null
    calculation_latency_ms: number|null
    calculation_age_ms: number|null
    p1_win_probability: number|null
    p2_win_probability: number|null
    advantage_score: number|null
    is_even: boolean|null
    origin: observed|physical_prediction|model|null
    calibration_id: string|null
    evaluated_positions: integer|null
  best_action:
    availability: available|pending|unavailable
    request_id: string|null
    input_generation: integer|null
    input_digest: string|null
    calculation_latency_ms: number|null
    calculation_age_ms: number|null
    p1_position_value: number|null
    p1_position_value_low: number|null
    p1_position_value_high: number|null
    scale: advantage_score
    aggregation_profile_id: string|null
    search_profile_id: string|null
    searched_depth: integer|null
    searched_nodes: integer|null
  player_adjusted: null
input:
  event_seq: integer|null
  p1_board_provenance: confirmed|physics_projected|unknown
  p2_board_provenance: confirmed|physics_projected|unknown
  recognition_quality:
    status: trusted|partial|untrusted
    reason_codes: [typed-enum]
  unresolved_physics: [typed-enum]
  physical_prediction_used: boolean
terminal:
  state: none|candidate|confirmed
  winner: 1P|2P|null
  evidence_kind: visual_result_logo_bilateral_2x2|null
  result_code: p1_win|p2_win|null
runtime:
  status: starting|healthy|degraded
  tier: lightweight|standard|high_accuracy|analysis
  tier_profile_id: string
  practical_queue_depth: integer
  best_action_queue_depth: integer
  practical_worker_health: starting|healthy|degraded|restarting|disabled
  best_action_worker_health: starting|healthy|degraded|restarting|disabled
  discarded_jobs_since_match_start: integer
  telemetry_health: healthy|degraded|unavailable
assets:
  app_build_id: string
  recognition_model_hash: string
  recognition_config_hash: string
  prediction_model_hash: string
  calibration_hash: string
```

通番の意味:

- `stream_seq`: subscriberへpublishした全snapshotのsession内通番。
- `reducer_revision`: reducerが受理して状態を更新した全eventのsession内通番。
- `match_state_seq`: matchに関係するdomain状態更新のsession内通番。formal boundaryでも増やし、
  試合ごとに0へ戻さない。
- `input_generation`: 新しい評価入力を発行した回数。旧job無効化の比較キーに使う。
- process再起動時は新しい`session_id`を発行し、各通番を0から始める。

時刻の意味:

- `source_available_frame/ms`はsnapshotへ取り込んだ最新入力の映像時刻。
- `capture_to_publish_latency_ms`はそのframeをlocal captureが得てからpublishするまでの単調時計差。
- laneの`calculation_latency_ms`はjob開始から完了まで、`calculation_age_ms`は結果完了から
  snapshot publishまでの経過時間。
- rootの`calculation_age_ms`は現在表示の主lane（practical優先、best_action-only時はbest-action）の
  結果age。両laneの詳細値は各laneを正とする。
- `last_confirmed_age_ms`は最後に採用したSTABLE確定入力の利用可能時点からpublishまでの経過時間。
- `hold_started_ms`はhold開始eventの`source_available_ms`を保存する監査用の映像時刻であり、
  `hold_elapsed_ms`はprocess内monotonic clockから算出する表示用経過時間である。hold中だけ非nullとする。
- UTCは人向けログ相関にだけ使い、deadlineと状態遷移はprocess内monotonic clockで判定する。

数値制約:

- 確率は0〜1で、`p1_win_probability + p2_win_probability = 1`
  （両方availableの場合）。
- `advantage_score`は-100〜+100。
- `is_even`は`abs(advantage_score) <= 3`と一致する。
- `advantage_score`と確率の相互変換は`calibration_id`が指す版付き単調関数を単一正本とし、
  producerとvalidatorが同じmanifestから照合する。独立監査は別実装で許容誤差を検査する。
- 最善行動時評価は較正が証明されるまで`win_probability`という名前を使わない。
- `p1_position_value`とlow/highは-100〜+100、正が1P有利、負が2P有利とする。
  low ≤ central ≤ highを必須とし、`aggregation_profile_id`が相手応答、未知ツモ、best/worst/medianの
  集約方法を固定する。単一の相手行動仮定だけからcentralを作らない。
- 同一`session_id + stream_seq`で異なる内容を許さない。
- `evaluation_mode=both`で両laneを同時にavailable表示する場合、`input_generation`と
  `input_digest`を一致させる。片laneが古ければ、そのlaneは値をnullにして`pending`とする。
- `status=live|physical_prediction`でavailableなlaneはrootの`identity.input_generation`と
  一致させる。`status=hold`だけ過去generationの最終確定値を許し、rootより大きいgenerationは
  常に拒否する。
- `visibility=hidden`または`integrity.status=fault`では両評価の値、request ID、input digestを
  null、availabilityを`unavailable`にする。
- `status=terminal_fact`では、allowlist済み`evidence_kind`、winner、確率1/0、score±100を
  必須とし、best-actionは`unavailable`にする。
- `status=result`では通常評価をnullにし、`result_code`をUIの固定文言へ写して表示する。
- `availability!=available`の評価値は全てnullにする。矛盾する組合せはpublish前に拒否する。
- `integrity.status=degraded`はschema、sequence、cutoff、assetsの内容整合が全て成立し、telemetry欠落で
  sessionを完全監査できない場合だけ許す。表示は継続できるが`audit_unavailable`を明示する。
  内容整合違反は`degraded`へ落とさず`fault`とし、数値を非表示にする。
- `player_adjusted`はbeta v1で常にnull。`recognition_quality`と`unresolved_physics`は
  上記の公開用要約だけとし、詳細証拠は監査endpointへ分離する。

schema v1では必須fieldの削除、意味変更、型変更を禁止する。consumerは同じmajor versionの
未知fieldを無視できるが、未知enum値では数値を隠して`schema_unsupported`を表示する。

### 7.1 enum台帳

`typed-enum`のv1集合を次に固定し、コードのEnumとJSON Schemaを同じ生成元から作る。
生成元は`docs/schemas/puyo_overlay_enums_v1.json`とする。

- `update_reason`: `initial_snapshot`、`observation_update`、`physical_fact_update`、
  `prediction_committed`、`hold_started`、`hold_reason_changed`、`hold_expired`、
  `terminal_confirmed`、`result_transition`、`formal_boundary`、`config_pending`、
  `worker_state_changed`、`timer_elapsed`、`integrity_fault`。
- `fault_codes`: `sequence_gap`、`sequence_reversed`、`id_content_conflict`、
  `asset_bundle_changed_mid_match`、`schema_invalid`、`cutoff_violation`、
  `snapshot_revision_stale`、`unexpected_boundary`、`internal_invariant_failed`。
- `recognition_quality.reason_codes`: `board_unstable`、`board_conflict`、`next_unknown`、
  `capture_gap`、`calibration_unavailable`、`recognition_asset_mismatch`。
- `unresolved_physics`: `chain_resolving`、`cancel_unresolved`、`garbage_pending`、
  `landing_ambiguous`、`terminal_candidate_pending`。
- hold理由は第8.2節の集合を正とする。

集合の追加はv1で許すが、producer、validator、consumerの未知値試験を同時に追加する。
既存値の削除、改名、意味変更はschema versionを上げる。

## 8. 更新、保留、非表示、終了

### 8.1 更新

- 信頼できる設置完了。
- 連鎖開始、各段、終了、攻撃確定、相殺、落下開始・完了など、評価を変える物理事実。
- 認識訂正により現在状態が変化したとき。
- 片側連鎖中に、終了後盤面が一意に物理計算でき、操作可能側が設置したとき。

操作途中の位置や操作時間から意図を推定して更新しない。

### 8.2 保留

- 認識が信頼できない。
- 物理状態に複数候補が残り、かつ現在状態baselineも既存の安全条件を満たさない。
- 両側連鎖中で新しい選択や確定事実がない。
- 現在generationの計算待ち。
- terminal候補はあるが確定根拠がない。

hold中は最後の確定値、経過時間、主要理由、全理由を表示する。理由は強いものが弱いものへ
黙って退行しない。原因別上限を超えたら画面全体は`visibility=visible`、`status=hold`、
`update_reason=hold_expired`のまま、対象laneを`pending`へ移して評価値だけをnullにし、理由表示と
`hold_started_ms`を残す。pending jobのrequest ID、generation、digestは保持してよい。
`hold + hidden`は禁止する。

projected-stateだけに端数着弾列の複数候補が残る場合はholdへ遷移しない。正式projectedモデルの
採用前はbaseline値を維持し、採用後は全候補の等確率平均を使う。両段階の診断badgeは
`unresolved_physics_reason=landing_ambiguous`、監査詳細は`landing_columns_ambiguous`とする。
本節の段階契約は`PHASE_J_GUARANTEED_LANDING_PROJECTION_ADDENDUM_2026-09-04.md` v0.3を正とする。

hold理由は自由文字列でなく、強い順に次のenumとする。

1. `terminal_confirmation_pending`
2. `recognition_unreliable`
3. `physics_ambiguous`
4. `prediction_worker_fault`
5. `prediction_deadline_missed`
6. `calculation_pending`
7. `both_chaining_no_new_fact`

`integrity_fault`と`match_boundary_awaiting_input`はholdでなく即時非表示である。主要理由は
activeな理由のうち最上位、表示上限はactiveな理由ごとの上限の最小値を使う。理由ごとに
開始、継続、解除、境界resetを記録する。通常表示のhold timerと、死亡候補等の証拠有効期限は
別のclock・別fieldとし、値を共用しない。

### 8.3 決着

- 死亡候補だけで100対0にしない。
- v1の`terminal_fact` allowlistは`visual_result_logo_bilateral_2x2`だけとする。左右の勝敗ロゴを
  2フレーム間隔で2回同方向に確認し、現在のmatch IDと一致した場合に限る。
- 攻撃量、死亡候補、死亡位置、次試合開始、片側ロゴだけでは`terminal_fact`へ移らない。
- 「2フレーム以内」は正本として`source_frame_seq`の差で判定する。`source_frame_seq`を取得不能と
  事前登録したprofileだけ、欠落・buffer置換・backend重複が0の連続`capture_seq`を代用できる。
  いずれかが発生した時点でterminal confirmationの連続回数をresetする。
- 下沈み演出等の新しい根拠は、別版の独立監査とユーザー採用決定までallowlistへ追加しない。
- `terminal_fact`受理時に全jobを無効化し、勝者100%、敗者0%を表示する。
- 最短表示timer後は予測値を消し、結果表示へ移る。ただしformal boundaryが先に来た場合は
  timerを待たず新matchのhidden/waitingへ移り、前試合の100対0を残さない。
- 次試合境界で前試合の値、死亡候補、hold理由、job tokenを全て破棄する。

### 8.4 初期時間値（ユーザー承認前の測定候補）

- 通常hold上限: 2秒。
- 100対0の最短表示: 1秒。
- 標準tierの設置確定から表示commit: P95 500ms、P99 1秒。

これらは合格値ではなく、実測とユーザー確認のための初期候補である。

### 8.5 reducer event・guard・effect

- `formal_boundary`: event IDが現在cutoff内、単一の正式通知元、未処理であることをguardとする。
  effectはmatch ID更新、generation更新、全job無効化、評価・候補・hold全消去、即時非表示。
- `observation_available`: observation/event sequenceが単調、digest一致、資産ID一致をguardとする。
  信頼入力ならgeneration発行とjob投入、不確実なら理由付きholdまたは非表示。
- `prediction_completed`: commit token、request ID、capture session/seq、cutoff digest、input digest、
  deadline、laneが現在値と一致することを
  guardとする。effectはsingle-writer内の採用、projection、Hub publish。不一致は破棄だけ。
- `terminal_evidence`: 現在match、cutoff内、allowlistの左右一致をguardとする。候補根拠は
  candidate更新だけ、allowlist成立時だけterminal確定、job全無効化、100対0。
- `capture_status`: 切断、再接続、途中起動を扱う。切断受理と同じ原子的更新でgenerationを進め、
  待機・実行中を含む全job tokenを無効化してhold後非表示へ移る。再接続ではCaptureSourceが発行した
  新しいcapture sessionを受理・記録する。通知順に関係なくsession ID変更自体でも全jobを無効化し、
  新しい信頼入力がcommitされるまで前値を復元しない。
- `config_change_requested`: 次試合境界までpendingにし、現在試合のtier、mode、資産を変えない。
- `timer_fired`: reducer自身が登録したtimer ID、deadline、現在session/revisionとの一致をguardとする。
  hold中はprocess monotonic由来の`hold_elapsed_ms`を1秒ごとに再publishし、hold上限、
  terminal_fact最短表示、worker deadlineを無入力中でも進める。古いtimerは破棄する。
- sequence欠番、逆順、同一ID異内容、資産変更は`integrity_fault`として評価null・非表示にする。

## 9. latest-winsとバックプレッシャー

新しい評価対象が来たときは、次の順序を固定する。

1. reducerが先に`input_generation`を進め、旧結果を無効化する。
2. 待機jobを新しいjobへ置換する。
3. 実行中jobへ協調cancelを通知する。
4. practical laneを常に優先する。
5. 結果をcommit gateで照合する。
6. token不一致、deadline超過、資産不一致を理由付きで破棄する。

### 9.1 認識clockと表示clock

過去に確定した「認識は30fps、判定表示は秒2回で十分」を維持し、二つを同じloopにしない。

- captureは30fps取得を試み、認識/reducerは受理したframeごとに物理事実と安全状態を逐次更新する。
  backlogは作らず最新frameへ置換できるが、backend drop、buffer置換、sequence gapを全件計測する。
  30fps無損失は無条件の契約ではなく、OBS併走時のbeta性能gateとして判定する。
- practicalの数値commitは通常最大2Hzとし、その間の候補は最新generationへ集約する。
- formal boundary、hold開始/期限、integrity fault、terminal candidate/confirmed、worker fault、
  接続healthは2Hz制限を受けず、即時publishする。
- hold中の`hold_elapsed_ms`更新は正しい`timer_fired`で1秒ごとに再publishし、数値commitの2Hz集約へ
  混ぜない。
- best-actionは独立laneで最大1秒を初期計算目標とし、未完了でもpracticalを待たせない。
- 計算時間の初期目標はpractical P95 250ms、hard deadline 500msとする。2Hz集約を含む
  capture-to-OBS表示commitはP95 500ms、P99 1秒を初期測定目標とする。
- 同じgenerationについて計算結果が複数ある場合も、stream_seqを遡らせず最新の合格結果だけを送る。

不合格条件:

- 前試合の結果が一度でも表示される。
- queue満杯時に最新jobまたは最新snapshotを捨てる。
- 遅いsubscriberが古いFIFOを消化する。
- best-actionがpracticalを待たせる。
- tier、モデル、較正器を試合中に変更する。
- deadline超過結果を最新値として表示する。
- 認識がenqueueや監査書込みで待つ。

## 10. SSE ADR

初版はSSEを採用し、WebSocketと両対応は行わない。

理由:

- 初版のデータ方向は解析サーバからOBSへの一方向だけ。
- 既存のHTTP/SSE、透過HTML、複数購読テストを再利用できる。
- 同一PC・localhost限定なので、初版では認証、CORS、TLSが不要。
- 双方向操作が必要になった時点で別ADRとしてWebSocketを検討できる。

SSE要件:

- `event: snapshot`
- `id: <session_id>:<stream_seq>`
- OBSブラウザソースURLは`http://127.0.0.1:<port>/v1/overlay/`とする。
- 接続時は履歴を再送せず、現在snapshotを即送る。
- HTTP受付前に`stream_seq=0`、`match_id=null`、`visibility=hidden`、`status=waiting`、
  両評価`unavailable`の初回snapshotをsingle-writerでHubへ登録する。Hub空の状態で受付を始めない。
- clientは同じsession内の古い`stream_seq`と`match_state_seq`を拒否する。新しい`session_id`を
  受けたら旧画面と旧sequenceを破棄し、初回snapshotから再開する。HTTPの`Last-Event-ID`は
  履歴再送に使わない。
- `/v1/overlay/`のHTML、`/latest`、`/health`は`Cache-Control: no-store`。
- `/events`は`no-cache, no-transform`。
- `/v1/overlay/health`はserver、session、最新stream sequence/snapshot age、telemetry health、
  subscriber数/上限、loopback bind、稼働開始/確認時刻、直近error codeを返す。正本は
  `docs/schemas/puyo_overlay_health_v1.schema.json`とする。
- SSE切断中に解析や認識を止めない。
- Phase J runnerは`127.0.0.1`以外のbind指定を起動前に拒否する。
- subscriber数を固定上限で拒否し、接続ごとに書込みtimeoutを設定する。遅いclientは切断し、
  認識・予測・他clientへ波及させない。
- server停止時は購読thread/socketを強制解除し、満杯queueへのsentinel成功を前提にしない。
- 接続、切断、timeout、上限拒否、送信例外を抑止せずTelemetrySinkへ送る。
- legacyの`/events`、`/latest`とPhase Jの`/v1/overlay/*`を型・event名・試験で隔離する。

## 11. 品質tier

全tierで次を共通にする。

- 同じ認識、物理規則、攻撃・相殺・死亡の記録。
- 見えているnext 2手の合法配置探索。
- 同じ実戦予測の意味と同じDTO。
- tier別の固定較正器。
- 試合中のtier変更禁止。

変更可能なのは、探索深度、**見えていない未来の候補数**、beam幅、相手応答深度、
未知色候補数、未来盤面評価回数だけ。見えているnext 2手の合法配置はどのtierでも省略しない。
各tierは名前でなく、予算、モデル、較正器、最大結果年齢を含むhash付きmanifestで固定する。

- `lightweight`: 共通最低契約。実戦予測と見えている範囲の最善候補。
- `standard`: 初版の既定候補。併走測定で性能を満たすこと。
- `high_accuracy`: 計算増加が未知試合の精度向上を示す場合だけ提供。
- `analysis`: 録画後replay専用。配信runnerは拒否する。

### 11.1 未来評価の段階導入

過去のユーザー合意どおり、難しい両者探索を初期betaの障害にせず、次を独立に採否する。

1. 現在の確定盤面と確定情報による実戦予測。
2. 進行中の連鎖・相殺・おじゃまを物理計算した解決後盤面による実戦予測。
3. 両者が操作可能な状態で、見えている現在ツモを双方1手ずつ進める最善行動時評価。
4. 見えているnextまで候補を絞る複数手探索。
5. 見えていないツモを確率分布、長手数を版付き近似で扱う探索。

段階3では手番速度を勝率入力にせず、「両者が1手ずつ進める一巡」として近似する。相手応答を
「常に最善」「平均的」など一つへ固定せず、最良、最悪、中央値付近、ばらつき、安全な手の割合、
即時攻撃・防御につながる割合を監査出力へ保持する。公開DTOはcentralとlow/highを示し、
`aggregation_profile_id`で集約法を固定する。段階4〜5も探索量、上限/下限、未知ツモのばらつきを
保存し、一つの確定勝率へ潰さない。

各段階は直前段階と同一試合・同一時点で比較し、精度、較正、重大逆判定、無event急変、物理整合、
実時間、保留率が全て非劣化の場合だけ次の標準tierへ入れる。改善しない重い段階は高精度または
分析tierへ隔離する。

### 11.2 同一tier内の状況別探索

tierを固定したまま、版付きsearch profileが現在の物理状態から探索量を決定してよい。これは負荷を
見て黙って品質を下げる処理ではなく、同じ入力なら同じ探索方針になる決定的な規則とする。

- 両者とも操作不能: 将来手を探索せず、確定した物理解決だけ進める。
- 片方だけ操作可能: 操作可能側だけ探索する。
- 両者とも操作可能: 双方の合法配置を組み合わせ、応答分布を保持する。
- 攻撃・落下が目前: 間に合う一手程度を優先する。
- 通常構築中: 見えているnextの範囲を候補化する。
- 長い先読み: 全場面で必須にせず、候補を絞った版付き近似にする。

実際に選ばれたprofile ID、depth、nodes、candidate数、打切り理由を監査ログへ残す。

## 12. UI要件

- 実戦予測と最善行動時評価を、ラベル、色、単位で明確に分ける。
- LIVE、物理予測、更新待ち、非表示、決着確定を文字でも示し、色だけに依存しない。
- HOLD時は値を動かさず、最終更新からの秒数を表示する。
- 最善行動時評価は較正完了まで%表記せず、-100〜+100の盤面評価として表示する。
- 最善行動時評価はcentralだけを確定値のように見せず、low〜highの範囲と探索段階を併記する。
- 盤面基準と対戦者補正を同じ数値へ混合しない。
- 透明背景、1920×1080基準、OBSのスケーリング、文字切れ、スマートフォン相当縮小を検収する。
- 旧HTMLの単一スコアバーは置換するが、透過表示と同一originのEventSource構成は再利用する。

### 12.1 初期betaの表示候補

ユーザー目視前の実装基準を次へ固定し、配色・位置・大きさは短いfixture動画で最終調整する。

```text
┌──────────────────────────────────────────────────────┐
│ P1  63%        実戦予測  +26        37%  P2          │
│ ████████████████████│████████████                    │
│ LIVE・標準・更新 0.2秒前                              │
│ 最善手評価 +18  範囲 -4〜+32  段階2     # both時だけ │
└──────────────────────────────────────────────────────┘
```

- practical-onlyでは1段目と状態行、best-action-onlyでは盤面評価・範囲・探索段階を主段にする。
- P1は左、P2は右、advantage scoreの正方向はP1で固定する。左右反転fixtureでもラベルを入れ替えない。
- EVEN領域は中央±3として文字でも`EVEN`を示す。bar長だけで判定させない。
- HOLDは数値を固定し、主理由と経過秒を状態行へ表示する。期限後も画面はvisibleなHOLDを維持し、
  対象laneをpendingへ移して数値/barだけを消し、理由と経過秒を残す。
- physical prediction、入力不良、worker fault、監査不能を同じ警告語へ潰さない。
- terminal_factは通常評価を置換して勝者と100対0を示し、resultでは予測数値を残さない。
- 背景は透明、文字とbarには半透明の縁取りを付け、明暗どちらのゲーム背景でも判読可能にする。
- 初期betaはP1/P2表記とし、選手名・個人補正を入れない。装飾animationは状態変化の意味を遅らせない。

## 13. Telemetry

少なくとも次をsession、match、tier別に記録する。

- capture、認識、job開始、完了、commit、SSE publishの時刻。
- job置換、cancel要求、commit拒否、deadline missと理由。
- in-match総時間を分母にしたLIVE、物理予測、HOLD、HIDDENの時間。
- hold理由別件数、合計、P95、最大。
- 認識frame drop、queue深度、CPU、GPU、VRAM、RAM、subscriber数。
- model、recognition、calibration、tier profile、buildのhash。
- practicalとbest-actionの差、両laneの入力generation、探索量、どちらかがstaleだった時間。
- telemetry故障時はsessionを`audit_unavailable`とし、正常と偽らない。

domain時間は単調frame/出来事通番で管理する。処理時間はprocess内monotonic clockで測り、
UTCは表示と監査の目印にだけ使う。

TelemetrySinkはbounded bufferとし、認識producerを待たせない。監査eventには優先度を付け、
buffer不足時は破棄件数、最初/最後の欠落sequence、理由を別の固定容量counterへ必ず残す。
永続化失敗または欠落発生後は`telemetry_health=degraded|unavailable`をsingle-writerへ通知し、
回復しても当該sessionの`audit_unavailable`印を消さない。buffer容量とflush間隔は負荷試験で固定する。

## 14. テスト

### 14.1 単体・性質試験

- reducerの全遷移、hold理由優先度、timer、境界reset。
- DTO schema、値域、左右反転、実戦/最善/選手補正の非混合。
- 同一入力、設定、資産で同一結果。
- 不明、範囲、候補集合をゼロや単一値に潰さない。
- `visibility`、`status`、`integrity`、`availability`の禁止組合せを全列挙して拒否する。
- Phase J runnerがloopback以外のbindをfail-closedで拒否する。

### 14.2 scheduler競合試験

- 完了順を故意に逆転させる。
- cancel不能job、前試合job、同一seq異内容、試合中tier変更。
- practicalとbest-actionの飢餓防止。
- 1000更新と遅いclientでも最後に最新snapshotを受信する。
- commit後publish前に境界または次generationを割り込ませ、旧snapshotがHubへ入らない。
- laneごとのhard timeout、worker再生成、連続timeoutでもtierが自動変更されない。

### 14.3 replay・未来参照試験

- 出来事原本を`available`順だけでreplayする。
- full処理とprefix処理の共通部分は、domain sequence、cutoff、input digest、予測値、表示状態を
  bit-identicalにする。UTC、処理時間、session ID、request ID等のruntime fieldは比較から除く。
- 未来の公式勝者を早期注入しても、その時点までの表示が変化しない破壊試験。
- replay速度を変えてもdomain sequenceと評価値を一致させる。
- terminal、境界、coverageを表示器と独立した実装で監査する。
- J4接続前に、同一試合・同一causal cutoffの学習特徴量表と`ObservationAdapter`出力を照合する。
  runtime専用fieldを除外後、列順、dtype、missing mask、normalized value、assets hashの差分を0とする。

### 14.4 OBS・性能試験

- OBS CEF実機で透過、初回snapshot、切断再接続、browser source再読込。
- 長時間soakでメモリ、subscriber、thread、socketを監視する。
- captureからOBS描画までのP50/P95/P99を測る。
- 認識単独と予測併走でframe dropと認識結果hashを比較する。
- tier別にlog loss、Brier、較正、重大逆判定、処理時間を比較する。
- local subscriber上限、SSE書込みtimeout、強制切断、停止時thread/socket回収を試験する。

### 14.5 ユーザーレビュー成果物

過去に確定したレビュー方針を維持する。

- 1画面・1場面・1論点の短い動画にし、変化前後を数秒含める。
- 時刻、対象side、認識した出来事、勝率の前後、主要理由を一般的な言葉で表示する。
- 確定情報、物理予測、最善行動、保留、不明を色と文字の両方で区別する。
- 大量の内部数値は監査詳細へ退避し、スマートフォン相当縮小でも本文を読めるようにする。
- 判定入力は「明確な誤り」「要調査」「大きな違和感なし」「判断不能」の4種類にする。
- 最初に短い一覧を示し、選択した場面だけ詳細確認できる構成にする。
- 可能な代表例は、第一段階で勝率と最終結果を隠して認識・物理だけを確認し、第二段階で
  モデル出力と根拠を開示する。意見が割れた場面を無理に正解へせず保留として残す。

合否値は測定項目と分離してtier profileへ固定する。草案時点の候補は更新P95/P99だけであり、
soak時間、許容frame drop、認識hash不一致、deadline miss率、subscriber上限、メモリ増加上限は
小規模実測前に合否値を確定しない。値を決めたcommitと、値を見ずに測った結果を別記録にする。

## 15. 採用ゲート

### 15.1 beta入口

- 第4節の入口条件を全て満たす。
- 旧結果表示0、試合跨ぎ0、黙ったtier変更0の自動試験。
- 認識単独と併走で意味結果が不変。
- 実OBSで初回snapshotと再接続を確認。
- 全pytest成功。
- `src/production_config.py`はまだ変更しない。

### 15.2 beta出口

- 難所を含むcoverage、stale、hold、hidden監査を提示する。
- 同一PCの長時間soakを通す。
- 未使用動画、未知選手、未来時点で精度と較正が非劣化。
- 代表画面、数値、タイムラインを独立検収する。
- ユーザー目視で表示意味、読みやすさ、hold/非表示動作を確認する。

### 15.3 production

- tier別の性能予算と較正を固定する。
- 配布用設定、障害bundle、復旧手順を検証する。
- Fable/Codexの最終独立レビューでP0/P1を閉じる。
- ユーザー承認後だけ、採用日と根拠を`src/production_config.py`へ登録する。

## 16. 実装順序

1. ADR、DTO、reducer状態遷移、commit tokenを仕様として固定する。
2. reserveキューを操作せず完走させる。
3. c36/c60境界修正、回帰試験、停止9件の別名再export、独立監査を先に閉じる。
4. 純粋reducer、DTO validator、scheduler競合試験を動画なしで作る。
5. v1 HTML/SSEを最新snapshot hubへ追加し、legacy互換とOBS再接続試験を作る。
6. event sourceのoffline replay adapterで、未来参照と試合跨ぎを検証する。
7. W36/W37、terminal、100本超モデルを閉じ、practical laneへ本番入力を接続する。
8. 併走性能測定からstandard tierの予算を固定する。
9. best-action laneを独立追加し、practical非干渉を証明する。
10. 実OBS beta、独立監査、ユーザー目視を経て本番採否を決める。

## 17. 初期推奨とユーザー判断待ち

以下を仕様v1の初期値として推奨する。ユーザー回答またはFableレビューで更新する。

1. 標準tierはpractical計算P95 250ms・hard deadline 500ms、2Hz集約込みの表示commitを
   P95 500ms・P99 1秒とする。
2. 通常holdは2秒、その後は理由を残して数値を非表示にする。
3. 決着確定の100対0は最低1秒表示し、その後結果表示へ移る。
4. 最善行動時評価は較正完了まで-100〜+100で示し、%と呼ばない。
5. 評価モードとtierの変更は、どちらも次の試合境界で反映する。
6. 対戦者実力補正は初回betaから外し、DTOの予約枠だけ用意する。
7. 同一PC内のライブ映像入力は、ゲーム/ウィンドウの直接取得を第一候補とし、OBS出力の
   再キャプチャは遅延と画質の比較で同等以上の場合だけ採用する。実際の配信入力形態はユーザー確認待ち。

## 18. 既存資産の扱い

再利用:

- `src/stream_overlay.py`のHTTP server、localhost既定、透過HTML、SSE heartbeat。
- `tests/test_stream_overlay.py`と`tests/test_stream_overlay_obs.py`のHTTP/SSE基礎試験。
- `RecognitionPipeline`の確定/推定盤面、state、next、provenance。
- 出来事原本の通番、利用可能時刻、資産hash、追記監査。
- 左右勝敗ロゴ検出の左右対称ロジック。ただしリアルタイム独立再検収後のみ。
  死亡候補ロジックはhold判断に限り、100対0根拠へ再利用しない。
- `production_config.py`の採用台帳方式。

切り離す:

- 旧`AnalysisResult`をPhase J DTOとして使う経路。
- subscriberごとの16件FIFOと、満杯時に最新値を捨てる挙動
  （`src/stream_overlay.py:116`）。
- `src.old`の8指標Scorer。
- レビュー動画用の遡及訂正と未来参照。
- `scripts/visualize_advantage_overlay.py`の巨大main loop。表示方針は純粋部品へ抽出する。
- 公開DTOへの全盤面、全指標、デバッグ証拠の混入。

## 19. レビュー状態

- 既存資産棚卸し: 完了。
- Codexアーキテクト独立レビュー: 完了。P0を本仕様へ反映。
- Codex検収レビュー: 初回P0 4件、P1 12件を修正。最終再検収は新規P0/P1なしでPASS。
- health追加分のCodex独立再検収: 初回P0 1件、P1 4件を修正。最終再検収はP0/P1残存なしでPASS。
- Capture追加分のCodex独立再検収: 初回P0 1件、P1 4件、追補P1 3件を修正。
  capture profile/session伝播、progress信号、stall境界、clean feed、A/B性能比較を含めP0/P1残存なしでPASS。
- Capture最終アーキ再検収: 旧job無効化、lease寿命、意味不変frameのcommit継続、独立正解盤面と
  paired cluster bootstrapまで確認し、P0/P1残存なしでPASS。
- v0.4〜v0.5契約: JSON Schema構文・全local ref・22 Enum集合・canonical snapshot 8例・health 5例・48 test ID・
  12 requirement IDの機械整合を確認。pending値残存、lane時刻欠落、hidden値残存、勝者方向不一致、
  terminal score逆向き、未知worker状態の負例6種は全て拒否した。`/health`も正例適合と
  healthy時のerror残存拒否を確認済み。
- Claude/Fable独立反対査読: 2026-09-04完了。初回P1 2件、P2 10件を本仕様・Schema・
  意味規則・既存test IDへ反映し、再検収はGO、新規P0/P1なし。
- DTO v1のSchema契約試験27件、canonical snapshot 8件を独立validatorでPASS。
- v1.0は合成fixture限定のJ1実装開始契約とする。本番入力接続とbeta採否は本書の入口条件を全て閉じた後に別判断する。
