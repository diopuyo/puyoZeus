# Phase J 実装計画（同一PC・OBS beta）

更新: 2026-09-04 JST  
状態: 仕様v1.0・合成J1完了。J2 SingleWriter／pure worker／transport純粋層まで最終検収GO、実listener未接続。  
仕様正本: `docs/PHASE_J_REALTIME_OVERLAY_SPEC_2026-09-02.md`  
DTO正本: `docs/schemas/puyo_overlay_snapshot_v1.schema.json`
Health正本: `docs/schemas/puyo_overlay_health_v1.schema.json`
相関制約: `docs/schemas/puyo_overlay_snapshot_v1_semantic_rules.md`
Enum正本: `docs/schemas/puyo_overlay_enums_v1.json`
テスト台帳: `docs/PHASE_J_TEST_MATRIX_2026-09-02.md`
要件追跡: `docs/PHASE_J_REQUIREMENT_TRACEABILITY_2026-09-02.md`

## 1. 到達点

同一PC上で、リアルタイム認識・有利不利評価・OBSブラウザ表示を一つの実行系として動かす。
初回betaは次を満たした時点で成立とする。

- OBSはlocalhostのSSEから常に最新snapshotだけを受け取る。
- 表示値は、その時点で利用可能だった入力だけから作る。未来参照と試合跨ぎを拒否する。
- 不安定盤面、未解決物理、認識資産不整合では、仕様どおりholdまたは非表示にする。
- 実戦予測と最善行動時評価を別laneにし、重い探索が実戦予測を止めない。
- 決着の100対0は左右結果ロゴを2フレーム差以内で2回確認した場合だけ表示する。
- 実行障害時に誤った数値を残さず、原因コードを監査ログに残す。

## 2. 実装を始める条件

### 2.1 今すぐ実施可能

- 仕様、JSON Schema、相関制約一覧、実装分割、テストケースの確定。
- 既存資産の読み取り専用調査。
- 動画や認識系を使わないfixture設計。

### 2.2 reserveキュー完走まで禁止

- `src/`、`scripts/`、認識資産の編集。
- 実行中`run_event_reserve_background_v1`の停止、再起動、重複起動。
- 動画解析、download、既存成果物の再生成・上書き。

### 2.3 本番入力接続の前提

1. reserveキュー完走を確認する。
2. observer/resolverのMATCH_BOUNDARY伝播を最小修正し、c36/c60を回帰試験する。
3. 停止9件を保存済みNPZ/sidecarから別名再exportし、独立監査PASSを確認する。
4. W36/W37補正後に認識精度を再測定する。
5. terminal検出をリアルタイム条件で独立検収する。
6. ティア確認済み100本超データで正式モデルを学習・較正する。

上の1より前にPhase Jのproduction codeは変更しない。2〜6の完了前でも、合成fixtureだけを入力する
transport/reducer試験は進められるが、beta合格とは扱わない。

## 3. 予定モジュール

既存`src/stream_overlay.py`はlegacy APIの互換層として保ち、Phase Jの新規責務を混在させない。
新規packageは`src/phase_j/`とする。

### `contracts.py`

- DTOのdataclass、Enum、serialize/deserializeを持つ。
- `schema_version`と全enumを一か所で定義する。
- JSON SchemaとEnumの差分をテストで検出する。
- 表示用DTOに盤面全体、45指標、debug証拠を入れない。

### `validator.py`

- JSON Schemaで表現できる型、required、enum、値域を検査する。
- Draft 2020-12のformat assertionを有効にし、snapshot/healthの`date-time`をRFC 3339として検査する。
- 以下の相関制約を純粋関数で検査する。
  - P1確率 + P2確率 = 1（許容誤差を版付きmanifestで固定）。
  - score、確率、`is_even`、`calibration_id`の整合。
  - hidden/fault/unavailable時の数値と識別子のnull化。
  - live/physical_prediction時のgeneration一致。
  - hold時は過去generationだけ許可し、未来generationを拒否。
  - both時のlane間generation/digest一致。
  - terminal_fact時のevidence、winner、1/0、±100の方向一致。
  - result時の通常評価null化。
- publish前の不正snapshotは送らず、fail-closed snapshotへ変換する。

### `reducer.py`

- 単一writerの純粋状態遷移を担当する。
- 入力は通番付きevent、出力は内部state。transportやmodelを呼ばない。
- `session_id`、`match_id`、`match_state_seq`、`input_generation`を管理する。
- formal boundaryでmatch固有状態、hold、terminal候補、未完了commit tokenを破棄する。
- timer eventも通常eventとして扱い、wall clockを直接読まない。
- intermissionをhidden/waiting/両lane unavailableへ決定的に射影する。
- hold開始時の映像時刻とprocess monotonic由来の経過時間を分離し、1秒timerで経過時間を更新する。

### `display_projector.py`

- 内部stateから公開snapshotへ射影する純粋関数。
- STABLE、物理予測、hold、hidden、terminal、resultを一か所で決める。
- 表示文言ではなく、typed enumとnull規則を確定する。
- 初期snapshotは必ずhidden/waitingで生成する。

### `scheduler.py`

- practical laneとbest-action laneのjob投入、deadline、cancel、結果commitを担当する。
- 認識30fpsと判定表示最大2Hzを別clockにし、数値jobだけを最新generationへ集約する。
- boundary、hold、fault、terminal、healthのpublishを2Hz制限へ入れない。
- commit tokenはsession/match/capture session+seq/causal cutoff/generation/evaluation/tier/assets/deadlineを
  含み、切断・再接続時に全jobを無効化する。
- 古い結果、境界前結果、asset hash不一致、deadline超過結果を破棄する。
- practical laneを高優先度とし、best-actionの飽和から隔離する。
- Windowsではworkerを入力受付前にpre-warmし、再生成中はfault holdを維持する。
  `worker_health=restarting`中の新resultはcommitしない。

### `snapshot_hub.py`

- validator合格済みの最新snapshotを一つだけ保持する。
- `stream_seq`採番と保存とsubscriber通知を同じlock内で原子的に行う。
- subscriber bufferは1件。満杯なら旧値を捨て、最新値を入れる。
- subscriber登録と現在snapshot投入をpublishと同じlock内で原子的に行い、接続raceを防ぐ。
- 接続直後に現在snapshotを送り、その後の更新を配信する。

### `transport.py`

- `/v1/overlay/`のOBS用HTMLと、`events`、`latest`、`health`を提供する。
- v1 HTMLは同じoriginのv1 eventsだけを購読し、legacy `/`と`/events`を変更しない。
- localhost bindを既定かつbeta唯一の対応範囲とする。
- SSE heartbeat、再接続、unknown field許容を維持する。
- HTML/latest/healthを`no-store`、eventsを`no-cache, no-transform`で返す。
- legacy `/events`、`/latest`とはpayloadを混在させない。

### `telemetry.py`

- latency、age、queue depth、discard、worker health、schema rejectを計測する。
- 高カーディナリティの盤面や生画像を常時ログへ出さない。
- 障害bundleは追記型で、既存成果物を上書きしない。

### `runtime.py`

- reducer、scheduler、projector、hub、transportをcomposition rootで接続する。
- HTTP開始前に初期hidden snapshotをpublishする。
- 認識adapterと評価workerを依存注入し、単体試験でfakeへ交換可能にする。
- shutdown時は数値を消したsnapshotを送ってからworkerを閉じる。

### adapter群

- `capture.py`: `CapturedFrame`契約と差し替え可能なcapture backend。capture session内の連番、
  monotonic時刻、semantic frame digest、source欠落・buffer置換・反復、画像ownershipを保持し、
  latest-winsの1件bufferで認識を待たせない。`CaptureStatus`は別eventとし、欠落時は連続frame判定を
  リセットする。read-only leaseは単一consumerが`finally`で1回だけreleaseし、画像をprocess間へ渡さない。
  直接ゲーム/ウィンドウ取得とOBS clean feed再キャプチャは同じinterfaceで比較する。
- `event_source.py`: リアルタイムobserver/resolver eventをPhase J eventへ変換する。
- `offline_replay.py`: 保存済みeventを利用可能時刻順に再生する。未来参照を禁止する。
- `practical_model.py`: 正式100本超モデルのadapter。未完成時はunavailableを返す。
- `best_action.py`: 独立process用adapter。相手応答を一仮定へ潰さず、central、low/high、
  aggregation profileと監査用分布を返す。固定tier内では物理状態から決定的なsearch profileを
  選び、profile/depth/nodes/candidate数/打切り理由を返す。初回betaでは無効でもよい。

## 4. 実装順とゲート

### J0: 仕様固定（キュー中に実施）

- JSON Schemaがparseでき、全local refが解決する。
- Draft 2020-12 format assertionを有効にし、snapshot/healthの不正RFC 3339日時を拒否する。
- 正常・hidden・fault・terminal・resultのcanonical fixtureを定義する。
- `hold_started_ms`と`hold_elapsed_ms`、intermission射影、telemetry欠落だけに限定した
  `integrity.degraded`の意味を契約とfixtureで固定する。
- JSON Schema外の相関制約を全て検査項目へ対応付ける。
- Claude/Fable独立レビュー結果はv1.0へ反映済み。今後のP2以下は対象stageのgateへ追記する。

完了条件: 仕様とテスト台帳だけで、実装者が値のnull条件と採否条件を推測せずに済む。

### J1: 契約と純粋状態機械（reserve復旧完了後）

- 2.3節の1〜3（キュー完走、c36/c60境界修正、停止9件再export監査）を先に完了する。
- `contracts.py`、`validator.py`、`reducer.py`、`display_projector.py`を実装する。
- 動画、GPU、HTTPを使わないunit testを先に通す。
- formal boundary、sequence gap/reversed、asset change、hold timeoutを重点試験する。

完了条件: J1で実装する契約・validator・純粋reducer・projector範囲の状態遷移、
禁止組合せ、左右対称性が自動試験でPASS。capture buffer lease/burst/stall、worker、terminal、
runtime接続の枝はそれぞれJ4、J2、J3以降の対応test IDへ明示的に残し、未実装eventは黙殺せず拒否する。

### J2: 競合と配信

- `scheduler.py`と`snapshot_hub.py`を実装する。
- 遅延結果、順不同完了、試合跨ぎ、queue飽和、複数subscriberを再現する。
- transport v1を追加し、OBS相当EventSource試験を実施する。
- cache headerをendpoint別に検査し、Windows workerのpre-warm、再生成中hold、
  restarting result拒否を合格gateへ含める。

完了条件: practicalの更新をbest-actionが阻害せず、各consumerが最後に同一stream_seqへ収束。

### J3: offline causal replay

- 保存済みeventを利用可能時刻順に投入する。
- 出力snapshot列を同じ入力で再現可能にする。
- 元動画の未来frameを使っていないことをcutoff監査で証明する。

完了条件: 代表試合で試合跨ぎ0、cutoff violation 0、schema reject 0。

### J4: production input接続

- 2.3節の認識・境界・terminal・モデル前提を閉じる。
- 実observer/resolver adapterとpractical modelを接続する。
- capture A/B manifestへprofile別HSV較正資産IDと入力SHAを固定し、画素同値未証明の資産共用を拒否する。
- terminalの2-frameを`source_frame_seq`正本で監査し、代用条件とdrop時resetを破壊試験する。
- 同一試合・同一cutoffで学習特徴量表とruntime ObservationAdapterを、runtime専用field除外後の
  列順、dtype、missing mask、normalized value、assets hashまで差分0と確認する。
- standard tierで遅延と欠落を測る。

完了条件: practical計算P95 250ms・hard deadline 500ms、2Hz集約込みの表示commit
P95 500ms・P99 1秒の初期目標を達成、または実測根拠でtier予算を改訂。

### J5: OBS beta

- OBSブラウザソースの再接続、透過、解像度、文字可読性を実機確認する。
- 実戦値、hold、非表示、決着、結果遷移を実画面で検収する。
- 1場面1論点の短いreview clipと4択判定一覧を作り、スマートフォン縮小でも検収する。
- 独立監査とユーザー目視の後だけproduction flagを登録する。

完了条件: P0/P1なし、誤った100対0なし、障害時fail-closed、復旧手順再現済み。

## 5. テスト配置

- `tests/phase_j/test_contracts.py`: serialize、enum、schema version、unknown field。
- `tests/phase_j/test_validator.py`: 相関制約とfail-closed。
- `tests/phase_j/test_reducer.py`: 通番、境界、hold、timer、terminal。
- `tests/phase_j/test_display_projector.py`: 表示/非表示、左右対称、null規則。
- `tests/phase_j/test_scheduler.py`: token、deadline、stale result、lane隔離。
- `tests/phase_j/test_snapshot_hub.py`: latest-wins、原子性、multi-client。
- `tests/phase_j/test_transport.py`: HTTP、SSE、heartbeat、再接続、localhost。
- `tests/phase_j/test_offline_replay.py`: causal cutoff、再現性、試合跨ぎ。
- `tests/phase_j/test_terminal.py`: 2-frame gap、2-confirm、偽陽性、左右対称。
- `tests/phase_j/test_runtime_faults.py`: worker停止、telemetry障害、asset差替え、shutdown。

既存`tests/test_stream_overlay*.py`はlegacy回帰として維持する。Phase J試験から旧`AnalysisResult`を
importしない。

## 6. 最初のfixture台帳

最低限、次のsnapshot/event列を固定fixtureにする。

1. 起動直後: hidden + waiting + unavailable。
2. 両者STABLE: practical available。
3. 片側連鎖中で予測一意: physical_prediction。
4. 予測不確定: hold開始、2秒後timer eventでも画面はvisibleなholdを維持し、対象laneをpendingへ
   移して数値だけを消す（理由・開始時刻とpending jobの識別子は維持可）。
5. practical結果が遅れて到着: token不一致で破棄。
6. best-action飽和: practical更新は継続。
7. formal boundary: match固有stateと古いjobを破棄。
8. asset hashが試合中に変化: integrity fault、数値null。
9. 左右ロゴ候補1回: terminal candidateのまま100対0禁止。
10. 左右ロゴ2回: terminal_factでwinner方向の1/0と±100。
11. result遷移: 通常評価null、result_codeのみ表示。
12. SSE consumer停止後復帰: 中間値を飛ばし最新stream_seqへ収束。

## 7. 変更単位

レビューを小さく保つため、次の単位で実装する。

1. contracts + schema fixtures。
2. validator + reducer + projector。
3. scheduler + snapshot hub。
4. v1 transport + OBS client試験。
5. offline replay + cutoff監査。
6. recognition/model adapter。
7. telemetry + recovery bundle。
8. beta UIとproduction config登録。

各単位で対象testを通し、最後に全pytestを実行する。`src/production_config.py`は採用判断の
最後まで変更しない。

## 8. 現時点の保留

- J2以降のscheduler、Hub、worker、terminal、capture/runtime実装に対する段階別再検収。
- P95/P99、hold 2秒、100対0表示1秒の最終承認。
- standard tierで使う具体的モデルとGPU予算。
- best-actionの探索器、較正法、表示開始時期。
- beta UIの最終レイアウト。
- ライブ映像の実入力形態（ゲーム/ウィンドウ直接、capture card、OBS出力再取得）。

これらはJ1の契約・状態機械実装を妨げない。意味が変わる場合はDTO v1を壊さず、profileまたは
optional fieldで差し替える。
