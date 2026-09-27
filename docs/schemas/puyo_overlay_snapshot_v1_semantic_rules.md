# Phase J DTO v1 相関制約台帳

更新: 2026-09-04 JST  
対象: `puyo-overlay-snapshot/v1`、`puyo-overlay-health/v1`

JSON Schema単体で表現しにくい時系列・算術・版付き較正の制約を含め、publish前validatorと
独立監査が同じrule IDで記録する。validatorが一件でも失敗したsnapshotは配信しない。

## Schemaで強制する規則

### S01 必須fieldと値域

11個のroot fieldと各nested requiredを必須とし、確率は0〜1、scoreは-100〜100、通番・age・
queue depthは非負とする。queue depthはlatest-winsのため0または1だけ許す。Draft 2020-12の
format assertionを有効にし、snapshot/healthの`date-time`をRFC 3339として検査する。

### S02 hidden/faultの非表示

`display.visibility=hidden`または`integrity.status=fault`なら、両laneを`unavailable`にし、
request ID、generation、digest、age、評価値、根拠をnullにする。

### S03 pending/unavailableの出力null

`availability=pending|unavailable`なら、そのlaneの評価値と結果metadataをnullにする。
pendingでは投入済みjobを追跡するrequest ID、generation、digestだけ保持してよい。
unavailableではそれらもnullにする。

### S04 terminal_factの必要条件

`display.status=terminal_fact`ならvisible、terminal confirmed、allowlist済みevidence、winner、
result code、practical available、確率1/0、score±100、`is_even=false`、`origin=observed`を必須にし、
best-actionはunavailableとする。

### S05 terminal方向

1P勝利は`p1_win`、確率1/0、score +100とする。2P勝利は`p2_win`、確率0/1、score -100とする。

### S06 resultの通常評価非表示

`display.status=result`ならterminal confirmedと整合するwinner/result codeを必須にし、両laneを
unavailableにする。

### S07 enumと固定値

仕様7.1節のenum以外を拒否し、`schema_version`、`practical_basis`、best-actionの`scale`を固定する。

## Runtime validatorで強制する規則

### R01 確率和

practical available時は`p1_win_probability + p2_win_probability = 1`を、較正manifestが定める
許容誤差内で満たす。

### R02 score・確率・EVENの整合

`calibration_id`が指す版付き単調関数で確率とscoreを照合する。`is_even`は
`abs(advantage_score) <= 3`と一致させる。terminal_factはS05を優先する。

### R03 rootとlaneのgeneration

liveまたはphysical_predictionでavailableなlaneは`identity.input_generation`と一致させる。
holdだけ過去generationを許し、rootより大きいgenerationは常に拒否する。

### R04 bothの同一入力

両laneを同時にavailable表示する場合、input generationとdigestを一致させる。不一致laneは
値を消してpendingへ落とす。

内部`match_phase=intermission`はdisplayをhidden/waiting、両laneをunavailableへ射影する。

### R05 input digest

同じcanonical inputとasset bundleから同じdigestを得る。digest計算対象とserialize順を版付きで
固定し、違う内容に同じdigestを再利用しない。

### R06 id-content一意性

同一`session_id + stream_seq`に異なるserialized contentを許さない。同一request IDに異なる
commit tokenまたは結果を許さない。

### R07 単調通番

同一session内で`stream_seq`と`reducer_revision`を単調増加させる。match内で
`match_state_seq`を単調増加させ、formal boundary後も0へ戻さない。process再起動で新しい
session IDを発行した場合だけ各sequenceを0から始めてよい。

### R08 commit token

model結果をcommitする時点でsession、match、generation、evaluation mode、tier profile、asset hash、
deadlineを現在stateと照合する。一項目でも違えば破棄してdiscard counterを増やす。

### R09 causal cutoff

snapshotが参照するevent、board、物理事実、terminal evidenceは`source_available_frame/ms`以下で
利用可能になった情報だけとする。offline replayでも同じ制約を使う。

### R10 asset bundle不変

match途中で認識・設定・予測・較正hashが変わった場合は値をcommitせず、
`asset_bundle_changed_mid_match`でfail-closedにする。次のformal boundaryでだけ切り替える。

### R11 表示状態と根拠

`physical_prediction`は少なくとも一方のprovenanceが`physics_projected`で、
`physical_prediction_used=true`の場合だけ許す。liveで両者確定を名乗る場合は両provenanceを
confirmedとする。

`confirmed_board`と既存45指標はNON-STABLE中に更新しない。`projected_state`は別object・別digest・
別provenanceにし、確定盤面へ書き戻さない。projected入力用に学習・較正されていない評価器では
physical_predictionをavailableにせずholdする。

### R12 hold timer

hold開始時に映像時刻`source_available_ms`を`hold_started_ms`へ固定する。表示用の
`hold_elapsed_ms`はprocess monotonic clockから算出し、hold中だけ非null、hold以外ではnullにする。
正しいtimer eventで1秒ごとに経過時間を2Hz数値集約外で再publishする。理由が解消されなければ2秒後に
対象laneを`pending`へ移して評価値だけをnullにする。画面全体は`visibility=visible`、
`status=hold`、`update_reason=hold_expired`を維持し、hold理由と開始時刻を残す。
投入済みjobのrequest ID、generation、digestはpending中だけ保持してよい。
wall clockをprojectorが直接読んで状態を変えない。

### R13 terminal evidenceの時系列

左右結果ロゴのframe差が2以下で、同一winnerを2回連続確認した場合だけconfirmedへ遷移する。
候補1回、攻撃予測、死亡候補、次試合盤面を100対0の根拠にしない。
frame差は`source_frame_seq`を正本とする。取得不能を事前登録したprofileだけ、欠落・buffer置換・
backend重複が0の連続`capture_seq`を代用し、いずれかの発生時に連続確認をresetする。

### R14 boundary reset

formal boundaryでhold、terminal candidate、未解決物理、match内discard count、未完了jobを破棄する。
境界前jobの完了通知は新matchへcommitしない。terminal_fact最短表示timerよりboundaryを優先し、
前試合の100対0を新matchへ持ち越さない。

### R15 modeとlane

live、physical_prediction、holdでは、modeがpracticalならbest-actionを、best_actionならpracticalを
公開しない。bothだけ両laneを公開できる。terminal_factとresultは勝敗事実を優先するためこの規則の
例外とし、S04/S06へ従う。mode変更はformal boundaryでのみ反映する。

### R16 available結果の識別情報

terminal_fact以外でavailableなlaneはrequest ID、input generation、input digest、calculation latency、
calculation ageを必須とする。practicalはcalibration ID、best-actionはsearch profile、depth、nodesを必須とする。
識別情報が欠ける結果はpendingへも採用せず破棄する。

### R17 表示状態の合法組合せ

holdは`visibility=visible`、primary reason、空でないall reasons、hold開始時刻を必須とし、
`hold + hidden`は禁止する。期限前の対象laneは最後の確定値を`available`で保持し、期限後は
`update_reason=hold_expired`として対象laneを`pending`へ移し、数値を消して理由を維持する。
hold以外では理由と開始時刻をnull・空配列にする。integrity_faultはintegrity status=faultと
空でないfault codesを必須とする。waiting、result、intermission相当では評価を公開しない。
visibleな通常数値はlive、physical_prediction、期限前holdに限る。

`integrity.status=degraded`はschema、sequence、cutoff、assetsの内容整合が成立し、telemetry欠落で
完全監査不能な場合だけ許す。表示継続時も`audit_unavailable`を明示する。内容整合違反はfaultとし、
degradedへの格下げを禁止する。

### R18 worker状態とlane

workerがdisabledなら対応laneをunavailable、queue depthを0にする。starting/restartingでは新しい
available結果をcommitしない。degradedは既にcommit済みの値をholdできるが、新結果は通常のcommit
gateとdeadlineを免除しない。
Windowsではworkerの入力受付前にpre-warmを完了させる。再生成中は`prediction_worker_fault` holdを
継続し、`worker_health=restarting`中に到着した新resultをcommitしない。

### R19 表示主laneのage

rootの`timing.calculation_age_ms`はpracticalが表示中ならpractical、best-action-onlyならbest-actionの
同名値と一致させる。表示可能なlaneがなければnullにする。`capture_to_publish_latency_ms`と
`last_confirmed_age_ms`は同一sessionのmonotonic clockから算出し、UTC時刻の差を状態判定に使わない。

### R20 best-actionの範囲と集約

best-action available時はlow ≤ central ≤ highを満たし、3値とも-100〜100とする。
`aggregation_profile_id`は相手応答と未知ツモの候補群からcentral/rangeを作る版付き規則を指し、
単一の相手行動仮定だけを黙ってcentralとして採用しない。profile不明・候補不足ではunavailableとする。

## Health runtime validatorで強制する規則

### H01 server状態とerror

`server_status=healthy`では`telemetry_health=healthy`かつ`last_error_code=null`とする。未解消の
配信障害、worker障害、または`telemetry_health=degraded|unavailable`があればserverもdegradedとし、
Enum台帳にある版付きerror codeを必須にする。未知値と空文字列を拒否する。

### H02 subscriber上限

`subscriber_count <= subscriber_limit`を常に満たす。上限到達後の接続は受付前に拒否し、既存subscriberや
認識・予測処理を待たせない。

### H03 Hubとの一致

healthの`session_id`と`latest_stream_seq`は同じlockで読んだSnapshotHubのlatestと一致させる。
`telemetry_health`もlatest snapshotの`runtime.telemetry_health`と一致させ、server statusは同じsnapshotの
runtime/worker状態と未解消health faultから決定する。`latest_snapshot_age_ms`はprocess内monotonic clockで
算出し、UTC時刻の差を状態判定に使わない。

### H04 bind実体との一致

`bound_host=127.0.0.1`を固定し、`bound_port`は実際にlisten済みのportと一致させる。設定値だけを返さない。

## 独立監査で追加確認する規則

### A01 較正再計算

producerとは別実装でR01/R02を再計算し、許容誤差とmanifest hashを記録する。

### A02 cutoff再構成

利用可能時刻順のevent logからsnapshot列を再構成し、未来frame参照、遡及訂正、試合跨ぎを検査する。

### A03 latest-wins収束

低速consumerと再接続consumerが中間snapshotを飛ばしても、最終的に同じ最新stream_seqとcontentへ
収束することを検査する。

### A04 左右対称

P1/P2を反転したfixtureで確率、score、winner、result code、provenanceが正しく反転し、side非依存の
絶対量が反転しないことを検査する。

### A05 fail-closed

worker停止、schema不正、asset変更、sequence gap、telemetry障害を注入し、誤った数値が残らないこと、
fault codeと復旧遷移が追記監査へ残ることを検査する。

## Canonical examples

`docs/schemas/examples/`の8例を正例としてJSON Schema検証する。負例は各ruleの正常例を一項目だけ
変更して作り、期待したrule IDで拒否されることをunit test化する。
