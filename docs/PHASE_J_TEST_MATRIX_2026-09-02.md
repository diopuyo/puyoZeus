# Phase J beta テスト台帳

更新: 2026-09-04 JST  
対象: 同一PC、localhost、SSE、practical優先、best-action独立lane  
件数: 48（Phase単位30〜50件の規約内）

全caseは動画・GPUなしで実行できる合成試験を先に作る。`P0`は一件でも失敗すればbetaを出さない。
`P1`は実装単位の完了前に閉じる。実動画でしか確認できない内容は、同じIDを独立監査へ引き継ぐ。

## C. 契約とserialize（6件）

1. `C01 P0` canonical snapshot 8 payloadとhealth 5 payloadがDraft 2020-12 Schemaを通る。
   format assertionを有効にし、snapshot/health双方の不正RFC 3339日時を拒否する。
2. `C02 P0` snapshot/healthのroot/nested requiredを各一つ欠落させると拒否される。
3. `C03 P0` snapshot/healthの全enumの未知値をproducer validatorが拒否する。
4. `C04 P1` consumerは同一major versionの未知fieldを無視する。
5. `C05 P0` Enum定義とJSON Schemaの集合差分が0である。
6. `C06 P0` 同一snapshotのserialize結果とcontent digestが決定的である。

## V. 相関validator（10件）

1. `V01 P0` hiddenまたはintegrity faultで両laneの数値・識別子が全てnullになる。
2. `V02 P0` pending/unavailableに評価値を残すと拒否される。unavailableでrequest ID、generation、
   digestのいずれかが残る負例も両laneで拒否し、pendingだけはjob識別子を保持できる。
3. `V03 P0` practicalの確率和がmanifest許容誤差を外れると拒否される。
4. `V04 P0` 確率、score、calibration IDの不整合を拒否する。
5. `V05 P0` `abs(score)<=3`と`is_even`の不一致を拒否する。
6. `V06 P0` live/physical_predictionでrootとlaneのgeneration不一致、または表示主laneとrootの
   calculation age不一致を拒否する。
7. `V07 P0` holdの過去generationは許し、未来generationは拒否する。
8. `V08 P0` bothで両laneのgenerationまたはdigestが違えば片laneをpendingへ落とし、best-actionの
   low ≤ central ≤ highまたはaggregation profileが不正なら拒否する。
9. `V09 P0` terminalのwinner、result code、確率、scoreの方向不一致を拒否する。
10. `V10 P0` mode、status、visibility、hold reason、integrityの禁止組合せを全列挙して拒否する。
   `hold + hidden`を禁止し、`hold_expired`はvisibleなholdと対象lane pendingの組合せだけ許す。
   `integrity.degraded`は内容整合済みかつtelemetry欠落の合法例だけを許し、内容整合違反を
   degradedとした負例、telemetry正常のdegraded負例を拒否する。

## R. Reducer状態遷移（10件）

1. `R01 P0` 起動直後はHTTP開始前からhidden/waiting snapshotを持つ。
2. `R02 P0` 信頼できる両者STABLE入力だけがgenerationを進めjobを投入する。
3. `R03 P0` event/capture sequenceの欠番、逆順、同一ID異内容を検出する。eventはintegrity faultへ
   遷移し、captureはsource欠落とbuffer置換を区別して品質理由付きholdへ移す。1件bufferのburst後に
   最新frameと正確な置換数を一度だけ渡し、欠落時は連続frame判定をresetする。lease中の画像変更、
   fan-out、process間受渡し、早期再利用、二重release、未release、pool枯渇、lost wakeupを拒否する。
   J1は純粋reducerで初回/正順/欠番/逆順/重複/同一ID異内容とfail-closed遷移までを固定する。
   buffer置換数、lease、fan-out、pool、lost wakeupはcapture adapterを実装するJ4のR03継続項目とする。
4. `R04 P0` formal boundaryだけがmatch IDを更新でき、旧評価・hold・terminal候補・jobを消す。
   intermissionはhidden/waiting/両lane unavailableへ射影し、前試合値を保持しない。
5. `R05 P0` 認識不確実化で理由付きholdへ入り、NON-STABLE中にconfirmed board/既存45指標を
   更新せず、未較正projected stateを表示しない。
6. `R06 P0` 正しいtimer IDだけが2秒後にvisibleなholdを維持したまま対象laneをpendingへ移し、
   数値を消して理由・開始時刻を残す。`hold_started_ms`は開始時の映像時刻、`hold_elapsed_ms`は
   process monotonic由来の非負経過時間とし、hold中だけ非nullにする。正しいtimerで1秒ごとに
   2Hz数値集約外の再publishを行い、古いtimerを無視する。
7. `R07 P1` hold理由の追加・解消時にprimary/all reasonsが決定的に更新される。
8. `R08 P0` capture切断後はholdからhiddenへ移る。再接続ではCaptureSourceだけが新capture sessionを
   発行し、reducerは受理・記録する。旧sessionの
   遅延frame/envelopeと切断前jobの遅延完了を拒否して、信頼できる新観測まで旧値を復元しない。
   正当な静止盤面をdigest一致だけで
   stall扱いしない。進行信号なしのprofileを起動前拒否し、stall閾値直前は正常、超過時だけholdへ移る。
   profile変更とbackend自動failoverは試合中拒否し、次のformal boundaryだけで反映する。
   J1は切断即hidden、新capture session初受理、旧session/job拒否までを固定する。stall、profile検証、
   backend failoverはcapture/runtime接続を持つJ4のR08継続項目とする。
9. `R09 P0` config変更要求はpendingとなり、次のformal boundaryまでmode/tier/assetsを変えない。
10. `R10 P0` 同一event列から同一snapshot列を再現できる。

## S. SchedulerとSnapshotHub（8件）

1. `S01 P0` laneごとに実行中1件・待機最新1件を超えない。
2. `S02 P0` 新入力で待機旧jobを捨て、実行中へcancelを通知するがcancel成功へ依存しない。
3. `S03 P0` session/match/capture session+seq/cutoff/generation/lane/tier/assets/deadlineの一項目でも
   現generationのimmutable入力snapshotと違う結果を破棄する。意味状態が不変の後続capture frameだけでは
   generation/tokenを変えず、正常な進行中結果を採用できる。
4. `S04 P0` formal boundary前の結果が境界後に完了しても一度もpublishされない。
5. `S05 P0` best-action飽和中もpracticalの投入・commit・publishが継続する。
6. `S06 P0` subscriber buffer満杯時は旧snapshotを捨て最新snapshotを保持する。
7. `S07 P0` stream_seq採番、latest保存、subscriber通知、およびsubscribe時の登録+latest投入が
   同じlockの原子的更新となり、接続raceで更新を失わない。
8. `S08 P1` 30fps相当の入力burstで数値publishを最大2Hzへ集約しつつ、安全eventは即時送信し、
   遅い複数consumerが最終的に同一stream_seq/contentへ収束する。

## T. HTTP/SSEとruntime（7件）

1. `T01 P0` v1のHTML/events/latest/healthがlegacy endpointとpayloadを混在させず、v1 HTMLが
   v1 eventsだけを購読する。HTML/latest/healthは`Cache-Control: no-store`、eventsは
   `no-cache, no-transform`を返すことをheaderで検査する。
2. `T02 P0` 既定bindはlocalhostで、初回beta設定から外部interfaceへ公開せず、healthのbind値が
   実際のlistenerと一致する。
3. `T03 P1` SSE event名、data、空行終端、heartbeatがEventSource互換である。
4. `T04 P0` 接続直後と再接続直後に現在のlatest snapshotを受け取り、healthのsession/stream sequenceが
   同じHub latestと一致する。
5. `T05 P1` H02としてsubscriber countが上限を超えるhealth payloadを拒否し、上限到達後の接続を
   既存処理へ波及させず拒否する。切断、server停止、再起動でthread/queueを残さない。
6. `T06 P0` worker fault時は対応laneの旧確定値を理由付きholdし、期限後も画面はvisibleなholdを
   維持して対応laneだけpendingへ移し、数値を消して理由・開始時刻を残す。
   Windowsでは入力受付前pre-warmを必須とし、再生成中は`prediction_worker_fault` holdを継続、
   `worker_health=restarting`中に到着した新resultを拒否する。
   telemetry fault時は評価表示を継続できるがruntime/healthをdegraded・監査不能とする。H01/H03として
   degradedのerror欠落、healthyのerror残存、telemetry状態とsnapshot/healthの不一致を拒否する。
7. `T07 P0` shutdown時にhidden snapshotを送ってからworkerとHTTPを閉じる。

## E. Causal replay・terminal・実機gate（7件）

1. `E01 P0` offline replayはavailable frame/event sequenceを超える情報を参照しない。
2. `E02 P0` 遅れて届いた訂正を過去snapshotへ遡及適用せず、新eventとして処理する。
3. `E03 P0` 結果ロゴ1回、片側だけ、frame差3以上、winner不一致で100対0を出さない。
4. `E04 P0` 左右ロゴframe差2以内・同一winner 2回だけがterminal confirmedになり、1秒timer前でも
   formal boundaryが来れば即hiddenへ移って100対0を持ち越さない。frame差は`source_frame_seq`を
   正本とし、取得不能profileだけ連続`capture_seq`を代用する。欠落・置換・重複を注入し、連続確認が
   resetされることと、各事象の母数を含むfail-safe結果を検査する。
5. `E05 P0` P1/P2反転fixtureでscore・確率・winnerだけが反転し、絶対量は反転しない。
6. `E06 P0` 代表試合の独立監査でcutoff violation、試合跨ぎ、schema rejectが各0件である。
   J4前に同一試合・同一cutoffの学習特徴量表とObservationAdapter出力をruntime専用field除外後に比較し、
   列順、dtype、missing mask、normalized value、assets hashの差分を各0とする。
7. `E07 P1` OBS実機で透過、再接続、hold、非表示、terminal 1秒、result遷移、clean feedの自己再帰0を
   確認する。直接取得とOBS再取得は5分同時の意味比較後、順序を反転した各30分の単独性能runで測定し、
   曖昧pair、欠落、buffer置換を報告する。採用backendは長時間soakし、1場面1論点の短いclipを
   スマートフォン縮小と4択判定で目視PASSする。P1/P2方向、EVEN、LIVE/予測/HOLD、理由・経過秒、
   both時の2段表示、terminal/resultで数値が混ざらないことも確認する。A/Bは仕様6.1の事前登録manifestに
   対するsample数、独立正解盤面、match/formal boundary clusterのpaired bootstrap、認識非劣化、coverage、
   遅延、資源、優先順位を機械判定する。manifestにはbackend/profile別のHSV較正資産IDと入力SHAを固定し、
   正規化後ROIの画素同値を証明できないbackend間で較正資産を共用しない。

## 実行順

1. J1でC01〜C06、V01〜V10と、R01〜R10の純粋reducer/projector範囲を実装する。
   R03/R08のcapture adapter固有項目はJ4で同じIDを継続し、延期を完了扱いにしない。
2. J2でS01〜S08、T01〜T06を実装する。
3. runtime接続時にT07を実装する。
4. offline replayとterminal接続時にE01〜E06を実装する。
5. 最後にE07と全pytestを実施する。

テスト名にはIDを含め、失敗ログと監査JSONにも同じIDを記録する。仕様変更でcaseを削除せず、
不要化した理由と代替caseを残す。
