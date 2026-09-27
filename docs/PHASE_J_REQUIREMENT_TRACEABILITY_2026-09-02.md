# Phase J 要件トレーサビリティ

更新: 2026-09-02 16:40 JST  
目的: Claude開発期のユーザー決定が、現仕様・実装計画・テストから抜け落ちないよう追跡する。

## J-REQ-01 同一PCのOBSブラウザソース

- 要件: 解析とOBSは同じPC。初期betaは外部公開を要しない。
- 仕様: 2節、6.1節、10節。localhost、HTTP+SSE、loopback以外は起動前拒否。入力方式は
  `CapturedFrame`契約のadapterとして分離する。
- 実装: `capture.py`、`transport.py`、`runtime.py`。
- 検証: `R03`、`R08`、`T01`〜`T05`、`E07`。
- 状態: 反映済み。実入力形態だけユーザー確認待ち。

## J-REQ-02 3つの表示モード

- 要件: 実戦予測のみ、最善行動時のみ、両方を設定で切替。既定は実戦予測。
- 根拠: `docs/agent_coordination/DECISIONS.md` 2026-08-28「実戦予測と最善行動時の評価」。
- 仕様: 3節、7節、17節。異なる意味の値を一つへ混合しない。
- 実装: `contracts.py`、`display_projector.py`、mode変更は次境界で適用。
- 検証: `V08`、`V10`、`R09`。
- 状態: 反映済み。

## J-REQ-03 更新・保留・終了

- 要件: 設置ごとに更新、片側連鎖中は物理解決後盤面、両側連鎖中は新事実だけ更新。
  不確実時は最後の確定値と経過時間、試合前後は非表示、物理確定時のみ100対0から結果へ遷移。
- 根拠: `DECISIONS.md` 2026-08-28「評価を更新・保留・終了する場面」。
- 仕様: 8節。通常hold 2秒、terminal_fact 1秒は初期測定値。
- 期限後状態: 画面全体はvisibleなholdを維持し、対象laneをpendingへ移して数値だけを消し、
  hold理由と開始時刻を残す。`hold + hidden`は禁止する。
- 実装: `reducer.py`、`display_projector.py`。
- 検証: `V01`、`V02`、`V10`、`R02`、`R05`〜`R08`、`E03`〜`E04`。
- 状態: 反映済み。

## J-REQ-04 因果的な出来事時系列

- 要件: 固定手数・固定秒を主軸にせず、設置、連鎖、攻撃、相殺、落下、全消し、死亡、勝敗の
  出来事時系列を使う。後から分かった情報を過去へ戻さない。
- 根拠: `DECISIONS.md` 2026-08-28「学習の時間軸」「由来・不確かさ」。
- 仕様: 4節、6.2節、8.5節。CausalCutoff、ObservationEnvelope、追記event。
- 実装: `event_source.py`、`reducer.py`、`offline_replay.py`。
- 検証: `R03`、`R04`、`R10`、`E01`、`E02`、`E06`。
- 状態: 反映済み。

## J-REQ-05 確定・物理予測・不明を分離

- 要件: 観測、物理的一意導出、複数候補、観測不能を混ぜず、STABLE確定盤面を凍結する。
- 根拠: AGENTS.md、`DECISIONS.md` 2026-08-28「由来・不確かさ」。
- 仕様: 5節、7節、8節。provenance、recognition quality、unresolved physicsを公開要約へ保持。
- 実装: `contracts.py`、`event_source.py`、`display_projector.py`。
- 検証: `V10`、`R02`、`R05`、`E01`。
- 状態: 反映済み。詳細証拠は公開DTOでなく監査endpointへ分離。

## J-REQ-06 性能別品質tier

- 要件: lightweight、standard、high_accuracy、analysisを選択可能にする。意味と物理規則は共通、
  実行中に黙って変更せず、段階ごとに較正する。
- 根拠: `DECISIONS.md` 2026-08-28「端末性能に応じた未来予測の品質段階」。
- 仕様: 11節。見えるnext 2手の全配置は全tier共通、analysisは配信拒否。
- 実装: tier manifest、`scheduler.py`、`runtime.py`。
- 検証: `S03`、`S05`、`R09`、性能gate。
- 状態: 反映済み。具体的予算は実機併走測定待ち。

## J-REQ-07 盤面基準・対戦者補正・最善行動を分離

- 要件: 公開既定は選手IDを使わない上級者平均の盤面基準。対戦者補正と最善行動は別値にする。
- 根拠: `DECISIONS.md` 2026-08-28「盤面の価値・対戦者の実力・最善行動時の評価」。
- 仕様: 3節、7節、12節。beta v1の`player_adjusted`はnull。
- 実装: practical/best-action別lane、player adjustment予約枠。
- 検証: `C01`、`V08`、`V10`。
- 状態: 反映済み。対戦者補正は初回beta対象外。

## J-REQ-08 認識30fps・判定表示2Hz

- 要件: 認識と判定表示を分離し、認識30fps、判定は秒2回で十分。
- 根拠: `docs/VIDEO_SCRIPT_FLOW_FACTS_2026-08-10.md`。
- 仕様: 9.1節。数値だけ最大2Hzへ集約し、boundary/hold/fault/terminal/healthは即時publish。
- 実装: `capture.py`、`scheduler.py`、`snapshot_hub.py`。
- 検証: `R03`、`S01`、`S02`、`S08`、`E07`、P50/P95/P99 telemetry。
- 状態: 反映済み。

## J-REQ-09 未来評価は段階導入

- 要件: 物理解決後盤面を先に、両者1手、見えるnext、未知ツモ、長手数近似を段階的に追加する。
  両者操作可能時は相手応答を一仮定へ固定せず、最良・最悪・中央・ばらつきを保持する。
- 根拠: `docs/ADVANTAGE_MODEL_WALL_DISCUSSION_TURN_BY_TURN_2026-08-28.md` 発言307〜315。
- 仕様: 11.1節。段階ごとに精度、較正、重大誤り、物理、時間、coverageを採否。
- 実装: practical lane、best-action lane、tier manifest。
- 検証: `S05`、`E05`、段階別offline比較。
- 状態: 反映済み。段階3以降は初期betaのblocking条件にしない。

## J-REQ-10 人がレビューしやすい成果物

- 要件: 1画面1論点、前後数秒、時刻/side/出来事/理由、確定と予測の見分け、スマートフォン可読、
  「明確な誤り・要調査・大きな違和感なし・判断不能」の4択。
- 根拠: 同turn-by-turn文書 発言305〜306。
- 仕様: 12節、14.5節。
- 実装: beta UIとreview clip generatorは別責務にする。
- 検証: `E07`。
- 状態: 反映済み。

## J-REQ-11 coverageを精度の分母から消さない

- 要件: LIVE、予測、hold、hiddenの時間、最長停止、staleを母数付きで報告し、難所を黙って除外しない。
- 根拠: `DECISIONS.md` 2026-08-28「評価を更新・保留・終了」。
- 仕様: 1節、13節、15節。
- 実装: `telemetry.py`と独立監査。
- 検証: `E06`とbeta gate。
- 状態: 反映済み。

## J-REQ-12 本番採用の前提

- 要件: W36/W37後の再測定、独立監査、ユーザー目視、production config単一正本。
- 根拠: AGENTS.md、`DECISIONS.md` 2026-08-24、今回の継続タスク。
- 仕様: 4節、15節、16節。
- 実装: reserve完走後にc36/c60境界修正、停止9件再export、W36/W37、terminal、100本超モデルの順。
- 検証: 全48case、全pytest、独立監査、実OBS。
- 状態: 未完。現在の最大blocking chain。

## 未確定でユーザーへ確認する内容

1. ライブ映像はゲーム/ウィンドウ直接取得、capture card、OBS出力再取得のどれか。
2. practical計算P95 250ms/hard 500ms、表示P95 500ms/P99 1秒を初期beta目標にしてよいか。
3. hold 2秒、terminal_fact 1秒を初期値にしてよいか。
4. best-actionを較正完了まで確率でなく-100〜+100表示にしてよいか。
5. player-adjustedを初期betaから外してよいか。

上記はJ1の契約・reducer単体実装を止めない。実入力adapter、UI、beta gateを固定する前に回答を反映する。
