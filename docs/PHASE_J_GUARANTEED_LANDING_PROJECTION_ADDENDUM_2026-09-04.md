# Phase J 返し切れない予告おじゃまの着弾後評価 追補仕様

作成日: 2026-09-04  
状態: v0.4（初回着弾が次設置より先の構造gateを追補。正式モデル学習・本番採用は未実施）  
適用先: `docs/PHASE_J_REALTIME_OVERLAY_SPEC_2026-09-02.md`、
`docs/FORMAL100_PROJECTED_STATE_PREREGISTRATION_2026-09-04.md`

## 1. 決定

予告おじゃまを受けている側が連鎖を発火し、その連鎖による相殺後も同じ側へ正味おじゃまが残り、
初回着弾まで追加の設置機会がない場合は、現在の凍結盤面だけで評価しない。

発火前の確定盤面から連鎖終了後盤面を物理計算し、返し切れず残るおじゃまを初回着弾させた
`projected_state`を実戦予測へ使う。投影盤面は`confirmed_board`へ書き戻さず、
`physics_projected`由来の別入力として保持する。

この投影は勝敗確定ではない。死亡位置が埋まる予測を含め、0%／100%を許す根拠は従来どおり
左右一致の勝敗ロゴだけとする。

## 2. 適用条件

次を全て満たす場合だけ`guaranteed_landing_after_active_chain`を成立させる。

1. 投影対象は、**いま発火中の連鎖を持つ側自身**である。非発火側への送付予測はこのgateに含めない。
2. 発火前のSTABLE確定盤面とactive chainのroot-familyが一意である。
3. `chain_started`直前の因果会計で、同じ側への確定予告おじゃまが1個以上存在する。
4. 連鎖結果、確定または信用可能な暫定攻撃量、実効レート、leftoverの連続性を検証できる。
5. 相互交換を解決した後も受け側が反転せず、同じactive chain側へ正味予告が1個以上残る。
6. 連鎖終了から初回着弾まで、その側に追加設置の選択機会がない。
7. event順序、match boundary、source/build/asset hashが現在のcausal cutoffと一致する。

条件6は映像時間の推測ではなく、受け側の`chain_active=true`、同じ側のactive root-familyが一意かつ
信用可能、現在発火中の両側familyを解決した後も受け側が変わらない、というevent物理から判定する。
この条件ではゲーム規則上、受け側の次の新ツモ設置より先に初回おじゃまが着弾する。相手側family、
event順序、相殺残量のいずれかが未解決なら成立させない。
`active_chain_first_drop_precedes_next_placement`を保証gateの最終条件とし、
active sideとrecipientの一致および正の残量が成立しなければ
`first_drop_timing_untrusted`として`guaranteed`を発行しない。

完全相殺、余剰攻撃による方向反転、anchor複数、rate不明、leftover不連続、境界不一致では、
このgateの数値を作らない。確定した別方向の投影へ移れる場合だけ、新しい受け側で計算し直す。

## 3. 投影する状態

初回着弾量は、相殺後残量と1回の最大着弾30個の小さい方とする。30個を超える分は
`leftover_after_first_drop`として保持し、まだ盤面へ置かない。

正式入力には少なくとも次を保持する。

- 発火前の確定盤面。
- active chainの連鎖終了後盤面。
- 初回着弾後盤面または着弾候補盤面集合。
- 相殺後残量、初回着弾量、初回後の残量。
- 受け側、root-family、causal cutoff、asset hash、盤面provenance。
- 物理的既知0、missing、着弾列不確実を区別するmask。

現在の盤面特徴と投影盤面特徴を同じ名前で上書きせず、学習時とPhase Jの
`ObservationAdapter`で列順、dtype、正規化、mask、hashを完全一致させる。

## 4. 端数列と将来着弾

初回着弾量を6で割った余りが0なら、列選択に関する不確実性はなく、盤面は一意に投影できる。
余りがある場合、個数は確定しても端数おじゃまの列はランダムであり、単一の疑似seed盤面を
「確定盤面」と表示してはならない。

端数は最大20通りの列組合せを有限分岐として保持する。正式モデルの集約法とRNG重みが
凍結され、較正・外部評価・Phase J parity・採用gateを通るまでは、各候補盤面の範囲だけを監査する。
この未採用段階では投影候補を利用不能として`landing_columns_ambiguous`を付け、現在状態のbaselineが
利用可能なら数値を維持する。candidate都合で評価行を分母から除外せず、新しいHOLDも追加しない。

`projected_set_residual_cnn_v1`が正式採用された後は、端数候補集合を利用可能入力とし、全候補を
等確率の単純平均で集約する。`landing_columns_ambiguous`は診断badgeとして残すがfallback理由にはしない。
単一疑似seed盤面を正式入力・死亡判定・表示へ使わない規則は採用後も変えない。
Phase J DTOの`unresolved_physics_reason`へ出すときは既存Enum値`landing_ambiguous`へ写像し、
`landing_columns_ambiguous`は監査用の詳細reason codeとして保持する。

30個を超える残量は次の着弾までに新しい設置が入り得るため、現時点で確定扱いできるのは
「初回着弾後盤面＋leftover」までである。leftover全量を現在盤面へ一括着弾させない。

## 5. Phase J表示

- 一意な投影または全端数候補集合と較正済み正式モデルがある場合だけ`status=physical_prediction`、
  `origin=physical_prediction`、盤面provenance=`physics_projected`で表示する。
- 正味量、初回着弾量、leftoverは診断値として表示できるが、未較正モデルで勝率を更新しない。
- 正式モデル採用前に端数列だけが曖昧ならbaseline値を維持する。採用後は全候補の等確率平均を使い、
  どちらの段階でも`landing_columns_ambiguous`を診断表示し、疑似seed由来値は公開しない。
- anchor、順序、rate、hashまたは現在状態自体が曖昧な場合だけ、既存の安全規約に従ってHOLDする。
  投影candidateの利用不能だけを理由とする新規HOLDは作らない。
- 新しい物理事実で正味残量や受け側が変わった場合だけ、新しいinput generationを発行する。
- 後から観測した実着弾列は精度検証にだけ使い、過去cutoffの入力へ戻さない。

## 6. 既存実装と不足部分

`src/event_causal_exchange_v1.py`と`src/event_root_family_v1.py`に因果的なroot-familyと相殺後残量、
`src/review_signed_physical_projection_v1.py`と`src/exchange_virtual_board.py`に連鎖後盤面と
初回最大30個の仮想着弾がある。Game43の「P2連鎖中、相殺後108個、初回30個、残り78個」も
回帰試験済みである。

投影した完全盤面と不確実性を保存するDTO、Formal100専用モデル基盤、
offline特徴と同一の`ObservationAdapter`は実装済み。以下の3点もモデル学習前の
構造gateとして実装した。

- 投影対象がactive chain側本人であることを保証し、通常の非発火側投影と分離するgate。
- `first_drop % 6 != 0`を明示的な曖昧理由として保持し、単一疑似seed由来の死亡判定・圧迫度を
  抑止する分岐列挙層。
- active chain側の次の新ツモ設置より初回着弾が先であることを、現在の両側root-familyと
  event順序から判定するgate。時間閾値や未来の観測着弾列には依存しない。

残る未実装はPhase J practical lane接続、正式学習と較正、実機検収である。
新しい物理計算系統を増やさず、既存投影結果の保持と接続で実装する。物理的既知0と未測定は
区別し、相殺後残量0を「安全が観測済み」とは扱わない。board未取得と列不確定も別理由にする。

## 7. 合格試験

- 自連鎖で一部相殺し、同じ側へ残る場合の連鎖後盤面、初回着弾、leftover保存則。
- 完全相殺、余剰による方向反転、左右交換対称。
- 初回着弾0、6の倍数、端数あり、30、31以上の境界。
- 6の倍数では実着弾盤面と完全一致し、端数ありでは実盤面が候補集合内にあること。
- 端数列の候補に死亡／非死亡の両方が含まれるfixtureで、単一seedの死亡判定を公開しないこと。
- 非発火側への予告は`guaranteed_landing_after_active_chain`を成立させず、通常経路へ戻ること。
- 初回着弾が次ツモ設置より先と判定できない場合は
  `first_drop_timing_untrusted`で`guaranteed`を拒否すること。
- anchor複数、leftover不連続、rate/hash/boundary不一致でHOLDとなること。
- 端数列だけの曖昧さでは新規HOLDを増やさず、正式モデル採用前はbaseline値＋診断badge、
  採用後は全候補の等確率平均＋同じ診断badgeになること。
- full replayとprefix replayの同一cutoff出力がbit-identicalで、未来の実着弾列を混入しないこと。
- offline学習特徴とPhase J入力の盤面、列順、dtype、mask、hashが完全一致すること。

本番採用はFormal100の正式学習・外部評価、terminal実機検収、独立レビュー、ユーザー承認後に限る。
`src/production_config.py`は本追補の作成だけでは変更しない。
