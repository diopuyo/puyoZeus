# Phase J 死亡確認・打ち合い物理投影 追加仕様

## 1. 目的

既存レビューと Phase J で、次の2問題を同じ因果契約で解決する。

1. 静的な死亡セル候補が新しい予測行ごとに再発生し、判定保留が1.5秒を超えて延命される。
2. 打ち合いの相殺後残量が物理的に計算できても、発火前盤面へ凍結された統計モデルだけが表示され、勝率が低く見積もられる。

本仕様は既存の学習済み重み、`src/production_config.py`、旧相互交換器、攻撃差10%補正を変更しない。

## 2. 不変条件

- 0% / 100% は、左右の勝敗ロゴを2フレーム間隔で2回同方向に確認した
  `visual_result_logo_bilateral_2x2` だけが出力できる。
- 死亡候補、時間的死亡確認、攻撃量、相殺後残量、片側ロゴは0% / 100%の根拠にしない。
- `confirmed_board` と既存45指標は、両者 STABLE のときだけ更新する。
- 物理計算後の盤面は `confirmed_board` へ書き戻さず、immutable な `projected_state` とする。
- `projected_state` 専用モデルと較正資産が無い、古い、支持外、またはhash不一致なら数値を更新せずHOLDにする。
- 未来の確定攻撃、公式勝者、試合後盤面をcutoff以前へ逆流させない。
- 新経路は既定OFFとし、旧成果物を上書きしない。

## 3. 死亡確認契約

### 3.1 観測源

`DeathConfirmTracker` を認識pipelineの外部wrapperから毎観測フレーム1回だけ呼ぶ。
入力は次に限定する。

- side別 `BoardState`
- STABLE時の `confirmed_board`
- side別 `next_pair`
- `is_match_active`
- 両sideの得点表示が読める既存causal信号 `_physical_match_evidence`
- 正式な `game_idx`

`death_margin`、`cnn_board`、`estimated_board`、直前STABLE盤面の独自再判定は使わない。
死亡セル占有は候補発生の入力であり、死亡確定や勝敗確定そのものではない。
`_physical_match_evidence=False` の間は新規候補と初回armを禁止する。既存pendingは
解除せず、state/NEXT不動clockを再開後に測り直す。`game_idx==0`単独のhard gateは、
途中開始動画の初戦を失うため使わない。

### 3.2 状態

side別状態は次のenumで持つ。

```text
clear | pending | released | confirmed
```

遷移は次に限定する。

```text
candidate_placement | candidate_ojama
released_placement | released_ojama
released_survival_placement | released_survival_ojama
confirmed_placement | confirmed_ojama
```

`death_generation` はcandidate、release、confirm、正式境界の原子的な状態変化時だけ増やす。
単なる観測フレームでは増やさない。

### 3.3 1.5秒の意味

1.5秒はHOLDの最大時間ではない。有効なNEXTが候補発生後も不動だった場合の最短確認時間である。
NEXT欠測時は1.5秒を超えても確定しない。own chain開始、NEXT変化、新ツモ開始は
生存証拠として即releaseする。後続STABLE `confirmed_board`の死亡セル空化だけでは
releaseしない。実測132.900秒で死亡演出中の傾いた盤面をSTABLE空盤面と誤認し、
132.467秒の真death candidateを誤解除したためである。確定後は同一gameの偽state遷移で
新candidateへ降格せず、正式境界または検証済み新試合開始までstickyに保つ。

通常表示の古い数値を隠すtimerと、死亡証拠のpending時間は別clock・別fieldで管理する。

### 3.4 正式境界とEOS

両sideの通常処理が終わった後、`game_idx` の `old -> old + 1` を1回だけ観測する。
境界時は加算前の `old` を渡して `resolve_boundary_confirmations()` を左右まとめて1回呼び、
その後に現在フレームを新gameへ入力する。片sideずつの境界処理は禁止する。

EOSは正式境界ではない。EOS時のpendingは未解決のまま保存し、死亡へ昇格させない。
decode失敗と要求区間完走も別状態として記録する。

### 3.5 レビューV5表示

新レビュー契約V5は、外部manifestに固定された期待SHAと一致するdense death sidecarを必須とする。
rendererは2Hz予測とは別にdense timelineを進める。

- `pending`: 数値を表示せず `terminal_confirmation_pending`。
- `confirmed`: 数値を表示せず、左右ロゴによる勝敗確認待ち。
- `released`: 1観測intervalだけrelease前の予測値を遮断し、同じ `death_generation` の新予測を待つ。
- `clear`: 他の品質・物理gateを評価する。
- 正式境界: 前gameの値、candidate、generationを即時破棄する。

V5では保存済みの静的 `death_confirmation_pending` と `death_candidate_p1/p2` を死亡根拠として再利用しない。
予測の `online_segment_index` と、その予測時点のdeath sidecar `game_idx` は一致必須とし、
旧gameの遅延予測へ新generationを後付けしない。sidecar欠落、範囲不一致、game不一致、
SHA不一致、EOS不正はlegacy表示へ退避せず停止する。
旧V1〜V4は互換確認専用として隔離し、新成果物で選択しない。

## 4. root-family契約

### 4.1 familyの単位

物理rootと会計childのordinal一致は仮定しない。family IDは `root_event_id` とする。
committed batch全体を原子的に読み、同じsideの新規childとrootを次の順で割り当てる。

1. 新規childと同一batchに同じsideのrootが1件なら、そのrootへ割り当てる。
2. 同一batchにrootが無ければ、そのsideの直前current rootへ割り当てる。
3. 一度割り当てたchildを後続rootへ移動しない。
4. batch処理後に新rootをcurrent rootとする。
5. 正式境界でcurrent rootとfamilyを破棄する。

同一batchではchildのevent seqがrootより小さい場合があるため、単純なseq比較でfamilyを決めない。
side内複数root、root欠落、重複所属、boundary衝突、source/build/segment跨ぎはHOLDとする。

### 4.2 発火前盤面anchor

anchorはroot単位でなく、family内の各新規child単位に選ぶ。条件は次のすべてである。

- 同一source、build、online segment、side。
- child初出batchより前のcommitted batch。
- `board_provenance=observed` かつ assertion=`confirmed/exact`。
- unknown cellが0。
- boardのavailable frameがchildを含むcommitted batchのavailable frameより前。
  `occurred`は事象の物理発生時刻であり、観測可能時刻としてanchor選択に使わない。
- anchorからchildまでに正式境界、data gap、資産変更が無い。

### 4.3 cutoff

`CausalCutoff` は次を固定する。

```text
source_video_id / build_id / attempt_id / boundary_segment
availability_batch_id / through_sequence / available_frame / available_ms
```

cutoffはcommitted batch末尾だけを許す。partial batch、逆順、重複、cutoff後訂正の混入は拒否する。

## 5. signed物理投影

### 5.1 計算

符号は1P有利を正、2P有利を負とする。

```text
family_observed = observed_family_p1 - observed_family_p2
outside_family  = global_observed_balance - family_observed
pure_family     = simulated_family_p1 - simulated_family_p2
projected_net   = outside_family + pure_family
```

root-familyの観測寄与を一度引き、物理simulate結果を一度だけ足す「置換」である。
観測childへsimulate値を加算して二重計上してはならない。暫定から確定への移行も加算でなく置換する。
同じsideの得点→おじゃま換算では、直前に確定した `leftover_after` を次childの
`prev_leftover` として因果順に継承する。boundaryでは0へ戻す。初期値、順序、
`leftover_before/after` の連続性を証明できなければ、exactな投影量を出さずHOLDとする。
途中cutoffだけを抜いた監査fixtureでは、最初の確定childが持つexactな`leftover_before`を初期anchorにできるが、
正式boundaryを観測した後は必ず0を期待値とし、非0 payloadによる上書きを許可しない。
同一prefire anchorが複数childへ重複した場合は同じ物理連鎖の二重simulateを避けるため、投影量をHOLDとする。

`projected_net > 0` は2Pへの予告、`projected_net < 0` は1Pへの予告である。
受け側盤面への着地は既存 `land_pending_ojama_onto_board()` だけを使い、1ターン30個上限とleftoverを分離する。
旧 `resolve_mutual_exchange()` は使わない。

受け側が連鎖中なら、STABLE凍結中の `a_p1_grid` / `a_p2_grid` へ直接着地させてはならない。
その連鎖のtrustedなprefire anchorを `ChainSimulator` で解決した `final_board` を着地元にする。
複数childの時間順やactive familyとの対応が一意でない場合は、死亡可否を推測せずHOLDとする。
受け側が連鎖中でない場合だけ、同じcutoffのSTABLE確定盤面を着地元にできる。
`recipient_dead_after_first_drop` はsoft投影の診断値であり、単独では0/100の根拠にしない。

### 5.2 Game43固定oracle

Game43はheldout回帰専用であり、特徴、cap、anchor条件、閾値、支持範囲の選択に使わない。

```text
P1 family: root219 <- child322 + child324
P2 family: root221 <- child325
family外:  child323 は直前root220に所属

global observed balance  = +173
family observed balance  = +176
outside family           = -3
simulated P1/P2          = 1200 / 1089
pure family              = +111
projected net            = +108
recipient                = p2
drop / leftover          = 30 / 78
P2 active-chain final    = 13連鎖後8個、死亡セル空
after first drop          = 38個、is_dead=False
```

旧診断の `recipient_dead_after_first_drop=True` は、連鎖中に凍結した69個の発火前盤面へ
30個を着地させた結果であり不採用とする。Game43で物理的に確定しているのは
`P2へ108個、初回30個、leftover 78個` までで、勝敗そのものではない。

### 5.3 fail-closed条件

次は診断値を保存しても、数値表示へ採用しない。

- root、child、anchor、rate、all-clear/leftoverの欠損または不一致。
- unknownを含むanchor、non-STABLE anchor、古すぎるanchor。
- family開始後の着地をfamily内外へ一意帰属できない。
- provisional/finalizeの訂正関係が曖昧、rateが途中で変化、oversettled。
- causal replayがquarantine中。
- source、build、attempt、cutoff、input、model、calibrationのhash不一致。
- projected-state専用モデル未較正または支持外。

未較正中は既存確率と既存特徴をbit-identicalに保ち、
`soft_physical_projection_uncalibrated` と計算済みの受け側・drop・leftoverだけを表示する。

## 6. 学習と評価の分離

- fit、feature selection、cap、support、Platt較正は固定48本だけを使う。
- 既存6-fold割当を変えず、seedは `20260831, 20260832, 20260833` とする。
- eval fold、tune fold、残りtrain foldのsource group交差を0件とする。
- heldout 57試合とGame43は最終評価だけに使う。1件でもfit/tune/support選択へ混入したら停止する。
- 追加特徴は左右対称pair `projected_pending_to_p1_norm` / `projected_pending_to_p2_norm` とmissing flagを初期候補とする。
  受け量 `x` は物理定数である1ターン最大落下量30を尺度として `x / (x + 30)` で0〜1化する。
  片側family欠損を既知0にせず、両側量を欠測、missing flagを1としてfail-closedにする。
- 左右反転時は入力を交換し、予測を厳密に `1-p` とする。

採用には全seedで全体log loss改善、cluster bootstrap 95%上端0以下、fold改善、Brier・高確信逆判定・局面別非劣化、
HOLD時間非悪化、heldoutで強制0/100が0件、defender-chain中の重大誤解除0件を要求する。

### 6.1 固定48本の結論と100本超の事前登録

2026-09-03の固定48本では、正味量4列の全体再fit、baseline固定weight=1のlogit残差、
初回着地後死亡列圧迫の順に一回ずつ検証し、全て不採用とした。着地後圧迫の修正版v2は
supportと方向監査に合格したが、raw/calibratedのlog-lossとBrierが全3seedで悪化した。
したがって同じ48本で特徴、係数、cap、閾値を追加探索せず、heldoutとGame43も使用しない。

着地盤面が曖昧でも受け側が一意なら、受け側pressureだけをmissing、非受け側pressureを既知0とする。
root-family自体が利用不能で受け側も信用できない場合は両側missingとする。

100本超での次候補は、一次元の死亡列圧迫ではなく、次を入力するprojected-state専用モデルに固定する。

- active chainを解決した受け側の完全盤面。
- 初回着地後の完全盤面。
- 初回drop量と残り着地予定量。
- 入力cutoff、root-family、盤面、モデル、較正のprovenance/hash。

候補のfit前にsupportと評価gateを固定し、同じ100本超集合で失敗後の再探索をしない。
未較正・支持外・hash不一致では既存勝率を変更せず、物理診断だけを理由付きで表示する。

## 7. 成果物と非上書き

新規成果物は別名rootへ排他的に作る。各成果物はsource SHA、sidecar SHA、code/asset hash、cutoff digest、
model/calibration ID、入力行数と除外理由の母数、manifest、validation、COMPLETEを持つ。

旧レビュー動画、旧予測Parquet、旧event run、既存NPZは変更・削除しない。新しい動画は
`D:\puyo_analyzer\videos\review\` 配下へ保存し、削除しない。

## 8. 必須回帰

- 死亡: 1.499秒pending、1.5秒confirm、NEXT欠測、NEXT変化、新ツモ、own chain、
  死亡演出中STABLE空化での誤解除防止、confirmed降格防止、左右同時候補、正式境界、EOS。
- 試合gate: 得点不可視中の偽設置/おじゃま遷移を候補化しない、pending中の一時欠測を
  即releaseにも不動時間にも使わない、game_idx=0の途中開始は得点可視後に追跡する。
- 表示: release後の旧値非表示、generation一致後だけ再表示、confirmed単独で0/100なし、左右ロゴだけ0/100。
- family: child-before-root同一batch、複数child、後続rootへの誤移動禁止、boundary reset、重複拒否。
- 物理: Game43の `+111 -> +108 -> drop30/leftover78`、13連鎖後盤面ではdeath false、
  左右反転、保存則、未来訂正非参照、連鎖中の凍結盤面へ着地しない。
- 互換: flag OFF、sidecarなしの旧経路、既存確率、既存重み、production設定がbit-identical。
- 完全性: sidecar/manifest/COMPLETEの改ざん、範囲不一致、source/build不一致をfail-closedで拒否。
