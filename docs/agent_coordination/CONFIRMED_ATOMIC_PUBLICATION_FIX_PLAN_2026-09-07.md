# 即時連鎖の原子的公開 修正計画（2026-09-07）

## 結論

### 14:48 実測による更新（以下の静的仮説より優先）

`video38_confirmed_collapse_2026-09-07_v3_details` は690 frameを103.845秒で完走した。
追加計装以外の1,527行がv2と文字列まで一致し、計装による挙動変化はなかった。
frame34704では、4連結ゲートへ到達する前に game-event NEXT 終了判定がactiveを消した。
新landing pseudoのtriggerは578.3667秒だが、start_next=[2,2]とentry_t=570.9333秒が
前イベントのまま残り、current_next=[5,5]によりpipeline:5193でstash/clearされた。
したがってFableの「消去済みfinalが4連結ゲートで自己拒否される」は構造上の別リスクであり、
この実フレームの直接原因と断定しない。古いNEXT開始情報の更新も独立した修正軸とする。

最初のshadowは因果を分けるため、`resolve_after_placement`のchain_count>0時の返却盤面を
`new_confirmed`（連鎖前の推定着地盤面）のcopyへ差し替えるだけとする。主・副の両call siteに
同じ契約を適用するが、chain_count、pseudo登録、会計、C-6、拒否時rollbackは変更しない。
以下の「両盤面rollback修正」は最終候補の必要条件であり、この一軸shadowには含まれない。

現行の即時着地連鎖経路には P0 のデータ整合性欠陥がある。`TSUMO_FALL -> STABLE`
となった同じ frame で、まだ画面観測していない連鎖完走後 `final_board` を
`STABLE / observed` の確定盤面として公開している。さらに次 frame の既存
4連結ゲートは、その消去済み `final_board` を検査するため、自分で作った
landing pseudo chain を拒否し得る。

最初の修正候補は、公開盤面だけを既存責務へ戻す最小差分とする。

- 着地 frame は連鎖前の `inferred_landing` を `confirmed/pending` に置く。
- 連鎖完走後 `final_board` の確定は、既存の CHAIN/GRAVITY_SETTLE 終了処理
  （C-6）だけに任せる。
- 大差分による pseudo 拒否時は、この即時commitで触った `confirmed` と
  `pending` の両方を直前盤面へ戻す。
- 同 frame の `chain_event`、外側会計、`ctx.state` の強制更新は今回に混ぜない。

これは直ちに本番採用せず、frame 34702 を含む shadow 差分で pseudo 自体の
正当性を確かめてから採否する。`src/production_config.py`、モデル、dataset、
学習成果物は変更しない。

## コード上で確定した原因

### 主経路

1. `src/recognition_pipeline.py:7322-7332` は state machine 更新前の
   `prev_state/prev_confirmed` を保存する。
2. `src/recognition_pipeline.py:7357-7377` は `_effective_chain_event` をこの時点で
   一度だけ確定する。着地処理で後から作る pseudo はここへ反映されない。
3. `src/recognition_pipeline.py:7669-7675` は `resolve_after_placement()` が返す
   連鎖完走後 `final_board` を、pseudo の採用より前に `ctx.confirmed_board` と
   `ctx.pending_board` へ書く。
4. `src/recognition_pipeline.py:7697-7748` は landing pseudo を
   `_active_chain_{1p,2p}` に登録するだけで、同 frame の `ctx.state`、引数
   `chain_event`、`_effective_chain_event` は変えない。
5. `src/recognition_pipeline.py:8502-8506` はその盤面を `_prev_stable_confirmed_*`
   に保存し、`src/recognition_pipeline.py:8523-8658` は
   `STABLE / confirmed=final / board_provenance="observed" / chain_event=None`
   として返す。

次 frame では `src/recognition_pipeline.py:4750-4780` が active pseudo を初めて
`chain_ev_*` に載せる。しかし `src/state_detectors.py:146-188` の
`_passes_erasable_gate()` は `ctx.confirmed_board` の4連結を検査する。
ここが既に消去済みfinalなら、UNKNOWNが3個未満の通常条件でゲートは拒否する。
landing mechanism は formula-read の既存bypass対象でもない。このため
「finalを先取りした結果、自分のpseudoを拒否する」という循環になる。

Fable独立監査結果は
`logs/fable_confirmed_collapse_2026-09-07_v1.json`。snapshot/current の
`recognition_pipeline.py` は同一SHAであり、この経路は両方に存在する。

### rollback不整合

`src/recognition_pipeline.py:7682-7696` の追加セル数ガードは、拒否時に
`confirmed_board` だけを `prev_confirmed` へ戻す。先にfinalを入れた
`pending_board` は戻らない。なお `resolve_after_placement()` 自体も
`src/placement_inferrer.py:849-863` で「総非空セル数差 > 6」または
`score_delta_observed == 0` を chain_count=0 にする。一方pipeline側は
「EMPTY/UNKNOWNから非空へ増えたセル数 > 6」を見るため、削除と追加が同時に
ある場合は両ガードが同値ではなく、pipeline側rollbackは到達可能である。

ここで要求する `confirmed == pending` は、今回の即時置きcommit直後と
拒否rollback直後だけの局所契約である。STABLE確定待ち・多数決投票中まで
常に両者同一とする一般契約ではない。

### 同型の副経路

`src/recognition_pipeline.py:7782-7860` の TSUMO_FALL 取りこぼし補強経路も、
`resolve_after_placement()` の `chain_count` を捨て、`final_b` を
`confirmed/pending` へ先取りする。ChainEventを作らないため、主経路より直接的に
未来盤面を `STABLE / observed` として公開する。

## 最小 shadow 差分

### A. 今回の主候補（公開盤面だけを直す）

`src/recognition_pipeline.py:7669-7748` の計算順とpseudo生成条件は維持し、commit
する盤面を次のように変える。

1. `resolve_after_placement()` は従来どおり呼び、`final_board/chain_count` を得る。
2. `ctx.confirmed_board` と `ctx.pending_board` には `final_board` ではなく
   `inferred_landing` とそのcopyを入れる。
3. 追加セル数ガードがpseudoを拒否したら、両方を `prev_confirmed` の独立copyへ
   戻す。`_active_chain_*` と chain estimate は作らない。
4. 正常なpseudoなら従来どおり `_active_chain_*` へ登録する。同 frame の返却
   `chain_event=None` と state遷移順は変えない。
5. 次 frame、既存 `ChainPhaseDetector` は連鎖前 `inferred_landing` の4連結を
   検査できる。通れば通常の CHAIN -> GRAVITY_SETTLE -> STABLE を通り、
   `src/recognition_pipeline.py:7873-7908` の既存C-6が
   `ChainEvent.before_board` を再simulateしてfinalを一度だけ確定する。

この候補では、frame 34702 の9cell future-final漏出は止まる。一方、そのpseudoが
本当に13連鎖だったかは別問題である。修正により、従来は自己矛盾ゲートで止まって
いたpseudoが次 frameから有効になる可能性があるため、下記shadow gateを通すまで
本番へ入れない。

### B. 副経路

`src/recognition_pipeline.py:7854-7860` も shadow では `final_b` を公開せず、
`inferred_b` とそのcopyだけをcommitする。ここで新しくpseudo生成や会計通知を
足すのは別変更とする。既存 VideoChainTracker / formula / score経路が後続CHAINを
検出し、C-6でfinalを確定できるかを実動画で確認する。検出できない正当chainが
あれば、その証拠をもとに副経路専用のイベント生成を別途設計する。

### C. 今回は行わない変更

- `ctx.state = CHAIN` の直接代入
- 同 frame の `SideResult.chain_event` をpseudoへ差し替えること
- `_effective_chain_event` の再計算
- p1/p2処理後に外側 `chain_ev_*` を返却eventで更新すること
- all-clear、攻撃、OJAMA、M1因果台帳を同 frame に進めること
- landing pseudo用のproduction flagや閾値追加

理由は、公開盤面のfuture leakとイベント会計の確度が別契約だからである。
landing pseudoは `mechanism="landing"` と `score_estimated` を持つ
（`src/recognition_pipeline.py:7697-7733`）が、本番既定の
`enable_pseudo_chain_score_fill` はFalse
（`src/recognition_pipeline.py:1368`, `scripts/visualize_advantage_overlay.py:8467`）。
それでもoverlay側 `_chain_event_gen_ojama()` は `total_score=0` のイベントを
`before_board` から再simulateして火力化する
（`scripts/visualize_advantage_overlay.py:322-345`）。score +1で推定13連鎖という
今回候補を同 frame会計へ昇格すると、偽火力をM1因果台帳へ混ぜる恐れがある。

イベント同期を将来行う場合は独立差分とし、少なくとも mechanism、
score_estimated、物理4連結、次frameの画面/score/formula corroboration、
ledger入力前のprovisional/observed区別を契約化して別回帰を通す。

## 既存テスト資産と不足

- `tests/test_placement_inferrer.py:368-425`
  - `resolve_after_placement()` の大差分拒否、通常+1 chain、閾値ちょうどを単体確認。
  - `ctx.confirmed/pending`、active pseudo、公開stateまでは見ない。
- `tests/test_recognition_pipeline.py:692-703`
  - 即時landing pseudoのhold式だけを確認。docstringにも
    TSUMO_FALL -> STABLEのfull integrationが無いと明記されている。
- `tests/test_recognition_pipeline.py:2884-2931`
  - active chainのstash/clearと1P/2Pを確認できる。
- `tests/test_recognition_pipeline.py:2934-3010`
  - `_force_confirmed_board()` と、別経路の
    CHAIN -> GRAVITY_SETTLE -> STABLEでC-6 finalを検証する統合テスト。
  - 新テストのfixture/完走確認に再利用できる。
- `tests/test_recognition_pipeline.py:4178-4245`
  - CHAIN中の `estimated_board` と `board_provenance`、confirmedとの分離を確認。
- `tests/test_formula_value_read.py:313-370`
  - `_passes_erasable_gate()` の「消去可能盤面なら通る／消去済み盤面なら拒否」
    を組み立てる既存fixtureがある。

不足は「TSUMO_FALL -> STABLE -> landing pseudo登録 -> 次frame CHAIN」を一続きで
駆動し、同時にconfirmed/pending/provenanceを確認する統合テストである。

## 追加する回帰テスト

実装時は `tests/test_recognition_pipeline.py` の既存 `_make_pipe*`、
`_force_confirmed_board()`、`_StubChainTracker` を再利用する。必要なら
`infer_placement` の戻りだけをmonkeypatchし、state machineと実
`ChainSimulator` は可能な限り本物を使う。

### 1. 正常landing pseudo（1P/2P parameterize）

Given:

- state machineを `TSUMO_FALL`、confirmedを連鎖前盤面にする。
- CNN/slideで同 frameにSTABLE復帰させる。
- `infer_placement` は1ツモ追加で4連結が成立する盤面を返す。
- `score_d_for_self > 0` とし、実 `resolve_after_placement` が
  `chain_count >= 1` とfinalを返す。

Then（着地frame）:

- `_active_chain_{side}` はlanding mechanismで作られる。
- `result.state == STABLE` と `result.chain_event is None` は今回変更しない。
- `result.confirmed_board == inferred_landing` で、simulated finalとは異なる。
- `result.board_provenance == "observed"` だが、それは未来finalでなく連鎖前盤面。
- `ctx.pending_board == ctx.confirmed_board == inferred_landing`。
- 反対sideのactive/state/boardは不変。

Then（次frame）:

- `_passes_erasable_gate()` が連鎖前盤面を見て通り、stateはCHAINになる。
- `estimated_board` はsimulated final、provenanceは既存
  `chain_estimate*` のいずれか。
- CHAIN/GRAVITY_SETTLE終了後、既存C-6が一度だけfinalをconfirmedへ入れる。

### 2. pseudo拒否rollback（1P/2P parameterize）

総非空セル差は6以下だがEMPTY/UNKNOWN -> 非空の追加が7個超になる
（同時に旧セルが減る）盤面を用意し、placement側guard通過・pipeline側guard拒否を
実際に通す。困難なら `resolve_after_placement` だけをmonkeypatchして
`distinct_final, chain_count=13` を返し、追加セル数判定は実コードを通す。

Then:

- `confirmed_board` と `pending_board` は両方 `prev_confirmed` と値一致し、
  simulated final / inferred hallucinationのどちらも残らない。
- 両Boardは同一object共有ではなく、後段mutationで相互汚染しない。
- `_active_chain_{side}`、chain estimate、返却chain_eventは作られない。
- 反対sideは完全不変。

### 3. chainなしの通常着地（1P/2P parameterize）

非連結の正当な+2 placementを返す。`chain_count == 0` のとき、従来どおり
`STABLE / observed / confirmed=inferred / pending=inferred copy / active=None` であることを
確認する。future leak修正が通常の置き確定を止めない陰性対照である。

### 4. 既存の正当chainとC-6

既存 `test_gravity_settle_to_stable_applies_final_board_via_stash` と
chain estimate統合テストを残し、landing起点版を追加する。次を固定する。

- pseudoの `before_board` は `inferred_landing` とbit-exact。
- 次frameのCHAIN突入、chain_count、hold終了時刻はshadow前後で一致、または
  現行の自己拒否だけが解消した説明可能差分である。
- C-6 finalは `ChainSimulator.simulate(before_board).final_board` とbit-exact。
- C-6の既存副作用（constraint再有効化、tsumo_count減算、事後verification）は維持。

### 5. 副経路 7854-7860

STABLE -> STABLEの隣接2cell追加をfixture化し、連鎖あり／なしを各1件作る。
着地frameに `final_b` がobserved公開されないこと、chainなしでは従来値一致、
chainありでは後続の既存検知がCHAIN/C-6へ到達するかを確認する。到達しなければ
副経路のevent生成は今回へ黙って混ぜず、未実装として差し戻す。

### 6. event/ledger非変更ロック

最小候補では着地frameの次を現行と同じに固定する。

- `SideResult.chain_event is None`
- `_all_clear_pending` の `chain_fired` はpseudoを見ない。
- 外側constraint invalidation、OJAMA会計、攻撃台帳へpseudoを即時通知しない。

翌frame以降は既存active-chain経路の挙動であり、frame 34702のpseudoが正当と確認
できた場合に限って差分を許容する。将来の同frame同期案には別テストと別採否を要求する。

## shadow差分の検査手順

### Step 0: baseline receipt

対象動画、境界、source SHA、snapshot/current SHA、production config SHA、実行CLI、
CUDA/environment fingerprintを固定する。baselineとcandidateは同じsource frame列を読む。

### Step 1: 診断列

少なくとも各frame/sideについて次をJSONLへ出す。動画や本番成果物は上書きしない。

- frame/time/side、prev_state、returned state
- prev/inferred/resolved-final/published confirmedの各SHAと非空cell数
- confirmed/pending一致、追加cell数、chain_count、score delta
- pseudo mechanism/score_estimated/total_score、active chain有無
- 次frame `_passes_erasable_gate` の結果とerasable group数、UNKNOWN数
- returned chain_event、estimated board SHA、board provenance
- all-clear、constraint、OJAMA/攻撃/因果台帳のdelta

### Step 2: frame 34702実再現

`video_38` の560--583秒（保存報告のframe 34702を含む）をbaseline/candidateで同一再走する。

合格条件:

- candidateの着地frameから9cellの `STABLE / observed` 行が消える。
- candidateの同frame confirmedはraw観測でなくても、少なくとも
  `inferred_landing` とbit-exactで、simulated finalとは不一致。
- 次frame gateの入力SHAが `inferred_landing` と一致する。
- 13chain pseudoが生映像の消去、score推移、formula、raw puyo減少のどれで裏付くかを
  明記する。裏付け不能ならproduction不採用。
- event1333の持続崩壊が解消し、別時刻への単なる移動でない。

Fableの「66+2系」は期待候補であり、GPU捕捉の実値が確定するまで固定値の合格条件には
しない。

### Step 3: 陰性対照とside対称性

- 本物のlanding即時chain 2件以上（1P/2P各1）
- chainなし通常着地 2件以上（1P/2P各1）
- 大差分pseudo拒否 2件以上（1P/2P各1）
- 既存formula/VideoChainTracker起点chain

各窓は発火前STABLEからC-6後STABLEまで全時間帯を比較する。スポットframeだけで
合格にしない。

### Step 4: 全域無悪化

同一動画全編で次をbefore/after集計する。

- `STABLE` かつ confirmed cell数がrawの1/3未満の件数/母数/最長連続長
- `STABLE / observed` なのにpublished SHAがresolved-final SHAとなる件数
- CHAINイベント数、mechanism別数、chain_count分布、発火/終了時刻差
- gate accept/reject数、C-6適用/verification mismatch数
- confirmed/raw agreement、UNKNOWN、空盤面、state占有時間
- provisional/observed攻撃、OJAMA、M1因果台帳の各delta

さらにcanonicalで問題行を含む5行（既報 global 719--723）を再生成し、入力SHAと
全列差分を記録する。変更行だけ新canonicalとして版を分け、旧dataset/checkpointと
無検証混合しない。

## 採否gate

### 最小候補を採用可能

- frame 34702のfuture-final漏出が0。
- 正当chainは次frame gateを通り、C-6 finalまで完走。
- chainなし着地と反対sideが無悪化。
- 拒否時のconfirmed/pendingにfuture-finalが残らない。
- 13chain pseudoが実映像・score/formula/raw減少の複数証拠で支持される、または
  誤pseudoを別の既存根拠で安全に拒否できる。
- 同frame event/ledger同期は変わっていない。

### 差し戻し

- 9cellが別のSTABLE frameへ移動しただけ。
- inferred landing自体がraw/prev/next/物理配置と不整合。
- pseudoが偽なのに、修正で翌frameからCHAIN/攻撃台帳へ昇格する。
- C-6 finalが未到達、二重適用、または既存verificationを迂回する。
- 1P/2Pで結果が非対称。

### event/ledger同期拡張を検討可能

最小候補が合格した後だけ、別差分として検討する。landing pseudoを
「物理予測/provisional」と「scoreまたはformulaで観測確定」に分け、M1因果台帳が
前者をobserved attackへ昇格しないことを先に契約化する。現時点ではユーザー判断を
求める段階ではなく、shadow証拠不足が唯一の保留理由である。

## 検収判定

**条件付き**。原因、最小責務境界、既存テスト再利用先は確定した。source、flag、
production設定はまだ変更しない。追加GPU捕捉とFable所見を入れたshadow比較が上記gateを
満たして初めて、コーダへ最小実装を渡せる。
