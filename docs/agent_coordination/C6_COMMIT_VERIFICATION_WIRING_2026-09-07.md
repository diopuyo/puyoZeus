# C-6 commit verification 配線最小案

調査日: 2026-09-07。本文は設計監査であり、pipeline、flag、snapshot、GPU を変更していない。参照時の `src/recognition_pipeline.py` SHA-256 は `6e945d7584025ae9803e14c9ac079b1a468f483d0beac681a1f0e580cdd10e02`。CPU transaction helper `src/chain_commit_candidate_v1.py` は修正ループ1後 `443b79a896c8e46a268fab8e9f54b0a5238940bf4495706041423ee7de9596c5`（150 tests PASS）で、二段階prepare/validate/finalizeを持つがpipeline未配線である。

## 結論

G2b の transaction helper は「候補を壊さず保持し、世代・underflow・一回性を検査して effect を作る」ところまで完成している。しかし現行 pipeline には、**連鎖が本当に終了したという positive evidence を候補世代へ結ぶ責務がない**。既存の5 STABLE多数決、単一CNN、STABLE遷移、timeout、途中chain count、NEXT/NextSlideのいずれも単独ではcommit権を発行してはならない。

最小の安全案は、C-6を即時書換えからper-side単一transaction生成へ変え、次の2証拠を別々に成立させてから一括適用すること。

1. `board_verified`: 候補世代中のSTABLE観測が候補finalを厳密に支持する。これは盤面一致証拠であり終了証明ではない。
2. `completion_verified`: 同じevent世代で、full simulationの予測総得点まで実測scoreまたはformula累積が到達した。これは終了進行証拠であり盤面一致証拠ではない。

両方と公開解除policyが揃わない限りfail-closedにする。観測不能時はavailabilityを落としてもcommitしない。13連鎖の早期9では盤面だけが一時的にfinalと合う一方、得点/formulaは途中なので止められる。

## 既存資産と不足

### 再利用できるもの

- `ChainEvent` はfrozenだがmutableな`before_board`を含む（`src/chain_detector.py:120-161`）。G2b factoryはbefore/finalをtupleへdeep-copyし、event digest、side、reset epoch、event/action revision、作成frame/timeを固定する（`src/chain_commit_candidate_v1.py:58-128,268-291`）。
- commitは検証policyと公開解除policyの両方が未接続なら失敗し、現在Counterから消去差分だけを引く。underflow、世代不一致、bool承認、callback再入、二重commit/discardを拒否する（同`:295-410`）。
- scoreは各frameで一度読み、current scoreを既に保持する（`src/recognition_pipeline.py:4788-4805,8129-8131`）。Formula accumulatorは`step_count`、`total_power`、`last_valid_t`を公開する（`src/score_ocr.py:1018,1110-1126`）。新しいformula段は継続中の直接証拠で、active eventを更新する既存集約点もある（`src/recognition_pipeline.py:6387-6467`）。
- NEXT確定値とNextSlideは既に取得される（同`:5107-5148`）。ただしNEXT誤解除とNextSlide偽陽性の実測があるため、positive completionではなく世代失効・補助corroborationに限定する。
- 現answer checkはSTABLE CNNを5件集め、許容差以下なら`verified_match`を返す（同`:6697-6744`）。盤面観測の素材には使えるが、13連鎖途中の9-cell false greenを通すため終了証明には使えない。
- 新着地の色Counter反映は`TSUMO_FALL→STABLE`かつpending dequeue時に集約される（同`:7385-7409`）。G2b effectは候補作成時snapshotへ戻さずcommit直前Counterを使うので、同世代中の正当増分を保持できる。

### 現在ないもの

- event開始前後の生score原観測、最初の正の`ScoreDelta.prev_score`と元frame、C-6 full simulateの`chain_count`および`calculate_chain_score(cr).total_score`を同じreset/event/action世代へ結ぶreceipt。
- board一致とcompletion一致を別判定し、両者を同一candidate IDへ結ぶ実policy。
- per-side候補の単一owner、消費済candidate ID registry、reset/resync/new actionでの失効処理。
- candidate final、Counter、constraint、T2/色memory、public boardを同一frameで切り替えるpipeline側のatomic apply。
- mismatch時に現在観測を公開してよいかを判定する「observed recovery」policy。candidateを捨てることと現在CNNを真値にすることは別である。

## 世代と単一所有者

### 初期化

`RecognitionPipeline.__init__`の既存verify state付近（`src/recognition_pipeline.py:1962-1963`）に、次だけをper-sideで持つ。

- `_chain_commit_tx_1p/2p: ChainCommitTransaction | None`
- `_chain_commit_event_revision_1p/2p: int`
- `_chain_commit_action_revision_1p/2p: int`
- `_chain_commit_consumed_ids_1p/2p: set[str]`

reset epochはpipeline全体で単調増加させる。候補自体を外部へ公開せず、生成は1個の内部helperだけを通す。既存pending transactionがある状態で別候補を作る場合、暗黙上書きせず理由付きdiscard→consumed registry登録→新規生成とする。同じcandidateから別transactionを再構成できる現在のG2b API上の余地は、この所有規約とregistryでpipeline内だけ塞ぐ。

### revision更新点

- 新しいtracker event採用: `src/recognition_pipeline.py:4682,4717`
- formula/score pseudo採用: 同`:6039-6055,6314-6326`
- formula段によるevent置換: 同`:6438,6453`
- landing pseudo採用: 同`:7717-7748`

上の全入口を`_register_chain_generation(side, event, frame_idx, time_sec)`へ集約し、event revisionを増やす。sticky/public scoreをentry scoreとは呼ばない。開始後最初の正の生OCR差分が得られた時、その`ScoreDelta.prev_score`と同じ値を最後に直接読めた元frameをanchor候補として一度だけ固定する。reset/action世代と結べないanchorはUNKNOWN扱いで、後段のformula revisionや新しいscoreで上書きしない。stash（同`:6518-6534`）は参照の所有場所が変わるだけなのでrevisionを増やさない。formula段はsessionとstep時刻を保持し、event置換時だけ旧candidateを失効させる。

action revisionはNEXT値だけで進めない。`sm.update`（同`:7332`）で実stateが新たに`TSUMO_FALL`へ入った時に増やし、pending candidateをdiscardする。これなら同色NEXTを見逃してもstate境界で失効し、NEXT一時誤読だけで旧候補を別世代へ誤帰属しない。着地Counter更新（`:7385-7409`）とは別revisionなので、会計増分と盤面世代を混同しない。

### reset/resync

- 公開`reset()`（同`:3943-4130`）で両候補をdiscard、consumedへ記録、reset epochを増やす。epoch変更後にregistryを空にしてよい。
- force-in-matchのscore境界は既に`self.reset()`へ集約される（同`:4807-4868`）。
- match inactiveでCounterをclearする前（同`:4491-4513`）に両候補をdiscardする。
- baseline broken reset（同`:6839-6843`）とdrift resync（同`:8076-8127`）は対象sideだけdiscardし、event/action revisionを増やす。resetを跨いだSTABLE票を混ぜない。

## 候補生成と保留

C-6入口（`src/recognition_pipeline.py:7873-7951`）では、stashをone-shot消費する既存責務を維持し、full simulate後に次だけを行う。

1. finalへgravity filterを適用し、`cr.steps`から色別消去差分を作る。
2. G2b candidateへ、現在side/reset/event/action revision、作成frame/time、effective event、final、差分を保存する。
3. 検証lineage用に`cr.chain_count`と`calculate_chain_score(cr).total_score`を同candidate IDへ別receiptとして保存する。`ChainResult`自身に`total_score`属性はない。途中eventの`chain_count/total_erased`との一致は要求せず、後から得た別prediction revisionを過去frameへ遡及適用しない。
4. `ctx.confirmed_board`、`ctx.pending_board`、`_tsumo_count`、`_constraint_valid`を変更しない。現行の即時更新（`:7907-7949`）は候補生成へ置換する。
5. provisional中の`published_confirmed`と`prob_board`は`None`にし、candidate finalはobservedではない専用estimated provenanceでのみ保持する。前値複製で採録穴を埋めない。

現行のstable color memory更新はC-6判定より前（`:7334-7355`）にある。candidate pending frameを通常STABLE memoryへ混ぜないため、この更新をC-6処理後の共通helperへ移し、commit/verified observed recoveryのときだけ同じ版を記録する。単なる表示None化では内部feedbackを止め切れない。

## 検証policyと公開解除policy

### board verification

既存`_update_chain_estimate_verification`（`:6697-6744`）を材料として再利用するが、`verified_match`をcommit permitへ直結しない。履歴にはcandidate ID、reset/event/action revision、frame、stateを持たせ、非STABLE、新formula段、resync、新着手で履歴全体を失効させる。

最初の安全なshadowでは、可視72cellがUNKNOWNなしで候補finalと一致する連続STABLE consensusを`board_verified`とする。既存の`diff<=6`は診断値として残せるが公開解除には使わない。5件という既存窓はノイズ除去の母数であって終了証明ではない。

row0は画面から直接読めないため、可視72cell一致だけでも、candidate finalのrow0が全EMPTYという自己申告だけでも、全78cellをcommitしない。非循環なEMPTY根拠候補は`build_hidden_row_probabilities`の`HiddenCellSource.GRAVITY_EMPTY`（`src/hidden_row_probability.py:124-160`）。候補finalではなく、同一世代で独立保存した生STABLE観測consensusのrow1を入力にし、row1がEMPTYの列だけrow0 EMPTYを重力で証明する。6列全てにこの証拠がある場合だけ「row0全EMPTY候補」を作れる。`UNINFORMED` fallbackもcertain EMPTYになり得るので`is_certain`だけで承認しない。非EMPTY/UNKNOWNは同action revisionの`infer_hidden_row`等の独立証拠を結べない限りfail-closed。今回receiptはrow1原値・元frame・GRAVITY_EMPTY列maskを保存するが、commit許可にはまだ接続しない。

### completion verification

推奨する最小policy候補は、生OCR anchorとC-6で得た`calculate_chain_score(cr).total_score`を同じreset/event/action revisionへ結び、次のどちらかを要求する。世代receiptが無い段階では数値一致を「候補」とだけ記録し、completion verifiedにしない。

- 通常scoreが生OCRで読め、`current_score - frozen_anchor == calculate_chain_score(cr).total_score`である。
- 同一formula sessionの`step_count == cr.chain_count`かつ`total_power == calculate_chain_score(cr).total_score`であり、その後に同世代STABLE観測へ入った。

いずれも値の増加途中は「継続中」でありpermitを出さない。score reset/wrap、OCR欠測、all-clear持越し等でexact closureを証明できない場合はfail-closedとする。timeout、通常scoreが一度見えた事実、formula不在、途中countとfull countの不一致は終了証明にしない。

NEXT変化/NextSlideは上記closure後のcorroborationまたはaction失効にだけ使う。単独permitは禁止する。特にNEXTが新着手を示した時点でaction revisionが進んでいれば、古いfinalを現在盤面へ遅れて適用せずdiscardする。

### release

`ChainCandidateVerificationPolicy.require_verified`は`board_verified AND completion_verified AND identity exact`だけを受理する。`ChainCandidateReleasePolicy.require_release`は、同frameがreset/resync/new actionでなく、公開side/stateが適用可能で、candidate IDが未消費であることを検査する。bool/tokenを自由生成するAPIは追加しない。

## 一括適用点

既存answer check呼出直後（`src/recognition_pipeline.py:8619-8629`）を唯一のcommit/correction境界候補にする。修正ループ1後helperでは、`prepare`でeffectを作ってもtransactionはPENDING、ownerの同一lock内で実context/Counterと単調`counter_revision`を`validate_prepared`し、外部一括swap成功後だけ`finalize(ticket)`する。外部適用前なら`abort_prepared`で同じtransactionをPENDING再試行へ戻せる。

- `ctx.confirmed_board`、`ctx.pending_board`、`published_confirmed`
- sideの`_tsumo_count`（effectのcommit直前Counter）
- sideの`_constraint_valid=True`
- `_prev_stable_confirmed`、`_stable_color_memory`
- T2に旧値を戻させない一frame marker
- transaction stateとconsumed registry

`_prev_confirmed`はSideResult返却後の既存更新（同`:5358-5360`）に委ねれば同じcommitted boardになる。確率根拠がないため`prob_board`を捏造せずNoneのままにする。重要な境界として、`abort_prepared`は**外部適用前専用**であり、属性を一部書いた後のrollback APIではない。現在helperだけでは外部swap途中失敗をPENDINGへ戻せない。全copy/effectを先に構築し、owner単一参照のswapを例外を起こさない最小操作にしてからfinalizeする。複数mutable objectへの逐次代入案は、適用途中fault injectionと復旧設計が通るまで採用不可。

候補不一致時は`discard`だけを先に行い、現在CNN多数決を直ちにobservedへ昇格させない。`verified observed recovery`を採るなら、completion証拠と厳密board consensusを満たした別effectとして同じ境界で適用し、Counterは増減せず`constraint_valid=False`を維持する。

## ケース別の採否

### 2P 13連鎖・早期9

- 実証済み: frame 34864/66/68/90でSTABLE 9色が公開された一方、実連鎖は約596秒まで継続した。frame 583以後もchain count 4→6→8、score 2311→6151→22571と増え、596秒の実score差は予測79080と一致した。
- 新policy: 581秒付近のboard一致/5 STABLEだけではcompletion未成立なのでcommit禁止。新formula段で旧revisionを失効。真の最終候補がscore/formula closureとfinal consensusを同時に満たした時だけcommit。
- 未成立: candidate IDとreset/event/action世代へ結ばれた生score anchor候補およびformula/score closureの実run receiptはまだない。

### 1P 旧before60→誤final56、実46

- 実証済み: frame 34370のC-6直前ctxは46、stale baseline beforeは60、simulated finalは56。現行は内部盤面とCounterを即更新した。
- 新policy: candidate化だけ行い、56をctx/Counterへ入れない。46支持は56のdiscard材料になり得るが、単一CNNや5票だけで46をobserved公開する根拠にはしない。新action/resyncで旧candidateを確実にdiscardし、消去差分は一度も適用しない。
- 未成立: 46を同frameで公開できるpositive completion evidence。証明できない期間はunavailableが安全側。

### 正常C-6復帰

- 必要条件: 同candidateのscore/formula closure、厳密final consensus、世代一致、未消費。
- 成功時: finalとCounter差分を一度だけ適用し、T2/色memory/public boardを同じ版へ同期する。
- 回帰gate: 既存の正常final/answer-check tests（`tests/test_recognition_pipeline.py:2850-2992,4321-4412`）を、保留→commit時刻、公開遅延、両sideへ拡張する。

### 次手増分

- 同action revision中の正当なCounter増分が存在しても、候補作成時snapshotへ戻さずcommit直前Counterから差分だけ引く。G2b CPU testで成立済み。
- 新しい`TSUMO_FALL`へ入った後は旧candidateをcommitせずdiscardする。次手着地のCounter増分（`:7398-7401`）を旧消去と混ぜない。既存lean delta drainで架空`on_tsumo_settled`が0であることをpipeline統合testでも維持する。

### reset/resync

- reset epochまたはside revision不一致でcommit拒否し、Counter/boardを不変にする。discard receiptを残し、次試合の候補とはIDを共有しない。
- explicit reset、score-reset、match inactive、baseline reset、drift resyncをそれぞれfault-injection testする。

## 追加計装は一点だけ

同じ診断を反復せず、次のshadow前に**candidate completion lineage receipt**だけを追加する。candidate/prediction ID、side/reset/event/action revision、生score値と元frame、最初の正の`ScoreDelta.prev_score` anchor候補、`cr.chain_count`と`calculate_chain_score(cr).total_score`、formula session/step時刻と`step_count/total_power/last_valid_t`、state、accounting前raw STABLE consensusのdigest/coverage/row1/GRAVITY_EMPTY列mask、NEXT/slide、失効理由を時系列で記録する。欠測は0やPASSにしない。異なる時点のscore exactとboard consensusを単純ANDせず、同一世代かつexact到達後の観測と結べるまでは未判定とする。

これを13連鎖の560–605秒、1P旧56を含む既存control窓、正常C-6対照へ同じ実装で当てる。目的はscore/formula closureが実際に正常例を通し早期9を止めるかの一回の判定であり、CNN再診断や既存C-6 lineageの再収集ではない。

## 実装可能な部分と親判断

今すぐ実装可能なのは、per-side owner/revision/registry、C-6候補化、全reset/resync/action失効、pending公開隔離、lineage receipt、G2b effectのfault-injection統合testである。これらは終了を推測せず安全側へ倒せる。

親判断が必要なのは次の2点だけ。

1. **availabilityとの交換条件**: score/formula exact closureを証明できない候補は最後までunavailableのままdiscardする厳格案を採るか、別の物理的終了観測器を追加するか。現情報だけで安全な緩和条件はない。
2. **mismatch後の公開**: candidateをdiscardした後、completion＋厳密consensusを満たす現在観測を`constraint_valid=False`で公開する別effectまで同時に実装するか、次の通常着地/resetまでunavailableを維持するか。誤finalを止めるだけなら後者が最小、復帰品質まで含めるなら前者が必要。

この文書は配線案であり、G2全体CLEAR、本番採用、13連鎖/旧56の実動画修復を宣言しない。
