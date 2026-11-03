## v2.1 — 公開補正の分離と先行NEXTの着地対応（2026-10-14）

起点 d9073afb。既定OFF・未採用を維持。pushなし。
本節を `<work>/ROW1_GUARD.md` にも追記する。

### 分解で確認した経路

PC2 elのinternal_historyを全件読んで比較した（early 2520、game36 5280コマ）。
`docs/ROW1_V21_decomposition.json` が再現可能な分解票。

- `ctx.confirmed_board` と `prev_stable_confirmed` の差は0。ただしこれは公開盤一致の意味ではない。
  SideResult.confirmed_boardにはearly 40側、game36 251側の差がある。
  それをcycle71dがprev_confirmedへ保存し、次の掛け算式発火の起点盤へ戻していた。
- 具体例はf66037/2P。formula_readのChainEvent.before_boardのr1c2がOFF緑3/ON空0。
  game36ではchain_eventが137コマ、estimated_boardが127コマ異なる。
  inferred_board・landing_diag・state・NEXT・CNNの差は0。
  elにはtsumo_count自体の原票がなく、手数一致をこの票だけで断定しない。
- 公開盤はB3のObservationCapture、RecordObserver、LiveQueuesにも直接入っていた。
  起点盤・連鎖予測、NEXT採用行とその時刻、盤面変更による推論更新契機、表示保持/EMAに
  過去の差が残り得る配線である。現在の盤面が戻っただけでは履歴は戻らない。
- f70721では、prob_boardを除いたSideResult全フィールドがOFF/ON一致しているのに、
  B3生予測は **31.690366235% / 87.658914913%**。
  次の推論はf70735なので、f70734の表示31.691207% / 87.658979%の差は、
  平滑化だけで作られた差ではない。既にB3状態/入力履歴に差が残っている。
- elのprob_boardは内容でなくrepr（アドレス）で保存されているため、差の集計から除外。
  全B3特徴列・採用NEXT行・手数はelに未保存。どの列が55.97ppの何割を作ったかは
  特定できていない。推測した特徴を個別に書き換える修正は行っていない。

### 修正

ガードを `_step_side` から `update` の最終公開境界へ移動した。
prev_confirmed、全消しラッチ、較正、擬似ラベルには介入前の盤を渡す。
補正が発生したSideResultだけにoptional `evaluation_confirmed_board` を保持する。
`row1_publication.evaluation_result` で元盤を取り出し、liveのB3観測/会計/イベント/NEXT、
R評価、認識スナップショット、DeviceSessionの較正にも補正前の入力を渡す。
プロセス間の差分通知にも元盤を配送し、評価完了後の描画だけ公開盤へ戻す。
評価用と公開用の盤を混同しない単体試験を追加した。

緑のNEXT移動はf70772で既に検出されていた。raw緑はf70776から、
TSUMO_FALLはf70777–70778、着地診断と確定盤の更新はf70779。
v2は公開候補が空の間に合図を消費したため、f70800のOJAMA_FALLまで保留した。
f70779–70799は3合図ともfalseで、NEXT検出器が同区間の移動を取りこぼしたという証拠はない。

v2.1は先行NEXTを最初の着地診断に一度だけ対応付ける。
有効範囲は既存着地投票窓と同じ0.4秒、着地診断・raw一致・下の支持が必要。
期限を過ぎれば破棄し、時間だけでは公開しない。対象セルのない着地でも合図を消費する。
連鎖/おじゃまの新しい合図、resetでも持ち越しを止める。
作動ログはschema version=2を維持し、guard_revision=2.1、settled、next_signal_secを追加。

### PC1検証結果と限界

1. **el保存ガード入力の再生**（認識・B3全体の再走ではない）:
   まずsettled=falseでv2出力を全件再現し、不一致0を確認した。
   その後v2.1で緑初公開f70779、接触f70778±1から **1±1コマ**（8以内）。
   紫はf68708、26±1コマのまま。OFFも同じため別件として記録する。
   #8は **0/67非空** を維持。
   early/game36採点区間の対象外差は **0/1,196,580セル**
   （対象を含めた全セル母数1,212,120、盤面None同士も同等として含む）。
2. **保存実画素128コマ、10Hz冷起動、B3同期**:
   内部盤、prev_confirmed、評価結果全体、手数、B3観測/イベント/NEXT入力、
   全特徴列、raw予測、表示の不一致はすべて **0/128**。
   公開盤も今回は差0（以前1コマ保留した青も先行合図で公開された）。
   したがって、この画素標本だけで「作動中の全動画安全性」を証明したとはしない。
   実際に公開セルを消す分離と差分通知は単体試験で確認。
   旧v2 OFFとの結果は追加Noneフィールドを除き全バイト一致、表示も一致。
   `docs/ROW1_V21_saved_comparison.json`、原票は`logs/row1_v21/{off,on}/`。
   配布uの資産/run_k認識設定を使用。既知のscore_zero画像2枚不在は両条件で同じ。
3. row1試験 **81/81**、認識pipeline/live橋渡し/差分通知等 **299/299**。
   最後のrow1/live橋渡し/device/production_config試験 **276/276**（前記と重複あり）。
   原票は`logs/row1_v21/{related_tests,final_tests}.txt`。
4. **修正後f70734の31.69%一致は未実測**。elにはB3再構築に必要な全観測がなく、
   PC1には当該連続動画もない。上記128コマをこの成功の代用にしない。
   f70734・実動画の緑8コマ・対象外差・#8を確認するPC2短区間手順と計装を同梱する。

再生コマンド（同梱Python、worktree cwd）:

```powershell
& C:/PuyoLive_20261014u/PuyoLive/python/python.exe scripts/diagnose_row1_v21.py docs/ROW1_V21_decomposition.json
& C:/PuyoLive_20261014u/PuyoLive/python/python.exe scripts/compare_row1_v21_saved.py logs/row1_v21/off logs/row1_v21/on logs/row1_v2/off_audit docs/ROW1_V21_saved_comparison.json
```

bundle: `D:/puyo_analyzer/share/live_redesign/live_row1guard_v21.bundle`。
前提d9073afb、HEAD/SHA/サイズは隣の`live_row1guard_v21.receipt.json`。
手順は`docs/ROW1_GUARD_V21_PC2.md`。inbox投入・PC2起動・段4は行わない。
採用判断は保留、既定OFFのまま。
