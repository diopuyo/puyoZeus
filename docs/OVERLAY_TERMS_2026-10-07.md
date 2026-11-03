# オーバーレイ表示文言の置き換え（2026-10-07）

対象は表示ページ・設定ページ・全5テーマ（従来を含む）・全配置と指定見本画像。内部項目名、URLパラメーター、公開データのキー・値、判定計算は変更しない。

## 置き換え一覧

- 来ているおじゃま／来る／来るおじゃま → 予告ぷよ
- 送った量／送った計／送った量（累計） → 送ったおじゃま（設定では「合計」を併記、表示行の説明でも合計と明示）
- 相殺した量／相殺計／相殺量（累計） → 相殺（設定・表示行の説明で合計と明示）
- 置ける手数 (H)／置ける手数 H／手数 H → 猶予（数値はN手）
- 返せる量 (Q)／返せる量 Q（中央値）／返す Q → 返し火力
- 差 (M)／差 M／差 M（返す−来る） → 返しの過不足 (＋なら返せる・−なら足りない)。説明を画面に常時表示し、正負の値にも＋・−を付与
- 連鎖の段／今の段 → 連鎖の進み（数値はN連鎖目）
- 予測連鎖長／予測長 → 予想連鎖（数値は予想 N連鎖）
- 残り段 → 残り連鎖（数値は残り N連鎖）
- 残り秒 → 残り時間（数値はN秒）
- 1手で撃てる最大量 (k1)／火力 k1／火力 k1（既知2手＋先1手） → すぐ撃てる火力
- B3 k1：既知2手＋理想ツモ先1手、総3組以内の最大火力 → 見えている2手と、その次に理想のツモが来た場合の1手を合わせ、3手以内で撃てる最大火力（説明）
- モデル：静止／モデル：撃ち合い／使用モデル → 通常判定／撃ち合い判定／判定方法
- モデル：— → 判定方法：—
- 上限で埋めた印／上限補完／上限補完の印 → (目安)（予想できないときの返し火力の上限であることを設定・表示行の説明に明記）
- 予測不能／予測不能の印 → 予想できず／予想できずの印
- 計算中 → 判定中
- 推定勝率／対戦の勝率推定 → 勝率／対戦の勝率
- 台帳 → おじゃま
- 時間軸・中央値 → 返し
- レート → おじゃまレート
- N個 (N段)／6個＝1段 → N個（段換算を削除）
- 観測済みの段数。予測不能時は実際の段番号と一致しない場合があります → 読み取れた連鎖数。予想できない場合は実際の何連鎖目かと一致しないことがあります（説明）
- 表示前の評価 → 変動をならす前の勝率
- 確定盤面の評価 → 確定した盤面の勝率
- 組み立て評価／撃ち合い評価 → 通常判定／撃ち合い判定
- 評価更新待ち／評価エラー → 判定待ち／判定エラー（入力データの「評価エラー」は従来どおり受信）
- 上段は平滑後 → 勝率は変動をならして表示
- 公開された評価値 · 1P基準 → 1Pの勝率
- 予測を破棄 · 確定盤面へ → 予想を使わず、確定した盤面で判定
- 連鎖後の予測／予測込み → 連鎖後の予想／予想込み
- 較正中 → 画面調整中
- 公開データ待ち／現在の公開データ → データ待ち／現在の対戦データ
- 評価データ付き → 判定の詳細付き
- 見本画像のB3 判定の中身／v7と一致 → 判定の中身／内部版番号を削除
- 見本画像の静止／予測不能・上限補完 → 通常判定／予想できず・(目安)

画面に出るmask・episode・static/exchangeの直接表記はなし。設定ファイル用JSON・URL内のキー、ソース内の識別子・コメントは維持。火力の計算条件も維持し、名称変更で「1手だけの探索」に変更しない。

## 英語表示の対応（配布版）

内部キーと計算条件を変えず、表示文言だけを切り替える。OSのUI言語を既定にし、日本語以外は英語。
設定から日本語／English／OSを選択でき、URLの `language=ja` / `language=en` が一時上書き。

- 予告ぷよ → Incoming garbage
- 送ったおじゃま（合計） → Garbage sent (total)
- 相殺（合計） → Garbage countered (total)
- 猶予 → Time to respond（N手 → N moves）
- 返し火力 → Response attack
- 返しの過不足 → Response margin
- ＋なら返せる・−なら足りない → + enough to counter / − shortfall
- 連鎖の進み → Chain progress（N連鎖目 → N links）
- 予想連鎖 → Predicted chain（予想 N連鎖 → Predicted N chain）
- 残り連鎖 → Remaining chain（残り N連鎖 → Remaining N chain）
- 残り時間 → Time remaining（N秒 → N s）
- すぐ撃てる火力 → Immediate attack potential
- 見えている2手と、その次に理想のツモが来た場合の1手を合わせ、3手以内で撃てる最大火力 → Maximum attack within three moves: the two visible pieces followed by one ideal piece.
- 通常判定 → Build evaluation
- 撃ち合い判定 → Exchange evaluation
- 判定方法 → Evaluation method
- (目安) → (Upper bound)（予想不能時の返し火力の上限）
- 予想できず → Prediction unavailable
- 判定中 → Evaluating
- 勝率／対戦の勝率 → Win probability / Match win probability
- おじゃま → Garbage（N個 → N puyos）
- 返し → Response
- おじゃまレート → Garbage conversion rate（点/個 → points/puyo）
- 読み取れた連鎖数。予想できない場合は実際の何連鎖目かと一致しないことがあります → Observed chain count; it may differ from the actual link when prediction is unavailable.
- 変動をならす前の勝率 → Unsmoothed win probability
- 確定した盤面の勝率 → Confirmed-board win probability
- 判定待ち／判定エラー → Waiting for evaluation / Evaluation error
- 勝率は変動をならして表示 → Win probability is smoothed
- 1Pの勝率 → 1P win probability
- 予想を使わず、確定した盤面で判定 → Using the confirmed board without prediction
- 連鎖後の予想／予想込み → Post-chain prediction / Includes prediction
- 画面調整中 → Calibrating
- データ待ち／現在の対戦データ → Waiting for data / Current match data
- 判定の詳細付き／判定の中身 → Analysis details
- 試合待ち／接続待ち／入力確認中 → Waiting for a match / Waiting for connection / Checking input
- 五分／有利／不利／勝ち／負け → Even / Ahead / Behind / Win / Loss
- ログを送る／送信内容の確認／確認して送信 → Send logs / Review report / Confirm and send

表示辞書の実装は `src/phase_j/overlay_i18n.js`。単位・符号・未知値「—」は保持。

## 見本画像

- `D:/puyo_analyzer/local_frames/overlay_design/mobile/` の `data.jpg`・`l_right.jpg`・`l_left.jpg`
- `D:/puyo_analyzer/local_frames/overlay_design/trend/mobile/` の `neon.jpg`・`arena.jpg`・`pop.jpg`・`noir.jpg`
- 各見本の元となる同フォルダ階層の静止画も更新。動画・配布フォルダは対象外。

## 検証

- ブラウザで全5テーマ×9配置条件の実値表示・旧用語不在・横幅を検査。
- 連鎖・手・個の単位、符号、欠測、項目選択、設定、既存DTO、通信状態、演出、四隅・縮小率・文字サイズ・横帯の組み合わせを検査。
- 文言が長くなった箇所のみ折り返し・項目名の列幅・横帯余白を調整。
- PID 2088・配布フォルダへの操作なし。コミットのみでpushなし。
