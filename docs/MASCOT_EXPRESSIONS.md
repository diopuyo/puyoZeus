# 勝率による任意のマスコット表情（2026-10-07）

既定はオフ。設定ページ・設定JSON・URLに `mascot_expressions` と
`mascot_flip_right` を追加した。表示中の平滑化後の1P勝率を参照する。
10/30/70/90%で区切り、境界ちょうどは上の帯（10→30の顔、30→50の顔、70→70の顔、90→90の顔）。
設定JSONの `overlay.mascot_band_edges` は `[10,30,70,90]` が既定。
0より大きく100未満の昇順4値で変更可能。設定ページでも編集でき、
URLの `mascot_band_edges=10,30,70,90` は保存値より優先する。
通常は切替から1500ms保持し、期限に到達した時点の最新の帯を使う。
途中で通過した帯を順に再生する待ち行列は作らない。
試合待ち・較正・通信断は50、確定決着は1P勝利=90／敗北=10へ即時切替し、
保留タイマーを解除する。確定していない0/100は通常の帯として扱う。
向きの変更と勝率視点は独立している。

## 素材と調整

元のGIFは未変更。同梱の透過WebP（320×320、6コマ）を基準に、
50はファイルをそのまま複製。その他は目・口だけを合成した。
上の大きな目玉・体・触角・頬・透明度と60/30/20/60/20/30msの時間は維持。
生成画像全体は形や位置が変わるため採用せず、目・口のみを抜き出して元の位置へ合わせた。
90の星は目の中に収め、ため息の雲や周辺のきらめきは追加していない。
輪郭外に効果を足すと「顔だけを変える」条件から外れるため。
コマごとに元の小さな目・口の領域を周辺の淡青色で補間し、同じ位置に素材を重ねた。
通常表示の小・中・大でも目と口を判別できる大きさに調整している。

再生成: `python scripts/build_mascot_faces.py`
生成元: `scripts/assets/mascot_faces/generated_sheet.png`。
組み込みのImageGenツールを使用（CLI/APIは未使用）。
顔以外の可視画素・全画素のalpha・コマ数・各コマ時間はテストで原本と比較する。
更新した帯の[見本PNG](mascot_expressions_preview.png)は配布物にも同梱。
作成済み5本と旧見本は `D:/puyo_analyzer/brand/mascot_faces/`。
差し替え方法は [配布ガイド](DISTRIBUTION_APP_GUIDE.md) を参照。

## 生成プロンプト

### v2（2026-10-07、preview.8）

70を半目＋ウインク＋小さな舌のにやり顔、10をしょんぼり眉＋涙目＋への字口へ変更。
おすすめ各1案を採用したため、別案の比較画像は作成していない。
30・50・90は旧版とバイト一致。顔以外の可視画素・alpha・6コマ・各コマ時間も維持。
組み込みImageGenで `generated_sheet.png` を編集し、結果を
`scripts/assets/mascot_faces/generated_sheet_v2.png` に保存した。
既存の合成処理を拡張し、10・70だけv2から眉・涙を含む目と口を切り出す。
シート右列は採用せず、30・90は元の生成シートを使う。
5段階見本は `docs/mascot_expressions_preview.png` と
`D:/puyo_analyzer/brand/mascot_faces/mascot_expressions_preview.png` に保存。
配布物の出力先は、今回の指定範囲内の
`D:/puyo_analyzer/brand/mascot_faces/release_v2/`。

使用プロンプト（組み込みツール、CLI/APIは未使用）:

```text
Use case: precise-object-edit. Edit this mascot expression sprite sheet, preserving 2x2 grid, framing and face positions. Change ONLY TOP LEFT and BOTTOM LEFT small eyes and mouth inside pale cyan face. TOP LEFT: much more dejected, wilted expression: sad inward-raised eyebrows immediately above both drooping eyes, small blue tears attached below eyes, strongly downturned small mouth. Readable cute sadness, not angry. BOTTOM LEFT: playful lightly teasing smug face: left eye half-lidded confident, right eye wink, asymmetrical upward smirk with a tiny pink tongue. Tasteful and cute. Keep TOP RIGHT and BOTTOM RIGHT entirely unchanged. Keep all bodies, red cheeks, large stalk eyeballs, silhouettes, colors, and positions unchanged. Dark plum outline matching source. Do not add text, props, or effects outside pale cyan face. This sheet is used to crop individual eye/brow/tear groups and mouths for compositing into the original animation. Keep each eye/brow/tear group compact and separated from mouth and red cheeks. Output square same arrangement as input.
```

### v1

Edit target: supplied 320x320 transparent mascot frame. Create one 2x2 sprite expression sheet,
equal square cells, all four use EXACTLY the same mascot body, stalk, top eyeball, scale,
placement, outline and colors as input. Change ONLY the two small dark eyes and mouth INSIDE
the pale blue face area at bottom. Top-left: tired half-lidded eyes, little sigh mouth
(sad, not asleep). Top-right: anxious round eyes and zigzag mouth. Bottom-left: happy eyes
and open smiling mouth with pink tongue. Bottom-right: sparkling star highlights inside eyes
and wide joyful open smile. Keep the red cheeks unchanged. Very dark purple outlines matching
original. No new objects, no effects outside face, no labels, no text. Preserve transparent
background. Make each cell identical framing to the input, no margins between cells, entire
2x2 sheet square. This is a sprite sheet used for precisely cropping just the small face
expressions; especially preserve the face position and pale cyan background.
