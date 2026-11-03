# 表示オプションとL字配置

既定は従来の中央カード。既存の `card` / `ribbon` / `minimal` / `data` を維持し、
`l-right`（ゲーム左上、データ右＋下）、`l-left`（ゲーム右上、データ左＋下）を追加した。
B3の採用行の中身を公開DTOの任意項目として追加した。設定ファイルの値をURLで項目単位に上書きできる。

## 設定ページ

更新版の起動後、`http://127.0.0.1:8765/settings` をブラウザで開く。
ゲームの実フレーム画像を選択すると、静止画を縮小配置して製品HTMLの表示を重ねて確認できる。
画像はブラウザ内のみに保持し、サーバへ送信しない。リポジトリ・配布物へのゲーム画像追加もない。
序盤・撃ち合い・決着のサンプル値、または現在の公開データを選べる。
現在データでも背景は選択した静止画。見本用の値は実測値ではない。
項目を選び「URLをコピー」すると、OBS用URLを取得できる。
設定ページはJSONや配信中の他ソースを直接変更せず、JSON例とURLを作る。

## OBSの手順

1. 基本キャンバスは1920×1080。ゲームキャプチャは元の1920×1080のまま用意する。
2. ブラウザソースに設定ページでコピーしたURLを指定し、幅1920・高さ1080、位置(0,0)で最上面に置く。
3. L字型ではゲームキャプチャの「変換→変換の編集」で、基準点を左上、クロップ0にする。
   設定ページに表示された位置・幅・高さを設定する。80%なら1536×864。
4. 80%時のゲーム位置は左上(0,0)、右上(384,0)、左下(0,216)、右下(384,216)。
   側列はゲームの反対側。下隅に寄せた場合、帯は上側へ移す。どの隅でもゲーム矩形は完全透明。
5. 仮想カメラはこれまでどおりゲームキャプチャだけの「ソース出力」。
   配信シーンの合成画面を認識入力にしない。縮小は配信シーン内の変換だけで行い、
   ソース出力の認識入力は1920×1080を維持する。
6. 横帯を無効にすると、判定の中身と試合全体の推移は側列へ移る。
   ゲームを縮小した残りの余白には他のOBSソースを配置できる。

## 設定項目

`puyo_live.json` の最上位に `"overlay": {...}` を追加する。他の既存項目は残す。
ファイルは次回起動で反映。URL指定は即時にそのブラウザソースへ反映する。

```json
"overlay": {
  "layout": "l",
  "corner": "top-left",
  "scale": 80,
  "font": "medium",
  "band": true,
  "numbers": true,
  "recent": true,
  "full": true,
  "status": true,
  "details": true
}
```

- `layout`: `card` / `ribbon` / `minimal` / `data` / `l` / `l-right` / `l-left`。
- `corner`: `top-left` / `top-right` / `bottom-left` / `bottom-right`。L字型だけに作用。
- `scale`: 70〜85（百分率、小数可）。L字型だけに作用。
- `font`: `small` / `medium` / `large`。狭い列はあふれ防止の調整を適用。
- `band`: 横帯を使うか。L字型だけに作用。
- `numbers` / `recent` / `full` / `status` / `details`: 各表示項目の独立したオン・オフ。

URL例：`http://127.0.0.1:8765/?layout=l&corner=bottom-right&scale=75&font=large&details=false`。
URLの真偽値は `true` / `false` または `1` / `0`。不正なURL値はその項目のファイル既定へ戻る。
不正なJSON設定は起動前に日本語エラーにする。URLで他の項目を指定しても、指定しない項目の設定は保持する。
省略時は配置の従来初期値を使う（最小表示の直近推移OFF、dataの詳細ON、L字の全体推移・詳細ON）。

## データの扱い

直近30秒と試合全体の推移は独立した履歴。全体はページで受信した最初の評価から描く。
途中接続・再読込では過去分を復元しないため、時間軸に「受信開始」と明記する。
同じ入力時刻の再送は置換。試合・セッション変更、巻き戻し、試合待ち・較正でリセットする。
一時的な計算待ちや通信断では過去の全体履歴を保持する。保持上限は2048点で、超過時は開始点を残して間引く。
確率の補間、待機再送、決着判定の条件は従来どおり。

連鎖後の予測は公開済み `evaluations.display_layers.predicted_p1` のみを表示する。
連鎖後勝率は現在のB3では欠測。判定の中身は `evaluations.b3_contents` から表示する。
量は生のおじゃま個数と段換算（6個＝1段、小数1桁表示）、欠測は「—」。
L字型では台帳・時間軸を帯、連鎖・盤面を横列に置く。帯なしなら全て横列。
data案は全項目を読める中央の縦長表示。従来カードなどの既定は変更しない。
設定ページで以下の `content_` 項目を個別選択できる（全て既定true、detailsが全体のスイッチ）。
`pending` / `sent` / `offset` / `rate` / `hands` / `return` / `margin` /
`bound` / `unknown` / `active` / `step` / `length` / `remaining` / `seconds` /
`k1` / `colors` / `ojama` / `model`。例：`content_k1=false`。
k1は評価行の定義を維持し、既知2手＋理想ツモ先1手（総3組以内）の最大火力。
来る量は台帳の受け予告、送った量と相殺量は試合累計。H・Q・Mは受け側基準。
Mは中央値Q−台帳予告。上限補完ありのQ/Mは算術上限で、確定火力ではない。

## 見本と検証

`python -m scripts.render_overlay_design2` で隔離したEdgeから製品HTMLを描画する。
出力先は `D:/puyo_analyzer/local_frames/overlay_design/`。
`l_right` / `l_left` と `options_bottom` / `options_column` / `options_minimal` の各3場面、
および `mobile/` の縦長5枚を生成する。数値と推移は見本用と明記する。
`manifest2.json` にURL引数とゲーム・表示の矩形を記録する。従来の見本は上書きしない。

ブラウザ試験で四隅×3縮小率×3文字サイズ×帯有無の72組合せ、各表示スイッチ、
全体履歴、ゲーム全画素の透明度、設定ページの実画像選択とHTTPの設定優先順位を検査する。
設定ファイル→ランチャ→パイプライン→配信HTMLの経路と既存の公開契約も検査する。

Windows・Python 3.14・Edgeで関連検査482 passed、2 skipped、既存seed警告22件。
スキップは `test_bundle_asset_list.py` の実資産ディレクトリ不在による2件。
設定ページの画像透過は、ブラウザ枠を表示／非表示にしたゲーム部分の画素一致でも検査した。
設定ページの見本は `settings.png`。PuyoLive実プロセスやOBS上での更新試験は行っていない。

稼働中PuyoLive（PID 2088）と `D:/puyo_analyzer/packaging/build/PuyoLive/` は操作せず、ビルド・pushも行わない。
本番への反映はユーザーが停止・更新した後。更新手順は `OVERLAY_DESIGN_2026-10-07.md` の末尾を参照。

## B3実値の見本更新

`python -m scripts.render_overlay_b3_contents --snapshots <照合出力>/snapshots.jsonl` で
`mobile/data.jpg`・`mobile/l_right.jpg`・`mobile/l_left.jpg` と各場面画像を更新する。
静止・撃ち合い・上限補完の値と推移はzenchi s1実再生。背景は従来の別場面で、画像にも明記する。
`manifest_b3_contents.json` に使用DTOを保存する。他のデザイン見本は従来のサンプル値のまま。
