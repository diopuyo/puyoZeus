# OBS表示の4案（2026-10-07）

既定は **A 中央カード**。1920×1080の対戦画面で中央の空きを使い、
左右の盤面・得点、上のNEXT・おじゃま予告、下のWINを避ける。
66px・太字の勝率、左右固定の1P/2P、青／暖色の赤、2Pバーの斜線、
半透明の濃紺背景と文字影を組み合わせる。勝率は整数表示で合計100%。

## 切り替え

OBSブラウザソースを幅1920・高さ1080で使用する。

- A 中央カード：`http://127.0.0.1:8765/` または `/?layout=card`。460×約267px、左730・上594。
- B 横長リボン：`http://127.0.0.1:8765/?layout=ribbon`。500×約216px、左710・上700。
- C 最小表示：`http://127.0.0.1:8765/?layout=minimal`。416×約127px、左752・上802。推移と詳細を省略。
- D 評価データ付き：`http://127.0.0.1:8765/?layout=data`。460×約403px、左730・上514。

異なる大きさのブラウザソースでも1920×1080座標を等比縮小し、余白は透明。
OBSで手動拡大・クロップすると位置も変わるため、ゲームと同じ16:9範囲に配置する。

## 状態と評価値

試合待ち／較正中／入力確認中／計算中／対戦中／決着を短く表示する。
死亡の物理予測だけでは決着扱いしない。既存の `terminal.state=confirmed` または
ライブB3の `source=confirmed_death` と端点確率から決着を表示する。
決着は即時反映し、通常の数字とバーは160msだけ補間する。
サーバ側の平滑処理に追加の長い平均を重ねない。動きを減らすOS設定にも対応。

直近30秒の1P勝率を入力映像の時刻で描く。中央線は50%、上ほど1P有利。
再送で同じ時刻の点を増やさず、試合・セッション変更や映像の巻き戻しでリセットする。
評価未取得時は数字・詳細・推移を消す。対戦中に3.5秒更新がなければ「更新待ち」、
SSEエラーだけを「接続待ち」とする。試合待ちの定期再送（ba39297）は変更していない。

D案は `evaluations.display_layers` の現在評価／予測値を表示する。
B3では現在評価が平滑前なので「表示前の評価」と明記する。
現行B3は連鎖後勝率を提供しないため「—」。判定の中身は任意の `b3_contents` から実値を表示（詳細は OVERLAY_OPTIONS_2026-10-07.md）。
公開スキーマ・API・評価計算・配布物は変更しない。

## 比較資料と再生成

画像はすべて `D:/puyo_analyzer/local_frames/overlay_design/` に保存する。
`{card,ribbon,minimal,data}_{early,exchange,result}.jpg` が全12枚の実フレーム見本。
`mobile/{card,ribbon,minimal,data}.jpg` は各案の3場面と拡大図をまとめた1080×2660画像。
各約0.9MB。勝率と推移はデザイン確認用のサンプルと画像内に明記した。
原フレームのパス・各表示の矩形は `manifest.json` に記録する。
obs_probeの画像には旧オーバーレイが焼き込まれていたため、評価記録の原フレームを使用。
大会ロゴ・スポンサー枠は中央にあるため隠れる場合がある。ゲームの操作情報とは分けて確認した。

再生成はWindowsのPillow・Playwright・Edge環境で `python -m scripts.render_overlay_design`。
画素から再認識せず製品HTMLを隔離ブラウザで描画し、実フレームに重ねる。
ゲーム画像はリポジトリに保存しない。判定データの見本は学習には使用しない。

## 検証

`tests/phase_j/test_overlay_browser.py` で実ブラウザ37ケース：全4案×0/50/100%、
スマホ相当の縮小、状態遷移、欠損値、公開DTO例、決着と予測の区別、履歴の時間軸、
試合待ち再送、通信断／更新遅延、補間終了を検査する。
ブラウザはWindowsではEdge、他環境ではPlaywright Chromium。
`OVERLAY_BROWSER_CHANNEL` で変更可能。Playwrightがない環境ではこのファイルのみskip。
公開DTO・較正・表示層・HTTP/SSE・旧表示の関連テストも実行する。
旧較正テスト1件の期待文言を、ba39297で採用済みの「試合待ち」に合わせた。
実行結果はWindows・Python 3.14・Edgeで357 passed（ブラウザ37件を含む、skipなし）。
旧指標の乱数seed未指定警告22件は既存のもの。稼働中のOBSでの切り替え試験は行っていない。

## 配布版を切り替えるときだけ実行

今回、稼働中のPID 2088と配布フォルダには操作していない。ビルドも未実行。
ユーザーがPuyoLiveを終了させてから、以下をPowerShellで実行する。
`build_b3_obs.py` は既存 `puyo_live.json` のバイト列を保存し、組み直し後に復元するため、
`device_index=4` を含め現在の設定を維持する。

```powershell
& D:/puyo_analyzer/packaging/build/PuyoLive/python/python.exe D:/puyo_analyzer/wt_b3_rt/packaging/build_b3_obs.py --model D:/puyo_analyzer/share/results/pc2/fast_pipeline/runs/v6/final/model.joblib
```

再起動後、OBSブラウザソースを再読み込みして上記URLへ切り替える。
