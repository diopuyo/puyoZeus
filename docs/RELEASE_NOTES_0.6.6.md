# puyoZeus Live 0.6.6

<!-- 0.6.5 の更新ボタンから 0.6.6 へ上がる実機の試験の合格を、サイトの更新の前に確認する。 -->

## 起動前の注意 / Before launching

ZIP をダブルクリックで開いただけでは動きません。ZIP を右クリック →「すべて展開」してから、展開したフォルダの puyoZeus Live.exe を起動してください。「python312.dll が見つからないため、コードの実行を続行できません」と出た場合も、ZIP をすべて展開してから起動し直してください。

Opening the ZIP with a double-click is not enough. Right-click the ZIP → Extract All, then run puyoZeus Live.exe from the extracted folder. If Windows says “The code execution cannot proceed because python312.dll was not found”, extract the entire ZIP and launch the app from the extracted folder.

Windows 11 の「スマート アプリ コントロール」が有効な PC では、計算部品が止められて解析が始まらないことがあります。その場合は画面にその旨を表示します。コード署名の準備を進めています。

On PCs with Windows 11 Smart App Control enabled, a required component may be blocked so that analysis cannot start. The app shows a message when this happens. We are preparing code signing.

ぷよぷよ eスポーツの配信に推定勝率を重ねる、非公式のファン制作ツールです。Windows 向けのアーリーアクセス版です。

## 0.6.5 からの変更

- ゲーム画面の画像を配布物に含めないようにしました。得点・連鎖数・試合の終わりなどの読み取りは、画像を持たない判定器で行います。0.6.5 と同じ動画 4 本で比べた推定勝率の差は平均 0.03 ポイントでした。
- 読み取りの処理は 0.6.5 より少し重くなります (手元の PC で 1 コマあたり約 6%)。
- 0.6.5 からは「その他」の「更新を確認する」で更新できます。設定・ログ・保存した OBS の連携・自動起動は引き継ぎます。

## Changes since 0.6.5

- The download no longer includes images taken from the game screen. Scores, chain counts, and match endings are now read by classifiers that do not contain images. Across the same four videos, estimated win probabilities differed from 0.6.5 by 0.03 percentage points on average.
- Recognition is slightly heavier than in 0.6.5 (about 6% per frame on our test PC).
- From 0.6.5, update with Check for updates under Other. Settings, logs, saved OBS integration, and start at sign-in are retained.
