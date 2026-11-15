# puyoZeus Live 0.6.8

<!-- 0.6.7 の更新ボタンから 0.6.8 へ上がる実機の試験の合格を、サイトの更新の前に確認する。 -->

## 起動前の注意 / Before launching

ZIP をダブルクリックで開いただけでは動きません。ZIP を右クリック →「すべて展開」してから、展開したフォルダの puyoZeus Live.exe を起動してください。「python312.dll が見つからないため、コードの実行を続行できません」と出た場合も、ZIP をすべて展開してから起動し直してください。

Opening the ZIP with a double-click is not enough. Right-click the ZIP → Extract All, then run puyoZeus Live.exe from the extracted folder. If Windows says “The code execution cannot proceed because python312.dll was not found”, extract the entire ZIP and launch the app from the extracted folder.

Windows 11 の「スマート アプリ コントロール」が有効な PC では、計算部品が止められて解析が始まらないことがあります。その場合は画面にその旨を表示します。コード署名の準備を進めています。

On PCs with Windows 11 Smart App Control enabled, a required component may be blocked so that analysis cannot start. The app shows a message when this happens. We are preparing code signing.

ぷよぷよ eスポーツの配信に推定勝率を重ねる、非公式のファン制作ツールです。Windows 向けのアーリーアクセス版です。

## 0.6.7 からの変更

- 試合と試合の切り替わりで前の試合の決着の表示が残り、次の試合の間ずっと推定勝率が出ないことがある問題を直しました (手元の試験ではおよそ 20 試合に 1 回起きていました)。
- CPU のコアが少ない PC で OBS と一緒に使った時に、推定勝率の表示が遅れにくくなるよう、処理の優先度の組み方を見直しました。4 コアのノート相当の試験で、表示の遅れが約 1 秒から約 0.3 秒に縮み、OBS の描画への影響は増えませんでした。
- 0.6.5 以降は「その他」の「更新を確認する」で更新できます。設定・ログ・保存した OBS の連携・自動起動は引き継ぎます。

## Changes since 0.6.7

- Fixed an issue where the result of the previous match could remain on screen at the start of the next match, so no estimated win probability was shown for that entire match (about once every 20 matches in our tests).
- Revised how processing is prioritized so the estimated win probability lags less on PCs with few CPU cores running OBS at the same time. In a test equivalent to a 4-core laptop, display delay dropped from about 1 second to about 0.3 seconds with no added impact on OBS rendering.
- From 0.6.5 or later, update with Check for updates under Other. Settings, logs, saved OBS integration, and start at sign-in are retained.
