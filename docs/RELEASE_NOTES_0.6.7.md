# puyoZeus Live 0.6.7

<!-- 0.6.6 の更新ボタンから 0.6.7 へ上がり、作業用フォルダーが消える実機の試験の合格を、サイトの更新の前に確認する。 -->

## 起動前の注意 / Before launching

ZIP をダブルクリックで開いただけでは動きません。ZIP を右クリック →「すべて展開」してから、展開したフォルダの puyoZeus Live.exe を起動してください。「python312.dll が見つからないため、コードの実行を続行できません」と出た場合も、ZIP をすべて展開してから起動し直してください。

Opening the ZIP with a double-click is not enough. Right-click the ZIP → Extract All, then run puyoZeus Live.exe from the extracted folder. If Windows says “The code execution cannot proceed because python312.dll was not found”, extract the entire ZIP and launch the app from the extracted folder.

Windows 11 の「スマート アプリ コントロール」が有効な PC では、計算部品が止められて解析が始まらないことがあります。その場合は画面にその旨を表示します。コード署名の準備を進めています。

On PCs with Windows 11 Smart App Control enabled, a required component may be blocked so that analysis cannot start. The app shows a message when this happens. We are preparing code signing.

ぷよぷよ eスポーツの配信に推定勝率を重ねる、非公式のファン制作ツールです。Windows 向けのアーリーアクセス版です。

## 0.6.6 からの変更

- 「更新を確認する」で更新した後、展開したフォルダーの隣に作業用のフォルダー (名前が「.PuyoLive-update-」で始まり、約 1.3 GB) が残っていました。起動時に自動で消すようにしました。0.6.5 から 0.6.6 へ更新した時に残ったものも消えます。
- 前の版から更新・上書きした時に残っていた、今は使っていないファイル (0.6.5 までのゲーム画面の画像など) を起動時に片付けるようにしました。中身が変更されたファイルは残します。
- 0.6.5・0.6.6 からは「その他」の「更新を確認する」で更新できます。設定・ログ・保存した OBS の連携・自動起動は引き継ぎます。

## Changes since 0.6.6

- After an update with Check for updates, a working folder (named “.PuyoLive-update-…”, about 1.3 GB) was left next to the app folder. It is now removed automatically at startup, including any folder left by the update from 0.6.5 to 0.6.6.
- Files that are no longer used but remained after updating or overwriting an earlier version (such as the game-screen images from 0.6.5 and earlier) are now removed at startup. Files whose contents were changed are kept.
- From 0.6.5 or 0.6.6, update with Check for updates under Other. Settings, logs, saved OBS integration, and start at sign-in are retained.
