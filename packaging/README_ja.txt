# puyoZeus Live 0.6.4 — ZIP 版 / ZIP edition

【起動前に必ずすべて展開 / Extract All before launching】
ZIP をダブルクリックで開いただけでは動きません。ZIP を右クリック →「すべて展開」してから、展開したフォルダの puyoZeus Live.exe を起動してください。「python312.dll が見つからないため、コードの実行を続行できません」と出た場合も、ZIP をすべて展開してから起動し直してください。
Opening the ZIP with a double-click is not enough. Right-click the ZIP → Extract All, then run puyoZeus Live.exe from the extracted folder. If Windows says “The code execution cannot proceed because python312.dll was not found”, extract the entire ZIP and launch the app from the extracted folder.

ぷよぷよ eスポーツの映像から推定勝率を OBS に表示する、非公式の Windows x64 用ソフトです。上級者対戦を対象とし、実際の勝敗を保証しません。ゲームの権利者による提供・承認を示すものではありません。

## はじめに

1. ZIP を書き込み可能な普通のフォルダ (例 `C:\puyoZeus`) にすべて展開します。OneDrive の中は避けてください。デスクトップやドキュメントも OneDrive の中にあることが多く、同期で重くなったり書き込みがぶつかったりすることがあります。ZIP の中から起動しないでください。
2. `puyoZeus Live.exe` をダブルクリックして設定を開きます。「はじめに」は最初に開いた時だけ表示します。
3. 初回は「配信」の「はじめに」①で OBS と連携します。OBS の「ツール → WebSocket サーバー設定」で有効にし、「接続情報を表示」のパスワードを入力して「OBS を自動で設定」を押します。ブラウザソースを追加し、仮想カメラを開始します。
4. ②は OBS での手作業です。「仮想カメラ開始」横の歯車 → 出力の種類「ソース」→ ゲームのキャプチャを選びます。ゲーム全体が見える 16:9、1920×1080 の映像を使い、本アプリの重ね表示を入力に含めないでください。この出力の選択は自動設定では変更できません。
5. ③「解析を始める」を押します。試合が映るまで「試合待ち」、映ると「解析中」になります。「解析を止める」で待機も止められます。自動設定の成功と、映り込み警告なしの解析を一度確認すると、準備は「準備: 済み ✓」にたたまれます。次回は OBS のゲーム映像を用意して「解析を始める」を押すだけです。「はじめにをもう一度見る」で手順を開けます。
6. 上部に状態と案内、OBS・ブラウザソース・仮想カメラの連携状態が出ます。「くわしく」でいまの状態を確認できます。映り込み警告が出たら手順②を確認してください。URLのコピー、手動追加 (幅1920・高さ1080)、WebSocketポート、パスワードの消去、「OBS の状態を確かめる」は「手動で設定する・その他」にあります。
7. 「アプリが見ている映像」で入力を確認できます。「入力を変える」を開くと入力選択と「映像を確認」があります。解析中は確かめられないため、解析を止めてから押してください。自動起動と更新確認は「その他」の「アプリの設定」で変更できます。

既定ポートは 8765 ですが、使用中の場合は別のポートになります。「見た目」タブで変えた設定は自動で保存され、開いている OBS 表示にも再読み込みなしで反映されます。URL は短い1本のままです。0.6.0 の長い URL でも保存した設定に従います。「画面調整中」が終わるまで待ち、短い対戦で映像と表示の遅れを確認してください。

Python と必要な計算部品は同梱しています。Python や CUDA の別途導入は不要です。AVX2 などの CPU 命令に対応しない古い PC では動かない場合があります。`msvcp140.dll` などが見つからない場合は Microsoft 公式の Visual C++ 再頒布可能パッケージ X64 を導入してください。CPU・メモリの最低仕様は未確定です。

ZIP 自体には署名がありません。入手元と公開 SHA256 を確認してください。入口の exe は署名付き Python の写しです。

## 表示と終了

既定の表示は「真ん中の柱」、ポップ黒、表情 ON です。目の動きとドーパミンモード、ドーパミンの強さは設定で変更できます。表情と演出はドーパミンモードに連動します。位置の固定を切り替え、L 字は 4 隅から選べます。表示する内容は配置ごとに固定です。勝率の数字とゲージは全配置、配置は真ん中の柱・小さい帯・カード・縦・L 字 4 隅です。カードは上段に数字とゲージ、下段に今の試合全体の横長グラフを表示します。直近 30 秒は小さい帯・カード以外、試合全体はカードと L 字の横帯に表示します。横帯を隠すと試合全体も隠れます。キャラクターの動きは目とドーパミン設定に応じた演出です。

ウインドウの × は通知領域へ隠すだけです。終了には通知領域の「終了 / Exit」を使います。通知領域の「開く / Open」で設定を開けます。起動時の更新確認と自動起動は既定 OFF、入力の自動判定は既定 ON です。

## ログ

自動送信はありません。設定ページのログの操作で内容を確認して確定します。既定の送信先は準備中のため、ネットワークへ送らず展開先の `logs/reports` に保存します。`tools/SendLogs.ps1` も同じです。ホスト名が `.invalid` で終わる宛先は保存のみです。実際の HTTPS 宛先を設定した場合は確認後に送信し、失敗時は手元に保存します。保存済みの内容は次の確認に含まれます。再起動だけで自動再送しません。

## ZIP 版の削除

設定画面に削除の欄はありません。アプリを終了後に、同梱の `アンインストール.exe` (互換入口は `tools/Uninstall.bat`) を実行し、ショートカット・自動起動の登録を解除します。データ削除の質問は Enter / N で保存、Y で削除です。必要な設定・ログを退避してから展開フォルダを削除します。

画像認識の誤りや入力の乱れにより勝率が揺れたり表示が遅れたりします。盤面・得点・勝敗表示を隠さないでください。ゲーム素材の権利は各権利者に帰属します。第三者部品は `THIRD_PARTY_NOTICES.md` と `LICENSES` を参照してください。

## English

puyoZeus Live is an unofficial Windows x64 overlay estimating win probability from Puyo Puyo Champions video. It targets advanced play. Estimates do not guarantee outcomes. The app is not provided or endorsed by the game's rights holders.

1. Extract the entire ZIP to a normal writable folder (for example `C:\puyoZeus`), outside OneDrive. Desktop and Documents are often inside OneDrive; syncing can slow the app down or cause write conflicts. Do not run inside the ZIP.
2. Double-click `puyoZeus Live.exe` to open Settings. The getting-started guide appears only on the first visit.
3. On Stream → Getting started, connect to OBS in step ①. In OBS, enable Tools → WebSocket Server Settings. Enter the password from Show Connect Info, then press Set up OBS automatically. This adds the browser source and starts Virtual Camera.
4. Step ② must be done in OBS: gear beside Start Virtual Camera → Output Type: Source → select your game capture. Use a complete 16:9, 1920×1080 game view without this app's overlay. Automatic setup cannot change the output selection.
5. In step ③, press Start analysis. The app waits for a match, then shows Analyzing. Stop analysis also cancels waiting. After successful OBS setup and analysis without an overlay feedback warning, the steps collapse to Ready ✓. On later launches, prepare the game feed in OBS and press Start analysis. Show getting started again reopens the steps.
6. The top shows analysis status and the OBS, browser source and Virtual Camera connection status. Details opens diagnostics. Follow step ② if overlay feedback is detected. Manual setup and other options contains Copy URL, manual browser setup (1920×1080), WebSocket port, password removal and Check OBS status.
7. Check What the app sees for a preview. Change input contains input selection and Check video. Stop analysis before checking video. Windows autostart and update checks are under Other → App settings.

The default port is 8765; another port is used if occupied. Changes on Appearance are saved automatically and applied to an already open OBS source without reloading. Keep the same short URL; old long URLs from 0.6.0 also follow saved settings. Wait for screen calibration and test a short match before streaming.

Python and computation libraries are bundled; separate Python or CUDA installation is unnecessary. Older CPUs without instructions such as AVX2 may not work. If `msvcp140.dll` or a similar runtime is missing, install Microsoft's official Visual C++ Redistributable X64. Minimum CPU and memory requirements are not established. The ZIP is unsigned. If SmartScreen warns, verify the source and published SHA256.

The default display is the center column, black Pop theme, with expressions enabled. Eye movement, dopamine mode and dopamine intensity are adjustable. Expressions and effects follow dopamine mode. Position locking is adjustable and the L layout offers four corners. Display contents are fixed by layout: numbers and gauges in all layouts, the recent 30 seconds except in Small strip and Card, and the whole match in Card and the L layout band. Layouts are Center column, Small strip, Card, Vertical and four L corners. Card shows numbers and a gauge above a horizontal graph of the current match. Hiding the band also hides the whole-match view. Motion comes from the eyes and dopamine effects.

Closing the window with X only hides it in the notification area. Use Exit in the notification-area menu to stop it. Open Settings from that menu. Startup update checks and Windows autostart are off by default; automatic input detection is on.

Logs are never sent automatically. Review their contents before confirming. The default destination is being prepared, so confirmed reports are saved in `logs/reports` without network access. `tools/SendLogs.ps1` behaves the same way. Destinations whose host ends in `.invalid` are local-only. With a real HTTPS destination configured, confirmed reports are sent there; failures are saved locally. Saved reports appear in the next review and are not automatically resent on restart.

Removal is not available in settings. To remove the ZIP version, exit the app and run the bundled `アンインストール.exe` (legacy entry: `tools/Uninstall.bat`) to remove shortcuts and startup registration. At the data-removal prompt, Enter / N keeps data and Y removes it. Back up needed settings and logs, then delete the extracted folder.

Recognition errors or disrupted input can cause fluctuations and delay. Keep boards, scores and result indicators visible. Game assets remain the property of their respective rights holders. See `THIRD_PARTY_NOTICES.md` and `LICENSES` for third-party notices and licenses.

OBS のブラウザソースには、アプリが表示する `http://127.0.0.1:<ポート>/` を入力してください。`localhost` への置き換えはできません。

解析開始時に前回の `output` を `output_history` へ退避します。履歴は直近3回、合計512MiB以内で、古いものから削除します。必要な記録は別の場所へ保存してください。

OBS パスワードは現在の Windows ユーザーの DPAPI で保存します。別のユーザーやPCへ設定を移すときは再入力してください。設定画面の「パスワードを消去」で削除できます。

## 入力の確認 / Input preview

「配信」タブでアプリが見ている映像と機器を確認・切り替えできます。解析中は同じ入力から毎秒1回ほど縮小して表示し、機器を二重に開きません。停止中は直近に確認した画像です。画像は手元のメモリとローカル HTTP だけで扱い、保存・外部送信しません。

On Stream, check the video and switch inputs. During analysis the preview uses the existing capture, about once per second; it does not open a second camera. While stopped it shows the last checked image. Images stay in memory and local HTTP, without saving or external transmission.

ZIP 直下の起動入口は `puyoZeus Live.exe`、削除入口は `アンインストール.exe` です。旧版の互換 bat/vbs は、配布時の名前と内容が一致するものを起動時に片付けます。自動起動は選んだときだけ直下の exe を使って登録します。スタートメニューとデスクトップのショートカットは作りません。旧版が作った、この展開先を指すリンクは起動時に片付けます。

The ZIP root has one launcher, `puyoZeus Live.exe`, and one uninstall entry, `アンインストール.exe`. Obsolete BAT/VBS files are removed at startup only when their names and contents match a previous release. Autostart uses the root exe only when enabled. Start menu and desktop shortcuts are no longer created. On launch, old shortcuts pointing to this extracted folder are removed.

補助ファイル / Helper files: `tools/SendLogs.ps1`, `tools/UninstallCleanup.ps1`, `tools/puyo_live.example.json`, `tools/mascot_expressions_preview.png`.

## 初回の起動 / First launch

起動は `puyoZeus Live.exe` を開きます。入口には Python Software Foundation の署名付き Python の写しを使い、スマート アプリ コントロール (SAC) が ON の PC で起動できる構成にしました。exe のアイコンは Python のものです。一番上の dll と ._pth はアプリの部品です。削除しないでください。

Python と計算ライブラリ、Rust 製の処理モジュールは同梱済みで、Python/Rust の開発環境は不要です。Visual C++ または WebView2 ランタイムが無い場合、起動用 exe が Microsoft 公式の部品を入れるか尋ねます。「はい」の場合だけ取得し、Microsoft の署名を検証して実行します。Windows が権限の確認を表示する場合があります。「いいえ」でも起動は続き、WebView2 が無ければブラウザで開きます。

Open `puyoZeus Live.exe`. The entry is an unchanged copy of Python signed by the Python Software Foundation, designed to launch with Smart App Control enabled. Its EXE icon is the Python icon. The DLL and ._pth files at the top level are required app components; do not delete them. Python, computation libraries and Rust modules are bundled. Missing Visual C++ or WebView2 runtimes are offered with your consent and Microsoft signature verification. Declining continues startup; without WebView2, the app opens in a browser.

## 0.6.4 の変更 / Changes in 0.6.4

Windows 11 の「スマート アプリ コントロール」が有効な PC では、計算部品が止められて解析が始まらないことがあります。その場合は画面にその旨を表示します。コード署名の準備を進めています。
On Windows 11 PCs with Smart App Control enabled, a required computation component may be blocked and analysis may not start. The app will display a notice if this happens. We are preparing code signing.

評価値を 1P 視点の大きな 1 つの数字で表示し、表示を 2 種類に整理しました。局面や撃ち合いの有無に応じて推定勝率のずれを補正します。評価値・グラフ・表情も補正後の勝率に従います。
The evaluation is shown as one large number from Player 1's perspective, with two display styles. Estimated win probabilities are calibrated by game phase and whether a chain exchange is active. The evaluation, graphs, and expressions follow the calibrated probability.

撃ち合いで、相手の長い連鎖の終わり際に、もう相殺したおじゃまを「これから受ける」と二重に数え、勝率が数秒間逆の向きに傾くことがあった問題を直しました。0.6.3 のフォルダーに上書きして更新できます。設定・OBS 連携・自動起動は引き継ぎます。
Fixed a problem in chain exchanges where garbage that had already been offset was counted again near the end of the opponent's long chain, tilting the win probability the wrong way for several seconds. You can overwrite your 0.6.3 folder to update. Settings, OBS integration, and start at sign-in are retained.

## 0.6.3 の変更 / Changes in 0.6.3

- 勝率の判定を学び直しました。学習に使う上級者の対戦のデータを約 2.4 倍に増やしています。
- 全消しのボーナスが残った状態で連鎖を始めた直後に、撃ち合いの勝率の計算がその時だけ抜けることがあった問題を直しました。
- 0.6.2 のフォルダーに上書きして更新できます。設定・OBS 連携・自動起動は引き継ぎます。

- Retrained the win probability model with about 2.4 times as much data from high-level matches.
- Fixed a problem where the exchange win probability could briefly drop out right after a chain was started while an all-clear bonus was still pending.
- You can overwrite your 0.6.2 folder to update. Settings, OBS integration, and start at sign-in are retained.

## 0.6.2 の変更 / Changes in 0.6.2

- 配信タブを作り直しました。上部で解析を始める・止める操作と状態を確認でき、OBS 連携の状態は1行にまとまります。初回は「はじめに」の3ステップで準備し、済んだら折りたためます。仮想カメラの出力設定も図で案内します。
- 配置に「カード」を追加しました。見た目の見本のずれを修正し、「詳しく」を「詳細設定」に改めました。
- デスクトップとスタートメニューのショートカットを作らなくなりました。0.6.1 が作ったスタートメニューのリンクは、同じフォルダーに上書きして起動すると片付きます。サインイン時の自動起動は引き続き使えます。
- 0.6.1 のフォルダーに上書きして更新できます。設定・OBS 連携・自動起動は引き継ぎます。

- The Stream tab has been redesigned. Start or stop analysis and check its status at the top, with OBS integration status on one line. Three Getting started steps guide the initial setup and can be collapsed when complete. A diagram explains the virtual camera output settings.
- Added the Card layout, corrected the appearance preview alignment, and renamed the advanced settings control to Advanced settings.
- Desktop and Start menu shortcuts are no longer created. When you overwrite the same folder and launch the app, the Start menu shortcut created by 0.6.1 is removed. Start at sign-in remains available.
- You can overwrite your 0.6.1 folder to update. Settings, OBS integration, and start at sign-in are retained.

## 0.6.1 の修正 / Fixes in 0.6.1

0.6.0 で ZIP をエクスプローラーで展開すると、専用の窓ではなくブラウザで開いていた不具合を直しました。
Fixed an issue in 0.6.0 where extracting the ZIP with File Explorer opened the app in a browser instead of its own window.

別の場所の puyoZeus Live が動いているときは、配信タブに案内が出ます。「そちらを終了してこちらに切り替える」で OBS の URL を変えずに切り替えられます。起動する展開先の「puyoZeus Live.exe」を開いてください。
If puyoZeus Live is running from another location, the Streaming tab offers to stop it and switch to this copy without changing the OBS URL. Open puyoZeus Live.exe in the extracted folder you want to use.
