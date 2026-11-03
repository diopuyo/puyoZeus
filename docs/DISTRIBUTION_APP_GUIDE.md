# puyoZeus Live 0.6.0 — ZIP 版 / ZIP edition

ぷよぷよ eスポーツの映像から推定勝率を OBS に表示する、非公式の Windows x64 用ソフトです。上級者対戦を対象とし、実際の勝敗を保証しません。ゲームの権利者による提供・承認を示すものではありません。

## はじめに

1. ZIP を書き込み可能なフォルダにすべて展開します。ZIP の中から起動しないでください。
2. `puyoZeus Live.vbs` をダブルクリックし、初回の案内で入力・表示・ショートカットを確認します。
3. OBS でゲーム全体が見える 16:9、1920×1080 の映像を用意します。仮想カメラの出力を「ソース」にし、ゲームのキャプチャを選んで開始します。認識映像には本アプリの重ね表示を含めないでください。
4. 案内で入力を確認して設定を完了し、OBS ブラウザソース用 URL をコピーします。
5. OBS に「ブラウザ」ソースを追加し、幅 1920、高さ 1080 にして URL を貼り付けます。既存のソースはアプリ起動後に再読み込みします。

既定ポートは 8765 ですが、使用中の場合は別のポートになります。配置を変えたときも設定ページの URL をコピーし直してください。「画面調整中」が終わるまで待ち、短い対戦で映像と表示の遅れを確認してください。

Python と必要な計算部品は同梱しています。Python や CUDA の別途導入は不要です。AVX2 などの CPU 命令に対応しない古い PC では動かない場合があります。`msvcp140.dll` などが見つからない場合は Microsoft 公式の Visual C++ 再頒布可能パッケージ X64 を導入してください。CPU・メモリの最低仕様は未確定です。

配布物は署名なしです。Windows SmartScreen の警告が出る場合は、入手元と公開されている SHA256 を確認してください。

## 表示と終了

既定の表示は「真ん中の柱」、ポップ黒、表情 ON です。目の動き、表情、ドーパミンの強さは設定で変更できます。キャラクターの動きは目とドーパミン設定に応じた演出です。

ウインドウの × は通知領域へ隠すだけです。終了にはアプリの「アプリを終了」か、通知領域の「終了 / Exit」を使います。通知領域の「設定を開く / Settings」で設定を開けます。起動時の更新確認と自動起動は既定 OFF、入力の自動判定は既定 ON です。

## ログ

自動送信はありません。設定ページのログの操作で内容を確認して確定します。0.6.0 の既定の送信先は準備中のため、ネットワークへ送らず展開先の `logs/reports` に保存します。`SendLogs.ps1` も同じです。ホスト名が `.invalid` で終わる宛先は保存のみです。実際の HTTPS 宛先を設定した場合は確認後に送信し、失敗時は手元に保存します。保存済みの内容は次の確認に含まれます。再起動だけで自動再送しません。

## ZIP 版の削除

終了後に `アンインストール.bat` または `Uninstall.bat` を実行し、ショートカット・自動起動の登録を解除します。データ削除の質問は Enter / N で保存、Y で削除です。必要な設定・ログを退避してから展開フォルダを削除します。

画像認識の誤りや入力の乱れにより勝率が揺れたり表示が遅れたりします。盤面・得点・勝敗表示を隠さないでください。ゲーム素材の権利は各権利者に帰属します。第三者部品は `THIRD_PARTY_NOTICES.md` と `LICENSES` を参照してください。

## English

puyoZeus Live is an unofficial Windows x64 overlay estimating win probability from Puyo Puyo Champions video. It targets advanced play. Estimates do not guarantee outcomes. The app is not provided or endorsed by the game's rights holders.

1. Extract the entire ZIP to a writable folder. Do not run inside the ZIP.
2. Double-click `puyoZeus Live.vbs` and review the first-run input, display and shortcut options.
3. Prepare a complete 16:9, 1920x1080 game view in OBS. Set Virtual Camera output to Source, select game capture and start it. Keep this application's overlay out of the recognition input.
4. Complete input setup and copy the displayed OBS browser-source URL.
5. Add an OBS Browser source with width 1920 and height 1080 and paste the URL. Refresh an existing source after starting the app.

The default port is 8765; another port is used if occupied. Copy the settings-page URL again after changing layout. Wait for screen calibration and test a short match before streaming.

Python and computation libraries are bundled; separate Python or CUDA installation is unnecessary. Older CPUs without instructions such as AVX2 may not work. If `msvcp140.dll` or a similar runtime is missing, install Microsoft's official Visual C++ Redistributable X64. Minimum CPU and memory requirements are not established. The ZIP is unsigned. If SmartScreen warns, verify the source and published SHA256.

The default display is the center column, black Pop theme, with expressions enabled. Eye movement, expressions and dopamine intensity are adjustable in settings. Motion comes from the eyes and dopamine effects.

Closing the window with X only hides it in the notification area. Use Exit in the app or notification-area menu to stop it. Open Settings from that menu. Startup update checks and Windows autostart are off by default; automatic input detection is on.

Logs are never sent automatically. Review their contents before confirming. The destination is being prepared for 0.6.0, so confirmed reports are saved in `logs/reports` without network access. `SendLogs.ps1` behaves the same way. Destinations whose host ends in `.invalid` are local-only. With a real HTTPS destination configured, confirmed reports are sent there; failures are saved locally. Saved reports appear in the next review and are not automatically resent on restart.

To remove the ZIP version, exit and run `Uninstall.bat` to remove shortcuts and startup registration. At the data-removal prompt, Enter / N keeps data and Y removes it. Back up needed settings and logs, then delete the extracted folder.

Recognition errors or disrupted input can cause fluctuations and delay. Keep boards, scores and result indicators visible. Game assets remain the property of their respective rights holders. See `THIRD_PARTY_NOTICES.md` and `LICENSES` for third-party notices and licenses.
