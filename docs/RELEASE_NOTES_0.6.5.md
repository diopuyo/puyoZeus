# puyoZeus Live 0.6.5

<!-- 更新ボタンは 0.6.5 で既定 ON。公開前に実機の更新試験の合格を確認する。 -->

## 起動前の注意 / Before launching

ZIP をダブルクリックで開いただけでは動きません。ZIP を右クリック →「すべて展開」してから、展開したフォルダの puyoZeus Live.exe を起動してください。「python312.dll が見つからないため、コードの実行を続行できません」と出た場合も、ZIP をすべて展開してから起動し直してください。

Opening the ZIP with a double-click is not enough. Right-click the ZIP → Extract All, then run puyoZeus Live.exe from the extracted folder. If Windows says “The code execution cannot proceed because python312.dll was not found”, extract the entire ZIP and launch the app from the extracted folder.

Windows 11 の「スマート アプリ コントロール」が有効な PC では、計算部品が止められて解析が始まらないことがあります。その場合は画面にその旨を表示します。コード署名の準備を進めています。

On PCs with Windows 11 Smart App Control enabled, a required component may be blocked so that analysis cannot start. The app shows a message when this happens. We are preparing code signing.

ぷよぷよ eスポーツの配信に推定勝率を重ねる、非公式のファン制作ツールです。Windows 向けのアーリーアクセス版です。

## 0.6.4 からの変更

- 連鎖中の光の影響を受けにくくし、連鎖の読み取りを改善しました。
- 中盤はフィールドの状況も使って推定勝率を補正するようにしました。
- 操作中のツモを置き終えたぷよと誤って扱い、評価がちらつく問題を抑えました。
- 時刻の丸めや検査の失敗で配信が止まる問題を修正しました。
- グラフの縦の幅を試合に合わせる設定を既定で有効にしました。L字・真ん中の柱・縦の配置では、縦向きのグラフを枠いっぱいに表示します。
- 連鎖の打ち合い直後の推定勝率を追加でずらす補正は、既定では無効です。
- 設定の「その他」から更新を確認・ダウンロードする機能を追加しました。0.6.5 では既定で有効です。日付と変更点を確認してから更新し、進み具合の表示と中断、SHA256 の照合、設定・ログ・OBS 連携・自動起動の引き継ぎ、失敗時の復旧に対応します。「起動時に更新を確認する」は既定 OFF です。
- 0.6.4 のフォルダーに上書きして更新できます。0.6.4 には更新ボタンがないため、最初の更新は ZIP を展開してください。設定・OBS 連携・自動起動は引き継ぎます。

## Changes since 0.6.4

- Improved chain recognition by reducing the effects of flashes during chains.
- Midgame win probability now also accounts for the field state.
- Reduced evaluation flicker caused by treating the pair being moved as already placed.
- Fixed streaming interruptions caused by timestamp rounding or failed validation.
- Graphs fit their vertical range to each match by default. Vertical graphs fill their frame in the L-shaped, center-column, and vertical layouts.
- The additional win-probability shift immediately after chain exchanges is disabled by default.
- Added in-app update checks and downloads under Other in Settings, enabled by default in 0.6.5. The feature shows the release date and changes, download progress and cancellation, SHA256 verification, preservation of settings, logs, OBS integration and start at sign-in, and recovery after a failed update. Check at startup is off by default.
- You can overwrite your 0.6.4 folder to update. Version 0.6.4 has no update button, so extract the ZIP for the initial update. Settings, OBS integration, and start at sign-in are retained.

グラフの縦の幅は既定で試合の最大の振れ＋余白に合わせます（最低40〜60%、試合ごとにリセット）。「見た目」の「グラフの縦の幅を試合に合わせる」は既定オンで、チェックを外すと従来の0〜100%固定になります。
The graph range fits each match’s largest swing plus padding by default (minimum 40–60%, reset each match). In Appearance, “Fit graph range to the match” is on by default; uncheck it for the previous fixed 0–100% range.
