# 0.6.0 公開手順（GitHub Releases + Cloudflare Pages）

この手順は user / PC1 が実行する。今回の作業はローカル準備までで、push、タグ作成、Release 作成、デプロイは実行していない。
【user 操作・承認】の印は実行前に内容を確認する箇所。

## 1. 正式版 ZIP の確定

2026-11-03追補: 言語切り替え時のログ案内を修正した final_060b を使用する。旧 final_060 は保存するが配布しない。評価経路は変更していない。

- ファイル: `puyoZeus-Live-0.6.0.zip`
- 容量: **323,705,140 bytes**
- SHA256: `1d9aa067edf771efe54671830269d6c580f38d09f135aa36d7ecbd17d3bde357`
- 製品 code_sha: `ad5e3322`。公開ソースのパス置換・非公開検査台本の除外は製品の評価動作を変更しない。
- 同値判断: PC1 が 2026-11-03 15:58 に合格と判断済み。評価コード・モデル・部品は RC とバイト一致。同じ RC の再生でも最大20.24ポイント・平均0.43ポイントの差が出る非同期採用時機の揺れ（KNOWN_WEAKNESSES W66）があり、正式版比較の最大20.2ポイントを版の差と扱わない。
- 【user 操作・承認】ZIP の実ファイルで容量と SHA256 を照合し、別 PC の初回導入確認の担当を決める。

```powershell
(Get-Item ./puyoZeus-Live-0.6.0.zip).Length
Get-FileHash ./puyoZeus-Live-0.6.0.zip -Algorithm SHA256
'1d9aa067edf771efe54671830269d6c580f38d09f135aa36d7ecbd17d3bde357  puyoZeus-Live-0.6.0.zip' | Set-Content ./SHA256SUMS -Encoding ascii
```

## 2. GitHub のソース・タグ・Release

公開先は `diopuyo/puyoZeus`。今回の公開用枝は **release/v0.6.0-final2**。
公開 main `9eb08b5c` の上に履歴を持ち込まない1コミット。作者・コミッターは `diopuyo <diopuyo2@gmail.com>`。
作業枝 `release/v0.6.0`、旧 `release/v0.6.0-public4-final` は送らない。

【user 操作・承認】送信対象の差分、画像一覧、個人情報の手元検査、試験結果を確認する。
公開 main が基点から進んでいた場合は実行を止めて PC1 が取り込み方を判断する。force push はしない。
以下はソースのリポジトリで実行する。pre-push の確認は LF 固定 bytes で行い、終了0だけでなく fatal/error/拒否0を確認する。

```powershell
git log -1 --format=fuller release/v0.6.0-final2
git diff --stat 9eb08b5c release/v0.6.0-final2
git ls-remote origin refs/heads/main refs/tags/v0.6.0
git push origin release/v0.6.0-final2:refs/heads/release/v0.6.0-final2
git tag -a v0.6.0 release/v0.6.0-final2 -m 'puyoZeus Live 0.6.0'
git push origin refs/tags/v0.6.0
```

既存の v0.6.0 タグがあれば作り直さず停止して確認する。
【user 操作・承認】GitHub の Release 作成画面で v0.6.0 を選び、正式版 ZIP と SHA256SUMS を添付して**下書きで保存**。
版名・添付ファイル名・容量・SHA・説明・タグのコミットを確認し、user の承認後に Release を公開する。

## 3. 公開された ZIP を取得して照合

取得先は次の固定 URL。
`https://github.com/diopuyo/puyoZeus/releases/download/v0.6.0/puyoZeus-Live-0.6.0.zip`

【user 操作】新しい確認フォルダにダウンロードし、323,705,140 bytes と上記 SHA256 の両方が一致することを確認する。
GitHub のリダイレクトは許可する。404、容量不一致、SHA 不一致なら Pages へ進まない。

```powershell
New-Item -ItemType Directory ./verify-v0.6.0
Invoke-WebRequest 'https://github.com/diopuyo/puyoZeus/releases/download/v0.6.0/puyoZeus-Live-0.6.0.zip' -OutFile ./verify-v0.6.0/puyoZeus-Live-0.6.0.zip
(Get-Item ./verify-v0.6.0/puyoZeus-Live-0.6.0.zip).Length
Get-FileHash ./verify-v0.6.0/puyoZeus-Live-0.6.0.zip -Algorithm SHA256
```

## 4. カタログ確定 → Pages プレビュー → 本番

以降は配布サイトのリポジトリで行う。`site/releases.json` と `drafts/releases-0.6.0.draft.json` に正式版の容量・SHA・取得先を記入済み。
公開前は `status: draft` のため配布ボタンを無効にしている。手順3を通過してから `site/releases.json` の0.6.0だけを `status: published` に変更する。

次回以降の候補生成は `node scripts/publish_release.mjs --provider github --version <版> --date <日付> --zip <ZIP> --notes <日英JSON>`。
既定はドライランで、GitHub API・R2 を呼ばず、未使用の `artifacts/releases.proposed.<時刻>.json` に候補を書くだけ。
同じ版の上書きは禁止。今回の0.6.0は登録済みの下書きを確認して反映する。
空の別カタログを入力する場合は `--catalog <JSON>`、別の未使用出力は `--output <JSON>` を指定できる。
GitHub 経路に `--publish` は使わない（拒否する）。

サイトの住所の設定は **site/site-config.json の siteUrl だけ**。
仮の住所は `https://puyolive-site.pages.dev`。Pages のプロジェクト作成時に実際の住所が違えばそこだけ直し、`node scripts/configure-site.mjs` で静的 canonical・og:url・OGP画像URLへ反映する。
公開対象のディレクトリは `site`。`drafts`・`artifacts`・`worker`・秘密の設定を含めない。

【user 操作・承認】Cloudflare Pages のプロジェクトを作成し、無料の pages.dev を使用。ZIP は Pages に置かない。
`node scripts/configure-site.mjs`、`npm test`、ローカルサーバー起動後 `npm run test:browser` を実行する。
Pages プレビューで日英トップ・ダウンロード・SHA256・GitHub URL・スマートフォン幅・canonical/OGPを確認する。
【user 承認】プレビュー確認後にだけ本番へ反映する。独自ドメインや R2 の設定は不要。

## 5. 公開後の確認

【user 操作】ログインしていないブラウザと別 PC で pages.dev → GitHub ZIP の導線を確認する。
ダウンロードした ZIP の容量・SHA を再確認し、展開 → puyoZeus Live.vbs → 初回案内 → OBS 仮想カメラ・ブラウザソースを確認する。
ログは手元へ保存される。報告の自動送信先は0.6.0では開かない。
失敗時は PC1 に症状と確認段階を報告し、別の ZIP で同じ版を上書きしない。

## 将来

- R2: `--provider r2 --bucket <バケット> --account-id <ID>` の既存経路を保持。既定はドライラン。独自ドメイン取得後にカタログの取得先を設計し直してから、承認を得て使用する。
- 独自ドメイン: 採用時は siteUrl を変更して静的HTMLを再生成する。
- Workers / D1: 障害報告の受け口は0.6.0で開設・デプロイしない。管理画面は将来用で、利用者向け導線は設けない。
