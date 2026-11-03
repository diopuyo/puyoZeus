# アプリ用ブランド素材（2026-10-07）

利用者提供の `D:/puyo_analyzer/brand/app_logo_v1.png` と `overlay_mascot_v1.gif` が原本です。
原本は変更していません。brand に既存の透過済み `app_logo_site.png/.ico` と
`overlay_mascot_site.webp` を確認し、`app_logo.png/.ico`、`overlay_mascot.webp` としてコピーしました。
サイトリポジトリにはアクセスしていません。12ポーズのサイト専用素材は同梱していません。

ICO: 16 / 24 / 32 / 48 / 64 / 128 / 256px、RGBA。
WebP: 320×320px、6フレーム、60/30/20/60/20/30ms、無限ループ。
ロゴのキャラクターはアイコン専用。OBSの既定は常に同じWebPです。
任意の表情連動では `mascot_10/30/50/70/90.webp` を使います。
50は元WebPとバイト一致。他の4本も顔以外の可視画素・透過・6コマの時間を維持しています。
生成素材の目・口のみを各コマへ合成し、体や上の目玉は描き直していません。
生成用プロンプト・合成方法は `docs/MASCOT_EXPRESSIONS.md` を参照してください。
BMPは透過ロゴを白地に配置したInno Setup用の形式変換です。
利用者による表示素材の差し替え手順は `docs/DISTRIBUTION_APP_GUIDE.md` を参照してください。
