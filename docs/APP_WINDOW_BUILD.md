# 専用の設定窓を含める配布手順

窓の中身は既存の設定ページ。`src.production_config.DISTRIBUTION_APP_WINDOW` の既定はOFF。
配布物の `distribution.json` に `"distribution_app_window": true` を保存すると、既存の `PuyoLive.vbs` → `pythonw.exe -m src.phase_j.distribution_app` で専用窓を開く。旗を変更したらアプリを終了して再起動する。

## 組み立て処理の差分

1. `prepare_window_wheels` が pywebview 6.1 とWindows/Python 3.12用の依存wheelを `camera_wheels/` に取得する。既存のpystray wheelがあってもこの処理は省略しない。
2. proxy_tools 0.1.0 は `pip wheel --no-deps` で純Python wheelを作る。作業場所はビルド出力フォルダ。Windows用のバイナリwheel取得は `--platform win_amd64 --python-version 312 --implementation cp --abi cp312` を指定する。
3. 既存の `finish_bundle` でwheelを `python/Lib/site-packages/` に展開し、`copy_licenses` で原文を収録する。`copy_window_licenses` がwheelに欠けるproxy_tools原文、cffiのlibffi部分の著作権表示、WebView2 SDK原文・第三者通知を補う。
4. ソースの `distribution_window.py` と既存の `brand/app_logo.ico` が同梱されていること、追加wheelの版、LICENSESの原文、THIRD_PARTY_NOTICES.mdを最終zipで照合する。

追加8パッケージの試験用wheel合計は1,190,398 bytes、展開ファイル合計は3,763,360 bytes。既存typing_extensionsとの重複、pyc、実行時キャッシュ、WebView2 Runtimeはこの増分見積もりから別扱い。最終zipのサイズではない。

## 配布前の確認

- 旗OFFとONを両方確認する。初回は/setup、設定済みは/settings。
- 閉じるボタンは隠すだけ。通知領域の「開く」、最小化からの復帰、二重起動で元の窓に戻ることを確認する。
- 画面内の終了と通知領域の終了で、自分の配信処理と窓が終わることを確認する。
- pywebviewが読めない場合、およびWebView2がない場合のブラウザ退避を確認する。実装はpywebview 6.1のrenderer初期化APIを使用しているため、同ライブラリを更新するときはこの確認を再実施する。
- WebView2 Runtimeを自動インストールする処理は追加していない。既存Runtimeと.NET Frameworkを使用する。

この変更の作業では配布物を組み立てていない。既定ONの採否と最終配布物の確認はPC1が行う。
