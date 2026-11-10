# SignPath 用の起動 exe（任意）

Windows 10/11 の .NET Framework 4.x と Python 3.12 を使う。

```powershell
python packaging/build_launcher_exe.py --output unsigned --version 0.6.4
```

同じソース・版・アイコン・コンパイラを使えば、出力先を変えても同じバイトに
なる。標準の旧 csc は決定的ビルドのスイッチがないため、署名前に PE の時刻を
0 にし、唯一の GUID（MVID）を内容の SHA-256 から設定する。署名済み exe には
この処理を適用しない。異なるコンパイラの出力一致は保証しない。

GitHub Actions の `launcher.yml` は main への push、手動実行、`v*` タグで
未署名成果物を保存する。main では版 0.6.4、タグではタグの版を使用する。
`SIGNPATH_API_TOKEN` secret と次の repository variables がすべてある時だけ署名を依頼する。
`SIGNPATH_ORGANIZATION_ID`、`SIGNPATH_PROJECT_SLUG`、
`SIGNPATH_SIGNING_POLICY_SLUG` を設定する。
SignPath 側で ZIP 内の 2 つの exe を署名する Artifact Configuration と、
版ごとに人が承認する signing policy を設定する必要がある。

署名後、成果物を新しいフォルダーへ展開し、配布組立時に指定する。

```powershell
python packaging/build_release.py --version 0.6.4 --signed-launcher-dir signed <その他の組立引数>
```

引数なしでは従来の PSF 署名付き Python の写しを使う。
指定時は 2 つとも Authenticode が Valid かつ署名者名に
SignPath Foundation を含み、FileVersion と ProductVersion が配布の `--version` と
一致することを検証する。署名後の版情報の書き換えは行わない。既存の配布物を更新せず、
従来どおり新しい出力先を使う。

通常起動は同梱 pythonw.exe から既存の signed_launcher.main を呼ぶ。
アンインストールは同梱 python.exe で distribution_uninstall モジュールを直接
起動する。親の Python を作らず、既存の停止確認と対話をそのまま利用する。
終了コードを返す前の Enter 待ちと、起動例外の理由表示は C# 側で行う。

公開ソースには `packaging/launcher_exe/`、`packaging/build_launcher_exe.py`、
`src/phase_j/brand/app_logo.ico`、`.github/workflows/launcher.yml` が必要。
配布の組立には既存の `packaging/launcher/`、`build_launchers.py`、
`build_release.py` も必要。いずれも Git の追跡対象にしてから公開用の木を作る。

公式の Action 入力定義:
https://github.com/SignPath/github-action-submit-signing-request/blob/v3/action.yml
