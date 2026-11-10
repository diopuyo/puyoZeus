# 計算部品の署名（明示指定時だけ使用）

`.github/workflows/native.yml` はmain への push、手動実行、`v*` タグで動く。main では版 0.6.4 を使う。手動入力は
`major.minor.patch`、タグは `v0.6.4` の形を使用する。

GitHub-hosted Windows 上で Python 3.12、Rust 1.97.1、maturin 1.9.4、
Zig 0.16.0 を使い、`packaging/build_signed_native.py` が
`puyo_core.cp312-win_amd64.pyd` と `puyo_live_exact.dll` を作る。
計算部品の機能指定は現行24w2と同じ7フラグ（依存込み9フラグ）。
ソースや依存の配置場所は Rust のパス再対応で除く。
PE の FileVersion / ProductVersion は署名前に Windows 標準 API で付ける。

未署名の2ファイルは `native-unsigned` に保存する。
次の秘密と変数がすべてある場合だけ、入口exeと同じSignPath v3 actionに渡す。

- secret: `SIGNPATH_API_TOKEN`
- variables: `SIGNPATH_ORGANIZATION_ID`、`SIGNPATH_PROJECT_SLUG`、`SIGNPATH_SIGNING_POLICY_SLUG`

SignPath側にはpyd/DLLを対象にしたartifact configurationと、人による承認を含む
signing policyが必要。入口exeと同じprojectを使用する場合は両方の成果物の定義を整合させる。
このワークフローでは署名者の承認を自動化しない。
トークン等が未設定なら署名段を省略し、ビルド・照合が成功すれば正常終了する。
署名済みの成果物名は `native-signed`。アップロード前にも両ファイルの署名・版・指紋を検査する。
PDBやimport libraryはアップロードしない。

配布作成時は、署名済みの2ファイルを同じフォルダーに置き、既存の引数に
`--signed-native-dir <フォルダー>` を加える。両方について有効な署名・SignPath
Foundationと完全一致する署名者名・配布版と一致するPE版を確認し、pydの既存9項目の指紋と
DLLの240盤面×隠し段2条件（全C ABI）を検査してから置き換える。
DLLの基準は `native_golden.json` に保存済みで、署名検査時に旧未署名DLLを実行する必要はない。
読込拒否・指紋不一致・欠落は採用しない。署名検査対象を一時コピーし、その同じバイトを同梱する。

引数の既定は `None`。指定しなければ既存の計算部品をそのまま使用する。
`--signed-launcher-dir` とは独立して指定できる。

WindowsでSACがRustの依存DLLや未署名候補を拒否する場合も、ポリシーは変更しない。
WSLで同じソースと機能指定をビルドし、保存済み結果やLinux版で機能を照合できるが、
これはWindows上で署名済み候補を受け入れる実機確認の代わりにはならない。
