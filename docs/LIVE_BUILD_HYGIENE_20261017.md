# 配布ZIPの実行状態混入防止

build_20261017a は配布用ステージで保存動画smokeを実行したため、
`app/config/device_calibration/` に既定のOBS機器の色較正JSONが生成され、
その後のZIPへ混入した。MANIFEST外のファイルであっても動作へ影響する。

以後は `packaging/build_bundle.py --zip` または共通の
`bundle_archive.write_bundle_zip` で実行時の較正・ログ・キャッシュ・出力を除外する。
資産収集段階でも除外し、インストーラー生成は汚れたステージを検出して停止する。
初回高速化のため事前生成したpycと、ライブラリのcacheモジュールは保持する。

起動検証の順序は必ず以下とする。

1. 製品設定を確定し、ZIPを生成する。
2. `bundle_archive.extract_smoke_copy(zip_path, new_directory)` で新しい別フォルダへ展開する。
3. 展開先のPythonとappでsmokeを実行し、ログ・出力は検証用フォルダへ置く。
4. 元ZIPのSHA256が検証前後で同一と確認する。smokeコピーを再ZIP化しない。

build_20261017b はaのZIPを `repack_archive` で再包装する。
アプリコードはbc5d6c17のまま。packagingの修正コードは配布appには含まれない。
全ファイルの内容・ZipInfo時刻を保持して較正JSONのみを除き、pycは再生成しない。
新旧の全ファイル名・SHA256、除外一覧、smoke結果を納品先へ保存する。
20261017aは保存する。

既知の不備（今回の修正対象外）: `LICENSES/numpy-2.4.4` の
LICENSE.md/ LICENSE.txtは収集時の同名上書きで内容が入れ替わっている。
前回からの不備としてBUILD_NOTESへ記載し、今回の再包装では変更しない。
