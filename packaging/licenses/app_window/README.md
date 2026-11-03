# 設定窓の追加ライセンス原文

2026-11-02に取得。`build_release.copy_window_licenses` が下記のディレクトリを、その名前のまま配布物の `LICENSES/` へコピーする。

## proxy_tools 0.1.0

- 原文: https://raw.githubusercontent.com/jtushman/proxy_tools/master/LICENSE.txt
- 元リポジトリ: https://github.com/jtushman/proxy_tools
- 保存先: `proxy_tools-0.1.0/LICENSE.txt`
- 原文のSHA-256: `a428fb8a2e762af3eb0a6edbbb88e9b42ccfee80fd9b423958bcacf9b9abbfe4`
- 1436 bytes。wheelには原文が含まれない。wheel内 `proxy_tools/__init__.py` のヘッダーもBSDを指定する。metadataのMIT表記だけに従って原文を置き換えない。

## Microsoft.Web.WebView2 SDK 1.0.2957.106

- 公式パッケージ: https://www.nuget.org/packages/Microsoft.Web.WebView2/1.0.2957.106
- 取得先: https://api.nuget.org/v3-flatcontainer/microsoft.web.webview2/1.0.2957.106/microsoft.web.webview2.1.0.2957.106.nupkg
- nupkgのSHA-256: `4c35a54835b63954159eac1d5b7a60ae617a41dbb5b73bfdb11c4870a891080a`
- nupkgは8,779,374 bytes。中の `LICENSE.txt`（1487 bytes）と `NOTICE.txt`（3894 bytes）を同名で保存した。
- pywebview 6.1 wheel内のCore.dll、WinForms.dll、win-x86/win-x64/win-arm64のWebView2Loader.dllの5ファイルすべてについて、同じSHA-256のファイルがこのnupkg内にあることを確認した。
- 原文はSDKのBSD-3-Clauseライセンス。WebView2 Runtimeそのものを同梱するためのライセンスではない。

pywebviewなど、wheelがLICENSEを含む7パッケージの原文は既存の `copy_licenses` が収録する。原文は翻訳・書き換えず、そのまま保存する。

## cffi 2.1.1のWindows x64向けlibffi部分

- 配布元: https://pypi.org/project/cffi/2.1.1/
- sdist: https://files.pythonhosted.org/packages/9e/ef/008a1939e372c06329a3fce4279c02f328488f3526744906eeec3da7ad5f/cffi-2.1.1.tar.gz
- sdistのSHA-256: `dd31f52ea1086513bb9df30f8fcee9b8918323ae067a3d5b78bc826a000712be`（PyPIのdigestsと照合）。
- setup.pyがWindows/MSVCで使用する `src/c/libffi_x86_x64/` の ffi.c、ffi.h、ffi_common.h、ffitarget.h、prep_cif.c、types.c の原文先頭コメント6件を、ファイル名とともに `cffi-2.1.1/libffi-NOTICES.txt` に保持した。MIT系の許諾・著作権表示があるため、cffi本体のMIT-0原文だけで置き換えない。
- win64.asm、fficonfig.hには別のライセンスヘッダーはなかった。Windows x64ではsetup.pyがwin32.cを除外しwin64.objを使う。
