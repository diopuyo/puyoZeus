# 第三者の部品とライセンス表記

puyoZeus Live のコードは MIT License（2026 diopuyo）です。同梱する第三者の部品・由来コードは、それぞれのライセンスと著作権表示に従います。この一覧は原文に代わるものではありません。

以下は2026-11-02に、11月1日の OBS 試験版（24w2）の Python 環境・LICENSES と、10月22日の棚卸しを照合した結果です。0.6.0 は ZIP で配布します。常駐版で追加する部品と未収録の原文を区別しています。パスは特記のない限り展開先からの相対パスです。試験版の起動ファイルは利用手順の根拠にしていません。

## Python と Python パッケージ

Python 3.12.3 embeddable: Python Software Foundation License（PSF-2.0）。原文は `python/LICENSE.txt`。Pythonが内包する部品の記載もこのファイルを参照してください。

下記31件は試験版の `python/Lib/site-packages/` 内のパッケージ情報を確認しました。dist-info 内に原文がある28件は `LICENSES/` にも収録されていますが、NumPyの原文は集約先だけでは揃いません。ONNX Runtimeの原文はパッケージ本体側にあります。

- attrs 26.1.0: MIT。原文: `LICENSES/attrs-26.1.0/LICENSE`。
- filelock 4.0.7: MIT。原文: `LICENSES/filelock-4.0.7/LICENSE`。
- flatbuffers 25.12.19: Apache 2.0。原文: 原文未収録（末尾の確認項目参照）。
- fsspec 2026.9.0: BSD-3-Clause。原文: `LICENSES/fsspec-2026.9.0/LICENSE`。
- Jinja2 3.1.6: BSD-3-Clause。原文: `LICENSES/jinja2-3.1.6/LICENSE.txt`。
- joblib 1.5.3: BSD-3-Clause。原文: `LICENSES/joblib-1.5.3/LICENSE.txt`。
- jsonschema 4.26.0: MIT。原文: `LICENSES/jsonschema-4.26.0/COPYING`。
- jsonschema-specifications 2025.9.1: MIT。原文: `LICENSES/jsonschema_specifications-2025.9.1/COPYING`。
- MarkupSafe 3.0.3: BSD-3-Clause。原文: `LICENSES/markupsafe-3.0.3/LICENSE.txt`。
- mpmath 1.3.0: BSD-3-Clause。原文: `LICENSES/mpmath-1.3.0/LICENSE`。
- networkx 3.7: BSD-3-Clause。原文: `LICENSES/networkx-3.7/LICENSE.txt`。
- numpy 2.4.4: BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0。原文: `python/Lib/site-packages/numpy-2.4.4.dist-info/licenses/` 以下（後述）。
- onnxruntime 1.30.0: MIT License。原文: `python/Lib/site-packages/onnxruntime/LICENSE` と `ThirdPartyNotices.txt`。
- opencv-python-headless 4.13.0.92: Apache 2.0。原文: `LICENSES/opencv_python_headless-4.13.0.92/LICENSE-3RD-PARTY.txt`、`LICENSES/opencv_python_headless-4.13.0.92/LICENSE.txt`。
- packaging 26.3: Apache-2.0 OR BSD-2-Clause。原文: `LICENSES/packaging-26.3/LICENSE`、`LICENSES/packaging-26.3/LICENSE.APACHE`、`LICENSES/packaging-26.3/LICENSE.BSD`。
- pandas 3.0.2: BSD-3-Clause。原文: `LICENSES/pandas-3.0.2/LICENSE`。
- pillow 12.2.0: MIT-CMU。原文: `LICENSES/pillow-12.2.0/LICENSE`。
- protobuf 7.36.2: 3-Clause BSD License。原文: `LICENSES/protobuf-7.36.2/LICENSE`。
- puyo_core 0.1.0: 自作コードはMIT、由来コード・依存crateは後述。原文: 由来コードの原文はリポジトリの `native/puyo_core/licenses/`（配布物は未収録）。
- python-dateutil 2.9.0.post0: Apache-2.0 / BSD-3-Clause。原文: `LICENSES/python_dateutil-2.9.0.post0/LICENSE`。
- referencing 0.37.0: MIT。原文: `LICENSES/referencing-0.37.0/COPYING`。
- rpds-py 2026.6.3: MIT。原文: `LICENSES/rpds_py-2026.6.3/LICENSE`。
- scikit-learn 1.8.0: BSD-3-Clause。原文: `LICENSES/scikit_learn-1.8.0/COPYING`。
- scipy 1.17.1: BSD-3-Clause（内包部品は原文参照）。原文: `LICENSES/scipy-1.17.1/LICENSE.txt`。
- setuptools 84.0.0: MIT。原文: `LICENSES/setuptools-84.0.0/LICENSE`。
- six 1.17.0: MIT。原文: `LICENSES/six-1.17.0/LICENSE`。
- sympy 1.13.1: BSD-3-Clause。原文: `LICENSES/sympy-1.13.1/LICENSE`。
- threadpoolctl 3.7.0: BSD-3-Clause。原文: `LICENSES/threadpoolctl-3.7.0/LICENSE`。
- torch 2.5.1+cpu: BSD-3-Clause。原文: `LICENSES/torch-2.5.1+cpu/LICENSE`、`LICENSES/torch-2.5.1+cpu/NOTICE`。
- typing_extensions 4.16.0: PSF-2.0。原文: `LICENSES/typing_extensions-4.16.0/LICENSE`。
- tzdata 2026.4: Apache-2.0。原文: `LICENSES/tzdata-2026.4/LICENSE`、`LICENSES/tzdata-2026.4/LICENSE_APACHE`。

## 内包する部品・ネイティブコード

- NumPyの原文は `python/Lib/site-packages/numpy-2.4.4.dist-info/licenses/` 以下に16件あります。`LICENSES/numpy-2.4.4/` は同名ファイルをまとめる作りのため、ハッシュ一致で確認できた原文は4件/16件です。NumPy本体・内包部品の原文は元のディレクトリ構造ごと保存してください。OpenBLAS、LAPACK、GCCランタイムなどの条件も原文を参照してください。
- SciPyの内包部品は `LICENSES/scipy-1.17.1/LICENSE.txt`、PyTorchの内包部品は `LICENSES/torch-2.5.1+cpu/LICENSE` と `NOTICE` に記載されています。BSD、MIT、Apacheなどの条件を、本体のライセンスだけに置き換えないでください。
- OpenCVの第三者表記は `LICENSES/opencv_python_headless-4.13.0.92/LICENSE-3RD-PARTY.txt`。試験版でファイル名に `ffmpeg` を含むものは0件でした。最終zipでも確認が必要です。
- setuptoolsの内包部品は `python/Lib/site-packages/setuptools/_vendor/` 内の各原文も参照してください。これらは `LICENSES/setuptools-84.0.0/LICENSE` だけでは網羅されません。
- `puyo_core` は本プロジェクトのRust拡張です。参照・移植した citrus610/ama は MIT、puyoai は MIT、rapidhash は BSD-2-Clause、LLVM由来部分は Apache-2.0 WITH LLVM-exception。リポジトリ内の原文は `native/puyo_core/licenses/ama-MIT.txt`、`puyoai-MIT.txt`、`rapidhash-BSD.txt`、`LLVM.txt` です。試験版の `LICENSES/` にはこれらの原文がなく、配布物への収録を要確認です。
- `app/src/phase_j/puyo_live_exact.dll` を確認しました。由来コードとビルドに含まれるランタイムについて、上記のMIT・LLVM等の原文を最終ビルドに合わせて収録してください。
- `app/models/ama_plain/ama_reference.exe` を確認しました。棚卸しでは puyoai/ama（MIT）、jsoncpp（MIT / Public Domain）、glog（BSD-3-Clause）、libstdc++・libgcc（GPL-3.0以降と GCC Runtime Library Exception）、mingw-w64 / winpthreads の表記が挙げられています。これらの原文一式は試験版の `LICENSES/` では確認できません。フォークの由来・権利者表記も含め、最終バイナリのビルド材料と照合が必要です。
- Rust依存crateには MIT / Apache-2.0、unicode-ident の Unicode-3.0 等が含まれます。試験版に `LICENSES/rust/` はありません。最終ビルドの `Cargo.lock` と使用部品に基づく一覧・原文が必要です。
- Microsoft Visual C++ランタイム（`vcruntime140*.dll`、`msvcp140*.dll`、`concrt140.dll` 等）を確認しました。Microsoftの再頒布可能コードの条件が適用されます。試験版の `LICENSES/` に条件文書は見当たらず、配布元・版に対応した条件の確認と原文収録が未了です。

## 常駐版で追加する部品

`packaging/build_release.py` は次のwheelを追加します。下記3件は照合した試験版の31パッケージには含まれていません。版・ライセンス名は棚卸しと組立処理に基づき、原文の最終格納先はzip組立時に確認します。six 1.17.0 は上の一覧に含めています。

- cv2-enumerate-cameras 1.4.0: MIT。追加wheelの原文と `LICENSES/cv2_enumerate_cameras-1.4.0/` の収録を確認。
- pystray 0.19.5: LGPL-3.0。追加wheelの `COPYING.LGPL` と関連原文、著作権表示、対応するソースの入手方法を明示する必要があります。純Pythonの独立パッケージとして `python/Lib/site-packages/pystray/` に置かれ、利用者が差し替えられる構成です。最終zipで差し替えと完全性検査の扱いを確認してください。
- websocket-client 1.8.0: Apache-2.0。追加wheelの原文と `LICENSES/websocket_client-1.8.0/` の収録を確認。

Inno Setup はインストーラーを作る場合に関係する別の部品です。今回の公開対象はzipですが、setup.exeも作る場合は Inno Setup License の原文と同梱条件を別途確認してください。

### 専用の設定窓（2026-11-02、配布既定ONのPC1案・user確認待ち）

`distribution_app_window` 用に以下の8パッケージを固定版で取得します。typing_extensions は既存部品と共通です。2026-11-02にwheelと原文を確認しました。`build_release.py` はwheelを展開して原文を `LICENSES/` へ集め、不足する原文を `packaging/licenses/app_window/` から補います。実際の最終配布物の組み立ては別途確認してください。

- pywebview 6.1: BSD-3-Clause。`LICENSES/pywebview-6.1/LICENSE`。
- bottle 0.13.4: MIT。`LICENSES/bottle-0.13.4/LICENSE`。
- pythonnet 3.2.1: MIT。`LICENSES/pythonnet-3.2.1/LICENSE`。
- clr_loader 0.3.1: MIT。`LICENSES/clr_loader-0.3.1/LICENSE`。
- cffi 2.1.1: MIT-0。`LICENSES/cffi-2.1.1/LICENSE`。Windows x64向けのlibffi部分はMIT系の原文を別に持つため、同版sdistのソースヘッダー6件を `LICENSES/cffi-2.1.1/libffi-NOTICES.txt` に保持します。
- pycparser 3.12: BSD-3-Clause。`LICENSES/pycparser-3.12/LICENSE`。
- typing_extensions 4.16.0: PSF-2.0。`LICENSES/typing_extensions-4.16.0/LICENSE`（上記と共通）。
- proxy_tools 0.1.0: ソースにBSDと記載され、上流の `LICENSE.txt` は2条項BSDの原文です。パッケージmetadataのMIT表記とは異なるため、原文を保持します。`LICENSES/proxy_tools-0.1.0/LICENSE.txt`。wheelが公開されていないため、固定版のsdistから純Python wheelを作ります。

pywebviewのwheelには Microsoft.Web.WebView2 SDK 1.0.2957.106 の Core/WinForms DLLと3アーキテクチャのWebView2Loader.dllが入っています。5ファイルとも同版の公式NuGetパッケージ内のバイナリとSHA-256が一致しました。SDKのBSD-3-Clause原文と第三者通知を `LICENSES/Microsoft.Web.WebView2-1.0.2957.106/LICENSE.txt`、`NOTICE.txt` に収録します。取得元とSHA-256は `packaging/licenses/app_window/README.md` に記録しています。

WebView2 Runtime自体と.NET Frameworkはこの追加wheel群には含まれません。Windowsに導入済みのものを使い、利用できない場合は通常のブラウザへ戻ります。Runtimeを別途再頒布する場合の条件は、上記SDKのライセンスと分けて確認してください。

## 画像・フォントなど

ゲーム認識用の切り抜きテンプレートはゲーム権利者の素材で、コードの MIT の対象ではありません。得点の数字・勝敗のロゴ・×の印・全消しのテロップを同梱する方針です。ゲーム全体の画面切り出し・検証用画像を公開リポジトリや配布物に入れる方針ではありません。本ソフトは非公式です。

独自のマスコット (ゼウスくん)・ロゴの画像は、著作者 diopuyo の決定 (2026-11-02) により、コードと同じ MIT License (ルートの LICENSE) で公開します。フォントはOSのものを指定する構成で、棚卸しではWebフォントや外部JavaScriptライブラリの同梱はありませんでした。

## 最終zipの組立時に補う・照合し直す項目

- [ ] 専用窓の追加wheel群とWebView2 SDKの原文・NOTICEが `LICENSES/` に入り、上記の版・格納先と一致していることを確認する。

- [ ] 最終zipの部品名・版・同梱バイナリをこの一覧と照合し、試験版との差を反映する。追加3部品の原文を含める。
- [ ] flatbuffersのApache-2.0原文が未収録。対応する原文・著作権表示を追加する。
- [ ] ONNX Runtimeの `python/Lib/site-packages/onnxruntime/LICENSE` と `ThirdPartyNotices.txt` を保持する（試験版には両方あり、`LICENSES/` への集約はなし）。
- [ ] NumPyの原文16件をディレクトリ構造付きで保持する。同名の原文を上書きしない。setuptools内包部品の原文も残す。
- [ ] Pythonの `LICENSE.txt`、torchの LICENSE / NOTICE、OpenCV・SciPy等の第三者表記を欠落させない。
- [ ] Rust依存crate、ama / puyoai / rapidhash / LLVM由来コード、ama_reference.exeの内包部品・ランタイムの原文を揃え、最終バイナリの由来と一致させる。
- [ ] Microsoftランタイムの配布元・版・条件文書を照合し、必要な原文を収録する。
- [ ] pystrayのLGPL原文・対応ソースの入手方法を明示し、差し替え可能な構成を最終zipで確認する。
- [ ] FFmpeg DLL、不要な開発用部品・フォント・外部JSが混入していないかを実物で確認する。追加部品があれば表記を足す。
- [ ] 認識用テンプレートと、公開しない画面切り出し・検証用画像を区別して検査する (マスコット・ロゴは MIT に決定済み)。
- [ ] ルートの LICENSE とこの文書をzipに同梱する。READMEの起動ファイル・設定URL・削除手順を最終常駐版で確認する。

未確認項目が残っているため、この文書は最終配布物のライセンス対応完了を宣言するものではありません。

## 11/2 リリース候補の組立追補（仮モデル）

C2 は既存の3 seed ensemble、EX は区切りの版（milestone 150）の3 seed ensemble（s1003/s1004/s1005）を収録する。モデル表記は c2ens3-exm150s3。C2 uses the existing three-seed ensemble; EX uses the milestone-150 three-seed ensemble. 参照ama_reference.exeは除外し、共用PythonはRustの下回りとして保持する（PC1 22:5x判断）。nativeのSHAはd631e3d40cb493eeaf1ad58ce97f5e2d0db2a8823407d5f00617cd15744c136f。ソース7d19831b・Rust 1.97.1・同じ機能フラグを使い、ビルドパスの文字列だけを再対応した。旧版との指紋9項目・乱数200盤面・見本40盤面の評価結果一致を確認した。

NumPyのlicenses以下を同じ相対ディレクトリでLICENSES/numpy-2.4.4へ収録する。実物は17ファイルで、以前の16件という数え方ではdragon4_LICENSE.txtが名前の先頭一致から漏れていた。今回その1件も含める。ONNX Runtimeの原文2件は元の場所を保持しLICENSES/onnxruntimeへも収録する。

pystrayのCOPYINGとCOPYING.LGPLをそのまま収録し、LICENSES/PYSTRAY_SOURCE.mdに同版ソースの入手先と差し替え方法を記載する。追加wheelの原文と、窓の補足原文も収録する。

Rustの原文は上記ビルド元のCargo.lockにあるregistry由来51パッケージに対して102ファイルをLICENSES/rustへ収録する。Cargo.lockはリンクされる部品以外も含みうるため、全51部品がバイナリにリンクされているという意味ではない。由来コード4原文はLICENSES/puyo_coreに収録する。

flatbuffersは同梱25.12.19、原文は上流v25.9.23のLICENSE (Apache-2.0)。版によらない原文としてPC1が11/2に採用を承認した（取得元とSHAはLICENSES/SOURCES.json）。Microsoft VCランタイムはSystem32から取り出したDLLを同梱せず、Python自身とwheelが持つ分だけ保持する。不足時は起動前検査の案内とREADMEに従い、Microsoft公式のX64再頒布可能パッケージを導入する。末尾チェック項目は窓の追補を含め12項目として検査する。

## 上流バイナリに含まれる公開ビルドパス（2026-11-03）

上流の配布物にもともと含まれる作者・CIのパスは、そのまま保持しています。該当バイナリを書き換えたり再ビルドしたりしていません。PythonのBarneyは説明文の例です。以下の件数はASCII文字列の出現数です。

- `python/Lib/site-packages/clr_loader/ffi/dlls/amd64/ClrLoader.dll` — 1件: `/home/benedikt`
- `python/Lib/site-packages/clr_loader/ffi/dlls/x86/ClrLoader.dll` — 1件: `/home/benedikt`
- `python/Lib/site-packages/rpds/rpds.cp312-win_amd64.pyd` — 35件: `C:\Users\runneradmin`
- `python/Lib/site-packages/scipy/_cyutility.cp312-win_amd64.pyd` — 1件: `C:/Users/RUNNER~1`
- `python/Lib/site-packages/scipy/cluster/_hierarchy.cp312-win_amd64.pyd` — 1件: `C:/Users/runneradmin`
- `python/Lib/site-packages/scipy/cluster/_optimal_leaf_ordering.cp312-win_amd64.pyd` — 1件: `C:/Users/runneradmin`
- `python/Lib/site-packages/scipy/cluster/_vq.cp312-win_amd64.pyd` — 1件: `C:/Users/runneradmin`
- `python/Lib/site-packages/scipy/fftpack/convolve.cp312-win_amd64.pyd` — 1件: `C:/Users/runneradmin`
- `python/Lib/site-packages/scipy/interpolate/_rgi_cython.cp312-win_amd64.pyd` — 1件: `C:/Users/runneradmin`
- `python/Lib/site-packages/scipy/io/matlab/_mio5_utils.cp312-win_amd64.pyd` — 1件: `C:/Users/runneradmin`
- `python/Lib/site-packages/scipy/io/matlab/_mio_utils.cp312-win_amd64.pyd` — 1件: `C:/Users/runneradmin`
- `python/Lib/site-packages/scipy/linalg/_decomp_interpolative.cp312-win_amd64.pyd` — 1件: `C:/Users/runneradmin`
- `python/Lib/site-packages/scipy/linalg/_decomp_lu_cython.cp312-win_amd64.pyd` — 1件: `C:/Users/runneradmin`
- `python/Lib/site-packages/scipy/linalg/_decomp_update.cp312-win_amd64.pyd` — 1件: `C:/Users/runneradmin`
- `python/Lib/site-packages/scipy/ndimage/_cytest.cp312-win_amd64.pyd` — 1件: `C:/Users/runneradmin`
- `python/Lib/site-packages/scipy/ndimage/_ni_label.cp312-win_amd64.pyd` — 1件: `C:/Users/runneradmin`
- `python/Lib/site-packages/scipy/optimize/_bglu_dense.cp312-win_amd64.pyd` — 1件: `C:/Users/runneradmin`
- `python/Lib/site-packages/scipy/optimize/_moduleTNC.cp312-win_amd64.pyd` — 1件: `C:/Users/runneradmin`
- `python/Lib/site-packages/scipy/optimize/_trlib/_trlib.cp312-win_amd64.pyd` — 1件: `C:/Users/runneradmin`
- `python/Lib/site-packages/scipy/signal/_peak_finding_utils.cp312-win_amd64.pyd` — 1件: `C:/Users/runneradmin`
- `python/Lib/site-packages/scipy/signal/_sosfilt.cp312-win_amd64.pyd` — 1件: `C:/Users/runneradmin`
- `python/Lib/site-packages/scipy/signal/_upfirdn_apply.cp312-win_amd64.pyd` — 1件: `C:/Users/runneradmin`
- `python/Lib/site-packages/scipy/sparse/_csparsetools.cp312-win_amd64.pyd` — 1件: `C:/Users/runneradmin`
- `python/Lib/site-packages/scipy/sparse/csgraph/_flow.cp312-win_amd64.pyd` — 1件: `C:/Users/runneradmin`
- `python/Lib/site-packages/scipy/sparse/csgraph/_matching.cp312-win_amd64.pyd` — 1件: `C:/Users/runneradmin`
- `python/Lib/site-packages/scipy/sparse/csgraph/_min_spanning_tree.cp312-win_amd64.pyd` — 1件: `C:/Users/runneradmin`
- `python/Lib/site-packages/scipy/sparse/csgraph/_reordering.cp312-win_amd64.pyd` — 1件: `C:/Users/runneradmin`
- `python/Lib/site-packages/scipy/sparse/csgraph/_shortest_path.cp312-win_amd64.pyd` — 1件: `C:/Users/runneradmin`
- `python/Lib/site-packages/scipy/sparse/csgraph/_tools.cp312-win_amd64.pyd` — 1件: `C:/Users/runneradmin`
- `python/Lib/site-packages/scipy/sparse/csgraph/_traversal.cp312-win_amd64.pyd` — 1件: `C:/Users/runneradmin`
- `python/Lib/site-packages/scipy/spatial/_ckdtree.cp312-win_amd64.pyd` — 1件: `C:/Users/runneradmin`
- `python/Lib/site-packages/scipy/spatial/_hausdorff.cp312-win_amd64.pyd` — 1件: `C:/Users/runneradmin`
- `python/Lib/site-packages/scipy/spatial/_qhull.cp312-win_amd64.pyd` — 1件: `C:/Users/runneradmin`
- `python/Lib/site-packages/scipy/spatial/_voronoi.cp312-win_amd64.pyd` — 1件: `C:/Users/runneradmin`
- `python/Lib/site-packages/scipy/spatial/transform/_rigid_transform_cy.cp312-win_amd64.pyd` — 1件: `C:/Users/runneradmin`
- `python/Lib/site-packages/scipy/spatial/transform/_rotation_cy.cp312-win_amd64.pyd` — 1件: `C:/Users/runneradmin`
- `python/Lib/site-packages/scipy/special/_ellip_harm_2.cp312-win_amd64.pyd` — 1件: `C:/Users/runneradmin`
- `python/Lib/site-packages/scipy/special/_specfun.cp312-win_amd64.pyd` — 1件: `C:/Users/runneradmin`
- `python/Lib/site-packages/scipy/special/_ufuncs.cp312-win_amd64.pyd` — 1件: `C:/Users/runneradmin`
- `python/Lib/site-packages/scipy/special/_ufuncs_cxx.cp312-win_amd64.pyd` — 1件: `C:/Users/runneradmin`
- `python/Lib/site-packages/scipy/stats/_ansari_swilk_statistics.cp312-win_amd64.pyd` — 1件: `C:/Users/runneradmin`
- `python/Lib/site-packages/scipy/stats/_biasedurn.cp312-win_amd64.pyd` — 1件: `C:/Users/runneradmin`
- `python/Lib/site-packages/scipy/stats/_qmc_cy.cp312-win_amd64.pyd` — 1件: `C:/Users/runneradmin`
- `python/Lib/site-packages/scipy/stats/_rcont/rcont.cp312-win_amd64.pyd` — 1件: `C:/Users/runneradmin`
- `python/Lib/site-packages/scipy/stats/_sobol.cp312-win_amd64.pyd` — 1件: `C:/Users/runneradmin`
- `python/Lib/site-packages/scipy/stats/_stats.cp312-win_amd64.pyd` — 1件: `C:/Users/runneradmin`
- `python/Lib/site-packages/scipy/stats/_unuran/unuran_wrapper.cp312-win_amd64.pyd` — 1件: `C:/Users/runneradmin`
- `python/Lib/site-packages/sklearn/_cyutility.cp312-win_amd64.pyd` — 1件: `C:/Users/RUNNER~1`
- `python/Lib/site-packages/sklearn/cluster/_hdbscan/_linkage.cp312-win_amd64.pyd` — 1件: `C:/Users/runneradmin`
- `python/Lib/site-packages/sklearn/cluster/_hdbscan/_reachability.cp312-win_amd64.pyd` — 1件: `C:/Users/runneradmin`
- `python/Lib/site-packages/sklearn/cluster/_hdbscan/_tree.cp312-win_amd64.pyd` — 1件: `C:/Users/runneradmin`
- `python/Lib/site-packages/sklearn/ensemble/_gradient_boosting.cp312-win_amd64.pyd` — 1件: `C:/Users/runneradmin`
- `python/Lib/site-packages/sklearn/feature_extraction/_hashing_fast.cp312-win_amd64.pyd` — 1件: `C:/Users/runneradmin`
- `python/Lib/site-packages/sklearn/manifold/_barnes_hut_tsne.cp312-win_amd64.pyd` — 1件: `C:/Users/runneradmin`
- `python/Lib/site-packages/sklearn/metrics/_dist_metrics.cp312-win_amd64.pyd` — 1件: `C:/Users/runneradmin`
- `python/Lib/site-packages/sklearn/metrics/_pairwise_distances_reduction/_radius_neighbors.cp312-win_amd64.pyd` — 1件: `C:/Users/runneradmin`
- `python/Lib/site-packages/sklearn/metrics/_pairwise_distances_reduction/_radius_neighbors_classmode.cp312-win_amd64.pyd` — 1件: `C:/Users/runneradmin`
- `python/Lib/site-packages/sklearn/neighbors/_ball_tree.cp312-win_amd64.pyd` — 1件: `C:/Users/runneradmin`
- `python/Lib/site-packages/sklearn/neighbors/_kd_tree.cp312-win_amd64.pyd` — 1件: `C:/Users/runneradmin`
- `python/Lib/site-packages/sklearn/neighbors/_quad_tree.cp312-win_amd64.pyd` — 1件: `C:/Users/runneradmin`
- `python/Lib/site-packages/sklearn/tree/_criterion.cp312-win_amd64.pyd` — 1件: `C:/Users/runneradmin`
- `python/Lib/site-packages/sklearn/tree/_partitioner.cp312-win_amd64.pyd` — 1件: `C:/Users/runneradmin`
- `python/Lib/site-packages/sklearn/tree/_splitter.cp312-win_amd64.pyd` — 1件: `C:/Users/runneradmin`
- `python/Lib/site-packages/sklearn/tree/_tree.cp312-win_amd64.pyd` — 1件: `C:/Users/runneradmin`
- `python/Lib/site-packages/sklearn/tree/_utils.cp312-win_amd64.pyd` — 1件: `C:/Users/runneradmin`
- `python/Lib/site-packages/sklearn/utils/_vector_sentinel.cp312-win_amd64.pyd` — 1件: `C:/Users/runneradmin`
- `python/python312.dll` — 2件: `C:/Users/Barney')`
