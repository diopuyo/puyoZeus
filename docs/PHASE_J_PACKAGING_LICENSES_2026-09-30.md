# Phase J 配布物のライセンス整理 (2026-09-30)

対象: `PuyoLive/` (埋め込み Python 3.12.3 + CPU 版 torch + アプリ)。一覧は `packaging/list_licenses.py` が
各パッケージの `dist-info/METADATA` から機械抽出したもの。**表記の転記であり、法務確認ではない**。
配布物には各ライブラリの LICENSE/NOTICE 計 44 ファイルが `LICENSES/<配布名>/` に入る (`build_bundle.py`)。

## 1. 学習済みモデルの再配布 (判断欄)

| 項目 | 状態 |
|---|---|
| 学習済みモデル (`models/exchange_event_v1〜v4`、`cnn_*.pt`、`landing_counter_prob_v1` 等) の再配布 | **user 承認済み 2026-09-30 (「問題なし」)** |
| プロジェクト自身 (`src/` `scripts/`) のライセンス表記・著作権表示 | **user 判断待ち** (現状リポジトリに LICENSE ファイルなし。配布物へ何を書くかの決定が必要) |

## 2. 同梱ライブラリ (自動抽出)

| パッケージ | 版 | ライセンス表記 | 種別 |
|---|---|---|---|
| attrs | 26.1.0 | MIT | 寛容 |
| filelock | 4.0.7 | MIT | 寛容 |
| fsspec | 2026.9.0 | BSD-3-Clause | 寛容 |
| Jinja2 | 3.1.6 | BSD | 寛容 |
| joblib | 1.5.3 | BSD-3-Clause | 寛容 |
| jsonschema / jsonschema-specifications | 4.26.0 / 2025.9.1 | MIT | 寛容 |
| MarkupSafe | 3.0.3 | BSD-3-Clause | 寛容 |
| mpmath / sympy | 1.3.0 / 1.13.1 | BSD | 寛容 |
| networkx | 3.7 | BSD-3-Clause | 寛容 |
| numpy | 2.4.4 | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 | 寛容 (+ 同梱 DLL に GPL+例外あり、§3-2) |
| opencv-python-headless | 4.13.0.92 | Apache-2.0 | 寛容 (+ **LGPL の FFmpeg DLL**、§3-1) |
| pandas | 3.0.2 | BSD-3-Clause | 寛容 |
| pillow | 12.2.0 | MIT-CMU (HPND) | 寛容 (+ FreeType は FTL/GPLv2 の選択制、§3-4) |
| python-dateutil | 2.9.0.post0 | Apache-2.0 / BSD-3-Clause (二重) | 寛容 |
| referencing / rpds-py | 0.37.0 / 2026.6.3 | MIT | 寛容 |
| scikit-learn / threadpoolctl | 1.8.0 / 3.7.0 | BSD-3-Clause | 寛容 |
| scipy | 1.17.1 | BSD-3-Clause (+ 同梱 OpenBLAS/GCC ランタイム、§3-2) | 寛容 |
| setuptools / six | 84.0.0 / 1.17.0 | MIT | 寛容 |
| torch | 2.5.1+cpu | BSD-3-Clause (+ third_party 各種、§3-3) | 寛容 |
| typing_extensions | 4.16.0 | PSF-2.0 | 寛容 |
| tzdata | 2026.4 | Apache-2.0 | 寛容 |
| Python 3.12.3 (embeddable) | 3.12.3 | PSF License | 寛容。同梱 libssl/libcrypto は OpenSSL 3.x (Apache-2.0)、sqlite3 (パブリックドメイン)、libffi (MIT) |
| VC++ ランタイム DLL | 14.x | Microsoft 再頒布可能コード | §3-5 |
| puyo_core (自社 Rust 拡張) | 0.1.0 | 自社 (依存 crate は MIT/Apache-2.0 が中心) | §3-6 |

## 3. コピーレフト系・要注意の扱い

### 3-1. FFmpeg 4.4.6 (LGPL-2.1-or-later) — `cv2/opencv_videoio_ffmpeg4130_64.dll` (28.6 MB)

- 実測 (DLL 内の文字列): 「libavcodec license: LGPL version 2.1 or later」、`--enable-gpl` / `--enable-nonfree` /
  `--enable-version3` **なし**。有効な外部コーデックは libaom / libopenh264 / libvpx。
- 配布条件 (LGPL-2.1):
  1. **別 DLL のまま動的リンクで同梱する** (現状のとおり。ユーザーが差し替え可能であること。DLL を静的に取り込まない・改変しない・
     ファイル名を変えて隠さない)。
  2. LGPL-2.1 の全文と著作権表示を同梱する (`cv2/LICENSE-3RD-PARTY.txt` に全文があり、`LICENSES/opencv-python-headless/` へ収集済み)。
  3. **対応するソースの入手先を明記する** (書面での申し出、または配布物内の記載):
     FFmpeg 4.4.6 ソース `https://ffmpeg.org/releases/ffmpeg-4.4.6.tar.xz` と、OpenCV の FFmpeg ラッパ/ビルド手順
     (`https://github.com/opencv/opencv-python`、`https://github.com/opencv/opencv`)。配布物の README_ja.txt に上記を記載済。
  4. アプリ側は LGPL の派生物にならない (ライブラリを呼ぶだけ)。ただし利用者が FFmpeg を差し替えて動作確認できる、
     というリバースエンジニアリング許諾の条項に反する利用規約を付けない。
- **代替**: 実機入力 (`source=dshow`) だけなら FFmpeg は使わない (動画ファイル入力 = `source=video` の時だけ)。
  `source=video` を配布版から外し、この DLL を配布物から除けば LGPL の義務自体が消える。要否は user 判断。
- **特許の注意**: libopenh264 は Cisco の特許許諾が「Cisco がビルドしたバイナリ」に限られる。OpenCV 版バイナリでの H.264 の
  特許リスクは本書では判断できない (H.264 の再生/デコードを配布する行為に当たるか要確認)。

### 3-2. GCC ランタイム (libgfortran/libgcc) — numpy.libs / scipy.libs の `libscipy_openblas*.dll`

- ライセンス: **GPL-3.0-or-later WITH GCC-exception-3.1** (numpy の LICENSE.txt に全文)。OpenBLAS は BSD-3-Clause。
- GCC ランタイム例外により、**非 GPL のアプリと一緒に配布できる**。ソース提供義務は生じない
  (例外の条件: 「適格なコンパイルプロセス」でリンクされたターゲットコードであること — numpy/scipy の公式 wheel は該当)。
- 条件: numpy の LICENSE.txt (GPL 文と例外文を含む) を同梱する (収集済み)。wheel の DLL を改変・再ビルドしない。

### 3-3. torch と第三者ライブラリ

- torch: BSD-3-Clause。`NOTICE` (456 行) と `LICENSE` (9,181 行、第三者ライセンス集を含む) を同梱 (収集済み)。
- `libiomp5md.dll` (Intel/LLVM OpenMP ランタイム): 表記が torch の LICENSE 内にある。**個別の条件は未確認 → 要確認**。
- その他 (oneDNN = Apache-2.0、protobuf = BSD、fbgemm = BSD、sleef = Boost、asmjit = zlib、libuv = MIT) は寛容。

### 3-4. Pillow の FreeType

- FreeType は「FreeType License (FTL、BSD 風、謝辞必須)」または GPLv2 の選択。Pillow の LICENSE は FTL を選択している旨を記す。
- **条件**: 配布物のドキュメントに「Portions of this software are copyright © The FreeType Project (www.freetype.org). All rights reserved.」
  相当の謝辞を入れる (README_ja.txt に記載済)。Pillow の LICENSE (収集済み) が根拠文書になる。

### 3-5. Microsoft VC++ ランタイム DLL

- 同梱対象 (`build_bundle.py: VC_RUNTIME_DLLS`): `vcruntime140.dll` `vcruntime140_1.dll` `msvcp140.dll`
  `msvcp140_1.dll` `msvcp140_2.dll` `concrt140.dll`。取得元は VC++ 再頒布可能パッケージが入れた `C:\Windows\System32`。
- これらは Microsoft の「再頒布可能コード」(Visual C++ 2015-2022 の CRT / 標準ライブラリ) に該当する。
  条件の要点 (**Visual Studio のライセンス条項 "Distributable Code" の記述に基づく理解であり、原文で確認すること**):
  アプリのライセンスに「Microsoft の免責を含む」旨と、配布先が Microsoft のコードを改変・逆コンパイルしない旨を含める、
  自分のアプリと一体で配布する (単体配布しない)。
- **UCRT (`api-ms-win-crt-*`, `ucrtbase.dll`) は Windows 10 以降の OS 標準**のため同梱しない (Windows 10/11 のみ対象)。

### 3-6. puyo_core (Rust)

- 依存 crate (pyo3 / rayon / numpy 等) は MIT または Apache-2.0 が中心。配布バイナリには crate の著作権表示・ライセンス文が必要
  (`Cargo.lock` から `cargo about` 等で生成して同梱する。**未実施**)。

## 4. 配布時に配布物へ入れる文書 (チェックリスト)

| 項目 | 状態 |
|---|---|
| 各ライブラリの LICENSE/NOTICE (`LICENSES/`) | 自動収集済み (44 ファイル。Rust crate 分は未) |
| FFmpeg 4.4.6 のソース入手先の明記 (LGPL) | 配布物の README_ja.txt へ記載済 (内容の最終確認は user) |
| FreeType の謝辞 | README_ja.txt へ記載済 |
| Microsoft 再頒布可能コードの帰属・条件の記載 | 帰属は README_ja.txt に記載済。**Microsoft の条項原文での確認は未** |
| 本アプリ自身のライセンス/著作権表示 | user 判断待ち (§1) |
| 学習済みモデルの再配布 | user 承認済み (2026-09-30) |

## 5. 判断が必要な選択肢 (user へ)

1. `source=video` を配布版から外して FFmpeg DLL を除くか (LGPL 義務と H.264 特許の懸念を根本から消せる。
   代償: 保存動画での動作確認・デモが配布版で出来なくなる)。
2. 本アプリ自身のライセンスの種類と著作権者表示。
3. 法務の最終確認をどこまで外部に依頼するか (本書は自動抽出と実測の整理であり、法的助言ではない)。
