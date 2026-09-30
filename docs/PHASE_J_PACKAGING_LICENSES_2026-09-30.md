# Phase J 配布物のライセンス整理 (2026-09-30)

対象: `PuyoLive/` (埋め込み Python 3.12.3 + CPU 版 torch + アプリ)。一覧は `packaging/list_licenses.py` が
各パッケージの `dist-info/METADATA` から機械抽出したもの。**表記の転記であり、法務確認ではない**。
配布物には各ライブラリの LICENSE/NOTICE 計 44 ファイルが `LICENSES/<配布名>/` に入る (`build_bundle.py`)。

## 1. 学習済みモデルの再配布 (判断欄)

| 項目 | 状態 |
|---|---|
| 学習済みモデル (`models/exchange_event_v1〜v4`、`cnn_*.pt`、`landing_counter_prob_v1` 等) の再配布 | **user 承認済み 2026-09-30 (「問題なし」)** |
| アプリ自身 (`src/` `scripts/` `packaging/` と学習済みモデル) のライセンス | **MIT (user 決定 2026-09-30)**。リポジトリ直下 `LICENSE` (英語原文、`Copyright (c) 2026 diopuyo`) を追加し、配布物の直下と `app/LICENSE` (MANIFEST 対象) に同梱。README_ja.txt に日本語の要約・非公式表記・無保証を記載。**リポジトリの公開 (visibility 変更) は行っていない** |

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
| opencv-python-headless | 4.13.0.92 | Apache-2.0 | 寛容 (FFmpeg DLL は配布物から除去済み、§3-1) |
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

### 3-1. FFmpeg (LGPL) — 配布物から除去した (user 決定 2026-09-30)

- 配布物から `cv2/opencv_videoio_ffmpeg4130_64.dll` (FFmpeg 4.4.6、LGPL-2.1-or-later、28.6 MB) を除去した
  (`build_bundle.py: FFMPEG_GLOB`、再ビルドのたびに削除。配布物内に `ffmpeg` / `avcodec` を名前に持つファイルは 0 件)。
- ランチャーは `source=video` を受け付けず「配布版は OBS 仮想カメラ等の入力 (DirectShow) のみ対応」と案内する。
  開発用の `--dev-allow-video` は残る (配布版では動画デコーダが無く動かない。開発環境 (WSL) の動画入力は既存コードのまま)。
- `cv2.pyd` 自体に FFmpeg のコードは含まれない (文字列走査で `libavcodec` / `FFmpeg version` / `LGPL` なし)。FFmpeg は
  プラグイン DLL としてのみ供給されていた。よって **LGPL の配布条件 (ライセンス全文・ソース入手先の明記・差し替え可能性) は不要になった**。
  libopenh264 / libvpx / libaom を介した H.264 等の特許上の懸念も、この DLL とともに消えた。
- 補足: `cv2/LICENSE-3RD-PARTY.txt` (opencv-python wheel 由来) には FFmpeg の LGPL 文が残っている。wheel の文書を改変しないため
  そのまま同梱するが、配布物に FFmpeg は含まれない (README_ja.txt に明記)。

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
| FreeType の謝辞 | README_ja.txt へ記載済 |
| Microsoft 再頒布可能コードの帰属・条件の記載 | 帰属は README_ja.txt に記載済。**Microsoft の条項原文での確認は未** |
| 本アプリ自身のライセンス/著作権表示 | user 判断待ち (§1) |
| 学習済みモデルの再配布 | user 承認済み (2026-09-30) |

## 5. 判断が必要な選択肢 (user へ)

1. ~~`source=video` の扱い~~ → **解決済み (2026-09-30)**: 配布版から除外し FFmpeg も外した (§3-1)。
2. ~~アプリ自身のライセンス~~ → **解決済み (2026-09-30)**: MIT (§1、§6)。
3. 法務の最終確認をどこまで外部に依頼するか (本書は自動抽出と実測の整理であり、法的助言ではない)。
4. 「ぷよぷよ」の商標表記 (README_ja.txt に「非公式・セガとは無関係・登録商標」を記載済み)。ゲーム画面由来の画像
   (`models/ui_templates/**` のテンプレート画像はゲーム画面の切り抜き) を MIT で再配布してよいかは、著作権者 (セガ) の許諾の
   問題で MIT の可否とは別。**本書では判断していない (user 判断欄)**。

## 6. 同梱ライブラリと MIT の両立確認 (一行ずつ)

判定の意味: 「両立」= 本ツールを MIT で配布しつつ、そのライブラリを同梱してよい (条件は各行に記す)。ライセンス表記は §2 の自動抽出
(METADATA) による。**法的助言ではなく、一般に知られた条件の照合**。

| 対象 | ライセンス | MIT との両立 | 守る条件 (配布物での対応) |
|---|---|---|---|
| attrs / filelock / jsonschema / jsonschema-specifications / referencing / rpds-py / setuptools / six | MIT | 両立 | 著作権表示・許諾表示の保持 (LICENSES/ へ収集) |
| fsspec / joblib / MarkupSafe / networkx / scikit-learn / threadpoolctl / torch / mpmath / sympy / Jinja2 / pandas | BSD-3-Clause (mpmath/sympy は BSD) | 両立 | 著作権表示・条件・免責の保持、名前を宣伝に使わない (LICENSES/) |
| numpy | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 | 両立 | LICENSE.txt (同梱ライブラリの分を含む) を保持 |
| scipy | BSD-3-Clause 系 | 両立 | LICENSES/scipy を保持 |
| Pillow | MIT-CMU (HPND) | 両立 | 著作権表示の保持。FreeType は FTL を選択 (謝辞を README に記載) |
| python-dateutil | Apache-2.0 / BSD-3-Clause 二重 | 両立 (BSD 側を選べる) | 表示の保持 |
| tzdata / opencv-python-headless | Apache-2.0 | 両立 (Apache-2.0 の成果物を MIT のものと一緒に配布可) | LICENSE と NOTICE の保持、改変時の明示 (改変なし) |
| typing_extensions / Python 3.12 本体 | PSF-2.0 | 両立 | PSF ライセンス文と著作権の保持 |
| OpenSSL 3 (python 同梱 libssl/libcrypto) | Apache-2.0 | 両立 | 同上 |
| sqlite3 / libffi / expat | パブリックドメイン / MIT / MIT | 両立 | libffi の著作権表示 (Python 同梱の文書) |
| OpenBLAS (numpy.libs / scipy.libs の DLL) | BSD-3-Clause | 両立 | numpy/scipy の LICENSE に含まれる文の保持 |
| libgfortran / libgcc (同 DLL に静的リンク) | GPL-3.0-or-later WITH GCC-exception-3.1 | 両立 (ランタイム例外により非 GPL のアプリと同梱配布可) | 例外文・GPL 文 (numpy の LICENSE.txt) の保持。DLL を改変しない |
| torch の第三者部品 (oneDNN Apache-2.0 / protobuf BSD / fbgemm BSD / sleef Boost / asmjit zlib / libuv MIT) | 寛容 | 両立 | torch の LICENSE / NOTICE の保持 |
| libiomp5md.dll (torch 同梱の OpenMP ランタイム) | 表記は torch の LICENSE 内 | **未確認** (個別条件を原文で確認していない) | 確認するまで「両立」と断定しない |
| Microsoft VC++ ランタイム DLL (vcruntime140 等) | Microsoft 再頒布可能コード | 両立 (MIT の適用範囲外のバイナリ。Microsoft の条件に従う) | 配布物の一部として同梱、改変しない。**条項原文は未確認** (§3-5) |
| puyo_core の Rust 依存 crate (pyo3 / rayon / numpy crate / ndarray 等) | crates.io 表記で MIT または Apache-2.0 が中心 (target-lexicon は Apache-2.0 WITH LLVM-exception) | 両立の見込み (**crate 一覧からの照合は未実施**) | 各 crate の著作権表示の同梱 (`cargo about` 等で生成。未実施) |
| FFmpeg (LGPL) | — | **同梱しないため対象外** (§3-1) | なし |
| 学習済みモデル (`models/`) | 本ツールと同じ MIT | user 承認済み (2026-09-30) | LICENSE に含まれる |

結論: 確認できた範囲では、**同梱ライブラリと MIT の間に矛盾はない**。コピーレフトで配布条件を課すのは GCC ランタイム (例外により非 GPL 可) のみで、
LGPL の FFmpeg は除去した。未確認は「libiomp5md の個別条件」「Microsoft 条項原文」「Rust crate の一覧照合」の 3 点。
