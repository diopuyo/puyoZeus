# puyo_core (Rust ネイティブ拡張) 進捗メモ — 2026-08-12

user確定指示 (2026-08-12): ぷよぷよ連鎖シミュレーション+ビームサーチの Rust
ネイティブ拡張。目標=深さ13〜16手/幅250のビームサーチを1探索100ms以内
(8スレッドで10〜20msが理想)。

**14:40 PCシャットダウン前のチェックポイントコミット**。このファイルは
再開する別セッションの自分が読む前提で書く。

## どこまで終わったか (完了)

優先順位「パリティ > ビームサーチ > ベンチ > 並列化」の順で実装、
**全項目着手済み・パリティ100%通過**。

1. **連鎖エンジン本体** (`src/bitboard.rs`): `src/chain_bitboard.py` の
   ビットボード方式 (ama citrus610 移植) をスカラー Rust へ再移植。
   `simulate_chain()` が `chain_bitboard.simulate_batch_with_approx_score`
   とビット一致 (score_approx は連結ボーナス0近似、chain_bitboard と同じ土俵)。
   幽霊連鎖ルール (`exclude_hidden_row_from_pop`) も対応、`simulate_batch`
   (厳密版、幽霊連鎖ルール対応) とも一致確認済み。
   設置列挙 `enumerate_placements()` は `_enumerate_placement_boards`
   (indicators_v2.py:3389、窒息除外あり) 準拠。
   Rust 単体テスト 6件 (`cargo test`) + Python パリティテスト 3件が
   全通過 (下記参照)。

2. **ビームサーチ** (`src/beam.rs`): running-max 方式 (`_near_future_known_expand`
   と同じ意味論 = 置く→即発火→残骸を次の手へ引き継ぐ、発火するかどうかを
   選べるゲームではない)。深さ・幅・幽霊連鎖ルール・並列有無を引数化。
   最良スコア・最良手順 (親ポインタで復元)・深さごとの running-max 配列を返す。
   動作確認済み (窒息盤面・空pairsでもpanicしない)。

3. **PyO3 バインディング** (`src/lib.rs`): `simulate_chain_py` /
   `enumerate_placements_py` / `beam_search_py` の3関数を公開。
   **設計判断**: numpy crate 依存を避け、盤面は flat な `Vec<u8>` (長さ78、
   行優先) で受け渡す (ビルド時間・リスク削減。task指示の「numpy盤面を
   受け取る薄いラッパ」は Python 側 `src/puyo_core_bridge.py` が担う設計)。
   rayon スレッドプールはプロセス全体で1個だけ再利用
   (`OnceLock<ThreadPool>`、探索ごとに作り直すとリアルタイム用途では
   無視できないオーバーヘッドになるため。初回呼び出しの num_threads で
   確定する制約あり、呼び出し側が毎回異なるスレッド数を渡す運用は未対応)。

4. **Python ブリッジ** (`src/puyo_core_bridge.py`, プロジェクトルート):
   optional import (`NATIVE_AVAILABLE` フラグ) + Python フォールバック実装。
   フォールバックは `src/indicators_v2.py` から import せず独立複製
   (編集禁止ファイルとの結合を避けるため)。フォールバックの
   `exclude_hidden_row_from_pop=True` は未対応 (`NotImplementedError`,
   fail-silent 回避)。

5. **パリティテスト** (`tests/test_puyo_core_parity.py`): 実盤面600件
   (`data/indicators_v2/boards_lean_phase_l_2026-08-11/` から12動画×60盤面)
   で3テスト全通過:
   - `test_simulate_chain_parity_with_chain_bitboard`: 600/600 一致
     (chain_count/total_erased/total_ojama/score_approx/final_board全て)
   - `test_simulate_chain_parity_ghost_rule`: 600/600 一致 (幽霊連鎖ルール、
     `simulate_batch` 厳密版と比較)
   - `test_enumerate_placements_parity_with_indicators_v2`: 120盤面×3ペアで
     `_enumerate_placement_boards` と盤面集合完全一致

6. **ベンチ** (`scripts/_bench_puyo_core_2026-08-12.py`): 実装済み・実行済み。
   **⚠️ 計測時、WSL側で148収集ジョブが10並列走行中 (loadavg 1分=28.6、
   16コアに対しほぼ2倍の過負荷)** だったため、数値は参考値に留まる
   (下記「ベンチ実測」参照、システム負荷が下がってからの再測定を推奨)。

## ビルド再現コマンド

前提: WSL Ubuntu に **build-essential (gcc) が未導入 + sudo がパスワード要求**
のため通常の `cc` リンカが使えない。回避策として pip配布の `ziglang`
(zig の cc互換ドライバ) を C リンカ代わりに使っている
(`native/puyo_core/zigcc.sh`, `~/.local/bin/zigcc` にコピー済み、
`native/puyo_core/.cargo/config.toml` で `linker` に指定)。
**この回避策は再現に必須** (gcc が導入できたら `.cargo/config.toml` の
linker 指定は削除してよい)。

```bash
# 1. Rust ツールチェーン (未導入なら)
curl https://sh.rustup.rs -sSf -o /tmp/rustup-init.sh
sh /tmp/rustup-init.sh -y --default-toolchain stable --profile minimal

# 2. zig (cc代替リンカ、gcc不要) — WSL ネイティブ fs (home) に置く
#    (drvfs=/mnt/c 上に置くと展開が極端に遅い、実測で数十分ハング)
python3 -m venv ~/zigvenv
~/zigvenv/bin/pip install ziglang
mkdir -p ~/.local/bin
cp /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer/native/puyo_core/zigcc.sh ~/.local/bin/zigcc
chmod +x ~/.local/bin/zigcc
# native/puyo_core/.cargo/config.toml が ~/.local/bin/zigcc を linker として参照する
# (このパスはこの環境固有、別環境では書き換えるか gcc 導入で削除)

# 3. maturin (プロジェクト venv へ)
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer
./venv/bin/python -m pip install maturin

# 4. ビルド + インストール (-j 4 制限、148収集ジョブと競合しないよう)
source $HOME/.cargo/env
cd native/puyo_core
VIRTUAL_ENV=$(pwd)/../../venv CARGO_BUILD_JOBS=4 ../../venv/bin/maturin develop --release -j 4

# 5. Rust 単体テスト
CARGO_BUILD_JOBS=4 cargo test -j 4 --release

# 6. Python パリティテスト
cd ../..
PYTHONPATH=. ./venv/bin/python -m pytest tests/test_puyo_core_parity.py -v

# 7. ベンチ
PYTHONPATH=. ./venv/bin/python -m scripts._bench_puyo_core_2026-08-12
```

## ベンチ実測 (2026-08-12、⚠️高負荷下の参考値)

盤面: puyo数29の中盤盤面。beam_width=250、色4色ランダムツモ。

| 項目 | 値 |
|---|---|
| 1盤面シミュレーション (median, n=300) | 1.2〜2.0 µs |
| 深さ13・単スレッド (median, n=5) | 14.5〜55.5 ms (実行毎に大きく変動) |
| 深さ13・8スレッド並列 | 58.6〜62.1 ms |
| 深さ16・単スレッド | 41.2〜59.5 ms |
| 深さ16・8スレッド並列 | 70.2〜109.1 ms |

**目標 (100ms以内) の達成可否**: 単スレッドでは条件により達成 (14.5ms) も
不達成 (59.5ms) もあり、**現状の計測はシステム負荷 (loadavg 28.6、
16コアに対し約2倍) の影響が大きく信頼できない**。並列が単スレッドより
遅いのも同じ理由 (rayon 8スレッドが既存の148収集10並列プロセスと直接競合)。
**148収集完了後 (または一時停止して) の再測定が必須**。

1盤面シミュレーションは 2µs 未満と極めて高速 (numpy版の
`chain_bitboard.simulate_single` は同等盤面で概算 数百µs〜1ms オーダーの
はず、要再比較) なので、探索が遅い場合はビームサーチのオーバーヘッド
(候補生成・ソート・Vec確保) 側を疑うべき。深さ13→16で単スレッドの伸びが
線形的でない (むしろ深さ13が遅く出た回もある) のも高負荷ノイズの疑いが濃い。

## 次にやること (優先順)

1. **148収集完了後の再ベンチ** (最優先、高負荷ノイズを排除した信頼できる数値)。
   `scripts/_bench_puyo_core_2026-08-12.py` はそのまま使える。
2. ビームサーチのアロケーション最適化 (現状: 深さごとに `Vec<BeamNode>` を
   `flat_map().collect()` — 幅250×22候補=5500件/深さのソート・複製コストを
   `Vec::with_capacity` 事前確保やスコアのみの部分ソート
   (`select_nth_unstable`) に変えられる余地あり。ベンチで律速箇所を
   `cargo flamegraph` 等で特定してから着手すべき (推測で最適化しない)。
3. `scripts/_verify_beam_miss_2026-08-09.py` への接続
   (task指示: 「今回は走らせなくてよい、接続可能な形にだけ」)。
   `src/puyo_core_bridge.beam_search(board, pairs, beam_width, ...)` は
   `Board` を受け取り `BeamSearchResult.best_score` (score_approx、
   お邪魔換算前の素点) を返すので、既存スクリプトの `_brute_force` 関数と
   同じ土俵で比較可能 (score_approx / OJAMA_RATE_STANDARD でお邪魔換算に
   揃える必要あり、既存スクリプトは `calculate_chain_score` の厳密値を
   使っているため、比較時は近似 vs 厳密の差を認識すること)。まだ実際に
   繋いで走らせてはいない。
4. 並列化の効果を正しく測るため、スレッドプールサイズ可変対応
   (現状 `OnceLock` は初回呼び出しのスレッド数で固定、複数スレッド数を
   比較するベンチでは毎回プロセスを再起動する必要がある。今回の
   `_bench_puyo_core_2026-08-12.py` はこの制約を回避できておらず、
   None (単スレッド) → 8スレッド の順で呼んでいるため8スレッド版の
   プール構築コストは1回目だけで、2回目以降は再利用されるはず。
   ただし単スレッド計測は毎回 `parallel=false` 経路 (プール未使用) なので
   問題ない。念のため要再確認)。
5. score_approx (連結ボーナス0近似) と厳密スコア (`src/scoring.py`
   の連結ボーナス反映版) の差がビームサーチの探索順位にどれだけ影響するか
   は未検証 (chain_bitboard.py 自身の既知の限界、docstring 参照)。
   価値検証時に論点になる可能性あり (今回のタスクは実装のみで価値検証は
   別、との user 指示通り本タスクでは踏み込まない)。

## 既知の制約・注意点

- **linker 回避策は環境固有**: `native/puyo_core/.cargo/config.toml` の
  `linker = "/home/ryouj/.local/bin/zigcc"` は絶対パス。別マシン/別WSL
  distroに移行する場合は書き換えが必要 (または gcc 導入で本設定自体を削除)。
- **numpy crate 未使用**: PyO3 バインディングは `Vec<u8>` (78要素flat) で
  やり取りする。numpy 配列との変換は `src/puyo_core_bridge.py` が担う。
  将来 numpy crate を追加する場合はゼロコピー化の余地があるが、
  現時点ではビルド簡素化を優先した。
- **フォールバック実装の幽霊連鎖ルール未対応**: 拡張なし環境で
  `exclude_hidden_row_from_pop=True` を呼ぶと `NotImplementedError`
  (fail-silent回避、CLAUDE.md規約)。
- **rayon スレッドプールは1プロセス1回だけ構築**: 上記「次にやること4」参照。
- **`native/puyo_core/target/` はビルド出力のため `.gitignore` 済み**
  (`native/puyo_core/.gitignore`)。コミットはソースのみ。

## 作成ファイル一覧

- `native/puyo_core/Cargo.toml`
- `native/puyo_core/pyproject.toml`
- `native/puyo_core/.gitignore`
- `native/puyo_core/.cargo/config.toml`
- `native/puyo_core/zigcc.sh`
- `native/puyo_core/src/bitboard.rs`
- `native/puyo_core/src/beam.rs`
- `native/puyo_core/src/lib.rs`
- `native/puyo_core/PROGRESS.md` (本ファイル)
- `native/puyo_core/Cargo.lock` (依存バージョン固定、コミット対象)
- `src/puyo_core_bridge.py`
- `tests/test_puyo_core_parity.py`
- `scripts/_bench_puyo_core_2026-08-12.py`
