"""boards_lean 系 npz (盤面グリッド原情報) → labeled_win 形式 CSV 変換ツール。

## 背景 (2026-08-12 user選択肢C確定)
CSV は使い捨ての派生物、npz (boards_lean_phase_l_2026-08-11/*.npz 等) が
恒久資産という前提で設計する。指標セットが増減しても npz 群を再収集せず
(動画も不要)、本ツールで CSV を安く再生成できるようにする。

## 薄い委譲構造 (指標が増減してもこのファイルは触らない)
指標の実計算は 100% `src/indicators_v2.py` の関数群に委譲する。本ツールが
持つのは「npz の 1 行 → Board 再構築 → レジストリ内の関数を呼んで列を書く」
という薄いループだけ。**新指標を追加したい場合は `GRID_ONLY_INDICATORS`
(または `GRID_ONLY_HEAVY_INDICATORS`) に 1 行足すだけで済む** (INDICATOR_
COLUMNS 末尾追加ルールと同じ精神)。

## 指標大整理 (2026-08-12 user確定、docs/INDICATOR_REORG_PROPOSAL_2026-08-12.md
「決定記録」節) の反映
- **a-1 完全重複の削除**: `*_raw` 列 (score と定数倍の完全重複) は CSV に
  一切出力しない。`saturated_chain_count` (current_max_chain と全19万場面で
  完全一致) は full profile のレジストリから削除。
  `absorption_capacity` (board_puyo_total と完全重複) は本ツールでは元々
  未収集のため対応不要 (既に GRID_ONLY_INDICATORS/HEAVY のいずれにも無い)。
- **b-1 center_bulge の分解**: 合成版 `center_bulge` (indicators_v2.py側は
  backwards compat のため関数自体は残す) の代わりに `center_bulge_color`
  (色ぷよのみ) / `center_bulge_ojama` (おじゃまのみ) の2列を出力する。
  新指標のため own/diff 両方を出す (DIFF_KEEP_OWN_NEW_COLUMNS)。
- **b-2 「相手との差」列**: `_analyze_reorg_diff_2026-08-12.py` の
  merge_asof(direction="backward") 方式を移植 (自分の各時刻に「相手の直近
  確定値」を対応付ける、対応成功率99.3%実測)。対象列の分類は下記
  DIFF_* 定数群を参照 (b-2決定記録の4分類 + 例外1分類、一括変換時の
  「diff化してよい列」明示リスト化 = 2026-08-10 user恒久指示に従う)。
  色ぷよ×おじゃまの関係は非単調 (おじゃま≈0なら色ぷよ多い=有利/材料律、
  おじゃま多いかつ色ぷよ多いなら「潰しが刺さった」不利シグナルの可能性、
  2026-08-12 user伝授) なため、own+diff+比率+交互作用の4列で表現する
  (単一の向きを決め打ちしない、詳細は DIFF_KEEP_OWN_PAIR_COLUMNS 直上の
  コメント)。
- **全消しボーナス予約中フラグ (`all_clear_bonus_pending`)**: 2026-08-12
  user伝授 (設計訂正版)。本質は「盤面が空かどうか」という瞬間状態ではなく、
  「全消しボーナス (2100点、おじゃま約30個相当) が未消費で残っている」
  という**持続状態**。npz の score 列 (post-hoc 復元可能) を使い、通常の
  落下ボーナス (最大 MAX_DROP_BONUS_SCORE=250) を大きく超え、かつ連鎖に
  紐付かない (chain_mechanism 未タグ) スコア増分を全消しボーナス計上と
  みなして ON にし、次の連鎖 (ボーナス消費) で OFF にする状態機械として
  実装する (詳細は `_compute_all_clear_bonus_pending` docstring)。相手の
  直近own値をそのまま carry する `opp_all_clear_bonus_pending` も出力する
  (diff ではない。フラグの差分は無意味で、相手が今ボーナス保持中かという
  生の状態そのものが受け側の判断材料になるため)。score OCR が信頼できない
  動画 (既知3本: c26/c58/c69) では NaN になる。
  **既知の精度限界**: `src/chain_detector.py` の `VideoChainTracker.
  all_clear_pending` が実運用 (認識パイプライン) で全く同じ状態を厳密に
  追跡済みだが、この値は npz に保存されていない (再収集が必要)。本関数は
  npz に既にある score/chain_mechanism からの post-hoc 近似であり、
  chain_mechanism のタグ網羅率が低いデータでは中型連鎖を誤って全消し
  ボーナスと判定する既知の過検出がある (実測: video_c143 で該当行約6.7%、
  詳細コメントは ALL_CLEAR_JUMP_UPPER_BOUND_SCORE 直上)。精度が要件を
  満たすかは要review、将来的な収集強化 (ChainEvent.is_all_clear の
  npz保存) が正確な解。

## 現状カバー範囲 (正直な記録、詳細は調査報告参照)
- npz に入っている原情報: grids / video_id / side / t_sec / game_idx /
  frame_idx / won / score / next1_a,b / dnext_a,b (--with-next 収集時) /
  chain_trigger_sec / chain_mechanism (--enable-chain-tracker 収集時)。
- **本ツールが計算するのは「盤面グリッドのみ」から求まる指標のみ**
  (GRID_ONLY_INDICATORS / GRID_ONLY_HEAVY_INDICATORS)。
- **タスク#8 (2026-08-13) で解消**: ojama_net_balance / ojama_forecast は
  npz からの事後復元を諦め、収集側 (`scripts/collect_boards_lean.py`) が
  `OjamaAccountingTracker` を実際に駆動して真値を記録するようになった
  (npz収集64本目以降)。本ツールはこの真値列を own-perspective のまま
  0-1 正規化して出力する (`OJAMA_TRUTH_COLUMNS` 直上コメント参照)。
  旧npz (真値列なし) は NaN + `ojama_source=1` で区別する。
  **側別サンプリングバイアス (docs/CROSS_CUTTING_AUDIT_2026-08-13.md P4
  決着)**: 各側の記録は「自分の設置直後 (=自分に有利な瞬間)」に偏るため、
  生の own 値をそのまま side 別に使うと構造的バイアスを学習してしまう。
  対処として `ojama_net_balance_synced` (b-2 と同じ merge_asof backward
  パターンで相手の直近値を対応付け、`(own−opp)/2` で平均する再同期版) を
  追加した。`net_2p=−net_1p` の厳密対称性 (会計コアは瞬時対称、P4決着済み)
  より、この平均は理論上「共通時刻での対称バイアス除去済み推定値」になる
  (両側の記録タイミング選好が対称という仮定の下)。**実測での注意**:
  真値npz42本の検証では平均バイアス量が21.7%縮小 (|生|1.998→|synced|1.646)
  する部分的な効果を確認したが、side別の符号一貫性 (2P>1P方向) は
  85.7%→81.0%とわずかな改善に留まり、完全な除去ではない (真の実力差との
  混同・両側の選好非対称性など残差要因は残る、過大な期待をしないこと)。
  さらに C-2 猶予量 `ojama_margin` (吸収余力−実飛来量) も追加した。
- next_pair 依存指標 (near_future_fire_power 等) は本ツールでは未実装
  (next1_a/b はレジストリの外で読み取り可能、拡張ポイントとしてコメントで
  明示するに留める)。
- 1 npz = 1 動画 (video_id 定数、side="1P"/"2P" 混在) を前提に diff 列を
  計算する (2026-08-12 実データで確認済み、data/indicators_v2/
  boards_lean_phase_l_2026-08-11/c143.npz 等)。

## 使い方
    python -m scripts.build_labeled_win_from_npz \\
        --npz-dir data/indicators_v2/boards_lean_phase_l_2026-08-11 \\
        --out data/indicators_v2/study/labeled_win_from_npz_2026-08-12.csv \\
        --profile light

--profile light: sub-ms 指標のみ (高速、反復開発向け)。
--profile full : current_max_chain 等の重い連鎖シミュ系も含む。既定で
    Rust拡張 puyo_core (`src/puyo_core_bridge.py`) に自動載せ替えられる
    (2026-08-13 追加、未ビルド環境では自動的に既存 Python 実装へ
    フォールバックする)。`--no-native` でこの載せ替えを強制的に無効化できる
    (パリティ検証・デバッグ用)。
--exclude-broken (既定ON): BROKEN_VIDEOS (score OCR破綻の4動画) を変換対象
    から除外する。`--no-exclude-broken` で無効化可 (デバッグ用)。
--with-saturation-chain (既定OFF): opt-in指標 saturation_chain (C-3) を
    full profile に含める。既定OFF (下記コスト実測節参照)。

## 2026-08-13 追加分 (docs/INDICATOR_PROPOSAL_ROUND2_2026-08-13.md, user採用済み)
- **A-1**: 旧収集 (collect_indicators_v2.py) から脱落していた11列のうち
  reach_fire_power 系 (next_pair必須、grid-onlyレジストリと非互換のため
  除外) を除く10列を GRID_ONLY_HEAVY_INDICATORS へ再接続 (full profile限定)。
  5列 (immediate_fire_power/chain_efficiency/second_chain_potential/
  ignition_point_count/multi_color_ignition) は native化済み。
  **緊急native化**: min_puyos_to_ignite (実測2000ms/行) も native化し
  43ms/行まで短縮 (下記)。
- **A-2**: BROKEN_VIDEOS (c26/c30/c58/c69、score OCR破綻・won欠損100%実測
  確認済み) を convert_dir の既定挙動として隔離する
  (`--no-exclude-broken` で無効化可)。
- **A-4**: 全消しボーナス予約中フラグ (all_clear_bonus_pending) は npz に
  `all_clear_pending` 真値列 (VideoChainTracker、収集64本目以降) があれば
  それを採用し、無ければ既存の近似ヒューリスティックにフォールバックする。
  `all_clear_source` 列 (0=真値/1=近似) でどちらを使ったか区別できる。
- **C-3**: 実装・テスト済みだった `saturation_chain` (飽和連鎖量、user確定
  定義) を追加したが、**既定 full profile からは除外 (opt-in、下記参照)**。
- **C-4**: 盤面直読みの新指標3種 (`color_diversity_evenness`/
  `buried_hole_count` は light profile、`chain_articulation_point_count`
  は重いため full profile 限定) を追加。

## 【重要】saturation_chain の opt-in化 (2026-08-13、user方針決定)
A-1/C-3 接続直後の素朴な実装では saturation_chain ≈12500ms/行という
桁違いのコストが判明した (システム高負荷下・30サンプル実測)。終端測定
ステップを native化しても改善せず (≈10173ms/行)、真因はビーム構築ステップ
`src/indicators_v2.py::_sat_expand_step` 自体 (simulate不要のはずが
1ステップ≈258ms、最大73ステップで最大17.8秒/行) と判明。この関数の
native化には `native/puyo_core` 本体への新規実装が必要 (本ファイルの
scope外)。**このため既定 full profile からは除外し `--with-saturation-
chain` の opt-in フラグに変更した** (`OPTIONAL_HEAVY_INDICATOR_NAMES`
直上コメント参照)。定義パラメータ (fill_ratio=0.93, beam_width=6) は
user確定 (2026-07-22) のため不変。採否判断用の少数動画サブセット測定を
先行させ、価値が確認されたら native ポートで本採用する方針。
min_puyos_to_ignite (実測2000ms/行) は native化で43ms/行まで短縮できた
ため通常の既定ON経路に残す (`GRID_ONLY_HEAVY_INDICATORS_NATIVE` 参照)。
native化しなかった chain_articulation_point_count (≈59ms/行) は許容範囲内
(native拡張に無い情報を要するため意図的に据え置き、モジュール内コメント参照)。
**simultaneous_pop_richness (≈149ms/行) は同日中に native化した** (タスク#10
「移植1」で puyo_core へ `simulate_chain_with_steps` [ステップごとの同時消し
グループ数/色数/消去個数を露出する新API] が追加されたため。
`_native_simultaneous_pop_richness` 参照、実測コストは下記追記節)。

## simultaneous_pop_richness native対応の実測 (2026-08-13 追記、タスク#10仕上げ)
実測 (300盤面サンプル、loadavg 15〜18の中負荷下): **12.97ms/行 → 0.81ms/行
(16.0倍)**。この列だけで full profile (grid-only heavy指標合計、150盤面
サンプル) の行あたりコストの **61.8%** を占めていたと確認 (native化前
20.43ms/行 → native化後7.80ms/行、2.6倍)。148動画換算 (実測109本の平均
7,141行/本から概算、総行数≈1,057,000行): grid-only heavy指標の計算だけで
**約6.0時間 → 約2.3時間** (savings≈3.7時間)。この数字は指標計算部分のみ
(npz読み込み・diff計算・CSV書き出し等のI/Oは含まない) であり、実際の
フルCSV生成時間はこれより大きい。完全一致は
`tests/test_build_labeled_win_from_npz.py::TestNativeHeavyIndicatorParity::
test_simultaneous_pop_richness_native_matches_python` (60盤面) で確認済み。

## タスク#8 (2026-08-13、docs/CROSS_CUTTING_AUDIT_2026-08-13.md P4 決着の反映)
おじゃま収支の真値列 (ojama_net_balance/ojama_forecast、npz収集64本目以降)
を初めてCSVに接続する。あわせて P4 で確定した「側別サンプリングバイアス」
(自分の設置直後=自分に有利な瞬間に記録が偏る) を打ち消す再同期列
`ojama_net_balance_synced` と、C-2 猶予量 `ojama_margin` (吸収余力−実飛来量)
を新設する。詳細・判断根拠は `OJAMA_TRUTH_COLUMNS` 直上コメント参照。
これら4列 (+ソース種別 `ojama_source`) は grid-only レジストリ (Board→
IndicatorV2Value の関数) の外から来る値のため、b-2 の DIFF_* 5分類には
**あえて含めない** (`all_clear_bonus_pending`/`all_clear_source` が
`TEMPORAL_STATE_COLUMNS` として同5分類の対象外になっている既存の前例と
同じ扱い — 5分類はあくまで grid-only レジストリの列を対象にした完全分割
テスト `tests/test_indicator_pipeline_registry_2026-08-13.py::
TestDiffClassificationCompleteness` の管轄であり、対象外にすることで
「テストが落ちない」を設計上保証する)。

## W12 (2026-08-16、根治P4第一歩) 生値2列の追加
`ojama_forecast` は `iv.ON_FIELD_CAP`(=72) で 0-1 正規化するため、予告個数が
72個を超えると全て同じ値(1.0)に飽和する。実測 (docs/KNOWN_WEAKNESSES.md
W12、2026-08-16 再測定・81動画582,493行) では予告0個=48.6%、72-143個=
44.3%、144-215個=43.4%、216個以上=26.2% (着弾前) と、72個を超えても量が
増えるほど実勝率が悪化し続けるが、現行の正規化ではこれらが全て score=1.0
で区別不能になっている。**旧数値「着弾前48.4%/着弾後18.3%」は再現不能と
判明し使用禁止**(同ファイル W12 冒頭の訂正注記)、上記が有効な再測定値。
この飽和が学習側の見落としの一因と考えられる。生値自体は
`convert_one_npz` 内で既に一時列 (`_ojama_forecast_raw_for_margin`/
`_OJAMA_NET_BALANCE_SYNC_RAW_COL`) として保持・計算に使われた後 pop
されるだけで、npz 側には元々ロスなく存在している(「データが無い」のでは
なく「配線されていない」だけ)。そこで診断・将来の特徴量設計検討用に、
0-1 正規化前の生の個数をそのまま **別列** `ojama_forecast_uncapped`/
`ojama_net_balance_uncapped` として追加する (`_attach_ojama_truth_own_
columns` 内で一時列と同時に書き込む、pop 対象にはしない)。

**列名について (`_raw` を避けた理由)**: 素直には `*_raw` と呼びたいところ
だが、a-1 決定記録 (docs/INDICATOR_REORG_PROPOSAL_2026-08-12.md 「a-1.
中身が完全に同じ重複」) で「加工前/加工後の2本立て8組 (*_raw 列、= 既存の
正規化列と完全重複・情報量ゼロ)」を削除・CSV非出力にすることが確定し、
回帰防止テスト `tests/test_build_labeled_win_from_npz.py::
test_convert_one_npz_never_emits_raw_columns` /
`test_csv_output_has_no_raw_columns` が **任意の `*_raw` 列を全面禁止**
するガードとして存在する。本タスクの2列は「正規化列とは違う情報を持つ
(72個超えの領域は正規化列からは復元不可能)」ため a-1 の「完全重複」には
該当しないが、テストは名前ベースの一律禁止のため `_raw` 接尾辞を使うと
無関係な既存の禁止事項に抵触して見える (かつ意味的にも a-1 の「完全重複
raw」と混同されうる)。よって意味が近い別名 `_uncapped` (0-1正規化の上限
飽和が掛かっていない、の意) を採用し、a-1 のテスト・決定記録は無改修で
維持する。

- **0-1正規化の対象外**: CLAUDE.md 「指標は0-1正規化必須」は学習に直接
  投入する指標列の規約であり、この2列は「診断・設計検討用の生の材料」
  という別カテゴリのため対象外とする (学習に入れる最終的な形=対数圧縮や
  上限拡張等は P3 で設計する、ここでは正しく取り出せることのみを保証)。
- **欠損の明示**: 真値の無い旧npzでは他の真値系列と全く同じ経路 (`_ojama_
  truth_raw_arrays` の nan_fill) から来るため、自動的に NaN になる (0埋めで
  誤魔化さない)。
- **対称化パイプライン非該当**: 本ツールに列を一括反転する処理は存在せず
  (b-2 の DIFF_* 5分類も個別列の明示リスト方式)、視覚オーバーレイ側の
  mirror/symmetrize (`scripts/visualize_advantage_overlay.py`) とは完全に
  別パイプラインのため、この2列がそちらの影響を受けることもない。
  この2列は own-perspective の絶対量 (個数、side非依存) であり、
  OJAMA_TRUTH_COLUMNS の他4列と同じく diff化・carry化もしない (概念的に
  「相手を含めた収支そのもの」であり相手との差を取る意味が無い点は上記と
  同じ)。

## W12 (2026-08-16、アーキ設計確定分) 追加3列
上記の uncapped 生値2列に続けて、実際に飽和の影響を緩和する3列を追加する
(実装は `_attach_ojama_forecast_log_columns`、詳細な数式・根拠は関数
docstring 参照):

| 列名 | 定義式 | 正規化 |
|---|---|---|
| `ojama_forecast_log` | `log1p(forecast_uncapped) / log1p(PENDING_ABS_CAP)` | 式自体が0-1有界 |
| `ojama_forecast_progress_interaction` | `ojama_forecast_log × match_progress` | 0-1×0-1で自動有界 |
| `color_forecast_ratio_own` | `color_raw / (color_raw + forecast_raw + COLOR_OJAMA_RATIO_EPS)` | 比率で自動0-1 |

`PENDING_ABS_CAP` (=216=ON_FIELD_CAP*3) は `src/ojama_accounting.py` の
`OjamaAccountingTracker` が forecast_incoming に実際に掛けている物理上限
そのもの (新規定数を作らず import して使う)。`COLOR_OJAMA_RATIO_EPS` も
既存の `color_ojama_ratio_own` と同じ定数を再利用する。match_progress は
中間値でありCSV列としては出力しない (新規の merge_asof を増やさず既存
`diff_board_puyo_total` から代数的に逆算するため、`_attach_opponent_diff_
columns` の後に呼ぶ必要がある、詳細は関数docstring)。3列とも own専用の
絶対量/比率 (side非依存) であり、`ojama_forecast_log`/`ojama_forecast_
progress_interaction` は `OJAMA_TRUTH_COLUMNS`、`color_forecast_ratio_own`
は `PAIR_INTERACTION_COLUMNS` (既存 `color_ojama_ratio_own` と対になる
色ぷよ×予告おじゃま版のため) にそれぞれ末尾追加し、b-2 の DIFF_* 5分類・
視覚オーバーレイ側の mirror/symmetrize いずれの一括変換も通さない。

採用しなかった設計 (アーキ確定): 容量との交互作用は非単調 (空き36-53が
54-71より高い逆転) で交絡濃厚のため保留 (既存 `ojama_margin` で代替)、
猶予時間はn不足で保留 (63本の再収集後に再検討)。位相は列を分けず
`ojama_forecast_progress_interaction` の乗算1列に圧縮する (列数節約)。

## saturation_chain_upper (2026-08-13、user簡略化決定)
`saturation_chain` (C-3) はコスト実測 (1行8〜18秒) が桁違いのため opt-in化
されたが、その後 user が「上部限定軽量版」への簡略化を決定した:
「疎らな盤面の飽和はおじゃまぷよ数 (既存 `board_ojama_count`) が現象を
代理するため計測不要、盤面が既に高く積まれている局面 (充填率
`iv.SATURATION_UPPER_MIN_FILL`=0.90 以上) だけ既存の積み上げ探索を実行する」
という設計 (`iv.saturation_chain_upper` 参照)。閾値未満は NaN (0 ではない、
「未計測」と区別) を返すだけの軽量判定のため **saturation_chain のような
opt-in化は不要 = full profile の既定に含める** (`OPTIONAL_HEAVY_
INDICATOR_NAMES` に追加していない)。閾値の導出根拠 (コスト実測→予算→
バッファのチェーン)、および「ビーム構築ステップ自体の native化は実データで
bit-identical要件を満たせず断念した」という負の実験結果は
`src/indicators_v2.py::SATURATION_UPPER_MIN_FILL` 直上コメントに集約する
(このファイル側では終端測定のみの native化 `_native_saturation_chain_upper`
のみ実装、`_native_saturation_chain` と同型)。
実測コスト見込み (63動画実データ→148動画換算、対象=fill>=0.90の行):
出現率0.671%・行数≈7,139行・追加時間≈0.57時間 (保守的な258ms/ステップ
[過去実測・システム高負荷下]前提、軽負荷実測では3-5ms/ステップで
さらに短い)。
"""
from __future__ import annotations

import argparse
import csv
import functools
import math
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.board import BOARD_COLS, COLOR_EMPTY, COLOR_UNKNOWN, Board  # noqa: E402
import src.indicators_v2 as iv  # noqa: E402
from src.ojama_accounting import PENDING_ABS_CAP  # noqa: E402
from src.production_config import GHOST_CHAIN_RULE_ENABLED  # noqa: E402
from src.puyo_core_bridge import NATIVE_AVAILABLE as _PUYO_CORE_AVAILABLE  # noqa: E402
from src.puyo_core_bridge import (  # noqa: E402
    chain_metrics_after_drops as _native_chain_metrics_after_drops,
    simulate_after_drops as _native_simulate_after_drops,
    simulate_chain as _native_simulate_chain,
    simulate_chain_with_steps as _native_simulate_chain_with_steps,
)
from src.scoring import ALL_CLEAR_BONUS  # noqa: E402

# ============================
# 指標レジストリ (薄い委譲構造の核心)
# ============================
# 値は IndicatorV2Value を返す Board -> value の関数のみ (score 列を自動で
# 書く。*_raw は a-1 決定 (2026-08-12) により出力しない)。新指標を増やす/
# 減らす場合はこの2つの dict を編集するだけで良く、変換ループ本体
# (_compute_row) は触らない。

# light: 実測 <0.15ms/行 (ベンチ scripts/_verify... 2026-08-12、n=500)。
# center_bulge (合成版) は b-1 決定により center_bulge_color/_ojama の
# 2列に分解済み (indicators_v2.py 側の center_bulge() 関数自体は
# backwards compat のため残っている)。
GRID_ONLY_INDICATORS: dict[str, Callable[[Board], "iv.IndicatorV2Value"]] = {
    "board_color_puyo_total": iv.board_color_puyo_total,
    "board_puyo_total": iv.board_puyo_total,
    "max_column_height": iv.max_column_height,
    "column_bumpiness": iv.column_bumpiness,
    "death_margin": iv.death_margin,
    "death_margin_neighbor": iv.death_margin_neighbor,
    "center_bulge_color": iv.center_bulge_color,
    "center_bulge_ojama": iv.center_bulge_ojama,
    "board_ojama_count": iv.board_ojama_count,
    # C-4 (2026-08-13 ラウンド2提案書 user採用済み): 盤面直読みの安価な新指標。
    # simulate 不要 (盤面走査のみ) のため light profile に収容。
    "color_diversity_evenness": iv.color_diversity_evenness,
    "buried_hole_count": iv.buried_hole_count,
}
# full: 連鎖シミュレーションを要する重い指標 (実測 1〜19ms/行)。
# --profile full 指定時のみ計算する。
# saturated_chain_count は a-1 決定 (2026-08-12) により削除済み
# (current_max_chain と19万場面で完全一致、src/production_config.py の
# ATTRIBUTION_EXCLUDED_INDICATORS にも同根拠で既に記録あり)。
#
# A-1 (2026-08-13 ラウンド2提案書、横展開監査 docs/CROSS_CUTTING_AUDIT_
# 2026-08-13.md P2): 旧 collect_indicators_v2.py には配線されていたが本
# ツールへの載せ替え時に脱落していた11列のうち、reach_fire_power 系 (3列)
# を除く10列を再接続する (reach_fire_power は next_pair/dnext_pair 必須の
# ため grid-only レジストリの型 `Callable[[Board], IndicatorV2Value]` と
# 非互換 — 除外理由は production_config.KNOWN_PIPELINE_GAPS に別エージェント
# により記録済み、本ファイルでの対応不要)。
# C-3 (同提案書、実装・テスト済みで検証中断していた saturation_chain の
# 正式接続) / C-4 の重い版 (chain_articulation_point_count) も同時に追加。
#
# **saturation_chain は opt-in 扱い (2026-08-13 コスト実測後、user方針決定)**:
# ここには「既知の heavy 指標カタログ」として残す (DIFF分類・native parity
# テストの対象台帳として)。実際の登録レジストリ (`_resolve_indicator_
# registry`) では既定で除外し、`--with-saturation-chain` 明示時のみ含める
# (下記 OPTIONAL_HEAVY_INDICATOR_NAMES 参照、理由は同定数直上コメント)。
GRID_ONLY_HEAVY_INDICATORS: dict[str, Callable[[Board], "iv.IndicatorV2Value"]] = {
    "current_max_chain": iv.current_max_chain,
    "dig_resistance": iv.dig_resistance,
    "ukeyasusa": iv.ukeyasusa,
    "sub_chain_count": iv.sub_chain_count,
    # --- A-1 再接続 (10列、next非依存のみ) ---
    "immediate_fire_power": iv.immediate_fire_power,
    "chain_efficiency": iv.chain_efficiency,
    "min_puyos_to_ignite": iv.min_puyos_to_ignite,
    "second_chain_potential": iv.second_chain_potential,
    "main_linked_pair_count": iv.main_linked_pair_count,
    "isolated_pair_count": iv.isolated_pair_count,
    "main_linked_ratio": iv.main_linked_ratio,
    "ignition_point_count": iv.ignition_point_count,
    "multi_color_ignition": iv.multi_color_ignition,
    "simultaneous_pop_richness": iv.simultaneous_pop_richness,
    # --- C-3 (saturation_chain 正式接続、既定OFFのopt-in。下記参照) ---
    "saturation_chain": iv.saturation_chain,
    # --- C-4 (重い版、find_groups だけでなく追加simulateを要するため) ---
    "chain_articulation_point_count": iv.chain_articulation_point_count,
    # --- 上部限定軽量版 (saturation_chain_upper, 2026-08-13 user簡略化決定) ---
    # 疎らな盤面 (fill<SATURATION_UPPER_MIN_FILL) は NaN を返すだけの軽量
    # ゲート判定のみ (count_puyos() 1回) のため、saturation_chain のような
    # opt-in化は不要 = full profile の既定に含める (OPTIONAL_HEAVY_
    # INDICATOR_NAMES に加えない)。閾値・native化の可否はコスト実測込みで
    # `iv.SATURATION_UPPER_MIN_FILL` 直上コメント参照。
    "saturation_chain_upper": iv.saturation_chain_upper,
}

# ============================
# saturation_chain の opt-in化 (2026-08-13 コスト実測後、user方針決定)
# ============================
# 実測 (2026-08-13): 1行8〜18秒 (システム高負荷下、ビーム構築ステップ
# `iv._sat_expand_step` が支配的コスト。native化した終端測定はほぼ無関係
# だったと判明、詳細はモジュール docstring「コスト実測」節)。148本フルは
# 非現実的なため既定 full profile から外す。採否判断用の少数動画サブセット
# 測定を先行させ、価値が確認されたら native/puyo_core へのビーム構築ポート
# (本ファイルの scope 外、別タスク) で本採用する方針。
# 定義パラメータ (fill_ratio=0.93, beam_width=6) は user確定 (2026-07-22)
# のため本対応では不変 (indicators_v2.py::saturation_chain 自体も無変更)。
OPTIONAL_HEAVY_INDICATOR_NAMES: frozenset[str] = frozenset({"saturation_chain"})

VALID_PROFILES: tuple[str, ...] = ("light", "full")

# ============================
# A-2: 壊れ動画の隔離 (2026-08-13 ラウンド2提案書 A-2)
# ============================
# score OCR完全破綻により won ラベル・全消し検出等が実質不能と判明した動画。
# npz ファイル名の stem (例 "c26.npz"→"c26") と照合する (npz内部の video_id
# 列は "video_c26" 形式のため、stem 側で比較する必要がある)。
#
# 実測根拠 (2026-08-13、data/indicators_v2/boards_lean_phase_l_2026-08-07/
# 各npz、convert_dir の主経路と同じ won/score 列で確認):
#   c26: n=8034,  won欠損率100%, score欠損率100% (既知、project_video_
#        difficulty_3broken_2026-07-29)
#   c30: n=9578,  won欠損率100%, score欠損率100% (新規発見、A-2)
#   c58: n=11059, won欠損率100%, score欠損率100% (既知)
#   c69: n=11467, won欠損率100%, score欠損率100% (既知)
# 参考 (健全動画の score欠損率、同ディレクトリ): c1=1.5%, c10=0.4%,
# c100=1.1%, c103=0.2% (0〜数%が通常域、100%は完全破綻)。
BROKEN_VIDEOS: tuple[str, ...] = ("c26", "c30", "c58", "c69")

# ============================
# Rust ネイティブ拡張 (puyo_core) 載せ替え (2026-08-13 追加)
# ============================
# GRID_ONLY_HEAVY_INDICATORS (上記4列) は ChainSimulator.simulate の繰り返し
# 呼び出しが支配的コスト (current_max_chain: 30回、dig_resistance: 4回、
# ukeyasusa: dig_resistance丸ごと再利用、sub_chain_count: 最大61回、実測
# 1〜19ms/行)。scripts/mc_counter_estimator.py と同じ「呼び出し側に native
# 分岐を足す」パターンで src/puyo_core_bridge.py 経由の Rust 実装に載せ替える
# (indicators_v2.py 自体は無変更、完全一致は
# tests/test_build_labeled_win_from_npz.py::TestNativeHeavyIndicatorParity
# で担保)。幽霊連鎖ルール (`_SHARED_SIMULATOR` の設定) と揃えるため、native
# 呼び出しは必ず exclude_hidden_row_from_pop=GHOST_CHAIN_RULE_ENABLED を
# 明示する。dig_resistance のおじゃま落下 (`ChainSimulator.drop_ojama`) は
# 乱数 (端数列のランダム選択) を含むため native化せず既存 Python 実装のまま
# 保持する (載せ替え対象は連鎖シミュレーション部分のみ、既存の非決定性は
# この載せ替えでは変えない・直さない)。
#
# **既知の native 制約 (2026-08-13 発見、`_board_is_gravity_consistent`
# 参照)**: puyo_core の重力実装は入力盤面が既に gravity一貫であることを
# 前提にした最適化であり、認識由来の浮きぷよ (既存の重力違反、既知欠陥) を
# 含む盤面では連鎖数がズレる場合がある (実測: 実盤面1,097件中1件で
# current_max_chain の chain_count が 2→1 に食い違った)。全 native
# 呼び出しの直前で `_board_is_gravity_consistent(board)` を確認し、違反時は
# 既存 Python 実装へフォールバックすることで「完全一致」を保証する
# (native の恒久修正は別課題、native/puyo_core 自体は本タスクで変更しない)。

# takapt定石 (列×色30通り) の (col, color) 一覧。`iv._takapt_best_drop` と
# 同一探索順 (col昇順→色昇順、scripts/mc_counter_estimator.py の
# `_DROP_CANDIDATES_30` と同一定義を独立複製 — 両ファイルとも編集しない
# 制約のため)。
_NATIVE_DROP_CANDIDATES_30: tuple[tuple[int, int], ...] = tuple(
    (col, color) for col in range(BOARD_COLS) for color in iv.IGNITION_TRIAL_COLORS
)


def _board_is_gravity_consistent(board: Board) -> bool:
    """各列に「浮きぷよ」由来のギャップが無いか判定する (native 載せ替えの安全弁)。

    **2026-08-13 実データ調査で発見した既知の native 制約**: puyo_core
    (Rust) の重力実装は、消去後の各列シフト量を「その列で消えたセル数」
    として計算するビット圧縮の定数時間実装であり、入力盤面が既に
    gravity一貫 (各列で占有セルの下に空きが無い) であることを前提に
    最適化されている。既に重力違反 (認識由来の浮きぷよ、
    `project_gravity_violation_regen_lead_2026-07-30` 系の既知欠陥) を含む
    盤面を渡すと、2手目以降の連鎖判定が食い違う場合がある (実測:
    1,097件の実盤面サンプル中1件で current_max_chain の chain_count が
    2→1 に食い違うことを確認済み。手動トレースで Python
    ChainSimulator の chain_count=2 が正しいゲーム挙動と確認済み)。
    この関数で事前検知し、違反時は呼び出し側が既存 Python 実装
    (`indicators_v2.py`、常に正しい) へフォールバックする
    (「完全一致」要件を守るための安全弁、native の恒久修正は別課題)。
    UNKNOWN セルは占有扱いしない (`height_of`/Rust `occ` と同じ意味論)。

    相互参照: `scripts/mc_counter_estimator.py::_board_is_gravity_consistent`
    に同一の意味論・実装を複製している (2026-08-13 追加、循環依存回避の
    ためscripts間import はせず複製、ロジックを変える場合は両ファイル修正)。
    """
    grid = board._grid
    unoccupied = (grid == COLOR_EMPTY) | (grid == COLOR_UNKNOWN)
    for col in range(BOARD_COLS):
        occupied_rows = np.where(~unoccupied[:, col])[0]
        if len(occupied_rows) == 0:
            continue
        top_row = int(occupied_rows[0])
        if np.any(unoccupied[top_row:, col]):
            return False
    return True


def _native_takapt_best_drop(
    board: Board,
) -> "tuple[int, Board | None, object | None]":
    """takapt定石探索の native 版 (`iv._takapt_best_drop` と同一意味論)。

    `sub_chain_count` が「本線発火後の final_board」を必要とするため、
    盤面付きバッチAPI `simulate_after_drops` で候補ごとの ChainSimResult も
    保持して返す (`iv._takapt_best_drop` 内で最良候補の simulate 結果を
    使い捨てていた分の再計算を省く最適化)。

    Returns:
        (最大連鎖数, 1個追加後の盤面 [連鎖解決前] または None, その
        ChainSimResult または None)。`>` による厳密な大小比較のため同値は
        先に見つかった (col昇順→色昇順) 候補を保持する
        (`iv._takapt_best_drop` の tie-break と同一)。
    """
    best_chain = 0
    best_board: "Board | None" = None
    best_result = None
    for r in _native_simulate_after_drops(
        board, _NATIVE_DROP_CANDIDATES_30,
        exclude_hidden_row_from_pop=GHOST_CHAIN_RULE_ENABLED,
    ):
        if r is None:
            continue
        if r.chain_result.chain_count > best_chain:
            best_chain = r.chain_result.chain_count
            best_board = r.dropped_board
            best_result = r.chain_result
    return best_chain, best_board, best_result


def _native_current_max_chain(
    board: Board, use_native: bool = True,
) -> "iv.IndicatorV2Value":
    """既存 III-1 `current_max_chain` の native 分岐版。

    use_native=False または拡張未導入時は既存 `iv.current_max_chain` に
    そのまま委譲する (完全一致、indicators_v2.py 自体は無変更)。
    """
    if not (use_native and _PUYO_CORE_AVAILABLE and _board_is_gravity_consistent(board)):
        return iv.current_max_chain(board)
    best_chain = 0
    for r in _native_chain_metrics_after_drops(
        board, _NATIVE_DROP_CANDIDATES_30,
        exclude_hidden_row_from_pop=GHOST_CHAIN_RULE_ENABLED,
    ):
        if r is None:
            continue
        chain_count, _exact_score = r
        if chain_count > best_chain:
            best_chain = chain_count
    raw = float(best_chain)
    return iv.IndicatorV2Value(score=iv._clamp01(raw / iv.NORM_MAX_CHAIN), raw=raw)


def _native_dig_resistance_one(
    board: Board, base_chain: int, n_ojama: int,
) -> float:
    """dig_resistance 1点分の native 版 (`iv._dig_resistance_one` と同一計算)。

    おじゃま落下 (`drop_ojama`) は乱数を含むため既存 Python 実装
    (`iv._SHARED_SIMULATOR`、上部コメント参照) をそのまま再利用し、連鎖
    シミュレーションのみ native に置き換える。
    """
    try:
        ojama_board = iv._SHARED_SIMULATOR.drop_ojama(board, n_ojama)
    except Exception:
        return 0.0
    if ojama_board.is_dead():
        return 0.0
    try:
        post_chain = _native_simulate_chain(
            ojama_board, exclude_hidden_row_from_pop=GHOST_CHAIN_RULE_ENABLED,
        ).chain_count
    except Exception:
        return 0.0
    survival = min(1.0, post_chain / float(base_chain))
    dig = 1.0 if post_chain >= iv.OJAMA_DEFENSE_DIG_MIN_CHAIN else 0.0
    return (
        iv.OJAMA_DEFENSE_SURVIVAL_WEIGHT * survival
        + iv.OJAMA_DEFENSE_DIG_WEIGHT * dig
    )


def _native_dig_resistance(
    board: Board, use_native: bool = True,
) -> "iv.IndicatorV2Value":
    """既存 VI-1 `dig_resistance` の native 分岐版 (`_native_dig_resistance_one`
    に3点分を委譲)。use_native=False または拡張未導入時は既存
    `iv.dig_resistance` にそのまま委譲する (完全一致)。
    """
    if not (use_native and _PUYO_CORE_AVAILABLE and _board_is_gravity_consistent(board)):
        return iv.dig_resistance(board)
    if board.is_dead():
        return iv.IndicatorV2Value(score=0.0, raw=0.0)
    base_chain = max(
        1, _native_simulate_chain(
            board, exclude_hidden_row_from_pop=GHOST_CHAIN_RULE_ENABLED,
        ).chain_count,
    )
    scores = [
        _native_dig_resistance_one(board, base_chain, n)
        for n in iv.OJAMA_DEFENSE_TEST_COUNTS
    ]
    avg = sum(scores) / len(scores) if scores else 0.0
    return iv.IndicatorV2Value(score=iv._clamp01(avg), raw=float(avg))


def _native_ukeyasusa(
    board: Board, use_native: bool = True,
) -> "iv.IndicatorV2Value":
    """既存 X-1 `ukeyasusa` の native 分岐版 (dig_resistance 部分のみ
    native化、absorption_capacity/death_margin は sim 不要のため既存関数を
    直接使う)。use_native=False または拡張未導入時は既存 `iv.ukeyasusa` に
    そのまま委譲する。
    """
    if not (use_native and _PUYO_CORE_AVAILABLE and _board_is_gravity_consistent(board)):
        return iv.ukeyasusa(board)
    s_abs = iv.absorption_capacity(board).score
    s_dig = _native_dig_resistance(board, use_native=True).score
    s_death = iv.death_margin(board).score
    score = (
        iv.UKEYASUSA_W_ABSORPTION * s_abs
        + iv.UKEYASUSA_W_DIG * s_dig
        + iv.UKEYASUSA_W_DEATH * s_death
    )
    raw_abs = iv.absorption_capacity(board).raw
    return iv.IndicatorV2Value(score=iv._clamp01(score), raw=raw_abs)


def _native_sub_chain_count(
    board: Board, use_native: bool = True,
) -> "iv.IndicatorV2Value":
    """既存 XII-4 `sub_chain_count` の native 分岐版。

    1手目探索の最良候補が保持する ChainSimResult (`_native_takapt_best_drop`)
    をそのまま「本線発火結果」として再利用する (`iv.sub_chain_count` の
    `sim.simulate(best_board)` 再計算と等価。ChainSimulator.simulate は
    決定的関数のため同一盤面には常に同一結果、再計算を省く最適化)。
    2手目探索は盤面を返さない軽量バッチAPIで行う。
    """
    if not (use_native and _PUYO_CORE_AVAILABLE and _board_is_gravity_consistent(board)):
        return iv.sub_chain_count(board)
    best_chain, best_board, best_result = _native_takapt_best_drop(board)
    if best_board is None or best_chain == 0 or best_result is None:
        return iv.IndicatorV2Value(score=0.0, raw=0.0)
    if best_result.chain_count == 0:
        return iv.IndicatorV2Value(score=0.0, raw=0.0)
    post_board = best_result.final_board
    sub_best_chain = 0
    for r in _native_chain_metrics_after_drops(
        post_board, _NATIVE_DROP_CANDIDATES_30,
        exclude_hidden_row_from_pop=GHOST_CHAIN_RULE_ENABLED,
    ):
        if r is None:
            continue
        chain_count, _exact_score = r
        if chain_count > sub_best_chain:
            sub_best_chain = chain_count
    raw = float(sub_best_chain)
    return iv.IndicatorV2Value(score=iv._clamp01(raw / iv.NORM_SUB_CHAIN), raw=raw)


def _native_simultaneous_pop_richness(
    board: Board, use_native: bool = True,
) -> "iv.IndicatorV2Value":
    """既存 XII-5 `simultaneous_pop_richness` の native 分岐版
    (2026-08-13、タスク#10「移植1」で puyo_core に追加された
    `simulate_chain_with_steps` [ステップごとの同時消しグループ数/色数/
    消去個数を露出する新API] を使って初めて native化可能になった。
    それまでは「native ChainSimResult にステップ内 erased_groups数の情報が
    無い」ため意図的に Python 実装のまま据え置いていた [直下「A-1再接続分」
    コメント参照、経緯を記録として残す]。

    `iv.simultaneous_pop_richness` と同一の探索 (takapt定石30通り→
    最良候補を simulate→ステップごとの同時消しグループ数の平均) だが、
    最良候補の再simulateを `simulate_chain_with_steps` (ステップ情報付き)
    1回で済ませる (`_native_takapt_best_drop` の `best_result` は
    ステップ情報を持たない `ChainSimResult` のため、そのままでは
    `erased_groups` 相当の値を取り出せない — これが native化できなかった
    理由そのものであり、`simulate_chain_with_steps` の追加で解消した)。

    use_native=False または拡張未導入時は既存 `iv.simultaneous_pop_richness`
    にそのまま委譲する (完全一致、indicators_v2.py 自体は無変更)。
    """
    if not (use_native and _PUYO_CORE_AVAILABLE and _board_is_gravity_consistent(board)):
        return iv.simultaneous_pop_richness(board)
    best_chain, best_board, _best_result = _native_takapt_best_drop(board)
    if best_board is None or best_chain == 0:
        return iv.IndicatorV2Value(score=0.0, raw=0.0)
    _result, steps = _native_simulate_chain_with_steps(
        best_board, exclude_hidden_row_from_pop=GHOST_CHAIN_RULE_ENABLED,
    )
    if not steps:
        return iv.IndicatorV2Value(score=0.0, raw=0.0)
    avg_groups = sum(s.num_groups for s in steps) / len(steps)
    return iv.IndicatorV2Value(
        score=iv._clamp01(avg_groups / iv.NORM_SIMULTANEOUS_POP), raw=avg_groups,
    )


# ============================
# A-1 再接続分の native 版 (2026-08-13 追加)
# ============================
# 5列 (immediate_fire_power/chain_efficiency/second_chain_potential/
# ignition_point_count/multi_color_ignition) は既存4列と同じ「takapt 30通り
# 探索 or 1回 simulate」パターンのため native化する。
# main_linked_pair_count/isolated_pair_count/main_linked_ratio は
# find_groups のみで simulate 不要 (実測 <1ms/行) のため native化不要、
# Python 実装のままとする (GRID_ONLY_HEAVY_INDICATORS_NATIVE に載せない =
# 常に iv.xxx が呼ばれる、_resolve_indicator_registry のフォールバック参照)。
# simultaneous_pop_richness は2026-08-13当初「native ChainSimResult に無い
# 情報 (ステップ内erased_groups数) を要する」として Python 実装のまま
# 据え置いていたが、同日中にタスク#10「移植1」で puyo_core へ
# `simulate_chain_with_steps` が追加されたため `_native_simultaneous_pop_
# richness` で native化済み (上記関数、GRID_ONLY_HEAVY_INDICATORS_NATIVE
# 登録済み)。
# chain_articulation_point_count は同種の制約が未解消のため Python 実装の
# まま (`_erase_groups` 等の内部詳細が puyo_core 未露出、実測59ms/行で
# 許容範囲)。
# min_puyos_to_ignite/saturation_chain は当初「無理はしない」対象だったが、
# 実測でそれぞれ2000ms/行・12500ms/行という桁違いのコストが判明したため
# (他列の数十〜数百倍、148本再学習が非現実的な時間になる) 例外的に
# native化する (下記「緊急native化」セクション参照)。


def _native_fire_ojama_from_chain_result(result: "object") -> int:
    """native ChainSimResult の exact_score からお邪魔換算する。

    `iv._board_fire_ojama` の native 版 (elapsed_sec=0.0 固定、grid-only
    レジストリは時間情報を持たないため既存 Python 経路の既定値と揃える)。
    """
    ojama = iv.score_to_ojama(
        score=result.exact_score, prev_leftover=0, elapsed_sec=0.0,
        rate_base=iv.OJAMA_RATE_STANDARD,
    )
    return int(ojama.ojama_count)


def _native_immediate_fire_power(
    board: Board, use_native: bool = True,
) -> "iv.IndicatorV2Value":
    """既存 III-2 `immediate_fire_power` の native 分岐版。

    `_native_takapt_best_drop` の best_result (exact_score 保持済み) を
    再利用し `iv._board_fire_ojama` の再simulateを省く。
    """
    if not (use_native and _PUYO_CORE_AVAILABLE and _board_is_gravity_consistent(board)):
        return iv.immediate_fire_power(board)
    _best_chain, best_board, best_result = _native_takapt_best_drop(board)
    if best_board is None or best_result is None:
        return iv.IndicatorV2Value(score=0.0, raw=0.0)
    ojama = _native_fire_ojama_from_chain_result(best_result)
    return iv.IndicatorV2Value(
        score=iv._clamp01(float(ojama) / iv.ON_FIELD_CAP), raw=float(ojama),
    )


def _native_chain_efficiency(
    board: Board, use_native: bool = True,
) -> "iv.IndicatorV2Value":
    """既存 III-4 `chain_efficiency` の native 分岐版 (best_result 再利用)。"""
    if not (use_native and _PUYO_CORE_AVAILABLE and _board_is_gravity_consistent(board)):
        return iv.chain_efficiency(board)
    _best_chain, best_board, best_result = _native_takapt_best_drop(board)
    if best_board is None or best_result is None:
        ojama = 0
    else:
        ojama = _native_fire_ojama_from_chain_result(best_result)
    color_count = iv._count_color_puyos(board)
    raw = 0.0 if color_count <= 0 else float(ojama) / float(color_count)
    return iv.IndicatorV2Value(score=iv._clamp01(raw / iv.CHAIN_EFF_MAX), raw=raw)


def _native_second_chain_potential(
    board: Board, use_native: bool = True,
) -> "iv.IndicatorV2Value":
    """既存 III-7 `second_chain_potential` の native 分岐版。

    ChainResult.participating_cells は native ChainSimResult.total_erased
    のエイリアス (`src/chain.py::ChainResult.participating_cells` docstring
    「= total_erased、indicators.py 用エイリアス」参照) のためそのまま使える。
    """
    if not (use_native and _PUYO_CORE_AVAILABLE and _board_is_gravity_consistent(board)):
        return iv.second_chain_potential(board)
    color_count = iv._count_color_puyos(board)
    participating = _native_simulate_chain(
        board, exclude_hidden_row_from_pop=GHOST_CHAIN_RULE_ENABLED,
    ).total_erased
    non_participating = max(0, color_count - participating)
    raw = float(non_participating)
    score = 0.0 if color_count <= 0 else float(non_participating) / float(color_count)
    return iv.IndicatorV2Value(score=iv._clamp01(score), raw=raw)


def _native_ignition_scan(board: Board) -> "list[tuple[int, int]]":
    """takapt 30通り native バッチ scan で発火可能な (col, color) 一覧を返す。

    `ignition_point_count`/`multi_color_ignition` 共通の探索部分
    (`iv._takapt_full_scan` の native 版、追加 simゼロで探索を共有)。
    呼び出し前提: `_board_is_gravity_consistent(board)` 確認済み。
    """
    raw = _native_chain_metrics_after_drops(
        board, _NATIVE_DROP_CANDIDATES_30,
        exclude_hidden_row_from_pop=GHOST_CHAIN_RULE_ENABLED,
    )
    hits: "list[tuple[int, int]]" = []
    for (col, color), r in zip(_NATIVE_DROP_CANDIDATES_30, raw):
        if r is None:
            continue
        chain_count, _exact_score = r
        if chain_count > 0:
            hits.append((col, color))
    return hits


def _native_ignition_point_count(
    board: Board, use_native: bool = True,
) -> "iv.IndicatorV2Value":
    """既存 XII-2 `ignition_point_count` の native 分岐版。"""
    if not (use_native and _PUYO_CORE_AVAILABLE and _board_is_gravity_consistent(board)):
        return iv.ignition_point_count(board)
    hits = _native_ignition_scan(board)
    raw = float(len(hits))
    return iv.IndicatorV2Value(
        score=iv._clamp01(raw / iv.NORM_IGNITION_POINT_COUNT), raw=raw,
    )


def _native_multi_color_ignition(
    board: Board, use_native: bool = True,
) -> "iv.IndicatorV2Value":
    """既存 XII-3 `multi_color_ignition` の native 分岐版 (探索を scan と共有)。"""
    if not (use_native and _PUYO_CORE_AVAILABLE and _board_is_gravity_consistent(board)):
        return iv.multi_color_ignition(board)
    hits = _native_ignition_scan(board)
    colors_hit = {color for _col, color in hits}
    raw = float(len(colors_hit))
    return iv.IndicatorV2Value(
        score=iv._clamp01(raw / iv.NORM_MULTI_COLOR_IGNITION), raw=raw,
    )


# ============================
# 緊急native化 (2026-08-13 コスト実測で発覚した2件): min_puyos_to_ignite /
# saturation_chain
# ============================
# A-1/C-3 接続直後の実測 (30サンプル、システム高負荷下) で
# min_puyos_to_ignite ≈2000ms/行、saturation_chain ≈12500ms/行 という
# 桁違いのコストが判明 (他の重い列は数十ms/行、native化済み4列は数ms/行)。
# 「無理はしない」方針の対象外とし (148本の全処理が現実的な時間で終わらず
# A-1/C-3 の価値が実質使えなくなるため)、コストの支配項のみ native化する。


def _native_search_min_ignite(board: Board, base_chain: int) -> int:
    """`iv._search_min_ignite` の native 分岐版 (N=1: 30通り, N=2: 900通り)。

    N=1 は `chain_metrics_after_drops` の1回のバッチ呼び出しで置換 (`iv.
    _search_min_ignite` の30回個別simulateがコスト支配項)。N=2 は「1手目を
    生の(連鎖未解決)盤面として保持し2手目をバッチ探索」を1手目30通り分
    繰り返す (`iv._search_min_ignite` の900回個別simulateに対応、native
    ラウンドトリップは最大31回に圧縮)。IGNITION_TRIAL_LIMIT>=2 前提の
    呼び出し元 (`_native_min_puyos_to_ignite`) からのみ呼ばれる。
    """
    raw1 = _native_chain_metrics_after_drops(
        board, _NATIVE_DROP_CANDIDATES_30,
        exclude_hidden_row_from_pop=GHOST_CHAIN_RULE_ENABLED,
    )
    for r in raw1:
        if r is not None and r[0] > base_chain:
            return 1
    first_drops = _native_simulate_after_drops(
        board, _NATIVE_DROP_CANDIDATES_30,
        exclude_hidden_row_from_pop=GHOST_CHAIN_RULE_ENABLED,
    )
    for r1 in first_drops:
        if r1 is None or r1.dropped_board.is_dead():
            continue
        raw2 = _native_chain_metrics_after_drops(
            r1.dropped_board, _NATIVE_DROP_CANDIDATES_30,
            exclude_hidden_row_from_pop=GHOST_CHAIN_RULE_ENABLED,
        )
        for r2 in raw2:
            if r2 is not None and r2[0] > base_chain:
                return 2
    return iv.IGNITION_TRIAL_LIMIT + 1


def _native_min_puyos_to_ignite(
    board: Board, use_native: bool = True,
) -> "iv.IndicatorV2Value":
    """既存 III-5 `min_puyos_to_ignite` の native 分岐版 (実測2000ms/行 → 緊急native化)。

    IGNITION_TRIAL_LIMIT が既存の2以外に変わった場合は非対応 (安全側で
    Python版へフォールバック、`_native_search_min_ignite` が N=1/N=2 決め打ち
    のため)。
    """
    if not (
        use_native and _PUYO_CORE_AVAILABLE
        and _board_is_gravity_consistent(board)
        and iv.IGNITION_TRIAL_LIMIT == 2
    ):
        return iv.min_puyos_to_ignite(board)
    base_chain = _native_simulate_chain(
        board, exclude_hidden_row_from_pop=GHOST_CHAIN_RULE_ENABLED,
    ).chain_count
    n = _native_search_min_ignite(board, base_chain)
    score = 1.0 - (float(n) / float(iv.IGNITION_MAX_PUYOS))
    return iv.IndicatorV2Value(score=iv._clamp01(score), raw=float(n))


def _native_sat_measure_terminal_chain(frontier: "list[Board]") -> "int | None":
    """saturation_chain 終端測定の native 版 (`iv._sat_measure_terminal_chain`
    相当、takapt 30通り native バッチ)。

    終端候補 (最大 beam_width 件) 全てが gravity-consistent の場合のみ
    最大到達連鎖数を返す。1件でも重力違反があれば None を返し、呼び出し側が
    既存 Python 実装にフォールバックする (完全一致を保証する安全弁、
    `_board_is_gravity_consistent` 参照)。
    """
    best = 0
    for candidate in frontier:
        if not _board_is_gravity_consistent(candidate):
            return None
        raw = _native_chain_metrics_after_drops(
            candidate, _NATIVE_DROP_CANDIDATES_30,
            exclude_hidden_row_from_pop=GHOST_CHAIN_RULE_ENABLED,
        )
        for r in raw:
            if r is not None:
                best = max(best, r[0])
    return best


def _native_saturation_chain(
    board: Board,
    fill_ratio: float = iv.SATURATION_FILL_RATIO_DEFAULT,
    beam_width: int = iv.SATURATION_BEAM_WIDTH_DEFAULT,
    use_native: bool = True,
) -> "iv.IndicatorV2Value":
    """既存 C-3 `saturation_chain` の native 分岐版 (実測12500ms/行 → 緊急native化)。

    ビーム構築 (`iv._sat_expand_step`) は元々 simulate 不要で高速なため
    無変更のまま再利用し、唯一の重い箇所 (終端 beam_width 候補×takapt30通り
    = 最大180回のfull simulate) だけ `_native_sat_measure_terminal_chain` に
    置き換える。この関数は `iv.saturation_chain` 本体のビーム構築ループを
    ミラーする (private helper 群を直接呼ぶため、`iv.saturation_chain` の
    アルゴリズムが変わった場合はここも追従が必要 — パリティテスト
    `TestNativeHeavyIndicatorParity` がドリフトを検出する)。
    """
    if not (use_native and _PUYO_CORE_AVAILABLE and _board_is_gravity_consistent(board)):
        return iv.saturation_chain(board, fill_ratio, beam_width)
    if board.is_dead():
        return iv.IndicatorV2Value(score=0.0, raw=0.0)
    target_cells = round(fill_ratio * iv.FULL_BOARD_CAP)
    frontier: "list[Board]" = [board]
    steps = min(
        max(0, target_cells - board.count_puyos()), iv.SATURATION_MAX_BUILD_STEPS,
    )
    for _ in range(steps):
        next_frontier = iv._sat_expand_step(frontier, beam_width)
        if not next_frontier:
            break
        frontier = next_frontier
    best_chain = _native_sat_measure_terminal_chain(frontier)
    if best_chain is None:
        best_chain = iv._sat_measure_terminal_chain(frontier, iv._SHARED_SIMULATOR)
    return iv.IndicatorV2Value(
        score=iv._clamp01(float(best_chain) / iv.NORM_SATURATED_CHAIN),
        raw=float(best_chain),
    )


def _native_saturation_chain_upper(
    board: Board,
    fill_ratio: float = iv.SATURATION_FILL_RATIO_DEFAULT,
    beam_width: int = iv.SATURATION_BEAM_WIDTH_DEFAULT,
    min_fill: float = iv.SATURATION_UPPER_MIN_FILL,
    use_native: bool = True,
) -> "iv.IndicatorV2Value":
    """既存 `iv.saturation_chain_upper` の native 分岐版 (2026-08-13、
    user簡略化決定タスク)。

    **ビーム構築ステップ自体の native化は断念済み** (`iv.
    SATURATION_UPPER_MIN_FILL` 直上コメント参照: 実データで native
    `chain_count` フィルタが既存 `_sat_group_size_after_drop` と食い違う
    ケースを実測、bit-identical要件を優先)。よって本関数がやっているのは
    `_native_saturation_chain` と全く同じ「終端測定のみ native化」であり、
    唯一の追加ロジックは閾値ゲート (min_fill 未満は native呼び出し自体を
    スキップして NaN を返す、無駄な native ラウンドトリップを避ける)。
    """
    current_fill = board.count_puyos() / iv.FULL_BOARD_CAP
    if current_fill < min_fill:
        return iv.IndicatorV2Value(score=float("nan"), raw=float("nan"))
    return _native_saturation_chain(board, fill_ratio, beam_width, use_native)


# full profile 重い4列 (既存) + A-1 native化5列 + 緊急native化2列
# (min_puyos_to_ignite/saturation_chain) + saturation_chain_upper (2026-08-13
# user簡略化決定) の native 版レジストリ (use_native は呼び出し側が
# functools.partial で bind する、`_resolve_indicator_registry` 参照)。
# ここに無いキー (main_linked_pair_count 等) は自動的に
# GRID_ONLY_HEAVY_INDICATORS の Python 実装のまま扱われる。
GRID_ONLY_HEAVY_INDICATORS_NATIVE: dict[str, Callable[..., "iv.IndicatorV2Value"]] = {
    "current_max_chain": _native_current_max_chain,
    "dig_resistance": _native_dig_resistance,
    "ukeyasusa": _native_ukeyasusa,
    "sub_chain_count": _native_sub_chain_count,
    "immediate_fire_power": _native_immediate_fire_power,
    "chain_efficiency": _native_chain_efficiency,
    "second_chain_potential": _native_second_chain_potential,
    "ignition_point_count": _native_ignition_point_count,
    "multi_color_ignition": _native_multi_color_ignition,
    "min_puyos_to_ignite": _native_min_puyos_to_ignite,
    "saturation_chain": _native_saturation_chain,
    "saturation_chain_upper": _native_saturation_chain_upper,
    # タスク#10「移植1」(2026-08-13、puyo_core に simulate_chain_with_steps
    # 追加) で native化可能になった (上記 _native_simultaneous_pop_richness
    # 直上コメント参照)。
    "simultaneous_pop_richness": _native_simultaneous_pop_richness,
}

# ============================
# b-2: 「相手との差」列 (2026-08-12 user確定、決定記録参照)
# ============================
# 一括変換の恒久指示 (2026-08-10 user指示) に従い、diff化してよい列を
# 下記4分類 + 例外1分類で明示リスト化する (リストに無い列は安全側デフォルト
# として diff化しない=own のみ)。

# (1) own を削除して diff_ に完全置換する列。b-2実測で11項目中10項目が
# 「自分のみ」より「差」で当てやすさ改善 (連結最大サイズは終盤0.508→0.567)。
DIFF_REPLACE_OWN_COLUMNS: tuple[str, ...] = (
    "max_column_height",
    "column_bumpiness",
    "death_margin",
    "death_margin_neighbor",
    "conn_pair_count",
    "conn_max_group_size",
)

# (2) own+diff 両方 (full profile 限定・重い連鎖シミュ系)。b-2本文は
# grid-only 系のみ実測済みで、この4列は「同様にdiff化するがownも残すか
# 迷う場合は両方出して報告」という task 側の指示に従い own+diff 両方を
# 出力する (2026-08-12 コーダ判断、user確認待ち)。
DIFF_KEEP_OWN_HEAVY_COLUMNS: tuple[str, ...] = (
    "current_max_chain",
    "dig_resistance",
    "ukeyasusa",
    "sub_chain_count",
)

# (3) own+diff 両方 (user指示8/12最重要: 色ぷよ総数は単独指標でなく
# おじゃま総数とペアで見る。差にすると向きが逆転する謎の答えとして、
# 単純な own/diff置換対象から外し、両方+比率/積の交互作用列で表現する)。
#
# ## 色ぷよ×おじゃまの関係は非単調 (2026-08-12 user伝授、重要)
# 色ぷよ総数とおじゃま総数の「多いほど有利/不利」という単一の向きは無い。
# 条件によって意味が反転する:
#   - おじゃま≈0 のとき: 色ぷよが多いほど有利 (材料律、reference_
#     color_puyo_material_law_2026-08-04「色ぷよの多さは未構造でも構造化
#     に向かえば強い構造につながる」)。
#   - おじゃまが多い かつ 色ぷよも多いとき: 「潰しが刺さった」
#     (色ぷよを溜めて構えていたところに攻撃を受け、構築中の連鎖ごと
#     埋まった) 可能性があり、この場合は不利シグナルになりうる。
# この非単調性は単純な差分・単一係数では表現できないため、own両方+比率+
# 交互作用の4列 (下記 PAIR_INTERACTION_COLUMNS 含む) で学習側に条件分岐的な
# 判断余地を残す設計にしている (単一の「向き」を決め打ちしない)。
DIFF_KEEP_OWN_PAIR_COLUMNS: tuple[str, ...] = (
    "board_color_puyo_total",
    "board_puyo_total",
    "board_ojama_count",
)

# (4) own+diff 両方 (b-1 新設の分解列。新指標のため測定用に両方出す)。
DIFF_KEEP_OWN_NEW_COLUMNS: tuple[str, ...] = (
    "center_bulge_color",
    "center_bulge_ojama",
)

# (5) diff化しない例外 (own のみ残す)。b-2本文: 3個連結は差にすると終盤の
# 不可解な逆転がさらに悪化するため対象外。
#
# 2026-08-13 追加分 (A-1再接続10列 + C-3 saturation_chain + C-4新指標3種):
# b-2 の diff/own 判断は個別に実測 (grid-only系は当てやすさ改善を確認済み)
# して初めて決めており、この新規14列は同種の実測がまだ無い。無評価のまま
# DIFF_KEEP_OWN_HEAVY_COLUMNS 等に混ぜると「diff化がb-2実測で検証済み」と
# 誤解される (過学習への警戒、feedback_overfitting_awareness_2026-08-04)
# ため、安全側デフォルトとして own のみに留める (このexempt扱いは分類
# 完全性テスト tests/test_indicator_pipeline_registry_2026-08-13.py の
# 要求を満たすための暫定分類であり、b-2同様の実測後に再評価すること)。
DIFF_EXEMPT_OWN_ONLY_COLUMNS: tuple[str, ...] = (
    "conn_triple_count",
    # --- A-1 再接続10列 (2026-08-13) ---
    "immediate_fire_power",
    "chain_efficiency",
    "min_puyos_to_ignite",
    "second_chain_potential",
    "main_linked_pair_count",
    "isolated_pair_count",
    "main_linked_ratio",
    "ignition_point_count",
    "multi_color_ignition",
    "simultaneous_pop_richness",
    # --- C-3 (2026-08-13) ---
    "saturation_chain",
    # --- C-4 新指標3種 (2026-08-13) ---
    "color_diversity_evenness",
    "buried_hole_count",
    "chain_articulation_point_count",
    # --- 上部限定軽量版 (2026-08-13 user簡略化決定、未評価のため own のみ) ---
    "saturation_chain_upper",
)

# connectivity_observation() は registry の外で個別に書く固定3列
# (このツール独自の慣習、GRID_ONLY_INDICATORS には乗らない)。
CONN_ALWAYS_PRESENT_COLUMNS: tuple[str, ...] = (
    "conn_pair_count", "conn_triple_count", "conn_max_group_size",
)

# ============================
# 全消しボーナス予約中フラグ (all_clear_bonus_pending)
# — 2026-08-12 user伝授 (設計訂正版)
# ============================
# 本質は「盤面が空かどうか」という瞬間状態ではなく、「全消しボーナス
# (2100点、おじゃま約30個相当 = 2100/OJAMA_RATE_STANDARD(70)) が未消費で
# 残っている」という持続状態。瞬間フラグだと学習が「空である」ことそのもの
# を意味だと誤解しかねないため、ボーナスが「予約中」の区間全体を ON にする
# 状態機械として実装する (_compute_all_clear_bonus_pending)。
# 全消しボーナスの固定値は src.scoring.ALL_CLEAR_BONUS が単一情報源
# (src/chain_detector.py VideoChainTracker が実運用でこの値を「次の連鎖
# 発火時に持ち越し加算」する仕様で実装済み、docs/PUYO_RULES_CONFIRMED_
# 2026-07-22.md 準拠)。ここでの再定義はしない (マジックナンバー重複回避)。
ALL_CLEAR_BONUS_SCORE: int = ALL_CLEAR_BONUS  # = 2100 (src.scoring と同一値)
MAX_DROP_BONUS_SCORE: int = 250  # 通常の落下ボーナス上限 (これ以下は通常運転)
# 連鎖外ジャンプ (ON トリガー) の判定帯。下限は通常の落下ボーナス上限と全消し
# ボーナスのちょうど中間 (双方から十分離してマージンを確保)。
#
# 【実データ検証 (2026-08-12、video_c143) で判明した既知の精度限界】
# chain_mechanism タグは実際の連鎖の一部にしか付かない (1P側の1175点超
# ジャンプ96件中、タグ付きはわずか4件、92件は untagged の実連鎖=数千〜
# 数万点)。下限のみだと数万点ジャンプまで誤検出するため上限も設ける
# (固定2100点から大きく外れる巨大ジャンプは明らかに普通の大型連鎖)。
# 上限を設けても [1175,3150] 帯だけで1動画中72件 (1P+2P) が候補になり、
# 現実的な全消し頻度より明らかに過大 → 帯に入るが実際は中型連鎖という
# 取りこぼしが残る (chain_mechanism のタグ網羅率向上が本質的な解、
# 現状は近似ヒューリスティックである旨をコメントで明示)。
ALL_CLEAR_JUMP_LOWER_BOUND_SCORE: float = (
    (MAX_DROP_BONUS_SCORE + ALL_CLEAR_BONUS_SCORE) / 2.0
)  # = 1175.0
ALL_CLEAR_JUMP_UPPER_BOUND_SCORE: float = ALL_CLEAR_BONUS_SCORE * 1.5  # = 3150.0

# 状態機械が own 列として生成する固定列 (registry の外、CONN_ALWAYS_
# PRESENT_COLUMNS と同じ位置付け)。all_clear_source (2026-08-13 A-4追加):
# all_clear_bonus_pending が真値 (VideoChainTracker.all_clear_pending、npz
# 収集64本目以降) 由来か近似ヒューリスティック由来かを区別する列
# (0=真値/1=近似、_apply_all_clear_truth/_compute_all_clear_bonus_pending
# 参照)。diff/carry対象ではない (own のみ、メタ情報)。
TEMPORAL_STATE_COLUMNS: tuple[str, ...] = (
    "all_clear_bonus_pending", "all_clear_source",
)

# all_clear_source の値 (マジックナンバー禁止のため定数化)。
ALL_CLEAR_SOURCE_TRUTH: float = 0.0  # VideoChainTracker 真値 (npz収集64本目以降)
ALL_CLEAR_SOURCE_APPROX: float = 1.0  # score差分ヒューリスティック近似 (既存)

# chain_mechanism が「連鎖タグ無し」を意味する値の集合 (小文字化して比較)。
# 空文字が通常だが、npz 側の型変換で "nan"/"none" 文字列化する経路があっても
# 安全に「タグ無し」として扱う (誤って「連鎖あり」と誤認しない安全側デフォルト)。
_NO_CHAIN_TAG_VALUES: frozenset[str] = frozenset({"", "nan", "none"})

# 相手側の「直近own値そのもの」を carry する列 (diff = own-opp ではなく
# opp_<col> = 相手の直近own値、2026-08-12 user伝授)。all_clear_bonus_pending
# はフラグ (0/1) なので差分には意味が無く、「相手が今ボーナス保持中か」と
# いう相手側の生の状態そのものが受け側の判断材料になる (自分が攻撃を受ける
# 側かどうかの判断に相手のボーナス保持状態が効くため)。
CARRY_OPPONENT_COLUMNS: tuple[str, ...] = (
    "all_clear_bonus_pending",
)

# ペア交互作用列 (b-2 user指示8/12「色ぷよ×おじゃまの関係性を特徴として
# 捉える」の最小実装)。上記の非単調性コメント参照: color_ojama_ratio_own は
# 「今どれだけ色ぷよ優勢か」、color_diff_x_ojama_diff は「色ぷよを増やしつつ
# おじゃまも増えた (催促を受けながら構築) / 減らしつつ減った (受けて掘った)
# 等の同時変化」を捉える。どちらも単調な有利/不利の向きを仮定しない
# (向きの決定は学習に委ねる、CLAUDE.md「観測軸を提供→学習で重要度を発見」)。
# 0除算防止の epsilon はマジックナンバー化を避けて定数化。
COLOR_OJAMA_RATIO_EPS: float = 1e-6
PAIR_INTERACTION_COLUMNS: tuple[str, ...] = (
    "color_ojama_ratio_own", "color_diff_x_ojama_diff",
    # --- W12根治 (2026-08-16、アーキ設計確定分) ---
    # color_ojama_ratio_own と対になる「色ぷよ×予告おじゃま」版。既存の
    # COLOR_OJAMA_RATIO_EPS をそのまま再利用する (新規epsilon定数を作らない、
    # アーキ指示)。own専用の比率で side非依存の絶対量、diff化・対称化の
    # 一括変換は通さない (このタプル自体が既に own-only レーン)。
    "color_forecast_ratio_own",
)

# ============================
# タスク#8: おじゃま収支の真値CSV統合 (2026-08-13、docs/CROSS_CUTTING_AUDIT_
# 2026-08-13.md P4 決着の反映)
# ============================
# npz 収集側 (`scripts/collect_boards_lean.py`) が64本目以降で
# OjamaAccountingTracker を実駆動して記録した ojama_net_balance/
# ojama_forecast (own-perspective: 1P はそのまま、2P は符号反転済み、
# 自分有利方向が正) を初めてCSVに接続する。旧npz (真値列なし) は列自体が
# 存在しないため NaN 全埋め + ojama_source=OJAMA_SOURCE_MISSING で区別する
# (all_clear_bonus_pending の all_clear_source と同じ「真値/近似(欠損)の
# 出自フラグ」パターン)。
#
# **P4決着 (2026-08-13、docs/CROSS_CUTTING_AUDIT_2026-08-13.md 該当節)**:
# 会計コア自体は瞬時対称 (net_1p(t)+net_2p(t)=0、合成9万ティックで残差ゼロ
# を確認済み・無罪) だが、各側の記録タイミングが独立 (自分の設置直後=自分の
# pending消化直後=自分に有利な瞬間にだけ記録) なため、生の own 値を side別
# にそのまま使うと構造的サンプリングバイアスが乗る (対称乱数合成テストで
# 実データと同符号・同水準の残差を再現し確定)。対処は「分析時の時刻再同期」
# であり、既存の b-2 diff列生成と全く同じ merge_asof(direction="backward")
# パターンを再利用して求める:
#   own(t1) は自分の直近確定値、opp_asof(t1) は同じ t1 時点での相手の
#   「直近確定値」(相手が記録した時刻 t2<=t1 のスナップショット)。
#   net_2p=−net_1p の厳密対称性 (真の同時刻なら own(t)=-opp(t) が常に成立)
#   より、-opp_asof(t1) は「t1時点の own を相手側の観測から推定した値」と
#   解釈できる。own(t1) 単独 (自分に有利な瞬間に偏る) と -opp_asof(t1)
#   (相手に有利な瞬間に偏る、符号反転で見れば自分に不利な瞬間に偏る) を
#   平均するとバイアスが打ち消し合う:
#     ojama_net_balance_synced = (own(t1) − opp_asof(t1)) / 2
#   既存 `_attach_opponent_diff_columns` が計算する diff_<col> = own−opp_asof
#   がまさにこの分子そのものなので、再実装せず同関数をそのまま再利用し
#   (`convert_one_npz` 内、一時列 `_OJAMA_NET_BALANCE_SYNC_RAW_COL` 経由)、
#   得られた diff_ 値を 2 で割るだけで済ませる (パターン完全一致、実装量最小)。
#
# **C-2 猶予量 (ojama_margin) の設計判断**: 「自分の吸収余力 − 実飛来量」を
# 求めるにあたり、受け側容量系の既存指標を検討した:
#   - `death_margin`: raw が「窒息列(3列目)1本の残り段数」であり単位が
#     「行数」。おじゃま個数 (セル数、floor(N/6)行×6列+端数の分配) と直接
#     引き算できない (単位不一致、かつ死亡列以外の空きを無視するため過小)。
#   - `ukeyasusa`: dig_resistance(生存確率)/absorption_capacity/death_margin
#     を重み付け合成した0-1複合スコアであり、そもそも「引き算できる生の
#     容量 (raw)」を持たない。
#   - `absorption_capacity` (raw=ON_FIELD_CAP−count_puyos、単位=セル数):
#     ojama_forecast の raw (単位=個数=セル数) と単位が完全に一致する唯一の
#     候補。**本タスクではこれを採用する** (2026-08-13 コーダ判断)。
#     ただし `absorption_capacity` 自体は a-1 決定 (2026-08-12) で
#     board_puyo_total と完全重複のため本ツールの登録レジストリには無い。
#     Board を再構築せずに済ませるため、既に計算済みの own
#     `board_puyo_total` (score=raw/ON_FIELD_CAP の正確な線形変換、raw は
#     常に0〜72でクランプが効かないため score から raw を完全に逆算できる)
#     から `ON_FIELD_CAP − board_puyo_total_score*ON_FIELD_CAP` として導出
#     する (Board再構築の重複を避ける既存の性能方針と一致、
#     `_attach_ojama_margin_column` docstring 参照)。
#
# **DIFF_* 5分類への非振り分け (意図的)**: 上記4列 (+ojama_source) は
# grid-only レジストリ (`GRID_ONLY_INDICATORS`/`GRID_ONLY_HEAVY_INDICATORS`)
# の外から来る値であり、`DIFF_REPLACE_OWN_COLUMNS` 等5分類の対象は
# 「Board→IndicatorV2Value のレジストリ関数が生成した own 列」に限定されて
# いる (`tests/test_indicator_pipeline_registry_2026-08-13.py::
# TestDiffClassificationCompleteness` の管轄範囲もそこまで)。
# `all_clear_bonus_pending`/`all_clear_source` が `TEMPORAL_STATE_COLUMNS`
# として同5分類の対象外になっている既存の前例と同じ扱いにする (own 列としては
# 出力するが、5分類には登録しない)。net_balance/forecast は元々「相手との
# 差」ではなく「相手を含めた収支そのもの」なので、これをさらに相手と diff化
# するのは概念的に無意味という判断もある (5分類のうち EXEMPT=own onlyと
# 実質同じ扱いになるため、無理に5分類の枠に押し込む必要が無い)。
# W12 (2026-08-16) 追加の `ojama_net_balance_uncapped`/`ojama_forecast_
# uncapped`/`ojama_forecast_log`/`ojama_forecast_progress_interaction`
# (OJAMA_TRUTH_COLUMNS 合計8列+ojama_source) も同じ理由 (grid-only
# レジストリ外・own-perspectiveの収支そのもの) で全く同じ扱いにする
# (diff化・carry化しない own only)。同日追加の `color_forecast_ratio_own`
# は PAIR_INTERACTION_COLUMNS 側 (既存 color_ojama_ratio_own と対、直上
# コメント参照) だが理由は同一。

OJAMA_SOURCE_TRUTH: float = 0.0  # OjamaAccountingTracker 真値 (npz収集64本目以降)
OJAMA_SOURCE_MISSING: float = 1.0  # 旧npz、ojama_net_balance/ojama_forecast 列が存在しない

# convert_one_npz 内でのみ使う一時列名 (最終CSVには出さない、
# `_final_fieldnames` に含めないことで DictWriter の extrasaction="ignore"
# が自動的に除外する)。相手との再同期 (merge_asof) 計算専用の own raw 値。
_OJAMA_NET_BALANCE_SYNC_RAW_COL: str = "_ojama_net_balance_raw_for_sync"

# grid-only レジストリの外から来る「おじゃま収支の真値系」own列 (常に own
# のみ、DIFF_* 5分類には含めない、上記コメント参照)。`_final_fieldnames` の
# own_candidates に `TEMPORAL_STATE_COLUMNS` と同じ扱いで直接追加する。
OJAMA_TRUTH_COLUMNS: tuple[str, ...] = (
    "ojama_net_balance", "ojama_forecast", "ojama_source",
    "ojama_net_balance_synced", "ojama_margin",
    # --- W12 (2026-08-16) 追加: 0-1正規化前の生の個数 (末尾追加、既存順序
    # 保持。EXTRA_INDICATOR_NAMES 末尾追加ルールと同じ精神)。診断・将来の
    # 特徴量設計検討用の材料であり、CLAUDE.md「指標は0-1正規化必須」の
    # 対象外 (モジュール docstring「W12 (2026-08-16) 生値2列の追加」節参照)。
    # `*_raw` でなく `*_uncapped` にした理由も同節参照 (a-1「完全重複raw列
    # 全面禁止」テストとの名前衝突回避)。
    "ojama_net_balance_uncapped", "ojama_forecast_uncapped",
    # --- W12根治 (2026-08-16、アーキ設計確定分。実装は _attach_ojama_
    # forecast_log_columns) ---
    # ojama_forecast_log: log1p(forecast_uncapped)/log1p(PENDING_ABS_CAP)。
    #   飽和(72個超で全て1.0)を対数圧縮で緩和する別表現 (0-1有界は式自体で
    #   保証、PENDING_ABS_CAP=216はOjamaAccountingTrackerの物理上限そのもの
    #   でありsrc.ojama_accountingからimportする、新規定数を作らない)。
    # ojama_forecast_progress_interaction: 上記 × match_progress (両者の
    #   board_puyo_total平均、既存 diff_board_puyo_total から逆算し新規の
    #   merge_asof計算を増やさない)。位相 (序盤/中盤/終盤) ごとに同じ予告量
    #   でも意味が違う (P1実測: 216+予告の実勝率が序盤52.4%/中盤42.1%/
    #   終盤11.3%) ことを1列に圧縮して表現する (位相を列で分けない、
    #   アーキ指示=列数節約)。
    "ojama_forecast_log", "ojama_forecast_progress_interaction",
    # --- 局面(試合の進み具合)を独立列として追加 (2026-08-21) ---
    # match_progress: 両者の board_puyo_total 平均 (0-1)。従来は
    #   ojama_forecast_progress_interaction の内部にだけ埋まっていたため、
    #   学習器 (HistGradientBoostingClassifier, max_depth=4) が「局面ごとに
    #   同じ数値の意味が変わる」構造を自力で学べなかった。素の列として渡すと
    #   全52列 × 局面 の交互作用を木が自力で発見できる (積列を人が決め打ち
    #   しない = 「観測軸を提供して学習で重要度を発見する」設計方針に沿う)。
    # 注意: 試合の相対進行率 (終端でスケールする形) にしてはならない。
    #   終端は未来情報でリークになり、リアルタイム実況でも計算不能。
    #   本列は「いま盤面にどれだけぷよが載っているか」の両者平均であり、
    #   その時点の情報だけで決まる。
    # 注意: 両者で同じ値になる (side 対称) ため単独では勝敗を識別できない。
    #   効果は交互作用経由でのみ出る = 単独 permutation が小さくても
    #   「効いていない」とは判定できない。
    "match_progress",
    # --- おじゃまの非線形な重み (2026-08-21、user指示) ---
    # ojama_damage_forecast: iv.ojama_damage(own_board, forecast_uncapped_raw)。
    #   これまで予告おじゃまは線形の正規化 (ojama_forecast = x/72、
    #   ojama_net_balance = (x+72)/144) だけで学習に入っていた。しかし user
    #   伝授では「おじゃま3個は無害、60個はほぼ死」「折れ点が12個(2段)と
    #   18個(3段)」「盤面が埋まるほど1個あたりの効きが増幅する」という
    #   明確な非線形がある (memory reference_ojama_damage_nonlinear_2026-07-29 /
    #   reference_ojama_damage_function_2026-07-29)。
    #   iv.ojama_damage() はこの構造を「発火点までの余裕段数 − おじゃまの
    #   段数」の引き算1本に統合して実装済みだったが、呼び出し元が打ち合いの
    #   測定スクリプトだけで、勝敗予測には繋がっていなかった (2026-08-21 調査)。
    #   同じ予告量でも盤面の埋まり具合で致命度が変わることを学習に渡す。
    # 上限なしの生値を渡す理由: 予告は 72個で頭打ちにすると「もう死ぬ」域の
    #   差が消える。ojama_damage 側が段数換算で飽和を扱うので、入力は
    #   打ち切らない方が情報が残る。
    "ojama_damage_forecast",
)


def _finite_or_nan_score(
    raw: float, normalize: "Callable[[float], iv.IndicatorV2Value]",
) -> float:
    """raw が有限値の場合のみ既存の正規化関数に委譲しスコアを返す (NaN安全)。

    `iv.ojama_forecast()` は内部で `max(0, forecast)` を使うが、Python の
    `max(0, float("nan"))` は (比較が常に False になるため) **NaN を静かに
    0 に変換してしまう既知の落とし穴**がある。「取得不能 (NaN)」を
    「予告0個」と誤解させないよう、ここで先に `np.isfinite` ガードを掛けて
    NaN を確実にそのまま伝播させる (サイレント破損防止)。
    """
    if not np.isfinite(raw):
        return float("nan")
    return normalize(raw).score


def _ojama_truth_raw_arrays(
    d: "np.lib.npyio.NpzFile", n: int,
) -> "tuple[np.ndarray, np.ndarray, float]":
    """npz からおじゃま収支の真値 raw 配列 (own-perspective) を取り出す。

    両列が揃っている npz (収集64本目以降) のみ真値として採用する。片方だけ
    存在する状態は収集側の実装上想定されないが、安全側 (見なかったことに
    しない) で「両方揃っていなければ両方 NaN 扱い」にする。

    Returns:
        (net_balance_raw, forecast_raw, source) の3値。source は npz 全体で
        1つの値 (OJAMA_SOURCE_TRUTH または OJAMA_SOURCE_MISSING)。行ごとの
        取得不能 (NaN) とは別概念 (OJAMA_TRUTH_COLUMNS 直上コメント参照)。
    """
    has_truth = "ojama_net_balance" in d.files and "ojama_forecast" in d.files
    if has_truth:
        return (
            np.asarray(d["ojama_net_balance"], dtype=np.float64),
            np.asarray(d["ojama_forecast"], dtype=np.float64),
            OJAMA_SOURCE_TRUTH,
        )
    nan_fill = np.full(n, float("nan"), dtype=np.float64)
    return nan_fill, nan_fill.copy(), OJAMA_SOURCE_MISSING


def _attach_ojama_truth_own_columns(
    rows: list[dict], net_raw: np.ndarray, forecast_raw: np.ndarray, source: float,
) -> None:
    """おじゃま収支の真値 own 列を in-place で rows に書き込む。

    ojama_net_balance/ojama_forecast は既存 IV-1/IV-2 正規化関数
    (`iv.ojama_net_balance`/`iv.ojama_forecast`) に委譲する (薄い委譲構造の
    原則、CLAUDE.md 0-1正規化必須ルール)。合わせて後段の計算用に own raw
    値を一時列に保持する: 再同期 (`_attach_ojama_net_balance_synced_column`)
    用の `_OJAMA_NET_BALANCE_SYNC_RAW_COL`、猶予量
    (`_attach_ojama_margin_column`) 用の `_ojama_forecast_raw_for_margin`。

    W12 (2026-08-16) 追加: 同じ raw 値を pop されない最終CSV列
    `ojama_net_balance_uncapped`/`ojama_forecast_uncapped` にも書き込む
    (0-1正規化前の生の個数、`_raw` でなく `_uncapped` にした理由はモジュール
    docstring「W12 生値2列の追加」節参照)。真値の無い旧npzでは net_raw/
    forecast_raw が既に NaN fill 済みのため、この2列も自動的に NaN になる
    (0埋めしない)。
    """
    for i, r in enumerate(rows):
        net = float(net_raw[i])
        forecast = float(forecast_raw[i])
        r["ojama_net_balance"] = _finite_or_nan_score(net, iv.ojama_net_balance)
        r["ojama_forecast"] = _finite_or_nan_score(forecast, iv.ojama_forecast)
        r["ojama_source"] = source
        r[_OJAMA_NET_BALANCE_SYNC_RAW_COL] = net
        r["_ojama_forecast_raw_for_margin"] = forecast
        r["ojama_net_balance_uncapped"] = net
        r["ojama_forecast_uncapped"] = forecast


def _attach_ojama_net_balance_synced_column(rows: list[dict]) -> list[dict]:
    """P4決着の再同期版収支 `ojama_net_balance_synced` を追加する。

    既存の b-2 diff列生成インフラ (`_attach_opponent_diff_columns`、
    merge_asof(direction="backward")) を一時列 `_OJAMA_NET_BALANCE_SYNC_RAW_
    COL` にそのまま適用し、得られる `diff_<一時列名>` (= own − opp の直近
    確定値、対称性の根拠は OJAMA_TRUTH_COLUMNS 直上コメント参照) を 2 で
    割って `iv.ojama_net_balance` で再正規化するだけで済ませる (既存パターン
    の再利用、新規の merge_asof 実装をしない)。
    """
    rows = _attach_opponent_diff_columns(rows, [_OJAMA_NET_BALANCE_SYNC_RAW_COL], ())
    diff_key = f"diff_{_OJAMA_NET_BALANCE_SYNC_RAW_COL}"
    for r in rows:
        delta = r.pop(diff_key, float("nan"))
        synced_raw = delta / 2.0 if np.isfinite(delta) else float("nan")
        r["ojama_net_balance_synced"] = _finite_or_nan_score(
            synced_raw, iv.ojama_net_balance,
        )
        r.pop(_OJAMA_NET_BALANCE_SYNC_RAW_COL, None)
    return rows


def _attach_ojama_margin_column(rows: list[dict]) -> None:
    """C-2 猶予量 `ojama_margin` (吸収余力−実飛来量) を in-place で追加する。

    吸収余力の raw (単位=セル数、absorption_capacity 相当) は own
    `board_puyo_total` の score から逆算する (`ON_FIELD_CAP -
    board_puyo_total_score*ON_FIELD_CAP`、Board再構築を避ける最適化。
    board_puyo_total の raw は常に0〜72でクランプが効かないため score
    からの逆算に精度損失は無い)。実飛来量の raw は own `ojama_forecast`
    (自分に向かう予告個数、既に0以上にクリップ済み) を使う。両者とも
    ojama_forecast と同じ OJAMA_NET_NORM_HALF/FULL (=ON_FIELD_CAP/144) の
    正規化を再利用する (`iv.ojama_net_balance` に委譲、0=収支ゼロ相当を
    0.5 に写す既存の慣習と揃える。死亡列限定の death_margin や複合スコアの
    ukeyasusa を不採用にした理由は本セクション先頭コメント参照)。
    """
    for r in rows:
        forecast_raw = r.pop("_ojama_forecast_raw_for_margin", float("nan"))
        board_total_score = float(r["board_puyo_total"])
        absorption_raw = iv.ON_FIELD_CAP - board_total_score * iv.ON_FIELD_CAP
        if not np.isfinite(forecast_raw):
            r["ojama_margin"] = float("nan")
            continue
        margin_raw = absorption_raw - max(0.0, forecast_raw)
        r["ojama_margin"] = iv.ojama_net_balance(margin_raw).score


def _attach_ojama_forecast_log_columns(
    rows: list[dict],
    grids: "np.ndarray | None" = None,
) -> None:
    """W12根治 (2026-08-16、アーキ設計確定分) の3列を in-place で追加する。

    docs/KNOWN_WEAKNESSES.md W12「学習モデルが未着弾おじゃま予告をほぼ
    無視する」の原因の一つである `ojama_forecast` の /72 飽和 (72個超は
    全て同じ値) を緩和する目的で、既に確保済みの `ojama_forecast_uncapped`
    (生の予告個数、pop されない列) を材料に以下3列を追加する。

    - `ojama_forecast_log`: log1p(forecast_uncapped)/log1p(PENDING_ABS_CAP)。
      式自体が [0,1] に有界 (PENDING_ABS_CAP が理論上の最大値のため)。
      浮動小数の丸め対策として `iv._clamp01` を通す (他の指標関数と同じ
      慣習、build_labeled_win_from_npz.py 内で既に多用されている private
      ヘルパーの共有利用)。`PENDING_ABS_CAP` (=ON_FIELD_CAP*3=216) は
      `src/ojama_accounting.py` の `OjamaAccountingTracker` が
      forecast_incoming に実際に掛けている物理上限そのもの。新規定数を
      作らずそのまま import して使う (アーキ設計指示)。
    - `ojama_forecast_progress_interaction`: 上記 × match_progress。
      match_progress (両者の board_puyo_total スコア平均、0-1、
      `scripts/visualize_advantage_overlay.py::_match_progress_from_totals`
      と同じ定義) は新規の merge_asof を増やさず、既に計算済みの
      `diff_board_puyo_total` (= own − opp_asof、b-2で計算済み) から
      代数的に逆算する: opp_asof = own − diff ⇒ (own+opp_asof)/2 =
      own − diff/2。**本関数は `_attach_opponent_diff_columns` の後に
      呼ぶこと** (diff_board_puyo_total が存在している前提、呼出順は
      `convert_one_npz` 参照)。同じ予告量でも位相 (序盤/中盤/終盤) で
      意味が違う (P1実測: 予告216+の実勝率が序盤52.4%/中盤42.1%/終盤
      11.3%) ことを列を増やさず1列の積で表現する (アーキ指示: 位相は
      列を分けず乗算1列に圧縮、列数節約)。match_progress 自体は中間値
      であり単独の CSV 列としては出力しない。
    - `color_forecast_ratio_own`: color_raw/(color_raw+forecast_raw+EPS)。
      既存 `color_ojama_ratio_own` (色ぷよ×盤面おじゃま) と対になる、
      色ぷよ×予告おじゃま版。EPS は既存 `COLOR_OJAMA_RATIO_EPS` をそのまま
      再利用する (新規epsilon定数を作らない、アーキ指示)。

    3列とも own-perspective の絶対量/比率 (side非依存) であり、
    `OJAMA_TRUTH_COLUMNS`/`PAIR_INTERACTION_COLUMNS` と同じ「grid-only
    レジストリ外・own only」レーンに乗せる (b-2 の DIFF_* 5分類・対称化の
    一括変換は通さない。side依存量を誤って一括変換に流すと壊れた過去事故
    への対策、feedback_symmetry_flip_column_types_2026-08-10 2026-08-10
    user恒久指示)。真値の無い旧npz、相手の直近確定値が未確定な先頭行
    (`diff_board_puyo_total` がNaN) では NaN になる (0埋めしない、既存の
    欠損明示パターンと同じ)。

    採用しなかった設計 (アーキ確定): 容量との交互作用は非単調 (空き
    36-53が54-71より高い逆転) で交絡濃厚のため保留 (既存 `ojama_margin`
    で代替)、猶予時間はn不足で保留 (63本の再収集後に再検討)。
    """
    log_denom = math.log1p(PENDING_ABS_CAP)
    # grids[i] を引くため enumerate にする (2026-08-21、ojama_damage_forecast)。
    # rows と grids は同一 npz の同じ行順なので index が一致する
    # (convert_one_npz が rows[i] を grids[i] から作っている)。
    for i, r in enumerate(rows):
        forecast_raw = float(r.get("ojama_forecast_uncapped", float("nan")))
        color_score = float(r.get("board_color_puyo_total", float("nan")))
        color_raw = color_score * iv.ON_FIELD_CAP

        if np.isfinite(forecast_raw):
            forecast_log = iv._clamp01(
                math.log1p(max(0.0, forecast_raw)) / log_denom,
            )
        else:
            forecast_log = float("nan")
        r["ojama_forecast_log"] = forecast_log

        own_total_score = float(r.get("board_puyo_total", float("nan")))
        diff_total = float(r.get("diff_board_puyo_total", float("nan")))
        # 局面は予告の可否とは独立に決まるので、先に単独で確定させる
        # (2026-08-21)。従来は交互作用の内部でしか計算しておらず、
        # 予告が読めない行では局面も失われていた。
        if np.isfinite(own_total_score) and np.isfinite(diff_total):
            match_progress = iv._clamp01(own_total_score - diff_total / 2.0)
        else:
            match_progress = float("nan")
        r["match_progress"] = match_progress
        if np.isfinite(forecast_log) and np.isfinite(match_progress):
            r["ojama_forecast_progress_interaction"] = forecast_log * match_progress
        else:
            r["ojama_forecast_progress_interaction"] = float("nan")

        if np.isfinite(color_raw) and np.isfinite(forecast_raw):
            r["color_forecast_ratio_own"] = color_raw / (
                color_raw + max(0.0, forecast_raw) + COLOR_OJAMA_RATIO_EPS
            )
        else:
            r["color_forecast_ratio_own"] = float("nan")

        # おじゃまの非線形な重み (2026-08-21、user指示)。
        # grids が渡されないとき (旧い呼び出し) は列を作らない = 従来の挙動。
        if grids is None:
            continue
        if not np.isfinite(forecast_raw):
            r["ojama_damage_forecast"] = float("nan")
            continue
        # Board の復元は _compute_row と同じ流儀 (from_list) を使う。
        # 例外は握りつぶさない: 握りつぶすと全行 NaN になっていても
        # 「欠測が多い列」に見えてしまい、実装ミスに気づけない
        # (2026-08-21 実際にそれで一度全欠測を作った)。
        board = Board.from_list(np.asarray(grids[i]).tolist())
        r["ojama_damage_forecast"] = float(
            iv.ojama_damage(board, int(max(0.0, forecast_raw))).score,
        )


def _resolve_indicator_registry(
    profile: str, use_native: bool = True, with_saturation_chain: bool = False,
) -> dict[str, Callable[[Board], "iv.IndicatorV2Value"]]:
    """profile に応じて使う指標レジストリを確定する (light はheavy除外)。

    use_native (2026-08-13 追加、既定 True): full profile の重い列を
    Rust拡張 puyo_core 経由で計算する native 版レジストリ
    (GRID_ONLY_HEAVY_INDICATORS_NATIVE) に切り替える。拡張未導入環境では
    各関数が自動的に既存 Python 実装 (GRID_ONLY_HEAVY_INDICATORS 相当) へ
    フォールバックするため、通常この引数は既定値のままでよい。False を
    渡すと native 分岐を無条件で無効化する (パリティ検証・デバッグ用)。

    with_saturation_chain (2026-08-13 追加、既定 False): OPTIONAL_HEAVY_
    INDICATOR_NAMES (現状 saturation_chain のみ) を registry に含めるか。
    実測で1行8〜18秒という桁違いのコストが判明したための opt-in化
    (定数直上コメント参照)。既定 False = 148本フル実行の既定挙動から除外。
    """
    if profile != "full":
        return dict(GRID_ONLY_INDICATORS)
    # Python実装を土台にし (2026-08-13 バグ修正: 以前は
    # GRID_ONLY_HEAVY_INDICATORS_NATIVE に無いキーが full profile から
    # 丸ごと消える潜在バグがあった。A-1 で native 版を持たない新規重い列
    # [min_puyos_to_ignite 等] を追加したことで顕在化する前に修正)、
    # native 版を持つキーだけ functools.partial で上書きする。
    heavy: dict[str, Callable[..., "iv.IndicatorV2Value"]] = dict(GRID_ONLY_HEAVY_INDICATORS)
    if not with_saturation_chain:
        for name in OPTIONAL_HEAVY_INDICATOR_NAMES:
            heavy.pop(name, None)
    for name, fn in GRID_ONLY_HEAVY_INDICATORS_NATIVE.items():
        if name not in heavy:
            continue  # opt-outされた列 (saturation_chain) を誤って復活させない
        heavy[name] = functools.partial(fn, use_native=use_native)
    return {**GRID_ONLY_INDICATORS, **heavy}


def _resolve_diff_target_columns(
    registry: dict[str, Callable[[Board], "iv.IndicatorV2Value"]],
) -> list[str]:
    """レジストリに実在する列のうち diff_ 化対象を返す (b-2 決定記録の4分類)。

    DIFF_EXEMPT_OWN_ONLY_COLUMNS (例外) は候補に含めない。レジストリに無い
    列 (light profile での heavy 系等) は自動的に除外される。
    """
    candidates = (
        DIFF_REPLACE_OWN_COLUMNS + DIFF_KEEP_OWN_PAIR_COLUMNS
        + DIFF_KEEP_OWN_NEW_COLUMNS + DIFF_KEEP_OWN_HEAVY_COLUMNS
    )
    available = set(registry.keys()) | set(CONN_ALWAYS_PRESENT_COLUMNS)
    return [c for c in candidates if c in available]


def _resolve_carry_target_columns(
    registry: dict[str, Callable[[Board], "iv.IndicatorV2Value"]],
) -> list[str]:
    """実在する列のうち opp_<col> carry 対象を返す。

    CARRY_OPPONENT_COLUMNS (相手の直近own値をそのまま残す列、
    all_clear_bonus_pending 等のフラグ向け、2026-08-12 user伝授) の実在
    チェック。TEMPORAL_STATE_COLUMNS は registry の外で常に生成されるため
    別途 available に含める。
    """
    available = set(registry.keys()) | set(TEMPORAL_STATE_COLUMNS)
    return [c for c in CARRY_OPPONENT_COLUMNS if c in available]


# CSV メタ列 (labeled_win.csv 既存フォーマットと同じ、tsumo は近似値)
META_COLUMNS: tuple[str, ...] = (
    "video_id", "game_idx", "t_sec", "frame", "tsumo", "side", "won",
)


def _final_fieldnames(
    profile: str, with_saturation_chain: bool = False,
) -> list[str]:
    """出力CSVの最終列順を構築する (a-1 raw削除・b-1 分解・b-2 diff/carry化を反映)。

    own 側の出力対象は DIFF_REPLACE_OWN_COLUMNS を除いた列のみ (diff_ に
    完全置換した列は own を書かない、CSVには乗せず内部計算だけに使う)。

    with_saturation_chain (2026-08-13 追加、既定 False): opt-in 指標
    saturation_chain を含めるか (`_resolve_indicator_registry` 参照)。
    """
    registry = _resolve_indicator_registry(
        profile, with_saturation_chain=with_saturation_chain,
    )
    own_candidates = (
        list(registry.keys()) + list(CONN_ALWAYS_PRESENT_COLUMNS)
        + list(TEMPORAL_STATE_COLUMNS) + list(OJAMA_TRUTH_COLUMNS)
    )
    own_output_cols = [c for c in own_candidates if c not in DIFF_REPLACE_OWN_COLUMNS]
    diff_output_cols = [f"diff_{c}" for c in _resolve_diff_target_columns(registry)]
    carry_output_cols = [f"opp_{c}" for c in _resolve_carry_target_columns(registry)]
    return (
        list(META_COLUMNS) + own_output_cols + diff_output_cols
        + carry_output_cols + list(PAIR_INTERACTION_COLUMNS)
    )


def _compute_row(
    grid: np.ndarray,
    registry: dict[str, Callable[[Board], "iv.IndicatorV2Value"]],
) -> dict[str, float]:
    """1 スナップショットの grid からレジストリ内の全指標を計算する。

    Board 再構築を 1 回だけ行い、レジストリの全関数に共有する (npz→Board
    復元コストの重複を避ける)。connectivity_observation (score を持たない
    タプル戻り値) は個別に追加する (collect_indicators_v2.py と同じ流儀)。

    a-1 決定 (2026-08-12) により `*_raw` はここで一切書かない (score のみ、
    CSV出力から完全重複列を削除)。
    """
    board = Board.from_list(grid.tolist())
    row: dict[str, float] = {}
    for name, fn in registry.items():
        row[name] = fn(board).score
    total_conn, _ = iv.connectivity_observation(board)
    row["conn_pair_count"] = float(total_conn.pair_count)
    row["conn_triple_count"] = float(total_conn.triple_count)
    row["conn_max_group_size"] = float(total_conn.max_group_size)
    return row


def _grouped_side_frames(
    df: pd.DataFrame, side: str, group_cols: list[str], diff_cols: list[str],
) -> dict[tuple, pd.DataFrame]:
    """指定 side の行を (video_id, game_idx) 単位の t_sec 昇順 df に分割する。

    scripts/_analyze_reorg_diff_2026-08-12.py の `_grouped_opp_dict` を移植
    (相手側として使う際の下ごしらえ、b-2)。
    """
    src = df[df["side"] == side][group_cols + ["t_sec"] + diff_cols]
    out: dict[tuple, pd.DataFrame] = {}
    for key, g in src.groupby(group_cols, sort=False):
        out[key] = g.sort_values("t_sec").drop(columns=group_cols).reset_index(drop=True)
    return out


def _asof_attach_opponent(
    self_df: pd.DataFrame,
    opp_groups: dict[tuple, pd.DataFrame],
    group_cols: list[str],
    diff_cols: list[str],
) -> pd.DataFrame:
    """自分の各行に「相手の直近確定値」を merge_asof(backward) で対応付ける。

    scripts/_analyze_reorg_diff_2026-08-12.py の `_asof_join_by_game` を移植
    (b-2)。相手の同時刻以前の最新スナップショットが無い先頭区間は NaN に
    なる (実対応成功率は同スクリプトの検証で99.3%、b-2 決定記録参照)。
    """
    opp_col_names = [f"_opp_{c}" for c in diff_cols]
    rename_map = dict(zip(diff_cols, opp_col_names))
    parts: list[pd.DataFrame] = []
    for key, g in self_df.groupby(group_cols, sort=False):
        gs = g.sort_values("t_sec").reset_index(drop=True)
        opp_g = opp_groups.get(key)
        if opp_g is None or len(opp_g) == 0:
            gs2 = gs.copy()
            for c in opp_col_names:
                gs2[c] = float("nan")
            parts.append(gs2)
            continue
        opp_renamed = opp_g.rename(columns=rename_map)
        merged = pd.merge_asof(gs, opp_renamed, on="t_sec", direction="backward")
        parts.append(merged)
    return pd.concat(parts, ignore_index=True) if parts else self_df


def _attach_opponent_diff_columns(
    rows: list[dict], diff_cols: list[str], carry_cols: tuple[str, ...] = (),
) -> list[dict]:
    """1P/2P 行に「相手の直近確定値」由来の列を追加する (b-2)。

    (video_id, game_idx) 単位で相手側の直近確定値を対応付け、
    diff_cols は diff_<col> = 自分の値 - 相手の直近値 を、carry_cols は
    opp_<col> = 相手の直近値そのもの (差分をとらない) を書く。carry は
    all_clear_bonus_pending 等の 0/1 フラグ向け (2026-08-12 user伝授:
    フラグの差分は無意味で、相手の生の状態そのものが判断材料になる)。
    1 npz = 1 動画である
    前提 (video_id は事実上定数だが、複数動画結合にも安全なよう group_cols
    に含める)。side が "1P"/"2P" 以外の行は対応不能として NaN で素通しする
    (サイレントにデータを消さない)。

    Args:
        rows: convert_one_npz が積んだ meta+indicator 行のリスト。
        diff_cols: diff_ 化する base 列名リスト (_resolve_diff_target_columns)。
        carry_cols: opp_ carry 化する base 列名リスト
            (_resolve_carry_target_columns、既定は空=後方互換)。

    Returns:
        list[dict]: 各行に diff_<col> / opp_<col> キーを追加したリスト。
    """
    pair_cols = list(diff_cols) + [c for c in carry_cols if c not in diff_cols]
    if not pair_cols or not rows:
        return rows
    df = pd.DataFrame(rows)
    group_cols = ["video_id", "game_idx"]
    known_sides = ("1P", "2P")
    parts: list[pd.DataFrame] = []
    for self_side, opp_side in (("1P", "2P"), ("2P", "1P")):
        self_df = df[df["side"] == self_side].reset_index(drop=True)
        if len(self_df) == 0:
            continue
        opp_groups = _grouped_side_frames(df, opp_side, group_cols, pair_cols)
        parts.append(_asof_attach_opponent(self_df, opp_groups, group_cols, pair_cols))
    other_df = df[~df["side"].isin(known_sides)].reset_index(drop=True)
    if len(other_df) > 0:
        for c in pair_cols:
            other_df[f"_opp_{c}"] = float("nan")
        parts.append(other_df)
    combined = pd.concat(parts, ignore_index=True) if parts else df
    for c in diff_cols:
        combined[f"diff_{c}"] = combined[c] - combined[f"_opp_{c}"]
        combined = combined.drop(columns=[f"_opp_{c}"])
    for c in carry_cols:
        combined[f"opp_{c}"] = combined[f"_opp_{c}"]
        combined = combined.drop(columns=[f"_opp_{c}"])
    return combined.to_dict("records")


def _add_pair_interaction_columns(rows: list[dict]) -> list[dict]:
    """色ぷよ×おじゃまのペア交互作用列を追加する (b-2 user指示8/12)。

    色ぷよ総数は単独指標でなくおじゃま総数とのペアで見る必要がある
    (2026-08-12 user指示)。own の比率 + diff の積の2列を新設する。
    diff_board_color_puyo_total / diff_board_ojama_count は常に
    DIFF_KEEP_OWN_PAIR_COLUMNS 経由で存在する前提 (存在しない場合は
    NaN を素通しする、サイレント失敗にしない)。
    """
    for row in rows:
        color = float(row.get("board_color_puyo_total", float("nan")))
        ojama = float(row.get("board_ojama_count", float("nan")))
        row["color_ojama_ratio_own"] = color / (color + ojama + COLOR_OJAMA_RATIO_EPS)
        diff_color = float(row.get("diff_board_color_puyo_total", float("nan")))
        diff_ojama = float(row.get("diff_board_ojama_count", float("nan")))
        row["color_diff_x_ojama_diff"] = diff_color * diff_ojama
    return rows


def _all_clear_next_state(state: float, delta: float, chain_tag: str) -> float:
    """1ステップ分の状態遷移 (ON/OFF 判定の核心。定数の意味は
    ALL_CLEAR_JUMP_LOWER/UPPER_BOUND_SCORE・MAX_DROP_BONUS_SCORE 直上の
    コメント、既知の精度限界は _all_clear_state_for_group docstring 参照)。
    """
    is_tagged = chain_tag.strip().lower() not in _NO_CHAIN_TAG_VALUES
    in_bonus_band = (
        ALL_CLEAR_JUMP_LOWER_BOUND_SCORE < delta <= ALL_CLEAR_JUMP_UPPER_BOUND_SCORE
    )
    if state == 0.0:
        return 1.0 if (not is_tagged and in_bonus_band) else 0.0
    return 0.0 if (is_tagged or delta > MAX_DROP_BONUS_SCORE) else 1.0


def _all_clear_state_for_group(
    scores: list[int], chain_tags: list[str],
) -> list[float]:
    """1つの (video_id, side, game_idx) 内、t_sec昇順の score/chain_mechanism
    列から all_clear_bonus_pending の状態遷移を計算する内部ヘルパー。

    状態機械の要点 (2026-08-12 user伝授、設計訂正版。全消しの本質は瞬間の
    空盤面でなく「ボーナス2100点が未消費で残っている」持続状態):
    新しい試合の最初の行は「ボーナス未保持」(0.0) から始まる (試合を
    またいでボーナスは持ち越されない)。ON/OFF 判定の実体は
    _all_clear_next_state() に委譲する。score が読めない行 (-1、または
    直前の既知scoreがまだ無い) は増分を計算できないため NaN を返し、
    状態自体は変更しない (score OCR破綻の既知3動画 c26/c58/c69 では、
    行0以外の列全体が実質NaNになる)。

    既知の精度限界 (要review): chain_mechanism のタグ網羅率が低いデータ
    では ON 判定帯に中型連鎖の得点も多数入り込み、全消し以外を誤って ON に
    する (実測: video_c143 1本で1P+2P合計72件が該当、現実的な全消し頻度
    より過大)。本実装は user 指定の検出方法をそのまま実装した近似
    ヒューリスティックであり、精度向上には chain_mechanism タグの網羅率
    向上または別の検出法が必要。
    """
    n = len(scores)
    out: list[float] = [float("nan")] * n
    state = 0.0
    last_valid_score: int | None = None
    for i in range(n):
        if i == 0:
            out[0] = state
            if scores[0] != -1:
                last_valid_score = scores[0]
            continue
        cur = scores[i]
        if cur == -1 or last_valid_score is None:
            if cur != -1:
                last_valid_score = cur
            continue
        delta = float(cur - last_valid_score)
        last_valid_score = cur
        state = _all_clear_next_state(state, delta, chain_tags[i])
        out[i] = state
    return out


def _compute_all_clear_bonus_pending(
    rows: list[dict], scores: list[int], chain_mechanisms: list[str],
) -> None:
    """(video_id, side, game_idx) 単位で全消しボーナス予約中フラグを計算し、
    rows[i]["all_clear_bonus_pending"] に in-place で書き込む。

    scores / chain_mechanisms は rows と同じ添字で対応する前提 (convert_
    one_npz が同じループで構築するため)。状態遷移の実体は
    _all_clear_state_for_group() 参照。近似ヒューリスティック経路のため
    all_clear_source は常に ALL_CLEAR_SOURCE_APPROX (1.0、A-4 2026-08-13)。
    """
    groups: dict[tuple, list[int]] = {}
    for i, r in enumerate(rows):
        key = (r["video_id"], r["side"], r["game_idx"])
        groups.setdefault(key, []).append(i)
    for idxs in groups.values():
        idxs.sort(key=lambda i: rows[i]["t_sec"])
        group_scores = [scores[i] for i in idxs]
        group_tags = [chain_mechanisms[i] for i in idxs]
        states = _all_clear_state_for_group(group_scores, group_tags)
        for pos, i in enumerate(idxs):
            rows[i]["all_clear_bonus_pending"] = states[pos]
            rows[i]["all_clear_source"] = ALL_CLEAR_SOURCE_APPROX


def _apply_all_clear_truth(rows: list[dict], all_clear_pending: np.ndarray) -> None:
    """npz の all_clear_pending 真値列 (A-4, 2026-08-13) を採用する。

    `src/chain_detector.py::VideoChainTracker.all_clear_pending` が実運用
    パイプラインで厳密に追跡した値 (npz収集64本目以降のみ収録)。近似
    ヒューリスティック (`_compute_all_clear_bonus_pending`) より精度が高い
    ため存在すればこちらを優先し、rows と同じ添字で対応する前提で
    all_clear_bonus_pending にそのまま書き込み、all_clear_source を
    ALL_CLEAR_SOURCE_TRUTH (0.0) に設定する (in-place)。

    Args:
        rows: convert_one_npz が積んだ meta+indicator 行のリスト。
        all_clear_pending: npz の "all_clear_pending" 配列 (rows と同じ添字)。
    """
    for i, r in enumerate(rows):
        r["all_clear_bonus_pending"] = float(all_clear_pending[i])
        r["all_clear_source"] = ALL_CLEAR_SOURCE_TRUTH


def _approx_tsumo(rows_meta: list[dict]) -> None:
    """(video_id, side, game_idx) 内の t_sec 順位で手数を近似する (in-place)。

    実際の tsumo_count (RecognitionPipeline.tsumo_count) とは異なり得る近似
    値 (STABLE スナップショット数 != 常に手数、ojama落下等でも新スナップ
    ショットが生まれるため)。現行 FEATURES/FEATURE_CANDIDATES はどの列も
    tsumo に依存しないため、メタ情報としての近似で実害はない
    (model_indicator_win.META_COLS が特徴量から除外する対象と同じ扱い)。
    """
    groups: dict[tuple, list[int]] = {}
    for i, r in enumerate(rows_meta):
        key = (r["video_id"], r["side"], r["game_idx"])
        groups.setdefault(key, []).append(i)
    for idxs in groups.values():
        idxs.sort(key=lambda i: rows_meta[i]["t_sec"])
        for rank, i in enumerate(idxs):
            rows_meta[i]["tsumo"] = rank


def _build_meta_rows(
    video_ids: np.ndarray, game_idxs: np.ndarray, t_secs: np.ndarray,
    frame_idxs: np.ndarray, sides: np.ndarray, wons: np.ndarray,
) -> list[dict]:
    """npz のメタ配列群から meta 列のみの行リストを組み立てる。"""
    return [
        {
            "video_id": str(video_ids[i]), "game_idx": int(game_idxs[i]),
            "t_sec": float(t_secs[i]), "frame": int(frame_idxs[i]),
            "side": str(sides[i]), "won": float(wons[i]),
        }
        for i in range(len(video_ids))
    ]


class _RowMaskedNpz:
    """np.load() の NpzFile を行マスクで透過的にフィルタするビュー
    (境界実装の仕上げ、2026-08-18、--exclude-match-end-locked 専用)。

    既存コードは全て `d[key]` (numpy 配列の再構築) / `key in d.files` の
    形式でのみアクセスするため、この最小プロキシで __getitem__ にだけ
    マスクを適用すればフィルタを一箇所に閉じ込められる (呼び出し側の
    convert_one_npz 本体は一切変更不要)。
    """

    def __init__(self, d: object, mask: np.ndarray) -> None:
        self._d = d
        self._mask = mask

    @property
    def files(self) -> list[str]:
        return self._d.files  # type: ignore[attr-defined]

    def __getitem__(self, key: str) -> np.ndarray:
        return self._d[key][self._mask]  # type: ignore[index]


def _apply_match_end_locked_filter(d: object) -> "tuple[object, int]":
    """match_end_locked==1 or post_match_lockdown_active==1 の行を除外する
    マスクビューを返す (--exclude-match-end-locked 専用、2026-08-18 追加、
    境界実装の仕上げ)。

    match_end_lockeds (W20/W21根治、2026-08-17) / post_match_lockdown_
    actives (2026-08-18) はいずれも「勝敗演出パネル表示中〜次試合開始まで
    のマーカー列」(collect_boards_lean.py 参照、除外は行わずマーキングのみ
    で収集する設計)。本フィルタは学習データ生成時にオプトインで初めて
    実際の除外を行う。

    値 -1 (未取得・旧npz、後方互換の UNKNOWN sentinel) は除外しない
    (fail-safe: 「わかっている」行だけを除外し、わからない行は残す)。
    両列が存在しない旧npzでは何もしない (mask 全True、後方互換)。

    Returns:
        (フィルタ後のビュー (除外0件なら元の d をそのまま返す), 除外行数)。
    """
    n = int(len(d["grids"]))  # type: ignore[index]
    mask = np.ones(n, dtype=bool)
    if "match_end_locked" in d.files:  # type: ignore[attr-defined]
        mask &= d["match_end_locked"] != 1  # type: ignore[index]
    if "post_match_lockdown_active" in d.files:  # type: ignore[attr-defined]
        mask &= d["post_match_lockdown_active"] != 1  # type: ignore[index]
    n_excluded = int(n - int(mask.sum()))
    if n_excluded == 0:
        return d, 0
    return _RowMaskedNpz(d, mask), n_excluded


def convert_one_npz(
    npz_path: Path, registry: dict[str, Callable[[Board], "iv.IndicatorV2Value"]],
    exclude_match_end_locked: bool = False,
) -> list[dict]:
    """1 npz ファイルを labeled_win 形式の行リストに変換する。

    npz は zlib 圧縮 (np.savez_compressed) のため、`d[key]` は毎回アーカイブ
    メンバを再展開するコストがある。ループの外で各配列を 1 回だけ取り出す
    (2026-08-12 実測: これを怠ると 1本 6693行で 99.8s、修正後は同じ内容が
    数秒で終わる。原情報を安く再変換できることが選択肢C成立の前提のため
    この最適化は必須)。

    b-2/全消しボーナスフラグの詳細はモジュール docstring 参照
    (「指標大整理」節)。1 npz = 1 動画・1P/2P混在という前提で処理する
    (実データで確認済み)。

    Args:
        exclude_match_end_locked: True で match_end_locked==1 または
            post_match_lockdown_active==1 の行を変換対象から除外する
            (境界実装の仕上げ、2026-08-18 追加、オプトイン・既定 False で
            後方互換)。列が存在しない旧npzでは無効化される (何も除外しない)。
    """
    d = np.load(str(npz_path), allow_pickle=True)
    if exclude_match_end_locked:
        d, n_excluded = _apply_match_end_locked_filter(d)
        if n_excluded:
            print(
                f"[exclude-match-end-locked] {npz_path.name}: "
                f"{n_excluded}行除外 (match_end_locked/post_match_lockdown_"
                f"active==1)",
            )
    grids = d["grids"]
    n = len(grids)
    scores = d["score"] if "score" in d.files else np.full(n, -1, dtype=np.int64)
    chain_mechanisms = (
        d["chain_mechanism"] if "chain_mechanism" in d.files else np.array([""] * n)
    )
    rows = _build_meta_rows(
        d["video_id"], d["game_idx"], d["t_sec"], d["frame_idx"], d["side"], d["won"],
    )
    _approx_tsumo(rows)
    for i in range(n):
        rows[i].update(_compute_row(grids[i], registry))
    # A-4 (2026-08-13): npz に VideoChainTracker 真値があればそれを優先採用、
    # 無い旧npzのみ近似ヒューリスティックにフォールバックする。
    if "all_clear_pending" in d.files:
        _apply_all_clear_truth(rows, d["all_clear_pending"])
    else:
        _compute_all_clear_bonus_pending(
            rows, [int(s) for s in scores], [str(m) for m in chain_mechanisms],
        )
    # タスク#8 (2026-08-13): おじゃま収支の真値統合。own列書き込み→猶予量→
    # 再同期 (相手との merge_asof を要するため own列書き込み後に行う) の順。
    net_raw, forecast_raw, ojama_source = _ojama_truth_raw_arrays(d, n)
    _attach_ojama_truth_own_columns(rows, net_raw, forecast_raw, ojama_source)
    _attach_ojama_margin_column(rows)
    rows = _attach_ojama_net_balance_synced_column(rows)
    diff_cols = _resolve_diff_target_columns(registry)
    carry_cols = _resolve_carry_target_columns(registry)
    rows = _attach_opponent_diff_columns(rows, diff_cols, tuple(carry_cols))
    # W12根治 (2026-08-16): diff_board_puyo_total (直上で計算済み) を要する
    # ため必ず _attach_opponent_diff_columns の後に呼ぶ (関数docstring参照)。
    _attach_ojama_forecast_log_columns(rows, grids)
    rows = _add_pair_interaction_columns(rows)
    return rows


def _split_broken_videos(
    npz_files: list[Path],
) -> "tuple[list[Path], list[tuple[str, int]]]":
    """npz ファイル一覧を (隔離対象を除いたリスト, 隔離した (ファイル名, 行数)) に分ける。

    A-2 (2026-08-13): BROKEN_VIDEOS (score OCR破綻・won欠損100%) を stem
    (拡張子・ディレクトリを除いたファイル名) で照合する。行数は grids 配列の
    長さのみを読む軽量操作 (指標計算はしない、隔離対象を数えるためだけ)。
    """
    kept: list[Path] = []
    excluded: list[tuple[str, int]] = []
    for p in npz_files:
        if p.stem in BROKEN_VIDEOS:
            n_rows = int(np.load(str(p), allow_pickle=True)["grids"].shape[0])
            excluded.append((p.name, n_rows))
        else:
            kept.append(p)
    return kept, excluded


def convert_dir(
    npz_dir: Path, out_csv: Path, profile: str = "light", use_native: bool = True,
    exclude_broken: bool = True, with_saturation_chain: bool = False,
    exclude_match_end_locked: bool = False,
) -> tuple[int, float]:
    """npz_dir 内の全 npz を変換し out_csv に書き出す。

    Args:
        use_native: full profile の重い4列を Rust拡張 puyo_core 経由で計算
            するか (2026-08-13 追加、既定 True、後方互換の optional 引数)。
            `_resolve_indicator_registry` 参照。
        exclude_broken: BROKEN_VIDEOS (score OCR破綻・won欠損100%、A-2
            2026-08-13追加) を変換対象から除外するか (既定 True、後方互換の
            optional 引数)。隔離した動画・行数は必ずログ出力する
            (黙って落とさない、feedback_progress_report 系の恒久方針)。
        with_saturation_chain: opt-in指標 saturation_chain を含めるか
            (2026-08-13追加、既定 False、後方互換の optional 引数)。実測で
            1行8〜18秒という桁違いのコストが判明したため既定OFF
            (`OPTIONAL_HEAVY_INDICATOR_NAMES` 直上コメント参照)。
        exclude_match_end_locked: True で match_end_locked==1 または
            post_match_lockdown_active==1 の行 (勝敗演出パネル〜次試合開始
            までのマーカー列、collect_boards_lean.py 参照) を学習データから
            除外する (境界実装の仕上げ、2026-08-18 追加、オプトイン・既定
            False で後方互換)。exclude_broken と同様に動画・行単位で必ず
            ログ出力する。両列が存在しない旧npzでは無効化される
            (何も除外しない)。

    Returns:
        (書き出し行数, 所要秒数)。
    """
    registry = _resolve_indicator_registry(
        profile, use_native=use_native, with_saturation_chain=with_saturation_chain,
    )
    t0 = time.time()
    all_rows: list[dict] = []
    npz_files = sorted(npz_dir.glob("*.npz"))
    if exclude_broken:
        npz_files, excluded = _split_broken_videos(npz_files)
        if excluded:
            total_excluded_rows = sum(n for _name, n in excluded)
            detail = ", ".join(f"{name}({n}行)" for name, n in excluded)
            print(
                f"[exclude-broken] BROKEN_VIDEOS を隔離: {len(excluded)}本"
                f" (計 {total_excluded_rows} 行) — {detail}",
            )
    for i, p in enumerate(npz_files):
        rows = convert_one_npz(
            p, registry, exclude_match_end_locked=exclude_match_end_locked,
        )
        all_rows.extend(rows)
        print(f"[{i+1}/{len(npz_files)}] {p.name}: {len(rows)} rows "
              f"(累計 {len(all_rows)}, {time.time()-t0:.1f}s)")
    if not all_rows:
        print("[WARN] 変換対象行が0件でした")
        return 0, time.time() - t0
    fieldnames = _final_fieldnames(profile, with_saturation_chain=with_saturation_chain)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        # extrasaction="ignore": diff_ に完全置換した own 列 (b-2) が行dict
        # には残っているが CSV には出さない (diff計算に own が必要なため
        # 削除せず保持している、a-1/b-2 の設計判断)。
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)
    elapsed = time.time() - t0
    print(f"[done] {len(all_rows)} 行 -> {out_csv} ({elapsed:.1f}s, "
          f"{len(npz_files)}本, profile={profile})")
    return len(all_rows), elapsed


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--npz-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--profile", choices=VALID_PROFILES, default="light")
    ap.add_argument(
        "--no-native", dest="use_native", action="store_false", default=True,
        help=(
            "full profile の重い4列で Rust拡張 puyo_core への載せ替えを"
            "無効化し既存Python実装のみで計算する (2026-08-13追加、"
            "パリティ検証・デバッグ用)。"
        ),
    )
    ap.add_argument(
        "--no-exclude-broken", dest="exclude_broken", action="store_false",
        default=True,
        help=(
            "BROKEN_VIDEOS (c26/c30/c58/c69、score OCR破綻・A-2 2026-08-13"
            "追加) の隔離を無効化し全npzを変換する (デバッグ用、既定は隔離ON)。"
        ),
    )
    ap.add_argument(
        "--with-saturation-chain", dest="with_saturation_chain",
        action="store_true", default=False,
        help=(
            "opt-in指標 saturation_chain (C-3, full profile限定) を含める。"
            "既定OFF (2026-08-13実測: 1行8〜18秒と桁違いに重く148本フルは"
            "非現実的なため。採否判断用の少数動画サブセット測定向け、"
            "`OPTIONAL_HEAVY_INDICATOR_NAMES` 直上コメント参照)。"
        ),
    )
    ap.add_argument(
        "--exclude-match-end-locked", dest="exclude_match_end_locked",
        action="store_true", default=False,
        help=(
            "境界実装の仕上げ (2026-08-18)。match_end_locked==1 または"
            "post_match_lockdown_active==1 の行 (勝敗演出パネル〜次試合"
            "開始までのマーカー列、collect_boards_lean.py --enable-post-"
            "match-lockdown-latch 併用収集で記録) を学習データから除外する。"
            "既定は無効 (オプトイン、後方互換)。両列が存在しない旧npzでは"
            "無効化される (何も除外しない)。"
        ),
    )
    a = ap.parse_args()
    convert_dir(
        a.npz_dir, a.out, profile=a.profile, use_native=a.use_native,
        exclude_broken=a.exclude_broken,
        with_saturation_chain=a.with_saturation_chain,
        exclude_match_end_locked=a.exclude_match_end_locked,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
