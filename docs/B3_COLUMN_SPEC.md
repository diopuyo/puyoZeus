# B3 列仕様台帳 (b3_rows_v1)

自動生成: `python -m scripts.b3_render_column_spec` (手編集しない。tests/test_b3_columns.py が一致を検査)。

総列数: 459。非飽和正規化 `nonsat(x,a)=z/(a+z), z=log1p(x)`, a 既定 3.0 (学習foldで決定、生値は別列)。

## 欠測理由コード (mk_* 列)

- 0: OK
- 1: LEGACY_SCHEMA_ABSENT
- 2: UNOBSERVABLE
- 3: NOT_YET_AVAILABLE
- 4: BOUNDARY_UNKNOWN
- 5: ONE_SIDE_UNOBSERVED
- 6: UNPREDICTABLE
- 7: SEARCH_TRUNCATED
- 8: NOT_APPLICABLE
- 9: DEFERRED
- 10: RECOGNITION_CONFLICT
- 11: HISTORY_INCOMPLETE

認識矛盾(10): 負の消費断片・材料超過は欠測。5色盤面は色依存指標・探索・時間軸を欠測とし、個数・形状は保持する。
得点の減少観測は直前の採用値を保持しmask=10、台帳の増分基準も下げない。
履歴不完全(11): 開始得点から履歴欠落が判明した試合は、両側のsl_pending/正規化値・sl_board_excessを欠測としmk_sl=11。時間軸の台帳予告と派生余力も欠測。観測開始後の累計は診断用に保持し、試合リセットまで解除しない。初回得点採用待ちはmk_sl=3。
ta_bound=1のQは残り色ぷよ＋追加ツモの算術上限。ビーム探索値（発火前1発火値を含む）は上限に使わない。連鎖中の段番号不確実時は最大係数、起点欠落時は全78セルを使う。
消費割合は式用/盤面用の分母を分離し、クリップしない。pf_before_colorは元の起点個数を保持。
レートはscoring.compute_effective_rateへ統一し、毎回切り捨て（70→52→39→29→21→15→11→8→6→4→3→2→1）。

## 側交換 (kind) の規則

- side_abs: p1/p2 の列を入れ替える (相手列=counterpart)
- diff_raw: 符号反転
- diff_centered: 1-x
- label: won_p1 -> 1-won_p1
- meta/common/weight: 不変 (位相・共通時刻・重み)

## 予告の扱い (統括決定 2026-10-05)

- 主入力 = 得点ベース台帳v2 (`sl_*`, src/b3_score_ledger.py の出力。未完成の間は全行 mask)。画面アイコン (`icon_*`) は台帳v2第4版では一切使われない参考列 (role=reference_only)。旧台帳 (lean の ojama_forecast 等) は特徴にも教師にも使わず `r_ledger_*` (role=R_only) に比較基準Rの再現専用で残す。

## 列一覧

| 列 | kind | dtype | 単位 | 正規化 | mask列 | available_at | 相手列 | role | 説明 |
|---|---|---|---|---|---|---|---|---|---|
| row_key | meta | str | - | none |  |  |  | meta | video|game|frame|orientation で一意 |
| video_id | meta | str | - | none |  |  |  | meta | 動画ID |
| game_idx | meta | i4 | - | none |  |  |  | meta | 試合番号 (B2 記録) |
| frame_idx | meta | i4 | - | none |  |  |  | meta | 行を生成した認識フレーム番号 |
| t_sec | meta | f8 | 秒 | none |  |  |  | meta | 評価時点 (行は available_at<=t_sec の情報のみ使う) |
| orientation | meta | i1 | - | none |  |  |  | meta | 0=原向き 1=1P/2P交換複製 |
| trigger_mask | meta | i4 | bit | none |  |  |  | meta | 更新契機ビット (b3_rows.TRIGGER_BITS) |
| held_sec | meta | f8 | 秒 | none |  |  |  | label | 次の行までの保持秒 (行重みの根拠) |
| weight | weight | f8 | 1/試合 | none |  |  |  | label | 行重み。試合内合計1(原向き0.5+複製0.5)。未来の行に依存するため特徴ではない |
| won_p1 | label | f4 | 0/1 | none |  |  |  | label | 教師: 1Pが最終勝者なら1 (lean.npz won 由来。特徴生成には渡さない) |
| av_max | meta | f8 | 秒 | none |  |  |  | audit | この行で使った全情報の available_at 最大 (<=t_sec が不変条件) |
| av_frame | meta | f8 | 秒 | none |  |  |  | audit | フレーム観測の available_at |
| lean_row_p1 | side_abs | i4 | index | none |  |  | lean_row_p2 | meta | lean.npz の対応する直前の盤面行 (-1=無し) |
| lean_row_p2 | side_abs | i4 | index | none |  |  | lean_row_p1 | meta | lean.npz の対応する直前の盤面行 (-1=無し) |
| av_board_p1 | side_abs | f8 | 秒 | none |  |  | av_board_p2 | audit | av_board の available_at (NaN=未使用) |
| av_board_p2 | side_abs | f8 | 秒 | none |  |  | av_board_p1 | audit | av_board の available_at (NaN=未使用) |
| av_queue_p1 | side_abs | f8 | 秒 | none |  |  | av_queue_p2 | audit | av_queue の available_at (NaN=未使用) |
| av_queue_p2 | side_abs | f8 | 秒 | none |  |  | av_queue_p1 | audit | av_queue の available_at (NaN=未使用) |
| av_fire_p1 | side_abs | f8 | 秒 | none |  |  | av_fire_p2 | audit | av_fire の available_at (NaN=未使用) |
| av_fire_p2 | side_abs | f8 | 秒 | none |  |  | av_fire_p1 | audit | av_fire の available_at (NaN=未使用) |
| av_formula_p1 | side_abs | f8 | 秒 | none |  |  | av_formula_p2 | audit | av_formula の available_at (NaN=未使用) |
| av_formula_p2 | side_abs | f8 | 秒 | none |  |  | av_formula_p1 | audit | av_formula の available_at (NaN=未使用) |
| av_icon_p1 | side_abs | f8 | 秒 | none |  |  | av_icon_p2 | audit | av_icon の available_at (NaN=未使用) |
| av_icon_p2 | side_abs | f8 | 秒 | none |  |  | av_icon_p1 | audit | av_icon の available_at (NaN=未使用) |
| phase | common | i1 | 0/1/2 | none | mk_phase | board |  | feature | user定義局面: 各側の色1-5+おじゃま9の総量(row0込み,UNKNOWN除外)の両側max。<=18序盤/19-47中盤/>=48終盤 |
| phase_total_max | common | f4 | 個 | none | mk_phase | board |  | feature | 位相の元になる両側 max 総量 |
| mk_phase | common | i1 | code | none |  |  |  | audit | 位相の欠測理由 (片側盤面無=5, UNKNOWN含む=2) |
| elapsed_first_move | common | f8 | 秒 | none |  | frame |  | feature | 両側のうち早い最初の1手からの経過秒 (G20) |
| rate_now | common | f4 | 点/個 | none |  | frame |  | feature | 有効マージンレート (from_first_move 起点) |
| mode_exchange | common | i1 | 0/1 | none |  | frame |  | feature | 撃ち合い(1)/静止(0)。連鎖中・OJAMA_FALL・着弾直後があれば1 (勝敗を見ない) |
| mode_bits | common | i1 | bit | none |  | frame |  | feature | 1=連鎖中 2=OJAMA_FALL 4=着弾直後 8=予告あり（B3_LEDGER_MODE_DEFAULTで台帳へ切替、既定は補正アイコン） |
| n_known_colors | common | i1 | 個 | none |  | board |  | feature | 試合内で因果的に観測済みの色数 (<4 のとき探索は5色) |
| cells_color_p1 | side_abs | f4 | 個 | none | mk_board_p1 | av_board_p1 | cells_color_p2 | feature | 色ぷよ(1-5)数 row0込み |
| cells_color_p2 | side_abs | f4 | 個 | none | mk_board_p2 | av_board_p2 | cells_color_p1 | feature | 色ぷよ(1-5)数 row0込み |
| cells_ojama_p1 | side_abs | f4 | 個 | none | mk_board_p1 | av_board_p1 | cells_ojama_p2 | feature | 盤面おじゃま(9)数 |
| cells_ojama_p2 | side_abs | f4 | 個 | none | mk_board_p2 | av_board_p2 | cells_ojama_p1 | feature | 盤面おじゃま(9)数 |
| cells_unknown_p1 | side_abs | f4 | 個 | none | mk_board_p1 | av_board_p1 | cells_unknown_p2 | feature | UNKNOWN(10)数 (総量に数えない) |
| cells_unknown_p2 | side_abs | f4 | 個 | none | mk_board_p2 | av_board_p2 | cells_unknown_p1 | feature | UNKNOWN(10)数 (総量に数えない) |
| cells_total_p1 | side_abs | f4 | 個 | none | mk_board_p1 | av_board_p1 | cells_total_p2 | feature | 色+おじゃま総量 (位相の元) |
| cells_total_p2 | side_abs | f4 | 個 | none | mk_board_p2 | av_board_p2 | cells_total_p1 | feature | 色+おじゃま総量 (位相の元) |
| max_height_p1 | side_abs | f4 | 段 | none | mk_board_p1 | av_board_p1 | max_height_p2 | feature | 最大列高さ |
| max_height_p2 | side_abs | f4 | 段 | none | mk_board_p2 | av_board_p2 | max_height_p1 | feature | 最大列高さ |
| board_age_p1 | side_abs | f4 | 秒 | none | mk_board_p1 | av_board_p1 | board_age_p2 | feature | 確定盤面の年齢 |
| board_age_p2 | side_abs | f4 | 秒 | none | mk_board_p2 | av_board_p2 | board_age_p1 | feature | 確定盤面の年齢 |
| is_dead_p1 | side_abs | i1 | 0/1 | none | mk_board_p1 | av_board_p1 | is_dead_p2 | feature | DEATH_ROW=1 の窒息盤面。火力を0にせず mask/flag で渡す |
| is_dead_p2 | side_abs | i1 | 0/1 | none | mk_board_p2 | av_board_p2 | is_dead_p1 | feature | DEATH_ROW=1 の窒息盤面。火力を0にせず mask/flag で渡す |
| mk_board_p1 | side_abs | i1 | code | none |  |  | mk_board_p2 | audit | board の欠測理由 (Mask) |
| mk_board_p2 | side_abs | i1 | code | none |  |  | mk_board_p1 | audit | board の欠測理由 (Mask) |
| ind_board_color_puyo_total_p1 | side_abs | f4 | 0-1 | 0-1(既存指標) | mk_ind_p1 | av_board_p1 | ind_board_color_puyo_total_p2 | feature | 既存静止指標 board_color_puyo_total (R1s の D 列の元) |
| ind_board_color_puyo_total_p2 | side_abs | f4 | 0-1 | 0-1(既存指標) | mk_ind_p2 | av_board_p2 | ind_board_color_puyo_total_p1 | feature | 既存静止指標 board_color_puyo_total (R1s の D 列の元) |
| ind_board_puyo_total_p1 | side_abs | f4 | 0-1 | 0-1(既存指標) | mk_ind_p1 | av_board_p1 | ind_board_puyo_total_p2 | feature | 既存静止指標 board_puyo_total (R1s の D 列の元) |
| ind_board_puyo_total_p2 | side_abs | f4 | 0-1 | 0-1(既存指標) | mk_ind_p2 | av_board_p2 | ind_board_puyo_total_p1 | feature | 既存静止指標 board_puyo_total (R1s の D 列の元) |
| ind_max_column_height_p1 | side_abs | f4 | 0-1 | 0-1(既存指標) | mk_ind_p1 | av_board_p1 | ind_max_column_height_p2 | feature | 既存静止指標 max_column_height (R1s の D 列の元) |
| ind_max_column_height_p2 | side_abs | f4 | 0-1 | 0-1(既存指標) | mk_ind_p2 | av_board_p2 | ind_max_column_height_p1 | feature | 既存静止指標 max_column_height (R1s の D 列の元) |
| ind_column_bumpiness_p1 | side_abs | f4 | 0-1 | 0-1(既存指標) | mk_ind_p1 | av_board_p1 | ind_column_bumpiness_p2 | feature | 既存静止指標 column_bumpiness (R1s の D 列の元) |
| ind_column_bumpiness_p2 | side_abs | f4 | 0-1 | 0-1(既存指標) | mk_ind_p2 | av_board_p2 | ind_column_bumpiness_p1 | feature | 既存静止指標 column_bumpiness (R1s の D 列の元) |
| ind_death_margin_p1 | side_abs | f4 | 0-1 | 0-1(既存指標) | mk_ind_p1 | av_board_p1 | ind_death_margin_p2 | feature | 既存静止指標 death_margin (R1s の D 列の元) |
| ind_death_margin_p2 | side_abs | f4 | 0-1 | 0-1(既存指標) | mk_ind_p2 | av_board_p2 | ind_death_margin_p1 | feature | 既存静止指標 death_margin (R1s の D 列の元) |
| ind_death_margin_neighbor_p1 | side_abs | f4 | 0-1 | 0-1(既存指標) | mk_ind_p1 | av_board_p1 | ind_death_margin_neighbor_p2 | feature | 既存静止指標 death_margin_neighbor (R1s の D 列の元) |
| ind_death_margin_neighbor_p2 | side_abs | f4 | 0-1 | 0-1(既存指標) | mk_ind_p2 | av_board_p2 | ind_death_margin_neighbor_p1 | feature | 既存静止指標 death_margin_neighbor (R1s の D 列の元) |
| ind_center_bulge_color_p1 | side_abs | f4 | 0-1 | 0-1(既存指標) | mk_ind_p1 | av_board_p1 | ind_center_bulge_color_p2 | feature | 既存静止指標 center_bulge_color (R1s の D 列の元) |
| ind_center_bulge_color_p2 | side_abs | f4 | 0-1 | 0-1(既存指標) | mk_ind_p2 | av_board_p2 | ind_center_bulge_color_p1 | feature | 既存静止指標 center_bulge_color (R1s の D 列の元) |
| ind_center_bulge_ojama_p1 | side_abs | f4 | 0-1 | 0-1(既存指標) | mk_ind_p1 | av_board_p1 | ind_center_bulge_ojama_p2 | feature | 既存静止指標 center_bulge_ojama (R1s の D 列の元) |
| ind_center_bulge_ojama_p2 | side_abs | f4 | 0-1 | 0-1(既存指標) | mk_ind_p2 | av_board_p2 | ind_center_bulge_ojama_p1 | feature | 既存静止指標 center_bulge_ojama (R1s の D 列の元) |
| ind_board_ojama_count_p1 | side_abs | f4 | 0-1 | 0-1(既存指標) | mk_ind_p1 | av_board_p1 | ind_board_ojama_count_p2 | feature | 既存静止指標 board_ojama_count (R1s の D 列の元) |
| ind_board_ojama_count_p2 | side_abs | f4 | 0-1 | 0-1(既存指標) | mk_ind_p2 | av_board_p2 | ind_board_ojama_count_p1 | feature | 既存静止指標 board_ojama_count (R1s の D 列の元) |
| ind_color_diversity_evenness_p1 | side_abs | f4 | 0-1 | 0-1(既存指標) | mk_ind_p1 | av_board_p1 | ind_color_diversity_evenness_p2 | feature | 既存静止指標 color_diversity_evenness (R1s の D 列の元) |
| ind_color_diversity_evenness_p2 | side_abs | f4 | 0-1 | 0-1(既存指標) | mk_ind_p2 | av_board_p2 | ind_color_diversity_evenness_p1 | feature | 既存静止指標 color_diversity_evenness (R1s の D 列の元) |
| ind_buried_hole_count_p1 | side_abs | f4 | 0-1 | 0-1(既存指標) | mk_ind_p1 | av_board_p1 | ind_buried_hole_count_p2 | feature | 既存静止指標 buried_hole_count (R1s の D 列の元) |
| ind_buried_hole_count_p2 | side_abs | f4 | 0-1 | 0-1(既存指標) | mk_ind_p2 | av_board_p2 | ind_buried_hole_count_p1 | feature | 既存静止指標 buried_hole_count (R1s の D 列の元) |
| ind_current_max_chain_p1 | side_abs | f4 | 0-1 | 0-1(既存指標) | mk_ind_p1 | av_board_p1 | ind_current_max_chain_p2 | feature | 既存静止指標 current_max_chain (R1s の D 列の元) |
| ind_current_max_chain_p2 | side_abs | f4 | 0-1 | 0-1(既存指標) | mk_ind_p2 | av_board_p2 | ind_current_max_chain_p1 | feature | 既存静止指標 current_max_chain (R1s の D 列の元) |
| ind_dig_resistance_p1 | side_abs | f4 | 0-1 | 0-1(既存指標) | mk_ind_p1 | av_board_p1 | ind_dig_resistance_p2 | feature | 既存静止指標 dig_resistance (R1s の D 列の元) |
| ind_dig_resistance_p2 | side_abs | f4 | 0-1 | 0-1(既存指標) | mk_ind_p2 | av_board_p2 | ind_dig_resistance_p1 | feature | 既存静止指標 dig_resistance (R1s の D 列の元) |
| ind_ukeyasusa_p1 | side_abs | f4 | 0-1 | 0-1(既存指標) | mk_ind_p1 | av_board_p1 | ind_ukeyasusa_p2 | feature | 既存静止指標 ukeyasusa (R1s の D 列の元) |
| ind_ukeyasusa_p2 | side_abs | f4 | 0-1 | 0-1(既存指標) | mk_ind_p2 | av_board_p2 | ind_ukeyasusa_p1 | feature | 既存静止指標 ukeyasusa (R1s の D 列の元) |
| ind_sub_chain_count_p1 | side_abs | f4 | 0-1 | 0-1(既存指標) | mk_ind_p1 | av_board_p1 | ind_sub_chain_count_p2 | feature | 既存静止指標 sub_chain_count (R1s の D 列の元) |
| ind_sub_chain_count_p2 | side_abs | f4 | 0-1 | 0-1(既存指標) | mk_ind_p2 | av_board_p2 | ind_sub_chain_count_p1 | feature | 既存静止指標 sub_chain_count (R1s の D 列の元) |
| ind_immediate_fire_power_p1 | side_abs | f4 | 0-1 | 0-1(既存指標) | mk_ind_p1 | av_board_p1 | ind_immediate_fire_power_p2 | feature | 既存静止指標 immediate_fire_power (R1s の D 列の元) |
| ind_immediate_fire_power_p2 | side_abs | f4 | 0-1 | 0-1(既存指標) | mk_ind_p2 | av_board_p2 | ind_immediate_fire_power_p1 | feature | 既存静止指標 immediate_fire_power (R1s の D 列の元) |
| ind_chain_efficiency_p1 | side_abs | f4 | 0-1 | 0-1(既存指標) | mk_ind_p1 | av_board_p1 | ind_chain_efficiency_p2 | feature | 既存静止指標 chain_efficiency (R1s の D 列の元) |
| ind_chain_efficiency_p2 | side_abs | f4 | 0-1 | 0-1(既存指標) | mk_ind_p2 | av_board_p2 | ind_chain_efficiency_p1 | feature | 既存静止指標 chain_efficiency (R1s の D 列の元) |
| ind_min_puyos_to_ignite_p1 | side_abs | f4 | 0-1 | 0-1(既存指標) | mk_ind_p1 | av_board_p1 | ind_min_puyos_to_ignite_p2 | feature | 既存静止指標 min_puyos_to_ignite (R1s の D 列の元) |
| ind_min_puyos_to_ignite_p2 | side_abs | f4 | 0-1 | 0-1(既存指標) | mk_ind_p2 | av_board_p2 | ind_min_puyos_to_ignite_p1 | feature | 既存静止指標 min_puyos_to_ignite (R1s の D 列の元) |
| ind_second_chain_potential_p1 | side_abs | f4 | 0-1 | 0-1(既存指標) | mk_ind_p1 | av_board_p1 | ind_second_chain_potential_p2 | feature | 既存静止指標 second_chain_potential (R1s の D 列の元) |
| ind_second_chain_potential_p2 | side_abs | f4 | 0-1 | 0-1(既存指標) | mk_ind_p2 | av_board_p2 | ind_second_chain_potential_p1 | feature | 既存静止指標 second_chain_potential (R1s の D 列の元) |
| ind_main_linked_pair_count_p1 | side_abs | f4 | 0-1 | 0-1(既存指標) | mk_ind_p1 | av_board_p1 | ind_main_linked_pair_count_p2 | feature | 既存静止指標 main_linked_pair_count (R1s の D 列の元) |
| ind_main_linked_pair_count_p2 | side_abs | f4 | 0-1 | 0-1(既存指標) | mk_ind_p2 | av_board_p2 | ind_main_linked_pair_count_p1 | feature | 既存静止指標 main_linked_pair_count (R1s の D 列の元) |
| ind_isolated_pair_count_p1 | side_abs | f4 | 0-1 | 0-1(既存指標) | mk_ind_p1 | av_board_p1 | ind_isolated_pair_count_p2 | feature | 既存静止指標 isolated_pair_count (R1s の D 列の元) |
| ind_isolated_pair_count_p2 | side_abs | f4 | 0-1 | 0-1(既存指標) | mk_ind_p2 | av_board_p2 | ind_isolated_pair_count_p1 | feature | 既存静止指標 isolated_pair_count (R1s の D 列の元) |
| ind_main_linked_ratio_p1 | side_abs | f4 | 0-1 | 0-1(既存指標) | mk_ind_p1 | av_board_p1 | ind_main_linked_ratio_p2 | feature | 既存静止指標 main_linked_ratio (R1s の D 列の元) |
| ind_main_linked_ratio_p2 | side_abs | f4 | 0-1 | 0-1(既存指標) | mk_ind_p2 | av_board_p2 | ind_main_linked_ratio_p1 | feature | 既存静止指標 main_linked_ratio (R1s の D 列の元) |
| ind_ignition_point_count_p1 | side_abs | f4 | 0-1 | 0-1(既存指標) | mk_ind_p1 | av_board_p1 | ind_ignition_point_count_p2 | feature | 既存静止指標 ignition_point_count (R1s の D 列の元) |
| ind_ignition_point_count_p2 | side_abs | f4 | 0-1 | 0-1(既存指標) | mk_ind_p2 | av_board_p2 | ind_ignition_point_count_p1 | feature | 既存静止指標 ignition_point_count (R1s の D 列の元) |
| ind_multi_color_ignition_p1 | side_abs | f4 | 0-1 | 0-1(既存指標) | mk_ind_p1 | av_board_p1 | ind_multi_color_ignition_p2 | feature | 既存静止指標 multi_color_ignition (R1s の D 列の元) |
| ind_multi_color_ignition_p2 | side_abs | f4 | 0-1 | 0-1(既存指標) | mk_ind_p2 | av_board_p2 | ind_multi_color_ignition_p1 | feature | 既存静止指標 multi_color_ignition (R1s の D 列の元) |
| ind_simultaneous_pop_richness_p1 | side_abs | f4 | 0-1 | 0-1(既存指標) | mk_ind_p1 | av_board_p1 | ind_simultaneous_pop_richness_p2 | feature | 既存静止指標 simultaneous_pop_richness (R1s の D 列の元) |
| ind_simultaneous_pop_richness_p2 | side_abs | f4 | 0-1 | 0-1(既存指標) | mk_ind_p2 | av_board_p2 | ind_simultaneous_pop_richness_p1 | feature | 既存静止指標 simultaneous_pop_richness (R1s の D 列の元) |
| ind_saturation_chain_upper_p1 | side_abs | f4 | 0-1 | 0-1(既存指標) | mk_ind_p1 | av_board_p1 | ind_saturation_chain_upper_p2 | feature | 既存静止指標 saturation_chain_upper (R1s の D 列の元) |
| ind_saturation_chain_upper_p2 | side_abs | f4 | 0-1 | 0-1(既存指標) | mk_ind_p2 | av_board_p2 | ind_saturation_chain_upper_p1 | feature | 既存静止指標 saturation_chain_upper (R1s の D 列の元) |
| ind_chain_articulation_point_count_p1 | side_abs | f4 | 0-1 | 0-1(既存指標) | mk_ind_p1 | av_board_p1 | ind_chain_articulation_point_count_p2 | feature | 既存静止指標 chain_articulation_point_count (R1s の D 列の元) |
| ind_chain_articulation_point_count_p2 | side_abs | f4 | 0-1 | 0-1(既存指標) | mk_ind_p2 | av_board_p2 | ind_chain_articulation_point_count_p1 | feature | 既存静止指標 chain_articulation_point_count (R1s の D 列の元) |
| ind_conn_pair_count_p1 | side_abs | f4 | 個 | none(D parity; 生個数) | mk_ind_p1 | av_board_p1 | ind_conn_pair_count_p2 | feature | 既存静止指標 conn_pair_count (R1s の D 列の元) |
| ind_conn_pair_count_p2 | side_abs | f4 | 個 | none(D parity; 生個数) | mk_ind_p2 | av_board_p2 | ind_conn_pair_count_p1 | feature | 既存静止指標 conn_pair_count (R1s の D 列の元) |
| ind_conn_triple_count_p1 | side_abs | f4 | 個 | none(D parity; 生個数) | mk_ind_p1 | av_board_p1 | ind_conn_triple_count_p2 | feature | 既存静止指標 conn_triple_count (R1s の D 列の元) |
| ind_conn_triple_count_p2 | side_abs | f4 | 個 | none(D parity; 生個数) | mk_ind_p2 | av_board_p2 | ind_conn_triple_count_p1 | feature | 既存静止指標 conn_triple_count (R1s の D 列の元) |
| ind_conn_max_group_size_p1 | side_abs | f4 | 個 | none(D parity; 生個数) | mk_ind_p1 | av_board_p1 | ind_conn_max_group_size_p2 | feature | 既存静止指標 conn_max_group_size (R1s の D 列の元) |
| ind_conn_max_group_size_p2 | side_abs | f4 | 個 | none(D parity; 生個数) | mk_ind_p2 | av_board_p2 | ind_conn_max_group_size_p1 | feature | 既存静止指標 conn_max_group_size (R1s の D 列の元) |
| mk_ind_p1 | side_abs | i1 | code | none |  |  | mk_ind_p2 | audit | ind の欠測理由 (Mask) |
| mk_ind_p2 | side_abs | i1 | code | none |  |  | mk_ind_p1 | audit | ind の欠測理由 (Mask) |
| k1_ojama_p1 | side_abs | f8 | 個 | none | mk_k_p1 | av_board_p1 | k1_ojama_p2 | feature | B3専用探索: 理想ツモ総3組以内の最大発火量(おじゃま個数, 因果4色/NEXT既知) |
| k1_ojama_p2 | side_abs | f8 | 個 | none | mk_k_p2 | av_board_p2 | k1_ojama_p1 | feature | B3専用探索: 理想ツモ総3組以内の最大発火量(おじゃま個数, 因果4色/NEXT既知) |
| k1_nf_p1 | side_abs | f4 | 0-1 | z/(a+z) a=3.0 | mk_k_p1 | av_board_p1 | k1_nf_p2 | feature | k1 の非飽和正規化 |
| k1_nf_p2 | side_abs | f4 | 0-1 | z/(a+z) a=3.0 | mk_k_p2 | av_board_p2 | k1_nf_p1 | feature | k1 の非飽和正規化 |
| k2_ojama_p1 | side_abs | f8 | 個 | none | mk_k_p1 | av_board_p1 | k2_ojama_p2 | feature | B3専用探索: 理想ツモ総4組以内の最大発火量(おじゃま個数, 因果4色/NEXT既知) |
| k2_ojama_p2 | side_abs | f8 | 個 | none | mk_k_p2 | av_board_p2 | k2_ojama_p1 | feature | B3専用探索: 理想ツモ総4組以内の最大発火量(おじゃま個数, 因果4色/NEXT既知) |
| k2_nf_p1 | side_abs | f4 | 0-1 | z/(a+z) a=3.0 | mk_k_p1 | av_board_p1 | k2_nf_p2 | feature | k2 の非飽和正規化 |
| k2_nf_p2 | side_abs | f4 | 0-1 | z/(a+z) a=3.0 | mk_k_p2 | av_board_p2 | k2_nf_p1 | feature | k2 の非飽和正規化 |
| k3_ojama_p1 | side_abs | f8 | 個 | none | mk_k_p1 | av_board_p1 | k3_ojama_p2 | feature | B3専用探索: 理想ツモ総5組以内の最大発火量(おじゃま個数, 因果4色/NEXT既知) |
| k3_ojama_p2 | side_abs | f8 | 個 | none | mk_k_p2 | av_board_p2 | k3_ojama_p1 | feature | B3専用探索: 理想ツモ総5組以内の最大発火量(おじゃま個数, 因果4色/NEXT既知) |
| k3_nf_p1 | side_abs | f4 | 0-1 | z/(a+z) a=3.0 | mk_k_p1 | av_board_p1 | k3_nf_p2 | feature | k3 の非飽和正規化 |
| k3_nf_p2 | side_abs | f4 | 0-1 | z/(a+z) a=3.0 | mk_k_p2 | av_board_p2 | k3_nf_p1 | feature | k3 の非飽和正規化 |
| k4_ojama_p1 | side_abs | f8 | 個 | none | mk_k_p1 | av_board_p1 | k4_ojama_p2 | feature | B3専用探索: 理想ツモ総6組以内の最大発火量(おじゃま個数, 因果4色/NEXT既知) |
| k4_ojama_p2 | side_abs | f8 | 個 | none | mk_k_p2 | av_board_p2 | k4_ojama_p1 | feature | B3専用探索: 理想ツモ総6組以内の最大発火量(おじゃま個数, 因果4色/NEXT既知) |
| k4_nf_p1 | side_abs | f4 | 0-1 | z/(a+z) a=3.0 | mk_k_p1 | av_board_p1 | k4_nf_p2 | feature | k4 の非飽和正規化 |
| k4_nf_p2 | side_abs | f4 | 0-1 | z/(a+z) a=3.0 | mk_k_p2 | av_board_p2 | k4_nf_p1 | feature | k4 の非飽和正規化 |
| k5_ojama_p1 | side_abs | f8 | 個 | none | mk_k_p1 | av_board_p1 | k5_ojama_p2 | feature | B3専用探索: 理想ツモ総7組以内の最大発火量(おじゃま個数, 因果4色/NEXT既知) |
| k5_ojama_p2 | side_abs | f8 | 個 | none | mk_k_p2 | av_board_p2 | k5_ojama_p1 | feature | B3専用探索: 理想ツモ総7組以内の最大発火量(おじゃま個数, 因果4色/NEXT既知) |
| k5_nf_p1 | side_abs | f4 | 0-1 | z/(a+z) a=3.0 | mk_k_p1 | av_board_p1 | k5_nf_p2 | feature | k5 の非飽和正規化 |
| k5_nf_p2 | side_abs | f4 | 0-1 | z/(a+z) a=3.0 | mk_k_p2 | av_board_p2 | k5_nf_p1 | feature | k5 の非飽和正規化 |
| search_done_h_p1 | side_abs | i1 | 手 | none | mk_k_p1 | av_board_p1 | search_done_h_p2 | feature | 探索済み総手数 (打切り記録) |
| search_done_h_p2 | side_abs | i1 | 手 | none | mk_k_p2 | av_board_p2 | search_done_h_p1 | feature | 探索済み総手数 (打切り記録) |
| mk_k_p1 | side_abs | i1 | code | none |  |  | mk_k_p2 | audit | kfire の欠測理由 (Mask) |
| mk_k_p2 | side_abs | i1 | code | none |  |  | mk_k_p1 | audit | kfire の欠測理由 (Mask) |
| queue_age_p1 | side_abs | f4 | 秒 | none | mk_queue_p1 | av_queue_p1 | queue_age_p2 | feature | NEXT既知情報の年齢 |
| queue_age_p2 | side_abs | f4 | 秒 | none | mk_queue_p2 | av_queue_p2 | queue_age_p1 | feature | NEXT既知情報の年齢 |
| mk_queue_p1 | side_abs | i1 | code | none |  |  | mk_queue_p2 | audit | queue の欠測理由 (Mask) |
| mk_queue_p2 | side_abs | i1 | code | none |  |  | mk_queue_p1 | audit | queue の欠測理由 (Mask) |
| score_p1 | side_abs | f8 | 点 | none | mk_score_p1 | av_frame | score_p2 | feature | 画面得点 (非null時) |
| score_p2 | side_abs | f8 | 点 | none | mk_score_p2 | av_frame | score_p1 | feature | 画面得点 (非null時) |
| mk_score_p1 | side_abs | i1 | code | none |  |  | mk_score_p2 | audit | score の欠測理由 (Mask) |
| mk_score_p2 | side_abs | i1 | code | none |  |  | mk_score_p1 | audit | score の欠測理由 (Mask) |
| icon_raw_total_p1 | side_abs | f8 | 個 | none | mk_icon_p1 | av_icon_p1 | icon_raw_total_p2 | reference_only | [参考列] 画面アイコン読みの総数 (生値・最新フレーム) |
| icon_raw_total_p2 | side_abs | f8 | 個 | none | mk_icon_p2 | av_icon_p2 | icon_raw_total_p1 | reference_only | [参考列] 画面アイコン読みの総数 (生値・最新フレーム) |
| icon_corr_total_p1 | side_abs | f8 | 個 | none | mk_icon_p1 | av_icon_p1 | icon_corr_total_p2 | reference_only | [照合用] 補正済み: 試合中ゲート+時間窓中央値(過大誤読抑制)+大額の上限整合 |
| icon_corr_total_p2 | side_abs | f8 | 個 | none | mk_icon_p2 | av_icon_p2 | icon_corr_total_p1 | reference_only | [照合用] 補正済み: 試合中ゲート+時間窓中央値(過大誤読抑制)+大額の上限整合 |
| icon_raw_small_p1 | side_abs | f4 | 個 | none | mk_icon_p1 | av_icon_p1 | icon_raw_small_p2 | reference_only | small アイコン個数 (生値) |
| icon_raw_small_p2 | side_abs | f4 | 個 | none | mk_icon_p2 | av_icon_p2 | icon_raw_small_p1 | reference_only | small アイコン個数 (生値) |
| icon_corr_small_p1 | side_abs | f4 | 個 | none | mk_icon_p1 | av_icon_p1 | icon_corr_small_p2 | reference_only | small アイコン個数 (補正値) |
| icon_corr_small_p2 | side_abs | f4 | 個 | none | mk_icon_p2 | av_icon_p2 | icon_corr_small_p1 | reference_only | small アイコン個数 (補正値) |
| icon_raw_large_p1 | side_abs | f4 | 個 | none | mk_icon_p1 | av_icon_p1 | icon_raw_large_p2 | reference_only | large アイコン個数 (生値) |
| icon_raw_large_p2 | side_abs | f4 | 個 | none | mk_icon_p2 | av_icon_p2 | icon_raw_large_p1 | reference_only | large アイコン個数 (生値) |
| icon_corr_large_p1 | side_abs | f4 | 個 | none | mk_icon_p1 | av_icon_p1 | icon_corr_large_p2 | reference_only | large アイコン個数 (補正値) |
| icon_corr_large_p2 | side_abs | f4 | 個 | none | mk_icon_p2 | av_icon_p2 | icon_corr_large_p1 | reference_only | large アイコン個数 (補正値) |
| icon_raw_rock_p1 | side_abs | f4 | 個 | none | mk_icon_p1 | av_icon_p1 | icon_raw_rock_p2 | reference_only | rock アイコン個数 (生値) |
| icon_raw_rock_p2 | side_abs | f4 | 個 | none | mk_icon_p2 | av_icon_p2 | icon_raw_rock_p1 | reference_only | rock アイコン個数 (生値) |
| icon_corr_rock_p1 | side_abs | f4 | 個 | none | mk_icon_p1 | av_icon_p1 | icon_corr_rock_p2 | reference_only | rock アイコン個数 (補正値) |
| icon_corr_rock_p2 | side_abs | f4 | 個 | none | mk_icon_p2 | av_icon_p2 | icon_corr_rock_p1 | reference_only | rock アイコン個数 (補正値) |
| icon_raw_star_p1 | side_abs | f4 | 個 | none | mk_icon_p1 | av_icon_p1 | icon_raw_star_p2 | reference_only | star アイコン個数 (生値) |
| icon_raw_star_p2 | side_abs | f4 | 個 | none | mk_icon_p2 | av_icon_p2 | icon_raw_star_p1 | reference_only | star アイコン個数 (生値) |
| icon_corr_star_p1 | side_abs | f4 | 個 | none | mk_icon_p1 | av_icon_p1 | icon_corr_star_p2 | reference_only | star アイコン個数 (補正値) |
| icon_corr_star_p2 | side_abs | f4 | 個 | none | mk_icon_p2 | av_icon_p2 | icon_corr_star_p1 | reference_only | star アイコン個数 (補正値) |
| icon_raw_moon_p1 | side_abs | f4 | 個 | none | mk_icon_p1 | av_icon_p1 | icon_raw_moon_p2 | reference_only | moon アイコン個数 (生値) |
| icon_raw_moon_p2 | side_abs | f4 | 個 | none | mk_icon_p2 | av_icon_p2 | icon_raw_moon_p1 | reference_only | moon アイコン個数 (生値) |
| icon_corr_moon_p1 | side_abs | f4 | 個 | none | mk_icon_p1 | av_icon_p1 | icon_corr_moon_p2 | reference_only | moon アイコン個数 (補正値) |
| icon_corr_moon_p2 | side_abs | f4 | 個 | none | mk_icon_p2 | av_icon_p2 | icon_corr_moon_p1 | reference_only | moon アイコン個数 (補正値) |
| icon_raw_crown_p1 | side_abs | f4 | 個 | none | mk_icon_p1 | av_icon_p1 | icon_raw_crown_p2 | reference_only | crown アイコン個数 (生値) |
| icon_raw_crown_p2 | side_abs | f4 | 個 | none | mk_icon_p2 | av_icon_p2 | icon_raw_crown_p1 | reference_only | crown アイコン個数 (生値) |
| icon_corr_crown_p1 | side_abs | f4 | 個 | none | mk_icon_p1 | av_icon_p1 | icon_corr_crown_p2 | reference_only | crown アイコン個数 (補正値) |
| icon_corr_crown_p2 | side_abs | f4 | 個 | none | mk_icon_p2 | av_icon_p2 | icon_corr_crown_p1 | reference_only | crown アイコン個数 (補正値) |
| icon_corr_nf_p1 | side_abs | f4 | 0-1 | z/(a+z) a=3.0 | mk_icon_p1 | av_icon_p1 | icon_corr_nf_p2 | reference_only | 補正済み総数の非飽和正規化 |
| icon_corr_nf_p2 | side_abs | f4 | 0-1 | z/(a+z) a=3.0 | mk_icon_p2 | av_icon_p2 | icon_corr_nf_p1 | reference_only | 補正済み総数の非飽和正規化 |
| icon_big_clipped_p1 | side_abs | i1 | 0/1 | none | mk_icon_p1 | av_icon_p1 | icon_big_clipped_p2 | reference_only | 大額(星/月/王冠)を相手得点由来の上限で切った |
| icon_big_clipped_p2 | side_abs | i1 | 0/1 | none | mk_icon_p2 | av_icon_p2 | icon_big_clipped_p1 | reference_only | 大額(星/月/王冠)を相手得点由来の上限で切った |
| icon_ub_p1 | side_abs | f8 | 個 | none | mk_icon_p1 | av_icon_p1 | icon_ub_p2 | reference_only | 相手得点由来の予告上限 (整合チェック用) |
| icon_ub_p2 | side_abs | f8 | 個 | none | mk_icon_p2 | av_icon_p2 | icon_ub_p1 | reference_only | 相手得点由来の予告上限 (整合チェック用) |
| icon_omitted_p1 | side_abs | i1 | 0/1 | none | mk_icon_p1 | av_icon_p1 | icon_omitted_p2 | reference_only | [照合用] 表示が省略されている疑い (アイコン数が表示上限6に達している) |
| icon_omitted_p2 | side_abs | i1 | 0/1 | none | mk_icon_p2 | av_icon_p2 | icon_omitted_p1 | reference_only | [照合用] 表示が省略されている疑い (アイコン数が表示上限6に達している) |
| mk_icon_p1 | side_abs | i1 | code | none |  |  | mk_icon_p2 | audit | icon の欠測理由 (Mask) |
| mk_icon_p2 | side_abs | i1 | code | none |  |  | mk_icon_p1 | audit | icon の欠測理由 (Mask) |
| sl_pending_p1 | side_abs | f8 | 個 | none | mk_sl_p1 | av_ledger_p1 | sl_pending_p2 | feature | [主入力] 得点ベース台帳v2: 予告おじゃま (未決済, 相殺後) |
| sl_pending_p2 | side_abs | f8 | 個 | none | mk_sl_p2 | av_ledger_p2 | sl_pending_p1 | feature | [主入力] 得点ベース台帳v2: 予告おじゃま (未決済, 相殺後) |
| sl_generated_p1 | side_abs | f8 | 個 | none | mk_sl_p1 | av_ledger_p1 | sl_generated_p2 | feature | [主入力] 得点ベース台帳v2: 累計生成量 (その側が送った) |
| sl_generated_p2 | side_abs | f8 | 個 | none | mk_sl_p2 | av_ledger_p2 | sl_generated_p1 | feature | [主入力] 得点ベース台帳v2: 累計生成量 (その側が送った) |
| sl_offset_p1 | side_abs | f8 | 個 | none | mk_sl_p1 | av_ledger_p1 | sl_offset_p2 | feature | [主入力] 得点ベース台帳v2: 累計相殺量 |
| sl_offset_p2 | side_abs | f8 | 個 | none | mk_sl_p2 | av_ledger_p2 | sl_offset_p1 | feature | [主入力] 得点ベース台帳v2: 累計相殺量 |
| sl_dropped_p1 | side_abs | f8 | 個 | none | mk_sl_p1 | av_ledger_p1 | sl_dropped_p2 | feature | [主入力] 得点ベース台帳v2: 累計着弾量 (その側へ落ちた) |
| sl_dropped_p2 | side_abs | f8 | 個 | none | mk_sl_p2 | av_ledger_p2 | sl_dropped_p1 | feature | [主入力] 得点ベース台帳v2: 累計着弾量 (その側へ落ちた) |
| sl_carry_p1 | side_abs | f8 | 点 | none | mk_sl_p1 | av_ledger_p1 | sl_carry_p2 | feature | [主入力] 得点ベース台帳v2: 得点の70点刻み端数繰越 |
| sl_carry_p2 | side_abs | f8 | 点 | none | mk_sl_p2 | av_ledger_p2 | sl_carry_p1 | feature | [主入力] 得点ベース台帳v2: 得点の70点刻み端数繰越 |
| sl_observed_landing_p1 | side_abs | f8 | 個 | none | mk_sl_p1 | av_ledger_p1 | sl_observed_landing_p2 | feature | [主入力] 得点ベース台帳v2: 累計 観測着弾 (受け側確定盤面のおじゃま増分のうち予告を差し引いた分。盤面確定まで1-3秒遅れて反映) |
| sl_observed_landing_p2 | side_abs | f8 | 個 | none | mk_sl_p2 | av_ledger_p2 | sl_observed_landing_p1 | feature | [主入力] 得点ベース台帳v2: 累計 観測着弾 (受け側確定盤面のおじゃま増分のうち予告を差し引いた分。盤面確定まで1-3秒遅れて反映) |
| sl_board_increase_p1 | side_abs | f8 | 個 | none | mk_sl_p1 | av_ledger_p1 | sl_board_increase_p2 | feature | [主入力] 得点ベース台帳v2: 累計 盤面おじゃま増分 (観測、予告で説明できない分も含む) |
| sl_board_increase_p2 | side_abs | f8 | 個 | none | mk_sl_p2 | av_ledger_p2 | sl_board_increase_p1 | feature | [主入力] 得点ベース台帳v2: 累計 盤面おじゃま増分 (観測、予告で説明できない分も含む) |
| sl_board_excess_p1 | side_abs | f8 | 個 | none | mk_sl_p1 | av_ledger_p1 | sl_board_excess_p2 | feature | [主入力] 得点ベース台帳v2: 累計 予告を超えた盤面増分 (予告に無い着弾=台帳の取りこぼし/頭打ちの手掛かり) |
| sl_board_excess_p2 | side_abs | f8 | 個 | none | mk_sl_p2 | av_ledger_p2 | sl_board_excess_p1 | feature | [主入力] 得点ベース台帳v2: 累計 予告を超えた盤面増分 (予告に無い着弾=台帳の取りこぼし/頭打ちの手掛かり) |
| sl_pending_nf_p1 | side_abs | f4 | 0-1 | z/(a+z) a=3.0 | mk_sl_p1 | av_ledger_p1 | sl_pending_nf_p2 | feature | sl_pending の非飽和正規化 |
| sl_pending_nf_p2 | side_abs | f4 | 0-1 | z/(a+z) a=3.0 | mk_sl_p2 | av_ledger_p2 | sl_pending_nf_p1 | feature | sl_pending の非飽和正規化 |
| sl_rate | common | f4 | 点/個 | none | mk_sl_c | av_ledger |  | feature | [主入力] 台帳v2の換算レート (マージンタイム込み) |
| mk_sl_c | common | i1 | code | none |  |  |  | audit | sl_rate の欠測理由 |
| av_ledger | common | f8 | 秒 | none |  |  |  | audit | 台帳v2 snapshot の available_at |
| mk_sl_p1 | side_abs | i1 | code | none |  |  | mk_sl_p2 | audit | score_ledger の欠測理由 (Mask) |
| mk_sl_p2 | side_abs | i1 | code | none |  |  | mk_sl_p1 | audit | score_ledger の欠測理由 (Mask) |
| land_icon_drop_p1 | side_abs | f8 | 個 | none | mk_land_p1 | av_frame | land_icon_drop_p2 | feature | 直近着弾: 既定は同一OJAMA_FALL区間の補正アイコン減少。B3_LEDGER_LANDING_DEFAULT時は台帳の観測着弾控除量（名前は互換維持） |
| land_icon_drop_p2 | side_abs | f8 | 個 | none | mk_land_p2 | av_frame | land_icon_drop_p1 | feature | 直近着弾: 既定は同一OJAMA_FALL区間の補正アイコン減少。B3_LEDGER_LANDING_DEFAULT時は台帳の観測着弾控除量（名前は互換維持） |
| land_board_net_p1 | side_abs | f8 | 個 | none | mk_land_p1 | av_frame | land_board_net_p2 | feature | 直近着弾: 盤面おじゃま純増(空きで頭打ちの下限値) |
| land_board_net_p2 | side_abs | f8 | 個 | none | mk_land_p2 | av_frame | land_board_net_p1 | feature | 直近着弾: 盤面おじゃま純増(空きで頭打ちの下限値) |
| land_age_p1 | side_abs | f8 | 秒 | none | mk_land_p1 | av_frame | land_age_p2 | feature | 直近着弾からの経過 |
| land_age_p2 | side_abs | f8 | 秒 | none | mk_land_p2 | av_frame | land_age_p1 | feature | 直近着弾からの経過 |
| land_count_p1 | side_abs | i4 | 回 | none | mk_land_p1 | av_frame | land_count_p2 | feature | この試合の着弾区間数 |
| land_count_p2 | side_abs | i4 | 回 | none | mk_land_p2 | av_frame | land_count_p1 | feature | この試合の着弾区間数 |
| mk_land_p1 | side_abs | i1 | code | none |  |  | mk_land_p2 | audit | land の欠測理由 (Mask) |
| mk_land_p2 | side_abs | i1 | code | none |  |  | mk_land_p1 | audit | land の欠測理由 (Mask) |
| r_ledger_pending_p1 | side_abs | f8 | 個 | none |  | av_frame | r_ledger_pending_p2 | R_only | 台帳の予告(未決済) [R再現専用] |
| r_ledger_pending_p2 | side_abs | f8 | 個 | none |  | av_frame | r_ledger_pending_p1 | R_only | 台帳の予告(未決済) [R再現専用] |
| r_ledger_forecast_p1 | side_abs | f8 | 個 | none |  | av_frame | r_ledger_forecast_p2 | R_only | 台帳の予告 forecast [R再現専用] |
| r_ledger_forecast_p2 | side_abs | f8 | 個 | none |  | av_frame | r_ledger_forecast_p1 | R_only | 台帳の予告 forecast [R再現専用] |
| r_ledger_net_balance | diff_raw | f8 | 個 | none |  | frame |  | R_only | 台帳 net_balance_capped (1P視点符号) [R再現専用] |
| chain_active_p1 | side_abs | i1 | 0/1 | none | mk_chain_p1 | av_fire_p1 | chain_active_p2 | feature | 連鎖進行中(fire_start受理後で終了合図前) |
| chain_active_p2 | side_abs | i1 | 0/1 | none | mk_chain_p2 | av_fire_p2 | chain_active_p1 | feature | 連鎖進行中(fire_start受理後で終了合図前) |
| chain_n_steps_p1 | side_abs | i4 | 段 | none | mk_chain_p1 | av_fire_p1 | chain_n_steps_p2 | feature | 観測済み式の段数 |
| chain_n_steps_p2 | side_abs | i4 | 段 | none | mk_chain_p2 | av_fire_p2 | chain_n_steps_p1 | feature | 観測済み式の段数 |
| chain_pred_l_p1 | side_abs | f4 | 段 | none | mk_chain_p1 | av_fire_p1 | chain_pred_l_p2 | feature | 観測した式の段と整合する最大段数候補の総段数。不整合時は欠測 |
| chain_pred_l_p2 | side_abs | f4 | 段 | none | mk_chain_p2 | av_fire_p2 | chain_pred_l_p1 | feature | 観測した式の段と整合する最大段数候補の総段数。不整合時は欠測 |
| chain_rem_stages_p1 | side_abs | f4 | 段 | none | mk_chain_p1 | av_fire_p1 | chain_rem_stages_p2 | feature | 残り段数 max(L-観測段,0) |
| chain_rem_stages_p2 | side_abs | f4 | 段 | none | mk_chain_p2 | av_fire_p2 | chain_rem_stages_p1 | feature | 残り段数 max(L-観測段,0) |
| chain_rem_sec_p1 | side_abs | f4 | 秒 | none | mk_chain_p1 | av_fire_p1 | chain_rem_sec_p2 | feature | 残り時間 (段時間表+端点補正; 終了合図で0) |
| chain_rem_sec_p2 | side_abs | f4 | 秒 | none | mk_chain_p2 | av_fire_p2 | chain_rem_sec_p1 | feature | 残り時間 (段時間表+端点補正; 終了合図で0) |
| chain_rem_sec_table_p1 | side_abs | f4 | 秒 | none | mk_chain_p1 | av_fire_p1 | chain_rem_sec_table_p2 | feature | 残り時間 (連鎖数別実測テーブル版) |
| chain_rem_sec_table_p2 | side_abs | f4 | 秒 | none | mk_chain_p2 | av_fire_p2 | chain_rem_sec_table_p1 | feature | 残り時間 (連鎖数別実測テーブル版) |
| chain_age_p1 | side_abs | f4 | 秒 | none | mk_chain_p1 | av_fire_p1 | chain_age_p2 | feature | fire_start の trigger からの経過 |
| chain_age_p2 | side_abs | f4 | 秒 | none | mk_chain_p2 | av_fire_p2 | chain_age_p1 | feature | fire_start の trigger からの経過 |
| chain_pred_score_p1 | side_abs | f8 | 点 | none | mk_chain_p1 | av_fire_p1 | chain_pred_score_p2 | feature | 観測済み得点+最大段数の同一候補の残り得点。不整合時は欠測 |
| chain_pred_score_p2 | side_abs | f8 | 点 | none | mk_chain_p2 | av_fire_p2 | chain_pred_score_p1 | feature | 観測済み得点+最大段数の同一候補の残り得点。不整合時は欠測 |
| chain_pred_ojama_p1 | side_abs | f8 | 個 | none | mk_chain_p1 | av_fire_p1 | chain_pred_ojama_p2 | feature | 予測完走おじゃま個数 (現在レート) |
| chain_pred_ojama_p2 | side_abs | f8 | 個 | none | mk_chain_p2 | av_fire_p2 | chain_pred_ojama_p1 | feature | 予測完走おじゃま個数 (現在レート) |
| endsig_stable_p1 | side_abs | i1 | 0/1 | none | mk_chain_p1 | av_fire_p1 | endsig_stable_p2 | feature | 終了合図: CHAIN解除後に2フレーム継続 |
| endsig_stable_p2 | side_abs | i1 | 0/1 | none | mk_chain_p2 | av_fire_p2 | endsig_stable_p1 | feature | 終了合図: CHAIN解除後に2フレーム継続 |
| endsig_next_p1 | side_abs | i1 | 0/1 | none | mk_chain_p1 | av_fire_p1 | endsig_next_p2 | feature | 終了合図: 連鎖後のNEXT移動 (STABLE/TSUMO_FALL中のslide) |
| endsig_next_p2 | side_abs | i1 | 0/1 | none | mk_chain_p2 | av_fire_p2 | endsig_next_p1 | feature | 終了合図: 連鎖後のNEXT移動 (STABLE/TSUMO_FALL中のslide) |
| endsig_ojama_p1 | side_abs | i1 | 0/1 | none | mk_chain_p1 | av_fire_p1 | endsig_ojama_p2 | feature | 終了合図: 連鎖側へおじゃま落下(OJAMA_FALL) |
| endsig_ojama_p2 | side_abs | i1 | 0/1 | none | mk_chain_p2 | av_fire_p2 | endsig_ojama_p1 | feature | 終了合図: 連鎖側へおじゃま落下(OJAMA_FALL) |
| mk_chain_p1 | side_abs | i1 | code | none |  |  | mk_chain_p2 | audit | chain の欠測理由 (Mask) |
| mk_chain_p2 | side_abs | i1 | code | none |  |  | mk_chain_p1 | audit | chain の欠測理由 (Mask) |
| chain_pred_final_p1 | side_abs | str | none | none |  | av_chain_p1 | chain_pred_final_p2 | audit | 同じ最大段数候補の完走盤面JSON。予測不能はnull |
| chain_pred_final_p2 | side_abs | str | none | none |  | av_chain_p2 | chain_pred_final_p1 | audit | 同じ最大段数候補の完走盤面JSON。予測不能はnull |
| chain_pred_reason_p1 | side_abs | str | none | none |  | av_chain_p1 | chain_pred_reason_p2 | audit | 予測不能理由。整合候補ありは空文字 |
| chain_pred_reason_p2 | side_abs | str | none | none |  | av_chain_p2 | chain_pred_reason_p1 | audit | 予測不能理由。整合候補ありは空文字 |
| chain_observed_score_p1 | side_abs | f8 | none | none |  | av_chain_p1 | chain_observed_score_p2 | audit | 観測済み式の積の累積得点 |
| chain_observed_score_p2 | side_abs | f8 | none | none |  | av_chain_p2 | chain_observed_score_p1 | audit | 観測済み式の積の累積得点 |
| av_chain_p1 | side_abs | f8 | none | none |  | av_chain_p1 | av_chain_p2 | audit | 予測に使用した最新観測のavailable_at |
| av_chain_p2 | side_abs | f8 | none | none |  | av_chain_p2 | av_chain_p1 | audit | 予測に使用した最新観測のavailable_at |
| ep_closed_n_p1 | side_abs | i4 | 回 | none | mk_ep_p1 | av_fire_p1 | ep_closed_n_p2 | feature | 閉じた発火エピソード数 (CHAIN復帰なし1秒で閉じる) |
| ep_closed_n_p2 | side_abs | i4 | 回 | none | mk_ep_p2 | av_fire_p2 | ep_closed_n_p1 | feature | 閉じた発火エピソード数 (CHAIN復帰なし1秒で閉じる) |
| ep_age_p1 | side_abs | f4 | 秒 | none | mk_ep_p1 | av_fire_p1 | ep_age_p2 | feature | 直近の閉じたエピソードからの経過 |
| ep_age_p2 | side_abs | f4 | 秒 | none | mk_ep_p2 | av_fire_p2 | ep_age_p1 | feature | 直近の閉じたエピソードからの経過 |
| ep_denominator_p1 | side_abs | f4 | 個 | none | mk_ep_b_p1 | av_fire_p1 | ep_denominator_p2 | feature | 盤面差分と同じ最初のpre盤面の色ぷよ数。pre不在時は起点盤面 |
| ep_denominator_p2 | side_abs | f4 | 個 | none | mk_ep_b_p2 | av_fire_p2 | ep_denominator_p1 | feature | 盤面差分と同じ最初のpre盤面の色ぷよ数。pre不在時は起点盤面 |
| ep_steps_p1 | side_abs | f4 | 段 | none | mk_ep_f_p1 | av_fire_p1 | ep_steps_p2 | feature | 式の段数合計 (完全なものだけ) |
| ep_steps_p2 | side_abs | f4 | 段 | none | mk_ep_f_p2 | av_fire_p2 | ep_steps_p1 | feature | 式の段数合計 (完全なものだけ) |
| ep_consumed_formula_p1 | side_abs | f4 | 個 | none | mk_ep_f_p1 | av_fire_p1 | ep_consumed_formula_p2 | feature | 式由来の消費色ぷよ数 (段完全時のみ) |
| ep_consumed_formula_p2 | side_abs | f4 | 個 | none | mk_ep_f_p2 | av_fire_p2 | ep_consumed_formula_p1 | feature | 式由来の消費色ぷよ数 (段完全時のみ) |
| ep_ratio_formula_p1 | side_abs | f4 | 0-1 | none | mk_ep_f_p1 | av_fire_p1 | ep_ratio_formula_p2 | feature | 消費割合(式) |
| ep_ratio_formula_p2 | side_abs | f4 | 0-1 | none | mk_ep_f_p2 | av_fire_p2 | ep_ratio_formula_p1 | feature | 消費割合(式) |
| ep_consumed_board_p1 | side_abs | f4 | 個 | none | mk_ep_b_p1 | av_fire_p1 | ep_consumed_board_p2 | feature | 盤面差分由来の消費数 (閉じた後のみ) |
| ep_consumed_board_p2 | side_abs | f4 | 個 | none | mk_ep_b_p2 | av_fire_p2 | ep_consumed_board_p1 | feature | 盤面差分由来の消費数 (閉じた後のみ) |
| ep_ratio_board_p1 | side_abs | f4 | 0-1 | none | mk_ep_b_p1 | av_fire_p1 | ep_ratio_board_p2 | feature | 消費割合(盤面差分) |
| ep_ratio_board_p2 | side_abs | f4 | 0-1 | none | mk_ep_b_p2 | av_fire_p2 | ep_ratio_board_p1 | feature | 消費割合(盤面差分) |
| ep_honsen60_formula_p1 | side_abs | f4 | 0/1 | none | mk_ep_f_p1 | av_fire_p1 | ep_honsen60_formula_p2 | feature | 本線: 消費>=60% (式)。NaN=不明層 |
| ep_honsen60_formula_p2 | side_abs | f4 | 0/1 | none | mk_ep_f_p2 | av_fire_p2 | ep_honsen60_formula_p1 | feature | 本線: 消費>=60% (式)。NaN=不明層 |
| ep_honsen60_board_p1 | side_abs | f4 | 0/1 | none | mk_ep_b_p1 | av_fire_p1 | ep_honsen60_board_p2 | feature | 本線: 消費>=60% (盤面差分) |
| ep_honsen60_board_p2 | side_abs | f4 | 0/1 | none | mk_ep_b_p2 | av_fire_p2 | ep_honsen60_board_p1 | feature | 本線: 消費>=60% (盤面差分) |
| ep_saisoku_formula_p1 | side_abs | f4 | 0/1 | none | mk_ep_f_p1 | av_fire_p1 | ep_saisoku_formula_p2 | feature | 催促: 消費<60%かつ送り>4個 (式) |
| ep_saisoku_formula_p2 | side_abs | f4 | 0/1 | none | mk_ep_f_p2 | av_fire_p2 | ep_saisoku_formula_p1 | feature | 催促: 消費<60%かつ送り>4個 (式) |
| ep_chain7plus_p1 | side_abs | f4 | 0/1 | none | mk_ep_f_p1 | av_fire_p1 | ep_chain7plus_p2 | feature | 7連鎖以上完走 (式の段数>=7)。<7かつ不完全は不明 |
| ep_chain7plus_p2 | side_abs | f4 | 0/1 | none | mk_ep_f_p2 | av_fire_p2 | ep_chain7plus_p1 | feature | 7連鎖以上完走 (式の段数>=7)。<7かつ不完全は不明 |
| ep_sent_ojama_p1 | side_abs | f8 | 個 | none | mk_ep_f_p1 | av_fire_p1 | ep_sent_ojama_p2 | feature | 式の得点合計のおじゃま換算 |
| ep_sent_ojama_p2 | side_abs | f8 | 個 | none | mk_ep_f_p2 | av_fire_p2 | ep_sent_ojama_p1 | feature | 式の得点合計のおじゃま換算 |
| pf_available_p1 | side_abs | i1 | 0/1 | none |  | av_fire_p1 | pf_available_p2 | feature | 直近発火で過去リング由来の発火前盤面があるか |
| pf_available_p2 | side_abs | i1 | 0/1 | none |  | av_fire_p2 | pf_available_p1 | feature | 直近発火で過去リング由来の発火前盤面があるか |
| pf_color_p1 | side_abs | f4 | 個 | none | mk_pf_p1 | av_fire_p1 | pf_color_p2 | feature | 発火前盤面(過去確定)の色ぷよ数 |
| pf_color_p2 | side_abs | f4 | 個 | none | mk_pf_p2 | av_fire_p2 | pf_color_p1 | feature | 発火前盤面(過去確定)の色ぷよ数 |
| pf_before_color_p1 | side_abs | f4 | 個 | none | mk_pf_p1 | av_fire_p1 | pf_before_color_p2 | feature | 発火開始時盤面の色ぷよ数 |
| pf_before_color_p2 | side_abs | f4 | 個 | none | mk_pf_p2 | av_fire_p2 | pf_before_color_p1 | feature | 発火開始時盤面の色ぷよ数 |
| mk_ep_p1 | side_abs | i1 | code | none |  |  | mk_ep_p2 | audit | episode の欠測理由 (Mask) |
| mk_ep_p2 | side_abs | i1 | code | none |  |  | mk_ep_p1 | audit | episode の欠測理由 (Mask) |
| mk_ep_f_p1 | side_abs | i1 | code | none |  |  | mk_ep_f_p2 | audit | episode の欠測理由 (Mask) |
| mk_ep_f_p2 | side_abs | i1 | code | none |  |  | mk_ep_f_p1 | audit | episode の欠測理由 (Mask) |
| mk_ep_b_p1 | side_abs | i1 | code | none |  |  | mk_ep_b_p2 | audit | episode の欠測理由 (Mask) |
| mk_ep_b_p2 | side_abs | i1 | code | none |  |  | mk_ep_b_p1 | audit | episode の欠測理由 (Mask) |
| mk_pf_p1 | side_abs | i1 | code | none |  |  | mk_pf_p2 | audit | episode の欠測理由 (Mask) |
| mk_pf_p2 | side_abs | i1 | code | none |  |  | mk_pf_p1 | audit | episode の欠測理由 (Mask) |
| ta_active_p1 | side_abs | i1 | 0/1 | none | mk_ta_p1 | av_fire_p1 | ta_active_p2 | feature | 攻撃側aが連鎖中(予測完走あり) |
| ta_active_p2 | side_abs | i1 | 0/1 | none | mk_ta_p2 | av_fire_p2 | ta_active_p1 | feature | 攻撃側aが連鎖中(予測完走あり) |
| ta_a_rem_sec_p1 | side_abs | f4 | 秒 | none | mk_ta_p1 | av_fire_p1 | ta_a_rem_sec_p2 | feature | aの残り連鎖時間 |
| ta_a_rem_sec_p2 | side_abs | f4 | 秒 | none | mk_ta_p2 | av_fire_p2 | ta_a_rem_sec_p1 | feature | aの残り連鎖時間 |
| ta_b_busy_p1 | side_abs | f4 | 秒 | none | mk_ta_p1 | av_fire_p1 | ta_b_busy_p2 | feature | 受け側bの操作不能残り (自連鎖中) |
| ta_b_busy_p2 | side_abs | f4 | 秒 | none | mk_ta_p2 | av_fire_p2 | ta_b_busy_p1 | feature | 受け側bの操作不能残り (自連鎖中) |
| ta_window_p1 | side_abs | f4 | 秒 | none | mk_ta_p1 | av_fire_p1 | ta_window_p2 | feature | bが置ける窓 max(a残-b操作不能,0) |
| ta_window_p2 | side_abs | f4 | 秒 | none | mk_ta_p2 | av_fire_p2 | ta_window_p1 | feature | bが置ける窓 max(a残-b操作不能,0) |
| ta_hand_med_sec_p1 | side_abs | f4 | 秒 | none | mk_ta_p1 | av_fire_p1 | ta_hand_med_sec_p2 | feature | 1手時間 中央値版 (高さ別表) |
| ta_hand_med_sec_p2 | side_abs | f4 | 秒 | none | mk_ta_p2 | av_fire_p2 | ta_hand_med_sec_p1 | feature | 1手時間 中央値版 (高さ別表) |
| ta_hand_fast_sec_p1 | side_abs | f4 | 秒 | none | mk_ta_p1 | av_fire_p1 | ta_hand_fast_sec_p2 | feature | 1手時間 下位10%版 (高さ別表) |
| ta_hand_fast_sec_p2 | side_abs | f4 | 秒 | none | mk_ta_p2 | av_fire_p2 | ta_hand_fast_sec_p1 | feature | 1手時間 下位10%版 (高さ別表) |
| ta_H_med_p1 | side_abs | f4 | 手 | none | mk_ta_p1 | av_fire_p1 | ta_H_med_p2 | feature | 置ける総手数 中央値版 floor(窓/d)+1 |
| ta_H_med_p2 | side_abs | f4 | 手 | none | mk_ta_p2 | av_fire_p2 | ta_H_med_p1 | feature | 置ける総手数 中央値版 floor(窓/d)+1 |
| ta_H_fast_p1 | side_abs | f4 | 手 | none | mk_ta_p1 | av_fire_p1 | ta_H_fast_p2 | feature | 置ける総手数 下位10%版 |
| ta_H_fast_p2 | side_abs | f4 | 手 | none | mk_ta_p2 | av_fire_p2 | ta_H_fast_p1 | feature | 置ける総手数 下位10%版 |
| ta_req_h_p1 | side_abs | f4 | 手 | none | mk_ta_p1 | av_fire_p1 | ta_req_h_p2 | feature | 探索に要求した総手数 |
| ta_req_h_p2 | side_abs | f4 | 手 | none | mk_ta_p2 | av_fire_p2 | ta_req_h_p1 | feature | 探索に要求した総手数 |
| ta_done_h_p1 | side_abs | f4 | 手 | none | mk_ta_p1 | av_fire_p1 | ta_done_h_p2 | feature | 探索済み総手数 |
| ta_done_h_p2 | side_abs | f4 | 手 | none | mk_ta_p2 | av_fire_p2 | ta_done_h_p1 | feature | 探索済み総手数 |
| ta_trunc_p1 | side_abs | i1 | 0/1 | none | mk_ta_p1 | av_fire_p1 | ta_trunc_p2 | feature | 要求Hが探索上限で打切られた |
| ta_trunc_p2 | side_abs | i1 | 0/1 | none | mk_ta_p2 | av_fire_p2 | ta_trunc_p1 | feature | 要求Hが探索上限で打切られた |
| ta_q_med_p1 | side_abs | f8 | 個 | none | mk_ta_p1 | av_fire_p1 | ta_q_med_p2 | feature | B3専用探索: bのH_med組以内の最大発火量 (理想ツモ, おじゃま個数) |
| ta_q_med_p2 | side_abs | f8 | 個 | none | mk_ta_p2 | av_fire_p2 | ta_q_med_p1 | feature | B3専用探索: bのH_med組以内の最大発火量 (理想ツモ, おじゃま個数) |
| ta_q_fast_p1 | side_abs | f8 | 個 | none | mk_ta_p1 | av_fire_p1 | ta_q_fast_p2 | feature | B3専用探索: bのH_fast組以内の最大発火量 |
| ta_q_fast_p2 | side_abs | f8 | 個 | none | mk_ta_p2 | av_fire_p2 | ta_q_fast_p1 | feature | B3専用探索: bのH_fast組以内の最大発火量 |
| ta_i_pred_p1 | side_abs | f8 | 個 | none | mk_ta_p1 | av_fire_p1 | ta_i_pred_p2 | feature | bの受け量: aの予測完走おじゃま |
| ta_i_pred_p2 | side_abs | f8 | 個 | none | mk_ta_p2 | av_fire_p2 | ta_i_pred_p1 | feature | bの受け量: aの予測完走おじゃま |
| ta_i_sl_p1 | side_abs | f8 | 個 | none | mk_ta_p1 | av_fire_p1 | ta_i_sl_p2 | feature | bの受け量: 台帳v2の予告 sl_pending_b (主入力。v2定義確定後に in-flight 連鎖の含有を再確認) |
| ta_i_sl_p2 | side_abs | f8 | 個 | none | mk_ta_p2 | av_fire_p2 | ta_i_sl_p1 | feature | bの受け量: 台帳v2の予告 sl_pending_b (主入力。v2定義確定後に in-flight 連鎖の含有を再確認) |
| ta_i_icon_p1 | side_abs | f8 | 個 | none | mk_ta_p1 | av_fire_p1 | ta_i_icon_p2 | feature | [照合用] bの画面アイコン補正値+aの予測完走 |
| ta_i_icon_p2 | side_abs | f8 | 個 | none | mk_ta_p2 | av_fire_p2 | ta_i_icon_p1 | feature | [照合用] bの画面アイコン補正値+aの予測完走 |
| ta_q_med_nf_p1 | side_abs | f4 | 0-1 | z/(a+z) a=3.0 | mk_ta_p1 | av_fire_p1 | ta_q_med_nf_p2 | feature | Q 中央値版の正規化 |
| ta_q_med_nf_p2 | side_abs | f4 | 0-1 | z/(a+z) a=3.0 | mk_ta_p2 | av_fire_p2 | ta_q_med_nf_p1 | feature | Q 中央値版の正規化 |
| ta_q_fast_nf_p1 | side_abs | f4 | 0-1 | z/(a+z) a=3.0 | mk_ta_p1 | av_fire_p1 | ta_q_fast_nf_p2 | feature | Q 下位10%版の正規化 |
| ta_q_fast_nf_p2 | side_abs | f4 | 0-1 | z/(a+z) a=3.0 | mk_ta_p2 | av_fire_p2 | ta_q_fast_nf_p1 | feature | Q 下位10%版の正規化 |
| ta_m_pred_med_p1 | side_abs | f8 | 個 | none | mk_ta_p1 | av_fire_p1 | ta_m_pred_med_p2 | feature | 余地 M=Q-I (pred_med) |
| ta_m_pred_med_p2 | side_abs | f8 | 個 | none | mk_ta_p2 | av_fire_p2 | ta_m_pred_med_p1 | feature | 余地 M=Q-I (pred_med) |
| ta_mg_pred_med_p1 | side_abs | f4 | 0-1 | 0.5+0.5*sign*z/(a+z) a=3.0 | mk_ta_p1 | av_fire_p1 | ta_mg_pred_med_p2 | feature | M の符号付き非飽和 (pred_med); M=0→0.5 |
| ta_mg_pred_med_p2 | side_abs | f4 | 0-1 | 0.5+0.5*sign*z/(a+z) a=3.0 | mk_ta_p2 | av_fire_p2 | ta_mg_pred_med_p1 | feature | M の符号付き非飽和 (pred_med); M=0→0.5 |
| ta_recv_pred_med_p1 | side_abs | f8 | 個 | none | mk_ta_p1 | av_fire_p1 | ta_recv_pred_med_p2 | feature | bの残受け max(I-Q,0) (pred_med) |
| ta_recv_pred_med_p2 | side_abs | f8 | 個 | none | mk_ta_p2 | av_fire_p2 | ta_recv_pred_med_p1 | feature | bの残受け max(I-Q,0) (pred_med) |
| ta_send_pred_med_p1 | side_abs | f8 | 個 | none | mk_ta_p1 | av_fire_p1 | ta_send_pred_med_p2 | feature | bの残送り max(Q-I,0) (pred_med) |
| ta_send_pred_med_p2 | side_abs | f8 | 個 | none | mk_ta_p2 | av_fire_p2 | ta_send_pred_med_p1 | feature | bの残送り max(Q-I,0) (pred_med) |
| ta_unc72_pred_med_p1 | side_abs | i1 | 0/1 | none | mk_ta_p1 | av_fire_p1 | ta_unc72_pred_med_p2 | feature | 相殺不能の予告 残受け>=72個 (pred_med) [特徴のみ・死亡証明には使わない] |
| ta_unc72_pred_med_p2 | side_abs | i1 | 0/1 | none | mk_ta_p2 | av_fire_p2 | ta_unc72_pred_med_p1 | feature | 相殺不能の予告 残受け>=72個 (pred_med) [特徴のみ・死亡証明には使わない] |
| ta_m_pred_fast_p1 | side_abs | f8 | 個 | none | mk_ta_p1 | av_fire_p1 | ta_m_pred_fast_p2 | feature | 余地 M=Q-I (pred_fast) |
| ta_m_pred_fast_p2 | side_abs | f8 | 個 | none | mk_ta_p2 | av_fire_p2 | ta_m_pred_fast_p1 | feature | 余地 M=Q-I (pred_fast) |
| ta_mg_pred_fast_p1 | side_abs | f4 | 0-1 | 0.5+0.5*sign*z/(a+z) a=3.0 | mk_ta_p1 | av_fire_p1 | ta_mg_pred_fast_p2 | feature | M の符号付き非飽和 (pred_fast); M=0→0.5 |
| ta_mg_pred_fast_p2 | side_abs | f4 | 0-1 | 0.5+0.5*sign*z/(a+z) a=3.0 | mk_ta_p2 | av_fire_p2 | ta_mg_pred_fast_p1 | feature | M の符号付き非飽和 (pred_fast); M=0→0.5 |
| ta_recv_pred_fast_p1 | side_abs | f8 | 個 | none | mk_ta_p1 | av_fire_p1 | ta_recv_pred_fast_p2 | feature | bの残受け max(I-Q,0) (pred_fast) |
| ta_recv_pred_fast_p2 | side_abs | f8 | 個 | none | mk_ta_p2 | av_fire_p2 | ta_recv_pred_fast_p1 | feature | bの残受け max(I-Q,0) (pred_fast) |
| ta_send_pred_fast_p1 | side_abs | f8 | 個 | none | mk_ta_p1 | av_fire_p1 | ta_send_pred_fast_p2 | feature | bの残送り max(Q-I,0) (pred_fast) |
| ta_send_pred_fast_p2 | side_abs | f8 | 個 | none | mk_ta_p2 | av_fire_p2 | ta_send_pred_fast_p1 | feature | bの残送り max(Q-I,0) (pred_fast) |
| ta_unc72_pred_fast_p1 | side_abs | i1 | 0/1 | none | mk_ta_p1 | av_fire_p1 | ta_unc72_pred_fast_p2 | feature | 相殺不能の予告 残受け>=72個 (pred_fast) [特徴のみ・死亡証明には使わない] |
| ta_unc72_pred_fast_p2 | side_abs | i1 | 0/1 | none | mk_ta_p2 | av_fire_p2 | ta_unc72_pred_fast_p1 | feature | 相殺不能の予告 残受け>=72個 (pred_fast) [特徴のみ・死亡証明には使わない] |
| ta_m_sl_med_p1 | side_abs | f8 | 個 | none | mk_ta_p1 | av_fire_p1 | ta_m_sl_med_p2 | feature | 余地 M=Q-I (sl_med) |
| ta_m_sl_med_p2 | side_abs | f8 | 個 | none | mk_ta_p2 | av_fire_p2 | ta_m_sl_med_p1 | feature | 余地 M=Q-I (sl_med) |
| ta_mg_sl_med_p1 | side_abs | f4 | 0-1 | 0.5+0.5*sign*z/(a+z) a=3.0 | mk_ta_p1 | av_fire_p1 | ta_mg_sl_med_p2 | feature | M の符号付き非飽和 (sl_med); M=0→0.5 |
| ta_mg_sl_med_p2 | side_abs | f4 | 0-1 | 0.5+0.5*sign*z/(a+z) a=3.0 | mk_ta_p2 | av_fire_p2 | ta_mg_sl_med_p1 | feature | M の符号付き非飽和 (sl_med); M=0→0.5 |
| ta_recv_sl_med_p1 | side_abs | f8 | 個 | none | mk_ta_p1 | av_fire_p1 | ta_recv_sl_med_p2 | feature | bの残受け max(I-Q,0) (sl_med) |
| ta_recv_sl_med_p2 | side_abs | f8 | 個 | none | mk_ta_p2 | av_fire_p2 | ta_recv_sl_med_p1 | feature | bの残受け max(I-Q,0) (sl_med) |
| ta_send_sl_med_p1 | side_abs | f8 | 個 | none | mk_ta_p1 | av_fire_p1 | ta_send_sl_med_p2 | feature | bの残送り max(Q-I,0) (sl_med) |
| ta_send_sl_med_p2 | side_abs | f8 | 個 | none | mk_ta_p2 | av_fire_p2 | ta_send_sl_med_p1 | feature | bの残送り max(Q-I,0) (sl_med) |
| ta_unc72_sl_med_p1 | side_abs | i1 | 0/1 | none | mk_ta_p1 | av_fire_p1 | ta_unc72_sl_med_p2 | feature | 相殺不能の予告 残受け>=72個 (sl_med) [特徴のみ・死亡証明には使わない] |
| ta_unc72_sl_med_p2 | side_abs | i1 | 0/1 | none | mk_ta_p2 | av_fire_p2 | ta_unc72_sl_med_p1 | feature | 相殺不能の予告 残受け>=72個 (sl_med) [特徴のみ・死亡証明には使わない] |
| ta_m_sl_fast_p1 | side_abs | f8 | 個 | none | mk_ta_p1 | av_fire_p1 | ta_m_sl_fast_p2 | feature | 余地 M=Q-I (sl_fast) |
| ta_m_sl_fast_p2 | side_abs | f8 | 個 | none | mk_ta_p2 | av_fire_p2 | ta_m_sl_fast_p1 | feature | 余地 M=Q-I (sl_fast) |
| ta_mg_sl_fast_p1 | side_abs | f4 | 0-1 | 0.5+0.5*sign*z/(a+z) a=3.0 | mk_ta_p1 | av_fire_p1 | ta_mg_sl_fast_p2 | feature | M の符号付き非飽和 (sl_fast); M=0→0.5 |
| ta_mg_sl_fast_p2 | side_abs | f4 | 0-1 | 0.5+0.5*sign*z/(a+z) a=3.0 | mk_ta_p2 | av_fire_p2 | ta_mg_sl_fast_p1 | feature | M の符号付き非飽和 (sl_fast); M=0→0.5 |
| ta_recv_sl_fast_p1 | side_abs | f8 | 個 | none | mk_ta_p1 | av_fire_p1 | ta_recv_sl_fast_p2 | feature | bの残受け max(I-Q,0) (sl_fast) |
| ta_recv_sl_fast_p2 | side_abs | f8 | 個 | none | mk_ta_p2 | av_fire_p2 | ta_recv_sl_fast_p1 | feature | bの残受け max(I-Q,0) (sl_fast) |
| ta_send_sl_fast_p1 | side_abs | f8 | 個 | none | mk_ta_p1 | av_fire_p1 | ta_send_sl_fast_p2 | feature | bの残送り max(Q-I,0) (sl_fast) |
| ta_send_sl_fast_p2 | side_abs | f8 | 個 | none | mk_ta_p2 | av_fire_p2 | ta_send_sl_fast_p1 | feature | bの残送り max(Q-I,0) (sl_fast) |
| ta_unc72_sl_fast_p1 | side_abs | i1 | 0/1 | none | mk_ta_p1 | av_fire_p1 | ta_unc72_sl_fast_p2 | feature | 相殺不能の予告 残受け>=72個 (sl_fast) [特徴のみ・死亡証明には使わない] |
| ta_unc72_sl_fast_p2 | side_abs | i1 | 0/1 | none | mk_ta_p2 | av_fire_p2 | ta_unc72_sl_fast_p1 | feature | 相殺不能の予告 残受け>=72個 (sl_fast) [特徴のみ・死亡証明には使わない] |
| ta_m_icon_med_p1 | side_abs | f8 | 個 | none | mk_ta_p1 | av_fire_p1 | ta_m_icon_med_p2 | feature | 余地 M=Q-I (icon_med) |
| ta_m_icon_med_p2 | side_abs | f8 | 個 | none | mk_ta_p2 | av_fire_p2 | ta_m_icon_med_p1 | feature | 余地 M=Q-I (icon_med) |
| ta_mg_icon_med_p1 | side_abs | f4 | 0-1 | 0.5+0.5*sign*z/(a+z) a=3.0 | mk_ta_p1 | av_fire_p1 | ta_mg_icon_med_p2 | feature | M の符号付き非飽和 (icon_med); M=0→0.5 |
| ta_mg_icon_med_p2 | side_abs | f4 | 0-1 | 0.5+0.5*sign*z/(a+z) a=3.0 | mk_ta_p2 | av_fire_p2 | ta_mg_icon_med_p1 | feature | M の符号付き非飽和 (icon_med); M=0→0.5 |
| ta_recv_icon_med_p1 | side_abs | f8 | 個 | none | mk_ta_p1 | av_fire_p1 | ta_recv_icon_med_p2 | feature | bの残受け max(I-Q,0) (icon_med) |
| ta_recv_icon_med_p2 | side_abs | f8 | 個 | none | mk_ta_p2 | av_fire_p2 | ta_recv_icon_med_p1 | feature | bの残受け max(I-Q,0) (icon_med) |
| ta_send_icon_med_p1 | side_abs | f8 | 個 | none | mk_ta_p1 | av_fire_p1 | ta_send_icon_med_p2 | feature | bの残送り max(Q-I,0) (icon_med) |
| ta_send_icon_med_p2 | side_abs | f8 | 個 | none | mk_ta_p2 | av_fire_p2 | ta_send_icon_med_p1 | feature | bの残送り max(Q-I,0) (icon_med) |
| ta_unc72_icon_med_p1 | side_abs | i1 | 0/1 | none | mk_ta_p1 | av_fire_p1 | ta_unc72_icon_med_p2 | feature | 相殺不能の予告 残受け>=72個 (icon_med) [特徴のみ・死亡証明には使わない] |
| ta_unc72_icon_med_p2 | side_abs | i1 | 0/1 | none | mk_ta_p2 | av_fire_p2 | ta_unc72_icon_med_p1 | feature | 相殺不能の予告 残受け>=72個 (icon_med) [特徴のみ・死亡証明には使わない] |
| ta_m_icon_fast_p1 | side_abs | f8 | 個 | none | mk_ta_p1 | av_fire_p1 | ta_m_icon_fast_p2 | feature | 余地 M=Q-I (icon_fast) |
| ta_m_icon_fast_p2 | side_abs | f8 | 個 | none | mk_ta_p2 | av_fire_p2 | ta_m_icon_fast_p1 | feature | 余地 M=Q-I (icon_fast) |
| ta_mg_icon_fast_p1 | side_abs | f4 | 0-1 | 0.5+0.5*sign*z/(a+z) a=3.0 | mk_ta_p1 | av_fire_p1 | ta_mg_icon_fast_p2 | feature | M の符号付き非飽和 (icon_fast); M=0→0.5 |
| ta_mg_icon_fast_p2 | side_abs | f4 | 0-1 | 0.5+0.5*sign*z/(a+z) a=3.0 | mk_ta_p2 | av_fire_p2 | ta_mg_icon_fast_p1 | feature | M の符号付き非飽和 (icon_fast); M=0→0.5 |
| ta_recv_icon_fast_p1 | side_abs | f8 | 個 | none | mk_ta_p1 | av_fire_p1 | ta_recv_icon_fast_p2 | feature | bの残受け max(I-Q,0) (icon_fast) |
| ta_recv_icon_fast_p2 | side_abs | f8 | 個 | none | mk_ta_p2 | av_fire_p2 | ta_recv_icon_fast_p1 | feature | bの残受け max(I-Q,0) (icon_fast) |
| ta_send_icon_fast_p1 | side_abs | f8 | 個 | none | mk_ta_p1 | av_fire_p1 | ta_send_icon_fast_p2 | feature | bの残送り max(Q-I,0) (icon_fast) |
| ta_send_icon_fast_p2 | side_abs | f8 | 個 | none | mk_ta_p2 | av_fire_p2 | ta_send_icon_fast_p1 | feature | bの残送り max(Q-I,0) (icon_fast) |
| ta_unc72_icon_fast_p1 | side_abs | i1 | 0/1 | none | mk_ta_p1 | av_fire_p1 | ta_unc72_icon_fast_p2 | feature | 相殺不能の予告 残受け>=72個 (icon_fast) [特徴のみ・死亡証明には使わない] |
| ta_unc72_icon_fast_p2 | side_abs | i1 | 0/1 | none | mk_ta_p2 | av_fire_p2 | ta_unc72_icon_fast_p1 | feature | 相殺不能の予告 残受け>=72個 (icon_fast) [特徴のみ・死亡証明には使わない] |
| mk_ta_p1 | side_abs | i1 | code | none |  |  | mk_ta_p2 | audit | time_axis の欠測理由 (Mask) |
| mk_ta_p2 | side_abs | i1 | code | none |  |  | mk_ta_p1 | audit | time_axis の欠測理由 (Mask) |
| ta2_q_med_p1 | side_abs | f8 | 個 | none | mk_ta2_p1 | av_fire_p1 | ta2_q_med_p2 | feature | 一往復 a→b→a の ta2_q_med [v1 未実装・全行mask=DEFERRED] |
| ta2_q_med_p2 | side_abs | f8 | 個 | none | mk_ta2_p2 | av_fire_p2 | ta2_q_med_p1 | feature | 一往復 a→b→a の ta2_q_med [v1 未実装・全行mask=DEFERRED] |
| ta2_m_med_p1 | side_abs | f8 | 個 | none | mk_ta2_p1 | av_fire_p1 | ta2_m_med_p2 | feature | 一往復 a→b→a の ta2_m_med [v1 未実装・全行mask=DEFERRED] |
| ta2_m_med_p2 | side_abs | f8 | 個 | none | mk_ta2_p2 | av_fire_p2 | ta2_m_med_p1 | feature | 一往復 a→b→a の ta2_m_med [v1 未実装・全行mask=DEFERRED] |
| mk_ta2_p1 | side_abs | i1 | code | none |  |  | mk_ta2_p2 | audit | time_axis2 の欠測理由 (Mask) |
| mk_ta2_p2 | side_abs | i1 | code | none |  |  | mk_ta2_p1 | audit | time_axis2 の欠測理由 (Mask) |
| mk_df_ind | common | i1 | code | none |  |  |  | audit | mk_df_ind: 差分列の欠測理由 = 両側 mask の最大値 (側交換で不変) |
| mk_df_k | common | i1 | code | none |  |  |  | audit | mk_df_k: 差分列の欠測理由 = 両側 mask の最大値 (側交換で不変) |
| mk_df_board | common | i1 | code | none |  |  |  | audit | mk_df_board: 差分列の欠測理由 = 両側 mask の最大値 (側交換で不変) |
| mk_df_icon | common | i1 | code | none |  |  |  | audit | mk_df_icon: 差分列の欠測理由 = 両側 mask の最大値 (側交換で不変) |
| df_ind_board_color_puyo_total | diff_centered | f4 | 0-1 | 0.5+0.5*(p1-p2) | mk_df_ind |  |  | feature | board_color_puyo_total の中心化差分 |
| df_ind_board_puyo_total | diff_centered | f4 | 0-1 | 0.5+0.5*(p1-p2) | mk_df_ind |  |  | feature | board_puyo_total の中心化差分 |
| df_ind_max_column_height | diff_centered | f4 | 0-1 | 0.5+0.5*(p1-p2) | mk_df_ind |  |  | feature | max_column_height の中心化差分 |
| df_ind_column_bumpiness | diff_centered | f4 | 0-1 | 0.5+0.5*(p1-p2) | mk_df_ind |  |  | feature | column_bumpiness の中心化差分 |
| df_ind_death_margin | diff_centered | f4 | 0-1 | 0.5+0.5*(p1-p2) | mk_df_ind |  |  | feature | death_margin の中心化差分 |
| df_ind_death_margin_neighbor | diff_centered | f4 | 0-1 | 0.5+0.5*(p1-p2) | mk_df_ind |  |  | feature | death_margin_neighbor の中心化差分 |
| df_ind_center_bulge_color | diff_centered | f4 | 0-1 | 0.5+0.5*(p1-p2) | mk_df_ind |  |  | feature | center_bulge_color の中心化差分 |
| df_ind_center_bulge_ojama | diff_centered | f4 | 0-1 | 0.5+0.5*(p1-p2) | mk_df_ind |  |  | feature | center_bulge_ojama の中心化差分 |
| df_ind_board_ojama_count | diff_centered | f4 | 0-1 | 0.5+0.5*(p1-p2) | mk_df_ind |  |  | feature | board_ojama_count の中心化差分 |
| df_ind_color_diversity_evenness | diff_centered | f4 | 0-1 | 0.5+0.5*(p1-p2) | mk_df_ind |  |  | feature | color_diversity_evenness の中心化差分 |
| df_ind_buried_hole_count | diff_centered | f4 | 0-1 | 0.5+0.5*(p1-p2) | mk_df_ind |  |  | feature | buried_hole_count の中心化差分 |
| df_ind_current_max_chain | diff_centered | f4 | 0-1 | 0.5+0.5*(p1-p2) | mk_df_ind |  |  | feature | current_max_chain の中心化差分 |
| df_ind_dig_resistance | diff_centered | f4 | 0-1 | 0.5+0.5*(p1-p2) | mk_df_ind |  |  | feature | dig_resistance の中心化差分 |
| df_ind_ukeyasusa | diff_centered | f4 | 0-1 | 0.5+0.5*(p1-p2) | mk_df_ind |  |  | feature | ukeyasusa の中心化差分 |
| df_ind_sub_chain_count | diff_centered | f4 | 0-1 | 0.5+0.5*(p1-p2) | mk_df_ind |  |  | feature | sub_chain_count の中心化差分 |
| df_ind_immediate_fire_power | diff_centered | f4 | 0-1 | 0.5+0.5*(p1-p2) | mk_df_ind |  |  | feature | immediate_fire_power の中心化差分 |
| df_ind_chain_efficiency | diff_centered | f4 | 0-1 | 0.5+0.5*(p1-p2) | mk_df_ind |  |  | feature | chain_efficiency の中心化差分 |
| df_ind_min_puyos_to_ignite | diff_centered | f4 | 0-1 | 0.5+0.5*(p1-p2) | mk_df_ind |  |  | feature | min_puyos_to_ignite の中心化差分 |
| df_ind_second_chain_potential | diff_centered | f4 | 0-1 | 0.5+0.5*(p1-p2) | mk_df_ind |  |  | feature | second_chain_potential の中心化差分 |
| df_ind_main_linked_pair_count | diff_centered | f4 | 0-1 | 0.5+0.5*(p1-p2) | mk_df_ind |  |  | feature | main_linked_pair_count の中心化差分 |
| df_ind_isolated_pair_count | diff_centered | f4 | 0-1 | 0.5+0.5*(p1-p2) | mk_df_ind |  |  | feature | isolated_pair_count の中心化差分 |
| df_ind_main_linked_ratio | diff_centered | f4 | 0-1 | 0.5+0.5*(p1-p2) | mk_df_ind |  |  | feature | main_linked_ratio の中心化差分 |
| df_ind_ignition_point_count | diff_centered | f4 | 0-1 | 0.5+0.5*(p1-p2) | mk_df_ind |  |  | feature | ignition_point_count の中心化差分 |
| df_ind_multi_color_ignition | diff_centered | f4 | 0-1 | 0.5+0.5*(p1-p2) | mk_df_ind |  |  | feature | multi_color_ignition の中心化差分 |
| df_ind_simultaneous_pop_richness | diff_centered | f4 | 0-1 | 0.5+0.5*(p1-p2) | mk_df_ind |  |  | feature | simultaneous_pop_richness の中心化差分 |
| df_ind_saturation_chain_upper | diff_centered | f4 | 0-1 | 0.5+0.5*(p1-p2) | mk_df_ind |  |  | feature | saturation_chain_upper の中心化差分 |
| df_ind_chain_articulation_point_count | diff_centered | f4 | 0-1 | 0.5+0.5*(p1-p2) | mk_df_ind |  |  | feature | chain_articulation_point_count の中心化差分 |
| df_ind_conn_pair_count | diff_raw | f4 | 個 | none | mk_df_ind |  |  | feature | conn_pair_count の差 p1-p2 |
| df_ind_conn_triple_count | diff_raw | f4 | 個 | none | mk_df_ind |  |  | feature | conn_triple_count の差 p1-p2 |
| df_ind_conn_max_group_size | diff_raw | f4 | 個 | none | mk_df_ind |  |  | feature | conn_max_group_size の差 p1-p2 |
| df_k1_nf | diff_centered | f4 | 0-1 | 0.5+0.5*(p1-p2) | mk_df_k |  |  | feature | k1 正規化の中心化差分 |
| df_k2_nf | diff_centered | f4 | 0-1 | 0.5+0.5*(p1-p2) | mk_df_k |  |  | feature | k2 正規化の中心化差分 |
| df_k3_nf | diff_centered | f4 | 0-1 | 0.5+0.5*(p1-p2) | mk_df_k |  |  | feature | k3 正規化の中心化差分 |
| df_k4_nf | diff_centered | f4 | 0-1 | 0.5+0.5*(p1-p2) | mk_df_k |  |  | feature | k4 正規化の中心化差分 |
| df_k5_nf | diff_centered | f4 | 0-1 | 0.5+0.5*(p1-p2) | mk_df_k |  |  | feature | k5 正規化の中心化差分 |
| df_cells_color | diff_raw | f4 | 個 | none | mk_df_board |  |  | feature | 色ぷよ数差 |
| df_cells_ojama | diff_raw | f4 | 個 | none | mk_df_board |  |  | feature | 盤面おじゃま差 |
| df_icon_corr_total | diff_raw | f4 | 個 | none | mk_df_icon |  |  | feature | 補正アイコン総数の差 |
| search_trunc_empty_p1 | side_abs | f4 | マス | none | mk_search_trunc_p1 | av_board_p1 | search_trunc_empty_p2 | audit | 要求H未達の上限打切り時、保持ビームの最大空き(row0込み78マス)。非打切りはNaN |
| search_trunc_empty_p2 | side_abs | f4 | マス | none | mk_search_trunc_p2 | av_board_p2 | search_trunc_empty_p1 | audit | 要求H未達の上限打切り時、保持ビームの最大空き(row0込み78マス)。非打切りはNaN |
| search_trunc_empty_nf_p1 | side_abs | f4 | 0-1 | 空き/78 | mk_search_trunc_p1 | av_board_p1 | search_trunc_empty_nf_p2 | audit | 打切り時空きの正規化。非打切りはNaN |
| search_trunc_empty_nf_p2 | side_abs | f4 | 0-1 | 空き/78 | mk_search_trunc_p2 | av_board_p2 | search_trunc_empty_nf_p1 | audit | 打切り時空きの正規化。非打切りはNaN |
| search_full_p1 | side_abs | f4 | 0/1 | 0-1 | mk_k_p1 | av_board_p1 | search_full_p2 | audit | 要求H以内に保持ビームの全候補で合法な次手が無く終了(満杯)。打切りと区別 |
| search_full_p2 | side_abs | f4 | 0/1 | 0-1 | mk_k_p2 | av_board_p2 | search_full_p1 | audit | 要求H以内に保持ビームの全候補で合法な次手が無く終了(満杯)。打切りと区別 |
| mk_search_trunc_p1 | side_abs | i1 | code | none |  |  | mk_search_trunc_p2 | audit | search_extra の欠測理由 (Mask) |
| mk_search_trunc_p2 | side_abs | i1 | code | none |  |  | mk_search_trunc_p1 | audit | search_extra の欠測理由 (Mask) |
| ta_trunc_empty_p1 | side_abs | f4 | マス | none | mk_ta_trunc_p1 | av_fire_p1 | ta_trunc_empty_p2 | audit | 要求H未達の上限打切り時、保持ビームの最大空き(row0込み78マス)。非打切りはNaN |
| ta_trunc_empty_p2 | side_abs | f4 | マス | none | mk_ta_trunc_p2 | av_fire_p2 | ta_trunc_empty_p1 | audit | 要求H未達の上限打切り時、保持ビームの最大空き(row0込み78マス)。非打切りはNaN |
| ta_trunc_empty_nf_p1 | side_abs | f4 | 0-1 | 空き/78 | mk_ta_trunc_p1 | av_fire_p1 | ta_trunc_empty_nf_p2 | audit | 打切り時空きの正規化。非打切りはNaN |
| ta_trunc_empty_nf_p2 | side_abs | f4 | 0-1 | 空き/78 | mk_ta_trunc_p2 | av_fire_p2 | ta_trunc_empty_nf_p1 | audit | 打切り時空きの正規化。非打切りはNaN |
| ta_full_p1 | side_abs | f4 | 0/1 | 0-1 | mk_ta_p1 | av_fire_p1 | ta_full_p2 | audit | 要求H以内に保持ビームの全候補で合法な次手が無く終了(満杯)。打切りと区別 |
| ta_full_p2 | side_abs | f4 | 0/1 | 0-1 | mk_ta_p2 | av_fire_p2 | ta_full_p1 | audit | 要求H以内に保持ビームの全候補で合法な次手が無く終了(満杯)。打切りと区別 |
| mk_ta_trunc_p1 | side_abs | i1 | code | none |  |  | mk_ta_trunc_p2 | audit | search_extra の欠測理由 (Mask) |
| mk_ta_trunc_p2 | side_abs | i1 | code | none |  |  | mk_ta_trunc_p1 | audit | search_extra の欠測理由 (Mask) |
| search_version | meta | str | version | none |  | frame |  | meta | B3専用2個組・潜在火力探索の版 |
| ta_bound_p1 | side_abs | i1 | 0/1 | 0-1 |  | av_fire_p1 | ta_bound_p2 | feature | 予測不能時の楽観上限補完=1。側は攻撃側、Qは反対側の返せる量。側交換のみ、反転なし |
| ta_bound_p2 | side_abs | i1 | 0/1 | 0-1 |  | av_fire_p2 | ta_bound_p1 | feature | 予測不能時の楽観上限補完=1。側は攻撃側、Qは反対側の返せる量。側交換のみ、反転なし |
| ep_denominator_formula_p1 | side_abs | f4 | 個 | none | mk_ep_f_p1 | av_fire_p1 | ep_denominator_formula_p2 | feature | 式全段の消去数＋同じ完走候補の残存色数（発火補完込み）。候補終端不在は起点個数 |
| ep_denominator_formula_p2 | side_abs | f4 | 個 | none | mk_ep_f_p2 | av_fire_p2 | ep_denominator_formula_p1 | feature | 式全段の消去数＋同じ完走候補の残存色数（発火補完込み）。候補終端不在は起点個数 |
