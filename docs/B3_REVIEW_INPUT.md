# B3 レビュー動画の入力形式 (b3_review_v1)

実装: `src/b3_review_input.py` (形式・変換)、`src/b3_review_panel.py` (データ帯)、
`src/b3_review_video.py` (動画・CSV・検収・スマホ分割)、`scripts/b3_review_render.py` (CLI)。
既存のレビュー生成器 (review_data_panel / visualize_advantage_overlay) には手を入れない。
学習・評価側とは疎結合で、モデルや評価器には依存しない。

## 1. 入力 = 1 行 1 評価時点の CSV (UTF-8、`t_sec` 昇順)

- `t_sec` は **元動画の先頭からの秒** (= `frame_idx / fps`)。動画のフレーム時刻 t には「t 以前で最後の行」を表示する (最初の行より前は「評価前」と出す)。
- 空欄 = 未観測 (画面には `--`)。**0 と欠測を区別する**。補間・前方埋めはしない (グラフも欠測区間は線を切る)。
- 確率はすべて **1P が勝つ確率** (0〜1)。検査 `validate_rows` が範囲・昇順・語彙を確認し、不正なら読込を拒否する。

### 共通列
| 列 | 内容 |
|---|---|
| t_sec / frame_idx / video_id / game_idx | 時刻・フレーム・動画ID・試合番号 |
| phase | 局面 0=序盤 1=中盤 2=終盤 (user 定義、B3 spec の `phase`) |
| p_static / p_exchange | F_static / F_exchange の 1P 勝率 (該当しない側は空) |
| p_selected | 状態選択後の採用値 |
| p_display | 表示値 (平滑後) |
| p_ref / p_ref_display | 比較基準 R (現行本番) の生値 / 表示値。差が見えるよう並べて出す |
| mode | `static` / `exchange` / `none` (状態選択) |
| state_reason | 状態選択の根拠 (自由文) |
| missing_code / missing_note | 行全体の欠測理由 (B3 spec の Mask コード 0〜9) と注記 |
| sl_rate | 台帳 v2 の換算レート (点/個) |
| won_p1 | 任意。最終勝者 (1P 勝ち=1)。レビューで答え合わせ用 |

### 側別列 (`p1_*` / `p2_*`)
**側の意味は B3 の `ta_*_p1/p2` 列に従う: 列の側 = 攻撃側 a、値は受け側 b についての量。**
`p1_*` は「1P 攻撃 (a) → 受け = 2P (b)」、`p2_*` は逆。台帳 (`sl_*`)・アイコン (`icon_*`) は各側自身の量。

| 列 (側接尾辞なしの名前) | 内容 | b3_rows の元列 |
|---|---|---|
| H_med / H_fast | 置ける総手数 (中央値版 / 速い版) | ta_H_med / ta_H_fast |
| hand_med / hand_fast | 1 手時間 (秒) | ta_hand_med_sec / ta_hand_fast_sec |
| req_h / done_h / trunc | 探索要求手数 / 探索済み / 打切り | ta_req_h / ta_done_h / ta_trunc |
| a_rem / b_busy / window | a の残り連鎖時間 / b の操作不能 / b が置ける窓 (秒) | ta_a_rem_sec / ta_b_busy / ta_window |
| active | a が連鎖中 | ta_active |
| Q_med / Q_fast | b の H 手以内の近未来火力 (個) | ta_q_med / ta_q_fast |
| I_sl / I_pred / I_icon | b の受け量: 台帳 v2 (主入力) / 予測完走 / アイコン (照合用) | ta_i_sl / ta_i_pred / ta_i_icon |
| M / recv / send | 余地 Q−I、残受け、残送り | ta_m_sl_med (無ければ ta_m_pred_med) / ta_recv_pred_med / ta_send_pred_med |
| sl_pending / sl_generated / sl_offset / sl_dropped / sl_carry | 台帳 v2: 予告・累計生成・累計相殺・累計着弾・端数繰越 | sl_* |
| icon_corr / icon_raw | 画面アイコン読みの総数 (補正済 / 生) | icon_corr_total / icon_raw_total |
| mk_ta / mk_sl | 時間軸 / 台帳の欠測コード | mk_ta / mk_sl |
| death_state | `none` / `suspect` / `proven_dead` | (死亡証明の出力。pred 側で渡す) |
| death_basis | 死亡証明の根拠 (自由文。例: 相殺不能の予告72個以上) | 同上 |

### 画面・CSV に追加される計算列 (入力には書かない)
`frame_out` (出力フレーム番号)、`p_diff_vs_ref` (B3 表示 − R 表示)、`p1/p2_ledger_icon_gap` (台帳 v2 の予告 − アイコン補正値。どちらか欠測、または両方 0 のときは空)、`*_missing_name` (欠測コードの日本語名)。

## 2. 学習・評価側の出力からの変換口

```python
from src.b3_review_input import from_b3_rows, from_b3_rows_iter, save_review_input
row = from_b3_rows(b3_rows_row, pred)         # b3_rows の 1 行 (列名は B3_COLUMN_SPEC) + 予測 dict
rows = from_b3_rows_iter(rows_iter, preds_iter)  # 行数不一致は例外
save_review_input(Path("rows.csv"), rows)
```
`pred` のキー: `p_static p_exchange p_selected p_display p_ref p_ref_display mode state_reason missing_code missing_note` と、側別の `p1_death_state p1_death_basis p2_death_state p2_death_basis`。**列に無い値は None のまま** (捏造しない)。変換の語幹がすべて実在の b3_rows 列であることは `tests/test_b3_review_input.py` が検査する。

## 3. 出力

`scripts/b3_review_render.py` (WSL venv で実行):
```
PYTHONPATH=. ./venv/bin/python -m scripts.b3_review_render --video V.mp4 --input rows.csv \
  --out-dir D:/puyo_analyzer/review_B3_xxx --start-sec S --end-sec E --mobile
```
- `<stem>.mp4` (PC 用、crf20、元音声つき)、`<stem>_frames.csv` (画面に出した値と同じものをフレーム別に)、`<stem>_frames.selfcheck.json` / `<stem>_complete.json` (自己検収)。
- `--mobile`: 幅1280・crf28・veryfast・aac96k・+faststart → 270 秒分割。映像上限は「1 本が 30MiB 以内」になるよう逆算 (約 780kbps)。**各本のサイズを検査し、超過なら例外**。

## 4. 自己検収 (user に出す前)
`selfcheck.json` の母数つき被覆: 区間内の評価行数、先頭の未被覆秒、5 秒超の行間欠落、各確率列の非欠測行数、状態・死亡証明の行数。
- 試合の開始〜終了を通しで切る。区間は試合境界 (matches/labels) から決める。
- 「最初の行より前」「長い欠落」が 0 でなければ、原因 (評価側の行生成) を直してから user に出す。
- 件数だけで合格としない (母数と並べる)。
