#!/bin/bash
# レビュー#49 再生成 v2 (2026-07-25): video_c34 game1 の境界修正版。
#
# 前回 (scripts/_run_full_game1_c34_2026-07-25.sh) は開始 465.6s (score0直読み)
# だったが、winners_probe strict 境界 (data/verify/winners_probe_2026-07-23/
# video_c34.json game_abs_idx=1: start_sec=472.0) と 6.4 秒ずれていた。
# 実画面ダンプ(465.6/468.0/468.7/469.8/470.5/471.5/472.0s)で確認した結果:
#   - 465.6-471.5s は前試合(game0)の「ばたんきゅー」演出 → 得点リセット →
#     ロード演出(暗転盤面+シルエットキャラ+ダミー加算スコア+装飾ぷよアイコン)
#     であり、実試合ではない。
#   - 472.0s で両側とも実際のツモ落下・盤面形成が視認できる(winners_probe と一致)。
# write_trace (data/verify/write_trace/c34_1P.jsonl 468.73s, c34_2P.jsonl 468.77s)
# で「ロード演出の装飾ぷよアイコン(青主体)」が P1_merge_diff_only/P2_infer_placement
# 経由で confirmed_board に直接書き込まれていたことを確認 (= user 指摘の「青2個」の正体)。
# 短い試しレンダ (472.0-482.0s, force_in_match True/False 両方) で検証した結果:
#   - force_in_match は True/False どちらでも 465.6-471.5s の区間で同じ誤書き込みが
#     発生した (MatchStateDetector 自体が盤面背景の暗さだけで in-match 判定するため、
#     ロード演出の暗い盤面パネルを in-match と誤判定し、force_in_match の値に関係なく
#     score_zero によるハード凍結が外れた瞬間に同じ経路で誤認識が起こる)。
#     → force_in_match は原因ではなく、境界(465.6→472.0)の修正のみで解消する。
#   - 472.0s 開始のレンダでは「青2個」は再現せず、1P/2P とも実ツモ色と一致した認識結果。
#     (2P 側で CHAIN 中の estimated_board 表示に一瞬だけ孤立した"B"ゴースト cell が
#     出たが、STABLE 復帰で消える一過性の表示専用挙動であり、評価に使う confirmed_board
#     には影響しない別件。今回のバグ(前試合残骸の書き込み)とは無関係)。
# end_sec=511.8 (game2 の score0 直読み、raw) は据え置き: game1→game2 の同様なロード
# 演出区間(511.8-518.0 相当、winners_probe game_abs_idx=2 start=518.0) が始まる前で
# 切れており、末尾側は元から安全 (対称バグの再発なし)。
set -u
PROJ_DIR="/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer"; cd "${PROJ_DIR}" || exit 1
OUT_DIR="data/verify/review_video_new_2026-07-25"
mkdir -p "${OUT_DIR}"
RAW="${OUT_DIR}/advantage_recog_c34_game1_full_score0to0_v2.mp4"
H264="${OUT_DIR}/advantage_recog_c34_game1_full_score0to0_v2_h264.mp4"

echo "[start] $(date)"
PYTHONPATH=. nice -n 10 ./venv/bin/python -m scripts._zap_1t \
  --video data/frames/video_c34.mp4 \
  --out "${RAW}" \
  --start-sec 472.0 --end-sec 511.8 --warmup-sec 30 \
  --show-recognition --landing-observed-color
echo "[render done] $(date)"

FF=$(./venv/bin/python -c 'import imageio_ffmpeg;print(imageio_ffmpeg.get_ffmpeg_exe())')
"$FF" -y -i "${RAW}" -c:v libx264 -preset medium -crf 20 -pix_fmt yuv420p -movflags +faststart "${H264}"
echo "[h264 done] $(date)"
ls -la "${RAW}" "${H264}"
echo "[all done] $(date)"
