#!/bin/bash
# 8区間のレンダ結果を「セット1」「セット2」の2本に結合し、元動画の音声を載せる。
# (2026-08-21 作成 / 2026-08-22 音声載せを追加、ffprobe 依存を除去)
#
# ## なぜ再エンコードなしで済むか
#
# 区間4が セット境界 (t=3626.0s) でちょうど終わるように分割点を組んだため、
#   セット1 = 区間1〜4 (0 〜 3626.0s)
#   セット2 = 区間5〜8 (3626.0 〜 7033.6s)
# を並べて連結するだけで納品用の2本になる。全区間が同一の書き出し設定なので
# `-c copy` (無再エンコード) で連結できる。
#
# セット境界の根拠 (目視確認済み):
#   t≈3483s で WIN星 29→30 (のらすけ選手が30本先取、13連勝)
#   t≈3490〜3625s に約140秒のリザルト+キャラ再選択+ロード演出
#   t≈3626s でスコア0にリセット→「レディ」、2P名が「のらすけしんえいたい」に変化
#
# ## 検証 (受け入れ条件)
#
#   1. 各区間が rc=0 で完走している
#   2. 区間のフレーム数の合計 = 結合後のフレーム数 (欠落・重複ゼロ)
#   3. 2本の再生時間の合計 = 元動画 117分 (7033.6s) ± 数秒
#   4. 音声トラックが載っている
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer || exit 1

DIR=data/verify/zenchi_render_2026-08-21
OUT=data/verify/zenchi_delivery_2026-08-21
SRC=data/frames/video_zenchi_c0BQoMJwwQU.mp4
SET_BOUNDARY=3626.0
mkdir -p "$OUT"

# WSL には ffmpeg が入っていないので、プロジェクトにバンドルされている
# imageio-ffmpeg のバイナリを使う (CLAUDE.md: 動画 mux は imageio-ffmpeg バンドル)。
FFMPEG=$(PYTHONPATH=. ./venv/bin/python -c \
  "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())" 2>/dev/null)
if [ -z "$FFMPEG" ] || [ ! -x "$FFMPEG" ]; then
  echo "=== 中止: ffmpeg が見つからない ($FFMPEG) ==="
  exit 1
fi
echo "[ffmpeg] $FFMPEG"

# ffprobe はバンドルに含まれないので、フレーム数・尺は ffmpeg で全フレームを
# 読んで取る (`-f null -` なので再エンコードはしない)。
count_frames() {
  "$FFMPEG" -v error -stats -i "$1" -map 0:v:0 -f null - 2>&1 |
    tr '\r' '\n' | grep -oE "frame= *[0-9]+" | tail -1 | grep -oE "[0-9]+"
}
duration_str() {
  "$FFMPEG" -v error -stats -i "$1" -map 0:v:0 -f null - 2>&1 |
    tr '\r' '\n' | grep -oE "time=[0-9:.]+" | tail -1 | cut -d= -f2
}
stream_info() {
  "$FFMPEG" -hide_banner -i "$1" 2>&1 |
    grep -oE "Stream #0:[0-9]+.*: (Video|Audio): [a-zA-Z0-9]+" | tr '\n' ' '
}

echo "=== 結合 start $(date +%F_%T) ==="

echo "--- 1. 各区間の完走確認 ---"
NG=0
TOTAL_SEG=0
for i in 01 02 03 04 05 06 07 08; do
  RC=$(grep -oE "done rc=[0-9]+" "logs/zenchi_render_2026-08-21/seg$i.log" 2>/dev/null | tail -1)
  F=$(ls "$DIR"/seg${i}_*.mp4 2>/dev/null | head -1)
  if [ -z "$F" ]; then
    echo "  seg$i: **出力が無い**"
    NG=1
    continue
  fi
  NB=$(count_frames "$F")
  DUR=$(duration_str "$F")
  printf "  seg%s: %s frames=%s dur=%s size=%sMB\n" \
    "$i" "${RC:-rc不明}" "${NB:-?}" "${DUR:-?}" "$(du -m "$F" | cut -f1)"
  [ -n "$NB" ] && TOTAL_SEG=$((TOTAL_SEG + NB))
  [ "$RC" = "done rc=0" ] || NG=1
done

if [ "$NG" -ne 0 ]; then
  echo "=== 中止: 完走していない区間がある ==="
  echo "    欠けたまま結合すると納品物が壊れるので、原因を確認すること。"
  exit 1
fi
echo "  区間のフレーム数合計: $TOTAL_SEG"

echo "--- 2. セット1 (区間1〜4) を結合 ---"
: > "$OUT/set1.txt"
for i in 01 02 03 04; do
  echo "file '$(realpath "$(ls "$DIR"/seg${i}_*.mp4 | head -1)")'" >> "$OUT/set1.txt"
done
"$FFMPEG" -y -v error -f concat -safe 0 -i "$OUT/set1.txt" -c copy "$OUT/zenchi_set1.mp4"
echo "  rc=$?"

echo "--- 3. セット2 (区間5〜8) を結合 ---"
: > "$OUT/set2.txt"
for i in 05 06 07 08; do
  echo "file '$(realpath "$(ls "$DIR"/seg${i}_*.mp4 | head -1)")'" >> "$OUT/set2.txt"
done
"$FFMPEG" -y -v error -f concat -safe 0 -i "$OUT/set2.txt" -c copy "$OUT/zenchi_set2.mp4"
echo "  rc=$?"

echo "--- 4. 元動画の音声を載せる (user 指示 2026-08-22) ---"
#
# 元動画は h264 + opus。オーバーレイ側のスクリプトは音声を扱わないので、
# 結合後の映像に元動画の該当区間の音声を後付けする。
# 映像は再エンコードせず (-c:v copy)、音声は mp4 互換のため aac に変換する
# (opus は mp4 コンテナでは再生互換性が低い環境がある)。
#
# 同期の根拠: overlay 出力は実時間を保つ設計 (`effective_fps = fps/stride` で
# 書き出し) なので元動画とタイムラインが1:1。同じ秒数から切り出せば合う。
echo "  セット1 (元動画 0 〜 ${SET_BOUNDARY}s)"
"$FFMPEG" -y -v error -i "$OUT/zenchi_set1.mp4" \
  -ss 0 -to "$SET_BOUNDARY" -i "$SRC" \
  -map 0:v:0 -map 1:a:0 -c:v copy -c:a aac -b:a 192k -shortest \
  "$OUT/zenchi_set1_audio.mp4"
echo "    rc=$?"

echo "  セット2 (元動画 ${SET_BOUNDARY} 〜 末尾)"
"$FFMPEG" -y -v error -ss "$SET_BOUNDARY" -i "$SRC" -i "$OUT/zenchi_set2.mp4" \
  -map 1:v:0 -map 0:a:0 -c:v copy -c:a aac -b:a 192k -shortest \
  "$OUT/zenchi_set2_audio.mp4"
echo "    rc=$?"

echo "--- 5. 検証 ---"
S1=$(count_frames "$OUT/zenchi_set1.mp4")
S2=$(count_frames "$OUT/zenchi_set2.mp4")
echo "  結合後のフレーム数: set1=$S1 / set2=$S2 / 合計=$((S1 + S2))"
echo "  区間の合計:         $TOTAL_SEG"
if [ "$TOTAL_SEG" -gt 0 ] && [ "$TOTAL_SEG" -eq "$((S1 + S2))" ]; then
  echo "  -> フレーム数一致 (欠落・重複ゼロ) **合格**"
else
  echo "  -> **フレーム数が一致しない**。欠落または重複がある"
fi
echo "  再生時間: set1=$(duration_str "$OUT/zenchi_set1.mp4") / set2=$(duration_str "$OUT/zenchi_set2.mp4")"
echo "            (元動画は 7033.6s = 1:57:13.6)"

echo "  --- 音声つき2本 ---"
for f in "$OUT/zenchi_set1_audio.mp4" "$OUT/zenchi_set2_audio.mp4"; do
  if [ ! -f "$f" ]; then
    echo "    $(basename "$f"): **生成されていない**"
    continue
  fi
  echo "    $(basename "$f"): $(stream_info "$f")"
  echo "      尺=$(duration_str "$f")  frames=$(count_frames "$f")  $(du -m "$f" | cut -f1)MB"
done
echo "  ※ 音声と映像のずれは、発火エフェクトと効果音の一致で目視確認する (検収項目)"

echo "=== 完了 $(date +%F_%T) ==="
ls -l "$OUT"/*.mp4
