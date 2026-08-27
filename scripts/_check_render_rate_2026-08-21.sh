#!/bin/bash
# 本番レンダの実効スループットを測る (2026-08-21)。
#
# 8並列で投入したが負荷が57まで上がっており (16コアに対して過剰)、
# 実測スループット (P=8 で 0.668 動画秒/壁秒) が出ていない疑いがある。
#
# 出力ファイルの成長量から実効速度を出す。
# 基準: 認識色重畳なしの1080p出力は動画1秒あたり約1MB (プローブ実測 300秒→300MB)。
DIR=/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer/data/verify/zenchi_render_2026-08-21
INTERVAL=60
BYTES_PER_VIDEO_SEC=1048576  # 約1MB/動画秒

sum() { du -cb "$DIR"/*.mp4 2>/dev/null | tail -1 | cut -f1; }

A=$(sum)
echo "[$(date +%H:%M:%S)] 合計 $((A / 1048576)) MB"
echo "  スレッド数:"
for p in $(pgrep -f visualize_advantage_overlay); do
  printf "    pid %s threads=%s\n" "$p" "$(ls /proc/"$p"/task 2>/dev/null | wc -l)"
done
cat /proc/loadavg

sleep "$INTERVAL"

B=$(sum)
D=$((B - A))
VSEC=$((D / BYTES_PER_VIDEO_SEC))
echo "[$(date +%H:%M:%S)] 合計 $((B / 1048576)) MB (+$((D / 1048576)) MB)"
echo "  ${INTERVAL}秒で動画 約${VSEC}秒ぶん処理 -> スループット $(echo "scale=4; $VSEC / $INTERVAL" | bc) 動画秒/壁秒"
echo "  残り: 合計7034秒のうち 約$((B / BYTES_PER_VIDEO_SEC))秒完了"
REMAIN=$((7034 - B / BYTES_PER_VIDEO_SEC))
if [ "$VSEC" -gt 0 ]; then
  ETA=$((REMAIN * INTERVAL / VSEC))
  echo "  残り約${REMAIN}動画秒 -> あと $((ETA / 3600))時間$(((ETA % 3600) / 60))分 (完了 $(date -d "+${ETA} seconds" +%H:%M))"
fi
cat /proc/loadavg
