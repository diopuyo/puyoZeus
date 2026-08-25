#!/bin/bash
# 検収用フレーム抜き出し (区間A/B/C、境界イベント前後 + 30秒おき定点)
set -e
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer
FFMPEG=venv/lib/python3.12/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2
OUTDIR=data/verify/zenchi_probe_2026-08-21/frames
mkdir -p "$OUTDIR"

extract() {
  # $1 = 入力mp4, $2 = ローカル秒(出力mp4内の相対秒), $3 = 出力ファイル名
  "$FFMPEG" -y -ss "$2" -i "$1" -frames:v 1 -q:v 2 "$OUTDIR/$3" >/dev/null 2>&1
}

# ---- 区間A (real t = local t、start_sec=0) ----
A=data/verify/zenchi_probe_2026-08-21/regionA_0_300.mp4
# 30秒おき定点
for t in 0 30 60 90 120 150 180 210 240 270 299; do
  extract "$A" "$t" "A_grid_t${t}.png"
done
# 境界イベント (real t): 86.5-91.4 / 136.1-139.9 / 227.6-231.4 / 273.2-277.0
for pair in "84:A_ev1_pre" "88:A_ev1_mid" "93:A_ev1_post" \
            "134:A_ev2_pre" "137:A_ev2_mid" "142:A_ev2_post" \
            "225:A_ev3_pre" "229:A_ev3_mid" "234:A_ev3_post" \
            "271:A_ev4_pre" "275:A_ev4_mid" "280:A_ev4_post"; do
  t="${pair%%:*}"; name="${pair##*:}"
  extract "$A" "$t" "${name}.png"
done

# ---- 区間B (real t = local t + 3300) ----
B=data/verify/zenchi_probe_2026-08-21/regionB_3300_3600.mp4
for t in 0 30 60 90 120 150 180 210 240 270 299; do
  extract "$B" "$t" "B_grid_t$((t+3300)).png"
done
# 境界イベント (real t): 3325.9-3329.7 / 3356.2-3360.0 / 3412.2-3416.0
for pair in "24:B_ev1_pre" "28:B_ev1_mid" "33:B_ev1_post" \
            "54:B_ev2_pre" "58:B_ev2_mid" "63:B_ev2_post" \
            "110:B_ev3_pre" "114:B_ev3_mid" "119:B_ev3_post"; do
  t="${pair%%:*}"; name="${pair##*:}"
  extract "$B" "$t" "${name}.png"
done

# ---- 区間C (real t = local t + 6733) ----
C=data/verify/zenchi_probe_2026-08-21/regionC_6733_7033.mp4
for t in 0 30 60 90 120 150 180 210 240 270 299; do
  extract "$C" "$t" "C_grid_t$((t+6733)).png"
done
# 境界イベント (real t): 6755.4-6759.2 / 6788.5-6792.2 / 6832.9-6836.7 /
#                        6874.4-6878.2 / 6931.7-6935.5 / 6972.9-6976.7
for pair in "20:C_ev1_pre" "24:C_ev1_mid" "29:C_ev1_post" \
            "53:C_ev2_pre" "57:C_ev2_mid" "62:C_ev2_post" \
            "98:C_ev3_pre" "102:C_ev3_mid" "107:C_ev3_post" \
            "139:C_ev4_pre" "143:C_ev4_mid" "148:C_ev4_post" \
            "196:C_ev5_pre" "200:C_ev5_mid" "205:C_ev5_post" \
            "237:C_ev6_pre" "241:C_ev6_mid" "246:C_ev6_post"; do
  t="${pair%%:*}"; name="${pair##*:}"
  extract "$C" "$t" "${name}.png"
done

echo "done: $(ls "$OUTDIR" | wc -l) files"
