#!/bin/bash
# 動画fd読み込み位置の増加を90秒測り、実処理速度(%/分)を出す
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer || exit 1
snap() {
  for pid in $(pgrep -f 'venv/bin/python -m scripts.collect_indicators_v2'); do
    out=$(tr '\0' ' ' < /proc/$pid/cmdline | grep -oE 'study/[a-z0-9_]+\.csv')
    for fd in /proc/$pid/fd/*; do
      tgt=$(readlink "$fd" 2>/dev/null)
      case "$tgt" in
        *.mp4)
          fn=$(basename "$fd")
          pos=$(grep '^pos:' "/proc/$pid/fdinfo/$fn" 2>/dev/null | grep -oE '[0-9]+')
          echo "$out $pos"
          ;;
      esac
    done
  done
}
declare -A before
while read -r name pos; do before[$name]=$pos; done < <(snap)
sleep 90
while read -r name pos; do
  b=${before[$name]:-0}
  d=$(( (pos - b) / 1000000 ))
  echo "$name: +${d}MB/90s"
done < <(snap)
