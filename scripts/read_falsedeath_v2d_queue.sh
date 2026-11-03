#!/bin/bash
# run_limitedが数えるPythonのモジュール名だけを読み取る。停止・優先度変更はしない。
set -eu
for pid in $(pgrep -f 'python.*(-m scripts\.|-m pytest)' || true); do
  case "$(cat "/proc/$pid/comm" 2>/dev/null || true)" in
    python*)
      argv=()
      mapfile -d '' -t argv < "/proc/$pid/cmdline" 2>/dev/null || continue
      for ((i=0; i+1<${#argv[@]}; i++)); do
        if [ "${argv[i]}" = -m ]; then
          printf '%s %s\n' "$pid" "${argv[i+1]}"
          break
        fi
      done
      ;;
  esac
done
awk '/MemAvailable/ {print}' /proc/meminfo
