#!/bin/bash
# 実行中プロセスの読取位置だけを確認する。新しいPythonは起動しない。
free -m
for PID in $(pgrep -f '[p]ython.*(scripts.prefire_v5_ledger|scripts.run_prefire_replay_20260930)'); do
  ps -p "$PID" -o pid=,etime=,rss=,args=
  for FD in /proc/"$PID"/fd/*; do
    TARGET=$(readlink "$FD" 2>/dev/null || true)
    case "$TARGET" in
      *records/*.jsonl.gz)
        POS=$(awk '/^pos:/ {print $2}' /proc/"$PID"/fdinfo/"${FD##*/}")
        SIZE=$(stat -c %s "$TARGET")
        echo "  compressed_bytes_read=$POS/$SIZE source=${TARGET##*/}"
        ;;
    esac
  done
done
