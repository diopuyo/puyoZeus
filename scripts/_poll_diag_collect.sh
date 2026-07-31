#!/bin/bash
# diag_v29_mid 収集プロセス完了を60秒間隔でポーリングする使い捨てスクリプト。
# MSYS パイプ特殊文字問題 (feedback_msys_pipe_escape.md) 回避のためスクリプトファイル化。
set -u
i=0
while pgrep -f 'scripts._collect_1t.*diag_v29_mid' > /dev/null; do
  i=$((i + 1))
  echo "[poll ${i}] $(date +%H:%M:%S) still running"
  if [ "${i}" -ge 9 ]; then
    echo "POLL_LIMIT_REACHED"
    break
  fi
  sleep 60
done

if pgrep -f 'scripts._collect_1t.*diag_v29_mid' > /dev/null; then
  echo STILL_RUNNING
else
  echo PROCESS_DONE
fi
