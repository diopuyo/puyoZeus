#!/bin/bash
# 自分の未起動待機シェルだけを排他下で止め、原票は残して確認間隔を短くする。
set -eu
cd /mnt/d/puyo_analyzer/wt_evalset
OUT=logs/eval_set/falsedeath_fix/v2b
worker_pid="${1:?待機シェルのPIDを指定する}"
worker_log="${2:?新規ログ名を指定する}"
[[ "$worker_pid" =~ ^[0-9]+$ ]]
[[ "$worker_log" =~ ^[a-zA-Z0-9_]+\.log$ ]]
test ! -e "$OUT/$worker_log"
exec 9> "$OUT/launch.lock"
flock 9
test "$(cat /proc/$worker_pid/comm)" = bash
test "$(readlink /proc/$worker_pid/cwd)" = /mnt/d/puyo_analyzer/wt_evalset
test "$(tr '\0' ' ' < /proc/$worker_pid/cmdline)" = 'bash scripts/run_falsedeath_v2_ladder.sh v2b '
for child in $(pgrep -P "$worker_pid" || true); do
  case "$(cat /proc/$child/comm)" in python*) echo '再生開始済みのため再起動しない'; exit 1;; esac
done
kill "$worker_pid"
flock -u 9
exec 9>&-
export FALSEDEATH_WORKER_LOG="$worker_log"
setsid -f bash scripts/run_falsedeath_v2_ladder.sh v2b
