#!/bin/bash
# long_improve_v2.py が何らかの理由で死んだら自動再起動するラッパー
# 40時間モードなのでスクリプト自身が時間管理して自然終了する
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer
# src パッケージを import できるようプロジェクトルートを PYTHONPATH へ
export PYTHONPATH="/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer:${PYTHONPATH}"
# FAST_MODE: Phase 2 (アーキ探索) スキップ + stagnation_limit=2 で時短
# data/disable_fast_mode が存在すれば無効化（fallback）
if [ ! -f data/disable_fast_mode ]; then
  export FAST_MODE=1
fi
RESTART_COUNT=0
MAX_RESTARTS=30
LOG=data/long_improve_v2_stdout.log

# 排他ロック: 二重起動を防ぐ (flock で自プロセスに対し非ブロッキングロック)
# 先行インスタンスが生きていれば即 exit する。終了時に自動解放。
LOCKFILE=data/wrapper.lock
exec 9>"$LOCKFILE"
if ! flock -n 9; then
  echo "[wrapper] $(date '+%Y-%m-%d %H:%M:%S') another wrapper holds $LOCKFILE; exit" >> $LOG
  exit 0
fi
echo "[wrapper] $(date '+%Y-%m-%d %H:%M:%S') acquired lock pid=$$" >> $LOG

# Watchdog 相互監視サブプロセス。
# watchdog.ps1 が data/watchdog_heartbeat の mtime を120秒毎に更新するので、
# 10分以上古くなったら Windows 側で bootstrap スクリプトを発火して再起動する。
WATCHDOG_HEARTBEAT=data/watchdog_heartbeat
WATCHDOG_BOOTSTRAP='/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer/scripts/watchdog_bootstrap.ps1'
PWSH='/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe'
(
  while true; do
    sleep 180
    if [ -f "$WATCHDOG_HEARTBEAT" ]; then
      HB_AGE=$(( $(date +%s) - $(stat -c %Y "$WATCHDOG_HEARTBEAT") ))
      if [ "$HB_AGE" -gt 600 ]; then
        echo "[wrapper-supervisor] $(date '+%Y-%m-%d %H:%M:%S') watchdog heartbeat stale (${HB_AGE}s > 600s); respawning" >> $LOG
        "$PWSH" -ExecutionPolicy Bypass -NoProfile -WindowStyle Hidden -File "$WATCHDOG_BOOTSTRAP" >/dev/null 2>&1 &
      fi
    else
      echo "[wrapper-supervisor] $(date '+%Y-%m-%d %H:%M:%S') no heartbeat file; bootstrapping" >> $LOG
      "$PWSH" -ExecutionPolicy Bypass -NoProfile -WindowStyle Hidden -File "$WATCHDOG_BOOTSTRAP" >/dev/null 2>&1 &
    fi
  done
) &
SUPERVISOR_PID=$!
trap "kill $SUPERVISOR_PID 2>/dev/null" EXIT INT TERM
echo "[wrapper] $(date '+%Y-%m-%d %H:%M:%S') supervisor subprocess pid=$SUPERVISOR_PID" >> $LOG
while [ $RESTART_COUNT -lt $MAX_RESTARTS ]; do
  echo "[wrapper] $(date '+%Y-%m-%d %H:%M:%S') start attempt #$RESTART_COUNT" >> $LOG
  ./venv/bin/python scripts/long_improve_v2.py
  EXIT=$?
  echo "[wrapper] $(date '+%Y-%m-%d %H:%M:%S') exited code=$EXIT, restart in 60s" >> $LOG
  RESTART_COUNT=$((RESTART_COUNT + 1))
  # 自然終了（時間キャップ到達）は exit 0 で戻るはず
  if [ $EXIT -eq 0 ]; then
    echo "[wrapper] clean exit, stopping wrapper" >> $LOG
    break
  fi
  # 異常終了は milestone.jsonl に fatal イベントを書いて通知ループに拾わせる
  ./venv/bin/python -c "
import json, datetime
with open('data/milestones.jsonl', 'a', encoding='utf-8') as f:
    rec = {
        'ts': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'kind': 'fatal',
        'summary': f'wrapper restart #$RESTART_COUNT (python exit code=$EXIT)',
        'restart_count': $RESTART_COUNT,
        'exit_code': $EXIT,
    }
    f.write(json.dumps(rec, ensure_ascii=False) + '\n')
" 2>>$LOG
  sleep 60
done
echo "[wrapper] wrapper finished at $(date '+%Y-%m-%d %H:%M:%S')" >> $LOG
