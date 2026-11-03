#!/bin/bash
# 起動口の排他で、空き一枠に複数ワーカーが同時起動することを防ぐ。
falsedeath_is_controller() {
  # 呼出側が指定し、役割を確認した制御モジュールだけを除外する。既定は除外なし。
  local pid="$1" index module
  local -a argv=()
  mapfile -d '' -t argv < "/proc/$pid/cmdline" 2>/dev/null || return 1
  for ((index=0; index+1<${#argv[@]}; index++)); do
    if [ "${argv[index]}" = -m ]; then
      # 2026-10-03 b2_recollect_run.pyを確認。--oneなしは待機・割当だけ。
      # --oneは子の収集を待つ間だけ除外し、子終了後の監査は数える。
      if [ "${argv[index+1]}" = scripts.b2_recollect_run ]; then
        case " ${argv[*]} " in
          *" --one "*) pgrep -P "$pid" -f 'python.*-m scripts[.]b2_recollect_worker' >/dev/null && return 0; return 1;;
          *) return 0;;
        esac
      fi
      # sealed_eval_prepare.pyも確認済み。DL・境界検出を子へ委譲する親だけ除外。
      [ "${argv[index+1]}" != scripts.sealed_eval_prepare ] || return 0
      # sealed_eval_run.pyのfullは準備待ちと子の割当だけ。pilotは配列比較があるため数える。
      if [ "${argv[index+1]}" = scripts.sealed_eval_run ]; then
        case " ${argv[*]} " in *" --phase full "*) return 0;; esac
      fi
      for module in ${FALSEDEATH_CONTROLLER_MODULES:-}; do
        [ "${argv[index+1]}" != "$module" ] || return 0
      done
      return 1
    fi
  done
  return 1
}

run_limited() {
  local logfile="$1"
  shift
  local available jobs child
  exec 9> "$OUT/launch.lock"
  while true; do
    flock 9
    available=$(awk '/MemAvailable/ {print $2}' /proc/meminfo)
    jobs=0
    for pid in $(pgrep -f 'python.*(-m scripts\.|-m pytest|scripts/)' || true); do
      case "$(cat "/proc/$pid/comm" 2>/dev/null || true)" in
        python*) if ! falsedeath_is_controller "$pid"; then jobs=$((jobs+1)); fi;;
      esac
    done
    if [ "$available" -ge 6291456 ] && [ "$jobs" -lt 6 ]; then
      "$@" > "$logfile" 2>&1 &
      child=$!
      sleep 1
      flock -u 9
      wait "$child"
      return $?
    fi
    flock -u 9
    sleep "${FALSEDEATH_POLL_SECONDS:-30}"
  done
}
