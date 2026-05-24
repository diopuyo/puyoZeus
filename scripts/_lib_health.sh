#!/bin/bash
# scripts/_lib_health.sh — source して使う健康監視 helper
# 使用例:
#   source scripts/_lib_health.sh
#   init_health phase_l
#   run_step cut "$cmd_str"      # step 単位 (= idempotent + skip)
#   run_item eval "v89m3" cmd... # item 単位 (= fail-tolerant)
#   finalize_health $rc

_HEALTH_ROOT=""
_HEALTH_START=""
_HEALTH_NAME=""

# 初期化: logs/<name>/_status.jsonl + start event
init_health() {
  local name="$1"
  _HEALTH_NAME="$name"
  _HEALTH_ROOT="logs/${name}"
  _HEALTH_START=$(date +%s)
  mkdir -p "$_HEALTH_ROOT"
  : > "${_HEALTH_ROOT}/_status.jsonl"
  echo "{\"phase\":\"$name\",\"event\":\"start\",\"ts\":$_HEALTH_START}" \
    >> "${_HEALTH_ROOT}/_status.jsonl"
  echo "[health] init $name @ $(date)"
}

# step 単位: idempotent (= done.flag あれば skip) + try/catch
# 使用: run_step <step_name> <command...>
run_step() {
  local step="$1"; shift
  local flag="${_HEALTH_ROOT}/step_${step}_done.flag"
  if [ -f "$flag" ]; then
    echo "[health][skip] step=$step (flag exists)"
    return 0
  fi
  local ts0=$(date +%s)
  echo "[health][start] step=$step @ $(date)"
  "$@"; local rc=$?
  local ts1=$(date +%s)
  local event="ok"; [ $rc -ne 0 ] && event="fail"
  echo "{\"step\":\"$step\",\"event\":\"$event\",\"rc\":$rc,\"dur\":$((ts1-ts0)),\"ts\":$ts1}" \
    >> "${_HEALTH_ROOT}/_status.jsonl"
  if [ $rc -eq 0 ]; then
    echo "rc=0 ts=$ts1" > "$flag"
    echo "[health][done] step=$step dur=$((ts1-ts0))s"
  else
    touch "${_HEALTH_ROOT}/step_${step}_failed.flag"
    echo "[health][fail] step=$step rc=$rc"
  fi
  return $rc
}

# item 単位: fail-tolerant (= rc 無視で次へ続行)
# 使用: run_item <step> <item_id> <command...>
run_item() {
  local step="$1" item="$2"; shift 2
  local flag="${_HEALTH_ROOT}/item_${step}_${item}.flag"
  if [ -f "$flag" ]; then
    echo "[health][skip] item=${step}/${item}"
    return 0
  fi
  local ts0=$(date +%s)
  "$@"; local rc=$?
  local ts1=$(date +%s)
  if [ $rc -eq 0 ]; then
    echo "rc=0" > "$flag"
  fi
  echo "{\"step\":\"$step\",\"item\":\"$item\",\"rc\":$rc,\"dur\":$((ts1-ts0)),\"ts\":$ts1}" \
    >> "${_HEALTH_ROOT}/_status.jsonl"
  return 0  # fail-tolerant: 常に 0
}

# 全 step 完了 + flag 出力
finalize_health() {
  local total_rc="${1:-0}"
  local ts=$(date +%s)
  local elapsed=$((ts - _HEALTH_START))
  echo "{\"event\":\"done\",\"total_rc\":$total_rc,\"ts\":$ts,\"elapsed\":$elapsed}" \
    >> "${_HEALTH_ROOT}/_status.jsonl"
  if [ "$total_rc" -eq 0 ]; then
    echo "[health][ALL_DONE] $_HEALTH_NAME elapsed=${elapsed}s @ $(date)" \
      | tee "${_HEALTH_ROOT}/all_done.flag"
  else
    echo "[health][ALL_FAIL] $_HEALTH_NAME total_rc=$total_rc elapsed=${elapsed}s @ $(date)" \
      | tee "${_HEALTH_ROOT}/all_fail.flag"
  fi
}

# 空 input 行 skip (= 今回真因再発防止)
# 使用: read_csv_line で IFS='|' のあと使う。 引数 1 個目が空ならスキップ示唆
is_empty_first() {
  [ -z "$1" ]
}
