#!/usr/bin/env bash
# watch_ab_and_gen_viz.sh
#
# A/B 評価完走を監視し、完走次第 B0/B1/B2/B3 x 4 動画 = 16 本の viz を自動生成する。
# 朝のユーザー起床時に比較動画が揃っている状態を目指す。
#
# 起動方法 (CLAUDE.md プロセス管理ルール):
#   wsl -d Ubuntu -- bash -c "cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer && \
#     setsid -f bash scripts/watch_ab_and_gen_viz.sh < /dev/null"
#
# ログ: logs/watcher.log
# 朝報告: data/match_clips_viz/_morning_report.md

set -euo pipefail

# ============================
# 定数定義
# ============================
PROJ_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${PROJ_ROOT}/venv/bin/python"
VIZMODULE="scripts.visualize_recognition"
LOG="${PROJ_ROOT}/logs/watcher.log"
VIZ_DIR="${PROJ_ROOT}/data/match_clips_viz"
REPORT="${VIZ_DIR}/_morning_report.md"
AB_SUMMARY_GLOB="${PROJ_ROOT}/data/verify/stable_cell_acc/ab_summary_*.json"

# viz 並列数 (GPU RAM: 8GB, 1 プロセス ~2-3 GB 程度想定 -> 並列 3 が上限)
VIZ_PARALLEL=3

# 評価対象動画: "仮称:実ファイル名ステム:動画ID"
# 新切り出し動画 (= 5 秒バッファ付き、 user 直前体験と整合)
# path = data/match_clips/${VID_ID}/${VID_STEM}.mp4
EVAL_VIDEOS=(
    "v40_match01:v40_match01:v40"
    "v57_match01:v57_match01:v57"
    "v89_match01:v89_match01:v89"
    "v29_match02:v29_match02:v29"
)

# 仮説定義: "名前:enable_warmup_guard(0/1):bg_fp_force_max_puyo(空=省略)"
# B0: baseline (warmup OFF, MAX_PUYO=144 デフォルト)
# B1: warmup guard ON, MAX_PUYO=144
# B2: warmup OFF, MAX_PUYO=4
# B3: warmup ON, MAX_PUYO=4
VARIANTS=(
    "B0:0:"
    "B1:1:"
    "B2:0:4"
    "B3:1:4"
)

# ============================
# ログ初期化 (LOG のみ書く。stdout は /dev/null へ)
# ============================
mkdir -p "${PROJ_ROOT}/logs"
mkdir -p "${VIZ_DIR}"

_log() {
    echo "$*" >> "${LOG}"
}

_log "[watcher] 起動 $(date '+%Y-%m-%d %H:%M:%S JST')"
_log "[watcher] PROJ_ROOT=${PROJ_ROOT}"
_log "[watcher] A/B 完走監視開始 (60s 間隔ポーリング)"

# ============================
# Step 1: A/B 完走監視
# ============================
# 最大 4 時間待機 (= 03:00+4h = 07:00 超えたら諦め)
WAIT_LIMIT=14400
ELAPSED=0
SLEEP_INTERVAL=60

SUMMARY_FILE=""
while true; do
    SUMMARY_FILE=$(ls ${AB_SUMMARY_GLOB} 2>/dev/null | head -1 || true)
    if [ -n "${SUMMARY_FILE}" ]; then
        _log "[watcher] A/B 完走検知 $(date '+%Y-%m-%d %H:%M:%S JST')"
        _log "[watcher] summary: ${SUMMARY_FILE}"
        break
    fi

    ELAPSED=$((ELAPSED + SLEEP_INTERVAL))
    if [ ${ELAPSED} -ge ${WAIT_LIMIT} ]; then
        _log "[watcher] タイムアウト: ${WAIT_LIMIT}s 待機したが A/B 完走を検知できなかった"
        _log "[watcher] ab_summary なしで viz 生成を試みる"
        break
    fi

    _log "[watcher] 待機中... ${ELAPSED}s / ${WAIT_LIMIT}s ($(date '+%H:%M:%S'))"
    sleep ${SLEEP_INTERVAL}
done

# ============================
# Step 2: viz 生成ジョブ投入
# ============================
_log "[watcher] viz 生成開始 $(date '+%Y-%m-%d %H:%M:%S JST')"

# ジョブ追跡用配列
JOB_PIDS=()
JOB_LABELS=()
JOB_OUTPUTS=()

# viz 1 本を起動する関数
_run_viz() {
    local label="$1"      # 出力ファイル名 prefix (例: B1_v40_match01)
    local video_path="$2" # フルパス
    local extra_args="$3" # 追加 argparse 引数

    local out_path="${VIZ_DIR}/${label}.mp4"
    local viz_log="${PROJ_ROOT}/logs/viz_${label}.log"

    if [ -f "${out_path}" ]; then
        _log "[watcher] skip (already exists): ${label}.mp4"
        return 0
    fi

    _log "[watcher] viz 開始: ${label} -> ${out_path}"
    # PYTHONPATH をセットして python -m 経由で起動
    # shellcheck disable=SC2086
    PYTHONPATH="${PROJ_ROOT}" \
        "${PYTHON}" -m "${VIZMODULE}" \
        --video "${video_path}" \
        --output "${out_path}" \
        ${extra_args} \
        > "${viz_log}" 2>&1 &

    local pid=$!
    _log "[watcher] PID=${pid} label=${label}"
    JOB_PIDS+=("${pid}")
    JOB_LABELS+=("${label}")
    JOB_OUTPUTS+=("${out_path}")
}

# 並列スロット制御: 実行中が VIZ_PARALLEL 以上なら完了まで待つ
_wait_for_slot() {
    while true; do
        local running=0
        local p
        for p in "${JOB_PIDS[@]:-}"; do
            if [ -n "${p}" ] && kill -0 "${p}" 2>/dev/null; then
                running=$((running + 1))
            fi
        done
        if [ ${running} -lt ${VIZ_PARALLEL} ]; then
            break
        fi
        sleep 10
    done
}

# 全 16 本をエンキュー (バリアント x 動画)
for VARIANT_DEF in "${VARIANTS[@]}"; do
    IFS=':' read -r VARIANT_NAME WARMUP_FLAG MAX_PUYO <<< "${VARIANT_DEF}"

    # argparse 引数を組み立て
    EXTRA_ARGS=""
    if [ "${WARMUP_FLAG}" = "1" ]; then
        EXTRA_ARGS="${EXTRA_ARGS} --enable-warmup-guard"
    fi
    if [ -n "${MAX_PUYO}" ]; then
        EXTRA_ARGS="${EXTRA_ARGS} --bg-fp-force-max-puyo ${MAX_PUYO}"
    fi

    for VIDEO_DEF in "${EVAL_VIDEOS[@]}"; do
        IFS=':' read -r VID_ALIAS VID_STEM VID_ID <<< "${VIDEO_DEF}"

        VIDEO_PATH="${PROJ_ROOT}/data/match_clips/${VID_ID}/${VID_STEM}.mp4"
        if [ ! -f "${VIDEO_PATH}" ]; then
            _log "[watcher] WARNING: 動画ファイル不在 ${VIDEO_PATH}"
            continue
        fi

        LABEL="${VARIANT_NAME}_${VID_ALIAS}"
        _wait_for_slot
        _run_viz "${LABEL}" "${VIDEO_PATH}" "${EXTRA_ARGS}"
    done
done

# ============================
# Step 3: 全ジョブ完了待ち
# ============================
_log "[watcher] 全 viz ジョブ投入完了、完走待機中... $(date '+%Y-%m-%d %H:%M:%S JST')"

FAILED_JOBS=()
if [ ${#JOB_PIDS[@]} -gt 0 ]; then
    for i in "${!JOB_PIDS[@]}"; do
        pid="${JOB_PIDS[$i]}"
        label="${JOB_LABELS[$i]}"
        wait "${pid}" && status=0 || status=$?
        if [ "${status}" -ne 0 ]; then
            _log "[watcher] FAILED: ${label} (PID=${pid} exit=${status})"
            FAILED_JOBS+=("${label}")
        else
            _log "[watcher] OK: ${label} (PID=${pid})"
        fi
    done
fi

# ============================
# Step 4: 朝報告 markdown 生成
# ============================
COMPLETE_TIME="$(date '+%Y-%m-%d %H:%M:%S JST')"
_log "[watcher] viz 全完了: ${COMPLETE_TIME}"

{
    echo "# 朝の比較動画ガイド (自動生成)"
    echo ""
    echo "生成完了時刻: ${COMPLETE_TIME}"
    echo ""

    if [ -n "${SUMMARY_FILE}" ]; then
        echo "## A/B 評価サマリ"
        echo ""
        echo "ファイル: \`${SUMMARY_FILE}\`"
        echo ""
        # JSON から表を生成 (python で整形)
        PYTHONPATH="${PROJ_ROOT}" "${PYTHON}" -c "
import json, sys
try:
    with open('${SUMMARY_FILE}') as f:
        d = json.load(f)
    variants = d.get('variants', {})
    print('| 仮説 | 説明 | stable_cell_acc | critical_frames |')
    print('|------|------|-----------------|-----------------|')
    for vname, vdata in variants.items():
        desc = vdata.get('description', '')
        acc  = vdata.get('stable_cell_acc_mean', 'N/A')
        crit = vdata.get('critical_frames_total', 'N/A')
        print(f'| {vname.upper()} | {desc} | {acc} | {crit} |')
except Exception as e:
    print(f'JSON 解析失敗: {e}')
" 2>/dev/null || echo "JSON 解析スキップ"
        echo ""
    fi

    echo "## 比較動画 一覧 (Windows パス)"
    echo ""
    echo "各仮説の違い:"
    echo "- **B0**: baseline (warmup guard OFF, MAX_PUYO=144)"
    echo "- **B1**: warmup guard ON のみ (STABLE 復帰直後 12 frame confirmed 凍結)"
    echo "- **B2**: MAX_PUYO=4 のみ (背景 FP 採取を序盤 puyo 4 個以下に限定)"
    echo "- **B3**: B1+B2 両方適用"
    echo ""

    for VID_ALIAS in v40_match01 v57_match01 v89_match01 v29_match02; do
        case "${VID_ALIAS}" in
            v40_match01) echo "### v40_match01 (= 真因対策確認の最重要動画)" ;;
            v57_match01) echo "### v57_match01" ;;
            v89_match01) echo "### v89_match01" ;;
            v29_match02) echo "### v29_match02" ;;
        esac
        for VN in B0 B1 B2 B3; do
            F="${VIZ_DIR}/${VN}_${VID_ALIAS}.mp4"
            # /mnt/c パス -> Windows パス変換
            WIN_PATH=$(echo "${F}" | sed 's|/mnt/c|C:|; s|/|\\|g')
            if [ -f "${F}" ]; then
                echo "- [x] ${VN}: \`${WIN_PATH}\`"
            else
                echo "- [ ] ${VN}: 生成失敗"
            fi
        done
        echo ""
    done

    echo "## 比較の着目点"
    echo ""
    echo "1. **ぷよ落ち着き直後** (= 連鎖/ツモ落下直後のフレーム)"
    echo "   - B0 と B1 を比較: warmup guard が STABLE 遷移直後の誤確定を防ぐか?"
    echo "   - 緑枠 (STABLE) に変わった瞬間のセルに誤色 (R/B/G/Y/P) が出ていないか確認"
    echo ""
    echo "2. **背景学習条件** (= 試合開始直後 2 秒間)"
    echo "   - B0 と B2 を比較: MAX_PUYO=4 絞りで背景 FP 採取誤りが減るか?"
    echo "   - v40 は背景誤認が最も顕著なので特に注目"
    echo ""
    echo "3. **両方適用** (B3 が総合的に最良か?)"
    echo "   - B3 と B0 を並べて: 改善 cell (誤認が消えた) vs 退行 cell (正解が消えた) をカウント"
    echo "   - B3 が全面改善なら即採用、退行があれば B1 or B2 単体採用を検討"
    echo ""

    if [ ${#FAILED_JOBS[@]} -gt 0 ]; then
        echo "## 生成失敗ジョブ"
        echo ""
        for j in "${FAILED_JOBS[@]}"; do
            echo "- ${j}: \`logs/viz_${j}.log\` 参照"
        done
        echo ""
    fi

    echo "## watcher ログ"
    echo ""
    echo "\`C:\\Users\\ryouj\\.gemini\\antigravity\\scratch\\puyo_analyzer\\logs\\watcher.log\`"
} > "${REPORT}"

_log "[watcher] 朝報告 -> ${REPORT}"
_log "[watcher] === 全処理完了 $(date '+%Y-%m-%d %H:%M:%S JST') ==="
