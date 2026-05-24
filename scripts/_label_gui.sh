#!/usr/bin/env bash
# ラベリング GUI 起動ヘルパー (= 短いコマンドで使えるように).
#
# 使い方 (Windows ターミナル):
#   wsl -d Ubuntu -- bash scripts/_label_gui.sh v50            # v50 全体 (デフォルト 2 秒間隔)
#   wsl -d Ubuntu -- bash scripts/_label_gui.sh v50 48 75 2    # v50 48-75s 2 秒間隔
#   wsl -d Ubuntu -- bash scripts/_label_gui.sh v91 0 75 2.5   # v91 全 75s 2.5 秒間隔
#   wsl -d Ubuntu -- bash scripts/_label_gui.sh test           # テスト用 (= v50 50-55s 1 秒間隔)
#
# 動画 ID マッピング:
#   v50   → data/test_unknown/v50_match1_75s_720p.mp4
#   v91   → data/test_unknown/v91_match1_75s_720p.mp4
#   test  → v50 の短い区間 (= 動作確認)
#
# CLI 引数 (positional):
#   $1 = video_id (v50 / v91 / test)
#   $2 = start_sec (省略時 0)
#   $3 = end_sec (省略時 75)
#   $4 = interval_sec (省略時 2.0)
#
# 環境変数 override:
#   LABEL_CLI=opencv   ← OpenCV 版 (scripts.label_cells) で起動
#   LABEL_CLI=gui      ← Tkinter 版 (scripts.label_cells_gui) で起動 (default)
set -e
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer

# WSLg 用 DISPLAY 環境変数を明示設定 (= wsl 直接実行ではプロファイルが読まれない).
# WSL2 + Windows 11 + WSLg なら以下のデフォルトで GUI が転送される.
export DISPLAY="${DISPLAY:-:0}"
export WAYLAND_DISPLAY="${WAYLAND_DISPLAY:-wayland-0}"
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/mnt/wslg/runtime-dir}"
export PULSE_SERVER="${PULSE_SERVER:-/mnt/wslg/PulseServer}"

VID="${1:-v50}"
START_SEC="${2:-0}"
END_SEC="${3:-75}"
INTERVAL="${4:-2.0}"
LABEL_CLI="${LABEL_CLI:-gui}"

# video_id → video path + ラベル保存 ID マッピング
case "$VID" in
    v50)
        VIDEO="data/test_unknown/v50_match1_75s_720p.mp4"
        STORE_ID="v50_match1"
        ;;
    v91)
        VIDEO="data/test_unknown/v91_match1_75s_720p.mp4"
        STORE_ID="v91_match1"
        ;;
    v89)
        VIDEO="data/test_unknown/v89_match1_75s_720p.mp4"
        STORE_ID="v89_match1"
        ;;
    v29)
        VIDEO="data/evaluation_videos/v29_match2_156s.mp4"
        STORE_ID="v29_match2"
        ;;
    v40)
        VIDEO="data/evaluation_videos/v40_match7_125s.mp4"
        STORE_ID="v40_match7"
        ;;
    v57)
        VIDEO="data/evaluation_videos/v57_match2_100s.mp4"
        STORE_ID="v57_match2"
        ;;
    v51)
        VIDEO="data/evaluation_videos/v51_match2_97s.mp4"
        STORE_ID="v51_match2"
        ;;
    v70)
        VIDEO="data/evaluation_videos/v70_match2_113s.mp4"
        STORE_ID="v70_match2"
        ;;
    v89m3)
        VIDEO="data/evaluation_videos/v89_match3_95s.mp4"
        STORE_ID="v89_match3"
        ;;
    test)
        VIDEO="data/test_unknown/v50_match1_75s_720p.mp4"
        STORE_ID="test_v50"
        START_SEC=50
        END_SEC=55
        INTERVAL=1.0
        ;;
    *)
        echo "[error] 未知の video_id: $VID (有効: v50 / v91 / v89 / v29 / v40 / v57 / v51 / v70 / v89m3 / test)" >&2
        exit 1
        ;;
esac

if [ ! -f "$VIDEO" ]; then
    echo "[error] 動画が存在しません: $VIDEO" >&2
    exit 1
fi

case "$LABEL_CLI" in
    opencv)
        MODULE="scripts.label_cells"
        echo "[label-gui] OpenCV 版で起動"
        ;;
    gui|tk|tkinter)
        MODULE="scripts.label_cells_gui"
        echo "[label-gui] Tkinter 版で起動"
        ;;
    *)
        echo "[error] LABEL_CLI=$LABEL_CLI 不正" >&2
        exit 1
        ;;
esac

echo "[label-gui] video=$VIDEO id=$STORE_ID start=$START_SEC end=$END_SEC interval=$INTERVAL"
exec env PYTHONPATH=. ./venv/bin/python -m "$MODULE" \
    --video "$VIDEO" \
    --video-id "$STORE_ID" \
    --start-sec "$START_SEC" \
    --end-sec "$END_SEC" \
    --interval-sec "$INTERVAL"
