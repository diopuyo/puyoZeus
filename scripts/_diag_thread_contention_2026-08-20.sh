#!/bin/bash
# 収集プロセスのスレッド数を数え、並列時のスレッド競合を確認する (2026-08-20)。
# 14並列で1本あたりが単独実行の4倍以上遅い原因として、cv2 は
# setNumThreads(1) で抑えているが torch (CNN推論) のスレッド数が
# 制限されていない疑いを検証する。
# 複雑なコマンド置換は wsl 直書きだと MSYS に壊されるためスクリプト化
# (memory feedback_msys_pipe_escape)。
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer || exit 1

echo "=== プロセスごとのスレッド数 ==="
total=0
count=0
for p in $(pgrep -f _collect_lean_1t); do
  n=$(ls /proc/$p/task 2>/dev/null | wc -l)
  total=$((total + n))
  count=$((count + 1))
  if [ "$count" -le 5 ]; then
    echo "  PID $p : $n スレッド"
  fi
done
echo "  ..."
echo "=== 合計 ==="
echo "  プロセス数: $count"
echo "  スレッド総数: $total"
echo "  CPUコア数: $(nproc)"

echo "=== スレッド関連の環境変数 (最初のプロセス) ==="
first=$(pgrep -f _collect_lean_1t | head -1)
if [ -n "$first" ]; then
  tr '\0' '\n' < /proc/$first/environ 2>/dev/null | grep -iE "OMP|MKL|THREAD|TORCH|BLAS" || echo "  (スレッド制限の環境変数なし)"
fi

echo "=== CPU使用率上位 ==="
ps -eo pid,pcpu,nlwp,comm --sort=-pcpu | head -6
