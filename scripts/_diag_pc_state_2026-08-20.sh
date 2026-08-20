#!/bin/bash
# 収集が遅い原因として PC 側の物理的制約 (熱によるクロック低下・メモリ圧・
# ディスクI/O飽和) を切り分ける (2026-08-20)。
# ノートPC (RTX 4060 Laptop) なので長時間高負荷でサーマルスロットリングが
# 起きうる。それが「14並列で1本あたり4.3倍遅い」の真因かを確認する。
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer || exit 1

echo "=== CPU ==="
grep -m1 "model name" /proc/cpuinfo
echo "  論理コア数: $(nproc)"
echo "  物理コア/スレッド:"
lscpu 2>/dev/null | grep -E "^CPU\(s\)|Core\(s\) per socket|Thread\(s\) per core|CPU max MHz|CPU min MHz"

echo "=== 現在のクロック (スロットリング判定) ==="
echo "  実効MHz (上位8コア):"
grep "cpu MHz" /proc/cpuinfo | head -8 | awk '{printf "    %.0f MHz\n", $4}'
echo "  平均:"
grep "cpu MHz" /proc/cpuinfo | awk '{s+=$4; n++} END {printf "    %.0f MHz (%d コア)\n", s/n, n}'

echo "=== メモリ ==="
free -h
echo "  スワップ使用:"
vmstat 1 2 | tail -1 | awk '{print "    si="$7" so="$8" (0以外ならスワップ発生=致命的)"}'

echo "=== 負荷 ==="
uptime
echo "  実行中の収集プロセス: $(pgrep -c -f _collect_lean_1t || echo 0)"

echo "=== ディスク I/O (収集は大きなmp4を並列で読む) ==="
vmstat 1 3 | tail -2 | awk '{print "    bi(読込)="$9" bo(書込)="$10" wa(I/O待ち%)="$16}'

echo "=== 動画の置き場所 ==="
df -h /mnt/c | tail -1
echo "  (WSL2 から /mnt/c は 9p 経由でアクセスが遅い可能性)"

echo "=== GPU ==="
if command -v nvidia-smi > /dev/null 2>&1; then
  nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu \
    --format=csv,noheader 2>/dev/null || echo "  nvidia-smi 実行失敗"
else
  echo "  nvidia-smi なし (WSL から見えていない)"
fi
