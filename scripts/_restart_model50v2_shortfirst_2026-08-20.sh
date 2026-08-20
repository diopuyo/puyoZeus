#!/bin/bash
# 48本収集を停止し、スレッド制限版ラッパー + 動画長昇順で再起動する (2026-08-20)。
#
# 経緯: 14並列で1本あたりが単独実行の4.3倍に劣化し、48本18時間の見積もりに
# なった。原因は物理8コア(論理16)に14プロセスを詰めたこと + 全コア稼働時の
# クロック低下 (4.2倍で実測4.3倍をほぼ説明)。load average 15.9 ≈ 論理コア数
# なので 1,008 スレッドの大半は sleep しており、スレッド制限の効果は
# 1.1〜1.5倍程度と見込まれる (再起動の損益分岐は 1.14倍)。
#
# 再起動の主目的は速度ではなく**検証の前倒し**: 2.5時間経過して完了0本という
# のは「2本ごとに確認して進める」(user方針) を満たせない軌道であり、
# 短い動画から処理すれば約1.5時間で検証2本が手に入る。
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer || exit 1

echo "=== 停止 ==="
pkill -f _regen_model50v2_2026-08-20
pkill -f _run_model50v2_2026-08-20
sleep 3
pkill -9 -f _collect_lean_1t
sleep 5
n=$(pgrep -c -f _collect_lean_1t)
echo "  収集プロセス残: ${n:-0}"
if [ "${n:-0}" -gt 0 ]; then
  echo "  [警告] まだ残っている。再度 kill する"
  pkill -9 -f _collect_lean_1t
  sleep 5
  echo "  収集プロセス残: $(pgrep -c -f _collect_lean_1t)"
fi
uptime
