#!/bin/bash
# CPU の交通整理 (2026-08-21、user 指示「コアの優先配置とかした方が良くね、交通整備」)。
#
# 状況: 全テスト (pytest、11.7コア相当) と速度の実測が同じ優先度で殴り合い、
# どちらも遅くなっていた (負荷23、pytest は通常23.6分が29分で38%)。
#
# 方針: 検証用の長時間ジョブを最低優先度 (nice 19) に落とす。
# nice 19 は「他が CPU を要求していない時だけ走る」設定なので、
# ジョブ自体が止まることはなく、実測の方が優先される。
#
# 注意: pgrep のパターンにスペースやスラッシュを入れると Git Bash 経由の wsl で
# 壊れる (memory feedback_msys_pipe_escape)。だからスクリプトファイルにしている。

echo "=== 交通整理 $(date +%H:%M:%S) ==="
echo "--- before ---"
cat /proc/loadavg

# 最低優先度に落とす対象 (検証・テスト系)。実測系は触らない。
for pat in pytest cargo; do
  for p in $(pgrep -f "$pat"); do
    cur=$(ps -o ni= -p "$p" 2>/dev/null | tr -d ' ')
    if [ -n "$cur" ] && [ "$cur" -lt 19 ]; then
      renice -n 19 -p "$p" > /dev/null 2>&1 && echo "  nice 19 <- pid $p ($pat, was $cur)"
    fi
  done
done

echo "--- 現在の優先度 ---"
ps -eo pid,ni,pcpu,etime,comm --no-headers | grep -E "python|pytest|cargo" | head -10

echo "=== 完了 $(date +%H:%M:%S) ==="
