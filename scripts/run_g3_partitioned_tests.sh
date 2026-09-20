#!/usr/bin/env bash
# 既存のc案を再現する。stdoutの末尾ではなく各pytestの実終了コードを合算する。
# 使用法: PYTHON=/path/to/python bash scripts/run_g3_partitioned_tests.sh /mnt/d/... [full|isolated]
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${PYTHON:-python3}"
OUT="${1:?新規のD保存ディレクトリを指定してください}"
OUT="$(realpath -m "$OUT")" || exit 2
MODE="${2:-full}"
case "$OUT" in /mnt/d/puyo_analyzer/verify/*) ;; *) echo '出力先はDのverify配下が必須' >&2; exit 2 ;; esac
case "$MODE" in full|isolated) ;; *) echo 'modeはfull又はisolated' >&2; exit 2 ;; esac
cd "$ROOT" || exit 2
mapfile -t FILES < tests/g3_isolated_files.txt
IGNORE=()
for F in "${FILES[@]}"; do
  [[ "$F" == tests/test_*.py && -f "$F" ]] || { echo "必須テスト欠落: $F" >&2; exit 2; }
  IGNORE+=("--ignore=$F")
done
[[ "${#FILES[@]}" -eq 22 ]] || { echo '隔離リストは22本が必須' >&2; exit 2; }
[[ "$(printf '%s\n' "${FILES[@]}" | sort -u | wc -l)" -eq 22 ]] || exit 2
mkdir "$OUT" || exit 2
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
printf 'mode=%s\nroot=%s\npython=%s\n' "$MODE" "$ROOT" "$PYTHON" > "$OUT/START.txt"
printf 'group\texit_code\n' > "$OUT/exit_codes.tsv"
RC=0
if [[ "$MODE" == full ]]; then
  "$PYTHON" -m pytest tests/ "${IGNORE[@]}" -q --tb=short -p no:cacheprovider \
    --basetemp="$OUT/main_tmp" --junitxml="$OUT/main.xml" > "$OUT/main.log" 2>&1
  CODE=$?
  [[ "$CODE" -ne 0 || -s "$OUT/main.xml" ]] || CODE=1
  printf 'main\t%s\n' "$CODE" >> "$OUT/exit_codes.tsv"
  [[ "$CODE" -eq 0 ]] || RC=1
fi
for F in "${FILES[@]}"; do
  NAME="$(basename "$F" .py)"
  "$PYTHON" -m pytest "$F" -q --tb=short -p no:cacheprovider \
    --basetemp="$OUT/${NAME}_tmp" --junitxml="$OUT/$NAME.xml" > "$OUT/$NAME.log" 2>&1
  CODE=$?
  [[ "$CODE" -ne 0 || -s "$OUT/$NAME.xml" ]] || CODE=1
  printf '%s\t%s\n' "$F" "$CODE" >> "$OUT/exit_codes.tsv"
  printf '%s rc=%s\n' "$F" "$CODE"
  [[ "$CODE" -eq 0 ]] || RC=1
done
printf '%s\n' "$RC" > "$OUT/exit_code.txt"
# isolated成功は本体群の成功やG3の動画品質合格を意味しない。
exit "$RC"
