"""案 R3 改 (= ぷよ色プロファイル DB) 朝報告 markdown 生成"""
import json
from pathlib import Path

ROOT = Path("data/verify/puyo_profile_eval")
cmp_data = json.loads((ROOT / "_comparison.json").read_text(encoding="utf-8"))

per = cmp_data["per_video"]
patch = cmp_data["patch_fp_per_video"]
axis = cmp_data["axis3b_per_video"]

improved: list[tuple[str, int, int, int]] = []
for v in per:
    p = per[v]["critical"]
    p_fp = patch[v]["critical"]
    d = p - p_fp
    improved.append((v, d, p, p_fp))
improved.sort(key=lambda x: x[1])

VIZ_BASE = "C:\\Users\\ryouj\\.gemini\\antigravity\\scratch\\puyo_analyzer\\data\\verify"
ORDER = ["v89m7", "v30_match11", "v30_5min", "v97_match11", "v29m2",
         "v40m7", "v51m2", "v57m2", "v70m2", "v89m3", "v95m15", "v97m11"]

lines: list[str] = []
lines.append("# 案 R3 改 朝報告 (ぷよ色プロファイル DB)")
lines.append("")
lines.append("生成: 2026-05-28 13:36 JST")
lines.append("")
lines.append("## 結論サマリ")
lines.append("")
lines.append(f'- **案 R3 改 12 動画合計 critical = {cmp_data["puyo_profile_critical_total"]}**')
lines.append(f'- patch_fp 12 動画合計 = {cmp_data["patch_fp_critical_total"]}')
lines.append(f'- axis3b_revert 12 動画合計 = {cmp_data["axis3b_revert_critical_total"]}')
lines.append(f'- baseline_v3 (= 8 動画) = {cmp_data["baseline_v3_critical_total"]}')
lines.append("")
lines.append(f'- **vs patch_fp = {cmp_data["diff_vs_patch_fp"]:+d} ({cmp_data["diff_vs_patch_fp"]/2147*100:+.1f}%)**')
lines.append(f'- **vs axis3b_revert = {cmp_data["diff_vs_axis3b_revert"]:+d}**')
lines.append(f'- **vs baseline_v3 = {cmp_data["diff_vs_baseline_v3"]:+d} ({cmp_data["pct_vs_baseline_v3"]:+.1f}%)** ← baseline_v3 を **初めて下回る**')
lines.append("")
lines.append(f'- **退行動画 = 0 (= regression_flags 空)**')
lines.append(f'- **verdict = {cmp_data["verdict"]}**')
lines.append("")

lines.append("## 12 動画 critical 比較")
lines.append("")
lines.append("| 動画 | R3改 | patch_fp | axis3b | diff vs patch | 評価 |")
lines.append("|---|---|---|---|---|---|")
for v in ORDER:
    p = per[v]["critical"]
    p_fp = patch[v]["critical"]
    a = axis[v]["critical"]
    d = p - p_fp
    if d <= -50:
        evl = "✅✅✅ 大幅改善"
    elif d <= -10:
        evl = "✅✅ 改善"
    elif d <= -5:
        evl = "✅ 軽改善"
    else:
        evl = "＝"
    lines.append(f"| {v} | **{p}** | {p_fp} | {a} | {d:+d} | {evl} |")
tot_r3 = cmp_data["puyo_profile_critical_total"]
tot_p = cmp_data["patch_fp_critical_total"]
tot_a = cmp_data["axis3b_revert_critical_total"]
lines.append(f"| **合計** | **{tot_r3}** | {tot_p} | {tot_a} | **{tot_r3 - tot_p:+d}** | **大幅改善** |")
lines.append("")

lines.append("## 注目発見")
lines.append("")
lines.append(f"1. **v29m2 (= 最大 critical 動画) = 163** (= patch_fp 497 から -334、 **約 1/3 まで削減**)")
lines.append(f"2. **v40m7 (= 真因動画) = 46** (= patch_fp 127 から -81、 真因解決維持 + 周辺誤認削減)")
lines.append(f"3. **v95m15 (= 退行動画) = 52** (= patch_fp 116 から -64、 アナリスト指摘 3 秒幻ぷよ解消推定)")
lines.append("")

lines.append("## ⚠️ fail-silent 警戒")
lines.append("")
lines.append("critical metric が大幅に下がっただけで「正常ぷよを過剰に EMPTY 化」 している可能性あり (memory `feedback_viz_eval_required.md`)。 アナリスト並列分析で検証中、 user 目視レビューも必須。")
lines.append("")

lines.append("## レビュー動画パス")
lines.append("")
lines.append("### 真因動画 (= 解決維持確認)")
lines.append("")
lines.append("```")
lines.append(f"{VIZ_BASE}\\puyo_profile_eval\\v40m7.mp4")
lines.append("```")
lines.append("")
lines.append("### 退行動画 (= 幻ぷよ解消確認)")
lines.append("")
lines.append("```")
lines.append(f"{VIZ_BASE}\\puyo_profile_eval\\v95m15.mp4")
lines.append("```")
lines.append("")
lines.append("### 最大 critical 動画 (= -334 改善確認)")
lines.append("")
lines.append("```")
lines.append(f"{VIZ_BASE}\\puyo_profile_eval\\v29m2.mp4")
lines.append("```")
lines.append("")
lines.append("### 全 12 動画 (= 大幅改善上位)")
lines.append("")
lines.append("```")
for v, d, _, _ in improved[:6]:
    lines.append(f"{VIZ_BASE}\\puyo_profile_eval\\{v}.mp4  ({d:+d})")
lines.append("```")
lines.append("")
lines.append("### 比較対象 (= patch_fp 版)")
lines.append("")
lines.append("```")
lines.append(f"{VIZ_BASE}\\patch_fp_eval\\v40m7.mp4")
lines.append(f"{VIZ_BASE}\\patch_fp_eval\\v95m15.mp4")
lines.append(f"{VIZ_BASE}\\patch_fp_eval\\v29m2.mp4")
lines.append("```")
lines.append("")

lines.append("## user 判断選択肢")
lines.append("")
lines.append("- **A. ACCEPT 即 commit** = critical -48.7% 改善 + 退行ゼロで採用、 fail-silent は user 動画 1-2 本目視で確認")
lines.append("- **B. ACCEPT + 詳細検証** = アナリスト fail-silent 検証結果待ち、 viz 目視結果と合わせて最終判断 (= 推奨)")
lines.append("- **C. REJECT** = critical 改善は fail-silent 疑い、 撤回")
lines.append("")
lines.append("推奨 = **B** (= 30 分以内にアナリスト結果出る、 user viz 目視と合わせて確定判断)")

out = ROOT / "_morning_report.md"
out.write_text("\n".join(lines), encoding="utf-8")
print(f"朝報告生成: {out}")
print(f"lines: {len(lines)}")
