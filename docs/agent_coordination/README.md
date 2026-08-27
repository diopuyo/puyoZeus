# Claude / Codex 共有ハブ

## 目的

Claude Code と Codex が同じ作業ツリーを扱う際に、現在地・担当・禁止事項・
次の命令を一か所で共有し、古い会話やメモリだけを根拠に作業を始めないための仕組み。

## ファイル所有権

- `CURRENT.md`: Codex が実プロセス・成果物を確認して更新する現在地。
- `PLAN.md`: 全体計画と完了条件。方針変更はユーザー承認後に反映する。
- `CODEX_TO_CLAUDE.md`: Claudeへの命令箱。Codexのみ更新する。
- `CLAUDE_TO_CODEX.md`: Claudeの報告箱。Claudeは既存記録を消さず末尾追記する。
- `DECISIONS.md`: 採否・不変条件・判断根拠の追記専用ログ。

## 運用

1. エージェントは開始時に共有ハブを読む。
2. `scripts/show_agent_coordination_status.ps1` で実プロセスを確認する。
3. write lock があれば調査・報告だけ行い、対象ソースを編集しない。
4. 作業完了時は担当側の報告箱へ、変更ファイル・テスト・成果物・残課題を追記する。
5. 採用フラグはユーザー承認後のみ `src/production_config.py` へ登録する。

