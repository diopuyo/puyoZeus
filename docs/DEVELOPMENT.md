# 開発者向け案内

この文書はソースから開発・検証する人向けです。配布zipの利用手順は [README](../README.md) を参照してください。

## 開発環境

Python 3.11以上（開発実環境は3.12）を使用します。配布zipの同梱環境と、学習・動画処理を含む開発環境は依存が異なります。リポジトリ直下で専用の仮想環境を作成してください。

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pytest tests/
```

学習・動画検証用の入力は別途用意します。入力の利用権限と、上級者対戦に限定するデータ条件を確認してください。入力動画・画面の切り出し・検証用画像を公開用のコミットや配布物に追加しないでください。

## 認識・評価の入口

`scripts.visualize_recognition` は動画から認識結果と盤面ログを作り、`scripts.evaluate_recognition` は盤面ログを評価します。利用可能な引数はそれぞれ `python -m <モジュール名> --help` で確認してください。配布版の利用者入口は常駐アプリで、これらの開発用動画処理とは別です。

認識は盤面の状態遷移を扱い、確定した盤面を評価します。画像認識の誤りや入力条件も含めて検証し、推定値だけで精度を判断しないでください。

## 関連文書

- `AGENTS.md`: データ条件・コード規約。
- `docs/INDICATOR_REFERENCE.md`: 指標の定義。
- `docs/INDICATOR_ROADMAP.md`: 認識・学習の開発資料。
- `docs/KNOWN_WEAKNESSES.md`: 既知の弱点と対応記録。
- `docs/RELEASE_DEFAULTS_20261101.md`: 配布入口の既定設定と検証記録。
- `packaging/build_release.py`: 常駐版の組立処理。
- `THIRD_PARTY_NOTICES.md`: 第三者部品の原文の場所と最終zipの照合項目。

ライセンスはルートの `LICENSE` を参照してください。コードの MIT はゲーム素材や第三者部品の条件を置き換えません。
