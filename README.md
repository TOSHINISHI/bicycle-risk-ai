# 駅前放置自転車リスク予測AI

東京都の駅別データと学習済みモデルを使い、2025年度の放置自転車台数から2026年度の予測台数とリスクを確認するStreamlitアプリです。駅名検索、全駅の予測一覧・CSVダウンロード、駅ごとの過去推移・駐輪場情報を利用できます。

## 使用データ

- `station_data_2025.csv`：2025年度の駅別放置自転車台数・駐輪場情報
- `station_history.csv`：2012〜2025年度の駅別推移
- `bicycle_count_model.pkl`：2026年度の放置自転車台数を予測する学習済み回帰モデル
- `bicycle_risk_model.pkl`：リスクあり・なしを予測する学習済み分類モデル
- データ出典：[東京都 駅前放置自転車等の現況と対策](https://www.tomin-anzen.metro.tokyo.lg.jp/kotsu/jitensha/houchi/0000001962?)

台数予測は0未満を0に補正して整数に四捨五入します。2026年度の数値はAIによる予測であり、実績ではありません。リスクあり・なしは「放置自転車が100台以上になるか」を基準とした課題独自の分類で、東京都の公式な危険度区分ではありません。予測は将来の状況を保証するものではありません。

## 使用アルゴリズム

- `bicycle_count_model.pkl`：RandomForestRegressorによる台数予測
- `bicycle_risk_model.pkl`：RandomForestClassifierによる二値分類

既存の学習済みモデルを使用し、アプリ起動時の再学習は行いません。学習Notebook・学習スクリプトはこのプロジェクト内にありません。

## 起動方法

Python 3.12を含むPython環境で依存パッケージをインストールしてから起動します。scikit-learnはモデルとの互換性のため1.6.1を使用します。

```bash
pip install -r requirements.txt
streamlit run app.py
```

`app.py`、モデルファイル、CSVファイルを同じプロジェクトフォルダに配置してください。Streamlit Community Cloudでは、Python 3.12を選択し、このリポジトリの `app.py` をメインファイルとして指定します。
