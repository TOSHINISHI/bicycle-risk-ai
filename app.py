import joblib
import pandas as pd
import streamlit as st
from pathlib import Path


st.set_page_config(
    page_title="駅前放置自転車リスク予測AI",
    page_icon="🚲",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
RISK_THRESHOLD = 100
SOURCE_URL = (
    "https://www.tomin-anzen.metro.tokyo.lg.jp/"
    "kotsu/jitensha/houchi/0000001962?"
)
COUNT_MODEL_PATH = BASE_DIR / "bicycle_count_model.pkl"

if not COUNT_MODEL_PATH.is_file():
    st.error(
        "台数予測モデル（bicycle_count_model.pkl）が見つかりません。"
        "アプリと同じフォルダに配置してください。"
    )
    st.stop()

count_model = joblib.load(COUNT_MODEL_PATH)
model = joblib.load(BASE_DIR / "bicycle_risk_model.pkl")
stations = pd.read_csv(BASE_DIR / "station_data_2025.csv")
history = pd.read_csv(BASE_DIR / "station_history.csv")

station_features = stations[["放置自転車台数"]]
predictions = model.predict(station_features)
raw_count_predictions = count_model.predict(station_features)
count_predictions = [
    int(max(0.0, float(prediction)) + 0.5)
    for prediction in raw_count_predictions
]

ranking = stations[["駅名", "放置自転車台数"]].copy()
ranking = ranking.rename(
    columns={"放置自転車台数": "2025年度放置台数"}
)
ranking["2026年度予測台数"] = count_predictions
ranking["前年比予測増減台数"] = (
    ranking["2026年度予測台数"] - ranking["2025年度放置台数"]
)
ranking["2026年度予測結果"] = [
    "リスクあり" if prediction == 1 else "リスクなし"
    for prediction in predictions
]
ranking = ranking.sort_values(
    "2025年度放置台数",
    ascending=False,
    kind="stable",
).reset_index(drop=True)

st.title("🚲 駅前放置自転車リスク予測AI")
st.write(
    "東京都の公開データと学習済みモデルを使って、"
    "2026年度の放置自転車リスクを確認できます。"
)

detail_tab, dashboard_tab = st.tabs(["駅別詳細", "全体ダッシュボード"])

with dashboard_tab:
    st.subheader("全体の概要")
    metric1, metric2, metric3 = st.columns(3)
    with metric1:
        st.metric("分析対象駅数", f"{len(ranking):,} 駅")
    with metric2:
        risk_count = int((predictions == 1).sum())
        st.metric("リスクありの駅数", f"{risk_count:,} 駅")
    with metric3:
        total_count = stations["放置自転車台数"].sum()
        st.metric("2025年度の放置自転車台数合計", f"{total_count:,.0f} 台")

    st.subheader("2026年度の駅別リスク・台数一覧")
    show_risk_only = st.checkbox("リスクありの駅のみ表示")
    visible_ranking = ranking
    if show_risk_only:
        visible_ranking = ranking[
            ranking["2026年度予測結果"] == "リスクあり"
        ]

    st.dataframe(visible_ranking, hide_index=True, width="stretch")
    csv_data = ranking.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "全駅の予測結果をCSVでダウンロード",
        data=csv_data,
        file_name="2026年度_駅別リスク予測.csv",
        mime="text/csv",
    )

with detail_tab:
    st.subheader("駅を検索して詳細を確認")
    station_names = sorted(stations["駅名"].dropna().unique())
    query = st.text_input("駅名で検索（部分一致）").strip()
    matching_stations = [name for name in station_names if query in name]

    if not matching_stations:
        st.info("該当する駅が見つかりません。別の駅名で検索してください。")
    else:
        station = st.selectbox("駅を選択してください", matching_stations)
        selected = stations[stations["駅名"] == station].iloc[0]
        selected_position = stations.index.get_loc(selected.name)
        selected_prediction = int(predictions[selected_position])
        selected_count_prediction = count_predictions[selected_position]
        predicted_change = (
            selected_count_prediction - selected["放置自転車台数"]
        )

        st.subheader("2026年度の予測結果")
        count_metric, forecast_metric, change_metric, risk_metric = st.columns(4)
        with count_metric:
            st.metric(
                "2025年度の放置自転車台数",
                f'{selected["放置自転車台数"]:,.0f} 台',
            )
        with forecast_metric:
                st.metric("2026年度の予測台数", f"{selected_count_prediction:,} 台")
        with change_metric:
            st.metric("前年比の予測増減台数", f"{predicted_change:+,.0f} 台")
        with risk_metric:
            if selected_prediction == 1:
                st.error("リスクあり：100台以上になると予測")
            else:
                st.success("リスクなし：100台未満になると予測")

        st.caption(
            "※ 2026年度の数値はAIによる予測であり、実績ではありません。"
        )
        st.caption(
            "※ 100台という基準は本課題独自のもので、"
            "東京都の公式な危険度区分ではありません。"
        )

        st.subheader("放置自転車台数の推移（2012〜2026年度）")
        station_history = history[
            history["駅名"] == station
        ].sort_values("年度")
        actual_column = "実績（2012〜2025年度）"
        forecast_column = "予測（2025〜2026年度）"
        actual_series = (
            station_history.set_index("年度")["放置自転車台数"]
            .rename(actual_column)
        )
        forecast_series = pd.Series(
            {
                2025: selected["放置自転車台数"],
                2026: selected_count_prediction,
            },
            name=forecast_column,
        )
        chart_data = pd.concat(
            [actual_series, forecast_series],
            axis=1,
        ).rename_axis("年度").reset_index().sort_values("年度")
        st.line_chart(
            chart_data,
            x="年度",
            y=[actual_column, forecast_column],
            color=["#1f77b4", "#ff7f0e"],
        )
        st.caption("青線：過去の実績、オレンジ線：2025年度実績から2026年度予測への推移")

        st.subheader("2025年度の駐輪場情報")
        capacity1, capacity2, capacity3 = st.columns(3)
        with capacity1:
            st.metric(
                "実収容台数",
                f'{selected["駐輪場実収容台数"]:,.0f} 台',
            )
        with capacity2:
            st.metric(
                "収容能力",
                f'{selected["駐輪場収容能力"]:,.0f} 台',
            )
        with capacity3:
            rate = selected["駐輪場利用率"]
            st.metric(
                "駐輪場利用率",
                f"{rate:.1f}%" if pd.notna(rate) else "データなし",
            )

st.divider()
st.markdown(
    "データ出典：[東京都 駅前放置自転車等の現況と対策"
    "（2012〜2025年度）]("
    f"{SOURCE_URL})"
)
st.caption(
    "2026年度の台数は学習済みRandomForestRegressorによる予測値で、"
    "実績ではありません。リスク判定はRandomForestClassifierによる二値分類です。"
    "いずれも将来の状況を保証するものではありません。"
)
