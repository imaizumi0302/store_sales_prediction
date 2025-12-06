import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import altair as alt
import shap
import streamlit.components.v1 as components

st.set_page_config(layout="wide") 

# === データの読み込み ===
# 例: Google Colabで生成済みのcsvをアップロードしておく
@st.cache_data
def load_data():
    predictions = pd.read_csv("predictions.csv")   # 各店舗・日付・実績・予測
    val_shap_values = pd.read_csv("val_shap_values.csv")   # shap値
    val_mean_shap = pd.read_csv("val_mean_shap.csv")       # 各特徴量の平均shap値
    test_shap_df = pd.read_csv("shap_values_test_avg.csv")
    test_mean_shap = pd.read_csv("test_mean_shap.csv")
    expected_value = pd.read_csv("expected_value.csv")
    return predictions, val_shap_values, val_mean_shap, test_shap_df, test_mean_shap, expected_value

predictions, val_shap_values, val_mean_shap, test_shap_df, test_mean_shap, expected_value = load_data()

# 1. predictions の 'date' 列を変換
predictions["date"] = pd.to_datetime(predictions["date"], errors='coerce')

# 2. test_shap_df の 'date' 列を変換 
test_shap_df["date"] = pd.to_datetime(test_shap_df["date"], errors='coerce')

# 3. val_shap_values の 'date' 列を変換 (新しい追加)
if "date" in val_shap_values.columns:
    val_shap_values["date"] = pd.to_datetime(val_shap_values["date"], errors='coerce')

# object型で残っている 'family' 列を category 型に変換
predictions["family"] = predictions["family"].astype("category")

test_shap_df["family"] = test_shap_df["family"].astype("category")

if "family" in val_shap_values.columns:
    val_shap_values["family"] = val_shap_values["family"].astype("category")

# ==========================================================
# 【デバッグ用】
# ==========================================================
print("=== predictions DataFrame Info ===")
predictions.info()
print("==================================")

print("=== test_shap_df DataFrame Info ===")
test_shap_df.info()
print("==================================")

print("=== val_shap_values DataFrame Info ===")
val_shap_values.info()
print("==================================")

# ==========================================================

expected_value = float(expected_value.loc[0, "expected_value"])  # CSVから1行目をfloatとして取り出す

#タイトルを表示
st.title("商品販売予測 Viewer")



# サイドバーで選択
store = st.sidebar.selectbox("店舗を選択", predictions["store_nbr"].unique())
#  日付を「YYYY-MM-DD」形式の文字列に変換して表示
date_options = (
    predictions.loc[predictions["store_nbr"] == store, "date"]
    .dt.strftime('%Y-%m-%d')  
    .unique()
)

date = st.sidebar.selectbox("日付を選択", date_options)

products = st.sidebar.multiselect(
    "商品群を選択",
    predictions.loc[(predictions["store_nbr"] == store) & (predictions["date"] == date), "family"].unique()
)




# フィルタリング prediction(予測値と各特徴量を格納しているデータフレーム)
filtered = predictions[
    (predictions["store_nbr"] == store) &
    (predictions["date"].dt.strftime('%Y-%m-%d') == date) &
    (predictions["family"].isin(products))
]

# フィルタリング テストデータの個別shap表示用
filtered_test_shap = test_shap_df[
    (test_shap_df["store_nbr"] == store) &
    (test_shap_df["date"].dt.strftime('%Y-%m-%d') == date) &
    (test_shap_df["family"].isin(products))
]

# === データセット表示　===
st.header("📊 特定店舗・日・商品群の行のデータ")

if filtered.empty:
    st.warning("データが存在しません。商品群を選択してください。")
else:
    st.dataframe(filtered.drop(columns=["pred_mean"]), width="stretch")



# === 1.特定店舗・日・商品群の予測販売個数 ===
st.header("📊 特定店舗・日・商品群の予測販売個数（3モデル平均）")
st.dataframe(filtered[["store_nbr", "date", "family", "pred_mean"]])


# === 2. 全体平均SHAPプロット ===
st.header("🌎 validation data平均SHAP値")

val_mean_shap_sorted = val_mean_shap.sort_values("mean_abs_shap", ascending=False)
# st.bar_chart(mean_shap_sorted.set_index("feature")["mean_abs_shap"])

chart = alt.Chart(val_mean_shap_sorted).mark_bar().encode(
    x=alt.X('feature', sort=list(val_mean_shap_sorted['feature'])),  # 順序を固定
    y='mean_abs_shap'
).properties(width=600, height=400)

st.altair_chart(chart)



# === 3.全体平均SHAP ===
st.header("🌎 テストデータ平均SHAP値")
test_mean_shap_sorted = test_mean_shap.sort_values("mean_abs_shap", ascending=False)
# st.bar_chart(mean_shap_sorted.set_index("feature")["mean_abs_shap"])
chart_test = alt.Chart(test_mean_shap_sorted).mark_bar().encode(
    x=alt.X('feature', sort=list(test_mean_shap_sorted['feature'])),  # 順序を固定
    y='mean_abs_shap'
).properties(width=600, height=400)

st.altair_chart(chart_test)




test_features_for_app = ["store_nbr_shap","family_shap","sales_by_store_nbr", "sales_by_family", "onpromotion", "year", "month", "day","weekday",
               "sales_by_store_nbr_family","rolling_mean_3","rolling_mean_7",
               "rolling_mean_30","sales_by_type","sales_by_cluster","dcoilwtico","oil_mean_30", "oil_mean_90"]

test_features_for_app_1 = ["store_nbr","date","family","store_nbr_shap","family_shap","sales_by_store_nbr", "sales_by_family", "onpromotion", "year", "month", "day","weekday",
               "sales_by_store_nbr_family","rolling_mean_3","rolling_mean_7",
               "rolling_mean_30","sales_by_type","sales_by_cluster","dcoilwtico","oil_mean_30", "oil_mean_90"]

# === 4.個別平均SHAP ===

# === SHAP値の表示（個別行） ===
st.header("🔍 SHAP 値（この行の特徴量の影響）")

if filtered.empty:
    st.warning("商品群を選択してください。データが存在しません。")
else:
    # 特徴量を抽出（pred_mean, date, categoryなどは除外）
    shap_row_features = filtered.drop(
        columns=["pred_mean", "date", "family"], 
        errors="ignore"
    )

    # shap_df の中から該当行だけ抽出
    shap_row = test_shap_df.loc[
        (test_shap_df["store_nbr"] == store) &
        (test_shap_df["date"] == date) &
        (test_shap_df["family"].isin(products))
    ]

    if shap_row.empty:
        st.warning("SHAP値が存在しません（モデルの学習データ外の可能性）。")
    else:
        # SHAP 値の表示
        st.subheader("📈 SHAP 値（特徴量ごとの寄与）")
        st.dataframe(shap_row, width="stretch")


# === 5.個別平均SHAP Force Plot===

st.header("🌎 テストデータ個別SHAP値 Force Plot(行ごと)")

if not filtered_test_shap.empty:
    # サイドバーで選択した行のインデックスを取得
    idx = filtered_test_shap.index[0]  # 最初の行
    # その行のSHAP値と特徴量値を取り出す
    shap_values_row = filtered_test_shap.loc[idx, test_features_for_app].values
    X_row = filtered_test_shap.loc[idx, test_features_for_app]

    # force_plot の処理
    # expected_value は事前に保存しておいたものを使用
    force_plot = shap.force_plot(
    expected_value,  # テストデータ用 expected_value
    shap_values_row,
    X_row,
    matplotlib=False
    )

    # HTMLとして保存して表示
    shap.save_html("temp.html", force_plot)
    with open("temp.html", "r", encoding="utf-8") as f:
        html = f.read()
    
    # Streamlit に表示（幅を100%に拡張）
    html = html.replace('<body>', '<body style="width:100%;">')
    components.html(html, height=400)  # 高さはお好みで調整

else:
    st.warning("該当データがありません。")


