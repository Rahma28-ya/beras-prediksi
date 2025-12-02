import streamlit as st
import pandas as pd
import plotly.express as px
from prophet import Prophet
from statsmodels.tsa.statespace.sarimax import SARIMAX
from streamlit_option_menu import option_menu
import warnings
warnings.filterwarnings("ignore")

# 1. KONFIGURASI HALAMAN
st.set_page_config(
    page_title="Prediksi Harga Beras Indonesia",
    page_icon="🌾",
    layout="wide"
)

# 2. CSS CUSTOM – mempercantik UI
custom_css = """
<style>
/* Background */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(120deg, #f5f9ff, #eef6ff);
}

/* Card Style */
.card {
    background: white;
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    border-left: 6px solid #1F618D;
    margin-bottom: 20px;
}

/* Title Center */
.title-center {
    text-align: center;
    font-size: 30px;
    font-weight: bold;
    color: #1F4E79;
}

/* Sub Title */
.sub-title {
    font-size: 22px;
    font-weight: 600;
    color: #0B5A8A;
}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# 3. SIDEBAR NAVIGASI
with st.sidebar:
    selected = option_menu(
        "Navigasi",
        ["🏠 Dashboard", "📈 Prediksi SARIMA", "🔮 Prediksi Prophet", "ℹ️ Tentang"],
        icons=["house", "graph-up", "activity", "info-circle"],
        default_index=0
    )

# 4. LOAD DATA
@st.cache_data
def load_data():
    df = pd.read_csv("bps_final.csv")
    df["tanggal"] = pd.to_datetime(df["tanggal"])
    df = df.rename(columns={"tanggal": "ds", "harga": "y"})
    df = df.sort_values("ds")
    return df

df = load_data()

# 5. DASHBOARD UTAMA
if selected == "🏠 Dashboard":

    st.markdown("<div class='title-center'>🌾 Dashboard Harga Beras Indonesia 🌾</div>", unsafe_allow_html=True)
    st.write("")

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='sub-title'>📌 Input Harga Bulan Ini (Manual)</div>", unsafe_allow_html=True)

    harga_default = int(df['y'].iloc[-1])
    harga_manual = st.number_input(
        "Harga bulan ini:",
        min_value=0,
        value=harga_default,
        help="Masukkan harga terbaru secara manual."
    )

    st.markdown(
        f"<h4 style='color:#0B5A8A;'>Harga manual bulan ini: <b>Rp {harga_manual:,.0f}</b></h4>",
        unsafe_allow_html=True
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # Copy data agar tidak merusak dataset asli
    df_manual = df.copy()
    df_manual.loc[df_manual.index[-1], "y"] = harga_manual

    # METRIC CARD
    col1, col2, col3 = st.columns(3)

    col1.metric("Harga Terbaru (Dataset)", f"Rp {df['y'].iloc[-1]:,.0f}")
    col2.metric("Harga Sebelumnya", f"Rp {df['y'].iloc[-2]:,.0f}")
    perubahan = ((df['y'].iloc[-1] - df['y'].iloc[-2]) / df['y'].iloc[-2]) * 100
    col3.metric("Perubahan (%)", f"{perubahan:.2f}%")

    st.write("")

    # GRAFIK TREN
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='sub-title'>📊 Tren Harga Beras</div>", unsafe_allow_html=True)

    fig = px.line(
        df, x="ds", y="y",
        title="Trend Harga Beras dari Waktu ke Waktu",
        markers=True
    )
    fig.update_layout(template="simple_white")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # PREDIKSI BULAN DEPAN
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='sub-title'>🔮 Prediksi Harga Bulan Depan</div>", unsafe_allow_html=True)

    model = Prophet()
    model.fit(df_manual)
    future = model.make_future_dataframe(periods=1, freq="MS")
    forecast = model.predict(future)
    pred_next = forecast['yhat'].iloc[-1]

    st.markdown(
        f"<h3 style='color:#0B5A8A;'>Prediksi: <b>Rp {pred_next:,.0f}</b></h3>",
        unsafe_allow_html=True
    )
    st.markdown("</div>", unsafe_allow_html=True)

# 6. PREDIKSI SARIMA
elif selected == "📈 Prediksi SARIMA":

    st.markdown("<div class='title-center'>📈 Prediksi SARIMA</div>", unsafe_allow_html=True)

    periode = st.slider("Pilih Periode Prediksi (bulan)", 1, 36, 12)

    model = SARIMAX(df["y"], order=(1,1,1), seasonal_order=(1,1,1,12))
    fit = model.fit()

    pred = fit.forecast(steps=periode)

    pred_df = pd.DataFrame({
        "Tanggal": pd.date_range(df["ds"].iloc[-1], periods=periode+1, freq="MS")[1:],
        "Prediksi Harga": pred
    })

    fig2 = px.line(
        pred_df, x="Tanggal", y="Prediksi Harga",
        title="Prediksi Harga Beras - SARIMA",
        markers=True
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.dataframe(pred_df, use_container_width=True)

# 7. PREDIKSI PROPHET
elif selected == "🔮 Prediksi Prophet":

    st.markdown("<div class='title-center'>🔮 Prediksi Prophet</div>", unsafe_allow_html=True)

    periode = st.slider("Pilih Periode Prediksi (bulan)", 1, 36, 12)

    model = Prophet()
    model.fit(df)

    future = model.make_future_dataframe(periods=periode, freq="MS")
    forecast = model.predict(future)

    pred_prophet = forecast[["ds", "yhat"]].tail(periode)

    fig3 = px.line(
        pred_prophet, x="ds", y="yhat",
        title="Prediksi Harga Beras - Prophet",
        markers=True
    )
    st.plotly_chart(fig3, use_container_width=True)

    st.dataframe(pred_prophet, use_container_width=True)

# 8. TENTANG
elif selected == "ℹ️ Tentang":

    st.markdown("<div class='title-center'>ℹ️ Tentang Aplikasi</div>", unsafe_allow_html=True)

    st.markdown("""
    Aplikasi ini dikembangkan untuk memprediksi harga beras di Indonesia
    menggunakan **SARIMA** dan **Prophet**.

    ### 🔧 Fitur:
    - Input harga bulan berjalan (manual)
    - Dashboard harga interaktif
    - Grafik tren interaktif (Plotly)
    - Prediksi SARIMA & Prophet
    - Tampilan UI modern dan responsif

    """)

