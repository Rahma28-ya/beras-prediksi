import streamlit as st
import pandas as pd
import plotly.express as px
from prophet import Prophet
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
warnings.filterwarnings("ignore")

# 1. KONFIGURASI HALAMAN
st.set_page_config(
    page_title="Prediksi Harga Beras Indonesia",
    page_icon="🌾",
    layout="wide"
)

# 2. SIDEBAR NAVIGASI
from streamlit_option_menu import option_menu

with st.sidebar:
    selected = option_menu(
        "Navigasi",
        ["Dashboard", "Prediksi SARIMA", "Prediksi Prophet", "Tentang"],
        icons=["bar-chart", "graph-up", "graph-up-arrow", "info-circle"],
        menu_icon="cast",
        default_index=0,
    )

# 3. LOAD DATA
@st.cache_data
def load_data():
    df = pd.read_csv("bps_final.csv")
    df["tanggal"] = pd.to_datetime(df["tanggal"])
    df = df.rename(columns={"tanggal": "ds", "harga": "y"})
    return df

df = load_data()

# 4. DASHBOARD UTAMA
if selected == "Dashboard":
    st.markdown(
        "<h2 style='color:#FF6F61; text-align:center;'>🌾 Dashboard Harga Beras Indonesia 🌾</h2>",
        unsafe_allow_html=True
    )
    st.write("---")

    # Input harga bulan ini
    harga_manual = st.number_input(
        "Masukkan harga bulan ini (Rp)", 
        min_value=0, 
        value=int(df['y'].iloc[-1])
    )
    st.markdown(
        f"<p style='color:#1F618D; font-weight:bold;'>Harga bulan ini: Rp {harga_manual:,.0f}</p>",
        unsafe_allow_html=True
    )

    # Card Metrics
    harga_terakhir = df["y"].iloc[-1]
    harga_awal = df["y"].iloc[-2]
    persen_perubahan = ((harga_terakhir - harga_awal) / harga_awal) * 100

    col1, col2, col3 = st.columns(3)
    col1.metric("Harga Terbaru", f"Rp {harga_terakhir:,.0f}")
    col2.metric("Harga Sebelumnya", f"Rp {harga_awal:,.0f}")

    warna = "green" if persen_perubahan >= 0 else "red"
    col3.markdown(
        f"<h3 style='color:{warna};'>{persen_perubahan:.2f}%</h3>",
        unsafe_allow_html=True
    )
    st.write("---")

    # Grafik harga
    fig = px.line(
        df, x="ds", y="y",
        title="Trend Harga Beras dari Waktu ke Waktu",
        markers=True,
        color_discrete_sequence=["#FF5733"]
    )
    fig.update_layout(template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

    # Prediksi bulan depan berdasarkan harga manual menggunakan Prophet
    df_manual = df.copy()
    df_manual.loc[df_manual.index[-1], "y"] = harga_manual

    model = Prophet()
    model.fit(df_manual)
    future = model.make_future_dataframe(periods=1, freq="MS")
    forecast = model.predict(future)
    pred_bulan_depan = forecast["yhat"].iloc[-1]

    st.markdown(
        f"<h3 style='color:#1F618D;'>Prediksi harga bulan depan berdasarkan harga manual: Rp {pred_bulan_depan:,.0f}</h3>",
        unsafe_allow_html=True
    )

# 5. PREDIKSI DENGAN SARIMA
elif selected == "Prediksi SARIMA":
    st.markdown(
        "<h2 style='color:#28B463; text-align:center;'>📈 Prediksi Harga Beras Menggunakan SARIMA</h2>",
        unsafe_allow_html=True
    )

    periode = st.slider("Pilih Periode Prediksi (bulan)", 1, 24, 12)
    model = SARIMAX(df["y"], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    model_fit = model.fit(disp=False)
    pred = model_fit.forecast(steps=periode)

    pred_df = pd.DataFrame({
        "Tanggal": pd.date_range(start=df["ds"].iloc[-1], periods=periode+1, freq="MS")[1:],
        "Prediksi Harga": pred
    })

    fig2 = px.line(
        pred_df, x="Tanggal", y="Prediksi Harga",
        title="Prediksi Harga Beras - SARIMA",
        markers=True,
        color_discrete_sequence=["#28B463"]
    )
    fig2.update_layout(template="plotly_white")
    st.plotly_chart(fig2, use_container_width=True)
    st.dataframe(pred_df.style.background_gradient(cmap='YlGn'))

# 6. PREDIKSI DENGAN PROPHET
elif selected == "Prediksi Prophet":
    st.markdown(
        "<h2 style='color:#AF7AC5; text-align:center;'>🔮 Prediksi Harga Beras Menggunakan Prophet</h2>",
        unsafe_allow_html=True
    )

    periode = st.slider("Pilih Periode Prediksi (bulan)", 1, 24, 12, key="p")
    model = Prophet()
    model.fit(df)
    future = model.make_future_dataframe(periods=periode, freq="MS")
    forecast = model.predict(future)
    pred_prophet = forecast[["ds", "yhat"]].tail(periode)

    fig3 = px.line(
        pred_prophet, x="ds", y="yhat",
        title="Prediksi Harga Beras - Prophet",
        markers=True,
        color_discrete_sequence=["#AF7AC5"]
    )
    fig3.update_layout(template="plotly_white")
    st.plotly_chart(fig3, use_container_width=True)
    st.dataframe(pred_prophet.style.background_gradient(cmap='Purples'))

# 7. TENTANG APLIKASI
elif selected == "Tentang":
    st.title("Tentang Aplikasi")
    st.write("""
    Aplikasi ini dikembangkan untuk memprediksi harga beras di Indonesia 
    menggunakan dua model peramalan: **SARIMA** dan **Prophet**.  
    Data berasal dari **BPS (data bulanan)** dan telah melalui proses pembersihan.  

    Fitur aplikasi:
    - Dashboard trend interaktif
    - Input harga bulan ini & prediksi bulan depan
    - Prediksi SARIMA
    - Prediksi Prophet
    - Grafik interaktif Plotly
    - Tampilan modern dengan sidebar navigation
    """)
