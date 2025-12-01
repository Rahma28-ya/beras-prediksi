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
    # Judul dashboard berwarna
    st.markdown(
        "<h2 style='color:#FF6F61; text-align:center;'>🌾 Dashboard Harga Beras Indonesia 🌾</h2>",
        unsafe_allow_html=True
    )
    st.write("---")

    # Input harga manual
    harga_input = st.number_input(
        "Masukkan harga manual (Rp)", 
        min_value=0, 
        value=int(df['y'].iloc[-1])
    )
    st.markdown(
        f"<p style='color:#1F618D; font-weight:bold;'>Harga manual yang dimasukkan: Rp {harga_input:,.0f}</p>",
        unsafe_allow_html=True
    )

    # Card Metrics
    harga_terakhir = df["y"].iloc[-1]
    harga_awal = df["y"].iloc[-2]
    persen_perubahan = ((harga_terakhir - harga_awal) / harga_awal) * 100

    col1, col2, col3 = st.columns(3)
    col1.metric("Harga Terbaru", f"Rp {harga_terakhir:,.0f}")
    col2.metric("Harga Sebelumnya", f"Rp {harga_awal:,.0f}")

    # Warna card perubahan (%)
    warna = "green" if persen_perubahan >= 0 else "red"
    col3.markdown(
        f"<h3 style='color:{warna};'>{persen_perubahan:.2f}%</h3>",
        unsafe_allow_html=True
    )
    st.write("---")

    # Grafik Interaktif Harga dengan warna garis custom
    fig = px.line(
        df, x="ds", y="y",
        title="Trend Harga Beras dari Waktu ke Waktu",
        markers=True,
        color_discrete_sequence=["#FF5733"]
    )
    fig.update_layout(template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

# 5. PREDIKSI DENGAN SARIMA
elif selected == "Prediksi SARIMA":
    st.markdown(
        "<h2 style='color:#FF5733; text-align:center;'>📈 Prediksi Harga Beras Menggunakan SARIMA</h2>",
        unsafe_allow_html=True
    )

    periode = st.slider("Pilih Periode Prediksi (bulan)", 1, 24, 12)

    # Model SARIMA
    model = SARIMAX(df["y"], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    model_fit = model.fit(disp=False)

    pred = model_fit.forecast(steps=periode)

    # Tampilkan dataframe prediksi
    pred_df = pd.DataFrame({
        "Tanggal": pd.date_range(start=df["ds"].iloc[-1], periods=periode+1, freq="MS")[1:],
        "Prediksi Harga": pred
    })

    # Grafik dengan warna garis custom
    fig2 = px.line(
        pred_df, x="Tanggal", y="Prediksi Harga",
        title="Prediksi Harga Beras - SARIMA",
        markers=True,
        color_discrete_sequence=["#28B463"]
    )
    fig2.update_layout(template="plotly_white")
    st.plotly_chart(fig2, use_container_width=True)

    # Tabel berwarna gradient
    st.dataframe(pred_df.style.background_gradient(cmap='YlGn'))

# 6. PREDIKSI DENGAN PROPHET
elif selected == "Prediksi Prophet":
    st.markdown(
        "<h2 style='color:#5DADE2; text-align:center;'>🔮 Prediksi Harga Beras Menggunakan Prophet</h2>",
        unsafe_allow_html=True
    )

    periode = st.slider("Pilih Periode Prediksi (bulan)", 1, 24, 12, key="p")

    # Prophet sudah menggunakan kolom ds & y
    model = Prophet()
    model.fit(df)

    future = model.make_future_dataframe(periods=periode, freq="MS")
    forecast = model.predict(future)

    # Ambil hanya bagian prediksi
    pred_prophet = forecast[["ds", "yhat"]].tail(periode)

    # Grafik dengan warna garis custom
    fig3 = px.line(
        pred_prophet, x="ds", y="yhat",
        title="Prediksi Harga Beras - Prophet",
        markers=True,
        color_discrete_sequence=["#AF7AC5"]
    )
    fig3.update_layout(template="plotly_white")
    st.plotly_chart(fig3, use_container_width=True)

    # Tabel berwarna gradient
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
    - Prediksi SARIMA
    - Prediksi Prophet
    - Grafik interaktif Plotly
    - Tampilan modern dengan sidebar navigation
    """)
