import streamlit as st
import pandas as pd
import plotly.express as px
from prophet import Prophet
from statsmodels.tsa.statespace.sarimax import SARIMAX
from streamlit_option_menu import option_menu
import warnings
warnings.filterwarnings("ignore")

# =================================================================
# 1. KONFIGURASI HALAMAN
# =================================================================
st.set_page_config(
    page_title="Prediksi Harga Beras Indonesia",
    page_icon="🌾",
    layout="wide"
)

# =================================================================
# 2. SIDEBAR NAVIGASI
# =================================================================
with st.sidebar:
    st.markdown("### 📂 Upload Dataset Manual (Opsional)")
    uploaded = st.file_uploader("Unggah file CSV", type=["csv"])

    selected = option_menu(
        "Navigasi",
        ["Dashboard", "Prediksi SARIMA", "Prediksi Prophet", "Tentang"],
        icons=["speedometer", "activity", "boxes", "info-circle"],
        menu_icon="cast",
        default_index=0,
    )

# =================================================================
# 3. LOAD DATA
# =================================================================
@st.cache_data
def load_default():
    df = pd.read_csv("bps_final.csv")
    df["tanggal"] = pd.to_datetime(df["tanggal"], errors="coerce")
    df = df.rename(columns={"tanggal": "ds", "harga": "y"})
    df = df.dropna(subset=["ds", "y"])
    return df

def load_uploaded(file):
    df = pd.read_csv(file)
    # pastikan kolom bernama "tanggal" dan "harga"
    if "tanggal" not in df.columns or "harga" not in df.columns:
        st.error("Dataset harus memiliki kolom: 'tanggal' dan 'harga'")
        return None
    df["tanggal"] = pd.to_datetime(df["tanggal"], errors="coerce")
    df = df.rename(columns={"tanggal": "ds", "harga": "y"})
    df = df.dropna(subset=["ds", "y"])
    return df

# pilih dataset
if uploaded:
    df = load_uploaded(uploaded)
    if df is None:
        st.stop()
else:
    df = load_default()

# =================================================================
# 4. DASHBOARD UTAMA (VERSI CANTIK & BANYAK INSIGHT)
# =================================================================
if selected == "Dashboard":
    st.markdown(
        "<h2 style='color:#FF6F61; text-align:center;'>🌾 Dashboard Harga Beras Indonesia 🌾</h2>",
        unsafe_allow_html=True
    )
    st.write("---")

    # =======================
    # INSIGHT UTAMA
    # =======================
    harga_terbaru = df["y"].iloc[-1]
    harga_bulan_lalu = df["y"].iloc[-2]
    rata_rata = df["y"].mean()
    min_harga = df["y"].min()
    max_harga = df["y"].max()
    volatilitas = df["y"].pct_change().std() * 100

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Harga Terbaru", f"Rp {harga_terbaru:,.0f}")
    col2.metric("Bulan Lalu", f"Rp {harga_bulan_lalu:,.0f}")
    col3.metric("Rata-rata", f"Rp {rata_rata:,.0f}")
    col4.metric("Termurah", f"Rp {min_harga:,.0f}")
    col5.metric("Volatilitas (%)", f"{volatilitas:.2f}")

    st.write("---")

    # =======================
    # GRAFIK TREND UTAMA
    # =======================
    fig = px.line(
        df, x="ds", y="y",
        title="📈 Trend Harga Beras dari Waktu ke Waktu",
        markers=True,
        template="plotly_white",
        color_discrete_sequence=["#FF5733"]
    )
    st.plotly_chart(fig, use_container_width=True)

    # =======================
    # GRAFIK MUSIMAN (HEATMAP)
    # =======================
    df["year"] = df["ds"].dt.year
    df["month"] = df["ds"].dt.month

    pivot = df.pivot_table(values="y", index="year", columns="month")

    st.subheader("🗓️ Pola Musiman Harga Beras (Heatmap)")
    fig_heatmap = px.imshow(
        pivot,
        labels=dict(x="Bulan", y="Tahun", color="Harga"),
        aspect="auto",
        color_continuous_scale="YlOrRd"
    )
    st.plotly_chart(fig_heatmap, use_container_width=True)

    # =======================
    # DISTRIBUSI HARGA
    # =======================
    st.subheader("📊 Distribusi Harga Beras (Boxplot)")
    fig_box = px.box(
        df, y="y",
        title="Distribusi Harga Beras",
        color_discrete_sequence=["#2980B9"]
    )
    st.plotly_chart(fig_box, use_container_width=True)

    # =======================
    # DETEKSI ANOMALI / OUTLIER
    # =======================
    Q1 = df["y"].quantile(0.25)
    Q3 = df["y"].quantile(0.75)
    IQR = Q3 - Q1
    outliers = df[(df["y"] < Q1 - 1.5 * IQR) | (df["y"] > Q3 + 1.5 * IQR)]

    st.subheader("🚨 Anomali Harga Beras")

    if outliers.empty:
        st.success("Tidak ada anomali harga yang signifikan. 👍")
    else:
        st.warning(f"Ditemukan {len(outliers)} anomali harga.")
        st.dataframe(outliers)

    # =======================
    # KENAIKAN TERBESAR PER BULAN
    # =======================
    df["delta"] = df["y"].diff()
    top_increase = df.nlargest(1, "delta")

    st.subheader("🔥 Bulan dengan Kenaikan Tertinggi")

    if not top_increase.empty:
        ds_top = top_increase["ds"].iloc[0].strftime("%B %Y")
        naik = top_increase["delta"].iloc[0]
        st.info(f"**Kenaikan tertinggi terjadi pada {ds_top} sebesar Rp {naik:,.0f}**")

    # =======================
    # MINI FORECAST (1 BULAN)
    # =======================
    model = Prophet()
    model.fit(df)
    future = model.make_future_dataframe(periods=1, freq="MS")
    forecast = model.predict(future)

    pred_next = forecast["yhat"].iloc[-1]

    st.subheader("📌 Prediksi Mini (1 Bulan ke Depan)")
    st.success(f"Perkiraan harga bulan depan: **Rp {pred_next:,.0f}**")

# =================================================================
# 5. PREDIKSI DENGAN SARIMA
# =================================================================
elif selected == "Prediksi SARIMA":
    st.markdown(
        "<h2 style='color:#28B463; text-align:center;'>📈 Prediksi Harga Beras Menggunakan SARIMA</h2>",
        unsafe_allow_html=True
    )

    periode = st.slider("Pilih Periode Prediksi (bulan)", 1, 36, 12)

    model = SARIMAX(df["y"], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    model_fit = model.fit(disp=False)
    pred = model_fit.forecast(steps=periode)

    pred_df = pd.DataFrame({
        "Tanggal": pd.date_range(start=df["ds"].iloc[-1], periods=periode + 1, freq="MS")[1:],
        "Prediksi Harga": pred
    })

    fig2 = px.line(
        pred_df, x="Tanggal", y="Prediksi Harga",
        title="Prediksi Harga Beras - SARIMA",
        markers=True,
        color_discrete_sequence=["#28B463"]
    )
    st.plotly_chart(fig2, use_container_width=True)
    st.dataframe(pred_df)

# =================================================================
# 6. PREDIKSI DENGAN PROPHET
# =================================================================
elif selected == "Prediksi Prophet":
    st.markdown(
        "<h2 style='color:#AF7AC5; text-align:center;'>🔮 Prediksi Harga Beras Menggunakan Prophet</h2>",
        unsafe_allow_html=True
    )

    periode = st.slider("Pilih Periode Prediksi (bulan)", 1, 36, 12, key="Pr")

    model = Prophet()
    model.fit(df)
    future = model.make_future_dataframe(periods=periode, freq="MS")
    forecast = model.predict(future)

    pred_df = forecast[["ds", "yhat"]].tail(periode)

    fig3 = px.line(
        pred_df, x="ds", y="yhat",
        title="Prediksi Harga Beras - Prophet",
        markers=True,
        color_discrete_sequence=["#AF7AC5"]
    )
    st.plotly_chart(fig3, use_container_width=True)
    st.dataframe(pred_df)

# =================================================================
# 7. HALAMAN TENTANG
# =================================================================
elif selected == "Tentang":
    st.title("Tentang Aplikasi")
    st.write("""
    Aplikasi ini dikembangkan untuk memprediksi harga beras di Indonesia 
    menggunakan dua model peramalan: **SARIMA** dan **Prophet**.  

    Fitur:
    - Upload dataset manual
    - Dashboard trend interaktif
    - Prediksi SARIMA & Prophet
    - Input harga bulan ini untuk prediksi manual
    - Grafik interaktif Plotly
    """)
