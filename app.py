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
# 4. DASHBOARD UTAMA (VERSI PASTEL GEMAS)
# =================================================================
if selected == "Dashboard":
    st.markdown(
        "<h2 style='color:#FF6F61; text-align:center;'>🌾 Dashboard Harga Beras Indonesia 🌾</h2>",
        unsafe_allow_html=True
    )
    st.write("---")

    # =======================
    # CARD METRICS UTAMA
    # =======================
    harga_terbaru = df["y"].iloc[-1]
    harga_bulan_lalu = df["y"].iloc[-2]
    rata_rata = df["y"].mean()
    min_harga = df["y"].min()
    max_harga = df["y"].max()
    volatilitas = df["y"].pct_change().std() * 100

    st.markdown("""
<style>
/* Background halaman utama */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(160deg, #FFF0F5, #E0FFFF);
}

/* Sidebar pastel */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #FFE4E1, #E6E6FA);
}

/* Card Metrics */
.card {
    border-radius:15px;
    padding:15px;
    text-align:center;
    font-weight:bold;
    color:#333;
    box-shadow: 0 4px 15px rgba(0,0,0,0.08);
    margin-bottom:10px;
}

/* Font besar gemas */
.big-font { font-size:28px; font-weight:700; }

/* Card pilihan periode hover */
.card-option:hover {
    transform: scale(1.05);
    background: #FDE2FF !important;
    border-color:#FFB6C1 !important;
}

/* Card terpilih */
.selected-card {
    border-color: #FF69B4 !important;
    background: #FFD1EC !important;
    transform: scale(1.05);
}
</style>
""", unsafe_allow_html=True)

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.markdown(f'<div class="card" style="background-color:#FFB6C1;"><div class="big-font">Rp {harga_terbaru:,.0f}</div>Harga Terbaru</div>', unsafe_allow_html=True)
    col2.markdown(f'<div class="card" style="background-color:#CBA0E3;"><div class="big-font">Rp {harga_bulan_lalu:,.0f}</div>Bulan Lalu</div>', unsafe_allow_html=True)
    col3.markdown(f'<div class="card" style="background-color:#A0E7E5;"><div class="big-font">Rp {rata_rata:,.0f}</div>Rata-rata</div>', unsafe_allow_html=True)
    col4.markdown(f'<div class="card" style="background-color:#FFFACD;"><div class="big-font">Rp {min_harga:,.0f}</div>Termurah</div>', unsafe_allow_html=True)
    col5.markdown(f'<div class="card" style="background-color:#FFDAB9;"><div class="big-font">{volatilitas:.2f}%</div>Volatilitas</div>', unsafe_allow_html=True)

    st.write("---")

    # =======================
    # TREND CHART
    # =======================
    fig = px.line(
        df, x="ds", y="y",
        title="📈 Trend Harga Beras dari Waktu ke Waktu",
        markers=True,
        template="plotly_white",
        color_discrete_sequence=["#FF6F61"]
    )
    st.plotly_chart(fig, use_container_width=True)

    # =======================
    # HEATMAP MUSIMAN
    # =======================
    df["year"] = df["ds"].dt.year
    df["month"] = df["ds"].dt.month
    pivot = df.pivot_table(values="y", index="year", columns="month")

    st.subheader("🗓️ Pola Musiman Harga Beras (Heatmap)")
    fig_heatmap = px.imshow(
        pivot,
        labels=dict(x="Bulan", y="Tahun", color="Harga"),
        aspect="auto",
        color_continuous_scale=px.colors.sequential.Mint
    )
    st.plotly_chart(fig_heatmap, use_container_width=True)

    # =======================
    # BOXPLOT DISTRIBUSI
    # =======================
    st.subheader("📊 Distribusi Harga Beras (Boxplot)")
    fig_box = px.box(
        df, y="y",
        title="Distribusi Harga Beras",
        color_discrete_sequence=["#B19CD9"]
    )
    st.plotly_chart(fig_box, use_container_width=True)

    # =======================
    # INSIGHT TAMBAHAN
    # =======================
    st.subheader("✨ Insight Tambahan")
    max_per_year = df.groupby("year")["y"].max()
    min_per_year = df.groupby("year")["y"].min()
    st.write("📌 Harga Maksimum per Tahun:")
    st.table(max_per_year.reset_index().rename(columns={"y":"Harga Maks"}))
    st.write("📌 Harga Minimum per Tahun:")
    st.table(min_per_year.reset_index().rename(columns={"y":"Harga Min"}))

    # Bulan dengan kenaikan tertinggi
    df["delta"] = df["y"].diff()
    top_inc = df.loc[df["delta"].idxmax()]
    st.info(f"🔥 Kenaikan tertinggi: {top_inc['ds'].strftime('%B %Y')} sebesar Rp {top_inc['delta']:,.0f}")

    # MINI FORECAST 1 BULAN
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

    # --- UI Estetik Pilih Periode ---
    st.markdown("""
    <style>
    .card-option {
        padding: 14px;
        border-radius: 12px;
        background: #ffffffcc;
        border: 2px solid #eee;
        text-align: center;
        transition: 0.2s;
        cursor: pointer;
    }
    .card-option:hover {
        border-color: #28b463;
        transform: scale(1.03);
        background: #e6f7ea;
    }
    .selected-card {
        border-color: #28B463 !important;
        background: #d4f3d4 !important;
        transform: scale(1.05);
    }
    </style>
    """, unsafe_allow_html=True)

    st.subheader("🎨 Pilih Periode Prediksi")

    cols = st.columns(3)
    labels = ["12 Bulan", "24 Bulan", "36 Bulan"]
    values = [12, 24, 36]

    if "periode_sarima" not in st.session_state:
        st.session_state.periode_sarima = 12

    for i, col in enumerate(cols):
        with col:
            if st.button(labels[i], key=f"sarima_btn_{i}"):
                st.session_state.periode_sarima = values[i]

            st.markdown(
                f"<div class='{'selected-card' if st.session_state.periode_sarima == values[i] else 'card-option'}'>"
                f"<h4>{labels[i]}</h4></div>",
                unsafe_allow_html=True
            )

    periode = st.slider(
        "Atau pilih manual (1–36 bulan)",
        1, 36,
        st.session_state.periode_sarima,
        key="sarima_slider"
    )
    st.session_state.periode_sarima = periode

    # Forecast SARIMA
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

    # --- UI Estetik Pilih Periode ---
    st.markdown("""
    <style>
    .card-option {
        padding: 14px;
        border-radius: 12px;
        background: #ffffffcc;
        border: 2px solid #eee;
        text-align: center;
        transition: 0.2s;
        cursor: pointer;
    }
    .card-option:hover {
        border-color: #AF7AC5;
        transform: scale(1.03);
        background: #f3eaff;
    }
    .selected-card {
        border-color: #AF7AC5 !important;
        background: #e8daef !important;
        transform: scale(1.05);
    }
    </style>
    """, unsafe_allow_html=True)

    st.subheader("🔮 Pilih Periode Prediksi Prophet")

    cols = st.columns(3)
    labels = ["12 Bulan", "24 Bulan", "36 Bulan"]
    values = [12, 24, 36]

    if "periode_prophet" not in st.session_state:
        st.session_state.periode_prophet = 12

    for i, col in enumerate(cols):
        with col:
            if st.button(labels[i], key=f"prophet_btn_{i}"):
                st.session_state.periode_prophet = values[i]

            st.markdown(
                f"<div class='{'selected-card' if st.session_state.periode_prophet == values[i] else 'card-option'}'>"
                f"<h4>{labels[i]}</h4></div>",
                unsafe_allow_html=True
            )

    periode = st.slider(
        "Atau pilih manual (1–36 bulan)",
        1, 36,
        st.session_state.periode_prophet,
        key="prophet_slider"
    )
    st.session_state.periode_prophet = periode

    # Forecast Prophet
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




