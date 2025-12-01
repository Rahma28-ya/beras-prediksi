import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pmdarima as pm

st.title("Prediksi Harga Beras di Indonesia")
st.write("Aplikasi ini menampilkan prediksi harga beras menggunakan SARIMA dan Prophet.")

# LOAD DATA
df = pd.read_csv("data/bps_final.csv")
df["tanggal"] = pd.to_datetime(df["tanggal"])
df = df.set_index("tanggal")

st.subheader("Data Harga Beras")
st.line_chart(df)

# TRAIN-TEST SPLIT
train_size = int(len(df) * 0.8)
train = df.iloc[:train_size]
test = df.iloc[train_size:]

# MODEL SARIMA
st.subheader("🔧 Training SARIMA...")

train_log = np.log(train["harga"])

model_auto = pm.auto_arima(
    train_log, seasonal=True, m=12, stepwise=True, error_action="ignore"
)

order = model_auto.order
seasonal_order = model_auto.seasonal_order

sarima_model = SARIMAX(train_log, order=order, seasonal_order=seasonal_order).fit()
sarima_pred_log = sarima_model.predict(start=test.index[0], end=test.index[-1])
sarima_pred = np.exp(sarima_pred_log)

# MODEL PROPHET
st.subheader(" Training Prophet...")

prophet_df = df.reset_index().rename(columns={"tanggal": "ds", "harga": "y"})
prophet_train = prophet_df.iloc[:train_size]

m = Prophet()
m.fit(prophet_train)

future_test = prophet_df.iloc[train_size:][["ds"]]
prophet_pred = m.predict(future_test)["yhat"]

# VISUALISASI
st.subheader(" Perbandingan Prediksi SARIMA vs Prophet")

fig, ax = plt.subplots(figsize=(10,5))
ax.plot(df.index, df["harga"], label="Data Aktual")
ax.plot(test.index, sarima_pred, label="Prediksi SARIMA")
ax.plot(test.index, prophet_pred, label="Prediksi Prophet")
ax.legend()
st.pyplot(fig)

# FORECAST 12 BULAN KE DEPAN
st.subheader(" Prediksi 12 Bulan ke Depan")

sarima_future_log = sarima_model.predict(
    start=df.index[-1] + pd.offsets.MonthBegin(1),
    end=df.index[-1] + pd.offsets.MonthBegin(12)
)
sarima_future = np.exp(sarima_future_log)

future = m.make_future_dataframe(periods=12, freq="MS")
forecast = m.predict(future).tail(12)[["ds", "yhat"]]

col1, col2 = st.columns(2)

with col1:
    st.write(" **Prediksi SARIMA 12 Bulan**")
    st.dataframe(sarima_future)

with col2:
    st.write(" **Prediksi Prophet 12 Bulan**")
    st.dataframe(forecast)
