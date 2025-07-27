import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
import numpy as np

# App Header
st.title("📈 Sales Forecast Studio")
st.markdown("Welcome! This interactive tool helps you explore and forecast your monthly sales data using ARIMA modeling.")

# Upload Section
st.header("🗂️ Upload Your Dataset")
uploaded_file = st.file_uploader("Please upload a CSV file containing sales data with a 'Date' and 'Sales' column.", type="csv")

if uploaded_file:
    # Load data
    df = pd.read_csv(uploaded_file, parse_dates=["Date"])
    df.set_index("Date", inplace=True)
    df = df.resample("M").sum()

    # Show original data
    st.subheader("📊 Visualizing Monthly Sales")
    st.line_chart(df['Sales'])

    # Stationarity Check
    st.subheader("📏 Stationarity Test (ADF)")
    adf_result = adfuller(df['Sales'].dropna())
    st.write(f"**ADF Statistic:** {adf_result[0]:.4f}")
    st.write(f"**p-value:** {adf_result[1]:.4f}")

    if adf_result[1] < 0.05:
        st.success("The time series appears to be stationary — great for ARIMA modeling!")
    else:
        st.warning("The series is not stationary. Applying first-order differencing.")

    # Differencing
    df["Sales_diff"] = df["Sales"].diff()

    st.subheader("🔄 Sales After Differencing")
    fig1, ax1 = plt.subplots()
    ax1.plot(df['Sales_diff'], color='darkgreen')
    ax1.set_title("Differenced Monthly Sales")
    ax1.set_ylabel("Sales Change")
    ax1.set_xlabel("Date")
    ax1.grid(True)
    st.pyplot(fig1)

    # ARIMA Model Training
    st.subheader("🧠 ARIMA Model Training")
    st.info("Fitting an ARIMA(1,1,1) model — adjust these values manually in code if needed.")
    model = ARIMA(df['Sales'], order=(1, 1, 1))
    model_fit = model.fit()
    st.text(model_fit.summary())

    # Forecasting
    st.subheader("📅 Sales Forecast (Next 6 Months)")
    forecast = model_fit.forecast(steps=6)
    forecast.index = pd.date_range(start=df.index[-1] + pd.DateOffset(months=1), periods=6, freq='M')
    st.dataframe(forecast.rename("Forecasted Sales"))

    # Forecast vs Actual Plot
    st.subheader("📉 Forecast vs Historical Sales")
    fig2, ax2 = plt.subplots()
    df['Sales'].plot(ax=ax2, label="Actual", color='navy')
    forecast.plot(ax=ax2, label="Forecast", color='crimson')
    ax2.set_title("Sales Trend: Past vs Future")
    ax2.set_ylabel("Sales")
    ax2.set_xlabel("Time")
    ax2.legend()
    ax2.grid(True)
    st.pyplot(fig2)

    # Model Performance Evaluation
    st.subheader("📊 Model Performance Metrics")
    try:
        pred = model_fit.predict(start=len(df)-6, end=len(df)-1, typ='levels')
        actual = df['Sales'][-6:]
        mape = mean_absolute_percentage_error(actual, pred)
        rmse = np.sqrt(mean_squared_error(actual, pred))

        st.metric("MAPE (Mean Absolute % Error)", f"{mape:.2%}")
        st.metric("RMSE (Root Mean Squared Error)", f"{rmse:.2f}")
    except Exception as e:
        st.warning("Could not compute evaluation metrics. Ensure the dataset has at least 12 months of data.")
