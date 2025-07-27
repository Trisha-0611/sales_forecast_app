import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
import numpy as np

# Step 1: Load and prepare data
df = pd.read_csv("sales_data.csv", parse_dates=["Date"])
df.set_index("Date", inplace=True)
df = df.resample("M").sum()  # Monthly sales aggregation

# Step 2: Plot original sales data
df['Sales'].plot(title="Monthly Sales", figsize=(10, 5))
plt.xlabel("Date")
plt.ylabel("Sales")
plt.grid(True)
plt.show()

# Step 3: Check stationarity using ADF Test
result = adfuller(df['Sales'].dropna())
print("ADF Test:")
print(f"ADF Statistic: {result[0]}")
print(f"p-value: {result[1]}")

# Step 4: Difference the data if not stationary
df['Sales_diff'] = df['Sales'].diff()

# Step 5: Plot differenced data
df['Sales_diff'].plot(title="Differenced Sales")
plt.grid(True)
plt.show()

# Step 6: ACF and PACF plots
plot_acf(df['Sales_diff'].dropna())
plot_pacf(df['Sales_diff'].dropna())
plt.show()

# Step 7: Build and train ARIMA model
model = ARIMA(df['Sales'], order=(1, 1, 1))  # Use (p,d,q) values from ACF/PACF
model_fit = model.fit()

# Step 8: Print model summary
print(model_fit.summary())

# Step 9: Forecast next 6 months
forecast = model_fit.forecast(steps=6)
print("Forecasted Sales for the Next 6 Months:")
print(forecast)

# Step 10: Plot forecast
df['Sales'].plot(label='Actual', figsize=(10, 5))
forecast.index = pd.date_range(start=df.index[-1] + pd.DateOffset(months=1), periods=6, freq='M')
forecast.plot(label='Forecast')
plt.title("Sales Forecast")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.legend()
plt.grid(True)
plt.show()

# Step 11: Optional - Evaluate model on past data
pred = model_fit.predict(start=len(df)-6, end=len(df)-1, typ='levels')
mape = mean_absolute_percentage_error(df['Sales'][-6:], pred)
rmse = np.sqrt(mean_squared_error(df['Sales'][-6:], pred))
print(f"MAPE: {mape:.2f}")
print(f"RMSE: {rmse:.2f}")
