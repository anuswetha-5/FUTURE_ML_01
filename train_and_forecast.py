import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load dataset
df = pd.read_csv("data/superstore_sales.csv", encoding="latin1")

# Convert date
df['Order Date'] = pd.to_datetime(df['Order Date'])

# Monthly aggregation
monthly_sales = (
    df.groupby(df['Order Date'].dt.to_period('M'))['Sales']
    .sum()
    .reset_index()
)

monthly_sales['Order Date'] = monthly_sales['Order Date'].dt.to_timestamp()

# Create time index
monthly_sales['Time_Index'] = range(len(monthly_sales))

# Train model
X = monthly_sales[['Time_Index']]
y = monthly_sales['Sales']

model = LinearRegression()
model.fit(X, y)

# Predict next 12 months
future_time = pd.DataFrame({
    'Time_Index': range(len(monthly_sales), len(monthly_sales) + 12)
})

future_sales = model.predict(future_time)

# Plot
plt.figure(figsize=(10, 5))
plt.plot(monthly_sales['Order Date'], monthly_sales['Sales'], label='Actual Sales')
plt.plot(
    pd.date_range(start=monthly_sales['Order Date'].iloc[-1], periods=13, freq='M')[1:],
    future_sales,
    linestyle='--',
    label='Forecast'
)

plt.legend()
plt.title("Sales Forecast (Next 12 Months)")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.tight_layout()
plt.savefig("output/forecast_plot.png")
plt.show()
