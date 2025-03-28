import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Load the dataset
df = pd.read_csv("Balaji_Fast_Food_Sales_with_Festivals.csv")

# ✅ Debugging Step: Print first 5 rows
print(df.head())  # <--- Add this here

# ✅ Fix Date Parsing (DD-MM-YYYY format)
df["date"] = pd.to_datetime(df["date"], format="%d-%m-%Y", errors="coerce")

# ✅ Check for invalid dates
if df["date"].isna().sum() > 0:
    print("⚠️ Warning: Some dates could not be parsed. Check your CSV format!")

# ✅ Ensure expected column exists
expected_column = "total_quantity"  # Adjust this if needed
if expected_column not in df.columns:
    raise ValueError(f"Column '{expected_column}' not found. Available columns: {df.columns}")

# ✅ Set Date as Index
df.set_index("date", inplace=True)

# ✅ Resample Data to Daily Frequency
daily_sales = df.resample("D").sum()

# ✅ Train SARIMA Model
model = SARIMAX(daily_sales[expected_column], order=(1,1,1), seasonal_order=(1,1,1,7))
model_fit = model.fit()

# ✅ Save the trained model
model_fit.save("demand_forecast_model.pkl")

print("✅ Model training complete! Saved as 'demand_forecast_model.pkl'.")
