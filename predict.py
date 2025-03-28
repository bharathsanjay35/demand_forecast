import pickle
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Load the trained model
with open("demand_forecast_model.pkl", "rb") as f:
    model = pickle.load(f)

# Define the number of days to predict
future_steps = 30  # Predict for the next 30 days

# Generate future dates based on the last date in the dataset
last_date = pd.to_datetime("2022-04-03")  # Replace with the actual last date
future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=future_steps, freq="D")

# Make predictions
predictions = model.get_forecast(steps=future_steps).predicted_mean

# Create a DataFrame to store results
forecast_df = pd.DataFrame({"Date": future_dates, "Predicted_Demand": predictions})

# Plot the predictions
plt.figure(figsize=(12, 6))
plt.plot(future_dates, predictions, marker="o", linestyle="-", color="b", label="Predicted Demand")
plt.xlabel("Date")
plt.ylabel("Predicted Quantity Sold")
plt.title("Future Demand Forecast")
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.show()

# Save predictions to a CSV file
forecast_df.to_csv("predictions.csv", index=False)
print("✅ Predictions saved to 'predictions.csv'")
