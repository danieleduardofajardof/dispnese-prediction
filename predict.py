import polars as pl
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from datetime import datetime
import json
# Load CSV into Polars DataFrame
dfp_model = pl.read_csv("model_input_export.csv",
    infer_schema_length=10000 )
# Suppose 'hour' is your index/timestamp column
index_col = "ts"

# Convert all other columns to float
dfp_model = dfp_model.with_columns([
    pl.col(c).cast(pl.Float64) for c in dfp_model.columns if c != index_col
])

# Preview the structure
dfp_model = dfp_model.fill_null(0)


# Ensure your date column is parsed as datetime
dfp_model = dfp_model.with_columns(
    pl.col("ts").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S %Z", strict=False)
)

# Sort by date
dfp_model = dfp_model.sort("ts")

dfp_model = dfp_model.tail(3000)

target_cols = [c for c in dfp_model.columns if  "grams" in c]
exog_cols = [c for c in dfp_model.columns if c not in target_cols and c != "ts"]



df = dfp_model.to_pandas()
df.set_index("ts", inplace=True)


def predict_to_pd(pdf, col, future_periods):
    lags = future_periods
    for lag in range(1, lags+1):
        df[f"lag_{lag}"] = df[col].shift(lag)

    df.dropna(inplace=True)

    X = df[[f"lag_{i}" for i in range(1, lags+1)] + exog_cols]
    y = df[col]

    # Keep index when splitting
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.02, shuffle=False
    )

    # Train model
    model = xgb.XGBRegressor()
    model.fit(X_train, y_train)
    # Last row of the data to start building lags
    last_row = pdf.iloc[-1]

    # Prepare a future DataFrame
    future_index = pd.date_range(start=pdf.index[-1] + pd.Timedelta(hours=1), periods=future_periods, freq='h')
    future_df = pd.DataFrame(index=future_index)
    
    # Initialize lag columns
    for lag in range(1, lags+1):
        future_df[f"lag_{lag}"] = None

    # Fill initial lag values from last observed data
    lag_values = pdf[col].values[-lags:].tolist()

    # Iteratively create lag features for future predictions
    future_predictions = []
    for ts in future_index:
        X_row = pd.DataFrame([lag_values + [last_row[col] for col in exog_cols]], columns=[f"lag_{i}" for i in range(1, lags+1)] + exog_cols)
        pred = model.predict(X_row)[0]
        # Apply your correction
        pred_corrected = max(pred*1.7 + 500, 500)
        future_predictions.append(pred_corrected)

        # Update lag_values for next step
        lag_values = lag_values[1:] + [pred_corrected]

    # Add predictions to future DataFrame
    future_df["predicted_"+col] = future_predictions
    future_df = future_df[["predicted_"+col]]
    # Export to CSV


    return future_df
    
future_periods = 24  # 24-hour horizon
all_preds = None
for col in target_cols:
    preds = predict_to_pd(df, col, future_periods)
    if(all_preds is None):
        all_preds= preds
    else:
        all_preds = all_preds.join(preds, how="outer")

print(all_preds.head())

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")


# Convert index (timestamps) to strings and rows to dicts
result_json = {str(ts): row.to_dict() for ts, row in all_preds.iterrows()}

# Save to a JSON file
with open("predictions_{}.json".format(timestamp), "w") as f:
    json.dump(result_json, f, indent=4)
