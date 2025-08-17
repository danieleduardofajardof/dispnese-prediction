import json
import polars as pl
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from datetime import datetime

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

def metrics_dict(pd_dataframe, lags, col_pred):
    for lag in range(1, lags+1):
        pd_dataframe[f"lag_{lag}"] = pd_dataframe[col_pred].shift(lag)

    pd_dataframe.dropna(inplace=True)

    X = pd_dataframe[[f"lag_{i}" for i in range(1, lags+1)] + exog_cols]
    y = pd_dataframe[col_pred]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = xgb.XGBRegressor()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    corrected_y_pred = [x*1.7+500 if x > 0 else 500 for x in y_pred]
    mun =  munderest(y_test, y_pred)
    mun_corrected =  munderest(y_test, corrected_y_pred)
    mun_pct = munderest_pct(y_test, y_pred)
    mun_pct_corrected = munderest_pct(y_test, corrected_y_pred)
    print("Results for column named: ", col_pred," with horizon of ", lags," steps ahead." )
    print(f"Mean Under Estimation: {mun:.4f}")
    print(f"Mean Under Estimation after correction: {mun_corrected:.4f}")
    print(f"Mean Percent Under Estimation: {mun_pct:.4f}")
    print(f"Mean Percent Under Estimation after correction: {mun_pct_corrected:.4f}")
    return {col_pred+"MUN_corrected":mun_corrected, col_pred+"MUN_pct":mun_pct_corrected}
    
def munderest(y_actual, y_hat):
    total_under = sum(max(0, yi - yh) for yi, yh in zip(y_actual, y_hat))
    return total_under / len(y_hat)

def munderest_pct(y_actual, y_hat):
    """
    Percent underestimation metric (like MAPE but only for underestimates)
    """
    under_pct = [
        (yi - yh) / yi * 100
        for yi, yh in zip(y_actual, y_hat)
        if yh < yi and yi != 0
    ]
    return sum(under_pct) / len(under_pct) if under_pct else 0
  
    
future_periods = 24  # 24-hour horizon
all_metrics = None
for col in target_cols:
    metrics = metrics_dict(df, future_periods, col)
    if(all_metrics is None):
        all_metrics = metrics
    else:
        all_metrics = all_metrics | metrics
# Generate timestamp for filename (no colons to avoid issues on some OS)
timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

# File name with timestamp
filename = f"metrics_{timestamp}.json"

all_metrics = {'ts':timestamp} | all_metrics

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        import numpy as np
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

with open(filename, "w") as f:
    json.dump(all_metrics, f, cls=NpEncoder, indent=2)
