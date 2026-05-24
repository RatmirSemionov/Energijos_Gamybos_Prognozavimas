# Naudojamos bibliotekos
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import xgboost as xgb

################################################################
# Duomenu apjungimas
################################################################

siaures = pq.read_table("df_top20_siaures.parquet").to_pandas()
pietus = pq.read_table("df_top20_pietu.parquet").to_pandas()
centrine = pq.read_table("df_top20_centrine.parquet").to_pandas()

siaures["plant_id"] = "siaures"
pietus["plant_id"] = "pietu"
centrine["plant_id"] = "centrine"

df = pd.concat([siaures, pietus, centrine], ignore_index=True)

df = df.sort_values(["ts_local", "plant_id"]).reset_index(drop=True)

df = pd.get_dummies(df, columns=["plant_id"])
df = df.iloc[:-3] # Pašalinama nepilna diena

# Apjungto duomenų rinkinio išsaugojimas, kad galima būtų panaudoti su TFT modeliu kitame kode
#df["plant_id"] = df["plant_id"].astype(str)
#df.to_parquet("apjungtos_duomenys.parquet", index=False)
#df.to_csv("apjungtos_duomenys.csv", index=False)

################################################################
# Požymiai
################################################################

target_col = "Energy"
feature_cols = [c for c in df.columns if c not in ["ts_local", target_col]]

################################################################
# Mokymo / Testavimo aibes
################################################################

split_ratio = 0.8

n_exo = len(df)
split_index_exo = int(n_exo * split_ratio)
train = df.iloc[:split_index_exo, :].copy()
test  = df.iloc[split_index_exo:, :].copy()

print("Exogenous split:", train.shape[0], "train |", test.shape[0], "test")
print(test.head(5))

################################################################
# XGBoost modelis
################################################################

xgb_params = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "eta": 0.05,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "seed": 42
}

model = xgb.train(
    params=xgb_params,
    dtrain=xgb.DMatrix(train[feature_cols], label=train[target_col]),
    num_boost_round=150
)

pred = []

for start_idx in range(0, len(test), 24):
    end_idx = start_idx + 24

    X_day = test[feature_cols].iloc[start_idx:end_idx]
    day_matrix = xgb.DMatrix(X_day)

    pred_day = model.predict(day_matrix)
    pred.extend(pred_day)

pred = np.array(pred)

################################################################
# Metrikos
################################################################

def wmape(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    return (
        np.nansum(np.abs(y_pred - y_true)) /
        np.nansum(np.abs(y_true))
    ) * 100


def evaluate_model(y_true, y_pred, ts_col=None, df=None):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    rmse = np.sqrt(np.nanmean((y_pred - y_true) ** 2))
    mae = np.nanmean(np.abs(y_pred - y_true))

    hourly_wmape_val = wmape(y_true, y_pred)

    daily_wmape_val = np.nan

    if ts_col is not None and df is not None:
        df_temp = df.copy()
        df_temp["pred"] = y_pred
        df_temp["y_true"] = y_true
        df_temp["date"] = pd.to_datetime(df_temp[ts_col]).dt.date

        df_daily = df_temp.groupby("date").agg(
            actual=("y_true", "sum"),
            pred=("pred", "sum")
        ).reset_index()

        daily_wmape_val = wmape(df_daily["actual"], df_daily["pred"])

    return {
        "RMSE": rmse,
        "MAE": mae,
        "Hourly_WMAPE": f"{hourly_wmape_val:.2f}%",
        "Daily_WMAPE": f"{daily_wmape_val:.2f}%" if not np.isnan(daily_wmape_val) else None
    }

################################################################
# Globalios metrikos
################################################################

print("\n================ Globalios metrikos ================\n")

global_metrics = evaluate_model(
    test[target_col].values,
    pred,
    ts_col="ts_local",
    df=test
)

print(global_metrics)

################################################################
# Metrikos pagal elektrine
################################################################

def evaluate_per_plant(df_test, y_pred):
    df_eval = df_test.copy()
    df_eval["pred"] = y_pred

    plants = {
        "plant_id_siaures": "siaures",
        "plant_id_pietu": "pietu",
        "plant_id_centrine": "centrine"
    }

    results = {}

    for col, name in plants.items():
        if col not in df_eval.columns:
            continue

        df_p = df_eval[df_eval[col] == 1]

        if len(df_p) == 0:
            continue

        results[name] = evaluate_model(
            df_p[target_col].values,
            df_p["pred"].values,
            ts_col="ts_local",
            df=df_p
        )

    return results

print("\n================ Metrikos pagal elektrine ================\n")

plant_results = evaluate_per_plant(test, pred)

for plant, metrics in plant_results.items():
    print(f"\n{plant}")
    print(metrics)