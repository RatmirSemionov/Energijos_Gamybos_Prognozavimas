################################################################
# Bibliotekos
################################################################
import pyarrow.parquet as pq
import pandas as pd
import numpy as np
import itertools
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import random
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Grafikams
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

#########################################################################
# Duomenų paruošimas (čia pasirenkama, kuri duomenų rinkinį naudoti kode)
#########################################################################
# Duomenų rinkiniai su visais požymiais
#df_exogenous = pq.read_table("df_exogenous_siaures_GTI.parquet").to_pandas()
#df_exogenous = pq.read_table("df_exogenous_pietu_GTI.parquet").to_pandas()
#df_exogenous = pq.read_table("df_exogenous_centrine_GTI.parquet").to_pandas()

# Duomenų rinkiniai su atrinktais požymiais
df_exogenous = pq.read_table("df_top20_siaures.parquet").to_pandas()
#df_exogenous = pq.read_table("df_top20_pietu.parquet").to_pandas()
#df_exogenous = pq.read_table("df_top20_centrine.parquet").to_pandas()


# Šita užkomentuota vieta skirta tam, kad galima būtų optimizuoti hiperparametrus apjungtam duomenų
#rinkiniui, kai to reikia

# siaures = pq.read_table("df_top20_siaures.parquet").to_pandas()
# pietine = pq.read_table("df_top20_pietu.parquet").to_pandas()
# centrine = pq.read_table("df_top20_centrine.parquet").to_pandas()
# siaures["plant_id"] = "siaures"
# pietine["plant_id"] = "pietine"
# centrine["plant_id"] = "centrine"
#
# common_cols = siaures.columns \
#     .intersection(pietine.columns) \
#     .intersection(centrine.columns)
#
# siaures = siaures[common_cols]
# pietine = pietine[common_cols]
# centrine = centrine[common_cols]
#
# df = pd.concat([siaures, pietine, centrine], ignore_index=True)
# df = df.sort_values(["ts_local", "plant_id"]).reset_index(drop=True)
# df_exogenous = pd.get_dummies(df, columns=["plant_id"])
print(df_exogenous.head())

print("Energijos reiksmes:")
print(df_exogenous["Energy"].min())
print(df_exogenous["Energy"].max())

# Pašalinu paskutinę dieną, kuri turi tik 3 valandas
df_exogenous = df_exogenous.iloc[:-3]

# Buvo žiurima ar geriau prognozuoja be nakties valandų, rezultatai beveik nesiskiria ir
#kartais valandinis nuokrypis net didesnis nei su nakties valandomis
# df_exogenous = df_exogenous[df_exogenous["ts_local"].dt.hour >= 7]

################################################################
# Sezoniškumo analizė
################################################################
import matplotlib.dates as mdates

files = {
    "Šiaurės": "df_top20_siaures.parquet",
    "Pietinės": "df_top20_pietu.parquet",
    "Centrinės": "df_top20_centrine.parquet",
}

weekly_all = {}
monthly_all = {}

# Žodynas, skirtas mėnesių priskyrimui konkretiems sezonams
season_map = {
    12: "Žiema", 1: "Žiema", 2: "Žiema",
    3: "Pavasaris", 4: "Pavasaris", 5: "Pavasaris",
    6: "Vasara", 7: "Vasara", 8: "Vasara",
    9: "Ruduo", 10: "Ruduo", 11: "Ruduo"
}

for name, file in files.items():
    df = pq.read_table(file).to_pandas()
    df["ts_local"] = pd.to_datetime(df["ts_local"])
    df = df.iloc[:-3]
    df["date"] = df["ts_local"].dt.date

    df_stats = df.copy()
    df_stats["month_num"] = df_stats["ts_local"].dt.month
    df_stats["sezonas"] = df_stats["month_num"].map(season_map)

    seasonal_stats = df_stats.groupby("sezonas")["Energy"].agg(
        Vidurkis="mean",
        Imties_Standartinis_Nuokrypis="std"
    ).reindex(["Žiema", "Pavasaris", "Vasara", "Ruduo"])

    print(f"\n==================================================")
    print(f"--- {name} Elektrinės sezoninė statistika (kWh) ---")
    print(f"==================================================")
    print(seasonal_stats.round(3).to_string())
    print("==================================================\n")
    # =========================================================================

    # -------------------------
    # Savaitės vidurkis
    # -------------------------
    weekly = df.resample("W", on="ts_local")["Energy"].mean()
    weekly = weekly.dropna()

    # -------------------------
    # Mėnesio vidurkis
    # -------------------------
    monthly = df.resample("M", on="ts_local").agg(
        Energy_mean=("Energy", "mean"), unique_days=("date", "nunique")
    )
    monthly = monthly.dropna(subset=["Energy_mean"])

    monthly_series = monthly["Energy_mean"]

    weekly_all[name] = weekly
    monthly_all[name] = monthly_series

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # --- Savaitės grafikas ---
    axes[0].plot(weekly.index, weekly.values, marker="o", markersize=3)
    axes[0].set_title(
        f"{name} elektrinės savaitinė vidutinė energijos gamyba"
    )
    axes[0].set_xlabel("Laikas")
    axes[0].set_ylabel("Energijos vidurkis (kWh)")
    axes[0].set_ylim(bottom=0)
    tick_months_weekly = pd.to_datetime(
        df["ts_local"].dt.strftime("%Y-%m-01").unique()
    ).sort_values()

    axes[0].set_xlim(tick_months_weekly.min(), weekly.index.max())
    axes[0].set_xticks(tick_months_weekly)

    axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    axes[0].tick_params(axis="x", rotation=45)

    # --- Mėnesio grafikas ---
    axes[1].plot(monthly_series.index, monthly_series.values, marker="o")
    axes[1].set_title(
        f"{name} elektrinės mėnesinė vidutinė energijos gamyba"
    )
    axes[1].set_xlabel("Laikas")
    axes[1].set_ylabel("Energijos vidurkis (kWh)")
    axes[1].set_ylim(bottom=0)

    if not monthly_series.empty:
        axes[1].set_xlim(
            monthly_series.index.min(), monthly_series.index.max()
        )
        axes[1].set_xticks(monthly_series.index)

    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    axes[1].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.show()

################################################################
# Elektrinių palyginimas
################################################################

fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# Savaitinis palyginimas
for name, series in weekly_all.items():

    norm_series = (series - series.min()) / (series.max() - series.min())

    axes[0].plot(
        norm_series.index,
        norm_series.values,
        marker="o",
        markersize=3,
        label=name
    )

axes[0].set_title("Savaitinė normalizuota energijos gamyba")
axes[0].set_xlabel("Laikas")
axes[0].set_ylabel("Normalizuota reikšmė (0–1)")
axes[0].set_ylim(0, 1)

axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
axes[0].tick_params(axis="x", rotation=45)
axes[0].legend()

all_weekly_months = pd.to_datetime(sorted({
    ts.to_period("M").to_timestamp()
    for series in weekly_all.values()
    for ts in series.index
}))

axes[0].set_xticks(all_weekly_months)
axes[0].set_xlim(all_weekly_months.min(), all_weekly_months.max())

# Mėnesinis palyginimas

for name, series in monthly_all.items():

    norm_series = (series - series.min()) / (series.max() - series.min())

    axes[1].plot(
        norm_series.index,
        norm_series.values,
        marker="o",
        label=name
    )

axes[1].set_title("Mėnesinė normalizuota energijos gamyba")
axes[1].set_xlabel("Laikas")
axes[1].set_ylabel("Normalizuota reikšmė (0–1)")
axes[1].set_ylim(0, 1)

axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
axes[1].tick_params(axis="x", rotation=45)
axes[1].legend()

all_months = sorted({
    ts for series in monthly_all.values()
    for ts in series.index
})

axes[1].set_xticks(all_months)
axes[1].set_xlim(min(all_months), max(all_months))

plt.tight_layout()
plt.show()

################################################################
# Sezonai
################################################################

def get_season(df):
    m = df["ts_local"].dt.month
    return np.where(
        m.isin([12, 1, 2]), "winter",
        np.where(
            m.isin([3, 4, 5]), "spring",
            np.where(
                m.isin([6, 7, 8]), "summer",
                "autumn"
            )
        )
    )

df_exogenous["season"] = get_season(df_exogenous)

################################################################
# Sezoninė statistika
################################################################

season_stats = df_exogenous.groupby("season")["Energy"].agg(
    ["mean", "median", "std", "min", "max"]
)

print("\nSeasonal Energy statistics:")
print(season_stats)


df_exogenous_full = df_exogenous.copy()

df_exogenous_winter = df_exogenous[df_exogenous["season"] == "winter"].copy()
df_exogenous_spring = df_exogenous[df_exogenous["season"] == "spring"].copy()
df_exogenous_summer = df_exogenous[df_exogenous["season"] == "summer"].copy()
df_exogenous_autumn = df_exogenous[df_exogenous["season"] == "autumn"].copy()

# Čia nustatoma, ar duomenų rinkinį naudoti pilnai, ar tik kažkokį sezoną
df_exogenous = df_exogenous_full
df_exogenous = df_exogenous.drop(columns=["season"])
print("Energijos reiksmes sezonui:")
print(df_exogenous["Energy"].min())
print(df_exogenous["Energy"].max())
################################################################
# Mokymo ir testavimo aibes
################################################################
split_ratio = 0.8

n_exo = len(df_exogenous)
split_index_exo = int(n_exo * split_ratio)
train_exogenous = df_exogenous.iloc[:split_index_exo, :].copy()
test_exogenous  = df_exogenous.iloc[split_index_exo:, :].copy()

print("Exogenous split:", train_exogenous.shape[0], "train |", test_exogenous.shape[0], "test")
print(test_exogenous.head(5))

# Duomenų rinkinio stulpeliai
target_col = "Energy"

feature_cols_exo = [c for c in train_exogenous.columns if c not in ["ts_local", target_col]]

################################################################
# Požymių atranka naudojant Borda rangų agregavimo metodą
################################################################

model_configs = {
    "Siaures": {
        "rf": {"n_estimators": 200, "max_features": 20, "max_depth": 10},
        "xgb": {"n_estimators": 150, "learning_rate": 0.05, "max_depth": 6, "subsample": 0.8, "colsample_bytree": 0.8}
    },
    "Pietu": {
        "rf": {"n_estimators": 800, "max_features": 20, "max_depth": 10},
        "xgb": {"n_estimators": 100, "learning_rate": 0.05, "max_depth": 6, "subsample": 0.9, "colsample_bytree": 0.7}
    },
    "Centrine": {
        "rf": {"n_estimators": 800, "max_features": 20, "max_depth": None},
        "xgb": {"n_estimators": 200, "learning_rate": 0.1, "max_depth": 6, "subsample": 0.8, "colsample_bytree": 0.8}
    }
}

data_files = [
    "df_exogenous_siaures_GTI.parquet",
    "df_exogenous_pietu_GTI.parquet",
    "df_exogenous_centrine_GTI.parquet"
]

borda_scores = defaultdict(int)
individual_lists = {}

for file in data_files:
    try:
        # Ištraukiamas regiono pavadinimas gražesniam spausdinimui (pvz., iš "df_exogenous_siaures.parquet" gausime "Siaures")
        region_name = file.split('_')[2].replace('.parquet', '').capitalize()

        # Paimame nustatymus pagal regioną
        config = model_configs.get(region_name, {})
        rf_params = config.get("rf", {})
        xgb_params = config.get("xgb", {})

        df_temp = pq.read_table(file).to_pandas()
        df_temp = df_temp.iloc[:-3]

        split_idx = int(len(df_temp) * 0.8)
        train_fs = df_temp.iloc[:split_idx].copy()

        y_fs = train_fs["Energy"]
        X_fs = train_fs.drop(columns=["ts_local", "Energy"], errors='ignore')
        feat_names = X_fs.columns.tolist()
        N = len(feat_names)

        rf_fs = RandomForestRegressor(
            **rf_params,
            random_state=SEED,
            n_jobs=-1
        )
        rf_fs.fit(X_fs, y_fs)

        rf_ranks = pd.Series(rf_fs.feature_importances_, index=feat_names).sort_values(ascending=False)
        individual_lists[f"{region_name} - Random Forest"] = rf_ranks.head(20)

        for rank, (feat, val) in enumerate(rf_ranks.items()):
            borda_scores[feat] += (N - rank)

        xgb_fs = xgb.XGBRegressor(
            **xgb_params,
            random_state=SEED
        )
        xgb_fs.fit(X_fs, y_fs)

        xgb_ranks = pd.Series(xgb_fs.feature_importances_, index=feat_names).sort_values(ascending=False)
        individual_lists[f"{region_name} - XGBoost"] = xgb_ranks.head(20)

        for rank, (feat, val) in enumerate(xgb_ranks.items()):
            borda_scores[feat] += (N - rank)

    except Exception as e:
        print(f"Klaida apdorojant {file}: {e}")

# Sąrašai
print("\n" + "#"*60)
print("Požymių sąrašai prieš agregavimą (TOP 20)")
print("#"*60)
for list_name, series in individual_lists.items():
    print(f"\n[{list_name}]")
    print(f"{'Eil.':<4} | {'Požymis':<30} | {'Svarba':<10}")
    print("-" * 50)
    for i, (feat, imp) in enumerate(series.items()):
        print(f"{i+1:<4} | {feat:<30} | {imp:.4f}")

# Galutinis agreguotas sąrašas
sorted_borda = sorted(borda_scores.items(), key=lambda x: x[1], reverse=True)
top_20_data = sorted_borda[:20]
top_20_features = [item[0] for item in top_20_data]
top_20_scores = [item[1] for item in top_20_data]

print("\n" + "=" * 60)
print(f"{'Eil.':<5} | {'Bendra suma (Požymis)':<35} | {'Borda balas':<10}")
print("-" * 60)
for i, (feat, score) in enumerate(top_20_data):
    print(f"{i + 1:<5} | {feat:<35} | {score:<10}")
print("=" * 60)

# Grafikas
plt.figure(figsize=(12, 8))
sns.set_style("white")

plot = sns.barplot(
    x=top_20_scores,
    y=top_20_features,
    hue=top_20_features,
    palette="viridis",
    legend=False
)

plot.xaxis.grid(False)
plot.yaxis.grid(False)
sns.despine(left=True, bottom=True)

plt.title('20 svarbiausių požymių pagal Borda balus', fontsize=14)
plt.xlabel('Borda balas', fontsize=12)
plt.ylabel('Požymio pavadinimas', fontsize=12)

for i, score in enumerate(top_20_scores):
    plot.text(score + 5, i, str(score), va='center', fontsize=10)

plt.tight_layout()
plt.show()

# ################################################################
# # Duomenų rinkinių išsaugojimas po požymių atrankos
# ################################################################
# required_cols = ["ts_local", "Energy"]
#
# for file in data_files:
#     try:
#         region_name = file.split('_')[2].replace('.parquet', '').lower()
#         df_full = pq.read_table(file).to_pandas()
#         cols_to_keep = [c for c in df_full.columns if c in top_20_features or c in required_cols]
#         df_filtered = df_full[cols_to_keep].copy()
#
#         csv_name = f"df_top20_{region_name}.csv"
#         df_filtered.to_csv(csv_name, index=False)
#
#         parquet_name = f"df_top20_{region_name}.parquet"
#         df_filtered.to_parquet(parquet_name, index=False)
#
#         print(f"Išsaugota: {csv_name} ir {parquet_name} (Stulpelių skaičius: {len(df_filtered.columns)})")
#
#     except Exception as e:
#         print(f"Klaida saugant {file}: {e}")

################################################################
# Tikslumo metrikų funkcija
################################################################
def evaluate_model(y_true, y_pred, ts_col=None, df=None):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    rmse = np.sqrt(np.nanmean((y_pred - y_true) ** 2))
    mae = np.nanmean(np.abs(y_pred - y_true))

    hourly_wmape = (np.nansum(np.abs(y_pred - y_true)) / np.nansum(np.abs(y_true))) * 100

    daily_wmape_val = np.nan

    if ts_col is not None and df is not None:
        df_temp = df.copy()
        df_temp["pred"] = y_pred
        df_temp["y_true"] = y_true
        df_temp["date"] = pd.to_datetime(df_temp[ts_col]).dt.date

        df_daily = df_temp.groupby("date").agg(
            actual_day=("y_true", lambda x: np.nansum(x)),
            pred_day=("pred", lambda x: np.nansum(x))
        ).reset_index()

        daily_wmape_val = (np.nansum(np.abs(df_daily["pred_day"] - df_daily["actual_day"])) /
                           np.nansum(np.abs(df_daily["actual_day"]))) * 100

    return {
        "RMSE": rmse,
        "MAE": mae,
        "Hourly_WMAPE": f"{round(hourly_wmape, 2)}%",
        "Daily_WMAPE": f"{round(daily_wmape_val, 2)}%" if not np.isnan(daily_wmape_val) else None
    }


################################################################
# Hiperparametrų optimizavimo funkcija
################################################################
def time_series_cv_mape(train_data, feature_cols, target_col,
                         model_func, param_grid,
                         ts_col="ts_local", folds=5):
    n = train_data.shape[0]
    fold_size = n // (folds + 1)
    results = []

    for idx, params in param_grid.iterrows():
        fold_metrics = []
        for k in range(1, folds + 1):
            train_end = fold_size * k
            val_start = train_end
            val_end = min(train_end + fold_size, n)

            train_fold = train_data.iloc[:train_end, :].copy()
            val_fold = train_data.iloc[val_start:val_end, :].copy()

            preds = model_func(train_fold, val_fold, feature_cols, params)

            metrics = evaluate_model(val_fold[target_col], preds, ts_col=ts_col, df=val_fold)
            hourly_wmape_num = float(str(metrics["Hourly_WMAPE"]).replace("%", ""))
            fold_metrics.append(hourly_wmape_num)

        results.append({"params": params, "Hourly_WMAPE": np.nanmean(fold_metrics)})

    best_idx = np.argmin([r["Hourly_WMAPE"] for r in results])
    return results[best_idx]["params"]

################################################################
# SVM modelio optimizavimas
################################################################
def svm_model_func(train_fold, val_fold, feature_cols, params):
    scaler_X = StandardScaler()

    X_train_scaled = scaler_X.fit_transform(train_fold[feature_cols])
    X_val_scaled   = scaler_X.transform(val_fold[feature_cols])

    model = SVR(
        kernel="rbf",
        C=params["cost"],
        gamma=params["gamma"],
        epsilon=params["epsilon"]
    )

    model.fit(X_train_scaled, train_fold["Energy"])
    y_val_pred = model.predict(X_val_scaled)

    return y_val_pred


# ----------------------
# SVM parametrų gardelė
# ----------------------
svm_costs = [1, 10, 100, 1000]
svm_gammas = ["scale", 0.01, 0.05, 0.1, 1]
svm_epsilons = [0.1, 1, 2, 5, 10]

svm_grid = pd.DataFrame(
    list(itertools.product(svm_costs, svm_gammas, svm_epsilons)),
    columns=["cost", "gamma", "epsilon"]
)

gamma_default = 1 / len(feature_cols_exo)
svm_default = pd.DataFrame({
    "cost": [1],
    "gamma": [gamma_default],
    "epsilon": [0.01]
})

svm_grid = pd.concat([svm_grid, svm_default]).drop_duplicates().reset_index(drop=True)

# Šita vieta atkomentavus optimizuoja hiperparametrus SVM modeliui

# best_svm_params = time_series_cv_mape(
#     train_exogenous,
#     feature_cols_exo,
#     "Energy",
#     svm_model_func,
#     svm_grid
# )
# print("\nGeriausi SVM parametrai:\n", best_svm_params)
################################################################
# RF modelio optimizavimas
################################################################
def rf_model_func(train_fold, val_fold, feature_cols, params):
    max_depth = params["max_depth"]
    if pd.isna(max_depth):
        max_depth = None
    else:
        max_depth = int(max_depth)

    mtry_val = params["mtry"]

    if not isinstance(mtry_val, str):
        mtry_val = int(mtry_val)

    model = RandomForestRegressor(
        n_estimators=int(params["ntree"]),
        max_features=mtry_val,
        max_depth=max_depth,
        random_state=42,
        n_jobs=-1
    )

    model.fit(train_fold[feature_cols], train_fold["Energy"])
    return model.predict(val_fold[feature_cols])


# -------------------------------
# RF parametrų gardelė
# -------------------------------
rf_ntrees = [200, 500, 800]
rf_mtrys = ["sqrt", 2, 3, 4, 5, 10, 20]
rf_depths = [None, 10, 20, 30, 50]

rf_grid = pd.DataFrame(
    list(itertools.product(rf_ntrees, rf_mtrys, rf_depths)),
    columns=["ntree", "mtry", "max_depth"])
rf_default = pd.DataFrame({
    "ntree": [500],
    "mtry": [3],
    "max_depth": [None]
})
rf_grid = pd.concat([rf_grid, rf_default]).drop_duplicates().reset_index(drop=True)

# Šita vieta atkomentavus optimizuoja hiperparametrus RF modeliui

# best_rf_params = time_series_cv_mape(train_exogenous, feature_cols_exo, "Energy",
#                                      rf_model_func, rf_grid)
# print("\nGeriausi RF parametrai::\n", best_rf_params)

################################################################
# XGBoost modelio optimizavimas
################################################################
def xgb_model_func(train_fold, val_fold, feature_cols, params):
    dtrain = xgb.DMatrix(train_fold[feature_cols], label=train_fold["Energy"])
    dval = xgb.DMatrix(val_fold[feature_cols], label=val_fold["Energy"])

    model = xgb.train(
        params={
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "eta": params["eta"],
            "max_depth": int(params["max_depth"]),
            "subsample": params["subsample"],
            "colsample_bytree": params["colsample"]
        },
        dtrain=dtrain,
        num_boost_round=int(params["nrounds"]),
        verbose_eval=False
    )
    return model.predict(dval)


# --------------------------
# XGBoost parametrų gardelė
# --------------------------
xgb_etas = [0.05, 0.1, 0.2]
xgb_depths = [4, 6, 8]
xgb_subsample = [0.5, 0.6, 0.7, 0.8, 0.9]
xgb_colsample = [0.5, 0.6, 0.7, 0.8, 0.9]
xgb_nrounds = [100, 150, 200]

xgb_grid = pd.DataFrame(list(itertools.product(xgb_etas, xgb_depths, xgb_subsample, xgb_colsample, xgb_nrounds)),
                        columns=["eta", "max_depth", "subsample", "colsample", "nrounds"])
xgb_default = pd.DataFrame({"eta": [0.1], "max_depth": [6], "subsample": [0.8], "colsample": [0.8], "nrounds": [200]})
xgb_grid = pd.concat([xgb_grid, xgb_default]).drop_duplicates().reset_index(drop=True)

# Šita vieta atkomentavus optimizuoja hiperparametrus XGBoost modeliui

# best_xgb_params = time_series_cv_mape(train_exogenous, feature_cols_exo, "Energy",
#                                       xgb_model_func, xgb_grid)
#
# print("\nGeriausi XGBoost parametrai:\n", best_xgb_params)
################################################################
# LSTM modelio optimizavimas
################################################################
# --------------------------
# LSTM parametrų gardelė
# --------------------------
lstm_hidden_sizes = [16, 32, 48]
lstm_num_layers = [1, 2, 3]
lstm_dropouts = [0.0, 0.2]
lstm_lrs = [0.001, 0.003]
lstm_epochs = [20, 30]

lstm_grid = pd.DataFrame(
    list(itertools.product(lstm_hidden_sizes, lstm_num_layers, lstm_dropouts, lstm_lrs, lstm_epochs)),
    columns=["hidden_size", "num_layers", "dropout", "lr", "epochs"]
)

def create_lstm_sequences(df, feature_cols, target_col, seq_len=24):
    X, y = [], []
    features = df[feature_cols].values
    target = df[target_col].values

    for i in range(seq_len, len(df)):
        X.append(features[i - seq_len:i + 1, :])
        y.append(target[i])

    return np.array(X), np.array(y)

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.0):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

def train_lstm(model, X_train, y_train, epochs=50, batch_size=32, lr=0.001):
    dataset = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float())
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for xb, yb in loader:
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * xb.size(0)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(dataset):.4f}")

def lstm_model_func(train_fold, val_fold, feature_cols, params, seq_len=24):
    X_train_seq, y_train_seq = create_lstm_sequences(train_fold, feature_cols, "Energy", seq_len)
    X_val_seq, y_val_seq = create_lstm_sequences(val_fold, feature_cols, "Energy", seq_len)

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_train_scaled = scaler_X.fit_transform(X_train_seq.reshape(-1, X_train_seq.shape[2])).reshape(X_train_seq.shape)
    y_train_scaled = scaler_y.fit_transform(y_train_seq.reshape(-1, 1))

    X_val_scaled = scaler_X.transform(X_val_seq.reshape(-1, X_val_seq.shape[2])).reshape(X_val_seq.shape)
    y_val_scaled = scaler_y.transform(y_val_seq.reshape(-1, 1))

    input_size = X_train_scaled.shape[2]
    model = LSTMModel(
        input_size=input_size,
        hidden_size=int(params["hidden_size"]),
        num_layers=int(params["num_layers"]),
        dropout=float(params["dropout"])
    )

    train_lstm(
        model,
        X_train_scaled,
        y_train_scaled,
        epochs=int(params["epochs"]),
        batch_size=32,
        lr=float(params["lr"])
    )

    model.eval()
    with torch.no_grad():
        y_pred_scaled = model(torch.from_numpy(X_val_scaled).float()).numpy()
    y_pred = scaler_y.inverse_transform(y_pred_scaled)

    val_fold_eval = val_fold.iloc[seq_len:seq_len + len(y_val_seq)]

    return y_pred.flatten(), y_val_seq, val_fold_eval

# Atskira funkcija LSTM modelio hiperparametrų optimizavimui, kadangi skiriasi architektūra ir
# modelis turi atmintį

def lstm_cv_wrapper(train_data, feature_cols, target_col, param_grid, seq_len=24, folds=5):
    results = []
    n = train_data.shape[0]
    fold_size = n // (folds + 1)

    for idx, params in param_grid.iterrows():
        fold_metrics = []
        for k in range(1, folds + 1):
            train_end = fold_size * k
            val_start = train_end
            val_end = min(train_end + fold_size, n)

            train_fold = train_data.iloc[:train_end, :].copy()
            val_fold = train_data.iloc[val_start:val_end, :].copy()

            y_pred, y_true, val_eval_df = lstm_model_func(train_fold, val_fold, feature_cols, params, seq_len=seq_len)

            metrics = evaluate_model(y_true, y_pred, ts_col="ts_local", df=val_eval_df)

            hourly_wmape_num = float(str(metrics["Hourly_WMAPE"]).replace("%", ""))
            fold_metrics.append(hourly_wmape_num)

        results.append({"params": params, "Hourly_WMAPE": np.nanmean(fold_metrics)})

    best_idx = np.argmin([r["Hourly_WMAPE"] for r in results])
    return results[best_idx]["params"]


# Šita vieta atkomentavus optimizuoja hiperparametrus LSTM modeliui
# best_lstm_params = lstm_cv_wrapper(train_exogenous, feature_cols_exo, "Energy", lstm_grid)

# Šitoje vietioje pateikiami visiems modeliams optimizuoti hiperparametrai (jei visi yra įjungti kode)
# Geriausi parametrai
# print("\nGeriausi SVM parametrai:\n", best_svm_params)
# print("\nGeriausi RF parametrai::\n", best_rf_params)
# print("\nGeriausi XGBoost parametrai:\n", best_xgb_params)
# print("Geriausi LSTM parametrai:\n", best_lstm_params)

################################################################
# Tiesioginio prognozavimo strategija
################################################################
# Gauti geriausi hiperparametrai kiekvienam naudojamam duomenų rinkiniui
CONFIG = {
    "Siaures_all": {
        "svm": {"C": 10, "gamma": "scale", "epsilon": 0.1},
        "rf": {"n_estimators": 200, "max_features": 20, "max_depth": 10},
        "xgb": {"eta": 0.05, "max_depth": 6, "subsample": 0.8, "colsample_bytree": 0.8, "num_boost_round": 150},
        "lstm": {"hidden_size": 48, "num_layers": 3, "dropout": 0.2, "lr": 0.001, "epochs": 20}
    },

    "Siaures_selected": {
        "svm": {"C": 100, "gamma": 0.01, "epsilon": 0.1},
        "rf": {"n_estimators": 200, "max_features": 10, "max_depth": 10},
        "xgb": {"eta": 0.05, "max_depth": 6, "subsample": 0.9, "colsample_bytree": 0.9, "num_boost_round": 200},
        "lstm": {"hidden_size": 32, "num_layers": 1, "dropout": 0.2, "lr": 0.001, "epochs": 30}
    },

    "Centrine_all": {
        "svm": {"C": 1000, "gamma": "scale", "epsilon": 0.1},
        "rf": {"n_estimators": 800, "max_features": 20, "max_depth": None},
        "xgb": {"eta": 0.1, "max_depth": 6, "subsample": 0.8, "colsample_bytree": 0.8, "num_boost_round": 200},
        "lstm": {"hidden_size": 48, "num_layers": 2, "dropout": 0.2, "lr": 0.003, "epochs": 20}
    },

    "Centrine_selected": {
        "svm": {"C": 100, "gamma": 0.01, "epsilon": 2.0},
        "rf": {"n_estimators": 200, "max_features": 10, "max_depth": 30},
        "xgb": {"eta": 0.05, "max_depth": 8, "subsample": 0.8, "colsample_bytree": 0.9, "num_boost_round": 200},
        "lstm": {"hidden_size": 48, "num_layers": 1, "dropout": 0.2, "lr": 0.001, "epochs": 20}
    },

    "Pietine_all": {
        "svm": {"C": 10, "gamma": "scale", "epsilon": 0.1},
        "rf": {"n_estimators": 800, "max_features": 20, "max_depth": 10},
        "xgb": {"eta": 0.05, "max_depth": 6, "subsample": 0.9, "colsample_bytree": 0.7, "num_boost_round": 100},
        "lstm": {"hidden_size": 48, "num_layers": 2, "dropout": 0.2, "lr": 0.003, "epochs": 20}
    },

    "Pietine_selected": {
        "svm": {"C": 100, "gamma": 0.01, "epsilon": 0.1},
        "rf": {"n_estimators": 800, "max_features": 10, "max_depth": 10},
        "xgb": {"eta": 0.1, "max_depth": 6, "subsample": 0.8, "colsample_bytree": 0.5, "num_boost_round": 100},
        "lstm": {"hidden_size": 16, "num_layers": 1, "dropout": 0.2, "lr": 0.001, "epochs": 30}
    }
}

# Čia pasirenkama, kuriuos hiperparametrus naudoti
ACTIVE_CASE = "Siaures_selected"
params = CONFIG[ACTIVE_CASE]


# SVM modelis
scaler_X = StandardScaler()

X_train_scaled = scaler_X.fit_transform(train_exogenous[feature_cols_exo])
X_test_scaled  = scaler_X.transform(test_exogenous[feature_cols_exo])

svm_params = params["svm"]

svm_exo = SVR(
    kernel="rbf",
    C=svm_params["C"],
    gamma=svm_params["gamma"],
    epsilon=svm_params["epsilon"]
)

start = time.time()
svm_exo.fit(X_train_scaled, train_exogenous["Energy"])
print("SVM training time:", time.time() - start)

pred_svm = []

for start_idx in range(0, len(test_exogenous), 24):
    end_idx = start_idx + 24

    X_day = X_test_scaled[start_idx:end_idx]
    pred_day = svm_exo.predict(X_day)

    pred_svm.extend(pred_day)

pred_svm = np.array(pred_svm)

metrics_svm = evaluate_model(
    test_exogenous["Energy"],
    pred_svm,
    ts_col="ts_local",
    df=test_exogenous
)

print("\n--- SVM ---")
print(metrics_svm)


# RF modelis
rf_params = params["rf"]

rf_exo = RandomForestRegressor(
    n_estimators=rf_params["n_estimators"],
    max_features=rf_params["max_features"],
    max_depth=rf_params["max_depth"],
    random_state=42,
    n_jobs=-1
)

start = time.time()
rf_exo.fit(train_exogenous[feature_cols_exo], train_exogenous["Energy"])
print("RF training time:", time.time() - start)

pred_rf = []

for start_idx in range(0, len(test_exogenous), 24):
    end_idx = start_idx + 24

    X_day = test_exogenous[feature_cols_exo].iloc[start_idx:end_idx]
    pred_day = rf_exo.predict(X_day)

    pred_rf.extend(pred_day)

pred_rf = np.array(pred_rf)

metrics_rf = evaluate_model(
    test_exogenous["Energy"],
    pred_rf,
    ts_col="ts_local",
    df=test_exogenous
)

print("\n--- Random Forest ---")
print(metrics_rf)


# XGBoost modelis
xgb_cfg = params["xgb"]

train_matrix = xgb.DMatrix(train_exogenous[feature_cols_exo], label=train_exogenous["Energy"])
test_matrix  = xgb.DMatrix(test_exogenous[feature_cols_exo], label=test_exogenous["Energy"])

xgb_params = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "eta": xgb_cfg["eta"],
    "max_depth": xgb_cfg["max_depth"],
    "subsample": xgb_cfg["subsample"],
    "colsample_bytree": xgb_cfg["colsample_bytree"],
    "seed": 42
}

start = time.time()

xgb_model = xgb.train(
    params=xgb_params,
    dtrain=train_matrix,
    num_boost_round=xgb_cfg["num_boost_round"],
    verbose_eval=0
)

print("XGBoost training time:", time.time() - start)

pred_xgb = []

for start_idx in range(0, len(test_exogenous), 24):
    end_idx = start_idx + 24

    X_day = test_exogenous[feature_cols_exo].iloc[start_idx:end_idx]
    day_matrix = xgb.DMatrix(X_day)

    pred_day = xgb_model.predict(day_matrix)
    pred_xgb.extend(pred_day)

pred_xgb = np.array(pred_xgb)

metrics_xgb = evaluate_model(
    test_exogenous["Energy"],
    pred_xgb,
    ts_col="ts_local",
    df=test_exogenous
)

print("\n--- XGBoost ---")
print(metrics_xgb)


# LSTM modelis
lstm_cfg = params["lstm"]

seq_len = 24

X_train_seq, y_train_seq = create_lstm_sequences(train_exogenous, feature_cols_exo, target_col, seq_len)

combined_df = pd.concat([
    train_exogenous.iloc[-seq_len:],
    test_exogenous
])

X_test_seq, y_test_seq = create_lstm_sequences(combined_df, feature_cols_exo, target_col, seq_len)

X_test_seq = X_test_seq[-len(test_exogenous):]
y_test_seq = y_test_seq[-len(test_exogenous):]

scaler_X_lstm = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X_lstm.fit_transform(X_train_seq.reshape(-1, X_train_seq.shape[2])).reshape(X_train_seq.shape)
y_train_scaled = scaler_y.fit_transform(y_train_seq.reshape(-1, 1))

X_test_scaled = scaler_X_lstm.transform(X_test_seq.reshape(-1, X_test_seq.shape[2])).reshape(X_test_seq.shape)

input_size = X_train_scaled.shape[2]

model_exo = LSTMModel(
    input_size=input_size,
    hidden_size=lstm_cfg["hidden_size"],
    num_layers=lstm_cfg["num_layers"],
    dropout=lstm_cfg["dropout"]
)

start = time.time()

train_lstm(
    model_exo,
    X_train_scaled,
    y_train_scaled,
    epochs=lstm_cfg["epochs"],
    lr=lstm_cfg["lr"]
)

print("LSTM training time:", time.time() - start)

predictions_scaled = []

model_exo.eval()

with torch.no_grad():
    for start_idx in range(0, len(X_test_scaled), 24):
        end_idx = start_idx + 24

        X_day = X_test_scaled[start_idx:end_idx]
        X_day_tensor = torch.from_numpy(X_day).float()

        pred_day_scaled = model_exo(X_day_tensor).numpy()
        predictions_scaled.extend(pred_day_scaled)

predictions_scaled = np.array(predictions_scaled)
y_pred = scaler_y.inverse_transform(predictions_scaled)

metrics_lstm = evaluate_model(
    test_exogenous["Energy"],
    y_pred.flatten(),
    ts_col="ts_local",
    df=test_exogenous
)

print("\n--- LSTM ---")
print(metrics_lstm)