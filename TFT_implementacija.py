# Bibliotekos
import os
import gc
import logging
import warnings
import psutil
import numpy as np
import pandas as pd
import lightning.pytorch as pl
import optuna

# Įspėjimų pašalinimas
logging.getLogger("lightning.pytorch").setLevel(logging.WARNING)
logging.getLogger("lightning").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", message=".*isinstance.*treespec.*LeafSpec.*")
warnings.filterwarnings("ignore", message=".*predict_dataloader.*num_workers.*")
warnings.filterwarnings("ignore", message=".*Attribute.*loss.*nn.Module.*")
warnings.filterwarnings("ignore", message=".*Attribute.*logging_metrics.*")

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import torch

from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss
import torch.serialization
from pytorch_forecasting.data.encoders import GroupNormalizer
from pytorch_forecasting.data.timeseries import TimeSeriesDataSet

torch.serialization.add_safe_globals([
    GroupNormalizer,
    TimeSeriesDataSet,
])

_original_torch_load = torch.load
torch.load = lambda *a, **kw: _original_torch_load(*a, **{**kw, "weights_only": False})

# Tikrinama, ar užtenka atminties
def check_memory(tag: str = "") -> None:
    ram  = psutil.virtual_memory()
    used = ram.percent
    msg  = f"[MEM{' ' + tag if tag else ''}] RAM {used:.0f}%  ({ram.used/1e9:.1f}/{ram.total/1e9:.1f} GB)"
    if torch.cuda.is_available():
        vram_used = torch.cuda.memory_allocated() / 1e9
        vram_res  = torch.cuda.memory_reserved()  / 1e9
        msg      += f"  |  VRAM alloc {vram_used:.2f} GB  reserved {vram_res:.2f} GB"
    print(msg)
    if used > 88:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise MemoryError(f"RAM usage {used:.0f}% exceeds 88% safety limit - aborting before OOM crash")


# ---------------------------------------------------------------------------------
# Papildoma funkcija, leidžianti modeliui turėti nulinę paklaidą nakties valandoms
# ---------------------------------------------------------------------------------
class DaylightQuantileLoss(QuantileLoss):
    def __init__(self, quantiles, night_threshold: float = 0.0):
        super().__init__(quantiles=quantiles)
        self.night_threshold = night_threshold

    def loss(self, y_pred: torch.Tensor, target) -> torch.Tensor:
        actuals    = target[0]
        base_loss  = super().loss(y_pred, target)
        night_mask = (actuals < self.night_threshold).unsqueeze(-1)
        return base_loss.masked_fill(night_mask, 0.0)


# -------------------------------------------------------
# Hiperparametrai
# -------------------------------------------------------

#Parametrai, kurie yra taikomi kiekvienam duomenų rinkiniui
BASE_PARAMS = {
    "max_encoder_length": 24,
    "max_prediction_length": 24,
    "val_length_hours": 672,
    "quantiles": [0.1, 0.5, 0.9],
}

# Optimizuoti hiperparametrai kiekvienam duomenų rinkiniui
BEST_PARAMS_REGIONS = {

    # -------------------------
    # Šiaurės
    # -------------------------
    "siaures": {
        "hidden_size": 64,
        "hidden_continuous_size": 64,
        "attention_head_size": 4,
        "lstm_layers": 1,
        "dropout": 0.3442049253841043,
        "learning_rate": 5.208256613362788e-05,
        "gradient_clip_val": 0.05,
        "batch_size": 64,
    },

    "siaures_selected": {
        "hidden_size": 64,
        "hidden_continuous_size": 32,
        "attention_head_size": 4,
        "lstm_layers": 1,
        "dropout": 0.2947807891787874,
        "learning_rate": 0.00026854186713749296,
        "gradient_clip_val": 0.2,
        "batch_size": 64,
    },

    # -------------------------
    # Pietinė
    # -------------------------
    "pietine": {
        "hidden_size": 64,
        "hidden_continuous_size": 64,
        "attention_head_size": 4,
        "lstm_layers": 1,
        "dropout": 0.38628897811869833,
        "learning_rate": 0.00021177370632407383,
        "gradient_clip_val": 0.05,
        "batch_size": 128,
    },

    "pietine_selected": {
        "hidden_size": 128,
        "hidden_continuous_size": 64,
        "attention_head_size": 4,
        "lstm_layers": 2,
        "dropout": 0.3807917575041345,
        "learning_rate": 0.00018098481661891575,
        "gradient_clip_val": 0.05,
        "batch_size": 64,
    },

    # -------------------------
    # Centrinė
    # -------------------------
    "centrine": {
        "hidden_size": 64,
        "hidden_continuous_size": 32,
        "attention_head_size": 4,
        "lstm_layers": 2,
        "dropout": 0.29558409125726703,
        "learning_rate": 7.049329588043546e-05,
        "gradient_clip_val": 0.05,
        "batch_size": 128,
    },

    "centrine_selected": {
        "hidden_size": 128,
        "hidden_continuous_size": 32,
        "attention_head_size": 8,
        "lstm_layers": 2,
        "dropout": 0.39841113904454206,
        "learning_rate": 9.751499586482743e-05,
        "gradient_clip_val": 0.1,
        "batch_size": 128,
    },
}

def build_best_params(name: str):
    return {**BASE_PARAMS, **BEST_PARAMS_REGIONS[name]}

# Čia įrašomas pavadinimas, kuris turi reikiamus hiperparametrus naudojamam duomenų rinkiniui
BEST_PARAMS = build_best_params("centrine_selected")


PREDICTION_LENGTH = BEST_PARAMS["max_prediction_length"]
ENCODER_LENGTH    = BEST_PARAMS["max_encoder_length"]
QUANTILES         = BEST_PARAMS["quantiles"]
MEDIAN_IDX        = 1


# -------------------------------------------------------
# Požymių rinkiniai
# -------------------------------------------------------
KNOWN_REALS = [
    # Laiko požymiai
    "daylight", "hour_sin", "hour_cos", "doy_sin", "doy_cos",
    "weekend", "month", "wday",
    # Apšvitos požymiai
    "GTI", "dir_frac", "gti_over_ghi",
    "GTI_max24", "gti_rel_max",
    # Sąveikos
    "GTI_x_Temp", "GTI_x_Clouds", "dir_frac_x_GTI",
    "WindSpeed_sq", "Humidity_sq",
    # Skirtumai
    "dGTI_1", "dTemp_1", "dClouds_1",
    # WeatherID one-hot kodavimai
    "WeatherID501", "WeatherID502",
    "WeatherID600", "WeatherID601",
    "WeatherID800", "WeatherID801",
    "WeatherID802", "WeatherID803", "WeatherID804",
    # Ilgesni velavimai (12h, 24h) - pasiekiami vidurnaktyje
    "GTI_lag12", "GTI_lag24",
    "Temp_lag12", "Temp_lag24",
    "WindSpeed_lag12", "WindSpeed_lag24",
    "Humidity_lag12", "Humidity_lag24",
    "Clouds_lag12", "Clouds_lag24",
    "dir_frac_lag12", "dir_frac_lag24",
    "gti_over_ghi_lag12", "gti_over_ghi_lag24",
    # Slenkantys statistiniai rodikliai (12h, 24h langai)
    "GTI_ma12", "GTI_sd12",
    "GTI_ma24", "GTI_sd24",
    "Temp_ma12", "Temp_ma24",
    "WindSpeed_ma12", "WindSpeed_ma24",
    "Humidity_ma12", "Humidity_ma24",
    "Clouds_ma12", "Clouds_ma24",
    "dir_frac_ma12", "dir_frac_ma24",
    "gti_over_ghi_ma12", "gti_over_ghi_ma24",
]

UNKNOWN_REALS = [
    # Prognozuojamas kintamasis
    "Energy",
    # Trumpesni velavimai (1-6h) - tik enkoderiui
    "GTI_lag01", "GTI_lag02", "GTI_lag03", "GTI_lag06",
    "Temp_lag01", "Temp_lag02", "Temp_lag03", "Temp_lag06",
    "WindSpeed_lag01", "WindSpeed_lag02", "WindSpeed_lag03", "WindSpeed_lag06",
    "Humidity_lag01", "Humidity_lag02", "Humidity_lag03", "Humidity_lag06",
    "Clouds_lag01", "Clouds_lag02", "Clouds_lag03", "Clouds_lag06",
    "dir_frac_lag01", "dir_frac_lag02", "dir_frac_lag03", "dir_frac_lag06",
    "gti_over_ghi_lag01", "gti_over_ghi_lag02", "gti_over_ghi_lag03", "gti_over_ghi_lag06",
    # Slenkantys statistiniai rodikliai (3h, 6h langai)
    "GTI_ma03", "Temp_ma03", "WindSpeed_ma03", "Humidity_ma03",
    "Clouds_ma03", "dir_frac_ma03", "gti_over_ghi_ma03", "GTI_sd03",
    "GTI_ma06", "Temp_ma06", "WindSpeed_ma06", "Humidity_ma06",
    "Clouds_ma06", "dir_frac_ma06", "gti_over_ghi_ma06", "GTI_sd06",
]


# -------------------------------------------------------
# Tikslumo metrikos funkcija
# -------------------------------------------------------
def evaluate_model(y_true, y_pred, ts_col=None, df=None):
    y_true = np.array(y_true, float)
    y_pred = np.array(y_pred, float)

    rmse = np.sqrt(np.nanmean((y_pred - y_true) ** 2))
    mae  = np.nanmean(np.abs(y_pred - y_true))

    denom = np.nansum(np.abs(y_true))
    wmape = np.nan
    if denom > 0:
        wmape = (np.nansum(np.abs(y_pred - y_true)) / denom) * 100

    daily_wmape = np.nan
    if ts_col is not None and df is not None:
        df_temp = df.copy()
        df_temp["pred"] = y_pred
        df_temp["y_true"] = y_true
        df_temp["date"] = pd.to_datetime(df_temp[ts_col]).dt.date

        df_daily = df_temp.groupby("date").agg(
            actual_day=("y_true", "sum"),
            pred_day=("pred", "sum")
        ).reset_index()

        denom_day = np.nansum(np.abs(df_daily["actual_day"]))
        if denom_day > 0:
            daily_wmape = (
                np.nansum(np.abs(df_daily["pred_day"] - df_daily["actual_day"])) /
                denom_day
            ) * 100

    return {
        "RMSE": rmse,
        "MAE": mae,
        "WMAPE": wmape,
        "Daily_WMAPE": daily_wmape
    }

# -------------------------------------------------------
# Duomenų paruošimas
# -------------------------------------------------------
def prepare_df(df_exogenous: pd.DataFrame) -> pd.DataFrame:
    df = df_exogenous.copy()
    df["Energy"] = df["Energy"].clip(lower=0.0001) #Kai neigiama energija, reikia konvertuoti į teigiamą dėl softplus transformacijos
    df["ts_local"] = pd.to_datetime(df["ts_local"])

    if df["ts_local"].dt.tz is not None:
        df["ts_local"] = df["ts_local"].dt.tz_localize(None)

    df = df.sort_values("ts_local").reset_index(drop=True)
    df["time_idx"] = np.arange(len(df), dtype=int)
    df["group"]    = "series_1"

    return df


# -------------------------------------------------------
# Apmokymas
# -------------------------------------------------------
def train_and_evaluate(
    df_exogenous: pd.DataFrame,
    best_params: dict,
    known_reals: list[str] = None,
    unknown_reals: list[str] = None,
    test_fraction: float = 0.20,
    num_workers: int = 0,
    max_epochs: int = 50,
    epsilon_mape: float = 0.0,
    device_precision: str = "bf16-mixed",
    allow_resume: bool = True,
) -> tuple[pd.DataFrame, dict]:

    known_reals   = known_reals   or KNOWN_REALS
    unknown_reals = unknown_reals or UNKNOWN_REALS

    encoder_length    = best_params["max_encoder_length"]
    prediction_length = best_params["max_prediction_length"]
    batch_size        = best_params["batch_size"]
    gradient_clip_val = best_params["gradient_clip_val"]

    model_params = {
        "learning_rate":          best_params["learning_rate"],
        "hidden_size":            best_params["hidden_size"],
        "hidden_continuous_size": best_params["hidden_continuous_size"],
        "attention_head_size":    best_params["attention_head_size"],
        "dropout":                best_params["dropout"],
        "lstm_layers":            best_params["lstm_layers"],
    }

    pl.seed_everything(42, workers=True)
    torch.set_float32_matmul_precision("high")
    check_memory("startup")

    df        = prepare_df(df_exogenous)
    known_reals = [feat for feat in KNOWN_REALS if feat in df.columns]
    unknown_reals = [feat for feat in UNKNOWN_REALS if feat in df.columns]
    n_total   = len(df)
    test_size = int(np.floor(n_total * test_fraction))
    train_end = n_total - test_size - 1

    print(f"\nTotal hours : {n_total}")
    print(f"Train hours : {train_end + 1}  (idx 0 → {train_end})")
    print(f"Test hours  : {test_size}  (idx {train_end + 1} → {n_total - 1})\n")

    train_df = df[df["time_idx"] <= train_end].copy()

    print("Building training dataset")
    training_ds = TimeSeriesDataSet(
        train_df,
        time_idx="time_idx",
        target="Energy",
        group_ids=["group"],
        max_encoder_length=encoder_length,
        min_encoder_length=encoder_length // 2,
        max_prediction_length=prediction_length,
        min_prediction_length=prediction_length,
        static_categoricals=[],
        time_varying_known_reals=known_reals,
        time_varying_unknown_reals=unknown_reals,
        target_normalizer=GroupNormalizer(
            groups=["group"], transformation="softplus",
        ),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )

    val_start     = max(0, train_end - best_params["val_length_hours"])
    validation_ds = TimeSeriesDataSet.from_dataset(
        training_ds,
        df[df["time_idx"] <= train_end].copy(),
        min_prediction_idx=val_start,
        stop_randomization=True,
        predict=True,
    )

    train_loader = training_ds.to_dataloader(
        train=True, batch_size=batch_size,
        num_workers=num_workers, persistent_workers=False,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = validation_ds.to_dataloader(
        train=False, batch_size=batch_size,
        num_workers=num_workers, persistent_workers=False,
        pin_memory=torch.cuda.is_available(),
    )

    check_memory("after datasets")

    loss_fn = DaylightQuantileLoss(quantiles=QUANTILES, night_threshold=epsilon_mape)
    model   = TemporalFusionTransformer.from_dataset(
        training_ds,
        loss=loss_fn,
        reduce_on_plateau_patience=4,
        **model_params,
    )
    print(f"Model parameters : {sum(p.numel() for p in model.parameters()):,}")
    print(f"Loss             : DaylightQuantileLoss (night_threshold={epsilon_mape} kWh)\n")

    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename="tft-epoch{epoch:02d}-val{val_loss:.4f}",
        monitor="val_loss",
        save_top_k=3, mode="min",
        save_last=True, verbose=True,
    )

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        gradient_clip_val=gradient_clip_val,
        enable_checkpointing=True,
        logger=False,
        precision=device_precision,
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=5, min_delta=1e-4),
            checkpoint_callback,
        ],
    )

    if allow_resume:
        last_ckpt = "checkpoints/last.ckpt"
        resume_from = last_ckpt if os.path.exists(last_ckpt) else None
    else:
        resume_from = None
    print("Resuming from checkpoint" if resume_from else "Starting fresh training run")

    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=resume_from,
    )
    print("Training complete.\n")

    import shutil
    best_path = checkpoint_callback.best_model_path
    print(f"Best checkpoint : {best_path}")
    print(f"Best val_loss   : {checkpoint_callback.best_model_score:.4f}")
    shutil.copy(best_path, "tft_solar_model.ckpt")
    print("Final model saved in tft_solar_model.ckpt\n")

    del train_loader, val_loader, validation_ds, trainer
    gc.collect()
    torch.cuda.empty_cache()
    check_memory("after training cleanup")

    results_df, metrics = _run_sliding_inference(
        df=df, model=model, training_ds=training_ds,
        train_end=train_end, n_total=n_total,
        prediction_length=prediction_length,
        encoder_length=encoder_length,
    )

    return results_df, metrics


# -------------------------------------------------------
# Prognozės
# -------------------------------------------------------
def _run_sliding_inference(
    df: pd.DataFrame,
    model: TemporalFusionTransformer,
    training_ds: TimeSeriesDataSet,
    train_end: int,
    n_total: int,
    prediction_length: int,
    encoder_length: int,
) -> tuple[pd.DataFrame, dict]:

    ts_index = df.set_index("time_idx")["ts_local"]
    test_df  = df[df["time_idx"] > train_end].copy()

    results_df = pd.DataFrame({
        "timestamp": test_df["ts_local"].values,
        "actual":    test_df["Energy"].values,
        "D1":     np.nan,
        "D1_p10": np.nan,
        "D1_p90": np.nan,
    }).set_index("timestamp")

    midnight_starts = list(range(train_end + 1, n_total - prediction_length + 1, 24))
    total_runs      = len(midnight_starts)

    model.eval()

    for run_num, pred_start in enumerate(midnight_starts, 1):
        ctx_start = max(0, pred_start - encoder_length)
        window_df = df[
            (df["time_idx"] >= ctx_start) &
            (df["time_idx"] <  pred_start + prediction_length)
        ].copy()

        if len(window_df) < (encoder_length // 2) + prediction_length:
            print(f"  Run {run_num}: skipping - insufficient context ({len(window_df)} rows)")
            continue

        try:
            window_ds = TimeSeriesDataSet.from_dataset(
                training_ds, window_df,
                min_prediction_idx=pred_start,
                stop_randomization=True, predict=True,
            )
            if len(window_ds) == 0:
                continue

            loader = window_ds.to_dataloader(
                train=False, batch_size=64,
                num_workers=0, persistent_workers=False,
            )

            with torch.no_grad():
                raw = model.predict(loader, mode="raw", return_x=True, return_y=False)

            preds = raw.output.prediction.detach().cpu().numpy()
            d_idx = raw.x["decoder_time_idx"].detach().cpu().numpy()

            p10    = preds[0, :, 0]
            median = preds[0, :, 1]
            p90    = preds[0, :, 2]

            for hour in range(prediction_length):
                target_idx = int(d_idx[0, hour])
                target_ts  = ts_index.get(target_idx)
                if target_ts is None or target_ts not in results_df.index:
                    continue
                if pd.isna(results_df.loc[target_ts, "D1"]):
                    results_df.loc[target_ts, "D1"]     = median[hour]
                    results_df.loc[target_ts, "D1_p10"] = p10[hour]
                    results_df.loc[target_ts, "D1_p90"] = p90[hour]

            del raw, preds, d_idx, loader, window_ds, window_df
            gc.collect()

        except Exception as e:
            print(f"  Run {run_num} (pred_start={pred_start}) failed: {e}")
            continue

        if run_num % 10 == 0:
            check_memory(f"run {run_num}/{total_runs}")
            print(f"  Progress : {run_num}/{total_runs} runs complete")

    results_df = results_df.reset_index()

    valid = results_df.dropna(subset=["actual", "D1"])

    metrics = evaluate_model(
        y_true=valid["actual"].values,
        y_pred=valid["D1"].values,
        ts_col="timestamp",
        df=valid
    )

    print("\n" + "=" * 45)
    print("  TEST SET METRICS")
    print("=" * 45)
    for k, v in metrics.items():
        if v is None or np.isnan(v):
            label = "N/A"
        elif "WMAPE" in k:
            label = f"{v:.2f}%"
        else:
            label = f"{v:.4f}"
        print(f"  {k:30s}: {label}")
    print("=" * 45 + "\n")

    return results_df, metrics

if __name__ == "__main__":
    if torch.cuda.is_available():
        precision = "bf16-mixed" if torch.cuda.is_bf16_supported() else "16-mixed"
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        precision = "32"
        print("No GPU - using CPU")

    #Pasirenkamas duomenų rinkinys, kuris bus naudojamas su modeliu
    import pyarrow.parquet as pq
    #df_exogenous = pq.read_table("df_exogenous_siaures_GTI.parquet").to_pandas()
    #df_exogenous = pq.read_table("df_exogenous_pietu_GTI.parquet").to_pandas()
    #df_exogenous = pq.read_table("df_exogenous_centrine_GTI.parquet").to_pandas()

    #df_exogenous = pq.read_table("df_top20_siaures.parquet").to_pandas()
    #df_exogenous = pq.read_table("df_top20_pietu.parquet").to_pandas()
    df_exogenous = pq.read_table("df_top20_centrine.parquet").to_pandas()


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
    df_exogenous_full = df_exogenous.copy()

    df_exogenous_winter = df_exogenous[df_exogenous["season"] == "winter"].copy()
    df_exogenous_spring = df_exogenous[df_exogenous["season"] == "spring"].copy()
    df_exogenous_summer = df_exogenous[df_exogenous["season"] == "summer"].copy()
    df_exogenous_autumn = df_exogenous[df_exogenous["season"] == "autumn"].copy()

    # Čia galima nustatyti, ar naudoti visą duomenų rinkinį, ar tik kažkokį sezoną
    df_exogenous = df_exogenous_full
    df_exogenous = df_exogenous.drop(columns=["season"])

    # Optuna hiperparametrų optimizavimas
    def objective(trial):
        import shutil
        if os.path.exists("checkpoints"):
            shutil.rmtree("checkpoints", ignore_errors=True)

        trial_params = {
            "max_encoder_length": 24,
            "hidden_size":           trial.suggest_categorical("hidden_size", [64, 128]),
            "hidden_continuous_size": trial.suggest_categorical("hidden_continuous_size", [32, 64]),
            "attention_head_size":    trial.suggest_categorical("attention_head_size", [4, 8]),
            "lstm_layers":            trial.suggest_categorical("lstm_layers", [1, 2]),
            "dropout":                trial.suggest_float("dropout", 0.05, 0.40),
            "learning_rate":          trial.suggest_float("learning_rate", 5e-5, 1e-2, log=True),
            "gradient_clip_val":      trial.suggest_categorical("gradient_clip_val", [0.05, 0.1, 0.2]),
            "batch_size":             trial.suggest_categorical("batch_size", [64, 128]),
            "max_prediction_length":  BEST_PARAMS["max_prediction_length"],
            "val_length_hours":       BEST_PARAMS["val_length_hours"],
            "quantiles":              BEST_PARAMS["quantiles"],
        }

        try:
            _, trial_metrics = train_and_evaluate(
                df_exogenous=df_exogenous,
                best_params=trial_params,
                max_epochs=20,
                device_precision=precision,
                allow_resume=False,
                num_workers=4
            )
            return trial_metrics["WMAPE"]
        except Exception as e:
            print("TRIAL FAILED:", repr(e))
            return float('inf')

    # Jei parašyti True, bus atliktas hiperparametrų optimizavimas
    RUN_OPTIMIZATION = False
    if RUN_OPTIMIZATION:
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=20)

        print(f"Best Hourly WMAPE: {study.best_value:.2f}%")
        print("Best Params:", study.best_params)

    import shutil
    if os.path.exists("checkpoints"):
        shutil.rmtree("checkpoints")
        print("Old checkpoints cleared - fresh training run\n")

    results_df, metrics = train_and_evaluate(
        df_exogenous=df_exogenous,
        best_params=BEST_PARAMS,
        known_reals=KNOWN_REALS,
        unknown_reals=UNKNOWN_REALS,
        test_fraction=0.20,
        num_workers=4,
        max_epochs=20,
        epsilon_mape=0.0,
        device_precision=precision,
    )

    results_df.to_csv("tft_test_results.csv", index=False)
    print(results_df.head(10).to_string())