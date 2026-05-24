library(readxl)
library(dplyr)
library(lubridate)
library(tidyr)
library(zoo)
set.seed(42)
#Stulpelių atrinkimas, duomenų failo pavadinimas
FILE_PATH <- "export_siaures.xlsx"
#FILE_PATH <- "SvenAly_pietu.xlsx"
#FILE_PATH <- "UbViev_centrine.xlsx"
LOCAL_TZ  <- "Europe/Vilnius"
TS_COL    <- "ts"
TARGET    <- "Energy"
REQUIRED_ALL <- c("Temp","WindSpeed","Humidity","Clouds","WeatherID")
IRRAD_COLS   <- c("GTI","DNI","DHI","GHI")
CATEGORICAL_VARS <- c("WeatherID")

#Duomenų skaitymas
df <- read_excel(FILE_PATH)

df[[TS_COL]] <- suppressWarnings(as_datetime(df[[TS_COL]], tz = "UTC"))

df <- df %>%
  filter(!is.na(.data[[TS_COL]])) %>%
  distinct(.data[[TS_COL]], .keep_all = TRUE) %>%
  arrange(.data[[TS_COL]])

df_full <- df

req_present <- REQUIRED_ALL %in% names(df)
if (!all(req_present)) {
  stop("Truksta stulpeliu: ", paste(REQUIRED_ALL[!req_present], collapse = ", "))
}
row_complete <- rowSums(is.na(df[REQUIRED_ALL])) == 0

if (all(row_complete)) {
  df_trim <- df
  cutoff_time <- NA
} else if (!any(row_complete)) {
  df_trim <- df[0, , drop = FALSE]
  cutoff_time <- df[[TS_COL]][1]
} else {
  last_good <- max(df[[TS_COL]][row_complete], na.rm = TRUE)
  df_trim <- df %>% filter(.data[[TS_COL]] <= last_good)
  cutoff_time <- last_good + hours(1)
}

sum(is.na(df_trim))
unique(df_trim$WeatherID)

# --------------------
# Pagalbines funkcijos
# --------------------
rollr_mean <- function(x, k) rollapplyr(x, k, mean, na.rm = TRUE, partial = FALSE, fill = NA)
rollr_sd   <- function(x, k) rollapplyr(x, k, sd,   na.rm = TRUE, partial = FALSE, fill = NA)

# ---------------------
# Laikas ir cikliškumas
# ---------------------
df_trim <- df_trim %>% arrange(.data[[TS_COL]])
options(dplyr.width = Inf)

df_trim <- df_trim %>%
  mutate(
    ts_local = with_tz(.data[[TS_COL]], tzone = LOCAL_TZ),
    hour     = hour(ts_local),
    wday     = wday(ts_local, week_start = 1),  # 1=Pirmadienis, ..., 7 = Sekmadienis
    month    = month(ts_local),
    doy      = yday(ts_local),
    week     = isoweek(ts_local),
    weekend  = as.integer(wday %in% c(6,7)),
    hour_sin = sin(2*pi*hour/24),  hour_cos = cos(2*pi*hour/24),
    doy_sin  = sin(2*pi*doy/366),  doy_cos  = cos(2*pi*doy/366)
  )

# -----------------
# Dienos šviesa
# -----------------

df_trim$daylight <- as.integer(df_trim$GTI > 20)

# -------------------------------------------------------
# Tiesioginės spinduliuotės dalis (dir_frac) ir pokyčiai
# -------------------------------------------------------
eps <- 1e-6
df_trim <- df_trim %>%
  mutate(
    dir_frac     = pmax(DNI, 0) / pmax(pmax(DNI, 0) + pmax(DHI, 0), eps),
    gti_over_ghi = GTI / pmax(pmax(GHI, 0), eps), #GTI santykis su GHI
    dGTI_1       = GTI    - lag(GTI, 1),
    dTemp_1      = Temp   - lag(Temp, 1),
    dClouds_1    = Clouds - lag(Clouds, 1)
  )

# -----------------
# Velavimai
# -----------------
lag_set  <- c(1,2,3,6,12,24)
lag_vars <- c("GTI","Temp","WindSpeed","Humidity","Clouds","dir_frac","gti_over_ghi")
for (v in lag_vars) for (L in lag_set) {
  df_trim[[sprintf("%s_lag%02d", v, L)]] <- lag(df_trim[[v]], L)
}

# ---------------------------------
# Slenkantys statistiniai rodikliai
# ---------------------------------
roll_windows   <- c(3,6,12,24)
roll_vars_mean <- c("GTI","Temp","WindSpeed","Humidity","Clouds","dir_frac","gti_over_ghi")
roll_vars_sd   <- c("GTI")
for (k in roll_windows) {
  for (v in roll_vars_mean) df_trim[[sprintf("%s_ma%02d", v, k)]] <- rollr_mean(df_trim[[v]], k)
  for (v in roll_vars_sd)   df_trim[[sprintf("%s_sd%02d", v, k)]] <- rollr_sd(df_trim[[v]], k)
}

# -------------------------------------------------------------
# Aiškios dienos spinduliuotės santykiai
# GTI_max24 = GTI maksimalus per paskutines 24 valandas
# gti_rel_max = GTI santykis su paskutinių 24 valandų maksimumu
# -------------------------------------------------------------
df_trim <- df_trim %>%
  mutate(
    GTI_max24   = rollapplyr(GTI, 24, max, partial = FALSE, fill = NA),
    gti_rel_max = GTI / pmax(GTI_max24, eps)
  )

# -----------------------------
# Netiesiniai efektai, sąveikos
# -----------------------------
df_trim <- df_trim %>%
  mutate(
    GTI_x_Temp     = GTI * Temp,
    GTI_x_Clouds   = GTI * Clouds,
    dir_frac_x_GTI = dir_frac * GTI,
    WindSpeed_sq   = WindSpeed^2,
    Humidity_sq    = Humidity^2
  )

# ------------------------------
# WeatherID paverčiamas faktoriu
# ------------------------------
df_trim$WeatherID <- as.factor(df_trim$WeatherID)

# ----------------------------------------------------------------------
# Eilučių pašalinimas, kuriose yra NA dėl velavimų / slenkančių rodiklių
# ----------------------------------------------------------------------
na_cut <- max(c(lag_set, roll_windows), na.rm = TRUE)
df_fe  <- df_trim %>% slice((na_cut + 1):n())

# ---------------------------
# One-hot kodavimas WeatherID
# ---------------------------
make_ohe <- function(d, factor_col = "WeatherID") {
  stopifnot(factor_col %in% names(d))
  d[[factor_col]] <- as.factor(d[[factor_col]])
  mm <- model.matrix(reformulate(factor_col, response = NULL), data = d)
  mm <- mm[, colnames(mm) != "(Intercept)", drop = FALSE]
  cbind(d[, setdiff(names(d), factor_col), drop = FALSE], mm)
}
df_ohe <- make_ohe(df_fe, "WeatherID")

# ----------------------------
# Galutiniai požymių rinkiniai
# ----------------------------
time_feats <- c("daylight","hour_sin","hour_cos","doy_sin","doy_cos","weekend","month","wday")

irr_feats_core <- c("GTI","dir_frac","gti_over_ghi","GTI_max24","gti_rel_max")

deltas <- c("dGTI_1","dTemp_1","dClouds_1")

interactions <- c("GTI_x_Temp","GTI_x_Clouds","dir_frac_x_GTI","WindSpeed_sq","Humidity_sq")

driver_keep_prefix <- c("GTI","dir_frac","gti_over_ghi","Temp","WindSpeed","Humidity","Clouds")
driver_pat <- paste0("^(", paste(driver_keep_prefix, collapse = "|"), ")_(lag|ma|sd)")
driver_lagroll_cols <- grep(driver_pat, names(df_ohe), value = TRUE)

# WeatherID stulpeliai
weather_ohe_cols <- grep("^WeatherID", names(df_ohe), value = TRUE)

TS_COL_LOCAL <- "ts_local"
base_cols <- c(TS_COL_LOCAL, TARGET)

keep_common <- unique(c(base_cols, time_feats, irr_feats_core, deltas, interactions, weather_ohe_cols))
keep_exogenous <- unique(c(keep_common, driver_lagroll_cols))
df_exogenous <- df_ohe[, intersect(keep_exogenous, names(df_ohe)), drop = FALSE]

#Duomenų rinkinių išsaugojimas
library(arrow)
#write_parquet(df_exogenous, "df_exogenous_siaures.parquet")
#write_parquet(df_exogenous, "df_exogenous_centrine.parquet")
#write_parquet(df_exogenous, "df_exogenous_pietu.parquet")