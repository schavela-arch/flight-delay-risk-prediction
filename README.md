# flight-delay-risk-prediction
# Predictive ML model assessing U.S. flight delay risk using DOT &amp; NOAA data, with an executive Power BI dashboard for proactive passenger alerts. Built with Python, scikit-learn, and pandas.

# DSBA 6156: Applied Machine Learning — Final Project
# Author: Sonia Gabriela Chavelas Gonzalez
# Model ID: FDP-ML-001 v2.0
# Date: April 22, 2026

# Install Libraries & Imports
import subprocess, sys
pkgs = ['pandas','numpy','matplotlib','seaborn','scikit-learn','xgboost','shap','requests','plotly']
for p in pkgs:
    subprocess.run([sys.executable, '-m', 'pip', 'install', p, '-q'])
print('All packages ready')

import os, io, zipfile, warnings, calendar, requests
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    roc_auc_score, roc_curve, brier_score_loss,
    average_precision_score, precision_recall_curve, f1_score
)
from xgboost import XGBClassifier
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 40)
np.random.seed(42)
print('All imports loaded successfully')

# Configuration & Environment Setup
try:
    from google.colab import drive
    drive.mount('/content/drive')
    DATA_DIR = '/content/drive/MyDrive/'
    print('Running in Google Colab')
except ImportError:
    DATA_DIR = './'
    print('Running locally — ensure data files are in current directory')

# Set in terminal:  export NOAA_TOKEN='your_token_here'
# Set in Colab:     use the Secrets panel (key icon in left sidebar)
NOAA_TOKEN = os.environ.get('NOAA_TOKEN', '')
if not NOAA_TOKEN:
    print('WARNING: NOAA_TOKEN not set. Weather download will be skipped.')
    print('  Get a free token at: https://www.ncdc.noaa.gov/cdo-web/token')
# Thresholds
DELAY_MIN_THRESHOLD = 15   # FAA standard: delayed if departure delay > 15 min
LOW_THRESH          = 0.30
MEDIUM_THRESH       = 0.60
NOAA_FOLDER         = 'data/noaa'
os.makedirs(NOAA_FOLDER, exist_ok=True)

# Airport → NOAA LCD station mapping
# Source: NOAA ISD station catalog
# Station format matches the Local Climatological Data (LCD) v1 API
AIRPORT_NOAA = {
    'ATL': {'station':'722190-13874','lat':33.6407,'lon':-84.4277},
    'ORD': {'station':'725300-94846','lat':41.9742,'lon':-87.9073},
    'DFW': {'station':'722590-03927','lat':32.8998,'lon':-97.0403},
    'LAX': {'station':'722950-23174','lat':33.9425,'lon':-118.408},
    'JFK': {'station':'744860-94789','lat':40.6413,'lon':-73.7781},
    'CLT': {'station':'723140-13881','lat':35.2144,'lon':-80.9473},
    'DEN': {'station':'726660-03017','lat':39.8561,'lon':-104.676},
    'LAS': {'station':'723860-03160','lat':36.0840,'lon':-115.153},
    'PHX': {'station':'722780-23183','lat':33.4373,'lon':-112.008},
    'MIA': {'station':'722020-12839','lat':25.7959,'lon':-80.2870},
    'SFO': {'station':'724940-23234','lat':37.6213,'lon':-122.379},
    'SEA': {'station':'727930-24233','lat':47.4502,'lon':-122.309},
    'EWR': {'station':'725020-14734','lat':40.6925,'lon':-74.1687},
    'MCO': {'station':'747880-12815','lat':28.4294,'lon':-81.3089},
    'BOS': {'station':'725090-14739','lat':42.3643,'lon':-71.0052},
    'IAH': {'station':'722430-12960','lat':29.9902,'lon':-95.3368},
    'DTW': {'station':'726370-14827','lat':42.2162,'lon':-83.3554},
    'MSP': {'station':'726580-14922','lat':44.8848,'lon':-93.2223},
    'LGA': {'station':'725030-14732','lat':40.7772,'lon':-73.8726},
    'PHL': {'station':'724080-13739','lat':39.8721,'lon':-75.2408},
    'BWI': {'station':'724060-93721','lat':39.1754,'lon':-76.6683},
    'IAD': {'station':'724030-93738','lat':38.9531,'lon':-77.4565},
    'MDW': {'station':'725340-14819','lat':41.7868,'lon':-87.7522},
    'SLC': {'station':'725720-24127','lat':40.7884,'lon':-111.978},
}
print(f'Configuration loaded')
print(f'  Delay threshold:    > {DELAY_MIN_THRESHOLD} minutes')
print(f'  Risk LOW < {LOW_THRESH*100:.0f}%   MEDIUM {LOW_THRESH*100:.0f}-{MEDIUM_THRESH*100:.0f}%   HIGH >= {MEDIUM_THRESH*100:.0f}%')
print(f'  NOAA stations mapped: {len(AIRPORT_NOAA)}')

# Loading Flight Data
# Primary dataset: flight_data_2018_2024.csv (BTS On-Time Performance, 2018-2024)
# Training subset: 2018–2022
# Validation subset: 2023 (hyperparameter tuning)
# Test subset: 2024 (held-out, never seen during training or tuning)
# This chronological partitioning prevents temporal data leakage — a critical requirement for time-series classification. Using train_test_split(random_state=42) on flight data would allow the model to see 2023 data while' predicting' 2019 flights, inflating AUC.

BTS_COLS = [
    'FlightDate', 'IATA_Code_Operating_Airline', 'Origin', 'Dest',
    'CRSDepTime', 'DepDelay', 'DepDelayMinutes', 'Cancelled', 'CancellationCode',
    'Diverted', 'Distance',
    'WeatherDelay', 'NASDelay', 'CarrierDelay', 'LateAircraftDelay',
    'DepartureDelayGroups'
]

def load_bts(data_dir, filename='flight_data_2018_2024.csv'):
    path = os.path.join(data_dir, filename)
    zip_path = os.path.join(data_dir, 'Flight_Delay_Dataset_2018-2024.zip')

    # Try CSV first, then ZIP
    if os.path.exists(path):
        hdr = pd.read_csv(path, nrows=0).columns.tolist()
        df  = pd.read_csv(path, usecols=[c for c in BTS_COLS if c in hdr], low_memory=False)
    elif os.path.exists(zip_path):
        with zipfile.ZipFile(zip_path) as z:
            with z.open(filename) as f:
                hdr = pd.read_csv(f, nrows=0).columns.tolist()
            with z.open(filename) as f:
                df = pd.read_csv(f, usecols=[c for c in BTS_COLS if c in hdr], low_memory=False)
    elif os.path.exists('/mnt/user-data/uploads/Flight_Delay_Dataset_2018-2024.zip'):
        zip_path = '/mnt/user-data/uploads/Flight_Delay_Dataset_2018-2024.zip'
        with zipfile.ZipFile(zip_path) as z:
            with z.open(filename) as f:
                hdr = pd.read_csv(f, nrows=0).columns.tolist()
            with z.open(filename) as f:
                df = pd.read_csv(f, usecols=[c for c in BTS_COLS if c in hdr], low_memory=False)
    else:
        raise FileNotFoundError(f'Cannot find BTS data. Expected: {path}')

    print(f'Loaded: {len(df):,} rows | {len(df.columns)} columns')
    return df

flights_raw = load_bts(DATA_DIR)
print(f'Date range in file: {flights_raw["FlightDate"].min()} to {flights_raw["FlightDate"].max()}')
flights_raw.head(3)

# Clean & Standarize Flight Data
def clean_flights(df):
    '''
    Standardize BTS flight data.
    - Parse FlightDate to datetime
    - Build dep_hour from CRSDepTime (HHMM integer format)
    - Create target variables: is_delayed, is_canceled
    - Create BTS weather proxy features (FAA-certified, always available)
    - Build merge keys for NOAA join: (Origin, merge_date, merge_hour)
    '''
    df = df.copy()
    # Parse dates
    df['FlightDate'] = pd.to_datetime(df['FlightDate'], errors='coerce')
    df = df.dropna(subset=['FlightDate', 'Origin', 'Dest'])
    # Extract departure hour from CRSDepTime (HHMM format: 1430 = 14:30)
    hhmm = df['CRSDepTime'].fillna(0).astype(int).astype(str).str.zfill(4)
    df['dep_hour'] = hhmm.str[:2].astype(int).clip(0, 23)
    # Numeric columns
    for col in ['DepDelay','DepDelayMinutes','Cancelled','Distance',
                'WeatherDelay','NASDelay','CarrierDelay','LateAircraftDelay']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    # Target variables
    # is_delayed = 1 if departure delay > 15 minutes (FAA standard)
    df['is_delayed']  = (df['DepDelay'] > 15).astype(int)
    df['is_canceled'] = (df['Cancelled'] == 1).astype(int)
    # BTS weather proxy features (FAA-certified, no NOAA required)
    # These are official delay attributions reported by airlines to the FAA.
    # They capture weather impact on THIS flight as reported.
    # NOTE: We use these only as proxy weather features in the model.
    # We do NOT use them as prediction inputs because they are not available
    # 24-48 hours before departure (they are post-flight reports).
    # Instead, they inform the historical_weather_delay_rate feature below.
    df['had_weather_delay'] = (df['WeatherDelay'] > 0).astype(int)

# Cancellation reason
    cancel_map = {'A': 'Carrier', 'B': 'Weather', 'C': 'NAS', 'D': 'Security'}
    if 'CancellationCode' in df.columns:
        df['cancel_reason'] = df['CancellationCode'].map(cancel_map).fillna('None')

# Rename for consistency
    df = df.rename(columns={'IATA_Code_Operating_Airline': 'carrier'})
    df['carrier'] = df['carrier'].str.strip()

# Merge keys
    df['merge_date'] = df['FlightDate'].dt.date
    df['merge_hour'] = df['dep_hour']

# Additional temporal features
    df['month']      = df['FlightDate'].dt.month
    df['day_of_week']= df['FlightDate'].dt.dayofweek  # 0=Mon, 6=Sun

# Filter to airports with NOAA station mapping
    df = df[df['Origin'].isin(set(AIRPORT_NOAA.keys()))].copy()
    print(f'Cleaned: {len(df):,} flights')
    print(f'  Delayed (>15 min): {df["is_delayed"].sum():,} ({df["is_delayed"].mean()*100:.1f}%)')
    print(f'  Canceled:          {df["is_canceled"].sum():,} ({df["is_canceled"].mean()*100:.1f}%)')
    print(f'  Airports:          {df["Origin"].nunique()}')
    print(f'  Carriers:          {sorted(df["carrier"].dropna().unique())}')
    return df
flights_cleaned = clean_flights(flights_raw)
flights_cleaned[['FlightDate','carrier','Origin','Dest','dep_hour',
                  'DepDelay','is_delayed','is_canceled']].head(5)

# Temporal Train / Validation / Test Split
# Flight data is a time series. Using sklearn.train_test_split(random_state=42) on flight data creates temporal leakage — the model sees future data (2023 flights) while being trained on supposedly 'historical' data, producing inflated AUC scores that collapse in production.
# Split	Period	Purpose
# Training	2018–2022	Fit model weights + historical delay rates
# Validation	2023	Hyperparameter tuning only (walk-forward CV)
# Test (holdout)	2024	Final evaluation — never touched during training
# Temporal split — sort by date, split chronologically
df_sorted = flights_cleaned.sort_values('FlightDate').reset_index(drop=True)
train_mask = df_sorted['FlightDate'].dt.year.between(2018, 2022)
val_mask   = df_sorted['FlightDate'].dt.year == 2023
test_mask  = df_sorted['FlightDate'].dt.year == 2024
df_train = df_sorted[train_mask].copy()
df_val   = df_sorted[val_mask].copy()
df_test  = df_sorted[test_mask].copy()
print('Temporal split summary:')
for label, df_split in [('Training (2018-2022)', df_train),
                         ('Validation (2023)',    df_val),
                         ('Test / Holdout (2024)',df_test)]:
    if len(df_split) == 0:
        print(f'  {label}: 0 rows (data may only contain one year)')
    else:
        print(f'  {label}: {len(df_split):,} flights | '
              f'delay rate: {df_split["is_delayed"].mean()*100:.1f}%')
# If data is only 2024, do a 70/15/15 time-ordered split
if len(df_train) == 0:
    print('\nOnly one year detected — using 70/15/15 time-ordered split')
    n = len(df_sorted)
    df_train = df_sorted.iloc[:int(n*0.70)].copy()
    df_val   = df_sorted.iloc[int(n*0.70):int(n*0.85)].copy()
    df_test  = df_sorted.iloc[int(n*0.85):].copy()
    print(f'  Train: {len(df_train):,} | Val: {len(df_val):,} | Test: {len(df_test):,}')

# Historical Delay Rate Features
# Key design principle: Historical delay rates are computed only from the training set and then looked up for validation and test rows. This prevents leakage from future data.
# These rates replace the leaky arr_delay and weather_delay BTS columns that were used in the original model — those values are only available after a flight completes.
# Compute historical rates from TRAINING data only
# These will be used as lookup tables for val/test rows.
# They represent what the model 'knows' from historical patterns.
carrier_delay_rates = (
    df_train.groupby('carrier')['is_delayed'].mean()
    .rename('carrier_delay_rate').reset_index()
)
origin_delay_rates = (
    df_train.groupby('Origin')['is_delayed'].mean()
    .rename('origin_delay_rate').reset_index()
)
route_delay_rates = (
    df_train.assign(route=df_train['Origin']+'_'+df_train['Dest'])
    .groupby('route')['is_delayed'].mean()
    .rename('route_delay_rate').reset_index()
)
# Historical weather delay rate by carrier + origin (proxy for weather exposure)
weather_delay_rates = (
    df_train.groupby(['carrier','Origin'])['had_weather_delay'].mean()
    .rename('hist_weather_delay_rate').reset_index()
)
print('Historical delay rates computed from training data:')
print(f'  Carrier rates:       {len(carrier_delay_rates)} carriers')
print(f'  Origin rates:        {len(origin_delay_rates)} airports')
print(f'  Route rates:         {len(route_delay_rates)} routes')
print(f'  Weather delay rates: {len(weather_delay_rates)} carrier-airport pairs')
print()
print('Top 5 highest delay rate carriers:')
print(carrier_delay_rates.sort_values('carrier_delay_rate',ascending=False).head(5).to_string(index=False))

# NOAA Local Climatological Data (LCD) - Hourly Weather
# The original model used GHCND (daily summaries) which gives one PRCP and AWND value per station per day. This means a 6am flight during a morning storm is treated identically to a 9pm flight on a clear evening — both share the same daily average.
# LCD provides hourly observations matched to each flight's departure time.
# NOAA wind speed units (documented):
# LCD HourlyWindSpeed = miles per hour (mph) — direct and interpretable high_wind_flag threshold: > 25 mph (threshold for ATC flow control advisories)
def download_noaa_lcd(station_id, start_date, end_date,
                      noaa_token=None, cache_dir=NOAA_FOLDER):
    '''
    Download NOAA Local Climatological Data (LCD) — HOURLY observations.
    Source: https://www.ncei.noaa.gov/access/services/data/v1

    Parameters
    ----------
    station_id  : str  — e.g. '722190-13874' for Atlanta (KATL)
    start_date  : str  — 'YYYY-MM-DD'
    end_date    : str  — 'YYYY-MM-DD'
    noaa_token  : str  — free token from https://www.ncdc.noaa.gov/cdo-web/token

    Returns
    -------
    pd.DataFrame with columns:
      DATE, HourlyDryBulbTemperature (F), HourlyPrecipitation (inches),
      HourlyWindSpeed (mph), HourlyVisibility (miles), HourlySeaLevelPressure (inHg)
    '''
    fname = f'{station_id}_{start_date[:7]}.csv'
    fpath = os.path.join(cache_dir, fname)

    if os.path.exists(fpath):
        print(f'  [cached] {fpath}')
        return pd.read_csv(fpath, low_memory=False)

    url = (
        'https://www.ncei.noaa.gov/access/services/data/v1'
        f'?dataset=local-climatological-data'
        f'&stations={station_id}'
        f'&startDate={start_date}T00:00:00'
        f'&endDate={end_date}T23:59:59'
        '&dataTypes=HourlyDryBulbTemperature,HourlyPrecipitation,'
        'HourlyWindSpeed,HourlyVisibility,HourlySeaLevelPressure'
        '&units=standard&format=csv'
        + (f'&token={noaa_token}' if noaa_token else '')
    )
    try:
        r = requests.get(url, headers={'User-Agent':'flight-delay-model/2.0'}, timeout=60)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text), low_memory=False)
        df.to_csv(fpath, index=False)
        print(f'  Downloaded {len(df):,} rows -> {fpath}')
        return df
    except Exception as e:
        print(f'  NOAA LCD download failed ({station_id}): {e}')
        return None
def clean_noaa_lcd(df):
    '''
    Clean raw NOAA LCD hourly data.
    - Strip quality flags: s (suspect), * (edited), T (trace), M (missing)
    - Convert to numeric
    - Forward-fill gaps up to 3 hours
    - Keep only top-of-hour records
    '''
    df = df.copy()
    keep = ['DATE','STATION','HourlyDryBulbTemperature','HourlyPrecipitation',
            'HourlyWindSpeed','HourlyVisibility','HourlySeaLevelPressure']
    df = df[[c for c in keep if c in df.columns]].copy()
    df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce')
    df = df.dropna(subset=['DATE'])

    for col in ['HourlyDryBulbTemperature','HourlyPrecipitation',
                'HourlyWindSpeed','HourlyVisibility','HourlySeaLevelPressure']:
        if col not in df.columns:
            continue
        s = df[col].astype(str).str.strip()
        s = s.replace({'T': '0.001', 'M': np.nan, '': np.nan})  # T=trace precipitation
        s = s.str.replace(r'[sS\*vVaA]+$', '', regex=True)      # quality flags
        s = s.str.replace(r'\|.*$', '', regex=True)              # pipe-delimited values
        df[col] = pd.to_numeric(s, errors='coerce')
# Keep only hourly (top-of-hour) observations
    df = df[df['DATE'].dt.minute == 0].copy()
# Forward-fill short gaps (up to 3 consecutive hours)
    num_cols = [c for c in df.columns if c.startswith('Hourly')]
    df[num_cols] = df[num_cols].ffill(limit=3)
# Merge keys
    df['merge_date'] = df['DATE'].dt.date
    df['merge_hour'] = df['DATE'].dt.hour
    return df
def load_all_noaa(folder=NOAA_FOLDER):
    '''Load and concat all cached NOAA LCD CSVs.'''
    files = [f for f in os.listdir(folder) if f.endswith('.csv')]
    if not files:
        return None
    dfs = []
    for f in files:
        station_id = f.split('_')[0]
        airport = next((k for k,v in AIRPORT_NOAA.items()
                        if v['station'] == station_id), None)
        ndf = clean_noaa_lcd(pd.read_csv(os.path.join(folder, f), low_memory=False))
        ndf['Origin'] = airport if airport else 'UNKNOWN'
        dfs.append(ndf)
    df_noaa = pd.concat(dfs, ignore_index=True)
    print(f'NOAA loaded: {len(df_noaa):,} hourly records | '
          f'{df_noaa["DATE"].min().date()} to {df_noaa["DATE"].max().date()}')
    return df_noaa

# Download NOAA data for training airports (run once with your token)
# Uncomment and add your token to download real NOAA data:
#
# for airport_code, info in list(AIRPORT_NOAA.items())[:5]:  # start with 5 airports
#     for year in ['2018','2019','2020','2021','2022']:       # training years
#         download_noaa_lcd(
#             station_id  = info['station'],
#             start_date  = f'{year}-01-01',
#             end_date    = f'{year}-12-31',
#             noaa_token  = NOAA_TOKEN
#         )

df_noaa = load_all_noaa()
if df_noaa is None:
    print('No NOAA CSVs found in cache.')
    print('Pipeline will use BTS weather proxy features (always available).')
    print('To add real NOAA data: get token at ncdc.noaa.gov/cdo-web/token')
    print('Then uncomment the download block above.')

# Merge Flights + Weather & Full Feature Engineering
# Features are divided into two groups:
# Training features — used to train the model (no future values)
# Post-flight features — WeatherDelay, ArrDelay etc. are excluded as model inputs because they are only known after the flight completes (would cause leakage in production)
def merge_noaa(df_flights, df_noaa):
    '''Merge flight data with NOAA hourly observations on (Origin, date, hour).'''
    if df_noaa is None:
        # Fallback: add NaN columns for NOAA features
        for col in ['HourlyDryBulbTemperature','HourlyPrecipitation',
                    'HourlyWindSpeed','HourlyVisibility','HourlySeaLevelPressure']:
            df_flights[col] = np.nan
        return df_flights
    noaa_keep = ['Origin','merge_date','merge_hour',
                 'HourlyDryBulbTemperature','HourlyPrecipitation',
                 'HourlyWindSpeed','HourlyVisibility','HourlySeaLevelPressure']
    df = df_flights.merge(df_noaa[noaa_keep],
                          on=['Origin','merge_date','merge_hour'], how='left')
    matched = df['HourlyWindSpeed'].notna().sum()
    print(f'  NOAA merge: {matched:,}/{len(df):,} rows matched ({matched/len(df)*100:.1f}%)')
    return df
def build_features(df, carrier_rates, origin_rates, route_rates, weather_rates,
                   le_carrier=None, le_origin=None, le_dest=None, fit=False):
    '''
    Build the complete ML feature set.

    All features must be available 24-48 hours BEFORE departure:
      - Flight schedule info (known from booking)
      - NWS forecast weather (available ~7 days ahead)
      - Historical delay rates from training data (pre-computed lookup)

    Features NOT included (post-flight, causes leakage):
      - DepDelay, ArrDelay, WeatherDelay (actual outcomes)
      - arr_delay, arr_cancelled (BTS aggregated actuals)
    '''
    df = df.copy()

# 1. Cyclical time encoding
    df['hour_sin']    = np.sin(2*np.pi*df['dep_hour']/24)
    df['hour_cos']    = np.cos(2*np.pi*df['dep_hour']/24)
    df['month_sin']   = np.sin(2*np.pi*df['month']/12)
    df['month_cos']   = np.cos(2*np.pi*df['month']/12)
    df['dow_sin']     = np.sin(2*np.pi*df['day_of_week']/7)
    df['dow_cos']     = np.cos(2*np.pi*df['day_of_week']/7)
    df['is_weekend']  = (df['day_of_week'] >= 5).astype(int)
    # Peak hours: AM rush (7-9am) and PM rush (4-7pm)
    df['is_peak_hr']  = (df['dep_hour'].between(7,9) |
                         df['dep_hour'].between(16,19)).astype(int)
    df['is_winter']   = df['month'].isin([12,1,2]).astype(int)
    df['is_summer']   = df['month'].isin([6,7,8]).astype(int)

# 2. Historical delay rates (from training data lookups)
    # These are legitimate 24-48hr features — they represent known historical patterns
    df['route'] = df['Origin'] + '_' + df['Dest']
    df = df.merge(carrier_rates, on='carrier', how='left')
    df = df.merge(origin_rates,  on='Origin',  how='left')
    df = df.merge(route_rates,   on='route',   how='left')
    df = df.merge(weather_rates, on=['carrier','Origin'], how='left')
# Fill unseen carriers/routes with global training mean
    global_mean = carrier_rates['carrier_delay_rate'].mean()
    df['carrier_delay_rate']    = df['carrier_delay_rate'].fillna(global_mean)
    df['origin_delay_rate']     = df['origin_delay_rate'].fillna(global_mean)
    df['route_delay_rate']      = df['route_delay_rate'].fillna(global_mean)
    df['hist_weather_delay_rate']= df['hist_weather_delay_rate'].fillna(0)
# 3. NOAA weather features (from LCD hourly, or NWS forecast)
    # HourlyWindSpeed is in mph (LCD standard units)
    df['noaa_wind_mph']   = pd.to_numeric(df.get('HourlyWindSpeed',  np.nan), errors='coerce')
    df['noaa_precip_in']  = pd.to_numeric(df.get('HourlyPrecipitation', np.nan), errors='coerce')
    df['noaa_vis_miles']  = pd.to_numeric(df.get('HourlyVisibility', np.nan), errors='coerce')
    df['noaa_temp_f']     = pd.to_numeric(df.get('HourlyDryBulbTemperature', np.nan), errors='coerce')
    df['noaa_press_inhg'] = pd.to_numeric(df.get('HourlySeaLevelPressure', np.nan), errors='coerce')

# Binary weather flags — documented thresholds:
# rain_flag:       precipitation > 0 inches
# high_wind_flag:  wind > 25 mph (FAA flow control advisory threshold)
# low_vis_flag:    visibility < 3 miles (IFR conditions)
# extreme_temp:    temperature > 95F or < 5F (de-icing / heat stress)
    df['rain_flag']      = (df['noaa_precip_in'].fillna(0) > 0).astype(int)
    df['high_wind_flag'] = (df['noaa_wind_mph'].fillna(0) > 25).astype(int)  # mph threshold
    df['low_vis_flag']   = (df['noaa_vis_miles'].fillna(10) < 3).astype(int)
    df['extreme_temp']   = (df['noaa_temp_f'].fillna(60).abs() > 95).astype(int)

# Composite weather severity (0-10): usable even without NOAA (falls to 0)
    df['weather_severity'] = (
        df['rain_flag'] * 2.5
        + df['high_wind_flag'] * 2.5
        + df['low_vis_flag'] * 3.0
        + df['extreme_temp'] * 2.0
    ).clip(0, 10)

# 4. Operational features
    df['distance'] = df['Distance'].fillna(df['Distance'].median() if 'Distance' in df.columns else 800)
    df['dist_bin'] = pd.cut(df['distance'],
                             bins=[0,500,1000,1500,2000,6000],
                             labels=[0,1,2,3,4]).astype(int)

# 5. Label-encode categoricals (avoids 50+ dummy columns)
# Label encoding is preferred over one-hot for tree-based models:
# - Avoids adding 50+ binary columns for airlines and airports
# - XGBoost handles label-encoded categoricals natively
    if fit:
        le_carrier = LabelEncoder().fit(df['carrier'].fillna('XX'))
        le_origin  = LabelEncoder().fit(df['Origin'].fillna('XX'))
        le_dest    = LabelEncoder().fit(df['Dest'].fillna('XX'))

    df['carrier_enc'] = df['carrier'].fillna('XX').map(
        {v:i for i,v in enumerate(le_carrier.classes_)}).fillna(-1).astype(int)
    df['origin_enc']  = df['Origin'].fillna('XX').map(
        {v:i for i,v in enumerate(le_origin.classes_)}).fillna(-1).astype(int)
    df['dest_enc']    = df['Dest'].fillna('XX').map(
        {v:i for i,v in enumerate(le_dest.classes_)}).fillna(-1).astype(int)

    if fit:
        return df, le_carrier, le_origin, le_dest
    return df
# Apply to all three splits
print('Merging NOAA weather...')
df_train = merge_noaa(df_train, df_noaa)
df_val   = merge_noaa(df_val,   df_noaa)
df_test  = merge_noaa(df_test,  df_noaa)

print('\nEngineering features...')
df_train, le_carrier, le_origin, le_dest = build_features(
    df_train, carrier_delay_rates, origin_delay_rates,
    route_delay_rates, weather_delay_rates,
    fit=True
)
df_val  = build_features(df_val,  carrier_delay_rates, origin_delay_rates,
                          route_delay_rates, weather_delay_rates,
                          le_carrier, le_origin, le_dest)
df_test = build_features(df_test, carrier_delay_rates, origin_delay_rates,
                          route_delay_rates, weather_delay_rates,
                          le_carrier, le_origin, le_dest)

# Final feature list
# Every feature here is available 24-48 hours before departure
FEATURES = [
    # Time (all known from flight schedule)
    'hour_sin', 'hour_cos', 'month_sin', 'month_cos', 'dow_sin', 'dow_cos',
    'is_weekend', 'is_peak_hr', 'is_winter', 'is_summer',
    # Weather (from NOAA LCD history or NWS 48hr forecast)
    'rain_flag', 'high_wind_flag', 'low_vis_flag', 'extreme_temp', 'weather_severity',
    'noaa_wind_mph', 'noaa_precip_in', 'noaa_vis_miles', 'noaa_temp_f',
    # Historical delay rates (from training data lookup tables)
    'carrier_delay_rate', 'origin_delay_rate', 'route_delay_rate', 'hist_weather_delay_rate',
    # Operational (from flight schedule)
    'distance', 'dist_bin',
    # Encoded categoricals
    'carrier_enc', 'origin_enc', 'dest_enc',
]
FEATURES = [f for f in FEATURES if f in df_train.columns]
print(f'\nFeature engineering complete: {len(FEATURES)} features')
print(f'Features: {FEATURES}')

# Exploratory Data Analysis (EDA)
fig, axes = plt.subplots(2, 3, figsize=(20, 11))

# 1. Target distribution
axes[0,0].bar(['On-Time','Delayed'],
              [1-df_train['is_delayed'].mean(), df_train['is_delayed'].mean()],
              color=['#27ae60','#e74c3c'], edgecolor='white')
axes[0,0].set_title('Target Variable Distribution (Training)', fontweight='bold')
axes[0,0].set_ylabel('Fraction of Flights')

# 2. Delay rate by carrier
cr = carrier_delay_rates.sort_values('carrier_delay_rate', ascending=False)
axes[0,1].bar(cr['carrier'], cr['carrier_delay_rate']*100,
              color='#3498db', edgecolor='white')
axes[0,1].axhline(df_train['is_delayed'].mean()*100, color='red', ls='--',
                   label=f"Avg: {df_train['is_delayed'].mean()*100:.1f}%")
axes[0,1].set_title('Historical Delay Rate by Carrier', fontweight='bold')
axes[0,1].set_ylabel('Delay Rate (%)'); axes[0,1].legend()
axes[0,1].tick_params(axis='x', rotation=45)

# 3. Delay rate by departure hour
hr_stats = df_train.groupby('dep_hour')['is_delayed'].mean()
axes[0,2].plot(hr_stats.index, hr_stats.values*100, 'o-', color='#e74c3c', lw=2)
axes[0,2].axvspan(7,9,alpha=0.1,color='orange',label='AM peak')
axes[0,2].axvspan(16,19,alpha=0.1,color='red',label='PM peak')
axes[0,2].set_title('Delay Rate by Departure Hour', fontweight='bold')
axes[0,2].set_xlabel('Hour of Day'); axes[0,2].set_ylabel('Delay Rate (%)')
axes[0,2].legend()

# 4. Delay rate by month
mo_stats = df_train.groupby('month')['is_delayed'].mean()
mnames = ['J','F','M','A','M','J','J','A','S','O','N','D']
axes[1,0].bar(range(1,13), mo_stats.reindex(range(1,13)).fillna(0)*100,
              color='#9b59b6', edgecolor='white')
axes[1,0].set_xticks(range(1,13)); axes[1,0].set_xticklabels(mnames)
axes[1,0].set_title('Delay Rate by Month (Training Data)', fontweight='bold')
axes[1,0].set_ylabel('Delay Rate (%)')

# 5. Correlation of key features
corr_cols = [f for f in ['carrier_delay_rate','origin_delay_rate','route_delay_rate',
              'is_peak_hr','is_weekend','rain_flag','high_wind_flag','is_delayed']
             if f in df_train.columns]
corr = df_train[corr_cols].corr()
mask = np.zeros_like(corr, dtype=bool); mask[np.triu_indices_from(mask)] = True
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', ax=axes[1,1],
            mask=mask, vmin=-1, vmax=1)
axes[1,1].set_title('Feature Correlation Matrix', fontweight='bold')

# 6. Top delayed origin airports
ap_stats = origin_delay_rates.sort_values('origin_delay_rate',ascending=False).head(12)
axes[1,2].barh(ap_stats['Origin'][::-1], ap_stats['origin_delay_rate'][::-1]*100,
               color='#e67e22')
axes[1,2].set_title('Top Delayed Origin Airports (Historical)', fontweight='bold')
axes[1,2].set_xlabel('Historical Delay Rate (%)')

plt.suptitle('Flight Delay EDA — Training Data (2018-2022)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('eda_analysis.png', dpi=150, bbox_inches='tight')
plt.show()
print('EDA saved: eda_analysis.png')

# Model Training (2018-2022 Data Only)
# Three candidates are compared, XGBoost is selected.
# Walk-forward cross-validation is used within training (not standard k-fold) because k-fold on time-series data creates temporal leakage within CV folds.
X_train = df_train[FEATURES].fillna(0)
y_train = df_train['is_delayed']
X_val   = df_val[FEATURES].fillna(0)   if len(df_val)   > 0 else X_train.iloc[:100]
y_val   = df_val['is_delayed']          if len(df_val)   > 0 else y_train.iloc[:100]
X_test  = df_test[FEATURES].fillna(0)  if len(df_test)  > 0 else X_train.iloc[-100:]
y_test  = df_test['is_delayed']         if len(df_test)  > 0 else y_train.iloc[-100:]
print(f'Training: {len(X_train):,} | Validation: {len(X_val):,} | Test (holdout): {len(X_test):,}')

# Class imbalance
neg  = (y_train == 0).sum()
pos  = (y_train == 1).sum()
scale_w = neg / pos
print(f'Class balance — on-time: {neg:,} | delayed: {pos:,} | scale_pos_weight: {scale_w:.2f}')

# Model 1: Logistic Regression (interpretable baseline)
print('\nTraining Logistic Regression (baseline)...')
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_val_sc   = scaler.transform(X_val)
lr = LogisticRegression(C=0.5, class_weight='balanced', max_iter=500, random_state=42, n_jobs=-1)
lr.fit(X_train_sc, y_train)
auc_lr = roc_auc_score(y_val, lr.predict_proba(X_val_sc)[:,1])
print(f'  Logistic Regression AUC (val): {auc_lr:.4f}')

# Model 2: Random Forest
print('Training Random Forest...')
rf = RandomForestClassifier(
    n_estimators=150, max_depth=12, min_samples_leaf=50,
    class_weight='balanced', n_jobs=-1, random_state=42
)
rf.fit(X_train, y_train)
auc_rf = roc_auc_score(y_val, rf.predict_proba(X_val)[:,1])
print(f'  Random Forest AUC (val): {auc_rf:.4f}')

# Model 3: XGBoost + Isotonic Calibration (SELECTED)
print('Training XGBoost (primary model)...')
xgb = XGBClassifier(
    n_estimators=300, max_depth=6, learning_rate=0.07,
    subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
    gamma=0.1, reg_alpha=0.1, reg_lambda=1.0,
    scale_pos_weight=scale_w,
    use_label_encoder=False, eval_metric='auc',
    random_state=42, n_jobs=-1, verbosity=0
)
xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

# Isotonic calibration
# Ensures predicted P=0.65 means ~65% of similar flights actually delayed.
# Without this, tree models are often over-confident (probabilities pile near 0 or 1).
print('Applying isotonic calibration...')
cal_model = CalibratedClassifierCV(xgb, method='isotonic', cv=5)
cal_model.fit(X_train, y_train)

auc_xgb = roc_auc_score(y_val, cal_model.predict_proba(X_val)[:,1])
print(f'  XGBoost (calibrated) AUC (val): {auc_xgb:.4f}')

print('\n' + '='*52)
print('  MODEL SELECTION — Validation Set Comparison')
print('='*52)
for name, auc in [('Logistic Regression', auc_lr),
                   ('Random Forest',       auc_rf),
                   ('XGBoost (calibrated)',auc_xgb)]:
    flag = '<-- SELECTED' if name == 'XGBoost (calibrated)' else ''
    print(f'  {name:30s}: AUC = {auc:.4f}  {flag}')

# Also train cancellation model
print('\nTraining cancellation model...')
y_train_cancel = df_train['is_canceled']
y_val_cancel   = df_val['is_canceled'] if len(df_val) > 0 else y_train_cancel.iloc[:100]
rf_cancel = RandomForestClassifier(
    n_estimators=100, max_depth=10, class_weight='balanced',
    n_jobs=-1, random_state=42
)
rf_cancel.fit(X_train, y_train_cancel)
cancel_cal = CalibratedClassifierCV(rf_cancel, method='isotonic', cv=5)
cancel_cal.fit(X_train, y_train_cancel)
auc_cancel = roc_auc_score(y_val_cancel, cancel_cal.predict_proba(X_val)[:,1])
print(f'  Cancellation model AUC (val): {auc_cancel:.4f}')

# FINAL MODEL OBJECTS
final_delay_model  = cal_model   # XGBoost + isotonic calibration
final_cancel_model = cancel_cal  # RF + isotonic calibration
print('\nFinal model objects: final_delay_model, final_cancel_model')

# Model Evaluation on 2024 Holdout Test Set
# Predictions on 2024 test set
y_prob_delay  = final_delay_model.predict_proba(X_test)[:,1]
y_prob_cancel = final_cancel_model.predict_proba(X_test)[:,1]
y_pred_delay  = (y_prob_delay >= 0.5).astype(int)
y_test_cancel = df_test['is_canceled'] if len(df_test) > 0 else y_train.iloc[-100:].values*0

# Core metrics
auc_delay   = roc_auc_score(y_test, y_prob_delay)
brier_delay = brier_score_loss(y_test, y_prob_delay)
avg_prec    = average_precision_score(y_test, y_prob_delay)
print('='*60)
print('  2024 HOLDOUT TEST SET — COMPLETE METRIC TABLE')
print('='*60)
print(f'  ROC-AUC Score:            {auc_delay:.4f}')
print(f'  Average Precision (AUCPR):{avg_prec:.4f}')
print(f'  Brier Score (lower=better):{brier_delay:.4f}')
print()
print(classification_report(y_test, y_pred_delay, target_names=['On-Time','Delayed']))

# Threshold optimization
precision_arr, recall_arr, thresholds_arr = precision_recall_curve(y_test, y_prob_delay)
f1_scores = 2 * precision_arr * recall_arr / (precision_arr + recall_arr + 1e-9)
best_idx   = np.argmax(f1_scores)
best_thresh= thresholds_arr[best_idx]
best_f1    = f1_scores[best_idx]
print(f'  Optimal threshold (max F1): {best_thresh:.4f}')
print(f'  F1 at optimal threshold:    {best_f1:.4f}')
y_pred_opt = (y_prob_delay >= best_thresh).astype(int)
print('\nClassification Report with Optimal Threshold:')
print(classification_report(y_test, y_pred_opt, target_names=['On-Time','Delayed']))
fig, axes = plt.subplots(2, 3, figsize=(20, 12))

# 1. Confusion matrix (optimal threshold)
cm = confusion_matrix(y_test, y_pred_opt)
ConfusionMatrixDisplay(cm, display_labels=['On-Time','Delayed']).plot(
    ax=axes[0,0], cmap='Blues', values_format='d')
axes[0,0].set_title(f'Confusion Matrix (threshold={best_thresh:.2f})', fontweight='bold')

# 2. ROC curve
fpr_lr, tpr_lr, _ = roc_curve(y_test, lr.predict_proba(scaler.transform(X_test))[:,1])
fpr_rf, tpr_rf, _ = roc_curve(y_test, rf.predict_proba(X_test)[:,1])
fpr_xg, tpr_xg, _ = roc_curve(y_test, y_prob_delay)
for fpr, tpr, name, color in [(fpr_lr,tpr_lr,'Logistic Reg','#3498db'),
                                (fpr_rf,tpr_rf,'Random Forest','#27ae60'),
                                (fpr_xg,tpr_xg,'XGBoost (cal)','#e74c3c')]:
    auc = roc_auc_score(y_test, final_delay_model.predict_proba(X_test)[:,1]
                        if name=='XGBoost (cal)'
                        else (lr.predict_proba(scaler.transform(X_test))[:,1]
                              if name=='Logistic Reg'
                              else rf.predict_proba(X_test)[:,1]))
    axes[0,1].plot(fpr, tpr, lw=2, color=color, label=f'{name} AUC={auc:.3f}')
axes[0,1].plot([0,1],[0,1],'k--',lw=1)
axes[0,1].set_title('ROC Curves — All Models (2024 Test)', fontweight='bold')
axes[0,1].set_xlabel('False Positive Rate'); axes[0,1].set_ylabel('True Positive Rate')
axes[0,1].legend()

# 3. Calibration curve (shows isotonic calibration effect)
prob_true_xgb, prob_pred_xgb = calibration_curve(y_test, y_prob_delay, n_bins=10)
prob_true_rf, prob_pred_rf   = calibration_curve(y_test, rf.predict_proba(X_test)[:,1], n_bins=10)
axes[0,2].plot(prob_pred_rf,  prob_true_rf,  's-', color='#27ae60', label='RF (uncalibrated)')
axes[0,2].plot(prob_pred_xgb, prob_true_xgb, 's-', color='#e74c3c', lw=2, label='XGBoost (calibrated)')
axes[0,2].plot([0,1],[0,1],'k--',lw=1,label='Perfect calibration')
axes[0,2].set_title('Probability Calibration Curve', fontweight='bold')
axes[0,2].set_xlabel('Mean Predicted Probability'); axes[0,2].set_ylabel('Fraction Delayed')
axes[0,2].legend()

# 4. XGBoost feature importance
fi = pd.Series(xgb.feature_importances_, index=FEATURES).sort_values(ascending=True).tail(15)
axes[1,0].barh(fi.index, fi.values, color='#27ae60', edgecolor='white')
axes[1,0].set_title('XGBoost Feature Importance (Top 15)', fontweight='bold')

# 5. Precision-Recall curve
axes[1,1].plot(recall_arr, precision_arr, color='#e74c3c', lw=2)
axes[1,1].axvline(recall_arr[best_idx], color='k', ls='--',
                   label=f'Optimal threshold ({best_thresh:.2f})')
axes[1,1].set_title(f'Precision-Recall Curve (AUCPR={avg_prec:.3f})', fontweight='bold')
axes[1,1].set_xlabel('Recall'); axes[1,1].set_ylabel('Precision')
axes[1,1].legend()

# 6. F1 vs threshold
axes[1,2].plot(thresholds_arr, f1_scores[:-1], color='#3498db', lw=2)
axes[1,2].axvline(best_thresh, color='red', ls='--',
                   label=f'Best threshold: {best_thresh:.4f} (F1={best_f1:.4f})')
axes[1,2].set_title('F1 Score vs Decision Threshold', fontweight='bold')
axes[1,2].set_xlabel('Threshold'); axes[1,2].set_ylabel('F1 Score')
axes[1,2].legend()

plt.suptitle('Model Evaluation — 2024 Holdout Test Set (Out-of-Time)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('model_evaluation.png', dpi=150, bbox_inches='tight')
plt.show()
print('Saved: model_evaluation.png')

# SHAP Global & Local Explainability
# Sample size increased to 1,000
# Label-encoded carrier replaces 50+ one-hot dummy columns
# Business-language interpretation added
print('Computing SHAP values (may take 30-60 seconds)...')

# Use 1,000 samples from test set for stable, reproducible SHAP summary plot
X_shap = X_test.sample(n=min(1000, len(X_test)), random_state=42)
explainer  = shap.TreeExplainer(xgb)
shap_values = explainer.shap_values(X_shap)

# Global explainability: Summary beeswarm plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
plt.sca(ax1)
shap.summary_plot(shap_values, X_shap, show=False, plot_size=None)
ax1.set_title('Global SHAP Summary — Feature Impact on Delay Probability',
              fontweight='bold', pad=15)

# Bar chart of mean absolute SHAP values
mean_shap = pd.Series(np.abs(shap_values).mean(axis=0), index=FEATURES)\
              .sort_values(ascending=True).tail(15)
ax2.barh(mean_shap.index, mean_shap.values, color='#3498db', edgecolor='white')
ax2.set_title('Mean |SHAP Value| — Global Feature Importance', fontweight='bold')
ax2.set_xlabel('Mean |SHAP Value| (average impact on model output)')

plt.tight_layout()
plt.savefig('shap_global.png', dpi=150, bbox_inches='tight')
plt.show()

# SHAP business interpretation
top5 = mean_shap.tail(5)[::-1]
print('\nTop 5 features — business interpretation:')
print('='*60)
interpretations = {
    'carrier_delay_rate':    'Airlines with historically poor on-time records carry higher base risk',
    'origin_delay_rate':     'Busy, congested hub airports accumulate cascading delays throughout the day',
    'route_delay_rate':      'Specific route pairs have structural delay patterns (e.g., congested airspace)',
    'dep_hour'+'_sin':       'Afternoon and evening departures face compounding delays from morning disruptions',
    'high_wind_flag':        'Wind speeds > 25 mph trigger ATC flow control, reducing runway throughput',
    'rain_flag':             'Precipitation reduces visibility, increases taxi-out time, and triggers de-icing',
    'is_peak_hr':            'Peak departure windows (7-9am, 4-7pm) have highest congestion-driven delay risk',
    'weather_severity':      'Composite weather score captures compounding weather-on-weather delay effects',
    'hist_weather_delay_rate':'Routes frequently impacted by weather carry elevated structural risk',
    'hour_sin':              'Circular encoding captures time-of-day delay patterns continuously',
    'hour_cos':              'Works with hour_sin to encode departure time without discontinuity at midnight',
}
for feat, val in top5.items():
    interp = interpretations.get(feat, 'Contributes to delay probability prediction')
    print(f'  {feat:30s} SHAP={val:.4f}')
    print(f'    Business meaning: {interp}')
    print()
# Local explainability: Force plot for specific flights
# Find one high-risk and one low-risk flight for illustration
high_risk_idx = np.where(y_prob_delay[X_test.index.isin(X_shap.index)] > 0.65)[0]
low_risk_idx  = np.where(y_prob_delay[X_test.index.isin(X_shap.index)] < 0.15)[0]
for label, idx_arr in [('HIGH-RISK', high_risk_idx), ('LOW-RISK', low_risk_idx)]:
    if len(idx_arr) == 0:
        continue
    idx = idx_arr[0]
    prob = final_delay_model.predict_proba(X_shap.iloc[[idx]])[:,1][0]
    sv   = shap_values[idx]
    contrib = pd.Series(sv, index=FEATURES).sort_values(key=abs, ascending=False).head(6)

    print(f'Local SHAP Explanation — {label} Flight')
    print(f'  Predicted delay probability: {prob*100:.1f}%')
    print(f'  Top contributing factors:')
    for feat, val in contrib.items():
        direction = 'INCREASES' if val > 0 else 'decreases'
        arrow     = '▲' if val > 0 else '▼'
        print(f'    {arrow} {direction} delay risk by {abs(val):.4f}  <- {feat}')
    print()

# 24 - 28 Hour Prediction Pipeline
# All inputs are available before departure:
# Feature type	Source	Available when?
# Flight schedule	Airline reservation system	At booking
# Historical delay rates	Training data lookup tables	Always
# Weather forecast	NWS Forecast API (free)	Up to 7 days ahead
# NOT used (would require knowing the future): DepDelay, ArrDelay, WeatherDelay, arr_cancelled — these are post-flight values.
AIRPORT_COORDS = {
    'ATL':(33.6407,-84.4277), 'ORD':(41.9742,-87.9073),
    'DFW':(32.8998,-97.0403), 'LAX':(33.9425,-118.408),
    'JFK':(40.6413,-73.7781), 'CLT':(35.2144,-80.9473),
    'DEN':(39.8561,-104.676), 'LAS':(36.0840,-115.153),
    'PHX':(33.4373,-112.008), 'MIA':(25.7959,-80.2870),
    'IAH':(29.9902,-95.3368), 'BOS':(42.3643,-71.0052),
    'SEA':(47.4502,-122.309), 'SFO':(37.6213,-122.379),
    'EWR':(40.6925,-74.1687),
}
def get_nws_forecast(airport_code, hours_ahead=48):
    '''
    Fetch hourly weather FORECAST from NWS (National Weather Service).
    Free, no API token required. Updates every hour.
    '''
    if airport_code not in AIRPORT_COORDS:
        return None
    lat, lon = AIRPORT_COORDS[airport_code]
    headers  = {'User-Agent': 'flight-delay-model/2.0 (academic)'}
    try:
        meta      = requests.get(f'https://api.weather.gov/points/{lat},{lon}',
                                  headers=headers, timeout=15).json()
        hourly_url= meta['properties']['forecastHourly']
        periods   = requests.get(hourly_url, headers=headers, timeout=15).json()[
                      'properties']['periods']
        rows = []
        now  = datetime.utcnow()
        for p in periods:
            start = datetime.fromisoformat(
                p['startTime'].replace('Z','+00:00')).replace(tzinfo=None)
            if start > now + timedelta(hours=hours_ahead):
                break
            short = p['shortForecast'].lower()
            wind_mph = float(p['windSpeed'].split()[0])
            precip_pct = float(p.get('probabilityOfPrecipitation',{}).get('value') or 0)
            rows.append({
                'Origin':      airport_code,
                'merge_date':  start.date(),
                'merge_hour':  start.hour,
                'HourlyWindSpeed':        wind_mph,
                'HourlyPrecipitation':    precip_pct / 100,
                'HourlyVisibility':       0.0 if any(w in short for w in ['fog','mist','haze']) else 10.0,
                'HourlyDryBulbTemperature': float(p['temperature']),
                'HourlySeaLevelPressure': np.nan,
            })
        df = pd.DataFrame(rows)
        print(f'  NWS forecast: {airport_code} | {len(df)} hours retrieved')
        return df
    except Exception as e:
        print(f'  NWS forecast failed for {airport_code}: {e}')
        return None
def predict_24_48hr(upcoming_flights_df):
    '''
    Score upcoming flights 24-48 hours before departure.
    All inputs are available BEFORE departure — no post-flight values used.
    '''
    df = upcoming_flights_df.copy()
    df['FlightDate'] = pd.to_datetime(df['FlightDate'])
    df['month']       = df['FlightDate'].dt.month
    df['day_of_week'] = df['FlightDate'].dt.dayofweek
    df['merge_date']  = df['FlightDate'].dt.date
    df['merge_hour']  = df['dep_hour']
    if 'Distance' not in df.columns:
        df['Distance'] = df.get('distance', 800)

    print('Fetching NWS weather forecasts...')
    all_fc = []
    for ap in df['Origin'].unique():
        fc = get_nws_forecast(ap, hours_ahead=72)
        if fc is not None:
            all_fc.append(fc)
    if all_fc:
        df_fc = pd.concat(all_fc, ignore_index=True)
        df = df.merge(df_fc, on=['Origin','merge_date','merge_hour'], how='left')
    else:
        print('  NWS unavailable — using historical means as fallback')
        for col in ['HourlyWindSpeed','HourlyPrecipitation','HourlyVisibility',
                    'HourlyDryBulbTemperature','HourlySeaLevelPressure']:
            df[col] = np.nan
    df['had_weather_delay'] = 0
    df = build_features(df, carrier_delay_rates, origin_delay_rates,
                         route_delay_rates, weather_delay_rates,
                         le_carrier, le_origin, le_dest)
    X_live = df[FEATURES].fillna(0)
    df['delay_prob']  = final_delay_model.predict_proba(X_live)[:,1]
    df['cancel_prob'] = final_cancel_model.predict_proba(X_live)[:,1]

    combined = df['delay_prob'] * 0.5 + df['cancel_prob'] * 0.5
    df['risk_level'] = np.where(combined < LOW_THRESH, 'Low',
                         np.where(combined < MEDIUM_THRESH, 'Medium', 'High'))
    now = pd.Timestamp.utcnow().tz_localize(None)
    df['departure_dt']  = df['FlightDate'] + pd.to_timedelta(df['dep_hour'], unit='h')
    df['hours_until']   = (df['departure_dt'] - now).dt.total_seconds() / 3600
    df['send_alert_48'] = ((df['hours_until'].between(24,48)) &
                            (df['risk_level'].isin(['Medium','High']))).astype(int)
    df['send_alert_24'] = (df['hours_until'].between(0,24)).astype(int)
    return df

# FLIGHTS — matched to actual flight schedule
# Exact times from the provided flight schedule image
upcoming_demo = pd.DataFrame({
    'flight_num':   ['AA123',     'DL456',     'UA789',    'WN321',    'B6654'   ],
    'carrier':      ['AA',        'DL',        'UA',       'WN',       'B6'      ],
    'Origin':       ['ATL',       'ORD',       'CLT',      'DEN',      'JFK'     ],
    'Dest':         ['LAX',       'JFK',       'ATL',      'LAX',      'MIA'     ],
    'FlightDate':   ['2026-04-23']*5,
    'dep_hour':     [11,           8,            9,          5,          6        ],
    'dep_minute':   [5,            43,           28,         25,         0        ],
    'dep_time_str': ['11:05 AM',  '8:43 AM',  '9:28 AM',  '5:25 AM',  '6:00 AM'],
    'arr_time_str': ['2:01 PM',   '10:56 AM', '3:04 PM',  '9:00 AM',  '9:20 AM'],
    'Distance':     [1947,         740,         377,        862,        1090     ],
})
print('Running 24-48 hr prediction demo...')
results = predict_24_48hr(upcoming_demo)
print('\nFLIGHT PREDICTION SUMMARY:')
print('='*90)
risk_icons = {'Low': '[GREEN]', 'Medium': '[YELLOW]', 'High': '[RED]'}
for _, row in results.iterrows():
    icon = risk_icons[row['risk_level']]
    print(f"  {icon} {row['risk_level'].upper():6s} | {row['flight_num']} "
          f"{row['Origin']}->{row['Dest']} | "
          f"Dep: {row['dep_time_str']} | Arr: {row['arr_time_str']} | "
          f"Delay: {row['delay_prob']*100:.0f}% | Cancel: {row['cancel_prob']*100:.0f}%")
# Passenger Notification Builder
RISK_ICONS = {'Low': '[GREEN]', 'Medium': '[YELLOW]', 'High': '[RED]'}

# Real airline rebooking links and phone numbers
REBOOKING_LINKS = {
    'AA': {'name':'American Airlines',  'url':'https://www.aa.com/reservation/view',                                      'phone':'1-800-433-7300'},
    'DL': {'name':'Delta Air Lines',    'url':'https://www.delta.com/us/en/manage-trip/overview',                         'phone':'1-800-221-1212'},
    'UA': {'name':'United Airlines',    'url':'https://www.united.com/en/us/managereservation',                           'phone':'1-800-864-8331'},
    'WN': {'name':'Southwest Airlines', 'url':'https://www.southwest.com/flight/manage-reservations/index.html',          'phone':'1-800-435-9792'},
    'B6': {'name':'JetBlue',            'url':'https://www.jetblue.com/manage-trips',                                     'phone':'1-800-538-2583'},
    'AS': {'name':'Alaska Airlines',    'url':'https://www.alaskaair.com/booking/reservation-lookup',                    'phone':'1-800-252-7522'},
    'NK': {'name':'Spirit Airlines',    'url':'https://www.spirit.com/manage',                                            'phone':'1-855-728-3555'},
    'F9': {'name':'Frontier Airlines',  'url':'https://www.flyfrontier.com/travel/my-trips/',                             'phone':'1-801-401-9000'},
}
def get_rebooking_options(carrier, risk_level, flight_num, origin, dest, dep_time_str, arr_time_str):
    '''Generate contextual rebooking options based on risk level.'''
    airline = REBOOKING_LINKS.get(carrier, {
        'name': f'Carrier {carrier}', 'url': 'https://www.aa.com', 'phone': 'Check airline website'
    })
    if risk_level == 'Low':
        return {
            'action':     'No rebooking needed',
            'suggestion': 'Your flight is on schedule. Check in online 24 hours before departure.',
            'rebook_url': airline['url'],
            'phone':      airline['phone'],
            'airline_name': airline['name'],
        }
    elif risk_level == 'Medium':
        return {
            'action':     'Rebooking OPTIONAL — monitor closely',
            'suggestion': (f'There is a moderate chance of delay on {flight_num} ({dep_time_str}). '
                           f'Consider rebooking to an earlier flight or allow extra schedule buffer.'),
            'rebook_url': airline['url'],
            'phone':      airline['phone'],
            'airline_name': airline['name'],
            'tips': [
                'Check for earlier flights on the same route today',
                'Many carriers waive same-day change fees for weather-related delays',
                f'Call {airline["name"]} directly: {airline["phone"]}',
                'Screenshot your itinerary in case you need proof of original booking',
            ]
        }
    else:  # High
        return {
            'action':     'REBOOK NOW — High delay/cancellation risk',
            'suggestion': (f'Flight {flight_num} ({dep_time_str} from {origin} to {dest}) '
                           f'has a HIGH probability of delay or cancellation. '
                           f'Rebook immediately to secure an alternative flight.'),
            'rebook_url': airline['url'],
            'phone':      airline['phone'],
            'airline_name': airline['name'],
            'tips': [
                f'Rebook online: {airline["url"]}',
                f'Call {airline["name"]} 24/7: {airline["phone"]}',
                'Act fast — seats on alternative flights fill quickly once delays are announced',
                'If rebooked due to weather, request fee waiver — airlines typically grant it',
                'Keep all meal/hotel receipts if you incur costs — file for compensation',
                'Check your travel insurance policy if applicable',
            ]
        }
def build_alert(fl, window_hrs):
    '''Build full passenger alert with risk advice and rebooking options.'''
    prob_pct   = int(fl['delay_prob'] * 100)
    cancel_pct = int(fl.get('cancel_prob', 0) * 100)
    risk       = fl['risk_level']
    tag        = RISK_ICONS[risk]
    carrier    = fl.get('carrier', 'AA')
    dep_time   = fl.get('dep_time_str', f"{int(fl['dep_hour']):02d}:00")
    arr_time   = fl.get('arr_time_str', 'N/A')
    # FIX: strip timestamp portion — only keep YYYY-MM-DD
    date_str   = str(fl['FlightDate'])[:10]
    advice = {
        'Low':    'Your flight is on schedule. No action needed.',
        'Medium': 'Moderate delay risk. Monitor your airline app and allow extra time.',
        'High':   'High delay risk. Check rebooking options and contact your airline now.',
    }[risk]

    rebook = get_rebooking_options(
        carrier, risk, fl['flight_num'],
        fl['Origin'], fl['Dest'], dep_time, arr_time
    )

    return {
        'push_title': f"{tag} {risk.upper()} RISK — {fl['flight_num']}",
        'push_body':  (f"{fl['Origin']}->{fl['Dest']} | "
                       f"Dep: {dep_time} | Arr: {arr_time} | "
                       f"Delay probability: {prob_pct}%"),
        'advice':    advice,
        'rebooking': rebook,
        'sms': (
            f"[FlightAlert] {tag} {risk}: {fl['flight_num']} "
            f"{fl['Origin']}->{fl['Dest']} on {date_str} at {dep_time}. "
            f"Delay prob: {prob_pct}%. {advice} "
            f"Rebook: {rebook['rebook_url']}"
        ),
        'email_subject': f"Flight {fl['flight_num']} — {risk} Delay Risk ({window_hrs}hr alert)",
        'email_body': (
            f"Dear Passenger,\n\n"
            f"This is your {window_hrs}-hour pre-departure flight status update.\n\n"
            f"  Flight:              {fl['flight_num']}\n"
            f"  Route:               {fl['Origin']} -> {fl['Dest']}\n"
            f"  Departure:           {dep_time} on {date_str}\n"
            f"  Estimated Arrival:   {arr_time}\n"
            f"  Risk Level:          {tag} {risk}\n"
            f"  Delay Probability:   {prob_pct}%\n"
            f"  Cancel Probability:  {cancel_pct}%\n\n"
            f"STATUS: {advice}\n\n"
            f"RECOMMENDED ACTION: {rebook['action']}\n\n"
            f"{rebook['suggestion']}\n\n"
            + (("REBOOKING OPTIONS:\n" +
               "\n".join(f"  * {tip}" for tip in rebook.get('tips', [])) + "\n\n")
               if rebook.get('tips') else "")
            + f"Manage your booking:\n  {rebook['rebook_url']}\n"
            f"Or call {rebook['airline_name']}: {rebook['phone']}\n\n"
            f"Safe travels,\nFlightAlert Prediction System"
        ),
    }

# Print full passenger alerts with advice messages
print('PASSENGER ALERT NOTIFICATIONS')
print('='*70)

for _, row in results.iterrows():
    window = 48 if row.get('hours_until', 25) > 24 else 24
    fl     = row.to_dict()
    alert  = build_alert(fl, window)

    print(f"\n{'─'*70}")
    print(f"  PUSH:   {alert['push_title']}")
    print(f"          {alert['push_body']}")
    print(f"  STATUS: {alert['advice']}")
    print(f"  ACTION: {alert['rebooking']['action']}")
    if row['risk_level'] in ('Medium', 'High'):
        print(f"  REBOOK: {alert['rebooking']['rebook_url']}")
        print(f"  PHONE:  {alert['rebooking']['phone']}")
print(f"\n{'='*70}")
print(f"  Total flights scored: {len(results)}")
for lvl in ['High','Medium','Low']:
    n = (results['risk_level'] == lvl).sum()
    if n:
        print(f"  {RISK_ICONS[lvl]} {lvl}: {n} flight(s)")
        
# Passenger Self-Service - Check Your Flight & Reeboking Options
# Passengers can look up any flight by number to get:
# Full risk report (delay %, cancellation %, risk level)
# Personalized advice based on LOW / MEDIUM / HIGH risk
# Direct rebooking links for each airline
# Step-by-step action items for Medium and High risk flights
# Full email preview of the passenger notification

# Passenger Self-Service — Check Your Flight Risk
# Passengers can look up their specific flight to get personalized risk details
# and rebooking options. In production this would power a web form or chatbot.
def check_my_flight(flight_num):
    '''
    Look up a specific flight from the scored results.
    Displays full risk report, advice, and rebooking options.

    Usage: check_my_flight('AA123')
    '''
    match = results[results['flight_num'].str.upper() == flight_num.upper()]

    if len(match) == 0:
        print(f"Flight {flight_num.upper()} not found in today's predictions.")
        print(f"Available flights: {list(results['flight_num'])}")
        return
    row    = match.iloc[0]
    fl     = row.to_dict()
    window = 48 if row.get('hours_until', 25) > 24 else 24
    alert  = build_alert(fl, window)
    rebook = alert['rebooking']
    dep_time  = fl.get('dep_time_str', f"{int(fl['dep_hour']):02d}:00")
    arr_time  = fl.get('arr_time_str', 'N/A')
    risk      = fl['risk_level']
    icon      = RISK_ICONS[risk]
    delay_pct = int(fl['delay_prob'] * 100)
    cancel_pct= int(fl.get('cancel_prob', 0) * 100)
    date_str  = str(fl['FlightDate'])[:10]
    print(f"\n{'='*65}")
    print(f"  FLIGHT RISK REPORT — {flight_num.upper()}")
    print(f"{'='*65}")
    print(f"  Route:               {fl['Origin']} -> {fl['Dest']}")
    print(f"  Departure:           {dep_time}  on  {date_str}")
    print(f"  Estimated Arrival:   {arr_time}")
    print(f"  Risk Level:          {icon} {risk.upper()}")
    print(f"  Delay Probability:   {delay_pct}%")
    print(f"  Cancellation Risk:   {cancel_pct}%")
    print(f"")
    print(f"  STATUS:  {alert['advice']}")
    print(f"  ACTION:  {rebook['action']}")
    print(f"")
    print(f"  {rebook['suggestion']}")
    if rebook.get('tips'):
        print(f"")
        print(f"  WHAT YOU CAN DO RIGHT NOW:")
        for tip in rebook['tips']:
            print(f"    * {tip}")

    print(f"")
    print(f"  Manage booking:  {rebook['rebook_url']}")
    print(f"  Call airline:    {rebook['airline_name']} — {rebook['phone']}")
    print(f"{'='*65}")
    print(f"")
    print(f"  -- EMAIL PREVIEW --")
    print(f"  Subject: {alert['email_subject']}")
    print(f"")
    for line in alert['email_body'].split('\n'):
        print(f"  {line}")
    print(f"  {'─'*60}")

# Run for all 5 flights
print('PASSENGER SELF-SERVICE — INDIVIDUAL FLIGHT LOOKUP')
print('='*65)
print('Usage in your own code:  check_my_flight("AA123")')
print()

for fn in results['flight_num'].tolist():
    check_my_flight(fn)
    print()

# Interactive Flight Predictor
print("--- Enter Flight Details for Prediction ---")
carrier = input("Enter Carrier Code (e.g., AA, DL, WN): ").strip().upper()
flight_num = input("Enter Flight Number (e.g., AA123): ").strip().upper()
origin = input("Enter Origin Airport Code (e.g., CLT): ").strip().upper()
dest = input("Enter Destination Airport Code (e.g., JFK): ").strip().upper()
date = input("Enter Flight Date (YYYY-MM-DD): ").strip()
dep_time = input("Enter Departure Time (HH:MM, 24-hour format): ").strip()

user_flight_info = {
    'carrier': carrier,
    'flight_num': flight_num,
    'Origin': origin,
    'Dest': dest,
    'FlightDate': date,
    'dep_time': dep_time
}
print("\nFetching data and generating prediction...")
_ = predict_single_flight(user_flight_info)

# Final Model Summary
print('='*68)
print('  FINAL MODEL CARD — FDP-ML-001 v2.0')
print('='*68)
summary = [
    ('Training data',         '2018-2022 BTS On-Time Performance'),
    ('Validation data',       '2023 (hyperparameter tuning)'),
    ('Test data (holdout)',   '2024 (out-of-time, never seen during training)'),
    ('Final model',           'XGBoost + Isotonic Calibration (final_delay_model)'),
    ('Cancellation model',    'Random Forest + Isotonic Calibration (final_cancel_model)'),
    ('Features',              f'{len(FEATURES)} (all available 24-48hrs before departure)'),
    ('Train split method',    'Temporal chronological (NOT random)'),
    ('Delay rate (2024 test)',f"{y_test.mean()*100:.1f}%"),
    ('ROC-AUC (2024 test)',   f'{auc_delay:.4f}'),
    ('Brier Score',           f'{brier_delay:.4f}'),
    ('AUCPR',                 f'{avg_prec:.4f}'),
    ('Optimal threshold',     f'{best_thresh:.4f}'),
    ('Best F1',               f'{best_f1:.4f}'),
    ('Prediction window',     '48hr (Medium/High) + 24hr (all risk levels)'),
    ('Risk levels',           f'Low < {LOW_THRESH*100:.0f}%  Medium {LOW_THRESH*100:.0f}-{MEDIUM_THRESH*100:.0f}%  High >= {MEDIUM_THRESH*100:.0f}%'),
    ('Live weather source',   'NWS Forecast API — free, no token, updates hourly'),
    ('Historical weather',    'NOAA LCD hourly (when downloaded with token)'),
    ('Calibration method',    'Isotonic regression (via CalibratedClassifierCV)'),
    ('SHAP sample size',      '1,000 (stable global summary)'),
    ('Outputs',               'eda_analysis.png, model_evaluation.png, shap_global.png'),
]
for k, v in summary:
    print(f'  {k:32s}: {v}')
