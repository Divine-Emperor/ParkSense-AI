import pandas as pd
import numpy as np
import os
import os

def parse_tmp(val):
    if pd.isna(val): return np.nan
    v = str(val).split(',')[0]
    if v.replace('+','').replace('-','').strip() == '9999': return np.nan
    return float(v) / 10.0

def parse_wnd(val):
    if pd.isna(val): return np.nan
    parts = str(val).split(',')
    if len(parts) >= 4:
        v = parts[3]
        if v.strip() == '9999': return np.nan
        return float(v) / 10.0
    return np.nan

def parse_slp(val):
    if pd.isna(val): return np.nan
    v = str(val).split(',')[0]
    if v.strip() == '99999': return np.nan
    return float(v) / 10.0

def parse_rain(val):
    if pd.isna(val): return 0.0
    parts = str(val).split(',')
    if len(parts) >= 3:
        v = parts[2].replace('+', '').replace('-', '')
        if v.strip() == '9999': return 0.0
        return float(v) / 10.0
    return 0.0

print("Processing NOAA Weather data...")
weather = pd.read_csv("/home/john/Dev-days/parksense/94866099999.csv", low_memory=False)
weather['DATE'] = pd.to_datetime(weather['DATE'])
weather['hour_start'] = weather['DATE'].dt.floor('h')

weather['air_temp'] = weather['TMP'].apply(parse_tmp)
weather['dew_point'] = weather['DEW'].apply(parse_tmp)
weather['wind_spd'] = weather['WND'].apply(parse_wnd)
weather['pressure'] = weather['SLP'].apply(parse_slp)
weather['rain_mm'] = weather['AA1'].apply(parse_rain)

# Aggregate to hourly (mean for temp/wind, max for rain)
hourly_weather = weather.groupby('hour_start').agg({
    'air_temp': 'mean',
    'dew_point': 'mean',
    'wind_spd': 'mean',
    'pressure': 'mean',
    'rain_mm': 'max'
}).ffill().bfill() # fill missing

print("Processing Parking Event data in chunks (this takes a few minutes)...")
# ArrivalTime is like 04/11/2017 07:24:35 AM
hourly_counts = pd.Series(dtype=float)

chunk_iter = pd.read_csv("/home/john/Dev-days/parksense/On-street_Car_Parking_Sensor_Data_-_2017.csv", 
                         chunksize=1_000_000, 
                         usecols=['ArrivalTime', 'Vehicle Present'])

for chunk in chunk_iter:
    # Filter for arrivals where a vehicle is present
    chunk = chunk[chunk['Vehicle Present'] == True]
    if len(chunk) == 0: continue
    
    # Extract date and hour
    chunk['hour_start'] = pd.to_datetime(chunk['ArrivalTime'], format='%m/%d/%Y %I:%M:%S %p').dt.floor('h')
    
    # Count arrivals per hour in this chunk
    counts = chunk.groupby('hour_start').size()
    
    # Accumulate
    hourly_counts = hourly_counts.add(counts, fill_value=0)

hourly_counts.name = 'arrivals'
hourly_counts.index = pd.to_datetime(hourly_counts.index)

print("Merging datasets...")
# Combine everything into an 8760-row dataframe (2017)
full_idx = pd.date_range(start='2017-01-01', end='2017-12-31 23:00:00', freq='h')
df = pd.DataFrame(index=full_idx)
df = df.join(hourly_counts).fillna(0) # 0 arrivals if no data
df = df.join(hourly_weather).ffill().bfill()

# Bin demand into 5 classes
print("Calculating occupancy classes...")
# Remove completely zero arrival hours (e.g. 3 AM) from quantile calculation if needed, 
# but let's just use regular quantiles
bins = [-1] + list(df['arrivals'].quantile([0.2, 0.4, 0.6, 0.8])) + [np.inf]
# Ensure bins are unique
bins = sorted(list(set(bins)))
if len(bins) < 6:
    # If quantiles are same (e.g. lots of 0s), linearly space the remaining space
    max_val = df['arrivals'].max()
    bins = [-1, 0, max_val*0.25, max_val*0.5, max_val*0.75, np.inf]

df['occ_class'] = pd.cut(df['arrivals'], bins=bins, labels=[0, 1, 2, 3, 4], include_lowest=True).astype(int)

# Extract temporal features
df['hour'] = df.index.hour
df['day'] = df.index.day
df['month'] = df.index.month
df['dow'] = df.index.dayofweek
df['weekend'] = (df['dow'] >= 5).astype(int)

from config import is_holiday
df['hol'] = [is_holiday(m, d) for m, d in zip(df['month'], df['day'])]

# Save to cache
df.to_csv("/home/john/Dev-days/parksense/processed_data_2017.csv", index=False)
print("✅ Saved to processed_data_2017.csv")
