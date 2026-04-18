import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from config import SEED, N_SYNTHETIC, TARGET_PER_CLASS, CFG, is_holiday
import gc

import os
import pandas as pd

def make_data(N=N_SYNTHETIC, seed=SEED):
    """
    Load real processed data from CSV, and bootstrap it to match N_SYNTHETIC
    size with slight noise to continuous features to prevent overfitting.
    """
    cache_path = "/home/john/Dev-days/parksense/processed_data_2017.csv"
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"Missing {cache_path}. Run process_csv.py first.")
        
    df = pd.read_csv(cache_path)
    
    # Bootstrap to N
    df_sample = df.sample(n=N, replace=True, random_state=seed).copy()
    
    rng = np.random.default_rng(seed)
    
    # Add slight noise to continuous features
    df_sample['air_temp'] += rng.normal(0, 0.5, N)
    df_sample['dew_point'] += rng.normal(0, 0.5, N)
    df_sample['wind_spd'] = np.clip(df_sample['wind_spd'] + rng.normal(0, 0.2, N), 0, None)
    df_sample['pressure'] += rng.normal(0, 1.0, N)
    df_sample['rain_mm'] = np.clip(df_sample['rain_mm'] + rng.normal(0, 0.1, N), 0, None)
    
    hour = df_sample['hour'].values
    day = df_sample['day'].values
    month = df_sample['month'].values
    dow = df_sample['dow'].values
    weekend = df_sample['weekend'].values
    hol = df_sample['hol'].values
    
    occ_class = df_sample['occ_class'].values.astype(np.int32)
    
    air_temp = df_sample['air_temp'].values
    dew_point = df_sample['dew_point'].values
    wind_spd = df_sample['wind_spd'].values
    pressure = df_sample['pressure'].values
    rain_mm = df_sample['rain_mm'].values
    
    # Free memory
    del df_sample, df
    gc.collect()

    # Cyclic encoding
    H_sin = np.sin(2*np.pi*hour/24);   H_cos = np.cos(2*np.pi*hour/24)
    D_sin = np.sin(2*np.pi*dow/7);     D_cos = np.cos(2*np.pi*dow/7)
    M_sin = np.sin(2*np.pi*month/12);  M_cos = np.cos(2*np.pi*month/12)

    X = np.column_stack([
        H_sin, H_cos, D_sin, D_cos, M_sin, M_cos,
        day, weekend, air_temp, dew_point, wind_spd, pressure, rain_mm
    ]).astype(np.float32)

    return X, occ_class, hour, dow, month, hol

def balance_classes(X, y, hours, dows, hols, target_per_class=TARGET_PER_CLASS, seed=SEED):
    """Downsample majority classes to balance the dataset."""
    parts = [np.where(y == c)[0] for c in range(CFG["n_classes"])]
    idx = np.concatenate([
        np.random.default_rng(seed).choice(p, min(target_per_class, len(p)), replace=False)
        for p in parts
    ])
    np.random.default_rng(seed).shuffle(idx)
    
    return X[idx], y[idx], hours[idx], dows[idx], hols[idx]

def get_train_val_test_data():
    """Generates, balances, splits, and scales the dataset."""
    X, y, hours, dows, months, hols = make_data(N_SYNTHETIC)
    
    X, y, hours, dows, hols = balance_classes(X, y, hours, dows, hols, TARGET_PER_CLASS)
    
    # Split
    (X_tmp, X_test, y_tmp, y_test,
     h_tmp, h_te, d_tmp, d_te, hf_tmp, hf_te) = train_test_split(
        X, y, hours, dows, hols, test_size=0.15, stratify=y, random_state=SEED)
    (X_train, X_val, y_train, y_val,
     h_tr, h_va, d_tr, d_va, hf_tr, hf_va) = train_test_split(
        X_tmp, y_tmp, h_tmp, d_tmp, hf_tmp, test_size=0.15, stratify=y_tmp, random_state=SEED)

    # Free memory
    del X, y, hours, dows, hols, X_tmp, y_tmp, h_tmp, d_tmp, hf_tmp
    gc.collect()

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val   = scaler.transform(X_val).astype(np.float32)
    X_test  = scaler.transform(X_test).astype(np.float32)
    
    # Class weights
    raw_w = compute_class_weight("balanced", classes=np.arange(CFG["n_classes"]), y=y_train)
    class_weights = {i: float(np.sqrt(raw_w[i])) for i in range(CFG["n_classes"])}
    
    return (X_train, X_val, X_test, 
            y_train, y_val, y_test,
            h_tr, h_va, h_te,
            d_tr, d_va, d_te,
            hf_tr, hf_va, hf_te,
            scaler, class_weights)

def compute_empirical_curves(hours, dows, hols, y, high_class=2):
    """Hourly + weekly + holiday empirical curves for Extended TSC."""
    mu_hour = np.array([
        np.mean(y[hours == h] == high_class) if np.any(hours == h) else 0.0
        for h in range(24)
    ], dtype=np.float32)

    mu_week = np.zeros((7, 24), dtype=np.float32)
    for d in range(7):
        for h in range(24):
            m = (dows == d) & (hours == h)
            mu_week[d, h] = np.mean(y[m] == high_class) if m.sum() > 0 else mu_hour[h]

    hol_mask = hols > 0
    mu_hol = np.array([
        np.mean(y[hol_mask & (hours == h)] == high_class)
        if (hol_mask & (hours == h)).sum() > 0 else mu_hour[h]
        for h in range(24)
    ], dtype=np.float32)

    return mu_hour, mu_week.reshape(168), mu_hol

def make_bns_ds(X, h, d, hf, y, shuffle=False):
    """BNS datasets (append hour, dow, hol as extra cols for TSC)"""
    X_ext = np.concatenate([
        X,
        h.reshape(-1,1).astype(np.float32),
        d.reshape(-1,1).astype(np.float32),
        hf.reshape(-1,1).astype(np.float32)
    ], axis=1)
    ds = tf.data.Dataset.from_tensor_slices((X_ext, y))
    if shuffle: ds = ds.shuffle(min(100_000, len(X)), seed=SEED)
    # Removing .cache() to prevent RAM doubling
    return ds.batch(CFG["batch_size"]).prefetch(tf.data.AUTOTUNE)

def make_bnn_ds(X, y, shuffle=False):
    """Standard dataset without extended features"""
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle: ds = ds.shuffle(min(100_000, len(X)), seed=SEED)
    return ds.batch(CFG["batch_size"]).prefetch(tf.data.AUTOTUNE)
