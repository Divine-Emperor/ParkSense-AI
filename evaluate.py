import numpy as np
import tensorflow as tf

def mc_predict(model, X, T=30):
    """Monte Carlo prediction using T forward passes."""
    # Process in batches to avoid OOM
    batch_size = 2048
    all_preds = []
    
    for _ in range(T):
        batch_preds = []
        for i in range(0, len(X), batch_size):
            batch = X[i:i+batch_size]
            logits = model(batch, training=True)
            batch_preds.append(tf.nn.softmax(logits).numpy())
        all_preds.append(np.concatenate(batch_preds, axis=0))
        
    preds = np.stack(all_preds)
    return preds.mean(0), preds.std(0)

def compute_ece(y_true, proba, n_bins=15):
    """Expected Calibration Error."""
    conf  = proba.max(axis=1)
    pred  = proba.argmax(axis=1)
    corr  = (pred == y_true).astype(float)
    bins  = np.linspace(0, 1, n_bins + 1)
    ece   = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum():
            ece += m.sum() * abs(corr[m].mean() - conf[m].mean())
    return ece / len(y_true)

def compute_tce(proba, hours, empirical_mu, high_class=2):
    """Temporal Consistency Error against empirical hourly distribution."""
    high_probs = proba[:, high_class]
    mu_hat = np.array([
        high_probs[hours == h].mean() if (hours == h).sum() > 0 else 0.0
        for h in range(24)
    ])
    tce = np.mean((mu_hat - empirical_mu)**2)
    return tce, mu_hat
