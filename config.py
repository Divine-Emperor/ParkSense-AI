"""
ParkSense AI - Global Configuration
"""

# Random seed for reproducibility
SEED = 42

# General configurations
CFG = dict(
    seed=SEED,
    n_classes=5,
    batch_size=2048, # Reduced from 4096 to prevent memory crashes
    epochs_bnn=20,
    epochs_bns=15,
    lr=1e-3,
    kl_base=None,
    lambda_bns=0.05,
    mc_samples=20, # Reduced from 30/50 to prevent memory crashes
    ece_bins=15,
)

# Synthetic data generation limits
N_SYNTHETIC = 500_000         # Reduced from 1M-2M to prevent OOM
TARGET_PER_CLASS = 100_000    # Class balancing target, reduced to match N_SYNTHETIC

# Melbourne 2017 Public Holidays (Improvement 6)
MELB_HOLIDAYS = {(1,1),(1,26),(3,13),(4,14),(4,17),(4,25),(6,12),(11,7),(12,25),(12,26)}

def is_holiday(month, day):
    return int((month, day) in MELB_HOLIDAYS)

ZONES = [
    {"id":"z01","name":"Flinders Street Station",   "lat":-37.8183,"lng":144.9671,"demand":"high"},
    {"id":"z02","name":"Federation Square",          "lat":-37.8179,"lng":144.9691,"demand":"high"},
    {"id":"z03","name":"Melbourne Central",          "lat":-37.8102,"lng":144.9628,"demand":"high"},
    {"id":"z04","name":"Queen Victoria Market",      "lat":-37.8065,"lng":144.9558,"demand":"medium"},
    {"id":"z05","name":"Crown Casino",               "lat":-37.8241,"lng":144.9551,"demand":"medium"},
    {"id":"z06","name":"Collins Street East",        "lat":-37.8154,"lng":144.9720,"demand":"medium"},
    {"id":"z07","name":"Bourke Street Mall",         "lat":-37.8142,"lng":144.9633,"demand":"high"},
    {"id":"z08","name":"Docklands",                  "lat":-37.8157,"lng":144.9480,"demand":"low"},
    {"id":"z09","name":"Carlton Gardens",            "lat":-37.8055,"lng":144.9714,"demand":"low"},
    {"id":"z10","name":"Chinatown",                  "lat":-37.8115,"lng":144.9657,"demand":"medium"},
    {"id":"z11","name":"State Library",              "lat":-37.8096,"lng":144.9642,"demand":"medium"},
    {"id":"z12","name":"Southbank Promenade",        "lat":-37.8220,"lng":144.9631,"demand":"medium"},
]

D_OFF = {"high": 1, "medium": 0, "low": -1}
CLASSES = ["Empty", "Low", "Medium", "High", "Full"]
