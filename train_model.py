"""
ParkSense AI — BNS Training Script
"""
import os, json, warnings, gc
import numpy as np
import tensorflow as tf
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

# Uncomment if XGBoost is installed
# import xgboost as xgb

from config import SEED, CFG, ZONES, D_OFF, CLASSES, MELB_HOLIDAYS
from data import get_train_val_test_data, make_bnn_ds, make_bns_ds, compute_empirical_curves
from models import build_mlp, build_bnn, BNS_Model, LambdaWarmup
from evaluate import mc_predict, compute_ece, compute_tce

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
np.random.seed(SEED)
tf.random.set_seed(SEED)

def main():
    print("=" * 60)
    print("ParkSense AI — Model Training (System Optimized)")
    print("=" * 60)

    print("\n[1/4] Loading and optimizing data...")
    (X_train, X_val, X_test, 
     y_train, y_val, y_test,
     h_tr, h_va, h_te,
     d_tr, d_va, d_te,
     hf_tr, hf_va, hf_te,
     scaler, class_weights) = get_train_val_test_data()

    D = X_train.shape[1]
    C = CFG["n_classes"]
    N_TRAIN = len(X_train)
    kl_base = 1.0 / N_TRAIN
    CFG["kl_base"] = kl_base
    
    print(f"  Train:{N_TRAIN:,}  Val:{len(X_val):,}  Test:{len(X_test):,}")

    mu_hour, mu_week, mu_hol = compute_empirical_curves(h_tr, d_tr, hf_tr, y_train)

    bnn_train = make_bnn_ds(X_train, y_train, shuffle=True)
    bnn_val   = make_bnn_ds(X_val, y_val)
    
    bns_train = make_bns_ds(X_train, h_tr, d_tr, hf_tr, y_train, shuffle=True)
    bns_val   = make_bns_ds(X_val,   h_va, d_va, hf_va, y_val)

    results = {}

    def save_result(name, acc, mf1, ece, tce):
        results[name] = {
            "name": name,
            "accuracy": float(acc),
            "macro_f1": float(mf1),
            "ece": float(ece) if ece is not None else None,
            "tce": float(tce) if tce is not None else None
        }
        print(f"  [{name}] Acc: {acc:.4f} | Mac-F1: {mf1:.4f} | ECE: {ece if ece else 0:.4f} | TCE: {tce if tce else 0:.4f}")

    # ── Classical Baselines ──────────────────────────────────────────────────
    print("\n[2/4] Training Classical Baselines...")
    
    # Logistic Regression
    lr = LogisticRegression(random_state=SEED, max_iter=200, class_weight='balanced')
    lr.fit(X_train, y_train)
    p_lr = lr.predict_proba(X_test)
    save_result("LR", accuracy_score(y_test, p_lr.argmax(1)), 
                f1_score(y_test, p_lr.argmax(1), average="macro"),
                compute_ece(y_test, p_lr), compute_tce(p_lr, h_te, mu_hour)[0])
    del lr, p_lr; gc.collect()

    # Random Forest (optimized for speed/memory)
    rf = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=SEED, class_weight='balanced', n_jobs=-1)
    rf.fit(X_train, y_train)
    p_rf = rf.predict_proba(X_test)
    save_result("RF", accuracy_score(y_test, p_rf.argmax(1)), 
                f1_score(y_test, p_rf.argmax(1), average="macro"),
                compute_ece(y_test, p_rf), compute_tce(p_rf, h_te, mu_hour)[0])
    del rf, p_rf; gc.collect()

    # XGBoost (Commented out as requested)
    """
    xgb_model = xgb.XGBClassifier(n_estimators=50, max_depth=6, learning_rate=0.1, random_state=SEED, n_jobs=-1)
    xgb_model.fit(X_train, y_train)
    p_xgb = xgb_model.predict_proba(X_test)
    save_result("XGBoost", accuracy_score(y_test, p_xgb.argmax(1)), 
                f1_score(y_test, p_xgb.argmax(1), average="macro"),
                compute_ece(y_test, p_xgb), compute_tce(p_xgb, h_te, mu_hour)[0])
    del xgb_model, p_xgb; gc.collect()
    """

    # ── Neural Models ────────────────────────────────────────────────────────
    print("\n[3/4] Training Neural Models...")
    cbs = [tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True, verbose=0)]

    models_to_train = [
        ("DET", build_mlp(D, C, [128, 64]), bnn_train, bnn_val, False),
        ("BNN-Small", build_bnn(D, C, [32, 16], kl_base), bnn_train, bnn_val, True),
        ("BNN-Large", build_bnn(D, C, [128, 64], kl_base), bnn_train, bnn_val, True),
        ("BNN-LowKL", build_bnn(D, C, [128, 64], kl_base*0.1), bnn_train, bnn_val, True),
        ("BNN-HighKL", build_bnn(D, C, [128, 64], kl_base*10), bnn_train, bnn_val, True),
    ]

    for name, model, ds_tr, ds_va, is_bnn in models_to_train:
        model.compile(optimizer=tf.keras.optimizers.Adam(CFG["lr"]),
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=["accuracy"])
        model.fit(ds_tr, validation_data=ds_va, epochs=CFG["epochs_bnn"], callbacks=cbs, verbose=0)
        
        if is_bnn:
            proba, _ = mc_predict(model, X_test, T=CFG["mc_samples"])
        else:
            proba = tf.nn.softmax(model.predict(X_test, batch_size=CFG["batch_size"], verbose=0)).numpy()
            
        save_result(name, accuracy_score(y_test, proba.argmax(1)), 
                    f1_score(y_test, proba.argmax(1), average="macro"),
                    compute_ece(y_test, proba) if is_bnn else None, 
                    compute_tce(proba, h_te, mu_hour)[0])
                    
        del model, proba; tf.keras.backend.clear_session(); gc.collect()

    # BNS Model
    print("\n[4/4] Training Flagship BNS Model...")
    backbone = build_bnn(D, C, [128, 64], kl_base, "BNN_BNS")
    BNS = BNS_Model(backbone, mu_hour, mu_week, mu_hol, N_TRAIN, lambda_max=CFG["lambda_bns"])
    BNS.compile(optimizer=tf.keras.optimizers.Adam(CFG["lr"]))
    
    bns_cbs = [
        LambdaWarmup(warmup_epochs=5),
        tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=False, verbose=0),
    ]
    BNS.fit(bns_train, validation_data=bns_val, epochs=CFG["epochs_bns"], callbacks=bns_cbs, verbose=0)

    bns_proba, bns_unc = mc_predict(backbone, X_test, T=CFG["mc_samples"])
    bns_acc = accuracy_score(y_test, bns_proba.argmax(1))
    bns_mf1 = f1_score(y_test, bns_proba.argmax(1), average="macro")
    bns_ece = compute_ece(y_test, bns_proba)
    bns_tce = compute_tce(bns_proba, h_te, mu_hour)[0]
    save_result("BNS", bns_acc, bns_mf1, bns_ece, bns_tce)

    # Save full results
    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Prediction grid for ParkSense UI
    print("\nGenerating zone prediction grid for UI...")
    predictions = {}
    for day in range(7):
        for hour in range(24):
            key = f"{day}_{hour}"
            zone_preds = []
            for zone in ZONES:
                h_s = np.sin(2*np.pi*hour/24); h_c = np.cos(2*np.pi*hour/24)
                d_s = np.sin(2*np.pi*day/7);   d_c = np.cos(2*np.pi*day/7)
                m_s = np.sin(2*np.pi*6/12);    m_c = np.cos(2*np.pi*6/12)
                is_wknd = 1.0 if day >= 5 else 0.0
                feat = np.array([[h_s,h_c,d_s,d_c,m_s,m_c,15,17,10,15,1013,0,is_wknd]], dtype=np.float32)
                feat_sc = scaler.transform(feat)
                preds_mc = np.stack([tf.nn.softmax(backbone(feat_sc,training=True)).numpy()[0] for _ in range(CFG["mc_samples"])])
                mean_p = preds_mc.mean(0)
                std_p  = preds_mc.std(0)

                pred_c = int(mean_p.argmax())
                adj_c  = int(np.clip(pred_c + D_OFF[zone["demand"]], 0, 4))
                conf   = float(np.clip(1.0 - std_p.mean()*4 + D_OFF[zone["demand"]]*0.05, 0.15, 0.95))

                zone_preds.append({
                    "zone_id":    zone["id"],
                    "pred_class": adj_c,
                    "confidence": round(conf, 3),
                    "proba":      [round(float(p), 4) for p in mean_p],
                    "uncertainty":round(float(std_p.mean()), 4),
                })
            predictions[key] = zone_preds

    output = {
        "zones": ZONES,
        "predictions": predictions,
        "classes": CLASSES,
        "metrics": results["BNS"],
        "improvement6": {
            "hourly_weight": 0.4, "weekly_weight": 0.4, "holiday_weight": 0.2,
            "holidays": [list(h) for h in sorted(MELB_HOLIDAYS)]
        }
    }

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static", "predictions.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n✅ Done! Results saved to results.json and static/predictions.json")

if __name__ == "__main__":
    main()
