"""
recompute_metrics.py
Recalcule les metriques reelles des DEUX modeles deployes (sans aucun
reentrainement) a partir des artefacts deja sauvegardes dans outputs_tp2/,
et ecrit metrics_bank.json + metrics_fashion.json.

Duree : quelques secondes (juste un preprocessing + model.predict()).

A LANCER EN LOCAL (pas sur Kaggle), depuis le dossier qui contient outputs_tp2/.
Necessite : pandas, numpy, scikit-learn, tensorflow.
"""


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Supprime les logs TensorFlow
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force l'utilisation du CPU

# Vérifiez si vous avez un GPU disponible
try:
    import tensorflow as tf
    print(f"TensorFlow version: {tf.__version__}")
    print(f"GPU disponible: {tf.config.list_physical_devices('GPU')}")
except ImportError:
    print("TensorFlow n'est pas installé correctement")

import os
import json
import pickle

import numpy as np
import pandas as pd

OUTPUT_DIR = "outputs_tp2"   # <-- adaptez le chemin si besoin


# ═════════════════════════════════════════════════════════════════
# PARTIE A : BANK TELEMARKETING — reseau Top-10 deploye
# ═════════════════════════════════════════════════════════════════
def recompute_bank():
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                                  recall_score, roc_auc_score)
    from tensorflow.keras.models import load_model

    # ── 1. Recharger le CSV original (meme fichier que dans le notebook) ──
    csv_candidates = [
        "bank-additional-full.csv",
        os.path.join(OUTPUT_DIR, "bank-additional-full.csv"),
        "data/bank-additional-full.csv",
    ]
    csv_path = next((p for p in csv_candidates if os.path.exists(p)), None)
    if csv_path is None:
        print("  ❌ bank-additional-full.csv introuvable.")
        print("     Placez-le a cote de ce script (ou dans outputs_tp2/) et relancez.")
        return None

    df = pd.read_csv(csv_path, sep=';')

    # ── 2. Reproduire EXACTEMENT le preprocessing du notebook ─────
    df_clean = df.copy()
    for col in ['job', 'education', 'contact']:
        if col in df_clean.columns:
            df_clean = df_clean[df_clean[col] != 'unknown']

    cat_enc = [c for c in df_clean.select_dtypes('object').columns if c != 'y']
    df_enc = pd.get_dummies(df_clean, columns=cat_enc, drop_first=True)

    # ── 3. Charger les artefacts sauvegardes ───────────────────────
    with open(os.path.join(OUTPUT_DIR, "telemarketing_features.pkl"), "rb") as f:
        feature_cols = pickle.load(f)
    with open(os.path.join(OUTPUT_DIR, "telemarketing_top10_features.pkl"), "rb") as f:
        top10_features = pickle.load(f)
    with open(os.path.join(OUTPUT_DIR, "telemarketing_top10_indices.pkl"), "rb") as f:
        top10_idx = pickle.load(f)
    with open(os.path.join(OUTPUT_DIR, "telemarketing_scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
    with open(os.path.join(OUTPUT_DIR, "telemarketing_label_encoder.pkl"), "rb") as f:
        le_target = pickle.load(f)

    # S'assurer que toutes les colonnes attendues par le scaler existent
    # (au cas ou get_dummies produirait des colonnes legerement differentes)
    for col in feature_cols:
        if col not in df_enc.columns:
            df_enc[col] = 0

    df_enc['y_encoded'] = le_target.transform(df_clean['y'])
    X = df_enc[feature_cols].values
    y = df_enc['y_encoded'].values

    # ── 4. Meme split que pendant l'entrainement (random_state=42) ─
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    X_test_sc = scaler.transform(X_test)
    X_test_t10 = X_test_sc[:, top10_idx]

    # ── 5. Charger le modele deploye et predire (PAS de fit/entrainement) ──
    model = load_model(os.path.join(OUTPUT_DIR, "telemarketing_optimal_top10.keras"))
    y_proba = model.predict(X_test_t10, verbose=0).ravel()
    y_pred = (y_proba >= 0.5).astype(int)

    metrics_bank = {
        "best_model": "Reseau Top-10 variables (deploye)",
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "f1_score": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "auc": round(float(roc_auc_score(y_test, y_proba)), 4),
        "n_instances_total": int(len(df)),
        "n_instances_clean": int(len(df_clean)),
        "n_features": int(len(feature_cols)),
        "taux_souscription_pct": round(float((df['y'] == 'yes').mean() * 100), 2),
        "top10_features": list(top10_features),
        "all_models": [
            {
                "model": "Reseau Top-10 variables (deploye)",
                "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
                "f1_score": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
                "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
                "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
            }
        ],
        "note": (
            "Seul le modele final deploye (reseau Top-10) a pu etre re-evalue, "
            "car les modeles intermediaires (Random Forest, Gradient Boosting, "
            "Decision Tree, etc.) n'ont pas ete sauvegardes individuellement."
        ),
    }

    with open(os.path.join(OUTPUT_DIR, "metrics_bank.json"), "w", encoding="utf-8") as f:
        json.dump(metrics_bank, f, indent=2, ensure_ascii=False)

    print("  ✅ metrics_bank.json recalcule et sauvegarde")
    print(f"     Accuracy={metrics_bank['accuracy']}  F1={metrics_bank['f1_score']}  "
          f"AUC={metrics_bank['auc']}")
    return metrics_bank


# ═════════════════════════════════════════════════════════════════
# PARTIE B : FASHION MNIST — meilleur modele sauvegarde
# ═════════════════════════════════════════════════════════════════
def recompute_fashion():
    from tensorflow.keras.datasets import fashion_mnist
    from tensorflow.keras.models import load_model
    from sklearn.metrics import accuracy_score, log_loss
    import numpy as np

    CLASS_NAMES = [
        'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
        'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
    ]

    print("  ⏳ Chargement de Fashion MNIST (dataset officiel Zalando, via le cache")
    print("     Keras — meme contenu et meme ordre train/test que le notebook,")
    print("     ~30 Mo telecharges une seule fois puis mis en cache localement)...")
    (_, _), (X_test, y_test) = fashion_mnist.load_data()
    X_test = X_test.astype("float32") / 255.0

    candidates = [
        "fashion_mnist_best.keras",
        "fashion_mnist_resnet.keras",
        "fashion_mnist_vgg_like.keras",
        "fashion_mnist_cnn_lenet.keras",
        "fashion_mnist_mlp.keras",
    ]
    all_models_metrics = []
    best = None

    for fname in candidates:
        path = os.path.join(OUTPUT_DIR, fname)
        if not os.path.exists(path):
            print(f"  ⚠️  {fname} non trouvé, ignoré")
            continue
        
        print(f"  🔍 Chargement de {fname}...")
        model = load_model(path)
        
        # 🔧 CORRECTION : Vérifier la forme d'entrée attendue par le modèle
        input_shape = model.input_shape
        
        # Si le modèle attend (28,28) ou (28,28,1) -> format 2D/image
        if len(input_shape) == 3:  # (None, 28, 28)
            X_in = X_test  # déjà en (batch, 28, 28)
            print(f"     → Format (28,28) pour {fname}")
        elif len(input_shape) == 4:  # (None, 28, 28, 1)
            X_in = X_test.reshape(-1, 28, 28, 1)
            print(f"     → Format (28,28,1) pour {fname}")
        elif len(input_shape) == 2:  # (None, 784)
            X_in = X_test.reshape(-1, 28 * 28)
            print(f"     → Format (784,) pour {fname}")
        else:
            # Fallback: essayer de deviner
            print(f"     ⚠️  Format inconnu: {input_shape}, tentative auto...")
            if input_shape[1] == 784:
                X_in = X_test.reshape(-1, 28 * 28)
            else:
                X_in = X_test.reshape(-1, 28, 28, 1)

        y_proba = model.predict(X_in, verbose=0)
        y_pred = np.argmax(y_proba, axis=1)
        acc = float(accuracy_score(y_test, y_pred))
        loss = float(log_loss(y_test, y_proba, labels=list(range(10))))

        label = fname.replace("fashion_mnist_", "").replace(".keras", "")
        all_models_metrics.append({"model": label, "accuracy": round(acc, 4), "loss": round(loss, 4)})
        print(f"  ✅ {fname:<35s} accuracy={acc:.4f}  loss={loss:.4f}")

        if fname == "fashion_mnist_best.keras":
            best = (label, acc, loss)

    if not all_models_metrics:
        print("  ❌ Aucun fichier fashion_mnist_*.keras trouve dans outputs_tp2/.")
        return None

    if best is None:
        # fallback : le meilleur parmi ceux trouves
        best_entry = max(all_models_metrics, key=lambda m: m["accuracy"])
        best = (best_entry["model"], best_entry["accuracy"], best_entry["loss"])

    metrics_fashion = {
        "best_model": best[0],
        "accuracy": round(best[1], 4),
        "loss": round(best[2], 4),
        "n_classes": 10,
        "class_names": CLASS_NAMES,
        "n_train": 60000,
        "n_test": int(len(X_test)),
        "all_models": all_models_metrics,
    }

    with open(os.path.join(OUTPUT_DIR, "metrics_fashion.json"), "w", encoding="utf-8") as f:
        json.dump(metrics_fashion, f, indent=2, ensure_ascii=False)

    print("  ✅ metrics_fashion.json recalcule et sauvegarde")
    return metrics_fashion


if __name__ == "__main__":
    print("═" * 60)
    print("  RECALCUL DES METRIQUES — Bank Telemarketing")
    print("═" * 60)
    recompute_bank()

    print()
    print("═" * 60)
    print("  RECALCUL DES METRIQUES — Fashion MNIST")
    print("═" * 60)
    recompute_fashion()
