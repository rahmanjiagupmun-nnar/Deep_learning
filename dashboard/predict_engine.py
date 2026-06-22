"""
predict_engine.py
Moteur de prediction en temps reel pour le Dashboard IA2.

Sous-partie A : Bank Telemarketing (reseau Top-10 variables)
Sous-partie B : Fashion MNIST (meilleur modele CNN/MLP sauvegarde)

Tous les modeles/artefacts sont charges une seule fois (lazy loading + cache)
a partir du dossier outputs_tp2/, puis reutilises a chaque appel de prediction
pour rester rapide.
"""

import os
import pickle
import numpy as np

# ── Cache global ─────────────────────────────────────────────────
_bank_cache = None
_fashion_cache = None

CLASS_NAMES_FASHION = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]


# ─────────────────────────────────────────────────────────────────
# SOUS-PARTIE A : BANK TELEMARKETING
# ─────────────────────────────────────────────────────────────────

def _load_pickle(folder, name):
    path = os.path.join(folder, name)
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def load_bank_model(tp2_dir):
    """Charge (et met en cache) le modele Top-10 + ses artefacts de pretraitement."""
    global _bank_cache
    if _bank_cache is not None:
        return _bank_cache

    if not tp2_dir or not os.path.isdir(tp2_dir):
        return None

    from tensorflow.keras.models import load_model

    keras_path = os.path.join(tp2_dir, "telemarketing_optimal_top10.keras")
    if not os.path.exists(keras_path):
        return None

    model = load_model(keras_path)
    scaler = _load_pickle(tp2_dir, "telemarketing_scaler.pkl")
    features = _load_pickle(tp2_dir, "telemarketing_features.pkl")
    top10_features = _load_pickle(tp2_dir, "telemarketing_top10_features.pkl")
    top10_idx = _load_pickle(tp2_dir, "telemarketing_top10_indices.pkl")
    label_encoder = _load_pickle(tp2_dir, "telemarketing_label_encoder.pkl")

    if scaler is None or features is None or top10_features is None or top10_idx is None:
        return None

    _bank_cache = {
        "model": model,
        "scaler": scaler,
        "features": features,            # liste complete des colonnes encodees (apres get_dummies)
        "top10_features": top10_features,  # noms des 10 variables retenues
        "top10_idx": top10_idx,          # indices de ces variables dans `features`
        "label_encoder": label_encoder,
    }
    return _bank_cache


def predict_bank(tp2_dir, raw_values):
    """
    raw_values : dict {nom_variable_top10: valeur_numerique}
    Le scaler a ete entraine sur le vecteur complet des features encodees (apres
    one-hot encoding). Pour rester fidele au pipeline du notebook, on construit un
    vecteur complet rempli de zeros/moyennes neutres, on y place les 10 valeurs
    saisies aux bons indices, on standardise avec le scaler complet, puis on
    n'extrait que les 10 colonnes du sous-modele Top-10 (top10_idx) avant de les
    donner au reseau Top-10.
    """
    bundle = load_bank_model(tp2_dir)
    if bundle is None:
        return {"error": (
            "Modele Bank Telemarketing introuvable. Verifiez que le dossier "
            "outputs_tp2/ contient bien telemarketing_optimal_top10.keras, "
            "telemarketing_scaler.pkl, telemarketing_features.pkl, "
            "telemarketing_top10_features.pkl et telemarketing_top10_indices.pkl."
        )}

    model = bundle["model"]
    scaler = bundle["scaler"]
    features = bundle["features"]
    top10_features = bundle["top10_features"]
    top10_idx = bundle["top10_idx"]
    label_encoder = bundle["label_encoder"]

    n_features = len(features)
    # Vecteur complet initialise a 0 (categories one-hot absentes = 0 ; les
    # variables numeriques non saisies restent a 0, ce qui correspond a leur
    # moyenne apres standardisation par le scaler).
    full_vector = np.zeros((1, n_features), dtype="float64")

    # On place chaque valeur saisie a l'indice correspondant a sa variable Top-10
    # dans le vecteur complet `features`.
    for fname in top10_features:
        if fname in raw_values and fname in features:
            col_idx = features.index(fname)
            try:
                full_vector[0, col_idx] = float(raw_values[fname])
            except (TypeError, ValueError):
                pass

    full_scaled = scaler.transform(full_vector)
    x_top10 = full_scaled[:, top10_idx]

    proba = float(model.predict(x_top10, verbose=0).ravel()[0])
    pred_class = int(proba >= 0.5)

    if label_encoder is not None:
        label = label_encoder.inverse_transform([pred_class])[0]
    else:
        label = "yes" if pred_class == 1 else "no"

    return {
        "probability": round(proba, 4),
        "prediction": str(label),
        "prediction_human": (
            "Le client va probablement SOUSCRIRE au produit bancaire."
            if label == "yes" else
            "Le client ne va probablement PAS souscrire au produit bancaire."
        ),
    }


# ─────────────────────────────────────────────────────────────────
# SOUS-PARTIE B : FASHION MNIST
# ─────────────────────────────────────────────────────────────────

def load_fashion_model(tp2_dir):
    """Charge (et met en cache) le meilleur modele Fashion MNIST sauvegarde."""
    global _fashion_cache
    if _fashion_cache is not None:
        return _fashion_cache

    if not tp2_dir or not os.path.isdir(tp2_dir):
        return None

    from tensorflow.keras.models import load_model

    # Ordre de preference : le meilleur modele explicitement sauvegarde,
    # puis repli sur les architectures individuelles si absent.
    candidates = [
        "fashion_mnist_best.keras",
        "fashion_mnist_resnet.keras",
        "fashion_mnist_vgg_like.keras",
        "fashion_mnist_cnn_lenet.keras",
        "fashion_mnist_mlp.keras",
        "resnet_best.keras",
        "vgg_like_best.keras",
        "cnn_lenet_best.keras",
        "mlp_best.keras",
    ]
    model = None
    used_file = None
    for fname in candidates:
        path = os.path.join(tp2_dir, fname)
        if os.path.exists(path):
            try:
                model = load_model(path)
                used_file = fname
                break
            except Exception:
                continue

    if model is None:
        return None

    _fashion_cache = {"model": model, "file": used_file}
    return _fashion_cache


def predict_fashion(tp2_dir, pixels_28x28):
    """
    pixels_28x28 : liste/array de 784 valeurs (0-255 ou 0-1), niveaux de gris,
    fond NOIR et trait BLANC (comme les images Fashion MNIST d'origine).
    """
    bundle = load_fashion_model(tp2_dir)
    if bundle is None:
        return {"error": (
            "Aucun modele Fashion MNIST trouve dans outputs_tp2/. Verifiez la "
            "presence de fashion_mnist_best.keras (ou des fichiers "
            "fashion_mnist_*.keras / *_best.keras)."
        )}

    model = bundle["model"]

    arr = np.array(pixels_28x28, dtype="float32").reshape(28, 28)
    if arr.max() > 1.0:
        arr = arr / 255.0

    # Adapte la forme d'entree selon l'architecture chargee (MLP = rank 2,
    # CNN/VGG/ResNet = rank 4), exactement comme dans le notebook.
    input_shape = model.input_shape
    rank = len(input_shape)

    if rank == 4:
        x = arr.reshape(1, 28, 28, 1)
    elif rank == 2:
        x = arr.reshape(1, 28 * 28)
    else:
        x = arr.reshape((1,) + input_shape[1:])

    proba = model.predict(x, verbose=0).ravel()
    pred_idx = int(np.argmax(proba))
    top3_idx = np.argsort(proba)[::-1][:3]

    return {
        "prediction": CLASS_NAMES_FASHION[pred_idx],
        "confidence": round(float(proba[pred_idx]), 4),
        "model_used": bundle["file"],
        "top3": [
            {"label": CLASS_NAMES_FASHION[i], "proba": round(float(proba[i]), 4)}
            for i in top3_idx
        ],
    }
