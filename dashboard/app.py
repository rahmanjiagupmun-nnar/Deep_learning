"""
TP Intelligence Artificielle 2 - Dashboard Flask
DIPES 2 | 4eme annee (S2) | 2025-2026
"""

from flask import Flask, render_template, jsonify, send_from_directory, request
import os
import json
import glob
import base64
from pathlib import Path

import predict_engine

app = Flask(__name__)

# ── Chemins des dossiers de sorties ──────────────────────────────
TP1_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs_tp1")
TP2_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs_tp2")

# Chemins alternatifs (adaptez selon votre environnement)
POSSIBLE_TP1 = [
    "outputs_tp1", "../outputs_tp1",
    os.path.join(os.path.dirname(__file__), "outputs_tp1"),
]
POSSIBLE_TP2 = [
    "outputs_tp2", "../outputs_tp2",
    os.path.join(os.path.dirname(__file__), "outputs_tp2"),
]


def find_dir(candidates):
    for p in candidates:
        if os.path.isdir(p):
            return os.path.abspath(p)
    return None


def get_images(folder):
    """Retourne la liste des images PNG d'un dossier, encodées en base64."""
    if not folder or not os.path.isdir(folder):
        return []
    images = []
    for path in sorted(glob.glob(os.path.join(folder, "*.png"))):
        name = os.path.basename(path)
        with open(path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")
        images.append({"name": name, "data": encoded})
    return images


def get_json_results(folder):
    """Lit les fichiers JSON de metriques si disponibles."""
    if not folder or not os.path.isdir(folder):
        return {}
    results = {}
    for path in glob.glob(os.path.join(folder, "*.json")):
        key = os.path.splitext(os.path.basename(path))[0]
        try:
            with open(path) as f:
                results[key] = json.load(f)
        except Exception:
            pass
    return results


# ── Metadonnees statiques des figures ────────────────────────────
TP1_FIGURE_META = {
    "01_analyse_exploratoire.png": {
        "title": "Analyse Exploratoire Complete",
        "description": (
            "Vue d'ensemble du jeu de donnees Census Income : distribution des classes "
            "de revenus, histogrammes d'age et d'heures travaillees, densite par classe "
            "et proportion de revenus >50K par niveau d'education."
        ),
        "section": "Exploration"
    },
    "02_correlation_matrix.png": {
        "title": "Matrice de Correlation",
        "description": (
            "Correlations de Pearson entre toutes les variables numeriques. "
            "La matrice triangulaire elimine la redondance et facilite la lecture des paires significatives."
        ),
        "section": "Exploration"
    },
    "03_pairplot.png": {
        "title": "Nuages de Points (Pairplot)",
        "description": (
            "Croisement deux a deux des variables cles (age, education-num, heures/semaine, capital-gain) "
            "avec coloration par classe de revenu et droites de regression superposees."
        ),
        "section": "Exploration"
    },
    "04_train_test_split.png": {
        "title": "Repartition Train / Test",
        "description": (
            "Distribution des classes dans les ensembles d'entrainement (80%) et de test (20%). "
            "La stratification garantit que les proportions sont preservees dans les deux sous-ensembles."
        ),
        "section": "Pretraitement"
    },
    "05_decision_tree_simple.png": {
        "title": "Arbre de Decision (depth=3)",
        "description": (
            "Visualisation de l'arbre de decision simple a profondeur 3. "
            "Chaque noeud indique la variable de coupure, la valeur seuil, le critere d'impurete Gini "
            "et la distribution des classes."
        ),
        "section": "Modeles"
    },
    "06_dt_depth_vs_accuracy.png": {
        "title": "Validation Croisee - Profondeur de l'Arbre",
        "description": (
            "Accuracy de validation croisee (5-folds) en fonction de la profondeur maximale de l'arbre. "
            "La bande bleue represente l'intervalle de confiance a +/- 1 ecart-type. "
            "La profondeur optimale est indiquee par la ligne rouge."
        ),
        "section": "Modeles"
    },
    "07_bagging_b_impact.png": {
        "title": "Impact du Nombre d'Arbres (Bagging)",
        "description": (
            "A gauche : accuracy en fonction du nombre d'arbres B pour le Bagging. "
            "A droite : temps d'entrainement correspondant. On observe la stabilisation "
            "de la performance au-dela d'un certain seuil de B."
        ),
        "section": "Ensemble"
    },
    "08_gb_learning_curve.png": {
        "title": "Courbe d'Apprentissage - Gradient Boosting",
        "description": (
            "Evolution de l'accuracy sur les ensembles d'entrainement et de test au fil des iterations "
            "du Gradient Boosting. Permet de detecter le sur-apprentissage et de choisir "
            "le nombre optimal d'arbres (early stopping)."
        ),
        "section": "Ensemble"
    },
    "09_model_comparison.png": {
        "title": "Comparaison Globale des Modeles",
        "description": (
            "A gauche : accuracy sur le jeu de test pour tous les classifieurs evalues. "
            "A droite : comparaison multi-metriques (Accuracy, F1, Precision, Rappel) "
            "pour les 5 meilleurs modeles."
        ),
        "section": "Comparaison"
    },
    "10_confusion_matrices.png": {
        "title": "Matrices de Confusion",
        "description": (
            "Matrices de confusion pour les 4 meilleurs classifieurs. "
            "Chaque cellule montre le nombre de predictions correctes (diagonale) "
            "et incorrectes (hors diagonale) par classe."
        ),
        "section": "Evaluation"
    },
    "11_roc_curves.png": {
        "title": "Courbes ROC et AUC",
        "description": (
            "Courbes ROC comparatives : Taux de Vrais Positifs (TPR) vs Taux de Faux Positifs (FPR) "
            "pour tous les modeles. L'AUC (Aire Sous la Courbe) mesure la capacite discriminante "
            "independamment du seuil de decision."
        ),
        "section": "Evaluation"
    },
    "12_feature_importance.png": {
        "title": "Importance des Variables",
        "description": (
            "Top 15 variables les plus importantes selon trois modeles : Random Forest, "
            "Gradient Boosting et Decision Tree. Scores bases sur la reduction moyenne "
            "de l'impurete Gini a chaque coupure."
        ),
        "section": "Interpretation"
    },
    "13_feature_importance_heatmap.png": {
        "title": "Heatmap des Importances Normalisees",
        "description": (
            "Visualisation croisee de l'importance normalisee des 15 variables les plus pertinentes "
            "selon les trois modeles d'ensemble. Permet d'identifier les variables consensuelles."
        ),
        "section": "Interpretation"
    },
    "14_learning_curves.png": {
        "title": "Courbes d'Apprentissage - Diagnostic Biais/Variance",
        "description": (
            "Courbes d'apprentissage pour les 3 meilleurs modeles : accuracy d'entrainement et de "
            "validation croisee en fonction de la taille du jeu d'entrainement. "
            "Diagnostique le sur-apprentissage (variance) et le sous-apprentissage (biais)."
        ),
        "section": "Diagnostic"
    },
}

TP2_FIGURE_META = {
    "01_analyse_exploratoire.png": {
        "title": "Analyse Exploratoire - Bank Telemarketing",
        "description": (
            "Distribution de la variable cible (souscription : yes/no), age et duree d'appel par "
            "classe, taux de souscription par emploi et par niveau d'education, matrice de correlation."
        ),
        "section": "Exploration",
        "souspartie": "bank"
    },
    "02_pairplot.png": {
        "title": "Nuages de Points Croises (Pairplot)",
        "description": (
            "Croisement deux a deux des variables cles (age, duree d'appel, campaign, euribor3m, "
            "nr.employed) avec coloration par classe et droites de regression superposees."
        ),
        "section": "Exploration",
        "souspartie": "bank"
    },
    "03_model_comparison.png": {
        "title": "Comparaison de Tous les Modeles",
        "description": (
            "F1-Score et comparaison multi-metriques (Accuracy, F1, Precision, Rappel) pour les "
            "classifieurs classiques (Arbre, Random Forest, Gradient Boosting, AdaBoost, etc.) et "
            "les 3 reseaux de neurones (Simple, Profond, Top-10 variables)."
        ),
        "section": "Comparaison",
        "souspartie": "bank"
    },
    "04_confusion_matrices.png": {
        "title": "Matrices de Confusion",
        "description": (
            "Matrices de confusion des meilleurs modeles (Decision Tree, Random Forest, "
            "Gradient Boosting, Reseau Profond) avec accuracy et F1-Score associes."
        ),
        "section": "Evaluation",
        "souspartie": "bank"
    },
    "05_roc_curves.png": {
        "title": "Courbes ROC et AUC",
        "description": (
            "Courbes ROC comparatives (TPR vs FPR) pour tous les modeles classiques et les 3 "
            "reseaux de neurones, avec classement des AUC. Metrique cle en contexte desequilibre."
        ),
        "section": "Evaluation",
        "souspartie": "bank"
    },
    "06_learning_curves_ann.png": {
        "title": "Courbes d'Apprentissage - Reseaux de Neurones",
        "description": (
            "Evolution de la loss (binary crossentropy) et de l'accuracy sur train/validation "
            "pour le Reseau Simple et le Reseau Profond, au fil des epoques."
        ),
        "section": "Reseaux de Neurones",
        "souspartie": "bank"
    },
    "07_feature_importance.png": {
        "title": "Importance des Variables (Top 15)",
        "description": (
            "Top 15 variables les plus importantes selon Random Forest, Gradient Boosting et "
            "Decision Tree. Les 10 premieres (selon RF) sont retenues pour le modele de deploiement."
        ),
        "section": "Interpretation",
        "souspartie": "bank"
    },
    "08_fashion_mnist_samples.png": {
        "title": "Echantillons Fashion MNIST",
        "description": (
            "Exemples d'images du jeu de donnees Fashion MNIST : 70 000 images 28x28 "
            "en niveaux de gris reparties en 10 categories (t-shirt, pantalon, pull, robe, etc.)."
        ),
        "section": "Deep Learning",
        "souspartie": "mnist"
    },
    "09_fashion_mnist_learning_curves.png": {
        "title": "Courbes d'Apprentissage - Architectures CNN",
        "description": (
            "Courbes d'apprentissage (loss / accuracy) des 4 architectures testees sur Fashion "
            "MNIST : MLP, CNN LeNet+, VGGNet-like et ResNet custom."
        ),
        "section": "Deep Learning",
        "souspartie": "mnist"
    },
    "10_fashion_mnist_confusion_matrix.png": {
        "title": "Matrice de Confusion - Meilleur Modele Fashion MNIST",
        "description": (
            "Matrice de confusion du meilleur modele retenu sur les 10 categories de "
            "vetements de Fashion MNIST."
        ),
        "section": "Deep Learning",
        "souspartie": "mnist"
    },
    "11_fashion_mnist_predictions.png": {
        "title": "Exemples de Predictions - Fashion MNIST",
        "description": (
            "Echantillon d'images de test avec la classe predite et la classe reelle, "
            "illustrant les succes et erreurs du meilleur modele."
        ),
        "section": "Deep Learning",
        "souspartie": "mnist"
    },
}


# ── Top-10 variables retenues pour le modele de deploiement ──────
# (issues de telemarketing_top10_features.pkl, ordre d'importance Random Forest)
TOP10_FEATURES_INFO = [
    {"name": "duration", "label": "Duree du dernier appel (secondes)",
     "desc": "Variable la plus discriminante : plus l'appel est long, plus la probabilite de souscription est elevee."},
    {"name": "euribor3m", "label": "Taux Euribor 3 mois",
     "desc": "Indicateur macroeconomique : un taux bas est associe a un contexte plus favorable a la souscription."},
    {"name": "nr.employed", "label": "Nombre d'employes (indicateur trimestriel)",
     "desc": "Indicateur macroeconomique national reflétant la conjoncture de l'emploi."},
    {"name": "age", "label": "Age du client",
     "desc": "Variable demographique numerique."},
    {"name": "campaign", "label": "Nombre de contacts durant la campagne",
     "desc": "Trop de contacts peut indiquer une moindre receptivite du client."},
    {"name": "pdays", "label": "Jours depuis le dernier contact (campagne precedente)",
     "desc": "999 signifie que le client n'a jamais ete contacte auparavant."},
    {"name": "cons.conf.idx", "label": "Indice de confiance des consommateurs",
     "desc": "Indicateur macroeconomique mensuel."},
    {"name": "cons.price.idx", "label": "Indice des prix a la consommation",
     "desc": "Indicateur macroeconomique mensuel."},
    {"name": "emp.var.rate", "label": "Taux de variation de l'emploi",
     "desc": "Indicateur macroeconomique trimestriel."},
    {"name": "previous", "label": "Nombre de contacts avant cette campagne",
     "desc": "Historique d'interaction avec le client."},
]




@app.route("/")
def index():
    tp1_dir = find_dir(POSSIBLE_TP1)
    tp2_dir = find_dir(POSSIBLE_TP2)
    tp1_count = len(glob.glob(os.path.join(tp1_dir, "*.png"))) if tp1_dir else 0
    tp2_count = len(glob.glob(os.path.join(tp2_dir, "*.png"))) if tp2_dir else 0
    return render_template(
        "index.html",
        tp1_available=tp1_dir is not None,
        tp2_available=tp2_dir is not None,
        tp1_count=tp1_count,
        tp2_count=tp2_count,
    )


@app.route("/partie1")
def partie1():
    tp1_dir = find_dir(POSSIBLE_TP1)
    images = []
    if tp1_dir:
        for path in sorted(glob.glob(os.path.join(tp1_dir, "*.png"))):
            name = os.path.basename(path)
            with open(path, "rb") as f:
                encoded = base64.b64encode(f.read()).decode("utf-8")
            meta = TP1_FIGURE_META.get(name, {
                "title": name.replace("_", " ").replace(".png", "").title(),
                "description": "Figure generee par le script d'analyse.",
                "section": "General"
            })
            images.append({"name": name, "data": encoded, **meta})
    return render_template("partie1.html", images=images, has_data=bool(images))


@app.route("/partie2")
def partie2():
    tp2_dir = find_dir(POSSIBLE_TP2)
    images = []
    if tp2_dir:
        for path in sorted(glob.glob(os.path.join(tp2_dir, "*.png"))):
            name = os.path.basename(path)
            with open(path, "rb") as f:
                encoded = base64.b64encode(f.read()).decode("utf-8")
            meta = TP2_FIGURE_META.get(name, {
                "title": name.replace("_", " ").replace(".png", "").title(),
                "description": "Figure generee par le notebook IA Partie 2.",
                "section": "General",
                "souspartie": "bank"
            })
            images.append({"name": name, "data": encoded, **meta})
    return render_template("partie2.html", images=images, has_data=bool(images))


@app.route("/methodes")
def methodes():
    return render_template("methodes.html")


@app.route("/predictions")
def predictions():
    tp2_dir = find_dir(POSSIBLE_TP2)
    json_results = get_json_results(tp2_dir)
    metrics_bank = json_results.get("metrics_bank")
    metrics_fashion = json_results.get("metrics_fashion")
    return render_template("predictions.html", top10=TOP10_FEATURES_INFO,
                            fashion_classes=predict_engine.CLASS_NAMES_FASHION,
                            metrics_bank=metrics_bank,
                            metrics_fashion=metrics_fashion)


@app.route("/api/predict/bank", methods=["POST"])
def api_predict_bank():
    tp2_dir = find_dir(POSSIBLE_TP2)
    payload = request.get_json(force=True, silent=True) or {}
    result = predict_engine.predict_bank(tp2_dir, payload)
    status = 400 if "error" in result else 200
    return jsonify(result), status


@app.route("/api/predict/fashion", methods=["POST"])
def api_predict_fashion():
    tp2_dir = find_dir(POSSIBLE_TP2)
    payload = request.get_json(force=True, silent=True) or {}
    pixels = payload.get("pixels")
    if not pixels or len(pixels) != 784:
        return jsonify({"error": "784 valeurs de pixels (image 28x28) attendues."}), 400
    result = predict_engine.predict_fashion(tp2_dir, pixels)
    status = 400 if "error" in result else 200
    return jsonify(result), status


@app.route("/api/stats")
def api_stats():
    tp1_dir = find_dir(POSSIBLE_TP1)
    tp2_dir = find_dir(POSSIBLE_TP2)
    return jsonify({
        "tp1": {
            "path": tp1_dir,
            "figures": len(glob.glob(os.path.join(tp1_dir, "*.png"))) if tp1_dir else 0,
        },
        "tp2": {
            "path": tp2_dir,
            "figures": len(glob.glob(os.path.join(tp2_dir, "*.png"))) if tp2_dir else 0,
        }
    })

"""
if __name__ == "__main__":
    app.run(debug=True, port=5050)
"""
