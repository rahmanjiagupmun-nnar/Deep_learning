"""
╔══════════════════════════════════════════════════════════════════╗
║     INTELLIGENCE ARTIFICIELLE 2 - TP : Partie 1                 ║
║     Analyse & Classification - Dataset Census Income            ║
║     DIPES 2 | 4ème année (S2) | 2025-2026                       ║
║     Auteur : Stéphane C. K. TÉKOUABOU (PhD & Ing.)              ║
╚══════════════════════════════════════════════════════════════════╝
"""

# ─────────────────────────────────────────────────────────────────
# 0. IMPORTS & CONFIGURATION GLOBALE
# ─────────────────────────────────────────────────────────────────
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from scipy import stats

from sklearn.model_selection import (train_test_split, cross_val_score,
                                     GridSearchCV, StratifiedKFold,
                                     learning_curve)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                              BaggingClassifier, AdaBoostClassifier)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_curve, auc,
                             roc_auc_score, f1_score, precision_score,
                             recall_score, ConfusionMatrixDisplay)
from sklearn.dummy import DummyClassifier
from sklearn.inspection import permutation_importance

import time
import os

# ── Palette & Style global ──────────────────────────────────────
PALETTE = {
    "primary":   "#6C3FC5",   # violet
    "secondary": "#F7931E",   # orange
    "accent":    "#00B4D8",   # cyan
    "success":   "#2DC653",   # vert
    "danger":    "#E63946",   # rouge
    "neutral":   "#8D99AE",   # gris
    "bg":        "#F8F9FA",
    "dark":      "#1A1A2E",
}

CMAP_MAIN  = LinearSegmentedColormap.from_list("custom",
                ["#EDE7F6", PALETTE["primary"]])
CMAP_DIVG  = LinearSegmentedColormap.from_list("divg",
                [PALETTE["accent"], "white", PALETTE["secondary"]])

# Seaborn global
sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams.update({
    "figure.facecolor":  PALETTE["bg"],
    "axes.facecolor":    "white",
    "axes.edgecolor":    "#DEE2E6",
    "axes.labelsize":    11,
    "axes.titlesize":    13,
    "axes.titleweight":  "bold",
    "axes.titlecolor":   PALETTE["dark"],
    "axes.grid":         True,
    "grid.alpha":        0.4,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "legend.fontsize":   9,
    "legend.framealpha": 0.85,
    "font.family":       "DejaVu Sans",
    "figure.dpi":        130,
    "savefig.dpi":       180,
    "savefig.bbox":      "tight",
    "savefig.facecolor": PALETTE["bg"],
})

OUTPUT_DIR = "outputs_tp1"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def save_fig(name, fig=None):
    path = os.path.join(OUTPUT_DIR, f"{name}.png")
    (fig or plt).savefig(path)
    print(f"   💾 Figure sauvegardée → {path}")
    plt.show()
    plt.close("all")

def section(title, emoji=""):
    bar = "═" * 55
    print(f"\n{bar}")
    print(f"  {emoji}  {title}")
    print(f"{bar}")

def subsection(title):
    print(f"\n  ► {title}")
    print("  " + "─" * 50)


# ─────────────────────────────────────────────────────────────────
# 1. CHARGEMENT & PREMIÈRE ANALYSE
# ─────────────────────────────────────────────────────────────────
section("PARTIE 1 — CHARGEMENT & ANALYSE EXPLORATOIRE", "📂")

column_names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num',
                'marital-status', 'occupation', 'relationship', 'race', 'sex',
                'capital-gain', 'capital-loss', 'hours-per-week',
                'native-country', 'income']

df = pd.read_csv('census.csv', header=None, names=column_names)

# Nettoyage des espaces
for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].str.strip()

numeric_cols = ['age', 'fnlwgt', 'education-num',
                'capital-gain', 'capital-loss', 'hours-per-week']
cat_cols     = [c for c in df.columns
                if c not in numeric_cols and c != 'income']

subsection("Dimensions & types")
print(f"  Instances        : {df.shape[0]:,}")
print(f"  Caractéristiques : {df.shape[1]}")
print(f"  Variables numériques ({len(numeric_cols)}) : {numeric_cols}")
print(f"  Variables catégorielles ({len(cat_cols)}) : {cat_cols}")

subsection("Distribution de la variable cible")
income_dist = df['income'].value_counts()
for k, v in income_dist.items():
    print(f"  {k:>6s} : {v:>6,}  ({v/len(df)*100:.1f}%)")

subsection("Valeurs manquantes ('?')")
missing_info = {}
for col in df.select_dtypes('object').columns:
    n = (df[col] == '?').sum()
    if n:
        missing_info[col] = n
        print(f"  {col:<20s}: {n:,}  ({n/len(df)*100:.1f}%)")
if not missing_info:
    print("  Aucune valeur manquante détectée.")

print("\n  Statistiques descriptives (variables numériques):")
print(df[numeric_cols].describe().round(2).to_string())


# ── 1.1 Figure : Vue d'ensemble exploratoire ──────────────────
fig = plt.figure(figsize=(18, 14))
fig.suptitle("Census Income — Analyse Exploratoire Complète",
             fontsize=16, fontweight='bold', color=PALETTE["dark"], y=1.01)
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

# 1a — Distribution cible (pie)
ax0 = fig.add_subplot(gs[0, 0])
wedges, texts, autotexts = ax0.pie(
    income_dist.values,
    labels=income_dist.index,
    autopct='%1.1f%%',
    startangle=90,
    colors=[PALETTE["primary"], PALETTE["secondary"]],
    wedgeprops=dict(edgecolor='white', linewidth=2),
    explode=(0.04, 0),
    shadow=True,
)
for at in autotexts:
    at.set_fontsize(11)
    at.set_fontweight('bold')
    at.set_color('white')
ax0.set_title("Distribution des revenus")

# 1b — Distribution âge
ax1 = fig.add_subplot(gs[0, 1])
ax1.hist(df['age'], bins=40,
         color=PALETTE["accent"], alpha=0.8, edgecolor='white', linewidth=0.5)
ax1.axvline(df['age'].mean(), color=PALETTE["danger"],
            linestyle='--', linewidth=2, label=f"Moy. = {df['age'].mean():.1f}")
ax1.axvline(df['age'].median(), color=PALETTE["secondary"],
            linestyle=':', linewidth=2, label=f"Méd. = {df['age'].median():.1f}")
ax1.set_title("Distribution — Âge")
ax1.set_xlabel("Âge")
ax1.legend()

# 1c — Heures/semaine
ax2 = fig.add_subplot(gs[0, 2])
ax2.hist(df['hours-per-week'], bins=35,
         color=PALETTE["secondary"], alpha=0.8, edgecolor='white', linewidth=0.5)
ax2.axvline(df['hours-per-week'].mean(), color=PALETTE["danger"],
            linestyle='--', linewidth=2,
            label=f"Moy. = {df['hours-per-week'].mean():.1f}")
ax2.set_title("Distribution — Heures/semaine")
ax2.set_xlabel("Heures par semaine")
ax2.legend()

# 1d — Revenu par âge (KDE)
ax3 = fig.add_subplot(gs[1, :2])
for inc, col in zip(['>50K', '<=50K'],
                    [PALETTE["primary"], PALETTE["secondary"]]):
    subset = df[df['income'] == inc]['age']
    subset.plot.kde(ax=ax3, label=inc, color=col, linewidth=2.5)
    ax3.axvline(subset.mean(), color=col, linestyle='--',
                linewidth=1.5, alpha=0.7)
ax3.fill_between(ax3.lines[0].get_xdata(), ax3.lines[0].get_ydata(),
                 alpha=0.08, color=PALETTE["primary"])
ax3.fill_between(ax3.lines[1].get_xdata(), ax3.lines[1].get_ydata(),
                 alpha=0.08, color=PALETTE["secondary"])
ax3.set_title("Densité d'âge par classe de revenu")
ax3.set_xlabel("Âge")
ax3.legend(title="Revenu")

# 1e — Top professions >50K
ax4 = fig.add_subplot(gs[1, 2])
top_occ = df[df['income'] == '>50K']['occupation'].value_counts().head(6)
bars = ax4.barh(top_occ.index[::-1], top_occ.values[::-1],
                color=PALETTE["primary"], edgecolor='white', linewidth=0.5)
for bar, val in zip(bars, top_occ.values[::-1]):
    ax4.text(bar.get_width() + 50, bar.get_y() + bar.get_height()/2,
             f'{val:,}', va='center', fontsize=8, color=PALETTE["dark"])
ax4.set_title("Top professions (>50K)")
ax4.set_xlabel("Nombre")

# 1f — Éducation vs revenu
ax5 = fig.add_subplot(gs[2, :])
edu_income = (df.groupby(['education', 'income'])
               .size().unstack(fill_value=0))
edu_income['ratio'] = (edu_income['>50K'] /
                       (edu_income['>50K'] + edu_income['<=50K']) * 100)
edu_sorted = edu_income.sort_values('ratio', ascending=False)
bars = ax5.bar(range(len(edu_sorted)), edu_sorted['ratio'],
               color=[plt.cm.RdYlGn(r/100) for r in edu_sorted['ratio']],
               edgecolor='white', linewidth=0.6)
ax5.set_xticks(range(len(edu_sorted)))
ax5.set_xticklabels(edu_sorted.index, rotation=35, ha='right', fontsize=9)
ax5.set_title("Proportion de revenus >50K par niveau d'éducation (%)")
ax5.set_ylabel("% revenus >50K")
ax5.axhline(income_dist['>50K']/len(df)*100, color=PALETTE["danger"],
            linestyle='--', linewidth=1.5, label='Moyenne globale')
ax5.legend()
for bar, val in zip(bars, edu_sorted['ratio']):
    ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f'{val:.0f}%', ha='center', fontsize=8, fontweight='bold')

save_fig("01_analyse_exploratoire", fig)


# ─────────────────────────────────────────────────────────────────
# 2. MATRICE DE CORRÉLATION & SCATTER PLOTS
# ─────────────────────────────────────────────────────────────────
section("MATRICE DE CORRÉLATION & NUAGES DE POINTS", "📊")

corr = df[numeric_cols].corr()
print("\n  Matrice de corrélation (Pearson):")
print(corr.round(3).to_string())

# Masque triangle sup
mask = np.triu(np.ones_like(corr, dtype=bool))

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Analyse de Corrélation — Variables Numériques",
             fontsize=14, fontweight='bold', color=PALETTE["dark"])

# Heatmap full
sns.heatmap(corr, annot=True, fmt='.2f', cmap=CMAP_DIVG,
            center=0, vmin=-1, vmax=1, ax=axes[0],
            annot_kws={"size": 10, "weight": "bold"},
            linewidths=0.5, linecolor='#DEE2E6',
            cbar_kws={"shrink": 0.8, "label": "Corrélation"})
axes[0].set_title("Matrice de corrélation complète")
axes[0].tick_params(axis='x', rotation=30)

# Heatmap triangulaire
corr_masked = corr.copy()
corr_masked[mask] = np.nan
sns.heatmap(corr_masked, annot=True, fmt='.2f', cmap=CMAP_DIVG,
            center=0, vmin=-1, vmax=1, ax=axes[1],
            annot_kws={"size": 10}, linewidths=0.5,
            linecolor='#DEE2E6', cbar=False)
axes[1].set_title("Triangulaire inférieure (sans redondance)")
axes[1].tick_params(axis='x', rotation=30)

plt.tight_layout()
save_fig("02_correlation_matrix", fig)

# Pairplot enrichi avec la cible en couleur
subsection("Nuages de points (pairplot)")
pair_cols = ['age', 'education-num', 'hours-per-week', 'capital-gain', 'income']
pair_df   = df[pair_cols].copy()
pair_df   = pair_df[pair_df['capital-gain'] < pair_df['capital-gain'].quantile(0.99)]

g = sns.pairplot(pair_df, hue='income', diag_kind='kde',
                 palette={'>50K': PALETTE["primary"], '<=50K': PALETTE["secondary"]},
                 plot_kws=dict(alpha=0.4, s=15),
                 diag_kws=dict(linewidth=2))
g.figure.suptitle("Nuages de points croisés — Variables clés",
                  y=1.02, fontsize=14, fontweight='bold')

# Ajouter les droites de régression
for i, col_y in enumerate(['age', 'education-num', 'hours-per-week', 'capital-gain']):
    for j, col_x in enumerate(['age', 'education-num', 'hours-per-week', 'capital-gain']):
        if i != j:
            ax = g.axes[i][j]
            for inc, col in zip(['>50K', '<=50K'],
                                [PALETTE["primary"], PALETTE["secondary"]]):
                sub = df[df['income'] == inc]
                m, b, r, p, _ = stats.linregress(sub[col_x], sub[col_y])
                x_range = np.linspace(sub[col_x].min(), sub[col_x].max(), 100)
                ax.plot(x_range, m * x_range + b, color=col,
                        linewidth=1.5, linestyle='--', alpha=0.7)

save_fig("03_pairplot", g.figure)


# ─────────────────────────────────────────────────────────────────
# 3. PRÉTRAITEMENT
# ─────────────────────────────────────────────────────────────────
section("PRÉTRAITEMENT DES DONNÉES", "⚙️")

df_clean = df.copy()
for col in df_clean.select_dtypes('object').columns:
    df_clean = df_clean[df_clean[col] != '?']

print(f"  Instances avant nettoyage : {len(df):,}")
print(f"  Instances après nettoyage : {len(df_clean):,}")
print(f"  Supprimées                : {len(df)-len(df_clean):,} ({(len(df)-len(df_clean))/len(df)*100:.1f}%)")

le_income = LabelEncoder()
df_clean['income_encoded'] = le_income.fit_transform(df_clean['income'])
print(f"\n  Encodage cible : {dict(zip(le_income.classes_, le_income.transform(le_income.classes_)))}")

encode_cols = ['workclass', 'education', 'marital-status', 'occupation',
               'relationship', 'race', 'sex', 'native-country']
df_enc = pd.get_dummies(df_clean, columns=encode_cols, drop_first=True)

feature_cols = [c for c in df_enc.columns if c not in ['income', 'income_encoded']]
X = df_enc[feature_cols].values
y = df_enc['income_encoded'].values

print(f"\n  Features après encodage one-hot : {len(feature_cols)}")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y)

print(f"\n  Jeu d'entraînement : {len(X_train):,} instances")
print(f"  Jeu de test        : {len(X_test):,} instances")
print(f"  Distribution train : {np.bincount(y_train)} (<=50K / >50K)")
print(f"  Distribution test  : {np.bincount(y_test)}")

# Visualisation de la répartition train/test
fig, ax = plt.subplots(figsize=(8, 4))
split_data = {
    'Entraînement': [np.bincount(y_train)[0], np.bincount(y_train)[1]],
    'Test':         [np.bincount(y_test)[0],  np.bincount(y_test)[1]]
}
x = np.arange(2)
w = 0.35
for i, (split, vals) in enumerate(split_data.items()):
    offset = (i - 0.5) * w
    b = ax.bar(x + offset, vals, w,
               label=split,
               color=[PALETTE["primary"], PALETTE["secondary"]][i],
               edgecolor='white', linewidth=0.8)
    for bar in b:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                f'{int(bar.get_height()):,}', ha='center', fontsize=9, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(['≤50K', '>50K'])
ax.set_title("Répartition des classes — Train vs Test")
ax.set_ylabel("Nombre d'instances")
ax.legend()
plt.tight_layout()
save_fig("04_train_test_split", fig)


# ─────────────────────────────────────────────────────────────────
# 4. CLASSIFIEUR CONSTANT (BASELINE)
# ─────────────────────────────────────────────────────────────────
section("CLASSIFIEUR CONSTANT (BASELINE)", "📏")

dummy = DummyClassifier(strategy='most_frequent', random_state=42)
dummy.fit(X_train, y_train)
dummy_acc = accuracy_score(y_test, dummy.predict(X_test))
print(f"  Accuracy baseline (classe majoritaire) : {dummy_acc:.4f}")
print(f"  → Tout classifieur utile doit dépasser {dummy_acc:.2%}")


# ─────────────────────────────────────────────────────────────────
# 5. ARBRES DE DÉCISION
# ─────────────────────────────────────────────────────────────────
section("ARBRES DE DÉCISION", "🌳")

subsection("Arbre simple (max_depth=3)")
dt_simple = DecisionTreeClassifier(max_depth=3, random_state=42)
dt_simple.fit(X_train, y_train)
y_pred_dt3 = dt_simple.predict(X_test)
print(f"  Accuracy (depth=3) : {accuracy_score(y_test, y_pred_dt3):.4f}")

fig, ax = plt.subplots(figsize=(22, 9))
try:
    plot_tree(dt_simple, feature_names=feature_cols,
          class_names=le_income.classes_,
          filled=True, rounded=True, fontsize=8,     # ✅ entier
          impurity=True, proportion=True, ax=ax)
except TypeError:
    # proportion=True non disponible sur certaines versions de sklearn
    plot_tree(dt_simple, feature_names=feature_cols,
          class_names=le_income.classes_,
          filled=True, rounded=True, fontsize=8,     # ✅ entier
          impurity=True, ax=ax)
ax.set_title("Arbre de Décision — max_depth = 3",
             fontsize=15, fontweight='bold', pad=15)
save_fig("05_decision_tree_simple", fig)

subsection("GridSearch — Optimisation de l'arbre")
param_grid_dt = {
    'max_depth':        [3, 5, 7, 10, 15, 20, None],
    'min_samples_split':[2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'criterion':        ['gini', 'entropy'],
}
cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
dt_gs = GridSearchCV(DecisionTreeClassifier(random_state=42),
                     param_grid_dt, cv=cv5,
                     scoring='f1', n_jobs=-1, verbose=0)
t0 = time.time()
dt_gs.fit(X_train, y_train)
print(f"  Temps GridSearch : {time.time()-t0:.1f}s")
print(f"  Meilleurs params : {dt_gs.best_params_}")
print(f"  Meilleur F1 CV   : {dt_gs.best_score_:.4f}")

dt_opt = dt_gs.best_estimator_
y_pred_dt_opt = dt_opt.predict(X_test)
print(f"  Accuracy test    : {accuracy_score(y_test, y_pred_dt_opt):.4f}")
print(f"  F1-Score test    : {f1_score(y_test, y_pred_dt_opt):.4f}")

# Courbe depth vs score
depths  = list(range(2, 25))
cv_mean = []
cv_std  = []
for d in depths:
    dt_d   = DecisionTreeClassifier(max_depth=d, random_state=42)
    scores = cross_val_score(dt_d, X_train, y_train,
                             cv=cv5, scoring='accuracy', n_jobs=-1)
    cv_mean.append(scores.mean())
    cv_std.append(scores.std())

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(depths, cv_mean, 'o-', color=PALETTE["primary"],
        linewidth=2.5, markersize=6, label="CV Accuracy (moy.)")
ax.fill_between(depths,
                np.array(cv_mean) - np.array(cv_std),
                np.array(cv_mean) + np.array(cv_std),
                alpha=0.2, color=PALETTE["primary"], label="±1 std")
best_d = dt_gs.best_params_['max_depth']
ax.axvline(best_d if best_d else max(depths),
           color=PALETTE["danger"], linestyle='--', linewidth=2,
           label=f"Depth optimale = {best_d}")
ax.axhline(dummy_acc, color=PALETTE["neutral"], linestyle=':',
           linewidth=1.5, label=f"Baseline = {dummy_acc:.3f}")
ax.set_title("Validation Croisée — Profondeur de l'arbre de décision")
ax.set_xlabel("Profondeur maximale (max_depth)")
ax.set_ylabel("Accuracy CV (5-folds)")
ax.legend()
plt.tight_layout()
save_fig("06_dt_depth_vs_accuracy", fig)


# ─────────────────────────────────────────────────────────────────
# 6. BAGGING & RANDOM FOREST
# ─────────────────────────────────────────────────────────────────
section("BAGGING & RANDOM FOREST", "🌲")

subsection("Impact du nombre d'arbres B (Bagging)")
b_values  = [5, 10, 20, 50, 100, 200, 300]
bag_scores = []
bag_times  = []
for b in b_values:
    t0 = time.time()
    bag = BaggingClassifier(
        estimator=DecisionTreeClassifier(max_depth=10),
        n_estimators=b, random_state=42, n_jobs=-1)
    bag.fit(X_train, y_train)
    bag_scores.append(bag.score(X_test, y_test))
    bag_times.append(time.time() - t0)
    print(f"  Bagging B={b:3d} → Accuracy={bag_scores[-1]:.4f}  ({bag_times[-1]:.1f}s)")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.plot(b_values, bag_scores, 's-', color=PALETTE["primary"],
         linewidth=2.5, markersize=8, label="Bagging")
ax1.axhline(dummy_acc, linestyle=':', color=PALETTE["neutral"],
            label="Baseline")
ax1.set_title("Accuracy vs Nombre d'arbres (Bagging)")
ax1.set_xlabel("B (nombre d'arbres)")
ax1.set_ylabel("Accuracy (test)")
ax1.legend()

ax2.plot(b_values, bag_times, 'D-', color=PALETTE["secondary"],
         linewidth=2.5, markersize=8)
ax2.set_title("Temps d'entraînement vs B")
ax2.set_xlabel("B (nombre d'arbres)")
ax2.set_ylabel("Temps (secondes)")
plt.tight_layout()
save_fig("07_bagging_b_impact", fig)

subsection("Random Forest — OOB & GridSearch")
rf_oob = RandomForestClassifier(n_estimators=200, oob_score=True,
                                max_features='sqrt', random_state=42, n_jobs=-1)
rf_oob.fit(X_train, y_train)
print(f"  OOB Score       : {rf_oob.oob_score_:.4f}")
print(f"  Accuracy test   : {rf_oob.score(X_test, y_test):.4f}")

# GridSearch RF
param_grid_rf = {
    'max_features': ['sqrt', 'log2', 0.3, 0.5],
    'n_estimators': [100, 200],
    'max_depth':    [10, 15, 20, None],
    'min_samples_leaf': [1, 2],
}
rf_gs = GridSearchCV(
    RandomForestClassifier(random_state=42, n_jobs=-1, oob_score=False),
    param_grid_rf, cv=StratifiedKFold(3, shuffle=True, random_state=42),
    scoring='f1', n_jobs=-1, verbose=0)
t0 = time.time()
rf_gs.fit(X_train[:6000], y_train[:6000])
print(f"\n  GridSearch RF terminé en {time.time()-t0:.1f}s")
print(f"  Meilleurs params : {rf_gs.best_params_}")

rf_opt = RandomForestClassifier(n_estimators=200, max_depth=15,
                                max_features='sqrt', random_state=42, n_jobs=-1)
rf_opt.fit(X_train, y_train)
y_pred_rf = rf_opt.predict(X_test)
print(f"  RF optimisé — Accuracy : {accuracy_score(y_test, y_pred_rf):.4f}")
print(f"  RF optimisé — F1       : {f1_score(y_test, y_pred_rf):.4f}")


# ─────────────────────────────────────────────────────────────────
# 7. GRADIENT BOOSTING
# ─────────────────────────────────────────────────────────────────
section("GRADIENT BOOSTING & ADABOOST", "🚀")

gb_base = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1,
                                     max_depth=3, random_state=42)
gb_base.fit(X_train, y_train)
print(f"  GB (défaut) — Accuracy : {gb_base.score(X_test, y_test):.4f}")

# Early stopping via n_iter_no_change
gb_es = GradientBoostingClassifier(
    n_estimators=500, learning_rate=0.05, max_depth=4,
    subsample=0.8, min_samples_leaf=10,
    validation_fraction=0.15, n_iter_no_change=15, tol=1e-4,
    random_state=42)
t0 = time.time()
gb_es.fit(X_train, y_train)
print(f"\n  GB (early stopping) — Arbres utilisés : {gb_es.n_estimators_}")
print(f"  GB (early stopping) — Accuracy        : {gb_es.score(X_test, y_test):.4f}")
print(f"  Temps                                  : {time.time()-t0:.1f}s")

# GridSearch GB
param_grid_gb = {
    'n_estimators':    [50, 100],
    'learning_rate':   [0.05, 0.1, 0.2],
    'max_depth':       [3, 5],
    'subsample':       [0.7, 1.0],
}
gb_gs = GridSearchCV(
    GradientBoostingClassifier(random_state=42),
    param_grid_gb, cv=3, scoring='f1', n_jobs=-1, verbose=0)
t0 = time.time()
gb_gs.fit(X_train[:4000], y_train[:4000])
print(f"\n  GridSearch GB terminé en {time.time()-t0:.1f}s")
print(f"  Meilleurs params : {gb_gs.best_params_}")

gb_opt = GradientBoostingClassifier(
    n_estimators=100, learning_rate=0.1, max_depth=5,
    subsample=0.8, random_state=42)
gb_opt.fit(X_train, y_train)
y_pred_gb = gb_opt.predict(X_test)
print(f"  GB optimisé — Accuracy : {accuracy_score(y_test, y_pred_gb):.4f}")
print(f"  GB optimisé — F1       : {f1_score(y_test, y_pred_gb):.4f}")

# AdaBoost
ada = AdaBoostClassifier(n_estimators=200, learning_rate=0.5,
                         algorithm='SAMME', random_state=42)
ada.fit(X_train, y_train)
y_pred_ada = ada.predict(X_test)
print(f"\n  AdaBoost (n=200) — Accuracy : {ada.score(X_test, y_test):.4f}")

# Courbe deviance GB
test_score = np.zeros(gb_opt.n_estimators_)
for i, y_pred_stage in enumerate(gb_opt.staged_predict(X_test)):
    test_score[i] = accuracy_score(y_test, y_pred_stage)
train_score = gb_opt.train_score_

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(range(gb_opt.n_estimators_), train_score, '-',
        color=PALETTE["primary"], linewidth=2, label="Train")
ax.plot(range(gb_opt.n_estimators_), test_score, '-',
        color=PALETTE["secondary"], linewidth=2, label="Test")
ax.set_title("Gradient Boosting — Évolution de l'accuracy par itération")
ax.set_xlabel("Nombre d'arbres (itérations)")
ax.set_ylabel("Accuracy")
ax.legend()
plt.tight_layout()
save_fig("08_gb_learning_curve", fig)


# ─────────────────────────────────────────────────────────────────
# 8. AUTRES CLASSIFIEURS
# ─────────────────────────────────────────────────────────────────
section("COMPARAISON DE CLASSIFIEURS SUPPLÉMENTAIRES", "🔍")

other_models = {
    'KNN (k=5)':   KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
    'KNN (k=15)':  KNeighborsClassifier(n_neighbors=15, n_jobs=-1),
    'Régression Logistique': LogisticRegression(max_iter=1000, C=1.0, random_state=42),
    'Naive Bayes': GaussianNB(),
}

for name, model in other_models.items():
    t0 = time.time()
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)
    f1  = f1_score(y_test, model.predict(X_test))
    print(f"  {name:<30s} | Accuracy={acc:.4f} | F1={f1:.4f} | {time.time()-t0:.1f}s")


# ─────────────────────────────────────────────────────────────────
# 9. TABLEAU COMPARATIF COMPLET
# ─────────────────────────────────────────────────────────────────
section("TABLEAU COMPARATIF DES MODÈLES", "📋")

all_models = {
    'Classifieur constant':     dummy,
    'Arbre (depth=3)':          dt_simple,
    'Arbre optimisé':           dt_opt,
    'Random Forest (200 arbres)': rf_opt,
    'Gradient Boosting (opt.)': gb_opt,
    'AdaBoost':                 ada,
    'Régression Logistique':    other_models['Régression Logistique'],
    'KNN (k=5)':                other_models['KNN (k=5)'],
    'Naive Bayes':              other_models['Naive Bayes'],
}

results = []
for name, model in all_models.items():
    y_pred = model.predict(X_test)
    results.append({
        'Modèle':    name,
        'Accuracy':  round(accuracy_score(y_test, y_pred), 4),
        'F1-Score':  round(f1_score(y_test, y_pred), 4),
        'Précision': round(precision_score(y_test, y_pred), 4),
        'Rappel':    round(recall_score(y_test, y_pred), 4),
    })

df_results = pd.DataFrame(results).sort_values('F1-Score', ascending=False)
print("\n" + df_results.to_string(index=False))

# Visualisation comparative
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Comparaison des Modèles de Classification",
             fontsize=14, fontweight='bold', color=PALETTE["dark"])

metrics = ['Accuracy', 'F1-Score', 'Précision', 'Rappel']
colors_models = plt.cm.tab10(np.linspace(0, 1, len(df_results)))

# Barres horizontales — Accuracy
ax = axes[0]
y_pos = range(len(df_results))
bars = ax.barh(y_pos, df_results['Accuracy'],
               color=colors_models, edgecolor='white', linewidth=0.7, height=0.65)
ax.set_yticks(y_pos)
ax.set_yticklabels(df_results['Modèle'], fontsize=9)
ax.axvline(dummy_acc, color=PALETTE["danger"], linestyle='--',
           linewidth=1.5, label='Baseline')
for bar, val in zip(bars, df_results['Accuracy']):
    ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
            f'{val:.4f}', va='center', fontsize=8.5, fontweight='bold')
ax.set_title("Accuracy (test set)")
ax.set_xlabel("Accuracy")
ax.legend(fontsize=8)
ax.set_xlim(0, 1.05)

# Multi-métriques radar → grouped bar
ax = axes[1]
x = np.arange(4)
w = 0.07
top5 = df_results.head(5)
for i, (_, row) in enumerate(top5.iterrows()):
    vals = [row[m] for m in metrics]
    ax.bar(x + i*w, vals, w, label=row['Modèle'],
           color=colors_models[i], edgecolor='white', linewidth=0.5)
ax.set_xticks(x + 2*w)
ax.set_xticklabels(metrics, fontsize=10)
ax.set_title("Multi-métriques — Top 5 modèles")
ax.set_ylabel("Score")
ax.set_ylim(0, 1.1)
ax.legend(fontsize=7.5, loc='lower right')

plt.tight_layout()
save_fig("09_model_comparison", fig)


# ─────────────────────────────────────────────────────────────────
# 10. MATRICES DE CONFUSION
# ─────────────────────────────────────────────────────────────────
section("MATRICES DE CONFUSION", "🎯")

top_models = {
    'Decision Tree (opt.)':   dt_opt,
    'Random Forest':          rf_opt,
    'Gradient Boosting':      gb_opt,
    'Régression Logistique':  other_models['Régression Logistique'],
}

fig, axes = plt.subplots(1, 4, figsize=(20, 5))
fig.suptitle("Matrices de Confusion — Top modèles",
             fontsize=14, fontweight='bold')

for ax, (name, model) in zip(axes, top_models.items()):
    cm = confusion_matrix(y_test, model.predict(X_test))
    disp = ConfusionMatrixDisplay(cm, display_labels=['≤50K', '>50K'])
    disp.plot(ax=ax, colorbar=False, cmap=CMAP_MAIN)
    acc = accuracy_score(y_test, model.predict(X_test))
    ax.set_title(f"{name}\nAcc={acc:.3f}", fontsize=10, fontweight='bold')
    ax.set_xlabel("Prédit")
    ax.set_ylabel("Réel")

plt.tight_layout()
save_fig("10_confusion_matrices", fig)


# ─────────────────────────────────────────────────────────────────
# 11. COURBES ROC COMPARATIVES
# ─────────────────────────────────────────────────────────────────
section("COURBES ROC & AUC", "📈")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle("Évaluation ROC — Classifieurs Optimisés",
             fontsize=14, fontweight='bold', color=PALETTE["dark"])

roc_models = {**top_models, 'KNN (k=5)': other_models['KNN (k=5)'],
              'Naive Bayes': other_models['Naive Bayes']}
colors_roc  = [PALETTE["primary"], PALETTE["secondary"],
               PALETTE["accent"], PALETTE["success"],
               PALETTE["danger"], PALETTE["neutral"]]

auc_results = {}
for (name, model), col in zip(roc_models.items(), colors_roc):
    if hasattr(model, 'predict_proba'):
        scores = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, 'decision_function'):
        scores = model.decision_function(X_test)
    else:
        continue
    fpr, tpr, _ = roc_curve(y_test, scores)
    roc_auc      = auc(fpr, tpr)
    auc_results[name] = roc_auc
    ax1.plot(fpr, tpr, color=col, linewidth=2.2,
             label=f"{name}  (AUC = {roc_auc:.3f})")

ax1.plot([0,1],[0,1],'k--', linewidth=1.2, alpha=0.6, label='Aléatoire (AUC=0.5)')
ax1.fill_between([0,1],[0,1], alpha=0.03, color='k')
ax1.set_xlabel("Taux de Faux Positifs (FPR)")
ax1.set_ylabel("Taux de Vrais Positifs (TPR)")
ax1.set_title("Courbes ROC Comparatives")
ax1.legend(fontsize=8, loc='lower right')
ax1.set_xlim([-0.01, 1.01])
ax1.set_ylim([-0.01, 1.05])

# Barres AUC
auc_sorted = dict(sorted(auc_results.items(), key=lambda x: x[1], reverse=True))
bars = ax2.barh(list(auc_sorted.keys()), list(auc_sorted.values()),
                color=colors_roc[:len(auc_sorted)],
                edgecolor='white', linewidth=0.7)
for bar, val in zip(bars, auc_sorted.values()):
    ax2.text(bar.get_width() + 0.002,
             bar.get_y() + bar.get_height()/2,
             f'{val:.4f}', va='center', fontsize=9, fontweight='bold')
ax2.axvline(0.5, color=PALETTE["neutral"], linestyle='--',
            linewidth=1.5, label='Aléatoire')
ax2.set_title("AUC par modèle")
ax2.set_xlabel("AUC")
ax2.set_xlim(0.4, 1.05)
ax2.legend(fontsize=8)

plt.tight_layout()
save_fig("11_roc_curves", fig)


# ─────────────────────────────────────────────────────────────────
# 12. IMPORTANCE DES VARIABLES
# ─────────────────────────────────────────────────────────────────
section("IMPORTANCE DES VARIABLES", "🏆")

fi_rf = pd.DataFrame({'feature': feature_cols,
                      'RF':      rf_opt.feature_importances_,
                      'GB':      gb_opt.feature_importances_,
                      'DT':      dt_opt.feature_importances_}).sort_values('RF', ascending=False)

print("\n  Top 10 variables (Random Forest):")
for _, row in fi_rf.head(10).iterrows():
    bar = "█" * int(row['RF'] * 500)
    print(f"  {row['feature']:<40s} RF={row['RF']:.4f}  {bar}")

TOP_N = 15
fig, axes = plt.subplots(1, 3, figsize=(20, 8))
fig.suptitle("Importance des Variables — Comparaison des Modèles",
             fontsize=14, fontweight='bold', color=PALETTE["dark"])

for ax, (metric, col, name) in zip(axes, [
        ('RF', PALETTE["primary"], "Random Forest"),
        ('GB', PALETTE["secondary"], "Gradient Boosting"),
        ('DT', PALETTE["accent"], "Decision Tree"),
]):
    top = fi_rf.nlargest(TOP_N, metric)
    feat_clean = top['feature'].str.replace('_', ' ').str[:35]
    bars = ax.barh(feat_clean[::-1], top[metric][::-1],
                   color=col, alpha=0.85, edgecolor='white', linewidth=0.5)
    for bar in bars:
        ax.text(bar.get_width() + 0.0005,
                bar.get_y() + bar.get_height()/2,
                f'{bar.get_width():.4f}', va='center', fontsize=7.5)
    ax.set_title(f"Top {TOP_N} — {name}", fontweight='bold')
    ax.set_xlabel("Importance")

plt.tight_layout()
save_fig("12_feature_importance", fig)

# Heatmap croisée des importances
top15_names = fi_rf.head(15)['feature'].tolist()
fi_matrix   = fi_rf[fi_rf['feature'].isin(top15_names)].set_index('feature')[['RF','GB','DT']]
fi_matrix.columns = ['Random Forest', 'Gradient Boosting', 'Decision Tree']
fi_norm = fi_matrix.div(fi_matrix.max())

fig, ax = plt.subplots(figsize=(10, 7))
sns.heatmap(fi_norm, annot=fi_matrix.round(4), fmt='.4f',
            cmap=CMAP_MAIN, ax=ax, linewidths=0.4, linecolor='#DEE2E6',
            annot_kws={"size": 8})
ax.set_title("Importance Normalisée — Top 15 Variables (3 modèles)",
             fontsize=12, fontweight='bold')
ax.set_ylabel("")
ax.tick_params(axis='x', labelsize=10)
plt.tight_layout()
save_fig("13_feature_importance_heatmap", fig)


# ─────────────────────────────────────────────────────────────────
# 13. LEARNING CURVES
# ─────────────────────────────────────────────────────────────────
section("COURBES D'APPRENTISSAGE", "📉")

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("Courbes d'Apprentissage — Diagnostic Biais/Variance",
             fontsize=14, fontweight='bold')

lc_models = [
    (dt_opt, "Decision Tree (opt.)", PALETTE["accent"]),
    (rf_opt, "Random Forest",        PALETTE["primary"]),
    (gb_opt, "Gradient Boosting",    PALETTE["secondary"]),
]

for ax, (model, name, col) in zip(axes, lc_models):
    train_sizes, train_scores, val_scores = learning_curve(
        model, X_train, y_train,
        train_sizes=np.linspace(0.1, 1.0, 8),
        cv=StratifiedKFold(3, shuffle=True, random_state=42),
        scoring='accuracy', n_jobs=-1)

    tr_mean = train_scores.mean(1)
    tr_std  = train_scores.std(1)
    va_mean = val_scores.mean(1)
    va_std  = val_scores.std(1)

    ax.plot(train_sizes, tr_mean, 'o-', color=col,
            linewidth=2.5, markersize=6, label="Entraînement")
    ax.fill_between(train_sizes, tr_mean-tr_std, tr_mean+tr_std,
                    alpha=0.15, color=col)
    ax.plot(train_sizes, va_mean, 's--', color=PALETTE["danger"],
            linewidth=2.5, markersize=6, label="Validation")
    ax.fill_between(train_sizes, va_mean-va_std, va_mean+va_std,
                    alpha=0.15, color=PALETTE["danger"])
    ax.set_title(name)
    ax.set_xlabel("Taille du jeu d'entraînement")
    ax.set_ylabel("Accuracy")
    ax.legend(fontsize=8)
    ax.set_ylim(0.7, 1.01)

plt.tight_layout()
save_fig("14_learning_curves", fig)


# ─────────────────────────────────────────────────────────────────
# 14. RAPPORT DE CLASSIFICATION DÉTAILLÉ
# ─────────────────────────────────────────────────────────────────
section("RAPPORT DE CLASSIFICATION — MEILLEUR MODÈLE (RF)", "📄")

y_pred_best = rf_opt.predict(X_test)
print(classification_report(y_test, y_pred_best,
                            target_names=le_income.classes_))


# ─────────────────────────────────────────────────────────────────
# 15. RÉSUMÉ FINAL
# ─────────────────────────────────────────────────────────────────
section("RÉSUMÉ FINAL", "✅")

best_model = df_results.iloc[0]
print(f"""
  ┌──────────────────────────────────────────────────────────┐
  │                 RÉSULTATS FINAUX                         │
  ├──────────────────────────────────────────────────────────┤
  │  Dataset                                                  │
  │    Instances totales    : {len(df):>8,}                       │
  │    Après nettoyage      : {len(df_clean):>8,}                       │
  │    Features encodées    : {len(feature_cols):>8}                       │
  │    Classe mino. (>50K)  : {income_dist['>50K']:>8,}  ({income_dist['>50K']/len(df)*100:.1f}%)             │
  ├──────────────────────────────────────────────────────────┤
  │  Meilleur modèle : {best_model['Modèle']:<38s}│
  │    Accuracy     : {best_model['Accuracy']:.4f}                              │
  │    F1-Score     : {best_model['F1-Score']:.4f}                              │
  │    Précision    : {best_model['Précision']:.4f}                              │
  │    Rappel       : {best_model['Rappel']:.4f}                              │
  │    AUC          : {auc_results.get(best_model['Modèle'], 0):.4f}                              │
  ├──────────────────────────────────────────────────────────┤
  │  Top 3 variables importantes (RF) :                       │
  │    1. {fi_rf.iloc[0]['feature']:<40s} │
  │    2. {fi_rf.iloc[1]['feature']:<40s} │
  │    3. {fi_rf.iloc[2]['feature']:<40s} │
  └──────────────────────────────────────────────────────────┘
""")

print(f"  📁 Toutes les figures sauvegardées dans → ./{OUTPUT_DIR}/")
print("  🎉 Analyse terminée avec succès!\n")