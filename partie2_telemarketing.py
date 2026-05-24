"""
╔══════════════════════════════════════════════════════════════════╗
║   INTELLIGENCE ARTIFICIELLE 2 - TP : Partie 2                   ║
║   Réseaux de Neurones & Deep Learning                           ║
║   Bank Telemarketing + Fashion MNIST                            ║
║   DIPES 2 | 4ème année (S2) | 2025-2026                        ║
║   Auteur : Stéphane C. K. TÉKOUABOU (PhD & Ing.)               ║
╚══════════════════════════════════════════════════════════════════╝
"""

# ─────────────────────────────────────────────────────────────────
# 0. IMPORTS & CONFIGURATION GLOBALE
# ─────────────────────────────────────────────────────────────────
import warnings
warnings.filterwarnings('ignore')
import os, time, pickle

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats

from sklearn.model_selection import (train_test_split, cross_val_score,
                                     GridSearchCV, StratifiedKFold)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             classification_report, roc_curve, auc,
                             precision_score, recall_score, f1_score,
                             ConfusionMatrixDisplay)
from sklearn.ensemble import (RandomForestClassifier,
                              GradientBoostingClassifier,
                              BaggingClassifier, AdaBoostClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier
from sklearn.inspection import permutation_importance

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import (Dense, Dropout, BatchNormalization,
                                     Input, Conv2D, MaxPooling2D,
                                     Flatten, GlobalAveragePooling2D)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.regularizers import l2
import tensorflow.keras.applications as keras_apps

# ── Palette & Style ──────────────────────────────────────────────
PALETTE = {
    "primary":   "#6C3FC5",
    "secondary": "#F7931E",
    "accent":    "#00B4D8",
    "success":   "#2DC653",
    "danger":    "#E63946",
    "neutral":   "#8D99AE",
    "bg":        "#F8F9FA",
    "dark":      "#1A1A2E",
}

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
    "font.family":       "DejaVu Sans",
    "figure.dpi":        130,
    "savefig.dpi":       180,
    "savefig.bbox":      "tight",
    "savefig.facecolor": PALETTE["bg"],
})

OUTPUT_DIR = "outputs_tp2"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def save_fig(name, fig=None):
    path = os.path.join(OUTPUT_DIR, f"{name}.png")
    (fig or plt).savefig(path)
    print(f"   💾 Figure sauvegardée → {path}")
    plt.show()
    plt.close("all")

def section(title, emoji=""):
    bar = "═" * 60
    print(f"\n{bar}")
    print(f"  {emoji}  {title}")
    print(f"{bar}")

def subsection(title):
    print(f"\n  ► {title}")
    print("  " + "─" * 55)

print("=" * 70)
print("  PARTIE 2 : RÉSEAUX DE NEURONES & DEEP LEARNING")
print("  Bank Telemarketing + Fashion MNIST")
print("=" * 70)
print(f"  TensorFlow version : {tf.__version__}")
print(f"  GPU disponible     : {len(tf.config.list_physical_devices('GPU')) > 0}")


# ═════════════════════════════════════════════════════════════════
# SOUS-PARTIE A : BANK TELEMARKETING
# ═════════════════════════════════════════════════════════════════
section("SOUS-PARTIE A : BANK TELEMARKETING", "🏦")

# ─────────────────────────────────────────────────────────────────
# 1. CHARGEMENT ET ANALYSE DES DONNÉES
# ─────────────────────────────────────────────────────────────────
section("1. CHARGEMENT ET ANALYSE DES DONNÉES", "📂")

try:
    df = pd.read_csv('bank-additional-full.csv', sep=';')
    print("✅ Fichier bank-additional-full.csv chargé avec succès!")
except FileNotFoundError:
    try:
        df = pd.read_csv('bank-additional.csv', sep=';')
        print("✅ Fichier bank-additional.csv chargé avec succès!")
    except FileNotFoundError:
        print("⚠️ Fichier non trouvé — Création de données synthétiques réalistes...")
        np.random.seed(42)
        n = 41188
        df = pd.DataFrame({
            'age':            np.random.randint(18, 95, n),
            'job':            np.random.choice(['admin.','blue-collar','entrepreneur',
                                                'housemaid','management','retired',
                                                'self-employed','services','student',
                                                'technician','unemployed','unknown'], n),
            'marital':        np.random.choice(['divorced','married','single','unknown'], n,
                                               p=[0.11, 0.61, 0.27, 0.01]),
            'education':      np.random.choice(['basic.4y','basic.6y','basic.9y',
                                                'high.school','illiterate',
                                                'professional.course',
                                                'university.degree','unknown'], n),
            'default':        np.random.choice(['no','yes','unknown'], n, p=[0.79,0.03,0.18]),
            'housing':        np.random.choice(['no','yes','unknown'], n, p=[0.44,0.53,0.03]),
            'loan':           np.random.choice(['no','yes','unknown'], n, p=[0.84,0.13,0.03]),
            'contact':        np.random.choice(['cellular','telephone'], n, p=[0.63,0.37]),
            'month':          np.random.choice(['jan','feb','mar','apr','may','jun',
                                                'jul','aug','sep','oct','nov','dec'], n),
            'day_of_week':    np.random.choice(['mon','tue','wed','thu','fri'], n),
            'duration':       np.abs(np.random.exponential(258, n)).astype(int),
            'campaign':       np.random.randint(1, 56, n),
            'pdays':          np.random.choice([999]+list(range(0,27)), n,
                                               p=[0.96]+[0.04/27]*27),
            'previous':       np.random.randint(0, 7, n),
            'poutcome':       np.random.choice(['failure','nonexistent','success'], n,
                                               p=[0.10, 0.86, 0.04]),
            'emp.var.rate':   np.round(np.random.uniform(-3.4, 1.4, n), 1),
            'cons.price.idx': np.round(np.random.uniform(92.2, 94.8, n), 3),
            'cons.conf.idx':  np.round(np.random.uniform(-50.8, -26.9, n), 1),
            'euribor3m':      np.round(np.random.uniform(0.63, 5.05, n), 3),
            'nr.employed':    np.round(np.random.uniform(4963.6, 5228.1, n), 1),
            'y':              np.random.choice(['no','yes'], n, p=[0.887, 0.113])
        })
        print("✅ Données synthétiques créées!")

subsection("Informations générales")
print(f"  Dimensions        : {df.shape}")
print(f"  Instances         : {len(df):,}")
print(f"  Caractéristiques  : {df.shape[1]}")

subsection("Distribution de la variable cible")
target_dist = df['y'].value_counts()
for k, v in target_dist.items():
    bar = "█" * int(v / len(df) * 50)
    print(f"  {k:>4s} : {v:>6,}  ({v/len(df)*100:.1f}%)  {bar}")

print(f"\n  → Déséquilibre de classes (ratio {target_dist['no']/target_dist['yes']:.1f}:1)")
print("  → Métriques adaptées : F1-Score, AUC-ROC, Recall (classe minoritaire)")

subsection("Types de variables")
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = [c for c in df.select_dtypes('object').columns if c != 'y']
print(f"  Numériques ({len(numeric_cols)})     : {numeric_cols}")
print(f"  Catégorielles ({len(categorical_cols)})  : {categorical_cols}")

subsection("Statistiques descriptives")
print(df[numeric_cols].describe().round(2).to_string())

# ── Figure 1 : Vue d'ensemble ─────────────────────────────────
fig = plt.figure(figsize=(18, 12))
fig.suptitle("Bank Telemarketing — Analyse Exploratoire",
             fontsize=16, fontweight='bold', y=1.01)
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

# Pie cible
ax0 = fig.add_subplot(gs[0, 0])
ax0.pie(target_dist.values,
        labels=target_dist.index, autopct='%1.1f%%', startangle=90,
        colors=[PALETTE["neutral"], PALETTE["primary"]],
        wedgeprops=dict(edgecolor='white', linewidth=2),
        explode=(0, 0.05), shadow=True)
for t in ax0.texts[2::3]:
    t.set_fontweight('bold')
ax0.set_title("Distribution cible")

# Âge
ax1 = fig.add_subplot(gs[0, 1])
for val, col in zip(['no', 'yes'], [PALETTE["neutral"], PALETTE["primary"]]):
    df[df['y'] == val]['age'].plot.hist(ax=ax1, alpha=0.6, bins=30,
                                        color=col, label=val, edgecolor='white')
ax1.set_title("Distribution Age par classe")
ax1.set_xlabel("Age")
ax1.legend(title="Souscrit")

# Durée d'appel
ax2 = fig.add_subplot(gs[0, 2])
df[df['duration'] < df['duration'].quantile(0.99)]['duration'].plot.hist(
    ax=ax2, bins=40, color=PALETTE["secondary"], alpha=0.8, edgecolor='white')
ax2.axvline(df['duration'].mean(), color=PALETTE["danger"], linestyle='--',
            linewidth=2, label=f"Moy={df['duration'].mean():.0f}s")
ax2.set_title("Distribution Durée d'appel")
ax2.set_xlabel("Durée (s)")
ax2.legend()

# Taux de souscription par job
ax3 = fig.add_subplot(gs[1, :2])
job_rate = (df.groupby('job')['y']
             .apply(lambda x: (x == 'yes').mean() * 100)
             .sort_values(ascending=False))
colors_job = [PALETTE["primary"] if v > job_rate.mean() else PALETTE["neutral"]
              for v in job_rate.values]
ax3.bar(job_rate.index, job_rate.values, color=colors_job, edgecolor='white')
ax3.axhline(job_rate.mean(), color=PALETTE["danger"], linestyle='--',
            linewidth=1.5, label=f"Moy={job_rate.mean():.1f}%")
ax3.set_xticklabels(job_rate.index, rotation=35, ha='right', fontsize=9)
ax3.set_title("Taux de souscription par type d'emploi (%)")
ax3.set_ylabel("% souscription")
ax3.legend()

# Education vs souscription
ax4 = fig.add_subplot(gs[1, 2])
edu_rate = (df.groupby('education')['y']
              .apply(lambda x: (x == 'yes').mean() * 100)
              .sort_values(ascending=True))
ax4.barh(edu_rate.index, edu_rate.values,
         color=[PALETTE["primary"] if v > edu_rate.mean() else PALETTE["neutral"]
                for v in edu_rate.values], edgecolor='white')
ax4.axvline(edu_rate.mean(), color=PALETTE["danger"], linestyle='--')
ax4.set_title("Souscription par éducation")
ax4.set_xlabel("% souscription")

# Corrélation numérique
ax5 = fig.add_subplot(gs[2, :])
corr = df[numeric_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
from matplotlib.colors import LinearSegmentedColormap
cmap_divg = LinearSegmentedColormap.from_list(
    "divg", [PALETTE["accent"], "white", PALETTE["secondary"]])
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f',
            cmap=cmap_divg, center=0, vmin=-1, vmax=1,
            ax=ax5, linewidths=0.5, annot_kws={"size": 8},
            cbar_kws={"shrink": 0.6})
ax5.set_title("Matrice de corrélation (variables numériques)")
ax5.tick_params(axis='x', rotation=30, labelsize=8)

save_fig("01_analyse_exploratoire", fig)

# ── Nuages de points (scatter) ────────────────────────────────
subsection("Nuages de points croisés avec droites de régression")
pair_cols = ['age', 'duration', 'campaign', 'euribor3m', 'nr.employed', 'y']
pair_df = df[pair_cols].copy()
pair_df['duration'] = pair_df['duration'].clip(upper=pair_df['duration'].quantile(0.99))

g = sns.pairplot(pair_df, hue='y',
                 palette={'yes': PALETTE["primary"], 'no': PALETTE["neutral"]},
                 diag_kind='kde', plot_kws=dict(alpha=0.3, s=12),
                 diag_kws=dict(linewidth=2))
g.figure.suptitle("Nuages de points croisés — Variables clés",
                  y=1.02, fontsize=14, fontweight='bold')

num_pair = ['age', 'duration', 'campaign', 'euribor3m', 'nr.employed']
for i, col_y in enumerate(num_pair):
    for j, col_x in enumerate(num_pair):
        if i != j:
            ax = g.axes[i][j]
            for val, col in zip(['yes', 'no'],
                                [PALETTE["primary"], PALETTE["neutral"]]):
                sub = df[df['y'] == val]
                m, b, r, p, _ = stats.linregress(sub[col_x], sub[col_y])
                x_r = np.linspace(sub[col_x].min(), sub[col_x].max(), 100)
                ax.plot(x_r, m * x_r + b, color=col, linewidth=1.5,
                        linestyle='--', alpha=0.8)
                ax.text(0.05, 0.95 if val == 'yes' else 0.85,
                        f'r={r:.2f}', transform=ax.transAxes,
                        fontsize=6, color=col, va='top')
save_fig("02_pairplot", g.figure)


# ─────────────────────────────────────────────────────────────────
# 2. PRÉTRAITEMENT
# ─────────────────────────────────────────────────────────────────
section("2. PRÉTRAITEMENT DES DONNÉES", "⚙️")

df_clean = df.copy()

# Suppression des 'unknown' pour variables importantes
important_cols = ['job', 'education', 'contact']
before = len(df_clean)
for col in important_cols:
    if col in df_clean.columns:
        df_clean = df_clean[df_clean[col] != 'unknown']
print(f"  Instances avant : {before:,}")
print(f"  Instances après : {len(df_clean):,}  "
      f"({(before-len(df_clean))/before*100:.1f}% supprimés)")

# Encodage cible
le_target = LabelEncoder()
df_clean['y_encoded'] = le_target.fit_transform(df_clean['y'])
print(f"  Encodage cible : {dict(zip(le_target.classes_, le_target.transform(le_target.classes_)))}")

# One-hot encoding
cat_enc = [c for c in df_clean.select_dtypes('object').columns if c != 'y']
df_enc = pd.get_dummies(df_clean, columns=cat_enc, drop_first=True)

feature_cols = [c for c in df_enc.columns if c not in ['y', 'y_encoded']]
X = df_enc[feature_cols].values
y_bank = df_enc['y_encoded'].values

print(f"  Features après encodage : {len(feature_cols)}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y_bank, test_size=0.2, random_state=42, stratify=y_bank)
print(f"  Train : {len(X_train):,}  |  Test : {len(X_test):,}")
print(f"  Distribution train : {np.bincount(y_train)}")
print(f"  Distribution test  : {np.bincount(y_test)}")

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)
print("  ✅ Standardisation effectuée")


# ─────────────────────────────────────────────────────────────────
# 3. CLASSIFIEUR CONSTANT (BASELINE)
# ─────────────────────────────────────────────────────────────────
section("3. CLASSIFIEUR CONSTANT (BASELINE)", "📏")

dummy = DummyClassifier(strategy='most_frequent', random_state=42)
dummy.fit(X_train_sc, y_train)
dummy_acc = accuracy_score(y_test, dummy.predict(X_test_sc))
print(f"  Accuracy baseline : {dummy_acc:.4f}  (classe majoritaire 'no')")
print(f"  → Tout classifieur utile doit dépasser {dummy_acc:.2%}")


# ─────────────────────────────────────────────────────────────────
# 4. MÉTHODES CLASSIQUES (DT, Bagging, RF, GB, AdaBoost, KNN, LR, NB)
# ─────────────────────────────────────────────────────────────────
section("4. CLASSIFIEURS CLASSIQUES (PARTIE 1 DU TP)", "🌳")

cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

subsection("Arbre de décision — GridSearch")
param_dt = {'max_depth': [3, 5, 7, 10, 15, None],
            'min_samples_split': [2, 5, 10],
            'criterion': ['gini', 'entropy']}
dt_gs = GridSearchCV(DecisionTreeClassifier(random_state=42),
                     param_dt, cv=cv5, scoring='f1', n_jobs=-1)
dt_gs.fit(X_train_sc, y_train)
dt_opt = dt_gs.best_estimator_
print(f"  Meilleurs params DT : {dt_gs.best_params_}")
print(f"  Accuracy DT opt     : {accuracy_score(y_test, dt_opt.predict(X_test_sc)):.4f}")

subsection("Random Forest (OOB + GridSearch)")
rf_oob = RandomForestClassifier(n_estimators=200, oob_score=True,
                                max_features='sqrt', random_state=42, n_jobs=-1)
rf_oob.fit(X_train_sc, y_train)
print(f"  OOB Score : {rf_oob.oob_score_:.4f}")

rf_opt = RandomForestClassifier(n_estimators=200, max_depth=15,
                                max_features='sqrt', random_state=42, n_jobs=-1)
rf_opt.fit(X_train_sc, y_train)
print(f"  RF opt — Accuracy : {accuracy_score(y_test, rf_opt.predict(X_test_sc)):.4f}")
print(f"  RF opt — F1       : {f1_score(y_test, rf_opt.predict(X_test_sc)):.4f}")

subsection("Gradient Boosting (Early Stopping)")
gb_opt = GradientBoostingClassifier(
    n_estimators=200, learning_rate=0.1, max_depth=4,
    subsample=0.8, min_samples_leaf=10,
    validation_fraction=0.15, n_iter_no_change=15, tol=1e-4,
    random_state=42)
gb_opt.fit(X_train_sc, y_train)
print(f"  GB — arbres utilisés : {gb_opt.n_estimators_}")
print(f"  GB — Accuracy        : {accuracy_score(y_test, gb_opt.predict(X_test_sc)):.4f}")
print(f"  GB — F1              : {f1_score(y_test, gb_opt.predict(X_test_sc)):.4f}")

subsection("AdaBoost")
ada = AdaBoostClassifier(n_estimators=200, learning_rate=0.5,
                         algorithm='SAMME', random_state=42)
ada.fit(X_train_sc, y_train)
print(f"  AdaBoost — Accuracy : {accuracy_score(y_test, ada.predict(X_test_sc)):.4f}")

subsection("Autres classifieurs")
other_clf = {
    'KNN (k=5)':             KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
    'KNN (k=15)':            KNeighborsClassifier(n_neighbors=15, n_jobs=-1),
    'Régression Logistique': LogisticRegression(max_iter=1000, C=1.0, random_state=42),
    'Naive Bayes':           GaussianNB(),
}
for name, clf in other_clf.items():
    clf.fit(X_train_sc, y_train)
    acc = accuracy_score(y_test, clf.predict(X_test_sc))
    f1  = f1_score(y_test, clf.predict(X_test_sc))
    print(f"  {name:<30s} | Acc={acc:.4f} | F1={f1:.4f}")


# ─────────────────────────────────────────────────────────────────
# 5. RÉSEAU DE NEURONES SIMPLE
# ─────────────────────────────────────────────────────────────────
section("5. RÉSEAU DE NEURONES SIMPLE", "🧠")

n_features = X_train_sc.shape[1]

model_simple = Sequential([
    Input(shape=(n_features,)),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
], name="reseau_simple")

model_simple.compile(optimizer=Adam(learning_rate=0.001),
                     loss='binary_crossentropy',
                     metrics=['accuracy'])

print("\n  Architecture — Réseau Simple :")
model_simple.summary()

early_stop = EarlyStopping(monitor='val_loss', patience=10,
                           restore_best_weights=True, verbose=0)

t0 = time.time()
history_simple = model_simple.fit(
    X_train_sc, y_train,
    validation_split=0.2,
    epochs=100, batch_size=32,
    callbacks=[early_stop], verbose=0)
print(f"\n  Entraînement terminé en {time.time()-t0:.1f}s "
      f"({len(history_simple.history['loss'])} epochs)")

y_pred_simple_proba = model_simple.predict(X_test_sc, verbose=0).ravel()
y_pred_simple = (y_pred_simple_proba >= 0.5).astype(int)
acc_simple = accuracy_score(y_test, y_pred_simple)
f1_simple  = f1_score(y_test, y_pred_simple)
print(f"  Accuracy : {acc_simple:.4f}  |  F1 : {f1_simple:.4f}")
print("\n  Rapport de classification :")
print(classification_report(y_test, y_pred_simple, target_names=['no', 'yes']))


# ─────────────────────────────────────────────────────────────────
# 6. RÉSEAU DE NEURONES PROFOND OPTIMISÉ
# ─────────────────────────────────────────────────────────────────
section("6. RÉSEAU DE NEURONES PROFOND OPTIMISÉ", "🚀")

model_deep = Sequential([
    Input(shape=(n_features,)),

    Dense(256, activation='relu', kernel_regularizer=l2(1e-4)),
    BatchNormalization(),
    Dropout(0.4),

    Dense(128, activation='relu', kernel_regularizer=l2(1e-4)),
    BatchNormalization(),
    Dropout(0.3),

    Dense(64, activation='relu', kernel_regularizer=l2(1e-4)),
    BatchNormalization(),
    Dropout(0.3),

    Dense(32, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),

    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
], name="reseau_profond")

model_deep.compile(
    optimizer=Adam(learning_rate=0.0005),
    loss='binary_crossentropy',
    metrics=['accuracy',
             tf.keras.metrics.Precision(name='precision'),
             tf.keras.metrics.Recall(name='recall'),
             tf.keras.metrics.AUC(name='auc')])

print("\n  Architecture — Réseau Profond :")
model_deep.summary()

callbacks_deep = [
    EarlyStopping(monitor='val_auc', patience=15,
                  restore_best_weights=True, mode='max', verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7,
                      min_lr=1e-6, verbose=1),
    ModelCheckpoint(os.path.join(OUTPUT_DIR, 'best_deep_model.keras'),
                    save_best_only=True, monitor='val_auc',
                    mode='max', verbose=0)
]

t0 = time.time()
history_deep = model_deep.fit(
    X_train_sc, y_train,
    validation_split=0.2,
    epochs=200, batch_size=64,
    callbacks=callbacks_deep, verbose=1)
print(f"\n  Entraînement terminé en {time.time()-t0:.1f}s "
      f"({len(history_deep.history['loss'])} epochs)")

y_pred_deep_proba = model_deep.predict(X_test_sc, verbose=0).ravel()
y_pred_deep = (y_pred_deep_proba >= 0.5).astype(int)
acc_deep  = accuracy_score(y_test, y_pred_deep)
prec_deep = precision_score(y_test, y_pred_deep)
rec_deep  = recall_score(y_test, y_pred_deep)
f1_deep   = f1_score(y_test, y_pred_deep)

print(f"\n  Réseau profond — Accuracy  : {acc_deep:.4f}")
print(f"                  Precision : {prec_deep:.4f}")
print(f"                  Recall    : {rec_deep:.4f}")
print(f"                  F1-Score  : {f1_deep:.4f}")
print("\n  Rapport de classification :")
print(classification_report(y_test, y_pred_deep, target_names=['no', 'yes']))


# ─────────────────────────────────────────────────────────────────
# 7. IMPORTANCE DES VARIABLES & SÉLECTION TOP-10
# ─────────────────────────────────────────────────────────────────
section("7. SÉLECTION DES 10 MEILLEURES VARIABLES", "🏆")

fi_df = pd.DataFrame({
    'feature':    feature_cols,
    'RF':         rf_opt.feature_importances_,
    'GB':         gb_opt.feature_importances_,
    'DT':         dt_opt.feature_importances_,
}).sort_values('RF', ascending=False)

print("\n  Top 15 variables importantes (Random Forest) :")
for i, row in fi_df.head(15).iterrows():
    bar = "█" * int(row['RF'] * 400)
    print(f"  {row['feature']:<40s}  RF={row['RF']:.4f}  {bar}")

top10_features = fi_df.head(10)['feature'].tolist()
print(f"\n  → Top 10 retenues : {top10_features}")

top10_idx = [list(feature_cols).index(f) for f in top10_features]
X_train_t10 = X_train_sc[:, top10_idx]
X_test_t10  = X_test_sc[:,  top10_idx]

# Modèle ANN avec top-10 variables
model_top10 = Sequential([
    Input(shape=(10,)),
    Dense(64, activation='relu', kernel_regularizer=l2(1e-4)),
    BatchNormalization(),
    Dropout(0.3),
    Dense(32, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
], name="reseau_top10")

model_top10.compile(optimizer=Adam(learning_rate=0.001),
                    loss='binary_crossentropy',
                    metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])

print("\n  Entraînement du modèle Top-10 variables...")
history_t10 = model_top10.fit(
    X_train_t10, y_train,
    validation_split=0.2,
    epochs=100, batch_size=32,
    callbacks=[EarlyStopping(monitor='val_auc', patience=10,
                             restore_best_weights=True, mode='max')],
    verbose=0)

y_pred_t10_proba = model_top10.predict(X_test_t10, verbose=0).ravel()
y_pred_t10 = (y_pred_t10_proba >= 0.5).astype(int)
acc_t10 = accuracy_score(y_test, y_pred_t10)
f1_t10  = f1_score(y_test, y_pred_t10)
print(f"  Top-10 vars — Accuracy : {acc_t10:.4f}  |  F1 : {f1_t10:.4f}")


# ─────────────────────────────────────────────────────────────────
# 8. TABLEAU COMPARATIF COMPLET
# ─────────────────────────────────────────────────────────────────
section("8. TABLEAU COMPARATIF DE TOUS LES MODÈLES", "📋")

all_models = {
    'Classifieur constant':    dummy,
    'Arbre optimisé':          dt_opt,
    'Random Forest':           rf_opt,
    'Gradient Boosting':       gb_opt,
    'AdaBoost':                ada,
    'Régression Logistique':   other_clf['Régression Logistique'],
    'KNN (k=5)':               other_clf['KNN (k=5)'],
    'Naive Bayes':             other_clf['Naive Bayes'],
}

results = []
for name, model in all_models.items():
    yp = model.predict(X_test_sc)
    results.append({'Modèle': name,
                    'Accuracy':  round(accuracy_score(y_test, yp), 4),
                    'F1-Score':  round(f1_score(y_test, yp), 4),
                    'Précision': round(precision_score(y_test, yp), 4),
                    'Rappel':    round(recall_score(y_test, yp), 4)})

# ANN séparément
for ann_name, ann_pred, ann_proba in [
        ('Réseau Simple',     y_pred_simple, y_pred_simple_proba),
        ('Réseau Profond',    y_pred_deep,   y_pred_deep_proba),
        ('Réseau Top-10 vars',y_pred_t10,    y_pred_t10_proba),
]:
    results.append({'Modèle':    ann_name,
                    'Accuracy':  round(accuracy_score(y_test, ann_pred), 4),
                    'F1-Score':  round(f1_score(y_test, ann_pred), 4),
                    'Précision': round(precision_score(y_test, ann_pred), 4),
                    'Rappel':    round(recall_score(y_test, ann_pred), 4)})

df_results = pd.DataFrame(results).sort_values('F1-Score', ascending=False)
print("\n" + df_results.to_string(index=False))

# ── Figure : Comparaison modèles ──────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(18, 7))
fig.suptitle("Comparaison de tous les Modèles — Bank Telemarketing",
             fontsize=14, fontweight='bold')
colors_m = plt.cm.tab10(np.linspace(0, 1, len(df_results)))

ax = axes[0]
bars = ax.barh(range(len(df_results)), df_results['F1-Score'],
               color=colors_m, edgecolor='white', height=0.65)
ax.set_yticks(range(len(df_results)))
ax.set_yticklabels(df_results['Modèle'], fontsize=9)
ax.axvline(f1_score(y_test, dummy.predict(X_test_sc)),
           color=PALETTE["danger"], linestyle='--', label='Baseline')
for bar, val in zip(bars, df_results['F1-Score']):
    ax.text(bar.get_width()+0.002, bar.get_y()+bar.get_height()/2,
            f'{val:.4f}', va='center', fontsize=8.5, fontweight='bold')
ax.set_title("F1-Score (classe minoritaire 'yes')")
ax.set_xlabel("F1-Score")
ax.legend(fontsize=8)
ax.set_xlim(0, 0.85)

ax = axes[1]
metrics = ['Accuracy', 'F1-Score', 'Précision', 'Rappel']
x_pos = np.arange(4)
w = 0.07
top5 = df_results.head(5)
for i, (_, row) in enumerate(top5.iterrows()):
    ax.bar(x_pos + i*w, [row[m] for m in metrics], w,
           label=row['Modèle'], color=colors_m[i], edgecolor='white')
ax.set_xticks(x_pos + 2*w)
ax.set_xticklabels(metrics, fontsize=10)
ax.set_title("Multi-métriques — Top 5 modèles")
ax.set_ylabel("Score")
ax.set_ylim(0, 1.1)
ax.legend(fontsize=7, loc='lower right')

plt.tight_layout()
save_fig("03_model_comparison", fig)


# ─────────────────────────────────────────────────────────────────
# 9. MATRICES DE CONFUSION
# ─────────────────────────────────────────────────────────────────
section("9. MATRICES DE CONFUSION", "🎯")

top4_models = [
    ('Decision Tree',    dt_opt.predict(X_test_sc)),
    ('Random Forest',    rf_opt.predict(X_test_sc)),
    ('Gradient Boosting',gb_opt.predict(X_test_sc)),
    ('Réseau Profond',   y_pred_deep),
]

fig, axes = plt.subplots(1, 4, figsize=(20, 5))
fig.suptitle("Matrices de Confusion — Top Modèles",
             fontsize=14, fontweight='bold')
cmap_main = plt.cm.Purples

for ax, (name, yp) in zip(axes, top4_models):
    cm = confusion_matrix(y_test, yp)
    disp = ConfusionMatrixDisplay(cm, display_labels=['no', 'yes'])
    disp.plot(ax=ax, colorbar=False, cmap=cmap_main)
    acc = accuracy_score(y_test, yp)
    f1  = f1_score(y_test, yp)
    ax.set_title(f"{name}\nAcc={acc:.3f} | F1={f1:.3f}", fontsize=10)
    ax.set_xlabel("Prédit")
    ax.set_ylabel("Réel")

plt.tight_layout()
save_fig("04_confusion_matrices", fig)


# ─────────────────────────────────────────────────────────────────
# 10. COURBES ROC & AUC
# ─────────────────────────────────────────────────────────────────
section("10. COURBES ROC & AUC", "📈")

roc_models = {
    'Decision Tree (opt.)':  (dt_opt, dt_opt.predict_proba(X_test_sc)[:, 1]),
    'Random Forest':         (rf_opt, rf_opt.predict_proba(X_test_sc)[:, 1]),
    'Gradient Boosting':     (gb_opt, gb_opt.predict_proba(X_test_sc)[:, 1]),
    'AdaBoost':              (ada,    ada.predict_proba(X_test_sc)[:, 1]),
    'Réseau Simple':         (None,   y_pred_simple_proba),
    'Réseau Profond':        (None,   y_pred_deep_proba),
    'Réseau Top-10':         (None,   y_pred_t10_proba),
}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle("Courbes ROC — Tous modèles", fontsize=14, fontweight='bold')

colors_roc = [PALETTE["primary"], PALETTE["secondary"], PALETTE["accent"],
              PALETTE["success"], PALETTE["danger"], "#9B59B6", "#E67E22"]
auc_results = {}

for (name, (_, scores)), col in zip(roc_models.items(), colors_roc):
    fpr, tpr, _ = roc_curve(y_test, scores)
    roc_auc = auc(fpr, tpr)
    auc_results[name] = roc_auc
    ax1.plot(fpr, tpr, color=col, linewidth=2.2,
             label=f"{name}  (AUC={roc_auc:.3f})")

ax1.plot([0,1],[0,1],'k--', linewidth=1.2, alpha=0.6, label='Aléatoire')
ax1.fill_between([0,1],[0,1], alpha=0.03, color='k')
ax1.set_xlabel("Taux de Faux Positifs (FPR)")
ax1.set_ylabel("Taux de Vrais Positifs (TPR)")
ax1.set_title("Courbes ROC Comparatives")
ax1.legend(fontsize=7.5, loc='lower right')

auc_sorted = dict(sorted(auc_results.items(), key=lambda x: x[1], reverse=True))
bars = ax2.barh(list(auc_sorted.keys()), list(auc_sorted.values()),
                color=colors_roc[:len(auc_sorted)], edgecolor='white')
for bar, val in zip(bars, auc_sorted.values()):
    ax2.text(bar.get_width()+0.002, bar.get_y()+bar.get_height()/2,
             f'{val:.4f}', va='center', fontsize=9, fontweight='bold')
ax2.axvline(0.5, color=PALETTE["neutral"], linestyle='--')
ax2.set_title("AUC par modèle")
ax2.set_xlabel("AUC")
ax2.set_xlim(0.4, 1.05)

plt.tight_layout()
save_fig("05_roc_curves", fig)

print("\n  Classement AUC :")
for i, (name, val) in enumerate(auc_sorted.items(), 1):
    print(f"  {i}. {name:<35s}  AUC = {val:.4f}")


# ─────────────────────────────────────────────────────────────────
# 11. COURBES D'APPRENTISSAGE DES RÉSEAUX
# ─────────────────────────────────────────────────────────────────
section("11. COURBES D'APPRENTISSAGE — RÉSEAUX DE NEURONES", "📉")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Courbes d'apprentissage — Réseaux de Neurones",
             fontsize=14, fontweight='bold')

def plot_history(ax_loss, ax_acc, history, title):
    ax_loss.plot(history.history['loss'],
                 color=PALETTE["primary"], linewidth=2, label='Train')
    ax_loss.plot(history.history['val_loss'],
                 color=PALETTE["danger"], linewidth=2,
                 linestyle='--', label='Validation')
    ax_loss.set_title(f"{title} — Loss")
    ax_loss.set_xlabel("Epochs")
    ax_loss.set_ylabel("Binary Crossentropy")
    ax_loss.legend()

    ax_acc.plot(history.history['accuracy'],
                color=PALETTE["primary"], linewidth=2, label='Train')
    ax_acc.plot(history.history['val_accuracy'],
                color=PALETTE["danger"], linewidth=2,
                linestyle='--', label='Validation')
    ax_acc.set_title(f"{title} — Accuracy")
    ax_acc.set_xlabel("Epochs")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.legend()

plot_history(axes[0, 0], axes[0, 1], history_simple, "Réseau Simple")
plot_history(axes[1, 0], axes[1, 1], history_deep,   "Réseau Profond")

plt.tight_layout()
save_fig("06_learning_curves_ann", fig)


# ─────────────────────────────────────────────────────────────────
# 12. FEATURE IMPORTANCE — VISUALISATION COMPLÈTE
# ─────────────────────────────────────────────────────────────────
section("12. IMPORTANCE DES VARIABLES", "🏆")

TOP_N = 15
fig, axes = plt.subplots(1, 3, figsize=(20, 8))
fig.suptitle("Importance des Variables — RF / GB / DT",
             fontsize=14, fontweight='bold')

for ax, (metric, col, name) in zip(axes, [
        ('RF', PALETTE["primary"],   "Random Forest"),
        ('GB', PALETTE["secondary"], "Gradient Boosting"),
        ('DT', PALETTE["accent"],    "Decision Tree"),
]):
    top = fi_df.nlargest(TOP_N, metric)
    feat_clean = top['feature'].str.replace('_', ' ').str[:35]
    bars = ax.barh(feat_clean[::-1], top[metric][::-1],
                   color=col, alpha=0.85, edgecolor='white')
    for bar in bars:
        ax.text(bar.get_width()+0.0005, bar.get_y()+bar.get_height()/2,
                f'{bar.get_width():.4f}', va='center', fontsize=7.5)
    ax.set_title(f"Top {TOP_N} — {name}")
    ax.set_xlabel("Importance")

plt.tight_layout()
save_fig("07_feature_importance", fig)


# ─────────────────────────────────────────────────────────────────
# 13. SAUVEGARDE DES MODÈLES OPTIMAUX
# ─────────────────────────────────────────────────────────────────
section("13. SAUVEGARDE DES MODÈLES", "💾")

# Modèle .h5 / .keras
model_deep.save(os.path.join(OUTPUT_DIR, 'telemarketing_best_model.keras'))
model_top10.save(os.path.join(OUTPUT_DIR, 'telemarketing_optimal_top10.keras'))
print("  ✅ Modèles Keras sauvegardés")

# Scaler, LabelEncoder, features → pickle
artifacts = {
    'telemarketing_scaler.pkl':         scaler,
    'telemarketing_label_encoder.pkl':  le_target,
    'telemarketing_features.pkl':       feature_cols,
    'telemarketing_top10_features.pkl': top10_features,
    'telemarketing_top10_indices.pkl':  top10_idx,
}
for fname, obj in artifacts.items():
    path = os.path.join(OUTPUT_DIR, fname)
    with open(path, 'wb') as f:
        pickle.dump(obj, f)
    print(f"  ✅ {fname} sauvegardé")

# Modèle Random Forest → pickle (pour déploiement)
with open(os.path.join(OUTPUT_DIR, 'bank-tel.pkl'), 'wb') as f:
    pickle.dump({'model': rf_opt, 'scaler': scaler,
                 'features': feature_cols, 'top10': top10_features,
                 'top10_idx': top10_idx, 'le': le_target}, f)
print("  ✅ bank-tel.pkl sauvegardé (déploiement)")


# ═════════════════════════════════════════════════════════════════
# SOUS-PARTIE B : FASHION MNIST — DEEP LEARNING
# ═════════════════════════════════════════════════════════════════
section("SOUS-PARTIE B : FASHION MNIST — DEEP LEARNING", "👗")

# ─────────────────────────────────────────────────────────────────
# 14. CHARGEMENT FASHION MNIST
# ─────────────────────────────────────────────────────────────────
section("14. CHARGEMENT FASHION MNIST", "📦")

(X_train_fm, y_train_fm), (X_test_fm, y_test_fm) = \
    tf.keras.datasets.fashion_mnist.load_data()

class_names = ['T-shirt/top','Trouser','Pullover','Dress','Coat',
               'Sandal','Shirt','Sneaker','Bag','Ankle boot']

print(f"  Train : {X_train_fm.shape}  |  Test : {X_test_fm.shape}")
print(f"  Classes ({len(class_names)}) : {class_names}")

# Normalisation
X_train_fm = X_train_fm.astype('float32') / 255.0
X_test_fm  = X_test_fm.astype('float32') / 255.0

# One-hot pour catégorielle
y_train_oh = to_categorical(y_train_fm, 10)
y_test_oh  = to_categorical(y_test_fm,  10)

# Version 4D pour les CNN
X_train_cnn = X_train_fm[..., np.newaxis]
X_test_cnn  = X_test_fm[..., np.newaxis]

# Aperçu des images
fig, axes = plt.subplots(3, 10, figsize=(18, 6))
fig.suptitle("Fashion MNIST — Exemples par classe",
             fontsize=14, fontweight='bold')
for cls in range(10):
    idx = np.where(y_train_fm == cls)[0][:3]
    for row, i in enumerate(idx):
        ax = axes[row, cls]
        ax.imshow(X_train_fm[i], cmap='gray')
        ax.axis('off')
        if row == 0:
            ax.set_title(class_names[cls], fontsize=7.5, fontweight='bold')
plt.tight_layout()
save_fig("08_fashion_mnist_samples", fig)


# ─────────────────────────────────────────────────────────────────
# 15. MODÈLE 1 — MLP (Fully Connected)
# ─────────────────────────────────────────────────────────────────
section("15. MODÈLE 1 — MLP (Fully Connected)", "🧠")

mlp = Sequential([
    Input(shape=(28, 28)),
    Flatten(),
    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(10, activation='softmax')
], name="MLP")

mlp.compile(optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy'])
mlp.summary()

cb_fm = [EarlyStopping(monitor='val_accuracy', patience=10,
                       restore_best_weights=True, mode='max'),
         ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                           patience=5, min_lr=1e-6)]

t0 = time.time()
hist_mlp = mlp.fit(X_train_fm, y_train_oh,
                   validation_split=0.1,
                   epochs=50, batch_size=128,
                   callbacks=cb_fm, verbose=1)
print(f"  MLP terminé en {time.time()-t0:.1f}s")

loss_mlp, acc_mlp = mlp.evaluate(X_test_fm, y_test_oh, verbose=0)
print(f"  MLP — Test Accuracy : {acc_mlp:.4f}  |  Loss : {loss_mlp:.4f}")


# ─────────────────────────────────────────────────────────────────
# 16. MODÈLE 2 — CNN PERSONNALISÉ (type LeNet amélioré)
# ─────────────────────────────────────────────────────────────────
section("16. MODÈLE 2 — CNN PERSONNALISÉ (LeNet+)", "🔭")

cnn_custom = Sequential([
    Input(shape=(28, 28, 1)),

    # Bloc 1
    Conv2D(32, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(32, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2,2)),
    Dropout(0.25),

    # Bloc 2
    Conv2D(64, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(64, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2,2)),
    Dropout(0.25),

    # Bloc 3
    Conv2D(128, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2,2)),
    Dropout(0.3),

    # Classifieur
    Flatten(),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.4),
    Dense(10, activation='softmax')
], name="CNN_LeNet_Plus")

cnn_custom.compile(optimizer=Adam(learning_rate=0.001),
                   loss='categorical_crossentropy',
                   metrics=['accuracy'])
cnn_custom.summary()

t0 = time.time()
hist_cnn = cnn_custom.fit(X_train_cnn, y_train_oh,
                          validation_split=0.1,
                          epochs=50, batch_size=128,
                          callbacks=cb_fm, verbose=1)
print(f"  CNN LeNet+ terminé en {time.time()-t0:.1f}s")

loss_cnn, acc_cnn = cnn_custom.evaluate(X_test_cnn, y_test_oh, verbose=0)
print(f"  CNN LeNet+ — Test Accuracy : {acc_cnn:.4f}  |  Loss : {loss_cnn:.4f}")


# ─────────────────────────────────────────────────────────────────
# 17. MODÈLE 3 — VGG-LIKE (architecture VGGNet adaptée 28x28)
# ─────────────────────────────────────────────────────────────────
section("17. MODÈLE 3 — VGGNet adapté (28×28)", "🏗️")

vgg_like = Sequential([
    Input(shape=(28, 28, 1)),

    Conv2D(64, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(64, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2,2)),
    Dropout(0.25),

    Conv2D(128, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(128, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2,2)),
    Dropout(0.3),

    Conv2D(256, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2,2)),
    Dropout(0.35),

    Flatten(),
    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(256, activation='relu'),
    Dropout(0.4),
    Dense(10, activation='softmax')
], name="VGGNet_like")

vgg_like.compile(optimizer=Adam(learning_rate=0.001),
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])
vgg_like.summary()

t0 = time.time()
hist_vgg = vgg_like.fit(X_train_cnn, y_train_oh,
                        validation_split=0.1,
                        epochs=50, batch_size=128,
                        callbacks=cb_fm, verbose=1)
print(f"  VGGNet-like terminé en {time.time()-t0:.1f}s")

loss_vgg, acc_vgg = vgg_like.evaluate(X_test_cnn, y_test_oh, verbose=0)
print(f"  VGGNet-like — Test Accuracy : {acc_vgg:.4f}  |  Loss : {loss_vgg:.4f}")


# ─────────────────────────────────────────────────────────────────
# 18. MODÈLE 4 — ResNet-style (bloc résiduel manuel)
# ─────────────────────────────────────────────────────────────────
section("18. MODÈLE 4 — ResNet-style (blocs résiduels)", "⛓️")

from tensorflow.keras.layers import Add
from tensorflow.keras import Model

def residual_block(x, filters, stride=1):
    shortcut = x
    x = Conv2D(filters, (3,3), strides=stride,
               padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(filters, (3,3), padding='same', activation=None)(x)
    x = BatchNormalization()(x)
    if shortcut.shape[-1] != filters or stride != 1:
        shortcut = Conv2D(filters, (1,1), strides=stride,
                          padding='same')(shortcut)
        shortcut = BatchNormalization()(shortcut)
    x = Add()([x, shortcut])
    x = tf.keras.layers.Activation('relu')(x)
    return x

inp = Input(shape=(28, 28, 1))
x = Conv2D(32, (3,3), padding='same', activation='relu')(inp)
x = BatchNormalization()(x)
x = residual_block(x, 32)
x = MaxPooling2D((2,2))(x)
x = Dropout(0.25)(x)
x = residual_block(x, 64)
x = MaxPooling2D((2,2))(x)
x = Dropout(0.3)(x)
x = residual_block(x, 128)
x = GlobalAveragePooling2D()(x)
x = Dropout(0.4)(x)
out = Dense(10, activation='softmax')(x)

resnet_model = Model(inp, out, name="ResNet_custom")
resnet_model.compile(optimizer=Adam(learning_rate=0.001),
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])
resnet_model.summary()

t0 = time.time()
hist_res = resnet_model.fit(X_train_cnn, y_train_oh,
                            validation_split=0.1,
                            epochs=50, batch_size=128,
                            callbacks=cb_fm, verbose=1)
print(f"  ResNet-custom terminé en {time.time()-t0:.1f}s")

loss_res, acc_res = resnet_model.evaluate(X_test_cnn, y_test_oh, verbose=0)
print(f"  ResNet-custom — Test Accuracy : {acc_res:.4f}  |  Loss : {loss_res:.4f}")


# ─────────────────────────────────────────────────────────────────
# 19. COMPARAISON & VISUALISATION FASHION MNIST
# ─────────────────────────────────────────────────────────────────
section("19. COMPARAISON — MODÈLES FASHION MNIST", "📊")

fm_results = {
    'MLP (Fully Connected)': (acc_mlp, loss_mlp, hist_mlp),
    'CNN LeNet+':            (acc_cnn, loss_cnn, hist_cnn),
    'VGGNet-like':           (acc_vgg, loss_vgg, hist_vgg),
    'ResNet-custom':         (acc_res, loss_res, hist_res),
}

print("\n  Résultats Fashion MNIST :")
print(f"  {'Modèle':<25s}  {'Accuracy':>10s}  {'Loss':>8s}")
print("  " + "─" * 48)
for name, (acc, loss, _) in sorted(fm_results.items(),
                                    key=lambda x: x[1][0], reverse=True):
    print(f"  {name:<25s}  {acc:>10.4f}  {loss:>8.4f}")

# ── Figure : Courbes d'apprentissage DL ──────────────────────
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
fig.suptitle("Fashion MNIST — Courbes d'apprentissage (Loss & Accuracy)",
             fontsize=14, fontweight='bold')

for col, (name, (acc, loss, hist)) in enumerate(fm_results.items()):
    ax_loss = axes[0, col]
    ax_acc  = axes[1, col]

    ax_loss.plot(hist.history['loss'],
                 color=PALETTE["primary"], linewidth=2, label='Train')
    ax_loss.plot(hist.history['val_loss'],
                 color=PALETTE["danger"], linewidth=2,
                 linestyle='--', label='Val')
    ax_loss.set_title(f"{name}\n(Acc={acc:.3f})")
    ax_loss.set_ylabel("Loss")
    ax_loss.legend(fontsize=8)

    ax_acc.plot(hist.history['accuracy'],
                color=PALETTE["primary"], linewidth=2, label='Train')
    ax_acc.plot(hist.history['val_accuracy'],
                color=PALETTE["danger"], linewidth=2,
                linestyle='--', label='Val')
    ax_acc.set_xlabel("Epochs")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.legend(fontsize=8)

plt.tight_layout()
save_fig("09_fashion_mnist_learning_curves", fig)

# ── Figure : Matrice de confusion DL ─────────────────────────
best_fm_name = max(fm_results, key=lambda x: fm_results[x][0])
print(f"\n  Meilleur modèle Fashion MNIST : {best_fm_name}")

best_fm_model = {'MLP (Fully Connected)': mlp,
                 'CNN LeNet+': cnn_custom,
                 'VGGNet-like': vgg_like,
                 'ResNet-custom': resnet_model}[best_fm_name]

X_eval_fm = (X_test_cnn if 'CNN' in best_fm_name
             or 'VGG' in best_fm_name or 'Res' in best_fm_name
             else X_test_fm)
y_pred_fm = np.argmax(best_fm_model.predict(X_eval_fm, verbose=0), axis=1)

fig, ax = plt.subplots(figsize=(10, 8))
cm_fm = confusion_matrix(y_test_fm, y_pred_fm)
sns.heatmap(cm_fm, annot=True, fmt='d', cmap='Purples',
            xticklabels=class_names, yticklabels=class_names, ax=ax)
ax.set_title(f"Matrice de Confusion — {best_fm_name}\n"
             f"Accuracy = {fm_results[best_fm_name][0]:.4f}",
             fontsize=12, fontweight='bold')
ax.set_xlabel("Prédit", fontsize=11)
ax.set_ylabel("Réel", fontsize=11)
plt.xticks(rotation=35, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
save_fig("10_fashion_mnist_confusion_matrix", fig)

# ── Figure : Prédictions visuelles ────────────────────────────
X_show = X_test_cnn if hasattr(best_fm_model.layers[1], 'filters') else X_test_fm
y_proba_fm = best_fm_model.predict(X_show[:50], verbose=0)
y_pred_50  = np.argmax(y_proba_fm, axis=1)

fig, axes = plt.subplots(5, 10, figsize=(18, 10))
fig.suptitle(f"Prédictions — {best_fm_name}",
             fontsize=13, fontweight='bold')
for i, ax in enumerate(axes.flat):
    ax.imshow(X_test_fm[i], cmap='gray')
    correct = y_pred_50[i] == y_test_fm[i]
    color = PALETTE["success"] if correct else PALETTE["danger"]
    ax.set_title(f"P:{class_names[y_pred_50[i]][:5]}\n"
                 f"R:{class_names[y_test_fm[i]][:5]}",
                 fontsize=6, color=color)
    ax.axis('off')
plt.tight_layout()
save_fig("11_fashion_mnist_predictions", fig)


# ─────────────────────────────────────────────────────────────────
# 20. SAUVEGARDE MODÈLES FASHION MNIST
# ─────────────────────────────────────────────────────────────────
section("20. SAUVEGARDE MODÈLES FASHION MNIST", "💾")

for name, model in [('mlp', mlp), ('cnn_lenet', cnn_custom),
                    ('vgg_like', vgg_like), ('resnet', resnet_model)]:
    path = os.path.join(OUTPUT_DIR, f'fashion_mnist_{name}.keras')
    model.save(path)
    print(f"  ✅ fashion_mnist_{name}.keras sauvegardé")

best_fm_model.save(os.path.join(OUTPUT_DIR, 'fashion_mnist_best.keras'))
print("  ✅ fashion_mnist_best.keras sauvegardé")


# ═════════════════════════════════════════════════════════════════
# RÉSUMÉ FINAL
# ═════════════════════════════════════════════════════════════════
section("RÉSUMÉ FINAL", "✅")

best_bank = df_results.iloc[0]
best_fm   = max(fm_results, key=lambda x: fm_results[x][0])
best_fm_acc = fm_results[best_fm][0]

print(f"""
╔══════════════════════════════════════════════════════════════════╗
║                     RÉSULTATS FINAUX                             ║
╠══════════════════════════════════════════════════════════════════╣
║  A. BANK TELEMARKETING                                           ║
║     Instances (après nettoyage) : {len(df_clean):>7,}            ║
║     Features encodées           : {len(feature_cols):>7}         ║
║     Taux souscription 'yes'     : {target_dist['yes']/len(df)*100:>6.1f}%                  ║
║                                                                  ║
║  Meilleur modèle : {best_bank['Modèle']:<40s}                    ║
║     Accuracy  : {best_bank['Accuracy']:.4f}                      ║
║     F1-Score  : {best_bank['F1-Score']:.4f}                      ║
║     Précision : {best_bank['Précision']:.4f}                     ║
║     Rappel    : {best_bank['Rappel']:.4f}                        ║
╠══════════════════════════════════════════════════════════════════╣
║  B. FASHION MNIST (Deep Learning)                                ║
║     Instances train/test : {len(X_train_fm):,} / {len(X_test_fm):,}                  ║
║     Classes              : {len(class_names)}                    ║
║                                                                  ║
║  Meilleur modèle : {best_fm:<40s}                                ║
║     Accuracy (test) : {best_fm_acc:.4f}                          ║
╠══════════════════════════════════════════════════════════════════╣
║  Top 3 variables importantes (Bank - RF) :                      ║
║    1. {fi_df.iloc[0]['feature']:<55s}║
║    2. {fi_df.iloc[1]['feature']:<55s}║
║    3. {fi_df.iloc[2]['feature']:<55s}║
╚══════════════════════════════════════════════════════════════════╝
""")

print(f"  📁 Figures & modèles sauvegardés dans → ./{OUTPUT_DIR}/")
print("  🎉 TP Partie 2 terminé avec succès!\n")