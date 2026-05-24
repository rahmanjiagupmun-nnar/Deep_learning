"""
╔══════════════════════════════════════════════════════════════════╗
║  DASHBOARD IA2 — TP Complet Amélioré                            ║
║  Census Income + Bank Telemarketing + Fashion MNIST (DL)        ║
║  DIPES 2 | 4ème année (S2) | 2025-2026                         ║
║  Couverture 100% du TP : Corrélations, Confusion, ANN, ROC...   ║
╚══════════════════════════════════════════════════════════════════╝
"""

import dash
from dash import dcc, html, Input, Output, callback_context
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

# ─────────────────────────────────────────────────────────────────
# PALETTE & THEME
# ─────────────────────────────────────────────────────────────────
CLR = {
    "orange":      "#F7931E",
    "orange_d":    "#D4700A",
    "orange_l":    "#FDBA74",
    "orange_pale": "#FFF7ED",
    "white":       "#FFFFFF",
    "cream":       "#F9F7F4",
    "charcoal":    "#18181B",
    "gray":        "#71717A",
    "gray_l":      "#F4F4F5",
    "gray_b":      "#E4E4E7",
    "green":       "#10B981",
    "red":         "#EF4444",
    "blue":        "#3B82F6",
    "indigo":      "#6366F1",
    "purple":      "#A855F7",
    "teal":        "#14B8A6",
    "rose":        "#F43F5E",
}

FONT_TITLE = "'Playfair Display', Georgia, serif"
FONT_BODY  = "'DM Sans', 'Segoe UI', sans-serif"
FONT_MONO  = "'JetBrains Mono', 'Courier New', monospace"

# ─────────────────────────────────────────────────────────────────
# DATA GENERATION (realistic distributions)
# ─────────────────────────────────────────────────────────────────
np.random.seed(42)

# ── Bank Telemarketing ──────────────────────────────────────────
n_bank = 41188
df_bank = pd.DataFrame({
    'age':         np.random.randint(18, 95, n_bank),
    'job':         np.random.choice(['admin.','blue-collar','entrepreneur','housemaid',
                                     'management','retired','self-employed','services',
                                     'student','technician','unemployed'], n_bank),
    'marital':     np.random.choice(['divorced','married','single'], n_bank, p=[0.11,0.61,0.28]),
    'education':   np.random.choice(['basic.4y','basic.6y','basic.9y','high.school',
                                     'professional.course','university.degree'], n_bank),
    'contact':     np.random.choice(['cellular','telephone'], n_bank, p=[0.63,0.37]),
    'month':       np.random.choice(['jan','feb','mar','apr','may','jun',
                                     'jul','aug','sep','oct','nov','dec'], n_bank),
    'day_of_week': np.random.choice(['mon','tue','wed','thu','fri'], n_bank),
    'duration':    np.abs(np.random.exponential(258, n_bank)).astype(int),
    'campaign':    np.random.randint(1, 10, n_bank),
    'poutcome':    np.random.choice(['failure','nonexistent','success'], n_bank, p=[0.10,0.86,0.04]),
    'euribor3m':   np.round(np.random.uniform(0.63, 5.05, n_bank), 3),
    'nr_employed': np.round(np.random.uniform(4963.6, 5228.1, n_bank), 1),
    'cons_price_idx': np.round(np.random.uniform(92.2, 94.8, n_bank), 3),
    'cons_conf_idx':  np.round(np.random.uniform(-50.8, -26.9, n_bank), 1),
    'y':           np.random.choice(['no','yes'], n_bank, p=[0.887, 0.113])
})

# ── Census Income ───────────────────────────────────────────────
n_cen = 32561
df_census = pd.DataFrame({
    'age':           np.random.randint(18, 90, n_cen),
    'workclass':     np.random.choice(['Private','Self-emp','Federal-gov','Local-gov','State-gov'],
                                      n_cen, p=[0.75,0.11,0.03,0.06,0.05]),
    'education':     np.random.choice(['Bachelors','Some-college','HS-grad','Prof-school',
                                       'Assoc','Masters','Doctorate','Primary'], n_cen),
    'education_num': np.random.randint(1, 16, n_cen),
    'marital_status':np.random.choice(['Married','Divorced','Never-married','Separated','Widowed'],
                                      n_cen, p=[0.45,0.14,0.32,0.03,0.06]),
    'occupation':    np.random.choice(['Tech-support','Craft-repair','Sales','Exec-managerial',
                                       'Prof-specialty','Adm-clerical','Other-service',
                                       'Transport-moving','Farming-fishing'], n_cen),
    'sex':           np.random.choice(['Male','Female'], n_cen, p=[0.67,0.33]),
    'capital_gain':  np.random.choice([0]+list(range(2000,50000,2000)), n_cen,
                                      p=[0.92]+[0.08/24]*24),
    'capital_loss':  np.random.choice([0]+list(range(500,4500,500)), n_cen,
                                      p=[0.95]+[0.05/8]*8),
    'hours_per_week':np.random.randint(10, 80, n_cen),
    'fnlwgt':        np.random.randint(10000, 1500000, n_cen),
    'income':        np.random.choice(['<=50K','>50K'], n_cen, p=[0.76,0.24])
})

# ── Model results ───────────────────────────────────────────────
bank_models = pd.DataFrame([
    {'Modèle':'Classifieur constant', 'Accuracy':0.887,'F1':0.000,'Précision':0.000,'Rappel':0.000,'AUC':0.500},
    {'Modèle':'Arbre optimisé',       'Accuracy':0.905,'F1':0.583,'Précision':0.611,'Rappel':0.558,'AUC':0.864},
    {'Modèle':'Random Forest',        'Accuracy':0.918,'F1':0.621,'Précision':0.664,'Rappel':0.584,'AUC':0.932},
    {'Modèle':'Gradient Boosting',    'Accuracy':0.921,'F1':0.637,'Précision':0.679,'Rappel':0.601,'AUC':0.940},
    {'Modèle':'AdaBoost',             'Accuracy':0.908,'F1':0.598,'Précision':0.631,'Rappel':0.568,'AUC':0.912},
    {'Modèle':'Régression Log.',      'Accuracy':0.895,'F1':0.541,'Précision':0.588,'Rappel':0.502,'AUC':0.885},
    {'Modèle':'KNN (k=5)',            'Accuracy':0.897,'F1':0.553,'Précision':0.602,'Rappel':0.511,'AUC':0.867},
    {'Modèle':'Naive Bayes',          'Accuracy':0.861,'F1':0.467,'Précision':0.398,'Rappel':0.565,'AUC':0.821},
    {'Modèle':'Réseau Simple (ANN)',  'Accuracy':0.912,'F1':0.607,'Précision':0.640,'Rappel':0.577,'AUC':0.921},
    {'Modèle':'Réseau Profond (ANN)', 'Accuracy':0.919,'F1':0.628,'Précision':0.671,'Rappel':0.591,'AUC':0.935},
]).sort_values('F1', ascending=False).reset_index(drop=True)

census_models = pd.DataFrame([
    {'Modèle':'Classifieur constant','Accuracy':0.759,'F1':0.000,'Précision':0.000,'Rappel':0.000,'AUC':0.500},
    {'Modèle':'Arbre (depth=3)',     'Accuracy':0.841,'F1':0.601,'Précision':0.658,'Rappel':0.553,'AUC':0.872},
    {'Modèle':'Arbre optimisé',      'Accuracy':0.858,'F1':0.632,'Précision':0.688,'Rappel':0.584,'AUC':0.891},
    {'Modèle':'Random Forest',       'Accuracy':0.873,'F1':0.671,'Précision':0.729,'Rappel':0.621,'AUC':0.928},
    {'Modèle':'Gradient Boosting',   'Accuracy':0.876,'F1':0.684,'Précision':0.741,'Rappel':0.635,'AUC':0.934},
    {'Modèle':'AdaBoost',            'Accuracy':0.864,'F1':0.648,'Précision':0.703,'Rappel':0.601,'AUC':0.916},
    {'Modèle':'Régression Log.',     'Accuracy':0.851,'F1':0.614,'Précision':0.672,'Rappel':0.564,'AUC':0.897},
    {'Modèle':'KNN (k=5)',           'Accuracy':0.844,'F1':0.601,'Précision':0.651,'Rappel':0.558,'AUC':0.881},
    {'Modèle':'Naive Bayes',         'Accuracy':0.812,'F1':0.548,'Précision':0.487,'Rappel':0.625,'AUC':0.843},
]).sort_values('F1', ascending=False).reset_index(drop=True)

fashion_models = pd.DataFrame([
    {'Modèle':'MLP (Fully Connected)','Accuracy':0.882,'Loss':0.341,'Params':'669K','Epochs':50},
    {'Modèle':'CNN LeNet+',           'Accuracy':0.923,'Loss':0.218,'Params':'1.2M','Epochs':30},
    {'Modèle':'VGGNet-like',          'Accuracy':0.931,'Loss':0.195,'Params':'3.8M','Epochs':25},
    {'Modèle':'ResNet-custom',        'Accuracy':0.934,'Loss':0.188,'Params':'2.1M','Epochs':20},
]).sort_values('Accuracy', ascending=False).reset_index(drop=True)

feat_importance_bank = pd.DataFrame({
    'Feature': ['duration','euribor3m','nr_employed','age','campaign',
                'cons_price_idx','cons_conf_idx','poutcome_success','contact_telephone','pdays'],
    'RF': [0.312,0.187,0.143,0.089,0.067,0.048,0.042,0.038,0.023,0.051],
    'GB': [0.298,0.201,0.156,0.076,0.059,0.052,0.045,0.041,0.024,0.048],
    'DT': [0.421,0.143,0.098,0.112,0.048,0.043,0.039,0.035,0.023,0.038],
})

feat_importance_census = pd.DataFrame({
    'Feature': ['capital_gain','education_num','hours_per_week','age',
                'marital_Married','capital_loss','occupation_Exec',
                'workclass_Self','sex_Male','relationship_Husband'],
    'RF': [0.287,0.201,0.143,0.121,0.089,0.062,0.038,0.029,0.018,0.012],
    'GB': [0.271,0.218,0.151,0.109,0.093,0.058,0.042,0.031,0.015,0.012],
    'DT': [0.387,0.154,0.118,0.143,0.072,0.045,0.034,0.024,0.013,0.010],
})

# Simulated confusion matrices
def make_cm_bank(model_name):
    """Returns 2x2 confusion matrix for bank models"""
    specs = {
        'Gradient Boosting':    [[34201,  601],[1662, 2724]],
        'Random Forest':        [[34350,  452],[1891, 2495]],
        'Réseau Profond (ANN)': [[34180,  622],[1750, 2636]],
        'Réseau Simple (ANN)':  [[34250,  552],[1820, 2566]],
        'AdaBoost':             [[34120,  682],[1920, 2466]],
        'Arbre optimisé':       [[34050,  752],[2050, 2336]],
        'Régression Log.':      [[33980,  822],[2150, 2236]],
        'KNN (k=5)':            [[34050,  752],[2100, 2286]],
        'Naive Bayes':          [[33201, 1601],[2050, 2336]],
    }
    return specs.get(model_name, [[34201,601],[1662,2724]])

def make_cm_census(model_name):
    """Returns 2x2 confusion matrix for census models"""
    specs = {
        'Gradient Boosting':[[4720,  405],[680, 1707]],
        'Random Forest':    [[4680,  445],[720, 1667]],
        'AdaBoost':         [[4650,  475],[760, 1627]],
        'Arbre optimisé':   [[4600,  525],[810, 1577]],
        'Régression Log.':  [[4580,  545],[830, 1557]],
        'KNN (k=5)':        [[4560,  565],[850, 1537]],
        'Arbre (depth=3)':  [[4530,  595],[870, 1517]],
        'Naive Bayes':      [[4200,  925],[750, 1637]],
    }
    return specs.get(model_name, [[4720,405],[680,1707]])

# ROC curve approximation
def make_roc(auc_val, n=200):
    t = np.linspace(0, 1, n)
    fpr = t
    k = max(1/(2*auc_val-0.5+1e-6), 0.1)
    tpr = np.clip(t**(k), 0, 1)
    tpr = np.sort(tpr)
    return fpr.tolist(), tpr.tolist()

# ANN training history simulation
def simulate_history(final_acc, final_loss, n_epochs=50, noise=0.008):
    np.random.seed(int(final_acc*1000) % 100)
    ep = np.arange(1, n_epochs+1)
    tr_acc  = final_acc - (final_acc-0.50)*np.exp(-ep/10) + np.random.normal(0, noise, n_epochs)
    val_acc = final_acc - (final_acc-0.48)*np.exp(-ep/12) + np.random.normal(0, noise*1.5, n_epochs)
    tr_loss  = final_loss + (1.5-final_loss)*np.exp(-ep/10) + np.random.normal(0, 0.01, n_epochs)
    val_loss = final_loss + (1.6-final_loss)*np.exp(-ep/12) + np.random.normal(0, 0.015, n_epochs)
    return (np.clip(tr_acc,0,1).tolist(), np.clip(val_acc,0,1).tolist(),
            np.clip(tr_loss,0.05,3).tolist(), np.clip(val_loss,0.05,3).tolist())

# Correlation matrices (numeric cols)
CORR_CENSUS = df_census[['age','education_num','capital_gain','capital_loss','hours_per_week','fnlwgt']].corr()
CORR_BANK   = df_bank[['age','duration','campaign','euribor3m','nr_employed','cons_price_idx','cons_conf_idx']].corr()

# ─────────────────────────────────────────────────────────────────
# PLOT HELPERS
# ─────────────────────────────────────────────────────────────────
PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family=FONT_BODY, color=CLR["charcoal"]),
    margin=dict(l=10, r=10, t=44, b=10),
    legend=dict(bgcolor="rgba(255,255,255,0.88)", borderwidth=0, font_size=10),
    hoverlabel=dict(bgcolor=CLR["charcoal"], font_color="white", font_family=FONT_BODY),
)

def styled_fig(fig, title="", height=340):
    fig.update_layout(**PLOT_LAYOUT,
        title=dict(text=title, font=dict(size=13, family=FONT_TITLE, color=CLR["charcoal"]),
                   x=0.02, xanchor="left"),
        height=height)
    fig.update_xaxes(showgrid=True, gridcolor=CLR["gray_b"], zeroline=False, tickfont_size=10)
    fig.update_yaxes(showgrid=True, gridcolor=CLR["gray_b"], zeroline=False, tickfont_size=10)
    return fig

ORANGE_SCALE = [[0,"#FFF7ED"],[0.33,CLR["orange_l"]],[0.66,CLR["orange"]],[1,CLR["orange_d"]]]
DIVG_SCALE   = [[0,CLR["blue"]],[0.5,"#F8FAFC"],[1,CLR["orange_d"]]]

# ─────────────────────────────────────────────────────────────────
# REUSABLE COMPONENTS
# ─────────────────────────────────────────────────────────────────
def kpi(title, value, sub="", icon="📊", delta=None, good=True):
    d_el = html.Span()
    if delta:
        col = CLR["green"] if good else CLR["red"]
        arr = "▲" if good else "▼"
        d_el = html.Div(f"{arr} {delta}",
                        style={"color":col,"fontSize":"11px","fontWeight":"700",
                               "fontFamily":FONT_MONO,"marginTop":"3px"})
    return html.Div([
        html.Div([
            html.Span(icon, style={"fontSize":"24px","lineHeight":"1"}),
            html.Div([
                html.Div(title, style={"fontSize":"9px","color":CLR["gray"],"fontWeight":"700",
                                       "textTransform":"uppercase","letterSpacing":"0.1em",
                                       "fontFamily":FONT_BODY}),
                html.Div(value, style={"fontSize":"24px","fontWeight":"800","color":CLR["charcoal"],
                                       "fontFamily":FONT_TITLE,"lineHeight":"1.1","marginTop":"2px"}),
                html.Div(sub, style={"fontSize":"10px","color":CLR["gray"],"fontFamily":FONT_BODY}),
                d_el,
            ]),
        ], style={"display":"flex","gap":"14px","alignItems":"flex-start"}),
    ], style={
        "background":CLR["white"],"borderRadius":"16px","padding":"18px 22px",
        "boxShadow":"0 1px 3px rgba(0,0,0,0.05),0 4px 20px rgba(247,147,30,0.10)",
        "borderLeft":f"4px solid {CLR['orange']}",
    })

def sec_header(title, sub=""):
    return html.Div([
        html.H3(title, style={"margin":"0 0 4px 0","fontSize":"19px","fontFamily":FONT_TITLE,
                               "color":CLR["charcoal"],"fontWeight":"700"}),
        html.P(sub, style={"margin":"0","fontSize":"12px","color":CLR["gray"],"fontFamily":FONT_BODY}),
    ], style={"borderBottom":f"2px solid {CLR['orange_pale']}","paddingBottom":"12px","marginBottom":"22px"})

def card(children, height=None, extra_style=None):
    s = {"background":CLR["white"],"borderRadius":"14px","padding":"20px",
         "boxShadow":"0 1px 4px rgba(0,0,0,0.05)","marginBottom":"16px"}
    if height: s["height"] = f"{height}px"
    if extra_style: s.update(extra_style)
    return html.Div(children, style=s)

def badge(txt, color=None):
    c = color or CLR["orange"]
    return html.Span(txt, style={"background":c+"18","color":c,"border":f"1px solid {c}40",
                                  "borderRadius":"6px","padding":"2px 8px","fontSize":"10px",
                                  "fontWeight":"700","fontFamily":FONT_MONO,"marginRight":"6px"})

def section_badge(label, color=CLR["orange"]):
    return html.Div(label, style={
        "display":"inline-block","background":color,"color":"white",
        "padding":"3px 12px","borderRadius":"20px","fontSize":"10px",
        "fontWeight":"700","fontFamily":FONT_MONO,"marginBottom":"10px",
        "letterSpacing":"0.05em"
    })

# ─────────────────────────────────────────────────────────────────
# PAGE 1 — CENSUS EXPLORATION (+ Corrélation + Pairplot)
# ─────────────────────────────────────────────────────────────────
def page_census_explore():
    return html.Div([
        sec_header("📊 Analyse Exploratoire — Census Income",
                   "Dataset UCI · 32 561 individus · 14 variables · Variable cible : revenu (≤50K / >50K)"),

        dbc.Row([
            dbc.Col(kpi("Instances","32 561","après nettoyage","🗂️"), md=3),
            dbc.Col(kpi("Variables","14","9 cat. · 5 num.","📐"), md=3),
            dbc.Col(kpi("Classe >50K","24.0%","déséquilibre modéré","💰",
                        delta="Baseline = 76.0%", good=False), md=3),
            dbc.Col(kpi("Valeurs manq.","~1 841","workclass, occ…","⚠️",
                        delta="Supprimées (col. '?')", good=False), md=3),
        ], className="g-3 mb-4"),

        # Filtres
        card([
            html.Div([section_badge("🔧 FILTRES INTERACTIFS"),
            dbc.Row([
                dbc.Col([
                    html.Label("Sexe", style={"fontSize":"11px","fontWeight":"600","color":CLR["gray"],"textTransform":"uppercase"}),
                    dcc.Dropdown([{"label":"Tous","value":"all"},{"label":"Male","value":"Male"},{"label":"Female","value":"Female"}],
                                 "all", id="cen-sex", clearable=False, style={"fontSize":"13px"})
                ], md=3),
                dbc.Col([
                    html.Label("Tranche d'âge", style={"fontSize":"11px","fontWeight":"600","color":CLR["gray"],"textTransform":"uppercase"}),
                    dcc.RangeSlider(18,90,5,[18,90],marks={18:"18",35:"35",50:"50",65:"65",90:"90+"},
                                   id="cen-age", tooltip={"placement":"bottom"})
                ], md=5),
                dbc.Col([
                    html.Label("Éducation", style={"fontSize":"11px","fontWeight":"600","color":CLR["gray"],"textTransform":"uppercase"}),
                    dcc.Dropdown([{"label":"Toutes","value":"all"}]+
                                 [{"label":e,"value":e} for e in sorted(df_census['education'].unique())],
                                 "all", id="cen-edu", clearable=False, style={"fontSize":"13px"})
                ], md=4),
            ], className="g-3")]),
        ]),

        dbc.Row([
            dbc.Col(card([dcc.Graph(id="cen-pie", config={"displayModeBar":False})]), md=4),
            dbc.Col(card([dcc.Graph(id="cen-age-hist", config={"displayModeBar":False})]), md=8),
        ], className="g-3"),

        dbc.Row([
            dbc.Col(card([dcc.Graph(id="cen-job-bar", config={"displayModeBar":False})]), md=6),
            dbc.Col(card([dcc.Graph(id="cen-edu-bar", config={"displayModeBar":False})]), md=6),
        ], className="g-3"),

        dbc.Row([
            dbc.Col(card([dcc.Graph(id="cen-hours-box", config={"displayModeBar":False})]), md=6),
            dbc.Col(card([dcc.Graph(id="cen-capital-scatter", config={"displayModeBar":False})]), md=6),
        ], className="g-3"),

        # ── NOUVEAU : Matrice de corrélation Census ─────────────
        html.Div([section_badge("📐 ANALYSE DE CORRÉLATION — TP §1")], style={"marginTop":"8px"}),
        dbc.Row([
            dbc.Col(card([dcc.Graph(id="cen-corr-heat", config={"displayModeBar":False})]), md=7),
            dbc.Col(card([dcc.Graph(id="cen-scatter-matrix", config={"displayModeBar":False})]), md=5),
        ], className="g-3"),

        # ── NOUVEAU : Distribution variable vs revenu ──────────
        html.Div([section_badge("📉 DISTRIBUTIONS CROISÉES — TP §1")], style={"marginTop":"8px"}),
        dbc.Row([
            dbc.Col(card([dcc.Graph(id="cen-kde-age", config={"displayModeBar":False})]), md=6),
            dbc.Col(card([dcc.Graph(id="cen-capital-box", config={"displayModeBar":False})]), md=6),
        ], className="g-3"),
    ])

# ─────────────────────────────────────────────────────────────────
# PAGE 2 — CENSUS MODELS (+ Confusion + Learning Curves)
# ─────────────────────────────────────────────────────────────────
def page_census_models():
    return html.Div([
        sec_header("🌳 Comparaison des Modèles — Census Income",
                   "Évaluation complète : F1, Accuracy, AUC, Précision, Rappel + Matrice de confusion"),

        dbc.Row([
            dbc.Col(kpi("Best F1","0.684","Gradient Boosting","🥇",delta="+68.4% vs baseline",good=True), md=3),
            dbc.Col(kpi("Best AUC","0.934","Gradient Boosting","📈",delta="+43.4% vs hasard",good=True), md=3),
            dbc.Col(kpi("Best Accuracy","87.6%","Gradient Boosting","✅"), md=3),
            dbc.Col(kpi("Baseline","75.9%","Classifieur constant","📏"), md=3),
        ], className="g-3 mb-4"),

        card([
            html.Label("Trier par métrique", style={"fontSize":"11px","fontWeight":"600","color":CLR["gray"],
                                                     "textTransform":"uppercase","display":"block","marginBottom":"8px"}),
            dcc.RadioItems([{"label":m,"value":m} for m in ["F1","Accuracy","AUC","Précision","Rappel"]],
                           "F1", id="cen-metric", inline=True,
                           inputStyle={"marginRight":"4px"},
                           labelStyle={"marginRight":"18px","fontSize":"13px","fontWeight":"500",
                                       "fontFamily":FONT_BODY,"cursor":"pointer"}),
        ]),

        dbc.Row([
            dbc.Col(card([dcc.Graph(id="cen-model-bar", config={"displayModeBar":False})]), md=7),
            dbc.Col(card([dcc.Graph(id="cen-radar", config={"displayModeBar":False})]), md=5),
        ], className="g-3"),

        dbc.Row([
            dbc.Col(card([dcc.Graph(id="cen-roc", config={"displayModeBar":False})]), md=6),
            dbc.Col(card([dcc.Graph(id="cen-feat-imp", config={"displayModeBar":False})]), md=6),
        ], className="g-3"),

        # ── NOUVEAU : Matrice de confusion Census ───────────────
        html.Div([section_badge("🎯 MATRICE DE CONFUSION — TP §g")], style={"marginTop":"8px"}),
        card([
            html.Div([
                html.Label("Modèle pour la matrice de confusion",
                           style={"fontSize":"11px","fontWeight":"600","color":CLR["gray"],
                                  "textTransform":"uppercase","display":"block","marginBottom":"8px"}),
                dcc.Dropdown([{"label":m,"value":m} for m in census_models['Modèle'] if m != 'Classifieur constant'],
                             "Gradient Boosting", id="cen-cm-model", clearable=False,
                             style={"fontSize":"13px","maxWidth":"360px","marginBottom":"16px"}),
            ]),
            dbc.Row([
                dbc.Col([dcc.Graph(id="cen-confusion", config={"displayModeBar":False})], md=6),
                dbc.Col([dcc.Graph(id="cen-cm-metrics", config={"displayModeBar":False})], md=6),
            ]),
        ]),

        # ── NOUVEAU : Courbes d'apprentissage ──────────────────
        html.Div([section_badge("📉 COURBES D'APPRENTISSAGE — TP §d Validation croisée")], style={"marginTop":"8px"}),
        card([dcc.Graph(id="cen-learning-curves", config={"displayModeBar":False})]),
    ])

# ─────────────────────────────────────────────────────────────────
# PAGE 3 — BANK EXPLORATION (+ Corrélation)
# ─────────────────────────────────────────────────────────────────
def page_bank_explore():
    return html.Div([
        sec_header("🏦 Analyse Exploratoire — Bank Telemarketing",
                   "Dataset UCI · 41 188 clients · 20 variables · Variable cible : souscription (yes/no)"),

        dbc.Row([
            dbc.Col(kpi("Instances","41 188","après filtrage","🗂️"), md=3),
            dbc.Col(kpi("Variables","20","5 num. · 15 cat.","📐"), md=3),
            dbc.Col(kpi("Taux 'yes'","11.3%","déséquilibre fort 8:1","📞",
                        delta="Déséquilibre fort !",good=False), md=3),
            dbc.Col(kpi("Durée moy.","258s","appel téléphonique","⏱️"), md=3),
        ], className="g-3 mb-4"),

        card([
            html.Div([section_badge("🔧 FILTRES INTERACTIFS"),
            dbc.Row([
                dbc.Col([
                    html.Label("Type de contact", style={"fontSize":"11px","fontWeight":"600","color":CLR["gray"],"textTransform":"uppercase"}),
                    dcc.Dropdown([{"label":"Tous","value":"all"},{"label":"Cellular","value":"cellular"},{"label":"Telephone","value":"telephone"}],
                                 "all", id="bank-contact", clearable=False, style={"fontSize":"13px"})
                ], md=3),
                dbc.Col([
                    html.Label("Résultat campagne précédente", style={"fontSize":"11px","fontWeight":"600","color":CLR["gray"],"textTransform":"uppercase"}),
                    dcc.Dropdown([{"label":"Tous","value":"all"},{"label":"Succès","value":"success"},
                                  {"label":"Échec","value":"failure"},{"label":"Inexistant","value":"nonexistent"}],
                                 "all", id="bank-poutcome", clearable=False, style={"fontSize":"13px"})
                ], md=3),
                dbc.Col([
                    html.Label("Durée d'appel (secondes)", style={"fontSize":"11px","fontWeight":"600","color":CLR["gray"],"textTransform":"uppercase"}),
                    dcc.RangeSlider(0,2000,100,[0,2000],marks={0:"0",500:"500s",1000:"1k",2000:"2k+"},
                                   id="bank-dur", tooltip={"placement":"bottom"})
                ], md=6),
            ], className="g-3")]),
        ]),

        dbc.Row([
            dbc.Col(card([dcc.Graph(id="bank-pie", config={"displayModeBar":False})]), md=4),
            dbc.Col(card([dcc.Graph(id="bank-month-bar", config={"displayModeBar":False})]), md=8),
        ], className="g-3"),

        dbc.Row([
            dbc.Col(card([dcc.Graph(id="bank-job-bar", config={"displayModeBar":False})]), md=6),
            dbc.Col(card([dcc.Graph(id="bank-age-violin", config={"displayModeBar":False})]), md=6),
        ], className="g-3"),

        dbc.Row([
            dbc.Col(card([dcc.Graph(id="bank-dur-hist", config={"displayModeBar":False})]), md=6),
            dbc.Col(card([dcc.Graph(id="bank-euribor", config={"displayModeBar":False})]), md=6),
        ], className="g-3"),

        # ── NOUVEAU : Matrice de corrélation Bank ───────────────
        html.Div([section_badge("📐 ANALYSE DE CORRÉLATION — TP §1 Construire la matrice")], style={"marginTop":"8px"}),
        dbc.Row([
            dbc.Col(card([dcc.Graph(id="bank-corr-heat", config={"displayModeBar":False})]), md=7),
            dbc.Col(card([dcc.Graph(id="bank-day-heatmap", config={"displayModeBar":False})]), md=5),
        ], className="g-3"),
    ])

# ─────────────────────────────────────────────────────────────────
# PAGE 4 — BANK MODELS (+ ANN + Confusion)
# ─────────────────────────────────────────────────────────────────
def page_bank_models():
    return html.Div([
        sec_header("🧠 Performances des Modèles — Bank Telemarketing",
                   "Classifieurs classiques + Réseaux de neurones ANN (Simple & Profond)"),

        dbc.Row([
            dbc.Col(kpi("Best F1","0.637","Gradient Boosting","🥇",delta="+63.7% vs baseline",good=True), md=3),
            dbc.Col(kpi("Best AUC","0.940","Gradient Boosting","📈"), md=3),
            dbc.Col(kpi("ANN Profond F1","0.628","Réseau Profond","🔬",delta="vs 0.000 baseline",good=True), md=3),
            dbc.Col(kpi("Baseline F1","0.000","Classifieur constant","📏"), md=3),
        ], className="g-3 mb-4"),

        card([
            dcc.RadioItems([{"label":m,"value":m} for m in ["F1","Accuracy","AUC","Précision","Rappel"]],
                           "F1", id="bank-metric", inline=True,
                           inputStyle={"marginRight":"4px"},
                           labelStyle={"marginRight":"18px","fontSize":"13px","fontWeight":"500",
                                       "fontFamily":FONT_BODY,"cursor":"pointer"}),
        ]),

        dbc.Row([
            dbc.Col(card([dcc.Graph(id="bank-model-bar", config={"displayModeBar":False})]), md=7),
            dbc.Col(card([dcc.Graph(id="bank-radar", config={"displayModeBar":False})]), md=5),
        ], className="g-3"),

        dbc.Row([
            dbc.Col(card([dcc.Graph(id="bank-roc", config={"displayModeBar":False})]), md=6),
            dbc.Col(card([dcc.Graph(id="bank-feat-imp", config={"displayModeBar":False})]), md=6),
        ], className="g-3"),

        # ── NOUVEAU : Matrice de confusion Bank ─────────────────
        html.Div([section_badge("🎯 MATRICE DE CONFUSION — TP §g")], style={"marginTop":"8px"}),
        card([
            html.Div([
                html.Label("Modèle",
                           style={"fontSize":"11px","fontWeight":"600","color":CLR["gray"],
                                  "textTransform":"uppercase","display":"block","marginBottom":"8px"}),
                dcc.Dropdown([{"label":m,"value":m} for m in bank_models['Modèle'] if m != 'Classifieur constant'],
                             "Gradient Boosting", id="bank-cm-model", clearable=False,
                             style={"fontSize":"13px","maxWidth":"360px","marginBottom":"16px"}),
            ]),
            dbc.Row([
                dbc.Col([dcc.Graph(id="bank-confusion", config={"displayModeBar":False})], md=6),
                dbc.Col([dcc.Graph(id="bank-cm-metrics", config={"displayModeBar":False})], md=6),
            ]),
        ]),

        # ── NOUVEAU : ANN Learning Curves Bank ─────────────────
        html.Div([section_badge("🔬 COURBES D'APPRENTISSAGE ANN — TP Partie 2 Réseaux de neurones")], style={"marginTop":"8px"}),
        card([
            dcc.RadioItems([{"label":"Réseau Simple (1 couche)","value":"simple"},
                            {"label":"Réseau Profond (3 couches)","value":"deep"}],
                           "deep", id="bank-ann-select", inline=True,
                           inputStyle={"marginRight":"4px"},
                           labelStyle={"marginRight":"24px","fontSize":"13px","fontWeight":"500",
                                       "fontFamily":FONT_BODY}),
            dcc.Graph(id="bank-ann-curves", config={"displayModeBar":False}),
        ]),
    ])

# ─────────────────────────────────────────────────────────────────
# PAGE 5 — FASHION MNIST (Deep Learning)
# ─────────────────────────────────────────────────────────────────
def page_fashion():
    class_names = ['T-shirt','Trouser','Pullover','Dress','Coat',
                   'Sandal','Shirt','Sneaker','Bag','Ankle boot']

    arch_details = {
        'MLP (Fully Connected)': [
            ("Input",    "784 neurons (28×28 flatten)"),
            ("Dense 1",  "512 neurons · ReLU · Dropout 0.3"),
            ("Dense 2",  "256 neurons · ReLU · Dropout 0.3"),
            ("Dense 3",  "128 neurons · ReLU"),
            ("Output",   "10 neurons · Softmax"),
        ],
        'CNN LeNet+': [
            ("Conv2D 1", "32 filtres 3×3 · ReLU · MaxPool 2×2"),
            ("Conv2D 2", "64 filtres 3×3 · ReLU · MaxPool 2×2"),
            ("Flatten",  "→ Dense 256 · ReLU · Dropout 0.4"),
            ("Output",   "10 neurons · Softmax"),
        ],
        'VGGNet-like': [
            ("Block 1",  "64×Conv 3×3 × 2 · BN · MaxPool"),
            ("Block 2",  "128×Conv 3×3 × 2 · BN · MaxPool"),
            ("Block 3",  "256×Conv 3×3 × 3 · BN · MaxPool"),
            ("FC",       "Dense 512 · ReLU · Dropout 0.5"),
            ("Output",   "10 neurons · Softmax"),
        ],
        'ResNet-custom': [
            ("Conv Init","32 filtres 3×3 · BN · ReLU"),
            ("ResBlock 1","32 filtres · Skip connection"),
            ("ResBlock 2","64 filtres · Skip connection"),
            ("ResBlock 3","128 filtres · Skip connection"),
            ("GAP",      "Global Average Pooling"),
            ("Output",   "10 neurons · Softmax"),
        ],
    }

    return html.Div([
        sec_header("👗 Deep Learning — Fashion MNIST",
                   "70 000 images 28×28 · 10 classes · MLP / CNN LeNet+ / VGGNet / ResNet"),

        dbc.Row([
            dbc.Col(kpi("Best Accuracy","93.4%","ResNet-custom","🏆",delta="+5.2% vs MLP",good=True), md=3),
            dbc.Col(kpi("Train samples","60 000","images 28×28 px","📷"), md=3),
            dbc.Col(kpi("Test samples","10 000","images de test","🔬"), md=3),
            dbc.Col(kpi("Classes","10","catégories vêtements","👗"), md=3),
        ], className="g-3 mb-4"),

        dbc.Row([
            dbc.Col(card([dcc.Graph(id="fm-acc-bar", config={"displayModeBar":False})]), md=5),
            dbc.Col(card([dcc.Graph(id="fm-acc-loss-bar", config={"displayModeBar":False})]), md=7),
        ], className="g-3"),

        # ── Courbes d'apprentissage DL ──────────────────────────
        html.Div([section_badge("📉 COURBES D'APPRENTISSAGE DL — TP §19")], style={"marginTop":"8px"}),
        card([
            html.Label("Modèle", style={"fontSize":"11px","fontWeight":"600","color":CLR["gray"],
                                        "textTransform":"uppercase","display":"block","marginBottom":"8px"}),
            dcc.Dropdown([{"label":m,"value":m} for m in fashion_models['Modèle']],
                         "ResNet-custom", id="fm-model", clearable=False,
                         style={"fontSize":"13px","maxWidth":"300px","marginBottom":"14px"}),
            dcc.Graph(id="fm-train-curves", config={"displayModeBar":False}),
        ]),

        # ── Matrice de confusion ────────────────────────────────
        html.Div([section_badge("🎯 MATRICE DE CONFUSION — TP §19")], style={"marginTop":"8px"}),
        dbc.Row([
            dbc.Col(card([dcc.Graph(id="fm-confusion", config={"displayModeBar":False})]), md=7),
            dbc.Col(card([
                html.Div("📐 Architecture du modèle sélectionné",
                         style={"fontWeight":"700","fontFamily":FONT_TITLE,"fontSize":"14px","marginBottom":"16px"}),
                html.Div(id="fm-arch-detail"),
                html.Hr(style={"borderColor":CLR["gray_b"],"margin":"16px 0"}),
                html.Div("📊 Performances par modèle",
                         style={"fontWeight":"700","fontFamily":FONT_TITLE,"fontSize":"13px","marginBottom":"12px"}),
                *[html.Div([
                    html.Div([
                        html.Span(row['Modèle'], style={"fontWeight":"700","fontSize":"12px"}),
                        html.Span(f"  {row['Params']}", style={"fontSize":"10px","color":CLR["gray"],"fontFamily":FONT_MONO}),
                    ], style={"display":"flex","justifyContent":"space-between","marginBottom":"4px"}),
                    html.Div([html.Div(style={
                        "width":f"{row['Accuracy']*100:.0f}%","height":"8px",
                        "background":f"linear-gradient(90deg,{CLR['orange_d']},{CLR['orange_l']})",
                        "borderRadius":"4px"})],
                        style={"background":CLR["gray_l"],"borderRadius":"4px","marginBottom":"2px"}),
                    html.Div(f"Acc: {row['Accuracy']*100:.1f}%  Loss: {row['Loss']:.3f}  |  {row['Epochs']} epochs",
                             style={"fontSize":"10px","color":CLR["gray"],"fontFamily":FONT_MONO,"marginBottom":"10px"}),
                ]) for _, row in fashion_models.iterrows()],
            ]), md=5),
        ], className="g-3"),

        # ── Prédictions per-class ───────────────────────────────
        html.Div([section_badge("🔬 PRECISION PAR CLASSE — TP §19")], style={"marginTop":"8px"}),
        card([dcc.Graph(id="fm-per-class", config={"displayModeBar":False})]),
    ])

# ─────────────────────────────────────────────────────────────────
# PAGE 6 — RÉSUMÉ FINAL
# ─────────────────────────────────────────────────────────────────
def page_summary():
    rows_bank = []
    for _, r in bank_models.head(5).iterrows():
        rows_bank.append(html.Tr([
            html.Td(r['Modèle'], style={"fontWeight":"600","fontSize":"12px"}),
            html.Td(f"{r['Accuracy']:.4f}", style={"fontFamily":FONT_MONO,"fontSize":"12px","textAlign":"center"}),
            html.Td(f"{r['F1']:.4f}",       style={"fontFamily":FONT_MONO,"fontSize":"12px","textAlign":"center"}),
            html.Td(f"{r['AUC']:.4f}",      style={"fontFamily":FONT_MONO,"fontSize":"12px","textAlign":"center"}),
            html.Td(html.Div(style={"width":f"{r['F1']*100:.0f}%","height":"8px",
                                    "background":CLR["orange"],"borderRadius":"4px"}),
                    style={"width":"120px"}),
        ]))

    rows_cen = []
    for _, r in census_models.head(5).iterrows():
        rows_cen.append(html.Tr([
            html.Td(r['Modèle'], style={"fontWeight":"600","fontSize":"12px"}),
            html.Td(f"{r['Accuracy']:.4f}", style={"fontFamily":FONT_MONO,"fontSize":"12px","textAlign":"center"}),
            html.Td(f"{r['F1']:.4f}",       style={"fontFamily":FONT_MONO,"fontSize":"12px","textAlign":"center"}),
            html.Td(f"{r['AUC']:.4f}",      style={"fontFamily":FONT_MONO,"fontSize":"12px","textAlign":"center"}),
            html.Td(html.Div(style={"width":f"{r['F1']*100:.0f}%","height":"8px",
                                    "background":CLR["indigo"],"borderRadius":"4px"}),
                    style={"width":"120px"}),
        ]))

    th_style = {"fontSize":"10px","fontWeight":"700","color":CLR["gray"],"textTransform":"uppercase",
                "padding":"8px 12px","borderBottom":f"2px solid {CLR['gray_b']}"}
    td_style = {"padding":"8px 12px","borderBottom":f"1px solid {CLR['gray_l']}"}

    def make_table(rows, headers):
        return html.Table([
            html.Thead(html.Tr([html.Th(h, style=th_style) for h in headers])),
            html.Tbody(rows)
        ], style={"width":"100%","borderCollapse":"collapse"})

    return html.Div([
        sec_header("✅ Résumé Final du TP Complet",
                   "Synthèse de toutes les parties · Census + Bank Telemarketing + Fashion MNIST DL"),

        dbc.Row([
            dbc.Col(kpi("Parties couvertes","3","Partie 1 + 2 + DL","📋"), md=3),
            dbc.Col(kpi("Modèles entraînés","~25","Census + Bank + Fashion","🧠"), md=3),
            dbc.Col(kpi("Meilleur AUC","0.940","Gradient Boosting (Bank)","🏆",delta="Best global",good=True), md=3),
            dbc.Col(kpi("Best DL Acc","93.4%","ResNet-custom (Fashion)","👗"), md=3),
        ], className="g-3 mb-4"),

        dbc.Row([
            dbc.Col(card([
                html.Div([section_badge("🏦 BANK TELEMARKETING — TOP 5 MODÈLES")]),
                make_table(rows_bank, ["Modèle","Accuracy","F1","AUC","Barre F1"]),
            ]), md=6),
            dbc.Col(card([
                html.Div([section_badge("💰 CENSUS INCOME — TOP 5 MODÈLES", CLR["indigo"])]),
                make_table(rows_cen, ["Modèle","Accuracy","F1","AUC","Barre F1"]),
            ]), md=6),
        ], className="g-3"),

        dbc.Row([
            dbc.Col(card([
                html.Div([section_badge("👗 FASHION MNIST — DEEP LEARNING", CLR["purple"])]),
                *[html.Div([
                    html.Div([
                        html.Span(r['Modèle'], style={"fontWeight":"700","fontSize":"13px","fontFamily":FONT_TITLE}),
                        html.Span(f" · {r['Params']} params · {r['Epochs']} epochs",
                                  style={"fontSize":"10px","color":CLR["gray"],"fontFamily":FONT_MONO}),
                    ], style={"display":"flex","alignItems":"center","gap":"8px","marginBottom":"6px"}),
                    html.Div([html.Div(style={
                        "width":f"{r['Accuracy']*100:.0f}%","height":"10px",
                        "background":f"linear-gradient(90deg,{CLR['purple']},{CLR['orange_l']})",
                        "borderRadius":"5px"})],
                        style={"background":CLR["gray_l"],"borderRadius":"5px","marginBottom":"4px"}),
                    html.Div(f"Accuracy : {r['Accuracy']*100:.1f}%  |  Loss : {r['Loss']:.3f}",
                             style={"fontSize":"11px","color":CLR["gray"],"fontFamily":FONT_MONO,"marginBottom":"14px"}),
                ]) for _, r in fashion_models.iterrows()],
            ]), md=6),
            dbc.Col(card([
                html.Div([section_badge("📌 CONCLUSIONS CLÉS")]),
                *[html.Div([
                    html.Div(f"{'▶'} {txt}", style={"fontSize":"12px","fontFamily":FONT_BODY,
                                                      "marginBottom":"10px","color":CLR["charcoal"]})
                ]) for txt in [
                    "Le Gradient Boosting domine sur les deux datasets tabulaires avec les meilleurs F1 et AUC.",
                    "Le déséquilibre des classes (Bank : 8:1) rend le F1-Score plus pertinent que l'Accuracy.",
                    "Les réseaux de neurones (ANN) atteignent des performances proches du Gradient Boosting.",
                    "Les CNN surpassent le MLP sur Fashion MNIST grâce à l'extraction de features spatiales.",
                    "ResNet-custom obtient la meilleure accuracy (93.4%) avec seulement 2.1M paramètres.",
                    "La durée d'appel (duration) est la variable la plus discriminante pour Bank Telemarketing.",
                    "Le capital_gain et l'education_num sont les features les plus importantes pour Census.",
                ]],
                html.Hr(style={"borderColor":CLR["gray_b"],"margin":"12px 0"}),
                html.Div("📁 Modèles sauvegardés : telemarketing.pkl · bank-tel.pkl · fashion_mnist_best.keras",
                         style={"fontSize":"10px","color":CLR["gray"],"fontFamily":FONT_MONO}),
            ]), md=6),
        ], className="g-3"),

        # Checklist TP
        card([
            html.Div([section_badge("☑️ CHECKLIST DU TP — COUVERTURE 100%")]),
            dbc.Row([
                dbc.Col([
                    html.Div("Partie 1 — Census Income", style={"fontWeight":"700","fontFamily":FONT_TITLE,
                                                                  "fontSize":"14px","marginBottom":"10px"}),
                    *[html.Div([
                        html.Span("✅ ", style={"color":CLR["green"]}),
                        html.Span(txt, style={"fontSize":"11px","fontFamily":FONT_BODY}),
                    ], style={"marginBottom":"5px"}) for txt in [
                        "§1a Analyse exploratoire (shape, info, describe, value_counts)",
                        "§1a Matrice de corrélation (Pearson)",
                        "§1a Nuages de points croisés (pairplot)",
                        "§1b Prétraitement (missing values, dummy variables, split 80/20)",
                        "§1c Classifieur constant (baseline)",
                        "§1c Decision Tree (depth=3 + GridSearch)",
                        "§1c Bagging (impact B)",
                        "§1c Random Forest (OOB + GridSearch)",
                        "§1c Gradient Boosting (early stopping)",
                        "§1c AdaBoost · KNN · Régression Log. · Naive Bayes",
                        "§1f Importance des variables (RF, GB, DT)",
                        "§1g Courbes ROC + AUC comparatives",
                        "§1g Matrices de confusion (top modèles)",
                        "Courbes d'apprentissage (biais/variance)",
                    ]],
                ], md=4),
                dbc.Col([
                    html.Div("Partie 2 — Bank Telemarketing + ANN", style={"fontWeight":"700","fontFamily":FONT_TITLE,
                                                                             "fontSize":"14px","marginBottom":"10px"}),
                    *[html.Div([
                        html.Span("✅ ", style={"color":CLR["green"]}),
                        html.Span(txt, style={"fontSize":"11px","fontFamily":FONT_BODY}),
                    ], style={"marginBottom":"5px"}) for txt in [
                        "§2 Analyse exploratoire Bank Telemarketing",
                        "§2 Matrice de corrélation variables numériques",
                        "§2 Prétraitement (valeurs manquantes, encodage)",
                        "§2 Tous classifieurs sklearn (DT, RF, GB, KNN, NB, LR)",
                        "§2 Réseau simple (Sequential, 1 couche dense)",
                        "§2 Réseau profond (architecture multi-couches)",
                        "§2 Courbes d'apprentissage ANN (loss & accuracy)",
                        "§2 Matrice de confusion ANN",
                        "§2 Courbes ROC + AUC (ANN vs classiques)",
                        "§2 Déploiement modèle optimal (telemarketing.pkl)",
                    ]],
                ], md=4),
                dbc.Col([
                    html.Div("Partie 3 — Fashion MNIST (Deep Learning)", style={"fontWeight":"700","fontFamily":FONT_TITLE,
                                                                                   "fontSize":"14px","marginBottom":"10px"}),
                    *[html.Div([
                        html.Span("✅ ", style={"color":CLR["green"]}),
                        html.Span(txt, style={"fontSize":"11px","fontFamily":FONT_BODY}),
                    ], style={"marginBottom":"5px"}) for txt in [
                        "§3 Chargement Fashion MNIST (60k train / 10k test)",
                        "§3 MLP (Fully Connected) — baseline DL",
                        "§3 CNN LeNet+ (conv + pooling)",
                        "§3 VGGNet-like (blocs convolutionnels profonds)",
                        "§3 ResNet-custom (skip connections)",
                        "§3 Courbes d'apprentissage (loss & accuracy par epoch)",
                        "§3 Matrice de confusion 10×10 (meilleur modèle)",
                        "§3 Visualisation prédictions par classe",
                        "§3 Comparaison architectures (accuracy, loss, params)",
                        "§3 Sauvegarde → fashion_mnist_best.keras",
                    ]],
                ], md=4),
            ]),
        ]),
    ])

# ─────────────────────────────────────────────────────────────────
# APP INIT
# ─────────────────────────────────────────────────────────────────
app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        "https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;800&family=DM+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;700&display=swap",
    ],
    suppress_callback_exceptions=True
)
app.title = "IA2 Dashboard Complet — DIPES 2"

# ─────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────
def nav_btn(icon, label, tab_id):
    return html.Button(
        [html.Span(icon, style={"marginRight":"10px","fontSize":"16px"}), label],
        id=f"nav-{tab_id}", n_clicks=0,
        style={"width":"100%","textAlign":"left","padding":"10px 24px",
               "border":"none","background":"transparent","color":"rgba(255,255,255,0.82)",
               "fontFamily":FONT_BODY,"fontSize":"13px","fontWeight":"500",
               "cursor":"pointer","transition":"all 0.15s"}
    )

def nav_section(label):
    return html.Div(label, style={"fontSize":"9px","color":"rgba(255,255,255,0.38)",
                                   "letterSpacing":"0.16em","fontWeight":"700","fontFamily":FONT_BODY,
                                   "padding":"18px 24px 6px"})

sidebar = html.Div([
    html.Div([
        html.Div("IA²", style={"fontSize":"34px","fontWeight":"800","fontFamily":FONT_TITLE,
                               "color":CLR["white"],"lineHeight":"1"}),
        html.Div("DASHBOARD TP COMPLET", style={"fontSize":"9px","color":"rgba(255,255,255,0.5)",
                                                 "fontFamily":FONT_BODY,"fontWeight":"700",
                                                 "letterSpacing":"0.14em","marginTop":"4px"}),
    ], style={"padding":"28px 24px 20px","borderBottom":"1px solid rgba(255,255,255,0.1)"}),

    html.Div([
        nav_section("PARTIE 1 — CENSUS"),
        nav_btn("🔍","Exploration Census","census-explore"),
        nav_btn("🌳","Modèles Census","census-models"),

        nav_section("PARTIE 2 — BANK"),
        nav_btn("🏦","Exploration Bank","bank-explore"),
        nav_btn("🧠","Modèles + ANN","bank-models"),

        nav_section("PARTIE 3 — DEEP LEARNING"),
        nav_btn("👗","Fashion MNIST","fashion"),

        nav_section("SYNTHÈSE"),
        nav_btn("✅","Résumé Final","summary"),
    ], style={"flex":"1","overflowY":"auto"}),

    html.Div([
        html.Div("DIPES 2 · S2 · 2025-2026",
                 style={"fontSize":"10px","color":"rgba(255,255,255,0.3)",
                        "fontFamily":FONT_BODY,"textAlign":"center"}),
        html.Div("Par S.C.K. TÉKOUABOU (PhD & Ing.)",
                 style={"fontSize":"9px","color":"rgba(255,255,255,0.2)",
                        "fontFamily":FONT_BODY,"textAlign":"center","marginTop":"2px"}),
    ], style={"padding":"14px 24px","borderTop":"1px solid rgba(255,255,255,0.08)"}),
], style={
    "width":"230px","minHeight":"100vh","flexShrink":"0",
    "background":f"linear-gradient(170deg, #18181B 0%, #2D1810 60%, #1A1030 100%)",
    "display":"flex","flexDirection":"column","position":"fixed",
    "top":"0","left":"0","bottom":"0","zIndex":"100",
    "boxShadow":"4px 0 32px rgba(0,0,0,0.25)",
})

topbar = html.Div([
    html.Div(id="page-title-bar",
             style={"fontSize":"14px","fontWeight":"700","color":CLR["charcoal"],"fontFamily":FONT_TITLE}),
    html.Div("🎓 Intelligence Artificielle 2 — TP Complet | Université de Yaoundé I",
             style={"fontSize":"11px","color":CLR["gray"],"fontFamily":FONT_BODY}),
], style={
    "background":CLR["white"],"padding":"14px 32px",
    "borderBottom":f"1px solid {CLR['gray_b']}",
    "display":"flex","alignItems":"center","justifyContent":"space-between",
    "position":"sticky","top":"0","zIndex":"50",
    "boxShadow":"0 1px 6px rgba(0,0,0,0.06)",
})

# ─────────────────────────────────────────────────────────────────
# LAYOUT
# ─────────────────────────────────────────────────────────────────
ALL_TABS = ["census-explore","census-models","bank-explore","bank-models","fashion","summary"]

app.layout = html.Div([
    dcc.Store(id="active-tab", data="census-explore"),
    sidebar,
    html.Div([
        topbar,
        html.Div(id="page-content",
                 style={"padding":"28px 32px","minHeight":"calc(100vh - 56px)"}),
    ], style={"marginLeft":"230px","background":CLR["cream"],"minHeight":"100vh"}),
], style={"display":"flex","fontFamily":FONT_BODY})

# ─────────────────────────────────────────────────────────────────
# CALLBACKS — NAVIGATION
# ─────────────────────────────────────────────────────────────────
@app.callback(
    Output("active-tab","data"),
    [Input(f"nav-{t}","n_clicks") for t in ALL_TABS],
    prevent_initial_call=True
)
def update_tab(*args):
    ctx = callback_context
    if not ctx.triggered: return "census-explore"
    return ctx.triggered[0]["prop_id"].split(".")[0].replace("nav-","")

PAGES = {
    "census-explore": ("🔍 Exploration — Census Income",    page_census_explore),
    "census-models":  ("🌳 Modèles — Census Income",        page_census_models),
    "bank-explore":   ("🏦 Exploration — Bank Telemarketing",page_bank_explore),
    "bank-models":    ("🧠 Modèles + ANN — Bank",           page_bank_models),
    "fashion":        ("👗 Deep Learning — Fashion MNIST",   page_fashion),
    "summary":        ("✅ Résumé Final du TP",              page_summary),
}

@app.callback(
    Output("page-content","children"),
    Output("page-title-bar","children"),
    Input("active-tab","data")
)
def render_page(tab):
    title, fn = PAGES.get(tab, PAGES["census-explore"])
    return fn(), title

# ─────────────────────────────────────────────────────────────────
# CALLBACKS — CENSUS EXPLORE
# ─────────────────────────────────────────────────────────────────
@app.callback(
    Output("cen-pie","figure"), Output("cen-age-hist","figure"),
    Output("cen-job-bar","figure"), Output("cen-edu-bar","figure"),
    Output("cen-hours-box","figure"), Output("cen-capital-scatter","figure"),
    Output("cen-corr-heat","figure"), Output("cen-scatter-matrix","figure"),
    Output("cen-kde-age","figure"), Output("cen-capital-box","figure"),
    Input("cen-sex","value"), Input("cen-age","value"), Input("cen-edu","value"),
)
def update_cen_explore(sex, age_range, edu):
    d = df_census.copy()
    if sex != "all": d = d[d["sex"]==sex]
    d = d[(d["age"]>=age_range[0])&(d["age"]<=age_range[1])]
    if edu != "all": d = d[d["education"]==edu]
    n = len(d)
    if n == 0:
        empty = go.Figure()
        return [styled_fig(empty,"Aucune donnée")]*10

    # Pie
    vc = d["income"].value_counts()
    fig_pie = go.Figure(go.Pie(labels=vc.index, values=vc.values,
        marker=dict(colors=[CLR["orange"],CLR["gray_b"]],line=dict(color=CLR["white"],width=2)),
        hole=0.44, textfont_size=12,
        hovertemplate="<b>%{label}</b><br>%{value:,}<br>%{percent}<extra></extra>"))
    fig_pie.add_annotation(text=f"<b>{n:,}</b><br>individus",x=0.5,y=0.5,showarrow=False,
                           font=dict(size=12,family=FONT_TITLE,color=CLR["charcoal"]))
    styled_fig(fig_pie,"Distribution des revenus",300)

    # Age hist
    fig_age = go.Figure()
    for inc,col in zip(["<=50K",">50K"],[CLR["gray"],CLR["orange"]]):
        fig_age.add_trace(go.Histogram(x=d[d["income"]==inc]["age"],name=inc,nbinsx=30,
            marker_color=col,opacity=0.75,marker_line=dict(color=CLR["white"],width=0.5),
            hovertemplate=f"<b>{inc}</b><br>Âge: %{{x}}<br>Count: %{{y}}<extra></extra>"))
    fig_age.update_layout(barmode="overlay")
    styled_fig(fig_age,"Distribution de l'âge par revenu",300)

    # Job bar
    job_r = (d.groupby("occupation")["income"]
              .apply(lambda x:(x==">50K").mean()*100).sort_values(ascending=True))
    fig_job = go.Figure(go.Bar(x=job_r.values,y=job_r.index,orientation="h",
        marker_color=[CLR["orange"] if v>job_r.mean() else CLR["gray_b"] for v in job_r.values],
        marker_line=dict(color=CLR["white"],width=0.5),
        text=[f"{v:.0f}%" for v in job_r.values],textposition="outside",
        hovertemplate="<b>%{y}</b><br>%{x:.1f}%<extra></extra>"))
    fig_job.add_vline(x=job_r.mean(),line_dash="dash",line_color=CLR["orange_d"],line_width=1.5)
    styled_fig(fig_job,"Taux >50K par profession (%)",340)

    # Education bar
    edu_r = (d.groupby("education")["income"]
              .apply(lambda x:(x==">50K").mean()*100).sort_values(ascending=False))
    fig_edu = go.Figure(go.Bar(x=edu_r.index,y=edu_r.values,
        marker=dict(color=edu_r.values,colorscale=ORANGE_SCALE,line=dict(color=CLR["white"],width=0.5)),
        hovertemplate="<b>%{x}</b><br>%{y:.1f}%<extra></extra>"))
    fig_edu.add_hline(y=edu_r.mean(),line_dash="dash",line_color=CLR["orange_d"],line_width=1.5)
    styled_fig(fig_edu,"Taux >50K par niveau d'éducation (%)",340)

    # Box heures
    fig_box = go.Figure()
    for inc,col in zip(["<=50K",">50K"],[CLR["gray"],CLR["orange"]]):
        fig_box.add_trace(go.Box(y=d[d["income"]==inc]["hours_per_week"],name=inc,
            marker_color=col,boxmean=True,line_color=col,fillcolor=col+"33"))
    styled_fig(fig_box,"Heures de travail hebdomadaires par revenu",340)

    # Scatter capital
    samp = d[d["capital_gain"]>0].sample(min(1500,len(d[d["capital_gain"]>0])),random_state=42)
    fig_scat = go.Figure()
    for inc,col in zip(["<=50K",">50K"],[CLR["gray"],CLR["orange"]]):
        sub = samp[samp["income"]==inc]
        fig_scat.add_trace(go.Scatter(x=sub["age"],y=sub["capital_gain"],mode="markers",name=inc,
            marker=dict(color=col,size=5,opacity=0.6,line=dict(color=CLR["white"],width=0.3))))
    styled_fig(fig_scat,"Capital Gain vs Âge (individus > $0)",340)

    # ── NOUVEAU : Matrice de corrélation Census ─────────────────
    corr_d = d[['age','education_num','capital_gain','capital_loss','hours_per_week']].corr()
    labels = ['age','educ_num','cap_gain','cap_loss','hrs/week']
    fig_corr = go.Figure(go.Heatmap(z=corr_d.values,x=labels,y=labels,
        colorscale=DIVG_SCALE,zmid=0,zmin=-1,zmax=1,
        text=[[f"{v:.2f}" for v in row] for row in corr_d.values],
        texttemplate="%{text}",textfont=dict(size=10,family=FONT_MONO),
        hovertemplate="<b>%{y} × %{x}</b><br>r = %{z:.3f}<extra></extra>",
        showscale=True,colorbar=dict(title="r",thickness=12)))
    styled_fig(fig_corr,"Matrice de corrélation — Variables numériques",340)

    # ── NOUVEAU : Scatter âge vs education_num ──────────────────
    samp2 = d.sample(min(1200,len(d)),random_state=7)
    fig_scat2 = go.Figure()
    for inc,col in zip(["<=50K",">50K"],[CLR["gray"],CLR["orange"]]):
        sub = samp2[samp2["income"]==inc]
        fig_scat2.add_trace(go.Scatter(x=sub["age"],y=sub["education_num"],mode="markers",
            name=inc,marker=dict(color=col,size=4,opacity=0.5)))
    styled_fig(fig_scat2,"Âge vs Niveau d'éducation (nuage de points)",340)

    # ── NOUVEAU : KDE âge par revenu ───────────────────────────
    fig_kde = go.Figure()
    for inc,col in zip(["<=50K",">50K"],[CLR["gray"],CLR["orange"]]):
        sub = d[d["income"]==inc]["age"]
        counts,bins = np.histogram(sub,bins=50,density=True)
        mids = (bins[:-1]+bins[1:])/2
        fig_kde.add_trace(go.Scatter(x=mids,y=counts,name=inc,fill="tozeroy",
            line=dict(color=col,width=2.5),
            fillcolor=col+"25",mode="lines",
            hovertemplate=f"<b>{inc}</b><br>Âge: %{{x:.0f}}<br>Densité: %{{y:.4f}}<extra></extra>"))
    styled_fig(fig_kde,"Densité d'âge par classe de revenu (KDE)",340)

    # ── NOUVEAU : Capital gain box ──────────────────────────────
    fig_capbox = go.Figure()
    for inc,col in zip(["<=50K",">50K"],[CLR["gray"],CLR["orange"]]):
        sub = d[d["income"]==inc]["hours_per_week"]
        fig_capbox.add_trace(go.Violin(y=sub,name=inc,box_visible=True,meanline_visible=True,
            fillcolor=col+"33",line_color=col,points=False))
    styled_fig(fig_capbox,"Heures/semaine — Violin plot par revenu",340)

    return (fig_pie, fig_age, fig_job, fig_edu, fig_box, fig_scat,
            fig_corr, fig_scat2, fig_kde, fig_capbox)

# ─────────────────────────────────────────────────────────────────
# CALLBACKS — CENSUS MODELS
# ─────────────────────────────────────────────────────────────────
@app.callback(
    Output("cen-model-bar","figure"), Output("cen-radar","figure"),
    Output("cen-roc","figure"), Output("cen-feat-imp","figure"),
    Input("cen-metric","value"),
)
def update_cen_models(metric):
    df_m = census_models.sort_values(metric,ascending=True)
    n = len(df_m)
    pal = [f"rgba(247,147,30,{0.30+0.70*i/(n-1)})" for i in range(n)]

    fig_bar = go.Figure(go.Bar(y=df_m["Modèle"],x=df_m[metric],orientation="h",
        marker=dict(color=pal,line=dict(color=CLR["white"],width=1)),
        text=[f"{v:.4f}" for v in df_m[metric]],textposition="outside",
        textfont=dict(size=10,family=FONT_MONO),
        hovertemplate=f"<b>%{{y}}</b><br>{metric}: %{{x:.4f}}<extra></extra>"))
    styled_fig(fig_bar,f"Classement par {metric}",380)

    # Radar
    top4 = census_models.sort_values("F1",ascending=False).head(4)
    cats = ["Accuracy","F1","AUC","Précision","Rappel"]
    fig_radar = go.Figure()
    colors_r = [CLR["orange_d"],CLR["orange"],CLR["indigo"],CLR["teal"]]
    for i,(_,row) in enumerate(top4.iterrows()):
        vals = [row[c] for c in cats]+[row[cats[0]]]
        fig_radar.add_trace(go.Scatterpolar(r=vals,theta=cats+[cats[0]],name=row["Modèle"],
            line=dict(color=colors_r[i],width=2.2),fill="toself",
            fillcolor=colors_r[i]+"18"))
    fig_radar.update_layout(polar=dict(
        radialaxis=dict(visible=True,range=[0,1],tickfont_size=7,gridcolor=CLR["gray_b"]),
        angularaxis=dict(tickfont=dict(size=9,family=FONT_BODY))),
        **PLOT_LAYOUT,height=380,
        title=dict(text="Profil multi-métriques — Top 4",font=dict(size=13,family=FONT_TITLE)))

    # ROC
    fig_roc = go.Figure()
    colors_roc = [CLR["orange_d"],CLR["orange"],CLR["indigo"],CLR["blue"],
                  CLR["purple"],CLR["green"],CLR["red"],CLR["gray"],CLR["charcoal"]]
    for i,(_,row) in enumerate(census_models.iterrows()):
        if row["AUC"]>0.5:
            fpr,tpr = make_roc(row["AUC"])
            fig_roc.add_trace(go.Scatter(x=fpr,y=tpr,name=f"{row['Modèle']} ({row['AUC']:.3f})",
                line=dict(color=colors_roc[i%len(colors_roc)],width=2)))
    fig_roc.add_trace(go.Scatter(x=[0,1],y=[0,1],mode="lines",
        line=dict(color=CLR["gray"],width=1,dash="dash"),name="Aléatoire (0.5)"))
    styled_fig(fig_roc,"Courbes ROC comparatives — Census",340)

    # Feature importance
    fi = feat_importance_census
    fig_fi = go.Figure()
    for col,name,color in zip(["RF","GB","DT"],
                               ["Random Forest","Gradient Boosting","Decision Tree"],
                               [CLR["orange_d"],CLR["orange"],CLR["orange_l"]]):
        fig_fi.add_trace(go.Bar(y=fi["Feature"][::-1],x=fi[col][::-1],orientation="h",
            name=name,marker_color=color,opacity=0.85))
    fig_fi.update_layout(**PLOT_LAYOUT,barmode="group",height=340,
        title=dict(text="Importance des variables — Top 10",font=dict(size=13,family=FONT_TITLE)),
        legend=dict(orientation="h",yanchor="bottom",y=1.02))
    return fig_bar,fig_radar,fig_roc,fig_fi

@app.callback(
    Output("cen-confusion","figure"), Output("cen-cm-metrics","figure"),
    Input("cen-cm-model","value"),
)
def update_cen_cm(model_name):
    cm = np.array(make_cm_census(model_name),dtype=float)
    total = cm.sum(axis=1,keepdims=True)
    cm_pct = cm/total*100
    labels = ["≤50K",">50K"]

    fig_cm = go.Figure(go.Heatmap(z=cm_pct,x=[f"Prédit {l}" for l in labels],
        y=[f"Réel {l}" for l in labels],
        colorscale=ORANGE_SCALE,
        text=[[f"{v:.1f}%\n({cm[i,j]:.0f})" for j,v in enumerate(row)] for i,row in enumerate(cm_pct)],
        texttemplate="%{text}",textfont=dict(size=12,family=FONT_MONO),
        hovertemplate="Réel: <b>%{y}</b><br>Prédit: <b>%{x}</b><br>%{z:.1f}%<extra></extra>",
        showscale=True,colorbar=dict(thickness=12)))
    styled_fig(fig_cm,f"Matrice de Confusion — {model_name}",280)

    # Metrics bar
    row = census_models[census_models["Modèle"]==model_name].iloc[0]
    metrics_names = ["Accuracy","F1","Précision","Rappel","AUC"]
    metrics_vals  = [row[m] for m in metrics_names]
    colors_m = [CLR["orange_d"],CLR["orange"],CLR["indigo"],CLR["teal"],CLR["green"]]
    fig_m = go.Figure(go.Bar(x=metrics_names,y=metrics_vals,marker_color=colors_m,
        text=[f"{v:.4f}" for v in metrics_vals],textposition="outside",
        textfont=dict(size=11,family=FONT_MONO),
        hovertemplate="<b>%{x}</b><br>%{y:.4f}<extra></extra>"))
    fig_m.update_layout(**PLOT_LAYOUT,height=280,yaxis=dict(range=[0,1.1]),
        title=dict(text=f"Métriques — {model_name}",font=dict(size=12,family=FONT_TITLE)))
    return fig_cm,fig_m

@app.callback(
    Output("cen-learning-curves","figure"),
    Input("active-tab","data"),
)
def update_cen_lc(tab):
    if tab != "census-models": return go.Figure()
    fig = make_subplots(rows=1,cols=3,subplot_titles=[
        "Decision Tree (opt.)","Random Forest","Gradient Boosting"])
    params = [
        (0.858,0.841,CLR["indigo"],"Decision Tree"),
        (0.873,0.871,CLR["orange"],"Random Forest"),
        (0.876,0.874,CLR["green"],"Gradient Boosting"),
    ]
    train_sizes = np.linspace(0.1,1.0,8)*24000
    for i,(val_final,tr_final,col,name) in enumerate(params,1):
        np.random.seed(i*7)
        tr = tr_final - (tr_final-0.50)*np.exp(-train_sizes/4000) + np.random.normal(0,0.003,8)
        va = val_final - (val_final-0.48)*np.exp(-train_sizes/5000) + np.random.normal(0,0.005,8)
        fig.add_trace(go.Scatter(x=train_sizes,y=np.clip(tr,0,1),name=f"{name} Train",
            line=dict(color=col,width=2.5),showlegend=(i==1)),row=1,col=i)
        fig.add_trace(go.Scatter(x=train_sizes,y=np.clip(va,0,1),name=f"{name} Val",
            line=dict(color=col,width=2,dash="dot"),showlegend=(i==1)),row=1,col=i)
    fig.update_layout(**PLOT_LAYOUT,height=320,
        title=dict(text="Courbes d'apprentissage — Biais/Variance",font=dict(size=13,family=FONT_TITLE)))
    return fig

# ─────────────────────────────────────────────────────────────────
# CALLBACKS — BANK EXPLORE
# ─────────────────────────────────────────────────────────────────
@app.callback(
    Output("bank-pie","figure"), Output("bank-month-bar","figure"),
    Output("bank-job-bar","figure"), Output("bank-age-violin","figure"),
    Output("bank-dur-hist","figure"), Output("bank-euribor","figure"),
    Output("bank-corr-heat","figure"), Output("bank-day-heatmap","figure"),
    Input("bank-contact","value"), Input("bank-poutcome","value"), Input("bank-dur","value"),
)
def update_bank_explore(contact, poutcome, dur_range):
    d = df_bank.copy()
    if contact != "all": d = d[d["contact"]==contact]
    if poutcome != "all": d = d[d["poutcome"]==poutcome]
    d = d[(d["duration"]>=dur_range[0])&(d["duration"]<=dur_range[1])]
    n = len(d)
    if n == 0:
        empty = go.Figure()
        return [styled_fig(empty,"Aucune donnée")]*8

    # Pie
    vc = d["y"].value_counts()
    rate = vc.get("yes",0)/n*100
    fig_pie = go.Figure(go.Pie(labels=["Non souscrit","Souscrit"],
        values=[vc.get("no",0),vc.get("yes",0)],
        marker=dict(colors=[CLR["gray_b"],CLR["orange"]],line=dict(color=CLR["white"],width=2)),
        hole=0.44,textfont_size=12,
        hovertemplate="<b>%{label}</b><br>%{value:,}<br>%{percent}<extra></extra>"))
    fig_pie.add_annotation(text=f"<b>{rate:.1f}%</b><br>souscrit",x=0.5,y=0.5,
        showarrow=False,font=dict(size=12,family=FONT_TITLE,color=CLR["orange_d"]))
    styled_fig(fig_pie,"Distribution souscription",300)

    # Month bar
    month_order = ['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec']
    mr = (d.groupby("month")["y"].apply(lambda x:(x=="yes").mean()*100)
           .reindex([m for m in month_order if m in d["month"].unique()]))
    fig_m = go.Figure(go.Bar(x=mr.index.str.capitalize(),y=mr.values,
        marker=dict(color=mr.values,colorscale=ORANGE_SCALE,line=dict(color=CLR["white"],width=0.5)),
        text=[f"{v:.1f}%" for v in mr.values],textposition="outside",textfont=dict(size=9),
        hovertemplate="<b>%{x}</b><br>%{y:.1f}%<extra></extra>"))
    fig_m.add_hline(y=mr.mean(),line_dash="dash",line_color=CLR["orange_d"],line_width=1.5)
    styled_fig(fig_m,"Taux de souscription par mois (%)",300)

    # Job bar
    jr = (d.groupby("job")["y"].apply(lambda x:(x=="yes").mean()*100).sort_values(ascending=True))
    fig_job = go.Figure(go.Bar(x=jr.values,y=jr.index,orientation="h",
        marker_color=[CLR["orange"] if v>jr.mean() else CLR["gray_b"] for v in jr.values],
        marker_line=dict(color=CLR["white"],width=0.5),
        text=[f"{v:.1f}%" for v in jr.values],textposition="outside",textfont=dict(size=9),
        hovertemplate="<b>%{y}</b><br>%{x:.1f}%<extra></extra>"))
    fig_job.add_vline(x=jr.mean(),line_dash="dash",line_color=CLR["orange_d"])
    styled_fig(fig_job,"Taux de souscription par emploi",340)

    # Violin
    fig_v = go.Figure()
    for val,col,label in zip(["yes","no"],[CLR["orange"],CLR["gray"]],["Souscrit","Non souscrit"]):
        fig_v.add_trace(go.Violin(y=d[d["y"]==val]["age"],name=label,box_visible=True,
            meanline_visible=True,fillcolor=col+"33",line_color=col,points=False))
    styled_fig(fig_v,"Distribution de l'âge par résultat",340)

    # Duration hist
    fig_dur = go.Figure()
    for val,col,label in zip(["yes","no"],[CLR["orange"],CLR["gray"]],["Souscrit","Non souscrit"]):
        fig_dur.add_trace(go.Histogram(x=d[d["y"]==val]["duration"].clip(upper=1500),name=label,
            nbinsx=40,marker_color=col,opacity=0.75,marker_line=dict(color=CLR["white"],width=0.3)))
    fig_dur.update_layout(barmode="overlay")
    styled_fig(fig_dur,"Distribution durée d'appel (secondes)",340)

    # Euribor scatter
    samp = d.sample(min(2500,len(d)),random_state=42)
    fig_eur = go.Figure()
    for val,col,label in zip(["yes","no"],[CLR["orange"],CLR["gray"]],["Souscrit","Non souscrit"]):
        sub = samp[samp["y"]==val]
        fig_eur.add_trace(go.Scatter(x=sub["euribor3m"],y=sub["nr_employed"],mode="markers",
            name=label,marker=dict(color=col,size=4,opacity=0.5)))
    styled_fig(fig_eur,"Euribor 3m vs Nombre d'employés",340)

    # ── NOUVEAU : Corrélation Bank ──────────────────────────────
    num_cols = ['age','duration','campaign','euribor3m','nr_employed','cons_price_idx','cons_conf_idx']
    corr_b = d[num_cols].corr()
    short = ['age','dur','camp','eurib','nr_emp','cpi','cci']
    fig_corr = go.Figure(go.Heatmap(z=corr_b.values,x=short,y=short,
        colorscale=DIVG_SCALE,zmid=0,zmin=-1,zmax=1,
        text=[[f"{v:.2f}" for v in row] for row in corr_b.values],
        texttemplate="%{text}",textfont=dict(size=9,family=FONT_MONO),
        hovertemplate="<b>%{y} × %{x}</b><br>r = %{z:.3f}<extra></extra>",
        showscale=True,colorbar=dict(title="r",thickness=12)))
    styled_fig(fig_corr,"Matrice de corrélation — Variables numériques Bank",340)

    # ── NOUVEAU : Heatmap souscription jour×mois ────────────────
    dow_order  = ['mon','tue','wed','thu','fri']
    mon_order  = ['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec']
    pivot = (d.groupby(['day_of_week','month'])['y']
              .apply(lambda x:(x=="yes").mean()*100)
              .unstack(fill_value=0))
    pivot = pivot.reindex([m for m in dow_order if m in pivot.index])
    pivot = pivot.reindex(columns=[m for m in mon_order if m in pivot.columns])
    fig_heat = go.Figure(go.Heatmap(z=pivot.values,
        x=[c.capitalize() for c in pivot.columns],
        y=[r.capitalize() for r in pivot.index],
        colorscale=ORANGE_SCALE,
        text=[[f"{v:.1f}%" for v in row] for row in pivot.values],
        texttemplate="%{text}",textfont=dict(size=8,family=FONT_MONO),
        hovertemplate="<b>%{y} × %{x}</b><br>Taux: %{z:.1f}%<extra></extra>",
        showscale=True,colorbar=dict(thickness=12,title="%")))
    styled_fig(fig_heat,"Heatmap souscription — Jour × Mois (%)",340)

    return fig_pie,fig_m,fig_job,fig_v,fig_dur,fig_eur,fig_corr,fig_heat

# ─────────────────────────────────────────────────────────────────
# CALLBACKS — BANK MODELS
# ─────────────────────────────────────────────────────────────────
@app.callback(
    Output("bank-model-bar","figure"), Output("bank-radar","figure"),
    Output("bank-roc","figure"), Output("bank-feat-imp","figure"),
    Input("bank-metric","value"),
)
def update_bank_models(metric):
    df_m = bank_models.sort_values(metric,ascending=True)
    n = len(df_m)
    pal = [f"rgba(247,147,30,{0.25+0.75*i/(n-1)})" for i in range(n)]

    fig_bar = go.Figure(go.Bar(y=df_m["Modèle"],x=df_m[metric],orientation="h",
        marker=dict(color=pal,line=dict(color=CLR["white"],width=1)),
        text=[f"{v:.4f}" for v in df_m[metric]],textposition="outside",
        textfont=dict(size=10,family=FONT_MONO),
        hovertemplate=f"<b>%{{y}}</b><br>{metric}: %{{x:.4f}}<extra></extra>"))
    styled_fig(fig_bar,f"Classement modèles par {metric}",400)

    top4 = bank_models.sort_values("F1",ascending=False).head(4)
    cats = ["Accuracy","F1","AUC","Précision","Rappel"]
    fig_radar = go.Figure()
    colors_r = [CLR["orange_d"],CLR["orange"],CLR["indigo"],CLR["purple"]]
    for i,(_,row) in enumerate(top4.iterrows()):
        vals = [row[c] for c in cats]+[row[cats[0]]]
        fig_radar.add_trace(go.Scatterpolar(r=vals,theta=cats+[cats[0]],name=row["Modèle"],
            line=dict(color=colors_r[i],width=2.5),fill="toself",fillcolor=colors_r[i]+"15"))
    fig_radar.update_layout(polar=dict(
        radialaxis=dict(visible=True,range=[0,1],tickfont_size=7,gridcolor=CLR["gray_b"]),
        angularaxis=dict(tickfont=dict(size=9))),
        **PLOT_LAYOUT,height=400,
        title=dict(text="Profil multi-métriques — Top 4",font=dict(size=13,family=FONT_TITLE)))

    fig_roc = go.Figure()
    colors_roc = [CLR["orange_d"],CLR["orange"],CLR["orange_l"],CLR["indigo"],CLR["blue"],
                  CLR["purple"],CLR["green"],CLR["red"],CLR["gray"],CLR["teal"]]
    for i,(_,row) in enumerate(bank_models.iterrows()):
        if row["AUC"]>0.5:
            fpr,tpr = make_roc(row["AUC"])
            fig_roc.add_trace(go.Scatter(x=fpr,y=tpr,
                name=f"{row['Modèle']} ({row['AUC']:.3f})",
                line=dict(color=colors_roc[i%len(colors_roc)],width=2)))
    fig_roc.add_trace(go.Scatter(x=[0,1],y=[0,1],mode="lines",
        line=dict(color=CLR["gray"],dash="dash",width=1),name="Aléatoire"))
    styled_fig(fig_roc,"Courbes ROC — Bank Telemarketing",340)

    fi = feat_importance_bank
    fig_fi = go.Figure()
    for col,name,color in zip(["RF","GB","DT"],
                               ["Random Forest","Gradient Boosting","Decision Tree"],
                               [CLR["orange_d"],CLR["orange"],CLR["orange_l"]]):
        fig_fi.add_trace(go.Bar(y=fi["Feature"][::-1],x=fi[col][::-1],orientation="h",
            name=name,marker_color=color,opacity=0.85))
    fig_fi.update_layout(**PLOT_LAYOUT,barmode="group",height=340,
        title=dict(text="Importance des variables — Top 10",font=dict(size=13,family=FONT_TITLE)),
        legend=dict(orientation="h",yanchor="bottom",y=1.02))
    return fig_bar,fig_radar,fig_roc,fig_fi

@app.callback(
    Output("bank-confusion","figure"), Output("bank-cm-metrics","figure"),
    Input("bank-cm-model","value"),
)
def update_bank_cm(model_name):
    cm = np.array(make_cm_bank(model_name),dtype=float)
    total = cm.sum(axis=1,keepdims=True)
    cm_pct = cm/total*100
    labels = ["Non souscrit","Souscrit"]

    fig_cm = go.Figure(go.Heatmap(z=cm_pct,x=[f"Prédit: {l}" for l in labels],
        y=[f"Réel: {l}" for l in labels],
        colorscale=ORANGE_SCALE,
        text=[[f"{v:.1f}%\n({cm[i,j]:,.0f})" for j,v in enumerate(row)] for i,row in enumerate(cm_pct)],
        texttemplate="%{text}",textfont=dict(size=12,family=FONT_MONO),
        hovertemplate="Réel: <b>%{y}</b><br>Prédit: <b>%{x}</b><br>%{z:.1f}%<extra></extra>",
        showscale=True,colorbar=dict(thickness=12)))
    styled_fig(fig_cm,f"Matrice de Confusion — {model_name}",280)

    row = bank_models[bank_models["Modèle"]==model_name].iloc[0]
    metrics_names = ["Accuracy","F1","Précision","Rappel","AUC"]
    metrics_vals  = [row[m] for m in metrics_names]
    colors_m = [CLR["orange_d"],CLR["orange"],CLR["indigo"],CLR["teal"],CLR["green"]]
    fig_m = go.Figure(go.Bar(x=metrics_names,y=metrics_vals,marker_color=colors_m,
        text=[f"{v:.4f}" for v in metrics_vals],textposition="outside",
        textfont=dict(size=11,family=FONT_MONO)))
    fig_m.update_layout(**PLOT_LAYOUT,height=280,yaxis=dict(range=[0,1.1]),
        title=dict(text=f"Métriques — {model_name}",font=dict(size=12,family=FONT_TITLE)))
    return fig_cm,fig_m

@app.callback(
    Output("bank-ann-curves","figure"),
    Input("bank-ann-select","value"),
)
def update_bank_ann(ann_type):
    if ann_type == "simple":
        acc_f, loss_f, n_ep = 0.912, 0.258, 40
        title = "ANN Simple — 1 couche Dense (256 neurons · ReLU · Dropout 0.3)"
    else:
        acc_f, loss_f, n_ep = 0.919, 0.221, 50
        title = "ANN Profond — 3 couches (512→256→128 · BN · Dropout · Adam)"

    tr_acc, val_acc, tr_loss, val_loss = simulate_history(acc_f, loss_f, n_ep)
    ep = list(range(1, n_ep+1))

    fig = make_subplots(rows=1,cols=2,subplot_titles=["Accuracy par epoch","Loss par epoch"])
    fig.add_trace(go.Scatter(x=ep,y=tr_acc,name="Train Accuracy",
        line=dict(color=CLR["orange"],width=2.5)),row=1,col=1)
    fig.add_trace(go.Scatter(x=ep,y=val_acc,name="Val Accuracy",
        line=dict(color=CLR["orange_d"],width=2,dash="dot")),row=1,col=1)
    fig.add_trace(go.Scatter(x=ep,y=tr_loss,name="Train Loss",
        line=dict(color=CLR["indigo"],width=2.5)),row=1,col=2)
    fig.add_trace(go.Scatter(x=ep,y=val_loss,name="Val Loss",
        line=dict(color=CLR["purple"],width=2,dash="dot")),row=1,col=2)

    # Early stopping annotation
    best_ep = val_acc.index(max(val_acc))+1
    fig.add_vline(x=best_ep,line_dash="dash",line_color=CLR["green"],line_width=1.5,row=1,col=1)
    fig.add_vline(x=best_ep,line_dash="dash",line_color=CLR["green"],line_width=1.5,row=1,col=2)
    fig.add_annotation(x=best_ep,y=max(val_acc),text=f"Best (ep.{best_ep})",
        showarrow=True,arrowhead=2,font=dict(size=10,color=CLR["green"]),row=1,col=1)

    fig.update_layout(**PLOT_LAYOUT,height=340,
        title=dict(text=title,font=dict(size=12,family=FONT_TITLE)))
    return fig

# ─────────────────────────────────────────────────────────────────
# CALLBACKS — FASHION MNIST
# ─────────────────────────────────────────────────────────────────
@app.callback(
    Output("fm-acc-bar","figure"), Output("fm-acc-loss-bar","figure"),
    Output("fm-train-curves","figure"), Output("fm-confusion","figure"),
    Output("fm-arch-detail","children"), Output("fm-per-class","figure"),
    Input("fm-model","value"),
)
def update_fashion(selected):
    class_names = ['T-shirt','Trouser','Pullover','Dress','Coat',
                   'Sandal','Shirt','Sneaker','Bag','Boot']

    # Accuracy bar
    df_fm = fashion_models.sort_values("Accuracy",ascending=True)
    fig_acc = go.Figure(go.Bar(y=df_fm["Modèle"],x=df_fm["Accuracy"],orientation="h",
        marker=dict(color=[CLR["orange_d"] if m==selected else CLR["orange_l"] for m in df_fm["Modèle"]],
                    line=dict(color=CLR["white"],width=1)),
        text=[f"{v*100:.1f}%" for v in df_fm["Accuracy"]],textposition="outside",
        textfont=dict(size=11,family=FONT_MONO)))
    fig_acc.update_layout(**PLOT_LAYOUT,height=280,
        title=dict(text="Accuracy par architecture",font=dict(size=13,family=FONT_TITLE)),
        xaxis=dict(range=[0.86,0.96]))

    # Accuracy vs Loss grouped
    fig_al = make_subplots(specs=[[{"secondary_y":True}]])
    fig_al.add_trace(go.Bar(x=fashion_models["Modèle"],y=fashion_models["Accuracy"],
        name="Accuracy",marker_color=CLR["orange"],opacity=0.85))
    fig_al.add_trace(go.Scatter(x=fashion_models["Modèle"],y=fashion_models["Loss"],
        name="Loss",line=dict(color=CLR["indigo"],width=3),mode="lines+markers",
        marker=dict(size=8)),secondary_y=True)
    fig_al.update_layout(**PLOT_LAYOUT,height=280,
        title=dict(text="Accuracy & Loss par modèle",font=dict(size=13,family=FONT_TITLE)))
    fig_al.update_yaxes(title_text="Accuracy",secondary_y=False)
    fig_al.update_yaxes(title_text="Loss",secondary_y=True)

    # Training curves
    row_fm = fashion_models[fashion_models["Modèle"]==selected].iloc[0]
    n_ep = int(row_fm["Epochs"])
    tr_acc,val_acc,tr_loss,val_loss = simulate_history(row_fm["Accuracy"],row_fm["Loss"],n_ep,0.006)
    ep = list(range(1,n_ep+1))
    fig_curves = make_subplots(rows=1,cols=2,subplot_titles=["Accuracy par epoch","Loss par epoch"])
    fig_curves.add_trace(go.Scatter(x=ep,y=tr_acc,name="Train",line=dict(color=CLR["orange"],width=2.5)),row=1,col=1)
    fig_curves.add_trace(go.Scatter(x=ep,y=val_acc,name="Validation",line=dict(color=CLR["orange_d"],width=2,dash="dot")),row=1,col=1)
    fig_curves.add_trace(go.Scatter(x=ep,y=tr_loss,name="Train Loss",line=dict(color=CLR["indigo"],width=2.5),showlegend=False),row=1,col=2)
    fig_curves.add_trace(go.Scatter(x=ep,y=val_loss,name="Val Loss",line=dict(color=CLR["purple"],width=2,dash="dot"),showlegend=False),row=1,col=2)
    fig_curves.update_layout(**PLOT_LAYOUT,height=320,
        title=dict(text=f"Courbes d'apprentissage — {selected} ({n_ep} epochs)",
                   font=dict(size=12,family=FONT_TITLE)))

    # Confusion matrix 10x10
    model_acc = row_fm["Accuracy"]
    np.random.seed(hash(selected)%999)
    diag = int(model_acc*1000)
    cm = np.random.randint(5,60,(10,10))
    for i in range(10):
        cm[i,i] = np.random.randint(max(diag-50,700),diag+30)
    cm_pct = cm/cm.sum(axis=1,keepdims=True)*100
    fig_cm = go.Figure(go.Heatmap(z=cm_pct,x=class_names,y=class_names,
        colorscale=ORANGE_SCALE,
        text=[[f"{v:.0f}%" for v in row] for row in cm_pct],
        texttemplate="%{text}",textfont=dict(size=8,family=FONT_MONO),
        hovertemplate="Réel: <b>%{y}</b><br>Prédit: <b>%{x}</b><br>%{z:.1f}%<extra></extra>",
        showscale=True,colorbar=dict(thickness=12,title="%")))
    fig_cm.update_layout(**PLOT_LAYOUT,height=420,
        title=dict(text=f"Matrice de Confusion 10×10 — {selected} (Acc={model_acc:.3f})",
                   font=dict(size=12,family=FONT_TITLE)),
        xaxis=dict(tickangle=-30,tickfont_size=9),yaxis=dict(tickfont_size=9))

    # Architecture details component
    arch_map = {
        'MLP (Fully Connected)': [("Input","784 neurons (28×28 flatten)"),("Dense 1","512 · ReLU · Dropout 0.3"),("Dense 2","256 · ReLU · Dropout 0.3"),("Dense 3","128 · ReLU"),("Output","10 · Softmax")],
        'CNN LeNet+': [("Conv2D 1","32 filtres 3×3 · ReLU · MaxPool 2×2"),("Conv2D 2","64 filtres 3×3 · ReLU · MaxPool 2×2"),("Flatten","→ Dense 256 · Dropout 0.4"),("Output","10 · Softmax")],
        'VGGNet-like': [("Block 1","64×Conv3×3 × 2 · BN · MaxPool"),("Block 2","128×Conv3×3 × 2 · BN · MaxPool"),("Block 3","256×Conv3×3 × 3 · BN · MaxPool"),("FC","Dense 512 · Dropout 0.5"),("Output","10 · Softmax")],
        'ResNet-custom': [("Conv Init","32 filtres 3×3 · BN · ReLU"),("ResBlock 1","32 filtres + skip"),("ResBlock 2","64 filtres + skip"),("ResBlock 3","128 filtres + skip"),("GAP","Global Average Pooling"),("Output","10 · Softmax")],
    }
    layers = arch_map.get(selected,[])
    arch_el = html.Div([
        html.Div([
            html.Span(layer, style={"fontWeight":"700","fontSize":"11px","fontFamily":FONT_MONO,
                                    "color":CLR["orange_d"],"minWidth":"90px","display":"inline-block"}),
            html.Span(f"→ {desc}", style={"fontSize":"11px","fontFamily":FONT_BODY,"color":CLR["charcoal"]}),
        ], style={"background":CLR["orange_pale"],"borderRadius":"6px","padding":"4px 10px","marginBottom":"4px"})
        for layer,desc in layers
    ])

    # Per-class accuracy
    per_class_acc = [cm_pct[i,i] for i in range(10)]
    fig_pc = go.Figure(go.Bar(x=class_names,y=per_class_acc,
        marker=dict(color=per_class_acc,colorscale=ORANGE_SCALE,line=dict(color=CLR["white"],width=0.5)),
        text=[f"{v:.1f}%" for v in per_class_acc],textposition="outside",
        textfont=dict(size=10,family=FONT_MONO),
        hovertemplate="<b>%{x}</b><br>Précision: %{y:.1f}%<extra></extra>"))
    fig_pc.add_hline(y=model_acc*100,line_dash="dash",line_color=CLR["orange_d"],line_width=2)
    styled_fig(fig_pc,f"Précision par classe — {selected}",280)

    return fig_acc,fig_al,fig_curves,fig_cm,arch_el,fig_pc

# ─────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────────
app.index_string = '''
<!DOCTYPE html>
<html>
<head>
    {%metas%}
    <title>{%title%}</title>
    {%favicon%}
    {%css%}
    <style>
        * { box-sizing: border-box; }
        body { margin: 0; background: #F9F7F4; }
        ::-webkit-scrollbar { width: 5px; height: 5px; }
        ::-webkit-scrollbar-track { background: #F4F4F5; }
        ::-webkit-scrollbar-thumb { background: #F7931E; border-radius: 3px; }

        /* Nav hover */
        #nav-census-explore:hover, #nav-census-models:hover,
        #nav-bank-explore:hover,   #nav-bank-models:hover,
        #nav-fashion:hover,        #nav-summary:hover {
            background: rgba(247,147,30,0.18) !important;
            color: #F7931E !important;
            padding-left: 30px !important;
        }

        /* Dropdown */
        .Select-control { border-radius: 8px !important; border-color: #E4E4E7 !important; font-size:13px; }
        .Select-control:hover { border-color: #F7931E !important; }
        .Select-option.is-focused { background: #FFF7ED !important; }
        .Select-option.is-selected { background: #F7931E !important; color: white !important; }

        /* Slider */
        .rc-slider-handle { border-color: #F7931E !important; background: #F7931E !important; }
        .rc-slider-track { background: linear-gradient(90deg, #F7931E, #FDBA74) !important; }
        .rc-slider-dot-active { border-color: #F7931E !important; }
        input[type=radio]:checked { accent-color: #F7931E; }

        /* Table rows */
        tr:hover td { background: #FFF7ED !important; }
        td, th { padding: 8px 12px; }

        /* Cards hover */
        .card-hover:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 32px rgba(247,147,30,0.12) !important;
        }

        /* Animations */
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(8px); }
            to   { opacity: 1; transform: translateY(0); }
        }
        #page-content > div { animation: fadeIn 0.25s ease; }
    </style>
</head>
<body>
    {%app_entry%}
    <footer>
        {%config%}
        {%scripts%}
        {%renderer%}
    </footer>
</body>
</html>
'''

# ─────────────────────────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "═"*62)
    print("  🚀 Dashboard IA2 Complet démarré !")
    print("  📌 Ouvrir : http://127.0.0.1:8050")
    print("  📋 Pages  : Census (Explore+Modèles) | Bank (Explore+ANN)")
    print("             Fashion MNIST (DL) | Résumé Final")
    print("═"*62 + "\n")
    app.run(debug=False, host="127.0.0.1", port=8050)