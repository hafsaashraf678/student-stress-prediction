"""
Student Stress Prediction System - Dark Theme Professional
Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Arc
import seaborn as sns
import joblib, os, warnings, time
warnings.filterwarnings("ignore")

from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (r2_score, mean_absolute_error,
                             mean_squared_error, confusion_matrix,
                             classification_report)

st.set_page_config(
    page_title="Student Stress Prediction System",
    page_icon="🧠", layout="wide",
    initial_sidebar_state="expanded"
)

# ════════════════════════════════════════════════════════════
# DARK THEME CSS
# ════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

/* ── Global dark background ── */
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
#MainMenu, footer, header { visibility: hidden; }

.stApp {
    background: #070E1A;
}
.block-container {
    padding: 0.5rem 2rem 3rem 2rem;
    max-width: 1500px;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0D1B2E !important;
    border-right: 1px solid #1E3A5F;
}
[data-testid="stSidebar"] * { color: #B8D4F0 !important; }
[data-testid="stSidebar"] .stRadio label {
    background: #0A1628;
    border: 1px solid #1E3A5F;
    border-radius: 10px;
    padding: 10px 14px !important;
    margin-bottom: 6px !important;
    display: block;
    cursor: pointer;
    transition: all 0.2s;
    font-size: 14px !important;
    font-weight: 500 !important;
}
[data-testid="stSidebar"] .stRadio label:hover {
    background: #1A3A6B !important;
    border-color: #3B7DD8 !important;
}

/* ── Top header ── */
.top-header {
    background: linear-gradient(135deg, #0A1628 0%, #0D2240 50%, #0F2D5A 100%);
    border: 1px solid #1E3A5F;
    padding: 24px 32px;
    border-radius: 16px;
    margin-bottom: 24px;
    display: flex; align-items: center; gap: 18px;
    box-shadow: 0 4px 30px rgba(0,0,0,0.5), 0 0 60px rgba(24,95,165,0.1);
}
.top-header h1 { color: #FFFFFF; font-size: 26px; font-weight: 800; margin: 0 0 4px; }
.top-header p  { color: #5B9BD5; font-size: 13px; margin: 0; }
.header-badge {
    background: linear-gradient(135deg, #1A3A6B, #0F4C8A);
    border: 1px solid #2E6DB4;
    border-radius: 8px; padding: 6px 14px;
    font-size: 11px; font-weight: 700;
    color: #7EC8F7; letter-spacing: 0.1em;
    text-transform: uppercase;
}

/* ── Metric cards ── */
.metric-card {
    background: #0D1B2E;
    border: 1px solid #1E3A5F;
    border-radius: 14px; padding: 20px;
    text-align: center;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    transition: transform 0.2s, box-shadow 0.2s;
    position: relative; overflow: hidden;
}
.metric-card::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 3px;
    background: linear-gradient(90deg, #1A3A6B, #3B7DD8, #1A3A6B);
}
.metric-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 30px rgba(24,95,165,0.3);
    border-color: #2E6DB4;
}
.metric-card .num { font-size: 28px; font-weight: 800; margin-bottom: 6px; }
.metric-card .lbl { font-size: 11px; font-weight: 600; color: #5B9BD5;
    text-transform: uppercase; letter-spacing: 0.08em; }

/* ── Section titles ── */
.section-title {
    font-size: 15px; font-weight: 700; color: #7EC8F7;
    border-left: 3px solid #3B7DD8;
    padding: 5px 0 5px 12px;
    margin: 28px 0 14px;
    background: linear-gradient(to right, rgba(30,58,95,0.4), transparent);
    border-radius: 0 8px 8px 0;
}

/* ── Info boxes ── */
.info-box {
    background: rgba(24,95,165,0.12);
    border: 1px solid #1E3A5F;
    border-left: 3px solid #3B7DD8;
    border-radius: 0 10px 10px 0;
    padding: 12px 16px; margin: 10px 0;
    font-size: 13px; color: #90C4F0; line-height: 1.7;
}
.warn-box {
    background: rgba(255,160,0,0.08);
    border: 1px solid rgba(255,160,0,0.2);
    border-left: 3px solid #FFA000;
    border-radius: 0 10px 10px 0;
    padding: 12px 16px; margin: 10px 0;
    font-size: 13px; color: #FFD080; line-height: 1.7;
}
.danger-box {
    background: rgba(244,67,54,0.08);
    border: 1px solid rgba(244,67,54,0.2);
    border-left: 3px solid #F44336;
    border-radius: 0 10px 10px 0;
    padding: 12px 16px; margin: 10px 0;
    font-size: 13px; color: #FF8A80; line-height: 1.7;
}
.success-box {
    background: rgba(76,175,80,0.08);
    border: 1px solid rgba(76,175,80,0.2);
    border-left: 3px solid #4CAF50;
    border-radius: 0 10px 10px 0;
    padding: 12px 16px; margin: 10px 0;
    font-size: 13px; color: #A5D6A7; line-height: 1.7;
}

/* ── Result cards ── */
.result-low {
    background: linear-gradient(135deg, rgba(27,94,32,0.3), rgba(46,125,50,0.2));
    border: 1px solid #2E7D32; border-radius: 16px; padding: 28px; text-align: center;
    box-shadow: 0 0 30px rgba(76,175,80,0.15);
}
.result-medium {
    background: linear-gradient(135deg, rgba(230,81,0,0.2), rgba(255,143,0,0.15));
    border: 1px solid #E65100; border-radius: 16px; padding: 28px; text-align: center;
    box-shadow: 0 0 30px rgba(255,160,0,0.15);
}
.result-high {
    background: linear-gradient(135deg, rgba(183,28,28,0.3), rgba(229,57,53,0.15));
    border: 1px solid #C62828; border-radius: 16px; padding: 28px; text-align: center;
    box-shadow: 0 0 30px rgba(244,67,54,0.2);
}

/* ── Level cards ── */
.level-card { border-radius: 14px; padding: 20px; margin-bottom: 12px; }

/* ── Step badge ── */
.step-badge {
    display: inline-flex; align-items: center; justify-content: center;
    width: 28px; height: 28px; border-radius: 50%;
    background: linear-gradient(135deg, #1A3A6B, #3B7DD8);
    color: white; font-size: 12px; font-weight: 800; margin-right: 10px;
    flex-shrink: 0;
    box-shadow: 0 2px 8px rgba(59,125,216,0.4);
}

/* ── Model comparison cards ── */
.model-card {
    background: #0D1B2E;
    border: 1px solid #1E3A5F;
    border-radius: 14px; padding: 18px;
    text-align: center;
    transition: all 0.2s;
}
.model-card:hover {
    border-color: #3B7DD8;
    box-shadow: 0 0 20px rgba(59,125,216,0.2);
}
.model-card.winner {
    border-color: #4CAF50;
    background: linear-gradient(135deg, rgba(27,94,32,0.2), #0D1B2E);
    box-shadow: 0 0 25px rgba(76,175,80,0.2);
}

/* ── Streamlit overrides ── */
.stSlider [data-baseweb="slider"] { padding: 8px 0 !important; }
.stSelectbox select, .stSelectbox [data-baseweb="select"] > div {
    background: #0D1B2E !important;
    border-color: #1E3A5F !important;
    color: #B8D4F0 !important;
}
div[data-testid="stMetric"] {
    background: #0D1B2E;
    border: 1px solid #1E3A5F;
    border-radius: 12px; padding: 14px !important;
}
div[data-testid="stMetric"] label { color: #5B9BD5 !important; }
div[data-testid="stMetric"] div   { color: #FFFFFF  !important; }

.stDataFrame { background: #0D1B2E !important; }
.stDataFrame [data-testid="stDataFrameResizable"] {
    background: #0D1B2E !important;
    border: 1px solid #1E3A5F !important;
    border-radius: 10px !important;
}

/* Form */
.stForm { background: #0D1B2E !important; border: 1px solid #1E3A5F !important;
    border-radius: 14px !important; padding: 20px !important; }
.stButton button {
    background: linear-gradient(135deg, #1A3A6B, #2E6DB4) !important;
    color: white !important; border: none !important;
    border-radius: 10px !important; font-weight: 700 !important;
    padding: 12px 24px !important; font-size: 15px !important;
    box-shadow: 0 4px 15px rgba(24,95,165,0.4) !important;
    transition: all 0.2s !important;
}
.stButton button:hover {
    background: linear-gradient(135deg, #2E6DB4, #3B7DD8) !important;
    box-shadow: 0 6px 25px rgba(59,125,216,0.5) !important;
    transform: translateY(-1px) !important;
}

/* Divider */
hr { border-color: #1E3A5F !important; }
</style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# CONSTANTS
# ════════════════════════════════════════════════════════════
DATA_PATH = "university_student_stress_dataset.csv"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

STRESS_INFO = {
    "Low": {
        "range":"0 – 10", "emoji":"😊",
        "color":"#4CAF50", "dim_color":"#A5D6A7",
        "bg":"rgba(27,94,32,0.25)", "border":"#2E7D32",
        "css":"result-low", "box":"success-box",
        "title":"LOW STRESS — Student is doing well!",
        "what":"Score is 10 or below. Student is managing academic life comfortably. Sleep, screen time, and workload are at healthy levels.",
        "causes":["Good sleep habits (7+ hours per night)",
                  "Screen time is under control",
                  "Strong family support available",
                  "Manageable exam and assignment load"],
        "tips":["✅ Keep maintaining good sleep schedule",
                "✅ Continue limiting social media",
                "✅ Regular exercise helps stay at this level",
                "✅ Stay connected with family and friends"],
        "action":"No immediate action needed. Student is coping well.",
    },
    "Medium": {
        "range":"11 – 20", "emoji":"😐",
        "color":"#FFA000", "dim_color":"#FFD080",
        "bg":"rgba(230,81,0,0.2)", "border":"#E65100",
        "css":"result-medium", "box":"warn-box",
        "title":"MEDIUM STRESS — Student needs attention",
        "what":"Score is between 11 and 20. Student is under noticeable stress. Not critical yet, but can get worse if ignored. May feel tired or anxious sometimes.",
        "causes":["Too much screen time (social media, gaming)",
                  "High exam frequency or heavy assignments",
                  "Not enough sleep — only 5-6 hours",
                  "Low family support or high peer pressure"],
        "tips":["⚠️ Reduce screen time by 2+ hours per day",
                "⚠️ Try to get 7-8 hours sleep every night",
                "⚠️ Talk to a friend or family member",
                "⚠️ Take short breaks during study",
                "⚠️ Even 20 min walk reduces stress"],
        "action":"Counselor should monitor and check in regularly.",
    },
    "High": {
        "range":"21 – 30", "emoji":"😟",
        "color":"#F44336", "dim_color":"#FF8A80",
        "bg":"rgba(183,28,28,0.25)", "border":"#C62828",
        "css":"result-high", "box":"danger-box",
        "title":"HIGH STRESS — Immediate help needed!",
        "what":"Score is above 20. This is serious. Student is likely overwhelmed and exhausted. May be struggling to function normally. Needs immediate counselor attention.",
        "causes":["Extreme screen time (8+ hours daily)",
                  "Very high exam frequency",
                  "Sleeping less than 5 hours per night",
                  "Very little or no family support",
                  "Heavy social media causing anxiety"],
        "tips":["🚨 Meet counselor IMMEDIATELY",
                "🚨 Reduce exam pressure — discuss with teachers",
                "🚨 Encourage 7+ hours of sleep",
                "🚨 Limit screen time to max 4 hours/day",
                "🚨 Family involvement is critical right now"],
        "action":"⚠️ URGENT: Counselor must contact this student immediately.",
    }
}

# ════════════════════════════════════════════════════════════
# DATA & MODELS
# ════════════════════════════════════════════════════════════
@st.cache_data
def load_data():
    if not os.path.exists(DATA_PATH):
        st.error(f"❌ Dataset not found: {DATA_PATH}")
        st.stop()
    df  = pd.read_csv(DATA_PATH)
    dc  = df.dropna().copy()
    for c in ["Anxiety_Level","Stress_Level","Age"]:
        if c in dc.columns: dc.drop(columns=[c], inplace=True)
    for c in ["Gender","Physical_Exercise","Tuition"]:
        if c in dc.columns: dc[c] = dc[c].astype("category").cat.codes
    if "Family_Income_Level" in dc.columns:
        dc["Family_Income_Level"] = dc["Family_Income_Level"].map(
            {"Low":0,"Medium":1,"High":2}).fillna(1)
    if "University_Type" in dc.columns:
        dum = pd.get_dummies(dc["University_Type"], prefix="UnivType")
        dc  = pd.concat([dc.drop("University_Type",axis=1), dum], axis=1)
    X = dc.drop(columns=["Stress_Score"])
    y = dc["Stress_Score"]
    return df, dc, X, y

def _cat(s):
    return pd.Series(s).apply(
        lambda x: "Low" if x<=10 else ("Medium" if x<=20 else "High"))

@st.cache_resource
def train_models(X, y):
    Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=.2,random_state=42)
    sc  = StandardScaler()
    Xs  = sc.fit_transform(Xtr)
    Xe  = sc.transform(Xte)
    lr  = LinearRegression().fit(Xs,ytr)
    svr = SVR(kernel="rbf",C=10,epsilon=.5).fit(Xs,ytr)
    rf  = RandomForestRegressor(n_estimators=100,random_state=42).fit(Xs,ytr)
    res = {}
    for name,model,yp in [("Linear Regression",lr,lr.predict(Xe)),
                           ("SVR (RBF)",svr,svr.predict(Xe)),
                           ("Random Forest",rf,rf.predict(Xe))]:
        # ensure predicted categories align with test indices to allow direct comparison
        # category arrays (numpy) — avoids pandas index-alignment issues
        ct = _cat(yte).values
        cp = _cat(pd.Series(yp, index=yte.index)).values
        res[name]={
            "model":model,
            "r2":round(r2_score(yte,yp),4),
            "mae":round(mean_absolute_error(yte,yp),4),
            "rmse":round(np.sqrt(mean_squared_error(yte,yp)),4),
            "acc":round((ct==cp).mean()*100,2),
            "y_true":yte.values,"y_pred":yp,
            "cm":confusion_matrix(ct,cp,labels=["Low","Medium","High"]),
            "cr":classification_report(ct,cp,labels=["Low","Medium","High"],output_dict=True),
        }
    res["Random Forest"]["feature_names"]=list(X.columns)
    res["Random Forest"]["importances"]=rf.feature_importances_
    joblib.dump(lr, f"{MODEL_DIR}/lr_model.pkl")
    joblib.dump(svr,f"{MODEL_DIR}/svr_model.pkl")
    joblib.dump(rf, f"{MODEL_DIR}/rf_model.pkl")
    joblib.dump(sc, f"{MODEL_DIR}/scaler.pkl")
    return res, sc

def predict_one(inp, model_name, scaler, feature_cols):
    row = pd.DataFrame([inp])
    for c in feature_cols:
        if c not in row.columns: row[c]=0
    row  = row[feature_cols]
    rows = scaler.transform(row)
    m    = joblib.load({"Linear Regression":f"{MODEL_DIR}/lr_model.pkl",
                        "SVR (RBF)":        f"{MODEL_DIR}/svr_model.pkl",
                        "Random Forest":    f"{MODEL_DIR}/rf_model.pkl"}[model_name])
    sc   = float(np.clip(m.predict(rows)[0],0,30))
    return round(sc,2), _cat([sc]).iloc[0]

# ════════════════════════════════════════════════════════════
# DARK CHARTS
# ════════════════════════════════════════════════════════════
DARK_BG    = "#070E1A"
CARD_BG    = "#0D1B2E"
BORDER_COL = "#1E3A5F"
TEXT_COL   = "#B8D4F0"
ACCENT     = "#3B7DD8"
MUTED      = "#5B9BD5"

def dark_fig(w=8,h=4):
    fig,ax = plt.subplots(figsize=(w,h))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(CARD_BG)
    for sp in ax.spines.values(): sp.set_color(BORDER_COL)
    ax.tick_params(colors=TEXT_COL, labelsize=9)
    ax.xaxis.label.set_color(TEXT_COL)
    ax.yaxis.label.set_color(TEXT_COL)
    ax.title.set_color("#7EC8F7")
    return fig,ax

def gauge_chart(score):
    fig,ax = plt.subplots(figsize=(6,4))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_aspect("equal"); ax.axis("off")
    # Background arc
    ax.add_patch(Arc((0,0),2.1,2.1,angle=0,theta1=0,theta2=180,
                 color=BORDER_COL,linewidth=30))
    # Colored zones
    for s,e,c in [(180,120,"#388E3C"),(120,60,"#F57F17"),(60,0,"#C62828")]:
        ax.add_patch(Arc((0,0),2,2,angle=0,theta1=e,theta2=s,
                 color=c,linewidth=26))
    # Glow effect on active zone
    if score<=10:   glow_s,glow_e,glow_c = 180,120,"#66BB6A"
    elif score<=20: glow_s,glow_e,glow_c = 120,60, "#FFA726"
    else:           glow_s,glow_e,glow_c = 60, 0,  "#EF5350"
    ax.add_patch(Arc((0,0),2,2,angle=0,theta1=glow_e,theta2=glow_s,
                 color=glow_c,linewidth=28,alpha=0.4))
    # Needle
    ang = np.radians(180-(score/30)*180)
    ax.annotate("",xy=(.82*np.cos(ang),.82*np.sin(ang)),xytext=(0,0),
                arrowprops=dict(arrowstyle="-|>",color="white",
                                lw=2.5,mutation_scale=22))
    ax.plot(0,0,"o",color="white",markersize=12,zorder=5)
    ax.plot(0,0,"o",color=CARD_BG,markersize=7,zorder=6)
    # Score
    ax.text(0,-.25,f"{score}",ha="center",va="center",
            fontsize=36,fontweight="900",color="white")
    ax.text(0,-.50,"out of 30",ha="center",va="center",
            fontsize=11,color=MUTED)
    cat   = "Low" if score<=10 else ("Medium" if score<=20 else "High")
    ccol  = "#66BB6A" if cat=="Low" else ("#FFA726" if cat=="Medium" else "#EF5350")
    ax.text(0,-.72,f"{cat} Stress",ha="center",va="center",
            fontsize=16,fontweight="800",color=ccol)
    ax.text(-1.15,.1,"LOW",  ha="center",fontsize=9,color="#66BB6A",fontweight="700")
    ax.text(0,   1.15,"MED", ha="center",fontsize=9,color="#FFA726",fontweight="700")
    ax.text( 1.15,.1,"HIGH", ha="center",fontsize=9,color="#EF5350",fontweight="700")
    ax.set_xlim(-1.4,1.4); ax.set_ylim(-.9,1.4)
    plt.tight_layout(pad=0)
    return fig

def score_ruler(score):
    fig,ax = plt.subplots(figsize=(10,1.8))
    fig.patch.set_facecolor(DARK_BG); ax.axis("off")
    for x,w,c,l in [(0,10,"#388E3C","LOW (0–10)"),
                     (10,10,"#F57F17","MEDIUM (11–20)"),
                     (20,10,"#C62828","HIGH (21–30)")]:
        ax.barh(0,w,left=x,color=c,height=.45,edgecolor=DARK_BG,lw=2)
        ax.text(x+w/2,0,l,ha="center",va="center",
                fontsize=11,fontweight="700",color="white")
    ax.annotate("▲",xy=(score,.27),ha="center",
                fontsize=20,color="white",fontweight="900")
    ax.text(score,.48,f"Score: {score}",ha="center",
            fontsize=10,fontweight="700",color="white")
    for v in [0,5,10,15,20,25,30]:
        ax.text(v,-.32,str(v),ha="center",fontsize=9,color=MUTED)
    ax.set_xlim(-.5,30.5); ax.set_ylim(-.45,.65)
    plt.tight_layout(pad=0)
    return fig

def chart_model_bars(results):
    models  = list(results.keys())
    colors  = ["#3B7DD8","#2ECC71","#E67E22"]
    metrics = [("r2","R² Score"),("acc","Category Acc %"),
               ("mae","MAE (lower=better)"),("rmse","RMSE")]
    fig,axes = plt.subplots(1,4,figsize=(14,4))
    fig.patch.set_facecolor(DARK_BG)
    fig.suptitle("Model Performance Comparison",fontsize=13,
                 fontweight="800",color="#7EC8F7",y=1.02)
    for i,(metric,label) in enumerate(metrics):
        ax=axes[i]
        ax.set_facecolor(CARD_BG)
        for sp in ax.spines.values(): sp.set_color(BORDER_COL)
        ax.tick_params(colors=TEXT_COL,labelsize=9)
        vals=[results[m][metric] for m in models]
        bars=ax.barh(models,vals,color=colors,height=.55,
                     edgecolor=DARK_BG,linewidth=2)
        ax.set_title(label,fontsize=10,fontweight="700",
                     color="#7EC8F7",pad=8)
        for bar,val in zip(bars,vals):
            ax.text(bar.get_width()+max(vals)*.03,
                    bar.get_y()+bar.get_height()/2,
                    f"{val}",va="center",fontsize=9,
                    fontweight="700",color="white")
        ax.set_xlim(0,max(vals)*1.3)
    plt.tight_layout(); return fig

def chart_scatter(res,name):
    fig,ax=dark_fig(7,5)
    yt,yp=res[name]["y_true"],res[name]["y_pred"]
    lims=[min(yt.min(),yp.min())-1,max(yt.max(),yp.max())+1]
    ax.plot(lims,lims,"--",color="#EF5350",lw=1.8,label="Perfect fit",alpha=.8)
    scatter=ax.scatter(yt,yp,alpha=.45,c=yp,cmap="coolwarm",
                       s=25,edgecolors="none",zorder=2)
    ax.set_xlabel("Actual Stress Score",fontsize=10,color=TEXT_COL)
    ax.set_ylabel("Predicted Stress Score",fontsize=10,color=TEXT_COL)
    ax.set_title(f"Actual vs Predicted — {name}",fontsize=12,fontweight="700")
    ax.legend(fontsize=9,facecolor=CARD_BG,edgecolor=BORDER_COL,
              labelcolor=TEXT_COL)
    ax.text(.05,.92,f"R² = {res[name]['r2']}  |  MAE = {res[name]['mae']}",
            transform=ax.transAxes,fontsize=9,color=TEXT_COL,
            bbox=dict(boxstyle="round,pad=.4",facecolor=CARD_BG,
                      edgecolor=BORDER_COL))
    plt.tight_layout(); return fig

def chart_confusion(res,name):
    fig,ax=dark_fig(5,4)
    cmap=sns.light_palette("#3B7DD8",as_cmap=True)
    sns.heatmap(res[name]["cm"],annot=True,fmt="d",cmap=cmap,
                xticklabels=["Low","Medium","High"],
                yticklabels=["Low","Medium","High"],
                ax=ax,linewidths=1,linecolor=DARK_BG,
                annot_kws={"fontsize":13,"fontweight":"bold","color":"white"},
                cbar_kws={"shrink":.8})
    ax.set_xlabel("Predicted",fontsize=10,color=TEXT_COL)
    ax.set_ylabel("Actual",fontsize=10,color=TEXT_COL)
    ax.set_title(f"Confusion Matrix\n{name}",fontsize=11,fontweight="700")
    ax.tick_params(colors=TEXT_COL)
    plt.tight_layout(); return fig

def chart_importance(res):
    d=res["Random Forest"]
    names=[n.replace("_"," ") for n in d["feature_names"]]
    imps=d["importances"]
    idx=np.argsort(imps)[::-1][:10]
    fn=[names[i] for i in idx]; fv=[imps[i]*100 for i in idx]
    fig,ax=dark_fig(8,5)
    colors=["#FFD700" if v==max(fv) else ACCENT for v in fv]
    ax.barh(fn[::-1],fv[::-1],color=colors[::-1],height=.6,edgecolor=DARK_BG,lw=1.5)
    for bar,val in zip(ax.patches,fv[::-1]):
        ax.text(bar.get_width()+.3,bar.get_y()+bar.get_height()/2,
                f"{val:.1f}%",va="center",fontsize=9,fontweight="700",color="white")
    ax.set_xlabel("Importance (%)",fontsize=10,color=TEXT_COL)
    ax.set_title("Top 10 Feature Importance (Random Forest)",
                 fontsize=12,fontweight="700")
    ax.set_xlim(0,max(fv)*1.25)
    plt.tight_layout(); return fig

def chart_distribution(df):
    if "Stress_Score" not in df.columns: return None
    fig,ax=dark_fig(7,4)
    n,bins,patches=ax.hist(df["Stress_Score"],bins=30,edgecolor=DARK_BG,lw=.8)
    for p,l in zip(patches,bins[:-1]):
        p.set_facecolor("#388E3C" if l<=10 else("#F57F17" if l<=20 else "#C62828"))
    ax.axvline(10,color="#66BB6A",linestyle="--",lw=1.5,alpha=.8)
    ax.axvline(20,color="#EF5350",linestyle="--",lw=1.5,alpha=.8)
    ax.set_xlabel("Stress Score (0–30)",fontsize=10,color=TEXT_COL)
    ax.set_ylabel("Number of Students",fontsize=10,color=TEXT_COL)
    ax.set_title("Stress Score Distribution",fontsize=12,fontweight="700")
    ax.legend(handles=[
        mpatches.Patch(color="#388E3C",label="Low (≤10)"),
        mpatches.Patch(color="#F57F17",label="Medium (11–20)"),
        mpatches.Patch(color="#C62828",label="High (>20)")],
        fontsize=9,facecolor=CARD_BG,edgecolor=BORDER_COL,labelcolor=TEXT_COL)
    plt.tight_layout(); return fig

def chart_correlation(df):
    if "Stress_Score" not in df.columns: return None
    num_df=df.select_dtypes(include=[np.number])
    corr=num_df.corr()["Stress_Score"].drop("Stress_Score").sort_values()
    fig,ax=dark_fig(8,5)
    colors=["#EF5350" if v<0 else ACCENT for v in corr]
    ax.barh(corr.index,corr.values,color=colors,height=.6,edgecolor=DARK_BG)
    ax.axvline(0,color=BORDER_COL,lw=1.5)
    for i,(v,n) in enumerate(zip(corr.values,corr.index)):
        ax.text(v+(0.01 if v>=0 else -.01),i,f"{v:.2f}",
                va="center",ha="left" if v>=0 else "right",
                fontsize=8,color=TEXT_COL)
    ax.set_xlabel("Correlation with Stress Score",fontsize=10,color=TEXT_COL)
    ax.set_title("Feature Correlation",fontsize=12,fontweight="700")
    plt.tight_layout(); return fig

# ════════════════════════════════════════════════════════════
# SIDEBAR NAVIGATION
# ════════════════════════════════════════════════════════════
def sidebar_nav():
    with st.sidebar:
        st.markdown("""
        <div style='text-align:center;padding:20px 0 10px;'>
            <div style='font-size:40px;'>🧠</div>
            <div style='font-size:15px;font-weight:800;color:#7EC8F7;
                        margin-top:8px;'>Stress Prediction</div>
            <div style='font-size:11px;color:#5B9BD5;margin-top:4px;'>
                ML Research System</div>
        </div>
        <hr style='border-color:#1E3A5F;margin:12px 0;'>
        """, unsafe_allow_html=True)

        page = st.radio("", [
            "🏠  Dashboard",
            "📊  Dataset Info",
            "🤖  How Models Work",
            "🔴  Live Demo",
            "📈  All Charts",
            "📉  Stress Levels Guide",
            "ℹ️   About",
        ], label_visibility="collapsed")

        st.markdown("""
        <hr style='border-color:#1E3A5F;margin:20px 0 12px;'>
        <div style='font-size:11px;color:#3A5A7A;text-align:center;'>
            Linear Regression · SVR · Random Forest
        </div>
        """, unsafe_allow_html=True)
    return page

# ════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════
def main():
    page = sidebar_nav()

    # Header
    st.markdown("""
    <div class="top-header">
        <span style="font-size:44px;">🧠</span>
        <div style="flex:1;">
            <h1>Student Stress Prediction System</h1>
            <p>Machine Learning Research Dashboard  ·
               IEEE Paper 2025  ·  3,000 University Students</p>
        </div>
        <div>
            <div class="header-badge">IEEE 2025</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Load & train
    with st.spinner("⏳ Loading dataset and training models..."):
        df_raw, df_clean, X, y = load_data()
        results, scaler        = train_models(X, y)
    feature_cols = list(X.columns)
    best = max(results, key=lambda m: results[m]["r2"])

    # ── Summary row ─────────────────────────────────────
    c1,c2,c3,c4,c5 = st.columns(5)
    for col,num,lbl,color in [
        (c1, f"{len(df_raw):,}", "Total Students",  "#3B7DD8"),
        (c2, results[best]["r2"],"Best R² Score",   "#4CAF50"),
        (c3, f"{results[best]['acc']}%","Best Acc", "#FFA726"),
        (c4, len(feature_cols), "Features Used",    "#CE93D8"),
        (c5, "Linear Reg",      "Best Model",       "#7EC8F7"),
    ]:
        with col:
            st.markdown(f"""<div class="metric-card">
            <div class="num" style="color:{color};">{num}</div>
            <div class="lbl">{lbl}</div></div>""",unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # ════════════════════════════════════════════════════
    # PAGE ROUTER
    # ════════════════════════════════════════════════════

    # ── 🏠 DASHBOARD ────────────────────────────────────
    if "Dashboard" in page:
        st.markdown('<div class="section-title">Project Overview</div>',
                    unsafe_allow_html=True)
        st.markdown("""<div class="info-box">
        This system uses <strong>machine learning</strong> to predict stress levels
        of university students. We trained 3 models on data from 3,000 students
        and compared their performance. The goal is to find high-stress students
        early so counselors can help them before things get worse.
        </div>""", unsafe_allow_html=True)

        st.markdown('<div class="section-title">Model Results at a Glance</div>',
                    unsafe_allow_html=True)
        m1,m2,m3 = st.columns(3)
        model_colors = {"Linear Regression":"#3B7DD8",
                        "SVR (RBF)":"#2ECC71",
                        "Random Forest":"#E67E22"}
        for col,(name,data) in zip([m1,m2,m3],results.items()):
            with col:
                is_best = name==best
                css = "model-card winner" if is_best else "model-card"
                badge = "⭐ BEST MODEL" if is_best else ""
                bc = model_colors[name]
                st.markdown(f"""
                <div class="{css}" style="border-top:3px solid {bc};">
                    <div style="font-size:13px;font-weight:800;
                                color:{bc};margin-bottom:12px;">
                        {name} {badge}
                    </div>
                    <div style="font-size:28px;font-weight:900;color:white;">
                        {data['r2']}</div>
                    <div style="font-size:11px;color:{MUTED};margin-bottom:10px;">
                        R² Score</div>
                    <div style="display:grid;grid-template-columns:1fr 1fr;
                                gap:8px;margin-top:8px;">
                        <div style="background:#070E1A;border-radius:8px;padding:8px;">
                            <div style="font-size:16px;font-weight:700;color:white;">
                                {data['acc']}%</div>
                            <div style="font-size:10px;color:{MUTED};">Accuracy</div>
                        </div>
                        <div style="background:#070E1A;border-radius:8px;padding:8px;">
                            <div style="font-size:16px;font-weight:700;color:white;">
                                {data['mae']}</div>
                            <div style="font-size:10px;color:{MUTED};">MAE</div>
                        </div>
                    </div>
                </div>""", unsafe_allow_html=True)

        st.markdown('<div class="section-title">Performance Comparison Chart</div>',
                    unsafe_allow_html=True)
        st.pyplot(chart_model_bars(results), use_container_width=True)

    # ── 📊 DATASET ───────────────────────────────────────
    elif "Dataset" in page:
        st.markdown('<div class="section-title">Dataset Overview</div>',
                    unsafe_allow_html=True)
        a,b,c,d = st.columns(4)
        a.metric("Total Students","3,000")
        b.metric("Training Set","2,400  (80%)")
        c.metric("Testing Set","600  (20%)")
        d.metric("Target Range","0 – 30")

        st.markdown('<div class="section-title">Feature Descriptions</div>',
                    unsafe_allow_html=True)
        st.dataframe(pd.DataFrame({
            "Feature":["Sleep_Hours","Screen_Time","Exam_Frequency",
                       "Assignment_Load","Family_Support","Social_Media_Use",
                       "Study_Hours","Class_Attendance","Peer_Pressure",
                       "Gender","Physical_Exercise","Family_Income_Level"],
            "What it means":["Hours of sleep per night",
                             "Total screen hours per day",
                             "Number of exams per month",
                             "Assignments per week",
                             "Family support score 1–10",
                             "Hours on social media per day",
                             "Study hours per day",
                             "Class attendance percentage",
                             "Peer pressure score 1–10",
                             "Male or Female",
                             "Does student exercise?",
                             "Low, Medium, or High income"],
            "Encoding":["Number"]*9+["0/1"]*2+["0/1/2"],
        }),use_container_width=True,hide_index=True)

        st.markdown("""<div class="danger-box">
        🚫 <strong>Removed features:</strong><br>
        &nbsp;&nbsp;• <strong>Anxiety_Level</strong> — directly linked to stress = data leakage (cheating)<br>
        &nbsp;&nbsp;• <strong>Stress_Level</strong> — text version of Stress_Score = same answer<br>
        &nbsp;&nbsp;• <strong>Age</strong> — only 6 unique values, no useful pattern
        </div>""", unsafe_allow_html=True)

        st.markdown('<div class="section-title">Stress Distribution</div>',
                    unsafe_allow_html=True)
        fig=chart_distribution(df_raw)
        if fig: st.pyplot(fig,use_container_width=True)

        st.markdown('<div class="section-title">Raw Data Preview</div>',
                    unsafe_allow_html=True)
        st.dataframe(df_raw.head(20),use_container_width=True)

    # ── 🤖 HOW MODELS WORK ──────────────────────────────
    elif "How Models" in page:
        st.markdown('<div class="section-title">What Are These 3 Models?</div>',
                    unsafe_allow_html=True)

        m1,m2,m3 = st.columns(3)
        for col,(name,color,icon,explain,how,result) in zip([m1,m2,m3],[
            ("Linear Regression","#3B7DD8","📏",
             "The simplest model. It draws the best straight line through data.",
             "Finds coefficients for: Stress = (w1×ScreenTime) + (w2×Sleep) + ...",
             f"R²={results['Linear Regression']['r2']} ⭐ Best"),
            ("SVR (RBF Kernel)","#2ECC71","🎯",
             "Smarter model. Handles curved patterns using RBF kernel trick.",
             "Creates a tube around the data. Points inside tube = ignored. Outside = error.",
             f"R²={results['SVR (RBF)']['r2']}"),
            ("Random Forest","#E67E22","🌲",
             "100 decision trees vote together. More stable than one tree.",
             "Each tree sees random features. All 100 trees predict. Average is final answer.",
             f"R²={results['Random Forest']['r2']}"),
        ]):
            with col:
                st.markdown(f"""
                <div class="model-card" style="border-top:3px solid {color};text-align:left;">
                    <div style="font-size:28px;">{icon}</div>
                    <div style="font-size:14px;font-weight:800;color:{color};
                                margin:8px 0 6px;">{name}</div>
                    <div style="font-size:12px;color:{TEXT_COL};
                                line-height:1.6;margin-bottom:10px;">{explain}</div>
                    <div style="background:#070E1A;border-radius:8px;padding:10px;
                                font-size:11px;color:{MUTED};font-style:italic;
                                line-height:1.6;margin-bottom:10px;">{how}</div>
                    <div style="font-size:13px;font-weight:700;color:{color};">
                        Result: {result}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown('<div class="section-title">Training Pipeline — Step by Step</div>',
                    unsafe_allow_html=True)
        steps = [
            ("Collect Data",        "3,000 student records with 18 columns each.",                               "📁"),
            ("Remove Leakage",      "Drop Anxiety_Level, Stress_Level, Age.",                                    "🗑️"),
            ("Encode Text",         "Convert Male/Female → 0/1. Low/Med/High → 0/1/2.",                         "🔢"),
            ("Normalize",           "StandardScaler: every feature gets mean=0, std=1.",                         "⚖️"),
            ("Split 80/20",         "2,400 students for training. 600 for testing.",                             "✂️"),
            ("Train 3 Models",      "Linear Regression, SVR, Random Forest all trained on same data.",           "🏋️"),
            ("Evaluate",            "R², MAE, RMSE, Category Accuracy all calculated on test set.",              "📊"),
            ("Compare & Select",    "Linear Regression wins with R²=0.85 and 81.33% accuracy.",                  "🏆"),
        ]
        for i,(title,desc,icon) in enumerate(steps):
            bg = "rgba(13,27,46,0.8)" if i%2==0 else "rgba(7,14,26,0.8)"
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:14px;
                        padding:14px 18px;background:{bg};
                        border-radius:12px;margin-bottom:8px;
                        border:1px solid {BORDER_COL};">
                <span class="step-badge">{i+1}</span>
                <span style="font-size:20px;">{icon}</span>
                <div>
                    <strong style="color:#7EC8F7;font-size:14px;">{title}</strong><br>
                    <span style="font-size:12px;color:{MUTED};">{desc}</span>
                </div>
            </div>""", unsafe_allow_html=True)

        st.markdown('<div class="section-title">Performance Comparison</div>',
                    unsafe_allow_html=True)
        df_sum = pd.DataFrame({
            "Model":         list(results.keys()),
            "R² Score":      [results[m]["r2"]  for m in results],
            "MAE":           [results[m]["mae"]  for m in results],
            "RMSE":          [results[m]["rmse"] for m in results],
            "Category Acc%": [results[m]["acc"]  for m in results],
        })
        st.dataframe(df_sum.style
            .highlight_max(subset=["R² Score","Category Acc%"],color="#1A3A2A")
            .highlight_min(subset=["MAE","RMSE"],color="#1A3A2A"),
            use_container_width=True,hide_index=True)

        st.markdown("""<div class="success-box">
        ✅ <strong>Why does Linear Regression win?</strong><br>
        Stress increases in a mostly straight-line way when Screen Time or Exam Frequency goes up.
        Linear Regression is built exactly for this. Random Forest and SVR need tuning to beat it.
        </div>""", unsafe_allow_html=True)

        # Scatter plots for all 3
        st.markdown('<div class="section-title">Actual vs Predicted — All 3 Models</div>',
                    unsafe_allow_html=True)
        sc1,sc2,sc3 = st.columns(3)
        for col,name in zip([sc1,sc2,sc3],results.keys()):
            with col:
                st.pyplot(chart_scatter(results,name),use_container_width=True)

        # Confusion matrices
        st.markdown('<div class="section-title">Confusion Matrices — All 3 Models</div>',
                    unsafe_allow_html=True)
        cm1,cm2,cm3 = st.columns(3)
        for col,name in zip([cm1,cm2,cm3],results.keys()):
            with col:
                st.pyplot(chart_confusion(results,name),use_container_width=True)

        st.markdown("""<div class="info-box">
        <strong>Reading the confusion matrix:</strong> Diagonal numbers (top-left to bottom-right)
        are correct predictions. Other numbers are mistakes.
        Linear Regression correctly found <strong>41 High-stress students</strong> —
        the most important group to catch.
        </div>""", unsafe_allow_html=True)

        st.markdown('<div class="section-title">Feature Importance</div>',
                    unsafe_allow_html=True)
        st.pyplot(chart_importance(results),use_container_width=True)

    # ── 🔴 LIVE DEMO ─────────────────────────────────────
    elif "Live Demo" in page:
        st.markdown("""<div class="info-box">
        🎯 <strong>LIVE DEMO</strong> — Change the sliders below and press
        <strong>Predict</strong>. All 3 models will run instantly and show their
        results side by side. This shows exactly how each model works differently.
        </div>""", unsafe_allow_html=True)

        st.markdown('<div class="section-title">Enter Student Data</div>',
                    unsafe_allow_html=True)

        with st.form("live_demo"):
            ca,cb,cc = st.columns(3)
            with ca:
                st.markdown(f"<span style='color:{MUTED};font-size:13px;font-weight:600;'>📱 SCREEN & SOCIAL</span>",unsafe_allow_html=True)
                screen_time  = st.slider("Screen Time (hrs/day)",    0,16,6)
                social_media = st.slider("Social Media (hrs/day)",    0,10,3)
                study_hours  = st.slider("Study Hours (day)",         0,12,4)
            with cb:
                st.markdown(f"<span style='color:{MUTED};font-size:13px;font-weight:600;'>😴 HEALTH</span>",unsafe_allow_html=True)
                sleep_hours = st.slider("Sleep Hours (night)",        3,10,6)
                attendance  = st.slider("Class Attendance (%)",      40,100,75)
                physical_ex = st.selectbox("Physical Exercise",["Yes (1)","No (0)"])
            with cc:
                st.markdown(f"<span style='color:{MUTED};font-size:13px;font-weight:600;'>📚 ACADEMIC</span>",unsafe_allow_html=True)
                exam_freq   = st.slider("Exam Frequency (month)",     1,10,4)
                assign_load = st.slider("Assignment Load (week)",     1,10,4)
                peer_pres   = st.slider("Peer Pressure (1–10)",       1,10,5)

            st.markdown("---")
            cd,ce = st.columns(2)
            with cd:
                fam_sup   = st.slider("Family Support (1–10)",        1,10,6)
                fam_inc   = st.selectbox("Family Income",["Low (0)","Medium (1)","High (2)"])
            with ce:
                gender    = st.selectbox("Gender",["Male (1)","Female (0)"])
                univ_type = st.selectbox("University Type",["National","Private","Public"])

            btn = st.form_submit_button(
                "⚡  RUN ALL 3 MODELS — PREDICT NOW",
                use_container_width=True)

        if btn:
            # Build input
            inp = {}
            for col in feature_cols:
                cl=col.lower()
                if   "screen"   in cl: inp[col]=screen_time
                elif "social"   in cl: inp[col]=social_media
                elif "study"    in cl: inp[col]=study_hours
                elif "sleep"    in cl: inp[col]=sleep_hours
                elif "attend"   in cl: inp[col]=attendance
                elif "physical" in cl: inp[col]=int(physical_ex.split("(")[1].replace(")",""))
                elif "exam"     in cl: inp[col]=exam_freq
                elif "assign"   in cl: inp[col]=assign_load
                elif "peer"     in cl: inp[col]=peer_pres
                elif "family_s" in cl: inp[col]=fam_sup
                elif "income"   in cl: inp[col]=int(fam_inc.split("(")[1].replace(")",""))
                elif "gender"   in cl: inp[col]=int(gender.split("(")[1].replace(")",""))
                elif "private"  in cl: inp[col]=1 if univ_type=="Private"  else 0
                elif "public"   in cl: inp[col]=1 if univ_type=="Public"   else 0
                elif "national" in cl: inp[col]=1 if univ_type=="National" else 0
                else:                  inp[col]=0

            # ── Run all 3 models with animated progress
            st.markdown('<div class="section-title">Running All 3 Models...</div>',
                        unsafe_allow_html=True)
            pbar = st.progress(0,"Starting...")
            predictions = {}
            model_names = list(results.keys())
            model_icons = {"Linear Regression":"📏","SVR (RBF)":"🎯","Random Forest":"🌲"}
            model_colors_map = {"Linear Regression":"#3B7DD8",
                                "SVR (RBF)":"#2ECC71",
                                "Random Forest":"#E67E22"}

            for i,name in enumerate(model_names):
                pbar.progress((i+1)*33, f"Running {name}...")
                time.sleep(0.3)
                score,cat = predict_one(inp,name,scaler,feature_cols)
                predictions[name] = (score, cat)
            pbar.progress(100,"Done! ✅")
            time.sleep(0.3)
            pbar.empty()

            # ── Show all 3 results side by side
            st.markdown('<div class="section-title">Results — All 3 Models</div>',
                        unsafe_allow_html=True)
            r1,r2,r3 = st.columns(3)
            for col,name in zip([r1,r2,r3],model_names):
                score,cat = predictions[name]
                info  = STRESS_INFO[cat]
                mc    = model_colors_map[name]
                icon  = model_icons[name]
                is_best_model = name==best
                badge = "⭐ BEST" if is_best_model else ""
                with col:
                    st.markdown(f"""
                    <div class="model-card" style="border-top:3px solid {mc};
                                padding:20px;text-align:center;">
                        <div style="font-size:13px;font-weight:800;color:{mc};">
                            {icon} {name} {badge}</div>
                        <div style="font-size:42px;font-weight:900;color:white;
                                    margin:14px 0 4px;">{score}</div>
                        <div style="font-size:12px;color:{MUTED};
                                    margin-bottom:14px;">out of 30</div>
                        <div style="font-size:22px;font-weight:800;
                                    color:{info['color']};margin-bottom:6px;">
                            {info['emoji']} {cat} Stress</div>
                        <div style="font-size:11px;color:{MUTED};">
                            Range: {info['range']}</div>
                    </div>""", unsafe_allow_html=True)

            # ── Gauges for all 3
            st.markdown('<div class="section-title">Gauge Charts</div>',
                        unsafe_allow_html=True)
            g1,g2,g3 = st.columns(3)
            for col,name in zip([g1,g2,g3],model_names):
                score,_ = predictions[name]
                with col:
                    st.markdown(f"<div style='text-align:center;font-size:12px;"
                                f"font-weight:700;color:{model_colors_map[name]};"
                                f"margin-bottom:4px;'>{name}</div>",unsafe_allow_html=True)
                    st.pyplot(gauge_chart(score),use_container_width=True)

            # ── Score ruler
            st.markdown('<div class="section-title">Score Position on Scale</div>',
                        unsafe_allow_html=True)
            for name in model_names:
                score,cat = predictions[name]
                st.markdown(f"<div style='font-size:12px;color:{model_colors_map[name]};"
                            f"font-weight:600;margin-bottom:2px;'>{model_icons[name]} {name}</div>",
                            unsafe_allow_html=True)
                st.pyplot(score_ruler(score),use_container_width=True)

            # ── Explanation
            # Use best model prediction for explanation
            score_best, cat_best = predictions[best]
            info = STRESS_INFO[cat_best]
            st.markdown('<div class="section-title">What Does This Mean?</div>',
                        unsafe_allow_html=True)
            st.markdown(f"""
            <div class="{info['css']}">
                <div style="font-size:42px;">{info['emoji']}</div>
                <div style="font-size:20px;font-weight:800;
                            color:{info['color']};margin-top:10px;">
                    {info['title']}</div>
                <div style="font-size:14px;color:{info['color']};
                            margin-top:10px;line-height:1.8;">
                    {info['what']}</div>
            </div>""", unsafe_allow_html=True)

            cx,cy = st.columns(2)
            with cx:
                st.markdown(f"**Common causes:**")
                for c in info["causes"]: st.markdown(f"• {c}")
            with cy:
                st.markdown(f"**Recommended actions:**")
                for t in info["tips"]:   st.markdown(t)

            st.markdown(f"""<div class="{info['box']}">
            <strong>Action:</strong> {info['action']}</div>""",
                unsafe_allow_html=True)

    # ── 📈 ALL CHARTS ─────────────────────────────────────
    elif "All Charts" in page:
        choice = st.selectbox("Select chart:", [
            "Feature Importance",
            "Stress Score Distribution",
            "Feature Correlation",
            "Actual vs Predicted — All Models",
            "Confusion Matrix — All Models",
        ])
        if   choice=="Feature Importance":
            st.pyplot(chart_importance(results),use_container_width=True)
        elif choice=="Stress Score Distribution":
            f=chart_distribution(df_raw)
            if f: st.pyplot(f,use_container_width=True)
        elif choice=="Feature Correlation":
            f=chart_correlation(df_raw)
            if f: st.pyplot(f,use_container_width=True)
        elif choice=="Actual vs Predicted — All Models":
            for m in results: st.pyplot(chart_scatter(results,m),use_container_width=True)
        elif choice=="Confusion Matrix — All Models":
            cc1,cc2,cc3=st.columns(3)
            for col,m in zip([cc1,cc2,cc3],results):
                with col: st.pyplot(chart_confusion(results,m),use_container_width=True)

    # ── 📉 STRESS LEVELS GUIDE ──────────────────────────
    elif "Stress Levels" in page:
        st.markdown('<div class="section-title">What Does Each Level Mean?</div>',
                    unsafe_allow_html=True)
        lc1,lc2,lc3=st.columns(3)
        for col,cat in zip([lc1,lc2,lc3],["Low","Medium","High"]):
            info=STRESS_INFO[cat]
            with col:
                st.markdown(f"""
                <div class="level-card"
                     style="background:#FFFFFF;border:1px solid {info['color']};">
                    <h3 style="color:{info['color']};">{info['emoji']} {cat.upper()}</h3>
                    <p style="font-size:24px;font-weight:900;color:{info['color']};">
                        {info['range']}</p>
                    <p style="color:#0C447C;font-size:13px;
                               margin-top:8px;line-height:1.7;">
                        {info['what']}</p>
                </div>""", unsafe_allow_html=True)

        # Score ruler
        st.markdown('<div class="section-title">Visual Score Guide</div>',
                    unsafe_allow_html=True)
        fig,ax=plt.subplots(figsize=(10,1.8))
        fig.patch.set_facecolor(DARK_BG); ax.axis("off")
        for x,w,c,l in [(0,10,"#388E3C","LOW  (0–10)"),
                         (10,10,"#F57F17","MEDIUM  (11–20)"),
                         (20,10,"#C62828","HIGH  (21–30)")]:
            ax.barh(0,w,left=x,color=c,height=.5,edgecolor=DARK_BG,lw=2)
            ax.text(x+w/2,0,l,ha="center",va="center",
                    fontsize=12,fontweight="700",color="white")
        for v in [0,5,10,15,20,25,30]:
            ax.text(v,-.38,str(v),ha="center",fontsize=10,color=MUTED)
        ax.set_xlim(-.5,30.5); ax.set_ylim(-.5,.5)
        plt.tight_layout(pad=0)
        st.pyplot(fig,use_container_width=True)

        for cat in ["Low","Medium","High"]:
            info=STRESS_INFO[cat]
            with st.expander(f"{info['emoji']} {cat} Stress — Causes & Actions",expanded=False):
                ea,eb=st.columns(2)
                with ea:
                    st.markdown("**Common causes:**")
                    for c in info["causes"]: st.markdown(f"• {c}")
                with eb:
                    st.markdown("**Recommended actions:**")
                    for t in info["tips"]:   st.markdown(t)
                st.markdown(f"""<div class="{info['box']}">
                <strong>Summary:</strong> {info['action']}</div>""",
                    unsafe_allow_html=True)

    # ── ℹ️ ABOUT ─────────────────────────────────────────
    elif "About" in page:
        st.markdown('<div class="section-title">About This Project</div>',
                    unsafe_allow_html=True)
        st.markdown("""<div class="info-box">
        This system predicts stress levels of university students using machine learning.
        Built as part of an IEEE research paper comparing three regression models on
        data from 3,000 university students.
        </div>""", unsafe_allow_html=True)
        ab1,ab2=st.columns(2)
        with ab1:
            st.markdown("**Models & Results:**")
            st.markdown("- ✅ Linear Regression — R²=0.85, Acc=81.33% ← Best")
            st.markdown("- ✅ SVR (RBF) — R²=0.80, Acc=79.50%")
            st.markdown("- ✅ Random Forest — R²=0.76, Acc=79.83%")
            st.markdown("**Score Categories:**")
            st.markdown("- 😊 0–10 → Low Stress")
            st.markdown("- 😐 11–20 → Medium Stress")
            st.markdown("- 😟 21–30 → High Stress")
        with ab2:
            st.markdown("**Top stress factors:**")
            st.markdown("- 📱 Screen Time (26.25%) — #1 cause")
            st.markdown("- 📝 Exam Frequency (15.38%)")
            st.markdown("- 👨‍👩‍👧 Family Support (15.25%) — reduces stress")
            st.markdown("- 📚 Assignment Load (14.32%)")
            st.markdown("- 📲 Social Media (12.12%)")
            st.markdown("**Libraries:**")
            st.code("streamlit  pandas  numpy\nscikit-learn  matplotlib  seaborn  joblib",
                    language="text")
        st.markdown('<div class="section-title">How to Run</div>',unsafe_allow_html=True)
        st.code("""# Install
pip install streamlit pandas numpy scikit-learn matplotlib seaborn joblib

# Put student_stress.csv in same folder as app.py

# Run
streamlit run app.py""", language="bash")

if __name__=="__main__":
    main()