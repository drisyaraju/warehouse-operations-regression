import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="WarehouseIQ",
    page_icon="🏭",
    layout="wide"
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0f1117; }
    .stApp { background-color: #0f1117; color: #e0e0e0; }
    .metric-card {
        background: linear-gradient(135deg, #1e2130, #2a2d3e);
        border-radius: 12px;
        padding: 20px;
        border: 1px solid #3a3d4e;
        text-align: center;
    }
    .metric-value { font-size: 2rem; font-weight: 700; color: #4ade80; }
    .metric-label { font-size: 0.85rem; color: #9ca3af; margin-top: 4px; }
    .prediction-box {
        background: linear-gradient(135deg, #1a2e1a, #1e3a1e);
        border: 2px solid #4ade80;
        border-radius: 16px;
        padding: 30px;
        text-align: center;
    }
    .prediction-value { font-size: 3rem; font-weight: 800; color: #4ade80; }
    .prediction-label { font-size: 1rem; color: #9ca3af; }
    h1, h2, h3 { color: #f0f0f0 !important; }
    .stSlider > div > div > div { background: #4ade80 !important; }
    .stSelectbox label { color: #9ca3af !important; }
    .stSlider label { color: #9ca3af !important; }
    div[data-testid="stMetricValue"] { color: #4ade80 !important; }
</style>
""", unsafe_allow_html=True)

# ─── Load & Train ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("warehouse_train.csv")
    return df

@st.cache_resource
def train_models(df):
    feature_cols = [
        "Num_Items_Ordered", "Warehouse_Distance_m", "Num_Workers_Available",
        "Order_Weight_kg", "Hour_of_Day", "Num_Pending_Orders",
        "Packing_Complexity_Score", "Equipment_Downtime_min"
    ]
    X = df[feature_cols]
    y = df["Order_Processing_Time_min"]

    models = {
        "🌲 Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "📈 Linear Regression": LinearRegression(),
        "🌳 Decision Tree": DecisionTreeRegressor(max_depth=5, random_state=42),
        "🌀 Polynomial Regression": Pipeline([
            ("poly", PolynomialFeatures(degree=2, include_bias=False)),
            ("lr", LinearRegression())
        ]),
        "⚡ SVM": SVR(kernel="rbf", C=10),
    }

    trained = {}
    metrics = {}
    for name, model in models.items():
        model.fit(X, y)
        preds = model.predict(X)
        trained[name] = model
        metrics[name] = {
            "R²": round(r2_score(y, preds), 4),
            "RMSE": round(np.sqrt(mean_squared_error(y, preds)), 4),
            "MAE": round(mean_absolute_error(y, preds), 4),
        }
    return trained, metrics, feature_cols

# ─── App ──────────────────────────────────────────────────────────────────────
st.markdown("# 🏭 WarehouseIQ")
st.markdown("#### Order Processing Time Predictor")
st.markdown("---")

try:
    df = load_data()
    trained_models, metrics, feature_cols = train_models(df)

    tab1, tab2, tab3 = st.tabs(["🔮 Predict", "📊 Model Comparison", "📂 Dataset"])

    # ── Tab 1: Predict ────────────────────────────────────────────────────────
    with tab1:
        st.markdown("### Configure Order Parameters")
        st.markdown("Adjust the sliders below and get an instant processing time prediction.")

        col1, col2 = st.columns([1, 1])

        with col1:
            model_choice = st.selectbox(
                "Choose Model",
                list(trained_models.keys()),
                index=0
            )
            num_items = st.slider("📦 Num Items Ordered (scaled)", -2.0, 2.0, 0.0, 0.01)
            distance = st.slider("📍 Warehouse Distance (scaled)", -2.0, 2.0, 0.0, 0.01)
            workers = st.slider("👷 Num Workers Available (scaled)", -2.0, 2.0, 0.0, 0.01)
            weight = st.slider("⚖️ Order Weight kg (scaled)", -2.0, 2.0, 0.0, 0.01)

        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            hour = st.slider("🕐 Hour of Day (scaled)", -2.0, 2.0, 0.0, 0.01)
            pending = st.slider("📋 Num Pending Orders (scaled)", -2.0, 2.0, 0.0, 0.01)
            complexity = st.slider("🔧 Packing Complexity Score (scaled)", -2.0, 2.0, 0.0, 0.01)
            downtime = st.slider("⛔ Equipment Downtime min (scaled)", -2.0, 2.0, 0.0, 0.01)

        input_data = pd.DataFrame([[
            num_items, distance, workers, weight,
            hour, pending, complexity, downtime
        ]], columns=feature_cols)

        prediction = trained_models[model_choice].predict(input_data)[0]

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"""
        <div class="prediction-box">
            <div class="prediction-label">Estimated Processing Time</div>
            <div class="prediction-value">{prediction:.1f} min</div>
            <div class="prediction-label">using {model_choice}</div>
        </div>
        """, unsafe_allow_html=True)

        m = metrics[model_choice]
        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{m["R²"]}</div><div class="metric-label">R² Score</div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{m["RMSE"]}</div><div class="metric-label">RMSE</div></div>', unsafe_allow_html=True)
        with c3:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{m["MAE"]}</div><div class="metric-label">MAE</div></div>', unsafe_allow_html=True)

    # ── Tab 2: Model Comparison ────────────────────────────────────────────────
    with tab2:
        st.markdown("### 📊 Model Performance Comparison")

        metrics_df = pd.DataFrame(metrics).T.reset_index()
        metrics_df.columns = ["Model", "R²", "RMSE", "MAE"]
        metrics_df = metrics_df.sort_values("R²", ascending=False)

        st.dataframe(
            metrics_df.style.highlight_max(subset=["R²"], color="#1a3a1a")
                            .highlight_min(subset=["RMSE", "MAE"], color="#1a3a1a"),
            use_container_width=True
        )

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        fig.patch.set_facecolor("#0f1117")

        for ax, metric in zip(axes, ["R²", "RMSE", "MAE"]):
            colors = ["#4ade80" if i == 0 else "#3a3d4e" for i in range(len(metrics_df))]
            bars = ax.barh(metrics_df["Model"], metrics_df[metric], color=colors)
            ax.set_facecolor("#1e2130")
            ax.tick_params(colors="white")
            ax.set_title(metric, color="white", fontsize=13, fontweight="bold")
            for spine in ax.spines.values():
                spine.set_edgecolor("#3a3d4e")

        plt.tight_layout()
        st.pyplot(fig)

    # ── Tab 3: Dataset ─────────────────────────────────────────────────────────
    with tab3:
        st.markdown("### 📂 Training Dataset")
        st.markdown(f"**{len(df)} rows × {len(df.columns)} columns**")
        st.dataframe(df, use_container_width=True)

        st.markdown("### 📈 Feature Statistics")
        st.dataframe(df.describe().round(3), use_container_width=True)

except FileNotFoundError:
    st.error("⚠️ `warehouse_train.csv` not found. Make sure it's in the same folder as `app.py`.")

# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<center style='color:#4a4d5e; font-size:0.85rem'>Built by <b>Drisya Raju</b> · WarehouseIQ · Powered by Streamlit</center>",
    unsafe_allow_html=True
)
