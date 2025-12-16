import io
import json
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score

st.set_page_config(page_title="Smart Logistics Delay Dashboard", layout="wide")

@st.cache_resource
def load_bundle():
    return joblib.load("artifacts/model_bundle.joblib")

@st.cache_data
def load_model_results():
    return pd.read_csv("artifacts/model_results.csv")

def preprocess_raw_for_inference(df_raw: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    df = df_raw.copy()

    # Basic validation
    required_cols = [
        "Timestamp","Asset_ID","Traffic_Status",
        "Latitude","Longitude","Inventory_Level","Temperature","Humidity",
        "User_Transaction_Amount","User_Purchase_Frequency","Asset_Utilization","Demand_Forecast"
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Drop leaky columns if present
    for c in bundle["leaky_features"]:
        if c in df.columns:
            df = df.drop(columns=[c])

    # Timestamp -> time categories
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df = df.sort_values("Timestamp").reset_index(drop=True)

    df["Month_num"] = df["Timestamp"].dt.month
    df["DayOfWeek_num"] = df["Timestamp"].dt.dayofweek
    df["Hour_num"] = df["Timestamp"].dt.hour

    month_map = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
    day_map = {0:"Mon",1:"Tue",2:"Wed",3:"Thu",4:"Fri",5:"Sat",6:"Sun"}

    df["Month_cat"] = df["Month_num"].map(month_map)
    df["Day_cat"] = df["DayOfWeek_num"].map(day_map)
    df["Hour_cat"] = df["Hour_num"].astype(str)

    # Lag + rolling (same as your notebook)
    lag_features = [
        "Inventory_Level","Temperature","Humidity",
        "User_Transaction_Amount","User_Purchase_Frequency",
        "Asset_Utilization","Demand_Forecast"
    ]
    for col in lag_features:
        df[f"{col}_lag1"] = df[col].shift(1)

    df["Temp_roll3"] = df["Temperature"].rolling(window=3).mean()
    df["Humidity_roll3"] = df["Humidity"].rolling(window=3).mean()
    df["Inventory_roll3"] = df["Inventory_Level"].rolling(window=3).mean()
    df["Demand_roll3"] = df["Demand_Forecast"].rolling(window=3).mean()

    df = df.dropna().reset_index(drop=True)

    # Asset_ID label encoding
    le_asset = bundle["asset_label_encoder"]
    df["Asset_ID_encoded"] = le_asset.transform(df["Asset_ID"])

    # One-hot: Traffic
    traffic_dummies = pd.get_dummies(df["Traffic_Status"], prefix="Traffic").astype(int)
    traffic_dummies = traffic_dummies.reindex(columns=bundle["traffic_order"], fill_value=0)

    # One-hot: Month/Day/Hour
    month_dummies = pd.get_dummies(df["Month_cat"], prefix="Month").astype(int)
    month_dummies = month_dummies.reindex(columns=bundle["month_order"], fill_value=0)

    day_dummies = pd.get_dummies(df["Day_cat"], prefix="Day").astype(int)
    day_dummies = day_dummies.reindex(columns=bundle["day_order"], fill_value=0)

    hour_dummies = pd.get_dummies(df["Hour_cat"], prefix="Hour").astype(int)
    hour_dummies = hour_dummies.reindex(columns=bundle["hour_order"], fill_value=0)

    df_feat = pd.concat([df, traffic_dummies, month_dummies, day_dummies, hour_dummies], axis=1)

    # Drop helper cols
    drop_cols = [
        "Timestamp","Asset_ID","Traffic_Status",
        "Month_num","DayOfWeek_num","Hour_num",
        "Month_cat","Day_cat","Hour_cat","Hour_num"
    ]
    df_feat = df_feat.drop(columns=[c for c in drop_cols if c in df_feat.columns], errors="ignore")

    # Align to training feature columns
    feature_cols = bundle["feature_cols"]
    X = df_feat.reindex(columns=feature_cols, fill_value=0)

    return X, df.loc[X.index].reset_index(drop=True)

def plot_confusion(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    fig = go.Figure(data=go.Heatmap(z=cm, x=["No Delay","Delay"], y=["No Delay","Delay"]))
    fig.update_layout(title="Confusion Matrix", xaxis_title="Predicted", yaxis_title="Actual")
    return fig

def plot_roc(y_true, y_prob):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"ROC (AUC={auc:.3f})"))
    fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", name="Random", line=dict(dash="dash")))
    fig.update_layout(title="ROC Curve", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
    return fig

st.title("🚚 Smart Logistics Delay Prediction Dashboard")

with st.sidebar:
    st.header("Artifacts")
    bundle = load_bundle()
    st.success(f"Loaded model: {bundle['best_model_name']}")
    st.caption("Upload CSV to score shipments. If your CSV is small, lag/rolling will drop first few rows.")

tab1, tab2, tab3 = st.tabs(["Overview", "Model Comparison", "Score a CSV"])

with tab1:
    st.subheader("What this app lets you do")
    st.markdown(
        "- Explore model results (F1, ROC-AUC, overfitting gap)\n"
        "- Inspect confusion matrix + ROC curve\n"
        "- Upload a dataset CSV and produce delay probability predictions\n"
        "- Download scored results for reporting / demo\n"
    )
    st.info(
        "Note: We intentionally drop leaky features (e.g., Shipment_Status, Logistics_Delay_Reason, Waiting_Time) "
        "before training/scoring to avoid target leakage."
    )

with tab2:
    st.subheader("Model leaderboard")
    results_df = load_model_results()
    st.dataframe(results_df, use_container_width=True)

    metric = st.selectbox("Metric to visualize", ["F1","ROC_AUC","Test_Acc","Gap"])
    fig = px.bar(results_df.sort_values(metric, ascending=False), x="Model", y=metric)
    fig.update_layout(xaxis_title="", yaxis_title=metric)
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.subheader("Upload CSV and score delay probability")
    uploaded = st.file_uploader("Upload smart_logistics_dataset.csv", type=["csv"])
    if uploaded is not None:
        df_raw = pd.read_csv(uploaded)

        st.write("Preview:")
        st.dataframe(df_raw.head(10), use_container_width=True)

        run_btn = st.button("Run scoring", type="primary")
        if run_btn:
            try:
                X, df_aligned = preprocess_raw_for_inference(df_raw, bundle)
                model = bundle["model"]

                if hasattr(model, "predict_proba"):
                    prob = model.predict_proba(X)[:, 1]
                else:
                    prob = model.decision_function(X)
                    prob = (prob - prob.min()) / (prob.max() - prob.min() + 1e-12)

                pred = (prob >= 0.5).astype(int)

                out = df_aligned.copy()
                out["pred_delay_prob"] = prob
                out["pred_delay_label"] = pred

                st.success(f"Scored rows: {len(out)} (rows dropped due to lag/rolling: {len(df_raw) - len(out)})")

                c1, c2 = st.columns(2)
                with c1:
                    st.plotly_chart(px.histogram(out, x="pred_delay_prob", nbins=30, title="Predicted Delay Probability Distribution"), use_container_width=True)
                with c2:
                    topk = out.sort_values("pred_delay_prob", ascending=False).head(20)
                    st.write("Top 20 highest-risk rows")
                    st.dataframe(topk, use_container_width=True)

                csv_bytes = out.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="Download scored CSV",
                    data=csv_bytes,
                    file_name="scored_logistics_delay.csv",
                    mime="text/csv"
                )

            except Exception as e:
                st.error(str(e))
