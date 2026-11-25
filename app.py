import json
import os

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Forest Fire - Andhra Pradesh", layout="wide")

LABELS = {
    "temp_c": "Temperature (deg C)", "rh": "Relative humidity (%)",
    "wind": "Wind speed (m/s)", "rain_mm": "Rainfall on the day (mm)",
    "rain_30d_mm": "Rainfall in previous 30 days (mm)",
    "ndvi_pre": "Vegetation greenness before fire (NDVI)",
    "elevation": "Elevation (m)", "slope": "Slope (degrees)",
    "treecover": "Tree cover (%)", "month": "Month",
    "past_fires_5km": "Past fires within 5 km (earlier years)",
    "frp": "Fire intensity, FRP (MW)",
}
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


@st.cache_resource
def load_assets():
    with open("meta.json") as f:
        meta = json.load(f)
    risk = joblib.load("fire_risk_model.pkl")
    dmg = joblib.load("damage_model.pkl") if os.path.exists("damage_model.pkl") else None
    pts = pd.read_csv("fire_points.csv")
    monthly = pd.read_csv("monthly_counts.csv")
    return meta, risk, dmg, pts, monthly


meta, risk_model, dmg_model, points, monthly = load_assets()


def inputs(features, key):
    """Build one widget per feature, ranges come from the training data."""
    vals, cols = {}, st.columns(3)
    for i, f in enumerate(features):
        c = cols[i % 3]
        lo, hi, med = meta["ranges"][f]
        label = LABELS.get(f, f)
        if f == "month":
            vals[f] = c.selectbox(label, list(range(1, 13)), index=2,
                                  format_func=lambda m: MONTHS[m - 1], key=f"{key}_{f}")
        elif f == "past_fires_5km":
            vals[f] = c.number_input(label, 0, int(max(hi, 1)), int(med), key=f"{key}_{f}")
        else:
            vals[f] = c.slider(label, float(lo), float(hi), float(med), key=f"{key}_{f}")
    return pd.DataFrame([vals])[features]


def level(p, lo, hi):
    return "Low" if p < lo else ("Moderate" if p < hi else "High")


st.title("Forest Fire Risk and Damage Assessment - Andhra Pradesh")
st.caption("Machine learning on NASA satellite fire detections with weather, vegetation and terrain data "
           f"({meta['first_year']}-{meta['last_year']}).")
if meta.get("demo"):
    st.warning("DEMO DATA: results come from synthetic data, not real fires.")

tab1, tab2, tab3, tab4 = st.tabs(["Fire risk predictor", "Damage estimator",
                                  "Fire patterns", "About"])

# ----------------------------------------------------------- risk
with tab1:
    st.write("Enter the conditions at a forest location to get its fire risk level.")
    X = inputs(meta["risk_features"], "risk")
    if st.button("Predict fire risk"):
        p = float(risk_model.predict_proba(X)[0, 1])
        lvl = level(p, *meta["risk_thresholds"])
        st.metric("Fire probability", f"{p:.1%}")
        st.progress(min(max(p, 0.0), 1.0))
        {"Low": st.success, "Moderate": st.warning, "High": st.error}[lvl](f"Risk level: {lvl}")
    st.subheader("What drives fire risk (model feature importance)")
    imp = pd.Series(meta["importance"]).sort_values()
    imp.index = [LABELS.get(i, i) for i in imp.index]
    st.bar_chart(imp)

# ----------------------------------------------------------- damage
with tab2:
    if dmg_model is None:
        st.info("Damage model not available.")
    else:
        st.write("Estimate vegetation damage (change in NDVI after the fire) for a fire.")
        Xd = inputs(meta["damage_features"], "dmg")
        if st.button("Estimate damage"):
            d = float(dmg_model.predict(Xd)[0])
            lo, hi = meta["damage_thresholds"]
            cls = "High" if d <= lo else ("Moderate" if d <= hi else "Low")
            st.metric("Predicted NDVI change", f"{d:+.3f}")
            {"Low": st.success, "Moderate": st.warning, "High": st.error}[cls](
                f"Damage level: {cls} (a more negative change means more vegetation lost)")

# ----------------------------------------------------------- patterns
with tab3:
    st.subheader("Where fires were detected")
    st.map(points, latitude="latitude", longitude="longitude", size=300)
    st.subheader("Monthly fire detections")
    m = monthly.copy()
    m["acq_date"] = pd.to_datetime(m["acq_date"])
    st.line_chart(m.set_index("acq_date")["fires"])

# ----------------------------------------------------------- about
with tab4:
    st.markdown(f"""
**Data:** {meta.get('fire_source', 'NASA satellite')} fire detections inside forest areas (ESA WorldCover),
with ERA5-Land weather (temperature, humidity, wind, rainfall), MODIS NDVI, SRTM terrain,
tree cover and past fire history.

**Fire risk model:** {meta['best_clf']} (test ROC-AUC {meta['clf_auc']:.2f}). Probability is
turned into Low (below {meta['risk_thresholds'][0]:.0%}), Moderate and High (above
{meta['risk_thresholds'][1]:.0%}).

**Damage model:** {meta['best_dmg']} predicts vegetation change (NDVI after minus before).

**Limitations:** satellite detections are a proxy for real fires, no-fire samples are random
forest points, and this is a prototype, not an early-warning system.
""")
