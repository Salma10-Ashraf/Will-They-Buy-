"""Streamlit UI for the Online Shoppers Purchase Intention project.

Two modes:
- If the API_URL environment variable (or Streamlit secret) is set, the app calls the FastAPI service.
- Otherwise it loads the model locally (this is what makes it work on Streamlit Community Cloud alone).
"""
import os

import requests
import streamlit as st

st.set_page_config(page_title="Purchase Intention Predictor", page_icon="🛒", layout="wide")

MONTHS = ["Feb", "Mar", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
VISITOR_TYPES = ["Returning_Visitor", "New_Visitor", "Other"]
SPECIAL_DAYS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]


def get_api_url():
    url = os.environ.get("API_URL", "")
    if not url:
        try:
            url = st.secrets.get("API_URL", "")
        except Exception:
            url = ""
    return url.rstrip("/")


API_URL = get_api_url()


@st.cache_resource
def load_local_predictor():
    import predictor
    return predictor


def get_prediction(payload):
    if API_URL:
        response = requests.post(API_URL + "/predict", json=payload, timeout=30)
        response.raise_for_status()
        return response.json()
    return load_local_predictor().predict(payload)


# ---------- default values and examples ----------
DEFAULTS = {
    "Administrative": 0, "Administrative_Duration": 0.0,
    "Informational": 0, "Informational_Duration": 0.0,
    "ProductRelated": 10, "ProductRelated_Duration": 300.0,
    "BounceRates": 0.02, "ExitRates": 0.05, "PageValues": 0.0,
    "SpecialDay": 0.0, "Month": "Nov",
    "OperatingSystems": 2, "Browser": 2, "Region": 1, "TrafficType": 2,
    "VisitorType": "Returning_Visitor", "Weekend": False,
}

EXAMPLES = {
    "buyer": {
        "Administrative": 3, "Administrative_Duration": 90.0,
        "Informational": 1, "Informational_Duration": 40.0,
        "ProductRelated": 45, "ProductRelated_Duration": 2000.0,
        "BounceRates": 0.0, "ExitRates": 0.015, "PageValues": 25.0,
        "SpecialDay": 0.0, "Month": "Nov", "VisitorType": "Returning_Visitor", "Weekend": False,
    },
    "browser": {
        "Administrative": 0, "Administrative_Duration": 0.0,
        "Informational": 0, "Informational_Duration": 0.0,
        "ProductRelated": 3, "ProductRelated_Duration": 60.0,
        "BounceRates": 0.2, "ExitRates": 0.2, "PageValues": 0.0,
        "SpecialDay": 0.0, "Month": "Feb", "VisitorType": "New_Visitor", "Weekend": False,
    },
}

for key, value in DEFAULTS.items():
    st.session_state.setdefault(key, value)


def load_example(name):
    for key, value in EXAMPLES[name].items():
        st.session_state[key] = value


# ---------- page ----------
st.title("🛒 Purchase Intention Predictor")
st.write("Enter a visitor's browsing behavior and the model predicts whether they will **make a purchase**. "
         "Store owners can use this to focus offers and ads on visitors who are likely to buy.")

st.sidebar.header("Quick examples")
st.sidebar.button("Likely buyer", on_click=load_example, args=("buyer",), use_container_width=True)
st.sidebar.button("Casual browser", on_click=load_example, args=("browser",), use_container_width=True)
st.sidebar.caption("Mode: " + ("API (" + API_URL + ")" if API_URL else "local model"))

st.subheader("Pages visited")
c1, c2, c3 = st.columns(3)
c1.number_input("Administrative pages", min_value=0, step=1, key="Administrative")
c2.number_input("Informational pages", min_value=0, step=1, key="Informational")
c3.number_input("Product pages", min_value=0, step=1, key="ProductRelated")

st.subheader("Time spent (seconds)")
c1, c2, c3 = st.columns(3)
c1.number_input("Administrative duration", min_value=0.0, step=10.0, key="Administrative_Duration")
c2.number_input("Informational duration", min_value=0.0, step=10.0, key="Informational_Duration")
c3.number_input("Product pages duration", min_value=0.0, step=10.0, key="ProductRelated_Duration")

st.subheader("Behavior metrics")
c1, c2, c3 = st.columns(3)
c1.number_input("Bounce rate", min_value=0.0, max_value=1.0, step=0.005, format="%.3f", key="BounceRates")
c2.number_input("Exit rate", min_value=0.0, max_value=1.0, step=0.005, format="%.3f", key="ExitRates")
c3.number_input("Page value", min_value=0.0, step=1.0, key="PageValues")

st.subheader("Visit details")
c1, c2, c3, c4 = st.columns(4)
c1.selectbox("Month", MONTHS, key="Month")
c2.selectbox("Visitor type", VISITOR_TYPES, key="VisitorType")
c3.selectbox("Special day (closeness)", SPECIAL_DAYS, key="SpecialDay")
c4.checkbox("Weekend", key="Weekend")

with st.expander("Technical details (codes)"):
    c1, c2, c3, c4 = st.columns(4)
    c1.number_input("Operating system code", min_value=1, max_value=8, step=1, key="OperatingSystems")
    c2.number_input("Browser code", min_value=1, max_value=13, step=1, key="Browser")
    c3.number_input("Region code", min_value=1, max_value=9, step=1, key="Region")
    c4.number_input("Traffic type code", min_value=1, max_value=20, step=1, key="TrafficType")

if st.button("Predict", type="primary"):
    payload = {key: st.session_state[key] for key in DEFAULTS}
    try:
        result = get_prediction(payload)
    except Exception as error:
        st.error("Could not get a prediction: " + str(error))
    else:
        probability = result["probability"]
        st.divider()
        if result["will_buy"]:
            st.success("✅ " + result["label"])
        else:
            st.warning("❌ " + result["label"])
        st.metric("Purchase probability", str(round(probability * 100, 1)) + "%")
        st.progress(min(max(probability, 0.0), 1.0))
        if result["will_buy"]:
            st.info("Suggestion: this visitor is a good target for a discount or a reminder, since they are likely to convert.")
        else:
            st.info("Suggestion: this visitor is unlikely to buy now. Consider a retargeting campaign instead of spending on instant offers.")
