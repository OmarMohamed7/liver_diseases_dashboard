import streamlit as st

from dashboard.constants import MODEL_META
from dashboard.loader import build_pipeline, load_models, load_raw_data
from dashboard.pages.eda import page_eda
from dashboard.pages.overview import page_overview
from dashboard.pages.predict import page_predict

st.set_page_config(
    page_title="Liver Disease Dashboard",
    page_icon="🫁",
    layout="wide",
)


def main():
    df = load_raw_data()
    models = load_models()
    cat_options, cat_encoders, target_le, scaler = build_pipeline()

    st.sidebar.title("Liver Disease Dashboard")
    page = st.sidebar.radio(
        "Navigate to",
        ["Overview", "EDA", "Predict"],
        format_func=lambda x: {
            "Overview": "📊  Overview",
            "EDA": "🔍  EDA",
            "Predict": "🔮  Predict",
        }[x],
    )

    st.sidebar.divider()
    st.sidebar.markdown("**Model Performance (test set)**")
    for name, meta in MODEL_META.items():
        st.sidebar.metric(name, f"{meta['accuracy']:.2%}")

    if page == "Overview":
        page_overview(df)
    elif page == "EDA":
        page_eda(df)
    else:
        page_predict(cat_options, cat_encoders, target_le, scaler, models)


main()
