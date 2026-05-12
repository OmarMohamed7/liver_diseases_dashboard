import numpy as np
import pandas as pd
import joblib
import streamlit as st
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler

from .constants import (
    CATEGORICAL, DATA_PATH, FEATURE_ORDER, MODEL_META, NUMERICAL, TARGET
)


@st.cache_data
def load_raw_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def load_models() -> dict:
    return {name: joblib.load(meta["file"]) for name, meta in MODEL_META.items()}


@st.cache_resource
def build_pipeline() -> tuple:
    """Reconstruct preprocessing so new patient inputs can be encoded and scaled."""
    df = pd.read_csv(DATA_PATH)

    imputer = SimpleImputer(strategy="most_frequent")
    df["Alcohol_Consumption"] = imputer.fit_transform(df[["Alcohol_Consumption"]]).ravel()
    df["Medication_History"] = imputer.fit_transform(df[["Medication_History"]]).ravel()

    cat_options = {col: sorted(df[col].dropna().unique().tolist()) for col in CATEGORICAL}

    target_le = LabelEncoder()
    df[TARGET] = target_le.fit_transform(df[TARGET])

    cat_encoders = {}
    for col in CATEGORICAL:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        cat_encoders[col] = le

    for col in NUMERICAL:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        df[col] = np.clip(df[col], q1 - 1.5 * iqr, q3 + 1.5 * iqr)

    scaler = StandardScaler()
    scaler.fit(df[FEATURE_ORDER])

    return cat_options, cat_encoders, target_le, scaler
