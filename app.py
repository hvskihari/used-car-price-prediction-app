"""Streamlit web app for used-car price prediction.

Place this file in the project root next to the data/ folder, then run:
    python -m pip install -r requirements_webapp.txt
    python -m streamlit run app.py
"""
from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


st.set_page_config(page_title="Used Car Price Predictor", page_icon="🚗", layout="wide")


def normalize_name(name: str) -> str:
    return "".join(ch.lower() for ch in str(name) if ch.isalnum())


def find_target_column(columns: list[str]) -> str:
    normalized = {normalize_name(col): col for col in columns}
    candidates = [
        "sellingprice", "price", "sellingpriceinlakhs", "sellingpriceinlacs",
        "selling_price", "carprice", "resaleprice", "presentprice"
    ]
    for candidate in candidates:
        if normalize_name(candidate) in normalized:
            return normalized[normalize_name(candidate)]
    raise ValueError(
        "Couldn't identify the price column. Rename the target column to 'Selling_Price' or 'price'. "
        f"Available columns: {', '.join(columns)}"
    )


def find_default_csv() -> Path | None:
    candidates = [
        Path("data/dataset.csv"), Path("data/car data.csv"), Path("data/car_data.csv"),
        Path("dataset.csv"), Path("car data.csv"), Path("car_data.csv")
    ]
    for path in candidates:
        if path.exists() and path.is_file():
            return path
    for folder in (Path("data"), Path(".")):
        if folder.exists():
            found = sorted(p for p in folder.glob("*.csv") if p.is_file())
            if found:
                return found[0]
    return None


@st.cache_resource(show_spinner="Training the prediction model...")
def prepare_model(csv_bytes: bytes):
    df = pd.read_csv(io.BytesIO(csv_bytes))
    df.columns = [str(col).strip() for col in df.columns]
    df = df.loc[:, ~pd.Index(df.columns).str.match(r"^Unnamed: ?\d+$", case=False)]
    df = df.drop_duplicates().copy()
    if df.empty:
        raise ValueError("The CSV has no usable rows.")

    target = find_target_column(list(df.columns))
    df[target] = pd.to_numeric(df[target], errors="coerce")
    df = df.dropna(subset=[target]).copy()
    if df.empty:
        raise ValueError(f"The price column '{target}' contains no numeric values.")

    X = df.drop(columns=[target])
    y = df[target]
    unusable = [c for c in X.columns if X[c].isna().all() or X[c].nunique(dropna=True) <= 1]
    if unusable:
        X = X.drop(columns=unusable)
    if X.shape[1] == 0:
        raise ValueError("No usable feature columns remain after cleaning the dataset.")

    numeric_features = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_features = [c for c in X.columns if c not in numeric_features]

    transformers = []
    if numeric_features:
        transformers.append(("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median"))
        ]), numeric_features))
    if categorical_features:
        transformers.append(("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), categorical_features))

    model = Pipeline([
        ("preprocess", ColumnTransformer(transformers=transformers, remainder="drop")),
        ("regressor", RandomForestRegressor(
            n_estimators=250, min_samples_leaf=2, random_state=42, n_jobs=-1
        ))
    ])
    model.fit(X, y)
    return model, X, y, target, numeric_features, categorical_features, len(df), unusable


st.title("🚗 Used Car Price Predictor")
st.write("Enter the car details below to estimate its selling price using a machine-learning model.")

# Prefer a dataset already present in the project's folder; an uploaded CSV can override it.
default_csv = find_default_csv()
if default_csv is not None:
    st.caption(f"Dataset detected: `{default_csv.as_posix()}`")
else:
    st.info("No CSV dataset was found automatically. Upload your dataset below.")

uploaded = st.file_uploader("Upload your car dataset (CSV)", type=["csv"])
if uploaded is not None:
    csv_bytes = uploaded.getvalue()
    data_label = uploaded.name
elif default_csv is not None:
    csv_bytes = default_csv.read_bytes()
    data_label = default_csv.name
else:
    st.stop()

try:
    model, X, y, target, numeric_features, categorical_features, row_count, unusable = prepare_model(csv_bytes)
except Exception as exc:
    st.error(f"Couldn't prepare the model: {exc}")
    st.markdown("Check that your CSV contains car features and a numeric price column such as `Selling_Price` or `price`.")
    st.stop()

col1, col2, col3 = st.columns(3)
col1.metric("Training records", f"{row_count:,}")
col2.metric("Input features", str(X.shape[1]))
col3.metric("Price column", target)

if unusable:
    st.caption("Ignored empty or constant columns: " + ", ".join(unusable))

st.subheader("Enter car details")
with st.form("car_prediction_form"):
    values = {}
    left, right = st.columns(2)
    for index, feature in enumerate(X.columns):
        container = left if index % 2 == 0 else right
        with container:
            series = X[feature]
            if feature in numeric_features:
                numeric_series = pd.to_numeric(series, errors="coerce").dropna()
                if numeric_series.empty:
                    default_value, min_value, max_value = 0.0, 0.0, 1.0
                else:
                    default_value = float(numeric_series.median())
                    min_value = float(numeric_series.min())
                    max_value = float(numeric_series.max())
                    if min_value == max_value:
                        max_value = min_value + 1.0
                is_integer = pd.api.types.is_integer_dtype(series.dtype) and all(
                    float(v).is_integer() for v in [default_value, min_value]
                )
                values[feature] = st.number_input(
                    str(feature),
                    min_value=int(min_value) if is_integer else min_value,
                    max_value=int(max_value) if is_integer else max_value,
                    value=int(default_value) if is_integer else default_value,
                    step=1 if is_integer else max((max_value - min_value) / 100.0, 0.01),
                    key=f"num_{feature}",
                )
            else:
                options = series.dropna().astype(str).value_counts().index.tolist()
                if not options:
                    options = ["Unknown"]
                values[feature] = st.selectbox(str(feature), options=options, key=f"cat_{feature}")

    submitted = st.form_submit_button("Predict Used Car Price", type="primary")

if submitted:
    input_df = pd.DataFrame([{col: values[col] for col in X.columns}], columns=X.columns)
    try:
        prediction = float(model.predict(input_df)[0])
        st.success(f"Estimated selling price: {prediction:,.2f}")
        st.caption("This is a model estimate based on the uploaded/training data, not a guaranteed market price.")
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
