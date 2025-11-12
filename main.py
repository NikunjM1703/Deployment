import pickle
import streamlit as st
import pandas as pd
import numpy as np
import time
import traceback
from pathlib import Path
from sklearn.compose import ColumnTransformer

st.set_page_config(page_title="Customer Segmentation Web App", layout="centered")
st.title("Customer Segmentation Web App")

MODEL_PATH = Path("classifier.pkl")

@st.cache_resource
def load_model(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Model file not found at: {path.resolve()}")
    with open(path, "rb") as f:
        obj = pickle.load(f)
    # Support both: direct model, or dict with {'model': model, 'feature_names': [...]}
    if isinstance(obj, dict) and 'model' in obj:
        return obj['model'], obj.get('feature_names', None)
    # else it's likely the model itself
    return obj, None

# small helper guesses for widgets / encodings (update if your training used different mapping)
PARTNER_MAP = {"Yes": 1, "No": 0}
EDU_MAP_LABELS = {"Undergraduate": 3, "Graduate": 3, "Postgraduate": 4, "Basic": 1, "2n Cycle": 2, "Graduation": 3, "Master": 4, "PhD": 5}
MARITAL_OPTIONS = ["Married", "Together", "Divorced", "Widow", "Alone", "Absurd", "YOLO", "Unknown"]
CLUSTER_LABELS = {0: "cluster 0", 1: "cluster 1", 2: "cluster 2", 3: "cluster 3"}

# load model
model = None
saved_feature_names = None
model_error = None
try:
    model, saved_feature_names = load_model(MODEL_PATH)
    loaded = True
except Exception as e:
    model_error = e
    loaded = False

with st.expander("Model status (click to open)", expanded=True):
    if loaded:
        st.success("Model loaded successfully.")
        st.write("Model type:", type(model))
        if saved_feature_names:
            st.info("Feature names were loaded from the pickle wrapper object.")
    else:
        st.error("Failed to load model.")
        st.text(traceback.format_exc())

def get_expected_columns(model, saved_feature_names=None):
    """Try several ways to infer the expected feature names/order for the pipeline."""
    # 1) saved list passed through pickle wrapper
    if saved_feature_names:
        return list(saved_feature_names)

    # 2) feature_names_in_ (scikit-learn estimators/pipelines may have it)
    try:
        if hasattr(model, "feature_names_in_"):
            return list(model.feature_names_in_)
    except Exception:
        pass

    # 3) inspect ColumnTransformer inside pipeline to get transformer column lists
    try:
        if hasattr(model, "named_steps"):
            for name, step in getattr(model, "named_steps", {}).items():
                if isinstance(step, ColumnTransformer):
                    # try step.feature_names_in_ first
                    try:
                        if hasattr(step, "feature_names_in_"):
                            return list(step.feature_names_in_)
                    except Exception:
                        pass
                    # fallback: collect column lists used in the transformers (if present)
                    cols = []
                    for trans_name, trans_obj, trans_cols in step.transformers:
                        if isinstance(trans_cols, (list, tuple)):
                            cols.extend(list(trans_cols))
                        # if trans_cols is a slice, string, or callable we can't easily recover
                    if cols:
                        return cols
    except Exception:
        pass

    # 4) if pipeline is a single ColumnTransformer (not in named_steps)
    try:
        if isinstance(model, ColumnTransformer):
            cols = []
            for trans_name, trans_obj, trans_cols in model.transformers:
                if isinstance(trans_cols, (list, tuple)):
                    cols.extend(list(trans_cols))
            if cols:
                return cols
    except Exception:
        pass

    # Can't discover
    return None

expected_cols = None
if loaded:
    expected_cols = get_expected_columns(model, saved_feature_names)

st.markdown("---")
st.subheader("Enter customer information")

st.write("Detected expected columns (inferred):")
if expected_cols:
    st.write(f"{len(expected_cols)} features detected")
    st.write(expected_cols)
else:
    st.warning("Could not detect full expected column list automatically. The app will attempt a sensible fallback form, but the model may still fail unless it can accept the provided columns.")

# Robust coercion / mapping function to make the input numeric / ordered
def coerce_row_to_numeric(expected_cols, df_input):
    """
    Convert values in df_input to numeric encodings that the pipeline expects.
    - Maps common marital/education text to numeric codes.
    - Tries to cast numeric strings to float.
    - If a value cannot be coerced, replaces with 0 and warns.
    Returns a DataFrame ordered as expected_cols.
    """
    MARITAL_TO_BINARY = {
        'married': 1, 'together': 1, 'partner': 1,
        'single': 0, 'divorced': 0, 'widow': 0, 'alone': 0,
        'absurd': 0, 'yolo': 0, 'widowed': 0, 'unknown': 0
    }
    EDU_TO_CODE = {
        'basic': 1, '2n cycle': 2, 'graduation': 3, 'master': 4, 'phd': 5,
        'undergraduate': 3, 'graduate': 3, 'postgraduate': 4, 'other': 0
    }

    df = df_input.copy()

    for col in df.columns:
        val = df.at[0, col]
        # numeric-like values: accept ints / floats
        if isinstance(val, (int, float, np.integer, np.floating)):
            continue
        # None or NaN
        if pd.isna(val):
            df.at[0, col] = 0
            continue
        s = str(val).strip()
        # try numeric cast
        try:
            # allow comma separators
            s_clean = s.replace(",", "")
            df.at[0, col] = float(s_clean)
            continue
        except Exception:
            pass

        # lower-case for mapping
        s_lower = s.lower()

        # handle marital/partner column heuristics
        if 'marital' in col.lower() or 'partner' in col.lower() or col.lower() in ('marital_status','maritalstatus','living_with_partner','partner'):
            mapped = MARITAL_TO_BINARY.get(s_lower, None)
            if mapped is None:
                mapped = 1 if 'mar' in s_lower or 'together' in s_lower or 'partner' in s_lower else 0
            df.at[0, col] = mapped
            continue

        # education heuristics
        if 'education' in col.lower() or 'edu' in col.lower():
            mapped = EDU_TO_CODE.get(s_lower, None)
            if mapped is None:
                # if string contains digits, try that
                import re
                digits = re.findall(r"\d+", s_lower)
                if digits:
                    try:
                        mapped = int(digits[0])
                    except Exception:
                        mapped = 0
                else:
                    mapped = 0
            df.at[0, col] = mapped
            continue

        # yes/no or boolean-like
        if s_lower in ('yes','y','true','t','1'):
            df.at[0, col] = 1
            continue
        if s_lower in ('no','n','false','f','0'):
            df.at[0, col] = 0
            continue

        # fallback
        st.warning(f"Could not coerce column '{col}' value '{val}'. Replacing with 0 (check expected encoding).")
        df.at[0, col] = 0

    # Reorder to expected_cols if possible
    if expected_cols:
        # include only expected cols present in df; fill missing expected cols with 0
        out = {}
        for c in expected_cols:
            if c in df.columns:
                out[c] = df.at[0, c]
            else:
                st.warning(f"Expected column '{c}' not present in input; filling with 0.")
                out[c] = 0
        df_final = pd.DataFrame([out])
        return df_final
    else:
        return df

# Build UI
if not expected_cols:
    # Minimal fallback form (user must update mappings at top if training mapping differs)
    st.warning("Using fallback simple form - this will only work if your pipeline accepts these columns or performs its own encoding.")
    Income = st.text_input("Household Income (numeric)", "0")
    Kidhome = st.selectbox("Number Of Kids In Household", ("0","1","2"))
    Teenhome = st.selectbox("Number Of Teens In Household", ("0","1","2"))
    Age = st.slider("Age", 18, 100, value=30)
    Partner = st.selectbox("Living With Partner?", ("Yes","No"))
    Education_Level = st.selectbox("Education", ("Undergraduate","Graduate","Postgraduate","Basic","2n Cycle","Graduation","Master","PhD"))

    submit = st.button("Segment Customer")
    if submit:
        if not loaded:
            st.error("Model not loaded — upload classifier.pkl or check model status.")
        else:
            try:
                raw = {
                    "Income": Income,
                    "Kidhome": Kidhome,
                    "Teenhome": Teenhome,
                    "Age": Age,
                    "marital_status": PARTNER_MAP.get(Partner, 0),
                    "Education_Encoded": EDU_MAP_LABELS.get(Education_Level, 0),
                }
                df_input = pd.DataFrame([raw])
                st.write("Raw input:")
                st.dataframe(df_input)
                df_input_coerced = coerce_row_to_numeric(expected_cols=None, df_input=df_input)
                st.write("After coercion (fallback):")
                st.dataframe(df_input_coerced)
                with st.spinner("Running pipeline..."):
                    prediction = model.predict(df_input_coerced)
                pred_int = int(np.asarray(prediction).ravel()[0])
                label = CLUSTER_LABELS.get(pred_int, f"cluster {pred_int}")
                st.success(f"Prediction: **{label}**")
                st.write("Raw model output:", prediction)
            except Exception:
                st.error("Prediction failed — see traceback.")
                st.text(traceback.format_exc())

else:
    # Build dynamic form with detected expected columns
    st.info(f"Detected {len(expected_cols)} expected features from the pipeline.")
    st.write("You can adjust values before prediction. (Defaults chosen sensibly.)")

    raw_values = {}
    with st.form("dynamic_form"):
        for col in expected_cols:
            col_display = col  # we can prettify if desired
            # Heuristic widget selection
            if any(k in col.lower() for k in ("age", "recency", "children", "kid", "teen", "count", "size")):
                if "age" in col.lower():
                    default = 30
                    val = st.number_input(col_display, min_value=0, max_value=120, value=default, step=1, format="%d")
                    raw_values[col] = int(val)
                else:
                    val = st.number_input(col_display, min_value=0, max_value=10000, value=0, step=1, format="%d")
                    raw_values[col] = int(val)
            elif any(k in col.lower() for k in ("mnt", "total", "spent", "amount", "income", "money")):
                val = st.number_input(col_display, min_value=0.0, max_value=1e9, value=0.0, step=1.0, format="%.2f")
                raw_values[col] = float(val)
            elif "num" in col.lower() or "visits" in col.lower() or "purchases" in col.lower() or "catalog" in col.lower():
                val = st.number_input(col_display, min_value=0, max_value=10000, value=0, step=1, format="%d")
                raw_values[col] = int(val)
            elif "partner" in col.lower() or col.lower() in ("partner","livingwithpartner","marital_status"):
                opt = st.selectbox(col_display, ("Yes","No"))
                raw_values[col] = opt  # keep text; coercion will map it
            elif "education" in col.lower() or "edu" in col.lower():
                opt = st.selectbox(col_display, ("Undergraduate","Graduate","Postgraduate","Basic","2n Cycle","Graduation","Master","PhD","Other"))
                raw_values[col] = opt
            elif col.lower() in ("response","acceptedcmp1","acceptedcmp2","acceptedcmp3","acceptedcmp4","acceptedcmp5","complain"):
                val = st.selectbox(col_display, (0,1))
                raw_values[col] = int(val)
            else:
                # fallback numeric input with text option
                try:
                    val = st.text_input(col_display, value="0")
                    # store string; coercion will try to cast or map
                    raw_values[col] = val
                except Exception:
                    raw_values[col] = 0

        submitted = st.form_submit_button("Segment Customer")

    if submitted:
        if not loaded:
            st.error("Model not loaded — cannot predict.")
        else:
            try:
                # Build DataFrame in exact expected column order
                df_input = pd.DataFrame([{c: raw_values.get(c, 0) for c in expected_cols}])
                st.write("Input sent to model (raw):")
                st.dataframe(df_input)

                # Coerce and map to numeric encodings & ensure correct order
                df_coerced = coerce_row_to_numeric(expected_cols, df_input)
                st.write("Input after coercion (ordered to expected columns):")
                st.dataframe(df_coerced)

                with st.spinner("Running pipeline..."):
                    prog = st.progress(0)
                    for p in range(0, 101, 20):
                        prog.progress(p)
                        time.sleep(0.02)

                prediction = model.predict(df_coerced)
                pred_int = int(np.asarray(prediction).ravel()[0])
                label = CLUSTER_LABELS.get(pred_int, f"cluster {pred_int}")
                st.success(f"Prediction: **{label}**")
                st.write("Raw model output:", prediction)
            except Exception:
                st.error("Model prediction failed — see traceback.")
                st.text(traceback.format_exc())

st.caption("If your model expects particular categorical encodings (e.g. specific integers for education/marital labels), update the small mapping at the top of this file to match your training data. If you trained the model and saved feature names separately, re-export the pickle in the form {'model': model, 'feature_names': list(X.columns)} to help Streamlit auto-detect the exact expected columns.")
