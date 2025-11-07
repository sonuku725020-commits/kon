import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict, RandomizedSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

# SciPy optional (for distributions); fallback to lists if missing
try:
    from scipy.stats import randint, uniform
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False

import plotly.express as px
import plotly.graph_objects as go


# ------------- Helpers -------------
def load_builtin(name):
    if name == "Iris":
        data = load_iris(as_frame=True)
    elif name == "Wine":
        data = load_wine(as_frame=True)
    elif name == "Breast Cancer":
        data = load_breast_cancer(as_frame=True)
    else:
        raise ValueError("Unknown dataset")
    df = data.frame.copy()
    X = df.drop(columns=[data.target.name])
    y = df[data.target.name]
    return X, y, list(X.columns), list(np.unique(y))

def build_preprocess(X):
    num_cols = X.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=["number"]).columns.tolist()

    numeric = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ])

    # Version-safe OneHotEncoder
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

    categorical = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", ohe)
    ])

    pre = ColumnTransformer(
        transformers=[
            ("num", numeric, num_cols),
            ("cat", categorical, cat_cols),
        ],
        remainder="drop"
    )
    return pre, num_cols, cat_cols

def get_feature_names(pre, num_cols, cat_cols):
    out_names = []
    if len(num_cols) > 0:
        out_names.extend([f"num__{c}" for c in num_cols])
    if len(cat_cols) > 0:
        try:
            ohe = pre.named_transformers_["cat"].named_steps["onehot"]
            try:
                ohe_names = ohe.get_feature_names_out(cat_cols)
            except AttributeError:
                ohe_names = ohe.get_feature_names(cat_cols)
            out_names.extend([f"cat__{n}" for n in ohe_names])
        except Exception:
            pass
    return out_names

def plot_confusion_matrix(y_true, y_pred, labels=None, title="Confusion Matrix"):
    try:
        fig = px.imshow(
            confusion_matrix(y_true, y_pred, labels=labels),
            x=labels, y=labels,
            text_auto=True,
            color_continuous_scale="Blues",
            aspect="equal",
            labels=dict(x="Predicted", y="True", color="Count"),
            title=title
        )
    except TypeError:
        # Fallback for older Plotly without text_auto
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        fig = px.imshow(
            cm,
            x=labels, y=labels,
            color_continuous_scale="Blues",
            aspect="equal",
            labels=dict(x="Predicted", y="True", color="Count"),
            title=title
        )
    return fig

def line_with_error(x, y_mean, y_std, x_title, y_title="CV Score"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y_mean, mode="lines+markers", name="mean"))
    fig.add_trace(go.Scatter(
        x=np.concatenate([x, x[::-1]]),
        y=np.concatenate([y_mean + y_std, (y_mean - y_std)[::-1]]),
        fill='toself',
        fillcolor='rgba(0, 123, 255, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        showlegend=False,
        name="±1 std"
    ))
    fig.update_layout(xaxis_title=x_title, yaxis_title=y_title)
    return fig

def heatmap_2d(x_vals, y_vals, z_matrix, x_name, y_name, title="Accuracy Heatmap"):
    try:
        fig = px.imshow(
            z_matrix,
            x=x_vals,
            y=y_vals,
            text_auto=".3f",
            color_continuous_scale="Viridis",
            aspect="auto",
            labels=dict(color="CV score"),
            title=title
        )
    except TypeError:
        fig = px.imshow(
            z_matrix,
            x=x_vals,
            y=y_vals,
            color_continuous_scale="Viridis",
            aspect="auto",
            labels=dict(color="CV score"),
            title=title
        )
    fig.update_xaxes(title=x_name)
    fig.update_yaxes(title=y_name)
    return fig


# ------------- App -------------
st.set_page_config(page_title="RandomForest Tuning Dashboard", layout="wide")
st.title("RandomForest Hyperparameter Tuning Dashboard")
st.caption("Explore how accuracy changes with hyperparameters (scikit-learn + Streamlit).")

with st.sidebar:
    st.header("Data")
    dataset_choice = st.selectbox(
        "Choose dataset",
        ["Iris", "Wine", "Breast Cancer", "Upload CSV"]
    )

    df, target_col = None, None

    if dataset_choice == "Upload CSV":
        uploaded = st.file_uploader("Upload a CSV file", type=["csv"])
        if uploaded is not None:
            try:
                df = pd.read_csv(uploaded, encoding="utf-8")
            except UnicodeDecodeError:
                # Fallback for non-UTF8 CSVs
                df = pd.read_csv(uploaded, encoding="latin1")
            except Exception as e:
                st.error(f"❌ Failed to read CSV: {e}")
                st.stop()

            if df.empty or df.shape[1] == 0:
                st.error("Uploaded file appears empty or has no valid columns.")
                st.stop()

            st.write(f"✅ Loaded data shape: {df.shape}")

            target_col = st.selectbox("Select target column", df.columns)

            if target_col:
                st.success(f"Target selected: {target_col}")
        else:
            st.info("Please upload a CSV file to continue.")
            st.stop()
    else:
        uploaded = None

    st.header("Cross-Validation")
    cv_splits = st.slider("CV folds", min_value=3, max_value=10, value=5)
    random_state = st.number_input("Random state", min_value=0, max_value=10000, value=42, step=1)
    scoring = st.selectbox("Scoring metric", ["accuracy", "balanced_accuracy", "f1_macro", "f1_weighted"], index=0)
    n_jobs_cv = st.selectbox("Parallel jobs (CV)", options=[1, -1], index=1, help="-1 = use all cores")

    st.header("Basic Hyperparameters")
    n_estimators = st.slider("n_estimators", 10, 1000, 200, step=10)

    max_depth_mode = st.radio("max_depth", ["None", "Set value"], horizontal=True)
    max_depth_val = st.slider("max_depth value", 1, 100, 20) if max_depth_mode == "Set value" else None

    max_features_type = st.radio("max_features", ["sqrt", "log2", "all (None)", "fraction (0-1)"], index=0)
    mf_fraction = st.slider("max_features fraction", 0.1, 1.0, 0.5, step=0.05) if max_features_type == "fraction (0-1)" else None

    min_samples_split = st.slider("min_samples_split", 2, 50, 2)
    min_samples_leaf = st.slider("min_samples_leaf", 1, 50, 1)
    bootstrap = st.checkbox("bootstrap", value=True)
    criterion = st.selectbox("criterion", ["gini", "entropy", "log_loss"], index=0)

    st.header("Advanced Hyperparameters")
    class_weight = st.selectbox("class_weight", [None, "balanced", "balanced_subsample"], index=0)

    max_leaf_nodes_mode = st.radio("max_leaf_nodes", ["None", "Set value"], horizontal=True)
    max_leaf_nodes_val = st.slider("max_leaf_nodes value", 10, 5000, 1000, step=50) if max_leaf_nodes_mode == "Set value" else None

    min_weight_fraction_leaf = st.slider("min_weight_fraction_leaf", 0.0, 0.5, 0.0, step=0.01)
    min_impurity_decrease = st.slider("min_impurity_decrease", 0.0, 0.05, 0.0, step=0.001)
    ccp_alpha = st.slider("ccp_alpha (post-pruning)", 0.0, 0.05, 0.0, step=0.001)

    oob_score = st.checkbox("oob_score (requires bootstrap=True)", value=False, help="Out-of-bag score")
    max_samples_frac = None
    if bootstrap:
        max_samples_toggle = st.checkbox("Limit max_samples (as a fraction of data)", value=False)
        if max_samples_toggle:
            max_samples_frac = st.slider("max_samples fraction", 0.1, 1.0, 0.8, step=0.05)

    warm_start = st.checkbox("warm_start", value=False)
    n_jobs_model = st.selectbox("Parallel jobs (model)", options=[1, -1], index=1, help="-1 = use all cores")

# Load data
if dataset_choice == "Upload CSV":
    if uploaded is not None and target_col is not None:
        y = df[target_col]
        X = df.drop(columns=[target_col])
        feature_cols = list(X.columns)
        class_labels = list(pd.Series(y).unique())
    else:
        st.info("Upload a CSV and choose a target column to continue.")
        st.stop()
else:
    X, y, feature_cols, class_labels = load_builtin(dataset_choice)

# Warn if target looks like regression
if pd.api.types.is_numeric_dtype(y) and y.nunique() > 30:
    st.warning("Target has many unique numeric values; this looks like regression. Please use a classification target.")

# Build preprocess + model
preprocess, num_cols, cat_cols = build_preprocess(X)

# Resolve max_features
if max_features_type == "sqrt":
    max_features = "sqrt"
elif max_features_type == "log2":
    max_features = "log2"
elif max_features_type == "all (None)":
    max_features = None
else:
    max_features = float(mf_fraction)

# Resolve depth/leaf nodes
max_depth = None if max_depth_mode == "None" else int(max_depth_val)
max_leaf_nodes = None if max_leaf_nodes_mode == "None" else int(max_leaf_nodes_val)

# Resolve max_samples
max_samples = None
if bootstrap and max_samples_frac is not None:
    max_samples = float(max_samples_frac)

rf = RandomForestClassifier(
    n_estimators=int(n_estimators),
    criterion=criterion,
    max_depth=max_depth,
    min_samples_split=int(min_samples_split),
    min_samples_leaf=int(min_samples_leaf),
    min_weight_fraction_leaf=float(min_weight_fraction_leaf),
    max_features=max_features,
    max_leaf_nodes=max_leaf_nodes,
    min_impurity_decrease=float(min_impurity_decrease),
    bootstrap=bool(bootstrap),
    oob_score=bool(oob_score) if bootstrap else False,
    n_jobs=n_jobs_model,
    random_state=int(random_state),
    warm_start=bool(warm_start),
    class_weight=class_weight,
    ccp_alpha=float(ccp_alpha),
    max_samples=max_samples
)

pipe = Pipeline(steps=[
    ("preprocess", preprocess),
    ("model", rf)
])

cv = StratifiedKFold(n_splits=int(cv_splits), shuffle=True, random_state=int(random_state))

# ------------- Layout -------------
tab_overview, tab_sweep, tab_heatmap, tab_search = st.tabs(["Overview", "1D Sweep", "2D Sweep", "Randomized Search"])

with tab_overview:
    st.subheader("Cross-validated performance")

    colA, colB = st.columns([1, 1])
    with colA:
        with st.spinner("Evaluating CV performance..."):
            scores = cross_val_score(pipe, X, y, cv=cv, scoring=scoring, n_jobs=n_jobs_cv)
        st.metric("Mean CV score", f"{scores.mean():.4f}")
        st.write(f"CV scores (n={len(scores)}):", np.round(scores, 4))

    with colB:
        with st.spinner("Computing cross-validated predictions..."):
            y_pred = cross_val_predict(pipe, X, y, cv=cv, n_jobs=n_jobs_cv)
        fig_cm = plot_confusion_matrix(y, y_pred, labels=sorted(np.unique(y)), title="CV Confusion Matrix")
        st.plotly_chart(fig_cm, use_container_width=True)
        st.text("Classification report:")
        st.text(classification_report(y, y_pred, zero_division=0))

    st.subheader("Feature importances")
    with st.spinner("Fitting model on full data to get feature importances..."):
        fitted_pipe = pipe.fit(X, y)
        model = fitted_pipe.named_steps["model"]
        pre = fitted_pipe.named_steps["preprocess"]
        try:
            feat_names = get_feature_names(pre, num_cols, cat_cols)
            importances = model.feature_importances_
            if len(importances) == len(feat_names) and len(importances) > 0:
                fi_df = pd.DataFrame({"feature": feat_names, "importance": importances})
                fi_df = fi_df.sort_values("importance", ascending=False)
                max_k = max(1, min(50, len(fi_df)))
                default_k = min(15, max_k)
                topk = st.slider("Top k features to show", 1, max_k, default_k)
                fig_fi = px.bar(
                    fi_df.head(topk).sort_values("importance"),
                    x="importance", y="feature", orientation="h",
                    title="Top feature importances"
                )
                st.plotly_chart(fig_fi, use_container_width=True)
            else:
                st.info("Feature names/importances not available or dimension mismatch.")
        except Exception as e:
            st.info(f"Feature importances unavailable: {e}")

    if bootstrap and oob_score:
        try:
            st.subheader("OOB Score")
            st.write(f"Out-of-bag score (fitted above): {model.oob_score_:.4f}")
        except Exception:
            st.info("OOB score not available (ensure bootstrap=True and enough samples).")

with tab_sweep:
    st.subheader("1D Hyperparameter sweep (Score vs. Parameter)")
    st.caption("Pick a hyperparameter and range; we’ll run CV for each value and plot the curve.")

    sweep_param = st.selectbox(
        "Parameter to sweep",
        [
            "n_estimators", "max_depth", "max_features",
            "min_samples_split", "min_samples_leaf",
            "max_leaf_nodes"
        ]
    )

    # Configure ranges
    if sweep_param == "n_estimators":
        lo, hi = st.slider("Range", 10, 1000, (50, 400), step=10)
        values = list(range(lo, hi + 1, 25))
        param_key = "model__n_estimators"
    elif sweep_param == "max_depth":
        lo, hi = st.slider("Range", 1, 100, (2, 40), step=1)
        values = list(range(lo, hi + 1, 2)) + [None]
        param_key = "model__max_depth"
    elif sweep_param == "max_features":
        st.write("If using numeric fractions, we’ll sweep fractions of features.")
        flo, fhi = st.slider("Fraction range", 0.1, 1.0, (0.2, 1.0), step=0.05)
        fractions = list(np.round(np.arange(flo, fhi + 1e-9, 0.1), 2))
        values = ["sqrt", "log2"] + fractions
        param_key = "model__max_features"
    elif sweep_param == "min_samples_split":
        lo, hi = st.slider("Range", 2, 100, (2, 30), step=2)
        values = list(range(lo, hi + 1, 2))
        param_key = "model__min_samples_split"
    elif sweep_param == "min_samples_leaf":
        lo, hi = st.slider("Range", 1, 50, (1, 20), step=1)
        values = list(range(lo, hi + 1, 1))
        param_key = "model__min_samples_leaf"
    elif sweep_param == "max_leaf_nodes":
        lo, hi = st.slider("Range", 10, 5000, (50, 1000), step=50)
        values = [None] + list(range(lo, hi + 1, 100))
        param_key = "model__max_leaf_nodes"
    else:
        values, param_key = [], ""

    if st.button("Run 1D sweep"):
        means, stds = [], []
        with st.spinner("Running sweep..."):
            for v in values:
                params = {param_key: v}
                sc = cross_val_score(pipe.set_params(**params), X, y, cv=cv, scoring=scoring, n_jobs=n_jobs_cv)
                means.append(sc.mean())
                stds.append(sc.std())
        x_disp = values
        fig = line_with_error(x_disp, np.array(means), np.array(stds), x_title=sweep_param, y_title=f"CV {scoring}")
        st.plotly_chart(fig, use_container_width=True)

        df_sweep = pd.DataFrame({sweep_param: x_disp, f"mean_{scoring}": means, "std": stds})
        st.dataframe(df_sweep.sort_values(f"mean_{scoring}", ascending=False), use_container_width=True)

with tab_heatmap:
    st.subheader("2D Hyperparameter sweep (Heatmap)")
    st.caption("Pick two hyperparameters; we’ll grid them and heatmap the CV score.")

    p1 = st.selectbox("Param 1", ["n_estimators", "max_depth", "max_features", "min_samples_split", "min_samples_leaf"])
    p2 = st.selectbox("Param 2", ["max_depth", "n_estimators", "max_features", "min_samples_split", "min_samples_leaf"], index=1)

    # FIXED: sliders return (lo, hi) only; we define step separately.
    def param_to_key_and_values(pname, label):
        if pname == "n_estimators":
            lo, hi = st.slider(f"{label} n_estimators", 10, 1000, (50, 300), step=25)
            step = 25
            vals = list(range(lo, hi + 1, step))
            key = "model__n_estimators"
        elif pname == "max_depth":
            lo, hi = st.slider(f"{label} max_depth", 1, 100, (2, 40), step=2)
            step = 2
            vals = list(range(lo, hi + 1, step)) + [None]
            key = "model__max_depth"
        elif pname == "max_features":
            lo, hi = st.slider(f"{label} max_features fraction", 0.1, 1.0, (0.2, 1.0), step=0.2)
            step = 0.2
            vals = ["sqrt", "log2"] + list(np.round(np.arange(lo, hi + 1e-9, step), 2))
            key = "model__max_features"
        elif pname == "min_samples_split":
            lo, hi = st.slider(f"{label} min_samples_split", 2, 100, (2, 30), step=4)
            step = 4
            vals = list(range(lo, hi + 1, step))
            key = "model__min_samples_split"
        elif pname == "min_samples_leaf":
            lo, hi = st.slider(f"{label} min_samples_leaf", 1, 50, (1, 20), step=2)
            step = 2
            vals = list(range(lo, hi + 1, step))
            key = "model__min_samples_leaf"
        else:
            key, vals = None, []
        return key, vals

    key1, vals1 = param_to_key_and_values(p1, "Param 1")
    key2, vals2 = param_to_key_and_values(p2, "Param 2")

    if st.button("Run 2D sweep"):
        with st.spinner("Computing grid..."):
            Z = np.zeros((len(vals1), len(vals2)))
            for i, v1 in enumerate(vals1):
                for j, v2 in enumerate(vals2):
                    params = {key1: v1, key2: v2}
                    try:
                        sc = cross_val_score(pipe.set_params(**params), X, y, cv=cv, scoring=scoring, n_jobs=n_jobs_cv).mean()
                    except Exception:
                        sc = np.nan
                    Z[i, j] = sc
        fig_hm = heatmap_2d(vals2, vals1, Z, x_name=p2, y_name=p1, title=f"{scoring} heatmap")
        st.plotly_chart(fig_hm, use_container_width=True)
        df_heat = pd.DataFrame(Z, index=vals1, columns=vals2)
        st.dataframe(df_heat, use_container_width=True)

with tab_search:
    st.subheader("Randomized search (tune many hyperparameters fast)")

    n_iter = st.slider("n_iter (trials)", 10, 200, 40, step=10)

    if HAS_SCIPY:
        param_dist = {
            "model__n_estimators": randint(50, 1000),
            "model__criterion": ["gini", "entropy", "log_loss"],
            "model__max_depth": [None] + list(range(2, 60)),
            "model__min_samples_split": randint(2, 50),
            "model__min_samples_leaf": randint(1, 50),
            "model__min_weight_fraction_leaf": uniform(0.0, 0.2),
            "model__max_features": ["sqrt", "log2", 0.3, 0.5, 0.7, 1.0],
            "model__max_leaf_nodes": [None, 200, 500, 1000, 2000],
            "model__min_impurity_decrease": uniform(0.0, 0.02),
            "model__bootstrap": [True, False],
            "model__class_weight": [None, "balanced", "balanced_subsample"],
            "model__ccp_alpha": uniform(0.0, 0.02),
            # Note: Skip max_samples to avoid invalid combos when bootstrap=False
        }
    else:
        # Pure-Python fallback (lists)
        param_dist = {
            "model__n_estimators": list(range(50, 1000, 25)),
            "model__criterion": ["gini", "entropy", "log_loss"],
            "model__max_depth": [None] + list(range(2, 60, 2)),
            "model__min_samples_split": list(range(2, 50, 2)),
            "model__min_samples_leaf": list(range(1, 50, 1)),
            "model__min_weight_fraction_leaf": [0.0, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2],
            "model__max_features": ["sqrt", "log2", 0.3, 0.5, 0.7, 1.0],
            "model__max_leaf_nodes": [None, 200, 500, 1000, 2000],
            "model__min_impurity_decrease": [0.0, 0.001, 0.005, 0.01, 0.02],
            "model__bootstrap": [True, False],
            "model__class_weight": [None, "balanced", "balanced_subsample"],
            "model__ccp_alpha": [0.0, 0.001, 0.005, 0.01, 0.02],
        }

    if st.button("Run randomized search"):
        with st.spinner("Searching..."):
            rs = RandomizedSearchCV(
                estimator=pipe,
                param_distributions=param_dist,
                n_iter=int(n_iter),
                scoring=scoring,
                cv=cv,
                random_state=int(random_state),
                n_jobs=n_jobs_cv,
                verbose=0,
                refit=True,
                return_train_score=False
            )
            rs.fit(X, y)

        st.success(f"Best CV {scoring}: {rs.best_score_:.4f}")
        st.write("Best parameters:")
        st.json(rs.best_params_)

        # Show top results
        res = pd.DataFrame(rs.cv_results_)
        cols = ["mean_test_score", "std_test_score", "rank_test_score"] + [c for c in res.columns if c.startswith("param_")]
        res = res[cols].sort_values("rank_test_score").reset_index(drop=True)
        st.dataframe(res.head(25), use_container_width=True)

        # Confusion matrix for best estimator (CV predictions)
        y_pred_best = cross_val_predict(rs.best_estimator_, X, y, cv=cv, n_jobs=n_jobs_cv)
        st.plotly_chart(
            plot_confusion_matrix(y, y_pred_best, labels=sorted(np.unique(y)), title="Best model CV confusion matrix"),
            use_container_width=True

        )
