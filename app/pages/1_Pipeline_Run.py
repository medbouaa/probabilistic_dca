import streamlit as st
import pandas as pd
import pathlib
import sys
import os
import re
import warnings
import pickle

# Fix import paths
sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent.parent / 'src'))

from probabilistic_dca.my_dca_models.processing import (
    load_data, remove_outliers, split_train_test
)
from probabilistic_dca.my_dca_models.pipeline import (
    process_data, montecarlo_sampling, fit_models, analyze_train_fits,
    hindcast_test, calculate_model_probabilities, future_forecast,
    multi_model_combination, generate_eur_boxplot, prepare_fit_results_for_export
)
from probabilistic_dca.logging_setup import setup_logger
from probabilistic_dca.config import MODEL_PARAM_NAMES

# Suppress the Streamlit “missing ScriptRunContext” warnings
warnings.filterwarnings(
    "ignore",
    message="Thread 'MainThread': missing ScriptRunContext.*",
    module="streamlit.runtime.scriptrunner_utils.script_run_context"
)

# Persistence path for reload
SAVE_PATH = "data/last_pipeline_run.pkl"

# Logger
logger = setup_logger(__name__)

# Page config
st.set_page_config(page_title="Pipeline Run", layout="wide")
st.title("🚀 Probabilistic DCA — Pipeline Execution")

# --- Session state init (with reload) ---
if "pipeline_results" not in st.session_state:
    if os.path.exists(SAVE_PATH):
        with open(SAVE_PATH, "rb") as f:
            st.session_state.pipeline_results = pickle.load(f)
        st.session_state.pipeline_stage = 4
    else:
        st.session_state.pipeline_results = {}
        st.session_state.pipeline_stage = 0

# --- Sidebar: Reload previous run ---
st.sidebar.markdown("---")
if st.sidebar.button("🔄 Reload previous run"):
    if os.path.exists(SAVE_PATH):
        with open(SAVE_PATH, "rb") as f:
            st.session_state.pipeline_results = pickle.load(f)
        st.session_state.pipeline_stage = 4
        st.sidebar.success("✅ Reloaded previous run!")

        # trigger a rerun in whichever API is available
        try:
            st.experimental_rerun()
        except AttributeError:
            try:
                st.script_request_rerun()
            except AttributeError:
                # no rerun API available: user can manually refresh
                pass
    else:
        st.sidebar.error("No saved run found.")

# --- Sidebar: Global configuration ---
st.sidebar.header("Global Configuration")

model_options = ["arps", "sem", "crm", "lgm"]
selected_models = st.sidebar.multiselect(
    "Select Models", model_options, default=model_options,
    help="Choose which decline curve models to include in the analysis."
)

forecast_years = st.sidebar.number_input(
    "Forecast Horizon (years)", min_value=1, value=15,
    help="Years to project future production for forecasting."
)

n_samples = st.sidebar.number_input(
    "Monte Carlo Samples", min_value=100, value=1000, step=100,
    help="Number of synthetic samples generated to model uncertainty."
)

n_inits = st.sidebar.number_input(
    "Initializations per Sample", min_value=1, value=10,
    help="Number of optimization attempts per sample to avoid local minima."
)

num_trials = st.sidebar.number_input(
    "Trials for Init Parameters", min_value=1, value=5,
    help="Number of trials to generate good initial guesses per fit attempt."
)

sse_threshold = st.sidebar.number_input(
    "SSE Threshold", min_value=100, value=250,
    help="Early stopping threshold for SSE during model fitting."
)

min_improvement_frac = st.sidebar.number_input(
    "Minimum Improvement Fraction", min_value=0.0, value=0.01,
    help="Minimum relative SSE improvement between fitting attempts. Below this, stop early."
)

train_pct_slider = st.sidebar.slider(
    "Train Percentage (%)", min_value=50, max_value=95, value=80, step=5,
    help="Percentage of data to use for model training (rest is used for hindcast testing)."
)
train_pcts = train_pct_slider / 100.0

# Outlier detection config
st.sidebar.header("Outlier Detection (LOF)")
lof_n_neighbors = st.sidebar.number_input(
    "LOF n_neighbors", min_value=5, value=20, step=1,
    help="Number of neighbors used for Local Outlier Factor (LOF) outlier detection."
)

lof_contamination = st.sidebar.slider(
    "LOF contamination", min_value=0.0, max_value=0.5, value=0.05, step=0.01,
    help="Expected fraction of data considered outliers."
)

# Parallel jobs config
# figure out how many CPUs *we actually have* under our cgroup
def real_container_cpus():
    """
    Return a tuple (raw_cpuset, count_of_cpus) by inspecting the cgroup cpuset file.
    Falls back to os.cpu_count() if we can’t parse anything.
    """
    paths = [
        "/sys/fs/cgroup/cpuset/cpuset.cpus",   # cgroup v1
        "/sys/fs/cgroup/cpuset.cpus",          # some cgroup v2 mounts
    ]
    for p in paths:
        try:
            s = open(p).read().strip()
            if not s:
                continue
            total = 0
            # e.g. “0-1,3,5-6”
            for part in s.split(","):
                part = part.strip()
                if "-" in part:
                    a, b = part.split("-", 1)
                    total += int(b) - int(a) + 1
                elif part.isdigit():
                    total += 1
                else:
                    # unknown token, skip it
                    continue
            if total > 0:
                return s, total
        except Exception:
            # file not readable or parse error—try next
            continue

    # fallback
    return None, (os.cpu_count() or 1)

# use it in your sidebar:
label, count = real_container_cpus()
if label:
    st.sidebar.write(f"⚙️  Container cpuset: `{label}` → {count} cores")
else:
    st.sidebar.write(f"⚙️  Using host CPU count: {count} cores")

n_jobs = st.sidebar.number_input(
    "Parallel jobs (n_jobs)",
    min_value=1,
    max_value=count,
    value=count,
    help=f"Spawn up to {count} worker processes (based on container cpuset)."
)

# Reset logic
st.sidebar.markdown("---")
if st.sidebar.button("🧹 Reset & Clear Cache"):
    st.cache_data.clear()
    st.session_state.pipeline_stage = 0
    st.session_state.pipeline_results = {}
    if os.path.exists(SAVE_PATH): os.remove(SAVE_PATH)
    st.sidebar.success("Pipeline reset!")
    st.experimental_rerun()

# Tabs for pipeline stages
tabs = st.tabs(["1. Data Cleaning", "2. Monte Carlo Sampling", "3. Model Fitting", "4. Forecast and EUR Summary"])

# ----- STAGE 1: Data Cleaning -----
with tabs[0]:
    st.header("📂 Stage 1: Data Upload & Cleaning")
    prod_file = st.file_uploader("Upload Production Data CSV", type=["csv"])
    if prod_file:
        prod_df = pd.read_csv(prod_file)
        cols = prod_df.columns.tolist()
        def guess_column(columns, keywords):
            for c in columns:
                if any(k.lower() in c.lower() for k in keywords): return c
            return columns[0]
        default_time = guess_column(cols, ["date","time","day"])
        default_rate = guess_column(cols, ["rate","prod","oil"])
        default_cum = guess_column(cols, ["cum","cumulative","total"])
        time_col = st.selectbox("🕒 Time Column", options=cols, index=cols.index(default_time))
        rate_col = st.selectbox("📉 Rate Column", options=cols, index=cols.index(default_rate))
        cum_col  = st.selectbox("🛢️ Cumulative Column", options=cols, index=cols.index(default_cum))
        if st.button("🚀 Run Stage 1: Clean & Prepare Data"):
            st.session_state.pipeline_stage = 1
            # clear downstream keys
            for key in ["sample_df","sample_stats_df","sample_fig","train_df","test_df",
                        "model_results","train_fits","hindcast_plots","prob_matrix",
                        "ranked_models","prob_plot","future_forecasts","model_cum_stats",
                        "model_eur_stats","future_forecast_plots","combined_forecast",
                        "combined_cum_stats","combined_eur_stats","eur_plot","eur_df"]:
                st.session_state.pipeline_results.pop(key, None)
            clean_data, lof_plot = remove_outliers(
                prod_df, time_col, rate_col, cum_col,
                int(lof_n_neighbors), lof_contamination
            )
            last_day, last_cum, x_train_i, models_df = process_data(clean_data, train_pcts)
            st.session_state.pipeline_results.update({
                "prod_df": prod_df,
                "clean_data": clean_data,
                "lof_plot": lof_plot,
                "last_day": last_day,
                "last_cum": last_cum,
                "x_train_i": x_train_i,
                "models_df": models_df,
                "selected_models": selected_models
            })
            st.success("✅ Stage 1 completed! Continue with 'Monte Carlo Sampling'.")
    if st.session_state.pipeline_stage >= 1:
        st.subheader("Outlier Detection (LOF) Plot")
        st.pyplot(st.session_state.pipeline_results["lof_plot"], use_container_width=False)
        st.subheader("Cleaned Data Preview")
        st.dataframe(st.session_state.pipeline_results["clean_data"].head())

# ----- STAGE 2: Monte Carlo Sampling -----
with tabs[1]:
    st.header("🧪 Stage 2: Monte Carlo Sampling")
    if st.session_state.pipeline_stage < 1:
        st.info("Please complete Stage 1 first.")
    else:
        if st.button("🚀 Run Stage 2: Monte Carlo Sampling", disabled=(st.session_state.pipeline_stage < 1)):
            sample_df, sample_stats_df, sample_fig = montecarlo_sampling(
                st.session_state.pipeline_results["models_df"], n_samples=n_samples
            )
            train_df, test_df = split_train_test(sample_df, st.session_state.pipeline_results["x_train_i"])
            st.session_state.pipeline_results.update({
                "sample_df": sample_df,
                "sample_stats_df": sample_stats_df,
                "sample_fig": sample_fig,
                "train_df": train_df,
                "test_df": test_df
            })
            st.session_state.pipeline_stage = 2
            st.success("✅ Stage 2 completed! Continue with 'Model Fitting'.")
    if st.session_state.pipeline_stage >= 2:
        st.subheader("Sample Distribution")
        st.pyplot(st.session_state.pipeline_results["sample_fig"], use_container_width=False)
        st.subheader("Sample Stats")
        st.dataframe(st.session_state.pipeline_results["sample_stats_df"].head())
        st.write("✅ Train DF Shape:", st.session_state.pipeline_results["train_df"].shape)
        st.write("✅ Test DF Shape:", st.session_state.pipeline_results["test_df"].shape)

# ----- STAGE 3: Model Fitting & Forecasting -----
with tabs[2]:
    st.header("⚙️ Stage 3: Fit Models & Forecast")
    if st.session_state.pipeline_stage < 2:
        st.info("Please complete Stage 2 first.")
    else:
        if st.button("🚀 Run Stage 3: Fit & Forecast", disabled=(st.session_state.pipeline_stage < 2)):
            train_df = st.session_state.pipeline_results["train_df"]
            test_df  = st.session_state.pipeline_results["test_df"]
            last_day = st.session_state.pipeline_results["last_day"]
            last_cum = st.session_state.pipeline_results["last_cum"]
            text_slot = st.empty()
            bar_slot  = st.empty()
            model_results = fit_models(
                train_df,
                st.session_state.pipeline_results["selected_models"],
                n_inits=n_inits,
                num_trials=num_trials,
                n_jobs=n_jobs,
                sse_threshold=sse_threshold,
                min_improvement_frac=min_improvement_frac,
                status_placeholder=text_slot,
                progress_bar=bar_slot
            )
            train_fit_plots   = analyze_train_fits(train_df, model_results, st.session_state.pipeline_results["selected_models"])
            hindcast_plots    = hindcast_test(test_df, model_results, st.session_state.pipeline_results["selected_models"])
            prob_matrix, ranked_models, prob_plot = calculate_model_probabilities(
                model_results, st.session_state.pipeline_results["selected_models"]
            )
            forecast_days = forecast_years * 360
            future_forecasts, model_cum_stats, model_eur_stats, future_forecast_plots = future_forecast(
                last_day, last_cum, model_results, st.session_state.pipeline_results["selected_models"], forecast_days
            )
            st.session_state.pipeline_results.update({
                "model_results": model_results,
                "train_fits":   train_fit_plots,
                "hindcast_plots": hindcast_plots,
                "prob_matrix": prob_matrix,
                "ranked_models": ranked_models,
                "prob_plot": prob_plot,
                "future_forecasts": future_forecasts,
                "model_cum_stats": model_cum_stats,
                "model_eur_stats": model_eur_stats,
                "future_forecast_plots": future_forecast_plots
            })
            st.session_state.pipeline_stage = 3
            st.success("✅ Stage 3 completed! Continue with 'Forecast and EUR Summary'.")
    if st.session_state.pipeline_stage >= 3:
        st.subheader("Marginal Posterior Probabilities")
        st.pyplot(st.session_state.pipeline_results["prob_plot"], use_container_width=False)
        selected_model = st.selectbox(
            "Select Model to View Results",
            st.session_state.pipeline_results["selected_models"]
        )
        if selected_model in st.session_state.pipeline_results["model_results"]:
            fit_results = st.session_state.pipeline_results["model_results"][selected_model]
            p50_params = fit_results["params"].median().round(4)
            names = MODEL_PARAM_NAMES.get(selected_model.lower(), list(p50_params.index))
            st.subheader(f"P50 Best-Fit Parameters for {selected_model.upper()}")
            for name, v in zip(names, p50_params.values):
                st.markdown(f"- **{name}**: {v}")
        if selected_model in st.session_state.pipeline_results["train_fits"]:
            st.subheader("Training Fit")
            st.pyplot(st.session_state.pipeline_results["train_fits"][selected_model], use_container_width=False)
        if selected_model in st.session_state.pipeline_results["hindcast_plots"]:
            st.subheader("Hindcast Test")
            st.pyplot(st.session_state.pipeline_results["hindcast_plots"][selected_model], use_container_width=False)
        if selected_model in st.session_state.pipeline_results["future_forecast_plots"]:
            st.subheader("Future Forecast")
            st.pyplot(st.session_state.pipeline_results["future_forecast_plots"][selected_model], use_container_width=False)

# ----- STAGE 4: EUR Calculation & Summary -----
with tabs[3]:
    st.header("📊 Stage 4: EUR Calculation & Summary")
    if st.session_state.pipeline_stage < 3:
        st.info("Please complete Stage 3 first.")
    else:
        if st.button("🚀 Run Stage 4: EUR Calculation", disabled=(st.session_state.pipeline_stage < 3)):
            combined_forecast, combined_cum_stats, combined_eur_stats = multi_model_combination(
                st.session_state.pipeline_results["future_forecasts"],
                st.session_state.pipeline_results["prob_matrix"],
                st.session_state.pipeline_results["last_cum"],
                st.session_state.pipeline_results["selected_models"]
            )
            eur_plot, eur_df = generate_eur_boxplot(
                st.session_state.pipeline_results["model_eur_stats"],
                combined_eur_stats,
                st.session_state.pipeline_results["selected_models"]
            )
            st.session_state.pipeline_results.update({
                "combined_forecast": combined_forecast,
                "combined_cum_stats": combined_cum_stats,
                "combined_eur_stats": combined_eur_stats,
                "eur_plot": eur_plot,
                "eur_df": eur_df
            })
            st.session_state.pipeline_stage = 4
            # Save for reload
            with open(SAVE_PATH, "wb") as f:
                pickle.dump(st.session_state.pipeline_results, f)
            st.success("✅ Stage 4 completed! (Results saved for reload)")
    if st.session_state.pipeline_stage >= 4:
        st.subheader("Cumulative Production Forecast Statistics per Model")
        cum_df = pd.DataFrame(st.session_state.pipeline_results["model_cum_stats"]).T
        cols = ["p10","p50","mean","p90"]
        st.dataframe(cum_df[cols].applymap(lambda x: f"{int(round(x,0)):,}"))
        st.subheader("Combined Cumulative Production Forecast Statistics")
        combined_df = pd.DataFrame([st.session_state.pipeline_results["combined_cum_stats"]])
        st.dataframe(combined_df[cols].applymap(lambda x: f"{int(round(x,0)):,}"))
        st.subheader("EUR Statistics per Model")
        eur_stats_df = pd.DataFrame(st.session_state.pipeline_results["model_eur_stats"]).T
        st.dataframe(eur_stats_df[cols].applymap(lambda x: f"{int(round(x,0)):,}"))
        st.subheader("Combined EUR Statistics")
        combined_eur_df = pd.DataFrame([st.session_state.pipeline_results["combined_eur_stats"]])
        st.dataframe(combined_eur_df[cols].applymap(lambda x: f"{int(round(x,0)):,}"))
        st.subheader("EUR Boxplot (Multi-Model)")
        st.pyplot(st.session_state.pipeline_results["eur_plot"], use_container_width=False)
        fit_results_df = prepare_fit_results_for_export(
            st.session_state.pipeline_results["model_results"]
        )
        csv = fit_results_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 Download Fit Results as CSV",
            data=csv,
            file_name='fit_results.csv',
            mime='text/csv'
        )
