import streamlit as st
import pandas as pd
import pathlib
import sys
import os
import warnings

# Fix import paths
sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent.parent / 'src'))

from probabilistic_dca.my_dca_models.processing import load_data, remove_outliers, split_train_test
from probabilistic_dca.my_dca_models.pipeline import (
    process_data, montecarlo_sampling, fit_models, analyze_train_fits, hindcast_test,
    calculate_model_probabilities, future_forecast, multi_model_combination, generate_eur_boxplot, prepare_fit_results_for_export
)
from probabilistic_dca.logging_setup import setup_logger
from probabilistic_dca.config import MODEL_PARAM_NAMES

# Suppress the Streamlit “missing ScriptRunContext” warnings
warnings.filterwarnings(
    "ignore",
    message="Thread 'MainThread': missing ScriptRunContext.*",
    module="streamlit.runtime.scriptrunner_utils.script_run_context"
)

# Logger
logger = setup_logger(__name__)

# Page config
st.set_page_config(page_title="Pipeline Run", layout="wide")
st.title("🚀 Probabilistic DCA — Pipeline Execution")

# --- Session state init ---
if "pipeline_stage" not in st.session_state:
    st.session_state.pipeline_stage = 0
if "pipeline_results" not in st.session_state:
    st.session_state.pipeline_results = {}

# --- Sidebar: Global configuration ---
st.sidebar.header("Global Configuration")

# Stage progress tracker
# def render_progress(current_stage):
#     st.sidebar.subheader("📈 Pipeline Progress")
#     stages = ["Stage 1: Data Cleaning", "Stage 2: Sampling", "Stage 3: Model Fitting", "Stage 4: EUR"]
#     for i, stage in enumerate(stages, start=1):
#         if current_stage >= i:
#             st.sidebar.markdown(f"✅ **{stage}**")
#         else:
#             st.sidebar.markdown(f"🚧 {stage}")

# render_progress(st.session_state.pipeline_stage)

# --- Config parameters ---
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

# --- Outlier Detection (LOF) ---
st.sidebar.header("Outlier Detection (LOF)")

lof_n_neighbors = st.sidebar.number_input(
    "LOF n_neighbors", min_value=5, value=20, step=1,
    help="Number of neighbors used for Local Outlier Factor (LOF) outlier detection."
)

lof_contamination = st.sidebar.slider(
    "LOF contamination", min_value=0.0, max_value=0.5, value=0.05, step=0.01,
    help="Expected fraction of data considered outliers."
)

# --- Total Cores Count ---
st.sidebar.header("Total Cores Count")
total_cores = os.cpu_count() or 1
n_jobs = st.sidebar.number_input(
    "Parallel jobs",
    min_value=1,
    max_value=total_cores,
    value=total_cores,
    help="How many worker processes to spawn"
)

# --- Reset logic ---
st.sidebar.markdown("---")
if st.sidebar.button("🧹 Reset & Clear Cache"):
    st.cache_data.clear()
    st.session_state.pipeline_stage = 0
    st.session_state.pipeline_results = {}
    st.sidebar.success("Pipeline reset!")
    st.experimental_rerun()

# --- Tab layout for stages ---
tabs = st.tabs(["1. Data Cleaning", "2. Monte Carlo Sampling", "3. Model Fitting", "4. Forecast and EUR Summary"])

# ============================
# STAGE 1: Data Cleaning
# ============================
with tabs[0]:
    st.header("📂 Stage 1: Data Upload & Cleaning")

    prod_file = st.file_uploader("Upload Production Data CSV", type=["csv"])

    if prod_file:
        prod_df = pd.read_csv(prod_file)
        columns = prod_df.columns.tolist()

        # Smart defaults
        def guess_column(columns, keywords):
            for col in columns:
                if any(k.lower() in col.lower() for k in keywords):
                    return col
            return columns[0]

        default_time = guess_column(columns, ["date", "time", "day"])
        default_rate = guess_column(columns, ["rate", "prod", "oil"])
        default_cum = guess_column(columns, ["cum", "cumulative", "total"])

        time_col = st.selectbox("🕒 Time Column", options=columns, index=columns.index(default_time))
        rate_col = st.selectbox("📉 Rate Column", options=columns, index=columns.index(default_rate))
        cum_col = st.selectbox("🛢️ Cumulative Column", options=columns, index=columns.index(default_cum))

        stage1_button = st.button("🚀 Run Stage 1: Clean & Prepare Data")

        if stage1_button:
            st.session_state.pipeline_stage = 1
            keys_to_clear = [
                "sample_df", "sample_stats_df", "sample_fig", "train_df", "test_df",
                "model_results", "train_fits", "hindcast_plots", "prob_matrix",
                "ranked_models", "prob_plot", "future_forecasts", "model_cum_stats", "model_eur_stats",
                "future_forecast_plots", "combined_forecast", "combined_cum_stats",
                "combined_eur_stats",
                "eur_plot", "eur_df"
            ]
            for key in keys_to_clear:
                st.session_state.pipeline_results.pop(key, None)

            clean_data, lof_plot = remove_outliers(
                prod_df, time_col, rate_col, cum_col,
                int(lof_n_neighbors), lof_contamination,
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
                "selected_models": selected_models,
            })
            st.success("✅ Stage 1 completed!, Continue with 'Monte Carlo Sampling'")

    if st.session_state.pipeline_stage >= 1:
        st.subheader("Outlier Detection (LOF) Plot")
        st.pyplot(st.session_state.pipeline_results["lof_plot"], use_container_width=False)
        st.subheader("Cleaned Data Preview")
        st.dataframe(st.session_state.pipeline_results["clean_data"].head())

# ============================
# STAGE 2: Sampling
# ============================
with tabs[1]:
    st.header("🧪 Stage 2: Monte Carlo Sampling")

    if st.session_state.pipeline_stage < 1:
        st.info("Please complete Stage 1 first.")
    else:
        stage2_button = st.button("🚀 Run Stage 2: Monte Carlo Sampling", disabled=(st.session_state.pipeline_stage < 1))

        if stage2_button:
            sample_df, sample_stats_df, sample_fig = montecarlo_sampling(
                st.session_state.pipeline_results["models_df"], n_samples=n_samples
            )
            train_df, test_df = split_train_test(sample_df, st.session_state.pipeline_results["x_train_i"])

            st.session_state.pipeline_results.update({
                "sample_df": sample_df,
                "sample_stats_df": sample_stats_df,
                "sample_fig": sample_fig,
                "train_df": train_df,
                "test_df": test_df,
            })
            st.session_state.pipeline_stage = 2
            st.success("✅ Stage 2 completed! Continue with 'Model Fitting'")

    if st.session_state.pipeline_stage >= 2:
        st.subheader("Sample Distribution")
        st.pyplot(st.session_state.pipeline_results["sample_fig"], use_container_width=False)
        st.subheader("Sample Stats")
        st.dataframe(st.session_state.pipeline_results["sample_stats_df"].head())
        st.write("✅ Train DF Shape:", st.session_state.pipeline_results["train_df"].shape)
        st.write("✅ Test DF Shape:", st.session_state.pipeline_results["test_df"].shape)

# ============================
# STAGE 3: Model Fitting & Forecasting
# ============================
with tabs[2]:
    st.header("⚙️ Stage 3: Fit Models & Forecast")

    if st.session_state.pipeline_stage < 2:
        st.info("Please complete Stage 2 first.")
    else:
        stage3_button = st.button("🚀 Run Stage 3: Fit & Forecast", disabled=(st.session_state.pipeline_stage < 2))

        if stage3_button:
            train_df = st.session_state.pipeline_results["train_df"]
            test_df = st.session_state.pipeline_results["test_df"]
            last_day = st.session_state.pipeline_results["last_day"]
            last_cum = st.session_state.pipeline_results["last_cum"]

            # placeholder for all status messages
            status_placeholder = st.empty()
            bar_placeholder    = st.empty()
            
            # inside your `if stage3_button:` block, before calling fit_models(...)
            #n_jobs = -1   # -1 means “use all available CPU cores”
            # if user or default gave -1, use every core
            #effective_n_jobs = os.cpu_count() if n_jobs < 1 else n_jobs
            
            # call the new fit_models (it will create its own progress bar)
            model_results = fit_models(
                train_df,
                st.session_state.pipeline_results["selected_models"],
                n_inits=n_inits,
                num_trials=num_trials,
                n_jobs=n_jobs,             # <-- use the remapped value
                sse_threshold=sse_threshold,
                min_improvement_frac=min_improvement_frac,
                status_placeholder=status_placeholder,  # ✅ Pass it in!
                progress_bar=bar_placeholder,   # NEW
            )
            train_fit_plots = analyze_train_fits(train_df, model_results, st.session_state.pipeline_results["selected_models"])
            hindcast_plots = hindcast_test(test_df, model_results, st.session_state.pipeline_results["selected_models"])
            prob_matrix, ranked_models, prob_plot = calculate_model_probabilities(model_results, st.session_state.pipeline_results["selected_models"])
            forecast_days = forecast_years * 360
            future_forecasts, model_cum_stats, model_eur_stats, future_forecast_plots = future_forecast(
                last_day, last_cum, model_results, st.session_state.pipeline_results["selected_models"], forecast_days
            )

            st.session_state.pipeline_results.update({
                "model_results": model_results,
                "train_fits": train_fit_plots,
                "hindcast_plots": hindcast_plots,
                "prob_matrix": prob_matrix,
                "ranked_models": ranked_models,
                "prob_plot": prob_plot,
                "future_forecasts": future_forecasts,
                "model_cum_stats": model_cum_stats,
                "model_eur_stats": model_eur_stats,
                "future_forecast_plots": future_forecast_plots,
            })
            st.session_state.pipeline_stage = 3
            st.success("✅ Stage 3 completed! Continue with 'Forecast and EUR Summary'")

    if st.session_state.pipeline_stage >= 3:
        st.subheader("Marginal Posterior Probabilities")
        st.pyplot(st.session_state.pipeline_results["prob_plot"], use_container_width=False)

        selected_model = st.selectbox("Select Model to View Results", st.session_state.pipeline_results["selected_models"])

        # ✅ P50 parameters as text
        if selected_model in st.session_state.pipeline_results["model_results"]:
            fit_results = st.session_state.pipeline_results["model_results"][selected_model]
            df_params = fit_results["params"]
            p50_params = df_params.median().round(4)

            param_names = MODEL_PARAM_NAMES.get(selected_model.lower(), list(p50_params.index))

            st.subheader(f"P50 Best-Fit Parameters for {selected_model.upper()}")
            for name, value in zip(param_names, p50_params.values):
                st.markdown(f"- **{name}**: {value}")

            # st.subheader(f"P50 Best-Fit Parameters for {selected_model.upper()}")
            # for param, value in p50_params.items():
            #     st.markdown(f"- **{param}**: {value}")

        if selected_model in st.session_state.pipeline_results["train_fits"]:
            st.subheader("Training Fit")
            st.pyplot(st.session_state.pipeline_results["train_fits"][selected_model], use_container_width=False)

        if selected_model in st.session_state.pipeline_results["hindcast_plots"]:
            st.subheader("Hindcast Test")
            st.pyplot(st.session_state.pipeline_results["hindcast_plots"][selected_model], use_container_width=False)

        if selected_model in st.session_state.pipeline_results["future_forecast_plots"]:
            st.subheader("Future Forecast")
            st.pyplot(st.session_state.pipeline_results["future_forecast_plots"][selected_model], use_container_width=False)

# ============================
# STAGE 4: EUR Calculation & Summary
# ============================
with tabs[3]:
    st.header("📊 Stage 4: EUR Calculation & Summary")

    if st.session_state.pipeline_stage < 3:
        st.info("Please complete Stage 3 first.")
    else:
        stage4_button = st.button("🚀 Run Stage 4: EUR Calculation", disabled=(st.session_state.pipeline_stage < 3))

        if stage4_button:
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
                "eur_df": eur_df,
            })
            st.session_state.pipeline_stage = 4
            st.success("✅ Stage 4 completed!")

    if st.session_state.pipeline_stage >= 4:
        st.subheader("Cumulative Production Forecast Statistics per Model")
        cum_stats_df = pd.DataFrame(st.session_state.pipeline_results["model_cum_stats"]).T
        desired_columns = ["p10", "p50", "mean", "p90"]
        cum_stats_df_clean = cum_stats_df[desired_columns].applymap(lambda x: f"{int(round(x, 0)):,}")
        st.dataframe(cum_stats_df_clean)
        
        st.subheader("Combined Cumulative Production Forecast Statistics")
        combined_cum_stats_df = pd.DataFrame([st.session_state.pipeline_results["combined_cum_stats"]])
        combined_cum_stats_df_clean = combined_cum_stats_df[desired_columns].applymap(lambda x: f"{int(round(x, 0)):,}")
        st.dataframe(combined_cum_stats_df_clean)
        
        st.subheader("EUR Statistics per Model")
        eur_stats_df = pd.DataFrame(st.session_state.pipeline_results["model_eur_stats"]).T
        desired_columns = ["p10", "p50", "mean", "p90"]
        eur_stats_df_clean = eur_stats_df[desired_columns].applymap(lambda x: f"{int(round(x, 0)):,}")
        st.dataframe(eur_stats_df_clean)

        st.subheader("Combined EUR Statistics")
        combined_eur_stats_df = pd.DataFrame([st.session_state.pipeline_results["combined_eur_stats"]])
        combined_eur_stats_df_clean = combined_eur_stats_df[desired_columns].applymap(lambda x: f"{int(round(x, 0)):,}")
        st.dataframe(combined_eur_stats_df_clean)
        
        st.subheader("EUR Boxplot (Multi-Model)")
        st.pyplot(st.session_state.pipeline_results["eur_plot"], use_container_width=False)

        fit_results_df = prepare_fit_results_for_export(st.session_state.pipeline_results["model_results"])
        csv = fit_results_df.to_csv(index=False).encode('utf-8')

        st.download_button("📥 Download Fit Results as CSV", data=csv, file_name='fit_results.csv', mime='text/csv')
