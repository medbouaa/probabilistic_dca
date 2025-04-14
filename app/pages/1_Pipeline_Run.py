import streamlit as st
import pandas as pd
import pathlib
import sys

# Fix import paths
sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent.parent / 'src'))

from probabilistic_dca.my_dca_models.processing import load_data, remove_outliers, split_train_test
from probabilistic_dca.my_dca_models.pipeline import (
    process_data, montecarlo_sampling, fit_models, analyze_train_fits, hindcast_test,
    calculate_model_probabilities, future_forecast, multi_model_combination, generate_eur_boxplot, prepare_fit_results_for_export
)
from probabilistic_dca.my_dca_models.plotting import plot_future_forecast, boxplot_eur
from probabilistic_dca.logging_setup import setup_logger

# Logger
logger = setup_logger(__name__)

# ✅ Caching functions
@st.cache_data
def cached_load_data(prod_file):
    return load_data(prod_file)

@st.cache_data
def cached_montecarlo_sampling(models_df, n_samples):
    return montecarlo_sampling(models_df, n_samples=n_samples)

# Page config
st.set_page_config(page_title="Pipeline Run", layout="wide")
st.title("🚀 Pipeline Execution")

# Step 1: Initialize session state
if "pipeline_results" not in st.session_state:
    st.session_state.pipeline_results = {}

# Sidebar Inputs
st.sidebar.header("Configuration")

# Step 1a: File uploader
prod_file = st.sidebar.file_uploader("Upload Production Data CSV", type=["csv"])

# Step 1b: Dynamic column selection — after upload
if prod_file:
    # Read uploaded data
    prod_df = cached_load_data(prod_file)
    columns = prod_df.columns.tolist()

    # Helper function to guess sensible defaults
    def guess_column(columns, keywords):
        for col in columns:
            if any(keyword.lower() in col.lower() for keyword in keywords):
                return col
        return columns[0]  # fallback to first column if no match

    # Guess defaults
    default_time = guess_column(columns, ["date", "time", "day"])
    default_rate = guess_column(columns, ["rate", "production", "oil_rate", "prod"])
    default_cum = guess_column(columns, ["cumulative", "cum", "total", "oil"])

    # Sidebar selectboxes for column mapping
    st.sidebar.markdown("### Column Selection")
    time_col = st.sidebar.selectbox("🕒 Time Column", options=columns, index=columns.index(default_time))
    rate_col = st.sidebar.selectbox("📉 Rate Column", options=columns, index=columns.index(default_rate))
    cum_col = st.sidebar.selectbox("🛢️ Cumulative Column", options=columns, index=columns.index(default_cum))

    # Preview uploaded data
    st.markdown("### Uploaded Data Preview")
    st.dataframe(prod_df.head())

else:
    # If no file, use empty placeholders to avoid errors
    time_col = rate_col = cum_col = None
    prod_df = None

# Step 2: Other Sidebar Configurations

lof_n_neighbors = st.sidebar.number_input("LOF-Neighbors", min_value=1, max_value=30, value=16, step=1)

lof_contamination = st.sidebar.number_input("LOF-Contamination", min_value=0.01, max_value=0.10, value=0.05, step=0.01, format="%.2f")

train_pct_slider = st.sidebar.slider("Train Percentage (%)", min_value=50, max_value=95, value=80, step=5)
train_pcts = train_pct_slider / 100.0  # convert to fraction

model_options = ["arps", "sem", "crm", "lgm"]
selected_models = st.sidebar.multiselect("Select Models", model_options, default=model_options)

n_samples = st.sidebar.number_input("Monte Carlo Samples", min_value=100, value=1000, step=100)
n_inits = st.sidebar.number_input("Initializations per Sample", min_value=1, value=10)
num_trials = st.sidebar.number_input("Trials for Init Parameters", min_value=1, value=5)
sse_threshold = st.sidebar.number_input("Threshold Error for Model Fitting", min_value=100, value=250)
min_improvement_frac = st.sidebar.number_input("Minimum Improvement for Early Stopping", min_value=0.0, value=0.010)
forecast_years = st.sidebar.number_input("Forecast Horizon (years)", min_value=1, value=15)

# Step 3: Sidebar utilities
st.sidebar.markdown("---")
if st.sidebar.button("🧹 Clear Cache"):
    st.cache_data.clear()
    st.sidebar.success("Cache cleared! Reloading app...")
    st.experimental_rerun()

run_pipeline = st.sidebar.button("🚀 Run Analysis")

# Step 4: Progress bar and status
progress_text = "Pipeline running..."
progress_bar = st.progress(0, text=progress_text)
status_placeholder = st.empty()

# Step 5: Trigger pipeline execution
if run_pipeline and prod_file:
    st.subheader("Pipeline Execution")
    logger.info("Pipeline execution started")
    progress_bar.progress(5, text="Loading data...")

    # Step 1: Data is already loaded in prod_df
    st.success("Data loaded!")
    logger.info("Data loaded successfully")
    progress_bar.progress(15, text="Removing outliers...")

    # Step 2: Remove outliers
    clean_data, lof_plot = remove_outliers(prod_df, time_col, rate_col, cum_col, lof_n_neighbors, lof_contamination)
    logger.info("Outlier removal completed")
    progress_bar.progress(30, text="Processing data...")

    # Step 3: Process data
    last_day, last_cum, x_train_i, models_df = process_data(clean_data, train_pcts)
    logger.info("Data processing completed")
    progress_bar.progress(45, text="Monte Carlo sampling...")

    # Step 4: Monte Carlo sampling
    sample_df, sample_stats_df, sample_fig = cached_montecarlo_sampling(models_df, n_samples)
    logger.info(f"Monte Carlo sampling completed with {n_samples} samples")
    progress_bar.progress(55, text="Splitting train/test...")

    # Step 5: Split train/test
    train_df, test_df = split_train_test(sample_df, x_train_i)
    logger.info("Data split into train and test sets")
    progress_bar.progress(60, text="Fitting models...")

    # Step 6: Fit models
    model_results = fit_models(
        train_df, selected_models,
        n_inits=n_inits, num_trials=num_trials,
        sse_threshold=sse_threshold, min_improvement_frac=min_improvement_frac,
        status_placeholder=status_placeholder,
    )
    train_fit_plots = analyze_train_fits(train_df, model_results, selected_models)
    hindcast_plots = hindcast_test(test_df, model_results, selected_models)
    logger.info("Model fitting completed")
    progress_bar.progress(75, text="Calculating model probabilities...")

    # Step 7: Model probabilities
    prob_matrix, ranked_models, prob_plot = calculate_model_probabilities(model_results, selected_models)
    logger.info("Model probability calculation completed")
    progress_bar.progress(80, text="Forecasting future production...")

    # Step 8: Future forecast
    forecast_days = forecast_years * 360
    future_forecasts, model_eur_stats, future_forecast_plots = future_forecast(
        last_day, last_cum, model_results, selected_models, forecast_days
    )
    logger.info("Future forecast completed")
    progress_bar.progress(90, text="Combining multi-model forecast...")

    # Step 9: Multi-model combination
    combined_forecast, combined_stats = multi_model_combination(
        future_forecasts, prob_matrix, last_cum, selected_models
    )
    logger.info("Multi-model forecast combination completed")
    progress_bar.progress(95, text="Generating EUR Boxplot...")

    # Step 10: Generate EUR Boxplot
    fig_eur, df_eur = generate_eur_boxplot(model_eur_stats, combined_stats, selected_models)
    logger.info("EUR Boxplot completed")

    progress_bar.progress(100, text="Pipeline completed!")

    # Step 11: Store results
    st.session_state.pipeline_results = {
        "clean_data": clean_data,
        "lof_plot": lof_plot,
        "sample_fig": sample_fig,
        "sample_stats_df": sample_stats_df,
        "model_results": model_results,
        "train_fits": train_fit_plots,
        "hindcast_plots": hindcast_plots,
        "prob_matrix": prob_matrix,
        "ranked_models": ranked_models,
        "prob_plot": prob_plot,
        "future_forecasts": future_forecasts,
        "model_eur_stats": model_eur_stats,
        "future_forecast_plots": future_forecast_plots,
        "combined_forecast": combined_forecast,
        "combined_stats": combined_stats,
        "selected_models": selected_models,
        "last_cum": last_cum,
        "eur_plot": fig_eur,
        "eur_df": df_eur,
    }

    st.success("Pipeline completed!")
    logger.info("Pipeline execution completed successfully")

elif run_pipeline and not prod_file:
    st.error("Please upload the Production Data CSV file.")
    st.stop()
    logger.error("Pipeline execution failed: missing production data input file")

# Step 6: If no pipeline results, stop
if "pipeline_results" not in st.session_state or not st.session_state.pipeline_results:
    st.warning("Please run the pipeline to see results.")
    st.stop()

st.success("Pipeline results loaded ✅")

# Step 7: Tabs for visualization
tabs = st.tabs(["Data Cleaning & QC", "Monte Carlo Sampling", "Model Fitting & Results", "Summary & EUR Results"])

# Tab 1: Data Cleaning & QC
with tabs[0]:
    st.subheader("Outlier Removal - LOF Plot")
    st.pyplot(st.session_state.pipeline_results["lof_plot"], use_container_width=False)

# Tab 2: Monte Carlo Sampling
with tabs[1]:
    st.subheader("Monte Carlo Sampling")
    st.pyplot(st.session_state.pipeline_results["sample_fig"], use_container_width=False)

# Tab 3: Model Fitting & Results
with tabs[2]:
    pipeline_results = st.session_state.pipeline_results
    st.subheader("Model Fitting & Forecast Results")
    selected_model = st.selectbox("Select model to view results", pipeline_results["selected_models"])

    fit_results = pipeline_results["model_results"][selected_model]
    df_params = fit_results["params"]
    p50_params = df_params.median().round(4)

    st.subheader(f"P50 Best-Fit Parameters for {selected_model.upper()}")
    st.dataframe(p50_params.to_frame(name="P50 Value"))

    if "train_fits" in pipeline_results and selected_model in pipeline_results["train_fits"]:
        st.subheader("Training Fit")
        st.pyplot(pipeline_results["train_fits"][selected_model], use_container_width=False)

    if "hindcast_plots" in pipeline_results and selected_model in pipeline_results["hindcast_plots"]:
        st.subheader("Hindcast Test")
        st.pyplot(pipeline_results["hindcast_plots"][selected_model], use_container_width=False)

    if "future_forecast_plots" in pipeline_results and selected_model in pipeline_results["future_forecast_plots"]:
        st.subheader("Future Forecast - 15 Years")
        st.pyplot(pipeline_results["future_forecast_plots"][selected_model], use_container_width=False)

# Tab 4: Summary & EUR Results
with tabs[3]:
    st.subheader("Marginal Posterior Probabilities")
    st.pyplot(st.session_state.pipeline_results["prob_plot"], use_container_width=False)

    st.subheader("Boxplot of Multi-Model Probabilistic EUR")
    st.pyplot(st.session_state.pipeline_results["eur_plot"], use_container_width=False)

    st.subheader("EUR Statistics per Model")
    eur_stats_df = pd.DataFrame(st.session_state.pipeline_results["model_eur_stats"]).T
    desired_columns = ["p10", "p50", "mean", "p90"]
    eur_stats_df_clean = eur_stats_df[desired_columns].applymap(lambda x: f"{int(round(x, 0)):,}")
    st.dataframe(eur_stats_df_clean)

    st.subheader("Combined EUR Statistics")
    combined_stats_df = pd.DataFrame([st.session_state.pipeline_results["combined_stats"]])
    combined_stats_df_clean = combined_stats_df[desired_columns].applymap(lambda x: f"{int(round(x, 0)):,}")
    st.dataframe(combined_stats_df_clean)

    if "model_results" in st.session_state.pipeline_results:
        fit_results_df = prepare_fit_results_for_export(st.session_state.pipeline_results["model_results"])
        csv = fit_results_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Fit Results as CSV", data=csv, file_name='fit_results.csv', mime='text/csv')
