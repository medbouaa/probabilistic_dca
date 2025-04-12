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

# Sidebar Inputs
st.sidebar.header("Configuration")
#wells_file = st.sidebar.file_uploader("Upload Wells Data CSV", type=["csv"])
prod_file = st.sidebar.file_uploader("Upload Production Data CSV", type=["csv"])
time_col = st.sidebar.text_input("Time Column", value="cum_eff_prod_day")
rate_col = st.sidebar.text_input("Rate Column", value="oil_month_bpd")
cum_col = st.sidebar.text_input("Cumulative Column", value="cum_oil_bbl")

model_options = ["arps", "sem", "crm", "lgm"]
selected_models = st.sidebar.multiselect("Select Models", model_options, default=model_options)

n_samples = st.sidebar.number_input("Monte Carlo Samples", min_value=100, value=1000, step=100)
n_inits = st.sidebar.number_input("Initializations per Sample", min_value=1, value=10)
num_trials = st.sidebar.number_input("Trials for Init Parameters", min_value=1, value=5)
sse_threshold = st.sidebar.number_input("Threshold Error for Model Fitting", min_value=100, value=250)
min_improvement_frac = st.sidebar.number_input("Minimum Improvement for Early Stopping", min_value=0.0, value=0.010)
forecast_years = st.sidebar.number_input("Forecast Horizon (years)", min_value=1, value=15)

# Step 1: Initialize session state
if "pipeline_results" not in st.session_state:
    st.session_state.pipeline_results = {}

# Step 2: Setup progress bar
progress_text = "Pipeline running..."
progress_bar = st.progress(0, text=progress_text)

status_placeholder = st.empty()

# Step 3: Trigger pipeline
run_pipeline = st.sidebar.button("🚀 Run Analysis")

# ✅ Add this cleanly below all sidebar controls
st.sidebar.markdown("---")
if st.sidebar.button("🧹 Clear Cache"):
    st.cache_data.clear()
    st.sidebar.success("Cache cleared! Reloading app...")
    st.experimental_rerun()

# Pipeline Execution
if run_pipeline and prod_file:
    st.subheader("Pipeline Execution")
    logger.info("Pipeline execution started")
    progress_bar.progress(5, text="Loading data...")

    # Step 1: Load data
    prod_df = cached_load_data(prod_file)
    st.success("Data loaded!")
    logger.info("Data loaded successfully")
    progress_bar.progress(15, text="Removing outliers...")

    # Step 2: Remove outliers
    clean_data, lof_plot = remove_outliers(prod_df, time_col, rate_col, cum_col)
    logger.info("Outlier removal completed")
    progress_bar.progress(30, text="Processing data...")

    # Step 3: Process data
    last_day, last_cum, x_train_i, models_df = process_data(clean_data)
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
        status_placeholder=status_placeholder,  # ✅ New!
    )
    # Prepare per-model Train fit plots
    train_fit_plots = analyze_train_fits(train_df, model_results, selected_models)
    # Hindcast test plots
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
    
    progress_bar.progress(95, text="Pipeline completed!")
    
    # Step 10: Generate EUR Boxplot
    fig_eur, df_eur = generate_eur_boxplot(
        model_eur_stats, combined_stats, selected_models
    )
    logger.info("EUR Boxplot completed")

    progress_bar.progress(100, text="Pipeline completed!")

    # Store results
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
    st.stop()  # Optional to avoid double rendering
    logger.error("Pipeline execution failed: missing production data input file")

# Step 4: Visualization section (tabs)   
if "pipeline_results" not in st.session_state or not st.session_state.pipeline_results:
    st.warning("Please run the pipeline to see results.")
    st.stop()
        
st.success("Pipeline results loaded ✅")

# Step 5: Tabs for visualization
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
    st.subheader("Model Fitting & Forecast Results")

    pipeline_results = st.session_state.pipeline_results
    selected_model = st.selectbox("Select model to view results", pipeline_results["selected_models"])
    
    # ✅ Show P50 parameters
    fit_results = pipeline_results["model_results"][selected_model]
    df_params = fit_results["params"]
    
    p50_params = df_params.median().round(4)
    
    st.subheader(f"P50 Best-Fit Parameters for {selected_model.upper()}")
    st.dataframe(p50_params.to_frame(name="P50 Value"))

    # Train fit plot
    if "train_fits" not in pipeline_results or selected_model not in pipeline_results["train_fits"]:
        st.warning("Train fitting plots not yet available.")
    else:
        st.subheader("Training Fit")
        st.pyplot(pipeline_results["train_fits"][selected_model], use_container_width=False)

    # Hindcast test plot
    if "hindcast_plots" not in pipeline_results or selected_model not in pipeline_results["hindcast_plots"]:
        st.warning("Hindcast plots not yet available.")
    else:
        st.subheader("Hindcast Test")
        st.pyplot(pipeline_results["hindcast_plots"][selected_model], use_container_width=False)

    # Future forecast plot
    if "future_forecast_plots" not in pipeline_results or selected_model not in pipeline_results["future_forecast_plots"]:
        st.warning("Future forecast plots not yet available.")
    else:
        st.subheader("Future Forecast - 15 Years")
        st.pyplot(pipeline_results["future_forecast_plots"][selected_model],use_container_width=False)

# Tab 4: Summary & EUR Results
with tabs[3]:
    st.subheader("Marginal Posterior Probabilities")
    st.pyplot(st.session_state.pipeline_results["prob_plot"], use_container_width=False)

    st.subheader("Boxplot of Multi-Model Probabilistic EUR")
    st.pyplot(st.session_state.pipeline_results["eur_plot"], use_container_width=False)

    st.subheader("EUR Statistics per Model")
    eur_stats_df = pd.DataFrame(st.session_state.pipeline_results["model_eur_stats"]).T
    st.dataframe(eur_stats_df)

    st.subheader("Combined EUR Statistics")
    combined_stats_df = pd.DataFrame([st.session_state.pipeline_results["combined_stats"]])
    st.dataframe(combined_stats_df)
    
    if "model_results" in st.session_state.pipeline_results:
        fit_results_df = prepare_fit_results_for_export(st.session_state.pipeline_results["model_results"])
        
        csv = fit_results_df.to_csv(index=False).encode('utf-8')
        
        st.download_button(
            label="📥 Download Fit Results as CSV",
            data=csv,
            file_name='fit_results.csv',
            mime='text/csv'
        )


