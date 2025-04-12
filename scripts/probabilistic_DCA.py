import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
import matplotlib.pyplot as plt
import seaborn as sns

# ---- Local imports from your package ----
from src.probabilistic_dca.my_dca_models.data_processing import (
    data_processing, crossval_loess, rolling_std, sample_sorted_datasets
)
from src.probabilistic_dca.my_dca_models.fitting import (
    fit_model_for_samples_mstart_para, gather_sse_matrix, gather_predictions, forecast_from_params, compute_forecast_stats
)
from src.probabilistic_dca.my_dca_models.utilities import (
    load_all_model_train_results, calc_model_probabilities, compute_marginal_model_probs, rank_models_by_probability, combine_forecasts_across_models
)
from src.probabilistic_dca.my_dca_models.plotting import (
    generate_lof_plot, plot_model_predictions, plot_hindcast, plot_future_forecast, plot_post_prob_models, boxplot_eur
)
from src.probabilistic_dca.my_dca_models.arps_model import ArpsModel       # Arps: piecewise hyperbolic->exponential
from src.probabilistic_dca.my_dca_models.sem_model import SEMModel         # SEM: Stretched Exponential Model
from src.probabilistic_dca.my_dca_models.crm_model import CRMModel         # CRM: Capacitance-Resistance Model
from src.probabilistic_dca.my_dca_models.lgm_model import LGMModel         # LGM: Logistic Growth Model

# -------------------------------------------------------------------
# 1) GLOBAL PARAMETERS
# -------------------------------------------------------------------
study_well     = "AF-6(h)"
model_names_all = ["arps", "sem", "crm", "lgm"] # Define available models
model_classes_all = [ArpsModel, SEMModel, CRMModel, LGMModel]  # in parallel with model_names
n_samples      = 1000   # Monte Carlo samples
kfolds         = 10     # cross-validation folds
train_pct      = 0.8    # fraction of data used for training

# -------------------------------------------------------------------
# 2) DATA LOADING
# -------------------------------------------------------------------
# Read the data for the selected well
data_wells_final_df = pd.read_csv("src/probabilistic_dca/data/wells_final_Q12020.csv")
data_production_final_df = pd.read_csv("src/probabilistic_dca/data/AF-6h_daily_prod.csv")

# For demonstration, we specify columns for time, rate, cumulative
time_column      = 'cum_eff_prod_day'
rate_column      = 'oil_month_bpd'
cum_prod_column  = 'cum_oil_bbl'

# Filter well data
well_df = data_production_final_df[[time_column, rate_column, cum_prod_column]]
# (Optionally filter by well_name if you had multiple wells in data_production_final_df)
# e.g.: well_tbl = data_wells_final_df.query("well_name == @study_well")

# ---------------------------
# 2.1) LOF on raw data to detect outliers
# ---------------------------
# Use rate column from the original raw data (well_df)
X_lof = well_df[[rate_column]].values  # Shape: (n_samples, 1)

# Initialize and fit LOF model
lof = LocalOutlierFactor(n_neighbors=16, contamination=0.05)
lof_labels = lof.fit_predict(X_lof)  # 1 = inlier, -1 = outlier

# Attach LOF results to the full dataset (for plotting later)
data_tbl_full = pd.DataFrame({
    'x': well_df[time_column],
    'y': well_df[rate_column],
    'z': well_df[cum_prod_column],
    'lof_flag': lof_labels
})

# Filter to keep only inliers
data_tbl = data_tbl_full[data_tbl_full['lof_flag'] == 1].drop(columns='lof_flag').reset_index(drop=True)

# ---------------------------
# 2.2) PLOT PRODUCTION CURVE + OUTLIERS
# ---------------------------

fig = generate_lof_plot(
    data_tbl_full=data_tbl_full,
    time_column=data_tbl_full['x'], 
    rate_column=data_tbl_full['y'],
    )

plt.show()

# -------------------------------------------------------------------
# 3) DATA PREPROCESSING
# -------------------------------------------------------------------
# 3a) Identify maximum production, cut data to decline portion, do LOWESS plot
last_day, last_day_cum_oil, x_train_i, models_df = data_processing(
    prod_df   = data_tbl,
    train_pct = train_pct,
    frac_value= 0.4,         # fraction for LOESS smoothing
    plotting  = True         # show the initial data + smoothed curve
)

# 3b) Cross-validation to find best half-window for LOESS residual smoothing
best_half_window, best_span, models_df = crossval_loess(
    dataframe = models_df,
    k_folds   = kfolds,
    plotting  = True
)
print(f"[Crossval LOESS] best_half_window={best_half_window}, best_span={best_span}")

# 3c) Rolling standard deviation for measurement-error modeling
models_df = rolling_std(
    dataframe     = models_df,
    half_window   = best_half_window,
    plotting      = True
)

# 3d) Monte Carlo sampling: create multiple “sample_i” columns
sample_sorted_df, sample_stats_df = sample_sorted_datasets(
    models_df, n_samples=n_samples, plotting=True
)

# Save Monte Carlo samples to CSV
sample_sorted_df.to_csv("src/probabilistic_dca/data/model_fit_results/sample_sorted_df.csv", index=False)
print("Sampled production data saved to CSV.")

# Load the saved samples file
sample_sorted_df = pd.read_csv("src/probabilistic_dca/data/model_fit_results/sample_sorted_df.csv")

# -------------------------------------------------------------------
# 4) SPLIT TRAIN / TEST
# -------------------------------------------------------------------
# x_train_i indicates how many rows from max-prod onward for training
train_cutoff     = x_train_i

train_df = sample_sorted_df.iloc[:train_cutoff+1].copy()  # from row=0 up to train_cutoff
test_df  = sample_sorted_df.iloc[train_cutoff+1:].copy()

# -------------------------------------------------------------------
# 5) TRAIN MODELS ON TRAIN_DF
# -------------------------------------------------------------------
# Here we fit each of our 4 decline-curve models (Arps, SEM, CRM, LGM)
# to all the sample_i columns in the TRAIN_DF. The result for each model is
# a dict with 'params', 'sse', 'predictions'.

# Define the training data
t_train = train_df["x"].values
q_train = train_df["y"].values

# Choose models to fit: "all", or specify one like "arps", "sem", "crm", or "lgm"
model_selection = "sem"  # Change to one of the model names to run only that model

# Filter based on selection
if model_selection == "all":
    model_names = model_names_all
    model_classes = model_classes_all
else:
    if model_selection not in model_names_all:
        raise ValueError(f"Invalid model_selection: {model_selection}")
    idx = model_names_all.index(model_selection)
    model_names = [model_names_all[idx]]
    model_classes = [model_classes_all[idx]]

model_train_results = {}

# Loop over each model (name, class)
for m_index, (m_name, m_class) in enumerate(zip(model_names, model_classes)):
    print(f"\n--- Fitting {m_name.upper()} Model to Training Data ---")

    base_path = f"src/probabilistic_dca/data/model_fit_results/{m_name}"
    param_path = f"{base_path}/{m_name}_parameters.csv"
    sse_path   = f"{base_path}/{m_name}_sse.csv"
    pred_path  = f"{base_path}/{m_name}_predictions.csv"

    print(f"Fitting {m_name.upper()} Model to Training Data...")
    
    # Step 1: Print one sample's initial parameters (for inspection only)
    test_model = m_class()
    test_model._initial_guess = test_model.initialize_parameters(
        num_trials=5,
        t_data=t_train,
        q_data=q_train,
        seed=123  # consistent randomization
    )
    print(f"{m_name.upper()} initial guess:", test_model._initial_guess)

    # Step 2: Fit model across all samples
    
    fit_results = fit_model_for_samples_mstart_para(
        model_class=m_class,
        sample_df=train_df,
        seed=123,
        n_inits=10,
        num_trials=5,
        use_shared_p50_init=False,
        n_jobs=-1
    )

    # Save in dictionary
    model_train_results[m_name] = fit_results

    # Extract results
    param_df = fit_results["params"]
    sse_arr  = fit_results["sse"]
    preds    = fit_results["predictions"]

    # Create model-specific paths
    base_path = f"src/probabilistic_dca/data/model_fit_results/{m_name}"
    param_path = f"{base_path}/{m_name}_parameters.csv"
    sse_path   = f"{base_path}/{m_name}_sse.csv"
    pred_path  = f"{base_path}/{m_name}_predictions.csv"

    # Save parameters
    param_df.to_csv(param_path)

    # Save SSE
    pd.DataFrame({"sample": param_df.index, "sse": sse_arr}).to_csv(sse_path, index=False)

    # Save predictions
    pred_df = pd.DataFrame(preds)
    pred_df.insert(0, "x", train_df["x"].values)
    pred_df.to_csv(pred_path, index=False)

# Load results, if needed
#
model_names = ["arps", "sem", "crm", "lgm"]
model_classes = [ArpsModel, SEMModel, CRMModel, LGMModel]

model_train_results=load_all_model_train_results(model_names, base_dir="src/probabilistic_dca/data/model_fit_results")
#
# -------------------------------------------------------------------
# 6) ANALYZE TRAINING FITS
# -------------------------------------------------------------------
# For each model, gather predictions into [time_i, sample_j] matrix, compute p10/p50/p90

for m_name in model_names:
    print(f"\n--- Analyzing {m_name.upper()} fit on TRAIN set ---")
    fit_results = model_train_results[m_name]

    # 6a) Gather predicted rates into shape (Ntrain, Msamples)
    pred_matrix_train = gather_predictions(fit_results, train_df)

    # 6b) Compute p10/mean/p90 at each time step
    train_stats = compute_forecast_stats(pred_matrix_train)

    # 6c) Plot the training P10/Mean/P90 vs actual data
    plot_model_predictions(train_df, train_stats)

    # 6d) Print median (p50) of the best-fit parameters across all samples
    df_params = fit_results["params"]
    p50_params = df_params.median()
    print(f"{m_name.upper()} - P50 Best-Fit Parameters:\n{p50_params}\n")

# -------------------------------------------------------------------
# 7) MODEL PROBABILITIES
# -------------------------------------------------------------------
# Use SSE arrays from each model to build a "posterior" for each model
# (Bayesian weighting). Then we rank them by their marginal probability.

model_dict_for_probs = {}
for m_name in model_names:
    m_sse       = model_train_results[m_name]["sse"]
    m_forecasts = model_train_results[m_name]["predictions"]
    model_dict_for_probs[m_name] = {"sse": m_sse, "forecast": m_forecasts}

sse_matrix = gather_sse_matrix(model_dict_for_probs, model_names)  # shape (M, N) => (4, #samples)
prob_matrix = calc_model_probabilities(sse_matrix)                # shape (M, N)

marginal_probs = compute_marginal_model_probs(prob_matrix, model_names)
ranked_models  = rank_models_by_probability(marginal_probs)

print("\n--- Marginal Posterior Probability of Each Model ---")
for name, pval in ranked_models:
    print(f"{name}: {pval:.3f}")

# Plot Model Probability
plot_post_prob_models(ranked_models)

# -------------------------------------------------------------------
# 8) HINDCAST (TEST PHASE) - Evaluate how each model extrapolates
# -------------------------------------------------------------------
t_test = test_df["x"].values
for m_name, m_class in zip(model_names, model_classes):
    print(f"\n--- Hindcasting {m_name.upper()} Model on TEST set ---")
    # Each sample_j had best-fit parameters. We'll forecast on t_test for each sample.
    fit_res  = model_train_results[m_name]
    param_df = fit_res["params"]

    # Forecast
    test_fc_matrix = forecast_from_params(m_class, param_df, t_test)
    # Stats
    test_stats = compute_forecast_stats(test_fc_matrix)
    # Plot
    plot_hindcast(test_df, test_stats)

# -------------------------------------------------------------------
# 9) FUTURE FORECAST (e.g. 15 yrs or 30 yrs)
# -------------------------------------------------------------------
# Example: 30-year forecast from last_day+1 to last_day+10950
days_future = 5400  # 15 yrs in daily increments, or 10950 for 30 yrs
t_future = np.arange(last_day + 1, last_day + 1 + days_future)

# We store each model’s future forecast in a dictionary, so we can combine them
# Initialize storage
future_forecasts = {}
model_eur_stats = {}

for m_name, m_class in zip(model_names, model_classes):
    print(f"\n--- Long-Term Forecast: {m_name.upper()} ---")
    fit_res    = model_train_results[m_name]
    param_df   = fit_res["params"]

    # 9a) Generate forecast (shape: (T_future, N_samples))
    fc_matrix  = forecast_from_params(m_class, param_df, t_future)

    # 9b) Basic stats + plot
    fc_stats   = compute_forecast_stats(fc_matrix)
    plot_future_forecast(t_future, fc_stats)

    # 9c) Integrate (cumulative), get stats
    dt = 1.0
    cum_matrix   = np.cumsum(fc_matrix, axis=0) * dt
    cum_dist     = cum_matrix[-1, :]  # Total future bbl per sample

    # CUMULATIVE production (future only)
    p10_cum  = np.nanpercentile(cum_dist, 10)
    p25_cum  = np.nanpercentile(cum_dist, 25)
    p50_cum  = np.nanpercentile(cum_dist, 50)
    mean_cum = np.nanmean(cum_dist)
    p75_cum  = np.nanpercentile(cum_dist, 75)
    p90_cum  = np.nanpercentile(cum_dist, 90)

    # EUR = historical + future
    p10_eur  = last_day_cum_oil + p10_cum
    p25_eur  = last_day_cum_oil + p25_cum
    p50_eur  = last_day_cum_oil + p50_cum
    mean_eur = last_day_cum_oil + mean_cum
    p75_eur  = last_day_cum_oil + p75_cum
    p90_eur  = last_day_cum_oil + p90_cum

    print(f"{m_name.upper()} 15-Yr Future CUM P10={p10_cum:.1f}, P50={p50_cum:.1f}, Mean={mean_cum:.1f}, P90={p90_cum:.1f}")
    print(f"{m_name.upper()} 15-Yr EUR  P10={p10_eur:.1f}, P50={p50_eur:.1f}, Mean={mean_eur:.1f}, P90={p90_eur:.1f}")

    # Store forecast matrix for multi-model combination later
    future_forecasts[m_name] = fc_matrix.T  # shape: (N_samples, T_future)

    # Store EUR stats in dictionary
    model_eur_stats[m_name] = {
        "p10":   p10_eur,
        "p25":   p25_eur,
        "p50":   p50_eur,
        "mean":  mean_eur,
        "p75":   p75_eur,
        "p90":   p90_eur
    }

# -------------------------------------------------------------------
# 10) MULTI-MODEL (COMBINED) PROBABILISTIC FORECAST
# -------------------------------------------------------------------

# Stack future forecasts into a tensor of shape (M, N, T)
model_future_arrays = [future_forecasts[m_name] for m_name in model_names]
future_forecast_tensor = np.stack(model_future_arrays, axis=0)

# Combine across models using the posterior probabilities
combined_future_forecast = combine_forecasts_across_models(future_forecast_tensor, prob_matrix)
T_future = combined_future_forecast.shape[1]

# Compute forecast distribution stats at each timestep
multi_future_p10  = np.nanpercentile(combined_future_forecast, 10, axis=0)
multi_future_p25  = np.nanpercentile(combined_future_forecast, 25, axis=0)
multi_future_p50  = np.nanpercentile(combined_future_forecast, 50, axis=0)
multi_future_p75  = np.nanpercentile(combined_future_forecast, 75, axis=0)
multi_future_p90  = np.nanpercentile(combined_future_forecast, 90, axis=0)
multi_future_mean = np.nanmean(combined_future_forecast, axis=0)

# Compute cumulative production per sample
cum_combined = np.cumsum(combined_future_forecast, axis=1)  # shape (N_samples, T_future)
final_cum_samples = cum_combined[:, -1]
final_cum_samples.shape

# Combined EUR statistics (includes historical production)
combined_stats = {
    "p10":   last_day_cum_oil + np.nanpercentile(final_cum_samples, 10),
    "p25":   last_day_cum_oil + np.nanpercentile(final_cum_samples, 25),
    "p50":   last_day_cum_oil + np.nanpercentile(final_cum_samples, 50),
    "p75":   last_day_cum_oil + np.nanpercentile(final_cum_samples, 75),
    "p90":   last_day_cum_oil + np.nanpercentile(final_cum_samples, 90),
    "mean":  last_day_cum_oil + np.nanmean(final_cum_samples)
}

df_eur = pd.DataFrame({
    "model_name": ["Arps", "SEM", "CRM", "LGM", "Combined"],
    "y10":   [model_eur_stats['arps']['p10'], model_eur_stats['sem']['p10'], model_eur_stats['crm']['p10'], model_eur_stats['lgm']['p10'], combined_stats["p10"]],
    "y25":   [model_eur_stats['arps']['p25'], model_eur_stats['sem']['p25'], model_eur_stats['crm']['p25'], model_eur_stats['lgm']['p25'], combined_stats["p25"]],
    "y50":   [model_eur_stats['arps']['p50'], model_eur_stats['sem']['p50'], model_eur_stats['crm']['p50'], model_eur_stats['lgm']['p50'], combined_stats["p50"]],
    "y75":   [model_eur_stats['arps']['p75'], model_eur_stats['sem']['p75'], model_eur_stats['crm']['p75'], model_eur_stats['lgm']['p50'], combined_stats["p75"]],
    "y90":   [model_eur_stats['arps']['p90'], model_eur_stats['sem']['p90'], model_eur_stats['crm']['p90'], model_eur_stats['lgm']['p90'], combined_stats["p90"]],
    "ymean": [model_eur_stats['arps']['mean'], model_eur_stats['sem']['mean'], model_eur_stats['crm']['p90'], model_eur_stats['lgm']['mean'],combined_stats["mean"]]
})

# Plot boxplot
boxplot_eur(df_eur)

combined_EUR=combined_stats["p50"]
print(f"The combined (MMP) EUR for the well {study_well} is: {combined_EUR:.1f} bbl")



# df_arps = pd.read_csv('src/probabilistic_dca/data/model_fit_results/arps/arps_sse.csv')
# print(df_arps.describe())
# print(df_arps['sse'].quantile([0.1, 0.2, 0.5, 0.8, 0.9, 0.99]))

# # or plot
# import matplotlib.pyplot as plt
# df_arps['sse'].hist(bins=50)
# plt.show()

# df_sem = pd.read_csv('src/probabilistic_dca/data/model_fit_results/sem/sem_sse.csv')
# print(df_sem.describe())
# print(df_sem['sse'].quantile([0.1, 0.2, 0.5, 0.8, 0.9, 0.99]))

# df_crm = pd.read_csv('src/probabilistic_dca/data/model_fit_results/crm/crm_sse.csv')
# print(df_crm.describe())
# print(df_crm['sse'].quantile([0.1, 0.2, 0.5, 0.8, 0.9, 0.99]))

# df_lgm = pd.read_csv('src/probabilistic_dca/data/model_fit_results/lgm/lgm_sse.csv')
# print(df_lgm.describe())
# print(df_lgm['sse'].quantile([0.1, 0.2, 0.5, 0.8, 0.9, 0.99]))