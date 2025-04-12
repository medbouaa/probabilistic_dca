import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from my_dca_models.data_processing import *
from my_dca_models.fitting import *

from my_dca_models.arps_model import ArpsModel
from my_dca_models.sem_model import SEMModel
from my_dca_models.crm_model import CRMModel
from my_dca_models.lgm_model import LGMModel

######################################################################
# Global parameters
######################################################################

study_well = "AF-6(h)"
number_of_models = 4
model_names = ["arps", "sem", "crm", "lgm"]
n_samples = 1000  # Number of samples in Monte Carlo simulation
kfolds = 10  # Number of folds for cross-validation
train_pct = 0.8  # Percentage of data used for training

######################################################
# 0) Data Reading
######################################################

#data_production_final_df = pd.read_csv("00_data_wrangled/Updated_Dataset_with_Cumulative_Oil.csv")

data_wells_final_df = pd.read_csv("00_data_wrangled/wells_final_Q12020.csv")
data_production_final_df = pd.read_csv("00_data_wrangled/AF-6h_daily_prod.csv")
#eur_estimated_df = pd.read_csv("Data/eur_estimados.csv")

######################################################
# 1) Data Preprocessing: Crossval, Rolling STD, etc.
#    sampling => sample_sorted_df (with train/test split).
######################################################
# Selecting well data for analysis

# time_column = 'day'
# rate_column = 'oil_rate_bpd'
# cum_prod_colum = 'cum_oil_bbl'

time_column = 'cum_eff_prod_day'
rate_column = 'oil_month_bpd'
cum_prod_colum = 'cum_oil_bbl'

data_tbl = data_production_final_df[[time_column, rate_column, cum_prod_colum]]
well_tbl = data_wells_final_df.query("well_name == @study_well")

data_tbl = pd.DataFrame({'x': data_tbl[time_column], 'y': data_tbl[rate_column], 'z': data_tbl[cum_prod_colum]})

last_day, last_day_cum_oil, x_train_i, models_df = data_processing(prod_df=data_tbl, train_pct=train_pct, frac_value=0.4, plotting=True)

######################################################################
# Optimum Half-Window Width with Cross Validation
######################################################################
# Using cross-validation to determine the best half-window width for smoothing residuals.

best_half_window, best_span, models_df = crossval_loess(
    dataframe=models_df,
    k_folds=kfolds,
    #w_min=3,          
    #dataframe=models_df,
    plotting="True"      
)

# Printing the results
print(f"Best Half-Window: {best_half_window}")
print(f"Best Span/Frac: {best_span}")
models_df.head()

######################################################################
# Rolling Standard Deviation
######################################################################
# Using a moving-window approach to estimate the standard deviation of measurement errors.

# Compute rolling standard deviation and variance
models_df = rolling_std(dataframe=models_df, half_window=best_half_window, plotting=True)

# Printing the results
models_df.head()

######################################################################
# Generating multiple samples for Monte Carlo simulation.
######################################################################
# Sample N Sorted Data Sets from Original Data

# Call the function with the default sample size of 1000
sample_sorted_df, sample_sorted_stats_df = sample_sorted_datasets(models_df, n_samples=n_samples)

# Printing the results
sample_sorted_df.head()
sample_sorted_stats_df.head()

# Saving the generated samples
sample_sorted_df.to_csv("00_data_wrangled/model_fit_results/sample_sorted_df.csv", index=False)

########### CHECK POINT -----------
# Load the saved samples file
sample_sorted_df = pd.read_csv("00_data_wrangled/model_fit_results/sample_sorted_df.csv")

######################################################################
# Training vs Testing datasets
######################################################################
# Row index for your training cutoff
train_cutoff =  x_train_i

# Separate training vs test
train_df = sample_sorted_df.iloc[:train_cutoff+1].copy()  # from row 0 to train_cutoff
test_df  = sample_sorted_df.iloc[train_cutoff+1:].copy()  # from train_cutoff to last_day

t_train = train_df["x"].values
q_train = train_df["y"].values

######################################################
# 2) TRAIN PHASE
######################################################
# Fitting multiple decline-curve models (Arps, SEM, CRM, LGM) to sampled data.

# Fit the Arps model on the training portion
# => "params" (DataFrame), "sse" (array), "predictions" (dict)

# Instantiate models (no initialization yet)
arps_model = ArpsModel()
sem_model = SEMModel()
crm_model = CRMModel()
lgm_model = LGMModel()

# Initialize parameters explicitly
arps_model._initial_guess = arps_model.initialize_parameters(t_data=t_train, q_data=q_train)
sem_model._initial_guess = sem_model.initialize_parameters(t_data=t_train, q_data=q_train)
crm_model._initial_guess = crm_model.initialize_parameters(t_data=t_train, q_data=q_train)
lgm_model._initial_guess = lgm_model.initialize_parameters(t_data=t_train, q_data=q_train)

print("Arps Model Initial Parameters:", arps_model._initial_guess)
print("SEM Model Initial Parameters:", sem_model._initial_guess)
print("CRM Model Initial Parameters:", crm_model._initial_guess)
print("LGM Model Initial Parameters:", lgm_model._initial_guess)

# Fit each model to training samples
arps_train_results = fit_model_for_samples(arps_model, train_df)
sem_train_results = fit_model_for_samples(sem_model, train_df)
crm_train_results = fit_model_for_samples(crm_model, train_df)
lgm_train_results = fit_model_for_samples(lgm_model, train_df)

#########################################
# Storing model_train_results in csv file

arps_param_path = '00_data_wrangled/model_fit_results/arps/arps_parameters.csv'
arps_sse_path = '00_data_wrangled/model_fit_results/arps/arps_sse.csv'
arps_pred_path = '00_data_wrangled/model_fit_results/arps/arps_predictions.csv'

sem_param_path = '00_data_wrangled/model_fit_results/sem/sem_parameters.csv'
sem_sse_path = '00_data_wrangled/model_fit_results/sem/sem_sse.csv'
sem_pred_path = '00_data_wrangled/model_fit_results/sem/sem_predictions.csv'

crm_param_path = '00_data_wrangled/model_fit_results/crm/crm_parameters.csv'
crm_sse_path = '00_data_wrangled/model_fit_results/crm/crm_sse.csv'
crm_pred_path = '00_data_wrangled/model_fit_results/crm/crm_predictions.csv'

lgm_param_path = '00_data_wrangled/model_fit_results/lgm/lgm_parameters.csv'
lgm_sse_path = '00_data_wrangled/model_fit_results/lgm/lgm_sse.csv'
lgm_pred_path = '00_data_wrangled/model_fit_results/lgm/lgm_predictions.csv'

# 1) Parameters
arps_train_results['params'].to_csv(arps_param_path)
sem_train_results['params'].to_csv(sem_param_path)
crm_train_results['params'].to_csv(crm_param_path)
lgm_train_results['params'].to_csv(lgm_param_path)

# 2) SSE
sse_df = pd.DataFrame({
    "sample": arps_train_results["params"].index,
    "sse": arps_train_results["sse"]
})
sse_df.to_csv(arps_sse_path, index=False)

sse_df = pd.DataFrame({
    "sample": sem_train_results["params"].index,
    "sse": sem_train_results["sse"]
})
sse_df.to_csv(sem_sse_path, index=False)

sse_df = pd.DataFrame({
    "sample": crm_train_results["params"].index,
    "sse": crm_train_results["sse"]
})
sse_df.to_csv(crm_sse_path, index=False)

sse_df = pd.DataFrame({
    "sample": lgm_train_results["params"].index,
    "sse": lgm_train_results["sse"]
})
sse_df.to_csv(lgm_sse_path, index=False)

# 3) Predictions
pred_dict = arps_train_results['predictions']
pred_df = pd.DataFrame(pred_dict)
pred_df.insert(0, "x", train_df["x"].values)
pred_df.to_csv(arps_pred_path, index=False)

pred_dict = sem_train_results['predictions']
pred_df = pd.DataFrame(pred_dict)
pred_df.insert(0, "x", train_df["x"].values)
pred_df.to_csv(sem_pred_path, index=False)

pred_dict = crm_train_results['predictions']
pred_df = pd.DataFrame(pred_dict)
pred_df.insert(0, "x", train_df["x"].values)
pred_df.to_csv(crm_pred_path, index=False)

pred_dict = lgm_train_results['predictions']
pred_df = pd.DataFrame(pred_dict)
pred_df.insert(0, "x", train_df["x"].values)
pred_df.to_csv(lgm_pred_path, index=False)

# 4) Reconstruct the original arps_train_results dictionary (csv)
# Load paths first....
arps_train_results = load_train_results(arps_param_path,arps_sse_path,arps_pred_path)

sem_train_results = load_train_results(sem_param_path,sem_sse_path,sem_pred_path)

crm_train_results = load_train_results(crm_param_path,crm_sse_path,crm_pred_path)

lgm_train_results = load_train_results(lgm_param_path,lgm_sse_path,lgm_pred_path)

######################################################################
# Predictions (Training dataset)
######################################################################

# Arps
# 1) Gather train predictions into a 2D array (time steps x samples)
pred_matrix_train = gather_predictions(arps_train_results, train_df) # shape (Ntrain, Msamples)

# 2) Compute stats: p10, mean, p90, etc.
train_stats = compute_forecast_stats(pred_matrix_train)

# 3) Plot training P10/P50/P90
plot_model_predictions(train_df, train_stats)

# Extract parameter DataFrame
df_params = arps_train_results["params"]

# Compute P50 (median) for each parameter
p50_params = df_params.median()

# Display results
print("P50 Best-Fit Arps Parameters:")
print(p50_params)

#----------------  
# SEM
# 1) Gather train predictions into a 2D array (time steps x samples)
pred_matrix_train = gather_predictions(sem_train_results, train_df) # shape (Ntrain, Msamples)

# 2) Compute stats: p10, mean, p90, etc.
train_stats = compute_forecast_stats(pred_matrix_train)

# 3) Plot training P10/P50/P90
plot_model_predictions(train_df, train_stats)

# Extract parameter DataFrame
df_params = sem_train_results["params"]

# Compute P50 (median) for each parameter
p50_params = df_params.median()

# Display results
print("P50 Best-Fit SEM Parameters:")
print(p50_params)

#----------------  
# CRM
# 1) Gather train predictions into a 2D array (time steps x samples)
pred_matrix_train = gather_predictions(crm_train_results, train_df) # shape (Ntrain, Msamples)

# 2) Compute stats: p10, mean, p90, etc.
train_stats = compute_forecast_stats(pred_matrix_train)

# 3) Plot training P10/P50/P90
plot_model_predictions(train_df, train_stats)

# Extract parameter DataFrame
df_params = crm_train_results["params"]

# Compute P50 (median) for each parameter
p50_params = df_params.median()

# Display results
print("P50 Best-Fit CRM Parameters:")
print(p50_params)

#----------------  
# LGM
# 1) Gather train predictions into a 2D array (time steps x samples)
pred_matrix_train = gather_predictions(lgm_train_results, train_df) # shape (Ntrain, Msamples)

# 2) Compute stats: p10, mean, p90, etc.
train_stats = compute_forecast_stats(pred_matrix_train)

# 3) Plot training P10/P50/P90
plot_model_predictions(train_df, train_stats)

# Extract parameter DataFrame
df_params = lgm_train_results["params"]

# Compute P50 (median) for each parameter
p50_params = df_params.median()

# Display results
print("P50 Best-Fit LGM Parameters:")
print(p50_params)

######################################################################
# Model Probabilities
######################################################################

model_results = {
  "arps": {"sse": arps_train_results["sse"], "forecast": arps_train_results["predictions"]},
  "sem":  {"sse": sem_train_results["sse"],  "forecast": sem_train_results["predictions"]},
  "crm":  {"sse": crm_train_results["sse"],  "forecast": crm_train_results["predictions"]},
  "lgm":  {"sse": lgm_train_results["sse"],  "forecast": lgm_train_results["predictions"]}
}

# 1) Build SSE matrix
sse_matrix = gather_sse_matrix(model_results, model_names)  # shape (M, N)

# 2) Compute prob_matrix
prob_matrix = calc_model_probabilities(sse_matrix)  # shape (M, N)

# 3) Summarize marginal model probabilities
marginal_probs = compute_marginal_model_probs(prob_matrix, model_names)
ranked_models = rank_models_by_probability(marginal_probs)

print("Marginal Posterior Probability of each model:")
for m_name, p_val in ranked_models:
    print(f"{m_name}: {p_val:.3f}")

# Plot Model Probability
post_prob_models_plot(ranked_models)

######################################################
# 3) TEST PHASE (Hindcast)
######################################################

# Arps
# Use fitted (Training) parameters to make prediction for testing set:
t_test = test_df["x"].values
arps_params_df = arps_train_results["params"]  # best-fit for each sample

# 1) forecast
test_forecast_matrix = forecast_from_params(ArpsModel, arps_params_df, t_test)

# 2) stats
test_stats = compute_forecast_stats(test_forecast_matrix)

# 3) plot
plot_hindcast(test_df, test_stats)

#----------------  
# SEM
# Use fitted (Training) parameters to make prediction for testing set:
t_test = test_df["x"].values
sem_params_df = sem_train_results["params"]  # best-fit for each sample

# 1) forecast
test_forecast_matrix = forecast_from_params(SEMModel, sem_params_df, t_test)

# 2) stats
test_stats = compute_forecast_stats(test_forecast_matrix)

# 3) plot
plot_hindcast(test_df, test_stats)

#----------------  
# CRM
# Use fitted (Training) parameters to make prediction for testing set:
t_test = test_df["x"].values
crm_params_df = crm_train_results["params"]  # best-fit for each sample

# 1) forecast
test_forecast_matrix = forecast_from_params(CRMModel, crm_params_df, t_test)

# 2) stats
test_stats = compute_forecast_stats(test_forecast_matrix)

# 3) plot
plot_hindcast(test_df, test_stats)

#----------------  
# LGM
# Use fitted (Training) parameters to make prediction for testing set:
t_test = test_df["x"].values
lgm_params_df = lgm_train_results["params"]  # best-fit for each sample

# 1) forecast
test_forecast_matrix = forecast_from_params(LGMModel, lgm_params_df, t_test)

# 2) stats
test_stats = compute_forecast_stats(test_forecast_matrix)

# 3) plot
plot_hindcast(test_df, test_stats)

######################################################
# 4a) FUTURE PHASE (30 yrs after last day)
######################################################

# Arps
# We will generate a brand-new array of times from last_day + 1 to last_day + 5400.
t_future = np.arange(last_day + 1, last_day + 5400 + 1) # daily steps

# 1) forecast
arps_future_forecast_matrix = forecast_from_params(ArpsModel, arps_params_df, t_future)

# 2) stats
arps_future_stats = compute_forecast_stats(arps_future_forecast_matrix)

# 3) plot
plot_future_forecast(t_future, arps_future_stats)

# 4b) CUMULATIVE PRODUCTION FORECAST
# Integrate Rates for “Cumulative” Forecast
# Suppose forecast_matrix is shape (N_future, M)
# dt = 1.0 day if you used np.arange in increments of 1

dt = 1.0  # 1 day
cum_matrix = np.cumsum(arps_future_forecast_matrix, axis=0) * dt # shape (T_future, N_samples)
# or trapezoid rule => np.trapz(..., axis=0)

# The final row cum_matrix[-1, :] is each sample's total future bbl from (last_day+1) to last_day+5400
cum_distribution = cum_matrix[-1, :]  # final row

# Cumulative Production Forecast
arps_p10_cum = np.nanpercentile(cum_distribution, 10)
arps_p25_cum = np.nanpercentile(cum_distribution, 25)
arps_p50_cum = np.nanpercentile(cum_distribution, 50)
arps_mean_cum= np.nanmean(cum_distribution)
arps_p75_cum = np.nanpercentile(cum_distribution, 75)
arps_p90_cum = np.nanpercentile(cum_distribution, 90)

print(f"Arps P10 Cumulative Production: {arps_p10_cum}")
print(f"Arps P50 Cumulative Production: {arps_p50_cum}")
print(f"Arps Mean Cumulative Production: {arps_mean_cum}")
print(f"Arps P90 Cumulative Production: {arps_p90_cum}")

# If you want EUR (i.e. total from well start):
# last_day_cum_oil, this is the historical production up to last_obs_day
arps_p10_eur  = last_day_cum_oil + arps_p10_cum
arps_p25_eur  = last_day_cum_oil + arps_p25_cum
arps_p50_eur  = last_day_cum_oil + arps_p50_cum
arps_mean_eur = last_day_cum_oil + arps_mean_cum
arps_p75_eur  = last_day_cum_oil + arps_p75_cum
arps_p90_eur  = last_day_cum_oil + arps_p90_cum

print(f"Arps P10 EUR: {arps_p10_eur}")
print(f"Arps P50 EUR: {arps_p50_eur}")
print(f"Arps Mean EUR: {arps_mean_eur}")
print(f"Arps P90 EUR: {arps_p90_eur}")

############## END ARPS MODEL
# SEM
# We will generate a brand-new array of times from last_day + 1 to last_day + 5400.
t_future = np.arange(last_day + 1, last_day + 5400 + 1) # daily steps

# 1) forecast
sem_future_forecast_matrix = forecast_from_params(SEMModel, sem_params_df, t_future)

# 2) stats
sem_future_stats = compute_forecast_stats(sem_future_forecast_matrix)

# 3) plot
plot_future_forecast(t_future, sem_future_stats)

# 4b) CUMULATIVE PRODUCTION FORECAST
# Integrate Rates for “Cumulative” Forecast
# Suppose forecast_matrix is shape (N_future, M)
# dt = 1.0 day if you used np.arange in increments of 1

dt = 1.0  # 1 day
cum_matrix = np.cumsum(sem_future_forecast_matrix, axis=0) * dt # shape (N_future, M)
# or trapezoid rule => np.trapz(..., axis=0)

# The final row cum_matrix[-1, :] is each sample's total future bbl from (last_day+1) to last_day+5400
cum_distribution = cum_matrix[-1, :]  # final row

# Cumulative Production Forecast
sem_p10_cum = np.nanpercentile(cum_distribution, 10)
sem_p25_cum = np.nanpercentile(cum_distribution, 25)
sem_p50_cum = np.nanpercentile(cum_distribution, 50)
sem_mean_cum= np.nanmean(cum_distribution)
sem_p75_cum = np.nanpercentile(cum_distribution, 75)
sem_p90_cum = np.nanpercentile(cum_distribution, 90)

print(f"SEM P10 Cumulative Production: {sem_p10_cum}")
print(f"SEM P50 Cumulative Production: {sem_p50_cum}")
print(f"SEM Mean Cumulative Production: {sem_mean_cum}")
print(f"SEM P90 Cumulative Production: {sem_p90_cum}")

# If you want EUR (i.e. total from well start):
# last_day_cum_oil, this is the historical production up to last_obs_day
sem_p10_eur  = last_day_cum_oil + sem_p10_cum
sem_p25_eur  = last_day_cum_oil + sem_p25_cum
sem_p50_eur  = last_day_cum_oil + sem_p50_cum
sem_mean_eur = last_day_cum_oil + sem_mean_cum
sem_p75_eur  = last_day_cum_oil + sem_p75_cum
sem_p90_eur  = last_day_cum_oil + sem_p90_cum

print(f"SEM P10 EUR: {sem_p10_eur}")
print(f"SEM P50 EUR: {sem_p50_eur}")
print(f"SEM Mean EUR: {sem_mean_eur}")
print(f"SEM P90 EUR: {sem_p90_eur}")

############## END SEM MODEL
# CRM
# We will generate a brand-new array of times from last_day + 1 to last_day + 5400.
t_future = np.arange(last_day + 1, last_day + 5400 + 1) # daily steps

# 1) forecast
crm_future_forecast_matrix = forecast_from_params(CRMModel, crm_params_df, t_future)

# 2) stats
crm_future_stats = compute_forecast_stats(crm_future_forecast_matrix)

# 3) plot
plot_future_forecast(t_future, crm_future_stats)

# 4b) CUMULATIVE PRODUCTION FORECAST
# Integrate Rates for “Cumulative” Forecast
# Suppose forecast_matrix is shape (N_future, M)
# dt = 1.0 day if you used np.arange in increments of 1

dt = 1.0  # 1 day
cum_matrix = np.cumsum(crm_future_forecast_matrix, axis=0) * dt # shape (N_future, M)
# or trapezoid rule => np.trapz(..., axis=0)

# The final row cum_matrix[-1, :] is each sample's total future bbl from (last_day+1) to last_day+5400
cum_distribution = cum_matrix[-1, :]  # final row

# Cumulative Production Forecast
crm_p10_cum = np.nanpercentile(cum_distribution, 10)
crm_p25_cum = np.nanpercentile(cum_distribution, 25)
crm_p50_cum = np.nanpercentile(cum_distribution, 50)
crm_mean_cum= np.nanmean(cum_distribution)
crm_p75_cum = np.nanpercentile(cum_distribution, 75)
crm_p90_cum = np.nanpercentile(cum_distribution, 90)

print(f"CRM P10 Cumulative Production: {crm_p10_cum}")
print(f"CRM P50 Cumulative Production: {crm_p50_cum}")
print(f"CRM Mean Cumulative Production: {crm_mean_cum}")
print(f"CRM P90 Cumulative Production: {crm_p90_cum}")

# If you want EUR (i.e. total from well start):
# last_day_cum_oil, this is the historical production up to last_obs_day
crm_p10_eur  = last_day_cum_oil + crm_p10_cum
crm_p25_eur  = last_day_cum_oil + crm_p25_cum
crm_p50_eur  = last_day_cum_oil + crm_p50_cum
crm_mean_eur = last_day_cum_oil + crm_mean_cum
crm_p75_eur  = last_day_cum_oil + crm_p75_cum
crm_p90_eur  = last_day_cum_oil + crm_p90_cum

print(f"CRM P10 EUR: {crm_p10_eur}")
print(f"CRM P50 EUR: {crm_p50_eur}")
print(f"CRM Mean EUR: {crm_mean_eur}")
print(f"CRM P90 EUR: {crm_p90_eur}")

############## END CRM MODEL
# LGM
# We will generate a brand-new array of times from last_day + 1 to last_day + 5400.
t_future = np.arange(last_day + 1, last_day + 5400 + 1) # daily steps

# 1) forecast
lgm_future_forecast_matrix = forecast_from_params(LGMModel, lgm_params_df, t_future)

# 2) stats
lgm_future_stats = compute_forecast_stats(lgm_future_forecast_matrix)

# 3) plot
plot_future_forecast(t_future, lgm_future_stats)

# 4b) CUMULATIVE PRODUCTION FORECAST
# Integrate Rates for “Cumulative” Forecast
# Suppose forecast_matrix is shape (N_future, M)
# dt = 1.0 day if you used np.arange in increments of 1

dt = 1.0  # 1 day
cum_matrix = np.cumsum(lgm_future_forecast_matrix, axis=0) * dt # shape (N_future, M)
# or trapezoid rule => np.trapz(..., axis=0)

# The final row cum_matrix[-1, :] is each sample's total future bbl from (last_day+1) to last_day+5400
cum_distribution = cum_matrix[-1, :]  # final row

# Cumulative Production Forecast
lgm_p10_cum = np.nanpercentile(cum_distribution, 10)
lgm_p25_cum = np.nanpercentile(cum_distribution, 25)
lgm_p50_cum = np.nanpercentile(cum_distribution, 50)
lgm_mean_cum= np.nanmean(cum_distribution)
lgm_p75_cum = np.nanpercentile(cum_distribution, 75)
lgm_p90_cum = np.nanpercentile(cum_distribution, 90)

print(f"LGM P10 Cumulative Production: {lgm_p10_cum}")
print(f"LGM P50 Cumulative Production: {lgm_p50_cum}")
print(f"LGM Mean Cumulative Production: {lgm_mean_cum}")
print(f"LGM P90 Cumulative Production: {lgm_p90_cum}")

# If you want EUR (i.e. total from well start):
# last_day_cum_oil, this is the historical production up to last_obs_day
lgm_p10_eur  = last_day_cum_oil + lgm_p10_cum
lgm_p25_eur  = last_day_cum_oil + lgm_p25_cum
lgm_p50_eur  = last_day_cum_oil + lgm_p50_cum
lgm_mean_eur = last_day_cum_oil + lgm_mean_cum
lgm_p75_eur  = last_day_cum_oil + lgm_p75_cum
lgm_p90_eur  = last_day_cum_oil + lgm_p90_cum

print(f"LGM P10 EUR: {lgm_p10_eur}")
print(f"LGM P50 EUR: {lgm_p50_eur}")
print(f"LGM Mean EUR: {lgm_mean_eur}")
print(f"LGM P90 EUR: {lgm_p90_eur}")

############## END LGM MODEL


######################################################
# 5) FINAL FORECAST COMBINATION
######################################################

# Final forecast combination in the multi-model probabilistic (MM-P)

# 1) Structure of Future Forecasts
arps_future_forecast = arps_future_forecast_matrix.T  # Now shape is (1000, 10950)
sem_future_forecast = sem_future_forecast_matrix.T
crm_future_forecast = crm_future_forecast_matrix.T
lgm_future_forecast = lgm_future_forecast_matrix.T 

# 2) Stack Future Forecasts Into a Tensor
# Suppose each model's forecast is shape (N, T). We'll stack them => shape (M, N, T).
future_forecast_tensor = np.stack([arps_future_forecast, 
                            sem_future_forecast, 
                            crm_future_forecast, 
                            lgm_future_forecast], axis=0)
# shape (M, N, T)
#future_forecast_tensor.shape

# Combine Forecasts Across Models
combined_future_forecast = combine_forecasts_across_models(future_forecast_tensor, prob_matrix)
# => shape (N, T). This is the mixture forecast.

#prob_matrix.shape
#combined_future_forecast.shape

# 4) Compute P10, P50, P90, and Mean for Future Production 
# => shape(N,T).
#If you want P10,P50,P90 at each time step:
T_future = combined_future_forecast.shape[1]

future_p10  = np.zeros(T_future)
future_p25  = np.zeros(T_future)
future_p50  = np.zeros(T_future)
future_p75  = np.zeros(T_future)
future_p90  = np.zeros(T_future)
future_mean = np.zeros(T_future)

for t in range(T_future):
    distribution_t = combined_future_forecast[:, t]  # shape(N,)
    future_p10[t]  = np.nanpercentile(distribution_t, 10)
    future_p25[t]  = np.nanpercentile(distribution_t, 25)
    future_p50[t]  = np.nanpercentile(distribution_t, 50)
    future_p75[t]  = np.nanpercentile(distribution_t, 75)
    future_p90[t]  = np.nanpercentile(distribution_t, 90)
    future_mean[t] = np.nanmean(distribution_t)

# 5) Compute Future Cumulative Production

dt = 1  # Daily production rate
combined_future_cumulative = np.cumsum(combined_future_forecast * dt, axis=1)  # Shape (N, T_future)

combined_future_cumulative.shape

final_cum_production_per_sample = combined_future_cumulative[:, -1]  # Shape (N,)

final_total_p10  = np.nanpercentile(final_cum_production_per_sample, 10)  # Single number
final_total_p25  = np.nanpercentile(final_cum_production_per_sample, 25)  # Single number
final_total_p50  = np.nanpercentile(final_cum_production_per_sample, 50)  # Single number
final_total_p75  = np.nanpercentile(final_cum_production_per_sample, 75)  # Single number
final_total_p90  = np.nanpercentile(final_cum_production_per_sample, 90)  # Single number
final_total_mean = np.nanmean(final_cum_production_per_sample)  # Single number

print("Final P10 Cumulative Production Comparison (bbls):")
print(f"Arps P10: {arps_p10_cum:.2f}")
print(f"SEM  P10: {sem_p10_cum:.2f}")
print(f"CRM  P10: {crm_p10_cum:.2f}")
print(f"LGM  P10: {lgm_p10_cum:.2f}")
print(f"Combined P10: {final_total_p10:.2f} ✅")

print("Final P50 Cumulative Production Comparison (bbls):")
print(f"Arps P50: {arps_p50_cum:.2f}")
print(f"SEM  P50: {sem_p50_cum:.2f}")
print(f"CRM  P50: {crm_p50_cum:.2f}")
print(f"LGM  P50: {lgm_p50_cum:.2f}")
print(f"Combined P50: {final_total_p50:.2f} ✅")

print("Final Mean Cumulative Production Comparison (bbls):")
print(f"Arps mean: {arps_mean_cum:.2f}")
print(f"SEM  mean: {sem_mean_cum:.2f}")
print(f"CRM  mean: {crm_mean_cum:.2f}")
print(f"LGM  mean: {lgm_mean_cum:.2f}")
print(f"Combined mean: {final_total_mean:.2f} ✅")

print("Final P90 Cumulative Production Comparison (bbls):")
print(f"Arps P90: {arps_p90_cum:.2f}")
print(f"SEM  P90: {sem_p90_cum:.2f}")
print(f"CRM  P90: {crm_p90_cum:.2f}")
print(f"LGM  P90: {lgm_p90_cum:.2f}")
print(f"Combined P90: {final_total_p90:.2f} ✅")

final_eur_p10 = final_total_p10 + last_day_cum_oil
final_eur_p25 = final_total_p25 + last_day_cum_oil
final_eur_p50 = final_total_p50 + last_day_cum_oil
final_eur_mean = final_total_mean + last_day_cum_oil
final_eur_p75 = final_total_p75 + last_day_cum_oil
final_eur_p90 = final_total_p90 + last_day_cum_oil

print("Final P10 EUR Comparison (bbls), P10:")
print(f"Arps P10: {arps_p10_eur:.2f}")
print(f"SEM  P10: {sem_p10_eur:.2f}")
print(f"CRM  P10: {crm_p10_eur:.2f}")
print(f"LGM  P10: {lgm_p10_eur:.2f}")
print(f"Combined P10: {final_eur_p10:.2f} ✅")

print("Final P50 EUR Comparison (bbls), P50:")
print(f"Arps P50: {arps_p50_eur:.2f}")
print(f"SEM  P50: {sem_p50_eur:.2f}")
print(f"CRM  P50: {crm_p50_eur:.2f}")
print(f"LGM  P50: {lgm_p50_eur:.2f}")
print(f"Combined P50: {final_eur_p50:.2f} ✅")

print("Final Mean EUR Comparison (bbls), Mean:")
print(f"Arps mean: {arps_mean_eur:.2f}")
print(f"SEM  mean: {sem_mean_eur:.2f}")
print(f"CRM  mean: {crm_mean_eur:.2f}")
print(f"LGM  mean: {lgm_mean_eur:.2f}")
print(f"Combined mean: {final_eur_mean:.2f} ✅")

print("Final P90 EUR Comparison (bbls), P90:")
print(f"Arps P90: {arps_p90_eur:.2f}")
print(f"SEM  P90: {sem_p90_eur:.2f}")
print(f"CRM  P90: {crm_p90_eur:.2f}")
print(f"LGM  P90: {lgm_p90_eur:.2f}")
print(f"Combined P90: {final_eur_p90:.2f} ✅")

df_eur = pd.DataFrame({
    "model_name": ["Arps", "SEM", "CRM", "LGM", "Combined"],
    "y10":   [arps_p10_eur, sem_p10_eur, crm_p10_eur, lgm_p10_eur, final_eur_p10],
    "y25":   [arps_p25_eur, sem_p25_eur, crm_p25_eur, lgm_p25_eur, final_eur_p25],
    "y50":   [arps_p50_eur, sem_p50_eur, crm_p50_eur, lgm_p50_eur, final_eur_p50],
    "y75":   [arps_p75_eur, sem_p75_eur, crm_p75_eur, lgm_p75_eur, final_eur_p75],
    "y90":   [arps_p90_eur, sem_p90_eur, crm_p90_eur, lgm_p90_eur, final_eur_p90],
    "ymean": [arps_mean_eur, sem_mean_eur, crm_mean_eur, lgm_mean_eur, final_eur_mean]
})

eur_boxplot(df_eur)

