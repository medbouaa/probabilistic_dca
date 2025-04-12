import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from scipy.optimize import curve_fit

"""
Single Model Probabilistic (SM-P) Decline Curve Analysis with Arps Model
Author: Alexis Ortega - Mar. 2020

This script applies the Single Model Probabilistic (SM-P) approach (Hong et al. 2018) to forecast oil production of horizontal wells in Vaca Muerta.
For a given decline curve model (in this case, modified Arps), the model parameters are determined through history matching,
minimizing a predefined loss function to adjust model parameters. Once the parameters are established, future oil production is forecasted.

References:
Hong, Aojie, et al. "Integrating Model Uncertainty in Probabilistic Decline Curve Analysis for Unconventional Oil Production Forecasting."
Unconventional Resources Technology Conference, Houston, Texas, 23-25 July 2018.
"""

# Load Data
# Reading well production data and estimated EURs
new_data_wells_final_df = pd.read_csv("Data/new_wells_final_Q42019.csv")
data_production_final_df = pd.read_csv("Data/production_final_Q42019_original.csv")
eur_estimated_df = pd.read_csv("Data/eur_estimados.csv")

# Global parameters
number_of_models = 1  # Only Arps Model is used
model_names = ["arps"]
n_samples = 1000  # Number of samples in Monte Carlo simulation
kfolds = 10  # Number of folds for cross-validation
train_pct = 1.0  # Percentage of data used for training

"""
## Objectives
Estimate Single-Model Probabilistic (Arps, SM-P) EUR for selected wells in Vaca Muerta.
"""

# Selecting wells for analysis
# Filtering horizontal oil wells with at least 12 months of production
well_list = (
    data_production_final_df
    .query("produced_fluid == 'Oil' & well_type == 'Horizontal' & oil_month_bbl > 0")
    .groupby("well_name")
    .filter(lambda x: len(x) >= 12)
    .drop_duplicates("well_name")
    .reset_index()[["well_name"]]
)

print(f"Total wells selected: {len(well_list)}")

"""
## EUR Calculation
Applying Arps model to estimate EUR for each well using probabilistic decline curve analysis.
"""

# Applying Arps model to estimate EUR for each well
calc_arps_list = [arps_oil_eur(well) for well in well_list["well_name"]]
calc_arps_df = pd.concat(calc_arps_list, keys=well_list["well_name"], names=["well_name"]).reset_index()

# Extracting EUR results
calc_eur_df = calc_arps_df[['well_name', 'p10', 'average', 'p50', 'p90']].dropna()
calc_par_df = calc_arps_df[['well_name', 'model_par', 'par_value']].dropna().pivot(index='well_name', columns='model_par', values='par_value').reset_index()

# Merging EUR results with original well dataset
new_data_wells_final_df = new_data_wells_final_df.merge(calc_eur_df, on="well_name", how="left")
new_data_wells_final_df = new_data_wells_final_df.merge(calc_par_df, on="well_name", how="left")

# Save updated dataset with EUR estimates
new_data_wells_final_df.to_csv("Data/new_wells_final_Q12020.csv", index=False)

"""
# Visualization - EUR Comparison
## EUR (P50) from Arps versus EUR from GiGa
"""

# Comparing estimated EUR (P50) from Arps model with GiGa estimates
eur_results_df = new_data_wells_final_df.query("well_status == 'Active' & produced_fluid == 'Oil' & well_type == 'Horizontal'")
eur_results_df = eur_results_df.merge(eur_estimated_df, on="well_name", how="left")
eur_results_df["est_eur_mbbl"] = eur_results_df["est_eur_mm3"] * 6.28981  # Convert mm3 to Mbbl

fig = px.scatter(
    eur_results_df,
    x="eur_oil_mbbl",
    y="p50",
    color="max_qo_bpd",
    hover_data=["well_name"],
    title="EUR (P50) from SM-P Arps versus EUR from GiGa"
)
fig.show()

"""
## EUR (P50) from SM-P Arps versus Maximum Oil Rate
"""

# Histogram of fitted b-Arps values
plt.figure(figsize=(10, 5))
sns.histplot(eur_results_df['b'], bins=15, kde=True, color='royalblue')
plt.title("Histogram of fitted b-Arps")
plt.xlabel("Estimated b")
plt.ylabel("Count")
plt.show()

"""
## Saving results
Saving fitting parameters for future reference.
"""

# Save fitting parameters for future reference
calc_par_df.to_csv("Data/arps_fit_par_Q12020_v1.csv", index=False)
