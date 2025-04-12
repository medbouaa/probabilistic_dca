import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt
from statsmodels.nonparametric.smoothers_lowess import lowess
from sklearn.model_selection import KFold

class ProbabilisticDCA:
    def __init__(self, data, num_samples=1000, train_pct=0.8, k_folds=10, random_state=42):
        self.data = data.copy()
        self.num_samples = num_samples
        self.train_pct = train_pct
        self.k_folds = k_folds
        self.random_state = random_state  # Set random seed
        
    def split_train_test(self):
        """Splits the dataset into training and testing based on the 80/20 rule."""
        max_prod_index = self.data['oil_month_bpd'].idxmax()
        max_prod_day = self.data.loc[max_prod_index, 'cum_eff_prod_day']

        last_day_index = self.data['cum_eff_prod_day'].idxmax()
        train_size = int((last_day_index - max_prod_index) * self.train_pct)
        train_cutoff_day = self.data.loc[max_prod_index + train_size, 'cum_eff_prod_day']

        # Define training and testing sets
        train_data = self.data[self.data['cum_eff_prod_day'] <= train_cutoff_day].reset_index(drop=True)
        test_data = self.data[self.data['cum_eff_prod_day'] > train_cutoff_day].reset_index(drop=True)

        print(f"Training cutoff day: {train_cutoff_day} (Training size: {len(train_data)}, Test size: {len(test_data)})")
        return train_data, test_data
        
    def cross_validate_window_size(self, min_w=3, max_w=8):
        """Finds the optimal half-window size using k-fold cross-validation."""
        best_w = min_w
        min_error = float('inf')
        
        for w in range(min_w, max_w + 1):
            total_error = 0
            kf = KFold(n_splits=self.k_folds, shuffle=True, random_state=42)
            
            for train_idx, val_idx in kf.split(self.data):
                train_data = self.data.iloc[train_idx]
                val_data = self.data.iloc[val_idx]
                
                # Apply LOESS with the given window size
                loess_fit = lowess(train_data['oil_month_bpd'], train_data['cum_eff_prod_day'], frac=w/len(self.data))
                fitted_values = np.interp(val_data['cum_eff_prod_day'], loess_fit[:, 0], loess_fit[:, 1])
                error = np.mean((val_data['oil_month_bpd'] - fitted_values) ** 2)
                total_error += error
            
            avg_error = total_error / self.k_folds
            if avg_error < min_error:
                min_error = avg_error
                best_w = w
        
        print(f"Optimal half-window size: {best_w}")
        return best_w
    
    def estimate_uncertainty(self):
        """Estimates residuals, moving standard deviation, and moving variance using optimized window size."""
        
        # Determine optimal half-window size
        optimal_w = self.cross_validate_window_size()
        frac = optimal_w / len(self.data)
        print(f"Optimal span (frac): {frac}")
        
        # Apply LOESS regression with optimized window size
        loess_fit = lowess(self.data['oil_month_bpd'], self.data['cum_eff_prod_day'], frac=frac)
        self.data['loess_fit'] = np.interp(self.data['cum_eff_prod_day'], loess_fit[:, 0], loess_fit[:, 1])
        self.data['residuals'] = self.data['oil_month_bpd'] - self.data['loess_fit']
        
        # Compute moving standard deviation and variance with optimized window size
        window_size = 2 * optimal_w + 1
        self.data['rolling_sd'] = self.data['residuals'].rolling(window=window_size, center=True).std()
        self.data['rolling_sigma2'] = self.data['rolling_sd'] ** 2  # Compute rolling variance
        
        # Fill NaN values at edges
        self.data['rolling_sd'].fillna(self.data['rolling_sd'].median(), inplace=True)
        self.data['rolling_sigma2'].fillna(self.data['rolling_sigma2'].median(), inplace=True)
        
        return self.data
    
    def monte_carlo_sampling(self):
        """Generates multiple sample datasets using Gaussian noise."""

        np.random.seed(self.random_state)  # Set random seed for reproducibilit
        
        samples = np.zeros((len(self.data), self.num_samples))

        for i in range(len(self.data)):
            day_sample = np.random.normal(
                loc=self.data['oil_month_bpd'].iloc[i], 
                scale=self.data['rolling_sd'].iloc[i], 
                size=self.num_samples)

            samples[i, :] = np.sort(day_sample)[::-1]  # Sort in descending order per row

        sample_sorted_df = pd.DataFrame(samples, columns=[f'sample_{i+1}' for i in range(self.num_samples)])
        sample_sorted_df.insert(0, 'x', self.data['cum_eff_prod_day'].values)
        sample_sorted_df.insert(1, 'y', self.data['oil_month_bpd'].values)
        sample_sorted_df.insert(2, 'sigma2', self.data['rolling_sigma2'].values)

        return sample_sorted_df

    def compute_monte_carlo_statistics(self, sampled_data):
        """Computes Monte Carlo statistics including standard deviation, percentiles (P10, P50, P90), and mean."""

        sample_std = sampled_data.iloc[:, 3:].std(axis=1)
        sample_p10 = sampled_data.iloc[:, 3:].apply(lambda x: np.percentile(x, 10), axis=1)
        sample_mean = sampled_data.iloc[:, 3:].mean(axis=1)
        sample_p50 = sampled_data.iloc[:, 3:].apply(lambda x: np.percentile(x, 50), axis=1)
        sample_p90 = sampled_data.iloc[:, 3:].apply(lambda x: np.percentile(x, 90), axis=1)

        sample_stats_df = pd.DataFrame({
            'x': sampled_data['x'],
            'y': sampled_data['y'],
            'sample_std': sample_std,
            'sample_p10': sample_p10,
            'sample_mean': sample_mean,
            'sample_p50': sample_p50,
            'sample_p90': sample_p90
        })

        return sample_stats_df

    def plot_monte_carlo_results(self, sample_stats_df, sampled_data):
        """Plots the Monte Carlo statistics results along with sampled trajectories."""

        plt.figure(figsize=(10, 6))
        plt.scatter(sample_stats_df['x'], sample_stats_df['y'], color='red', alpha=0.5, label='Original Data')
        plt.plot(sample_stats_df['x'], sample_stats_df['sample_mean'], color='black', label='Mean Trend')
        plt.plot(sample_stats_df['x'], sample_stats_df['sample_p10'], color='red', linestyle='dashed', label='P10')
        plt.plot(sample_stats_df['x'], sample_stats_df['sample_p90'], color='red', linestyle='dashed', label='P90')

        # Plot individual sampled points
        sample_indices = [1, 10, 200, 300, 400, 500, 600, 700, 800, 990, 1000]
        colors = ['blue'] * 10 + ['black']
        for idx, color in zip(sample_indices, colors):
            if f'sample_{idx}' in sampled_data.columns:
                plt.scatter(sampled_data['x'], sampled_data[f'sample_{idx}'], color=color, alpha=0.5, s=10)

        plt.title("Sorted Sampled Data")
        plt.xlabel("Days")
        plt.ylabel("Oil Rate, bbl/day")
        plt.legend()
        plt.grid(True)
        plt.show()
        
# --- Arps Model Definition ---
def arps_model(t, q0, Di, b):
    """Arps decline curve model."""
    if b == 0:
        return q0 * np.exp(-Di * t)  # Exponential decline
    else:
        return q0 * (1 + b * Di * t) ** (-1 / b)  # Hyperbolic decline
    
    
# Define Arps Model with Hyperbolic-to-Exponential Transition
def hyp2exp_transition(Di, b, Df):
    """Computes time to transition from hyperbolic to exponential decline."""
    HARMONIC_EPS = 1e-10
    EXPONENTIAL_EPS = 1e-10

    if Di < EXPONENTIAL_EPS or Df < EXPONENTIAL_EPS or b < HARMONIC_EPS:
        return np.inf  # No transition
    elif abs(Df - Di) < EXPONENTIAL_EPS:
        return 0  # Immediate transition
    else:
        return (Di / Df - 1) / (b * Di)

def hyp2exp_q(qi, Di, b, Df, t):
    """Computes production rate using Arps model with transition to exponential decline."""
    t_trans = hyp2exp_transition(Di, b, Df)  # Compute transition time
    q_trans = arps_model(t_trans, qi, Di, b)  # Rate at transition

    q = arps_model(t, qi, Di, b)  # Apply hyperbolic decline
    mask = t > t_trans  # Transition to exponential decline
    q[mask] = q_trans * np.exp(-Df * (t[mask] - t_trans))  # Exponential decline after transition

    return q

def compute_wse(y_true, y_pred, sigma2):
    """Computes weighted squared error for Bayesian Model Averaging."""
    return ((y_true - y_pred) ** 2) / sigma2


def fit_arps_mle(t, q_obs):
    """Fits the Arps model to data using Maximum Likelihood Estimation (MLE)."""
    def negative_log_likelihood(params, t, q_obs):
        q0, Di, b, sigma = params
        q_pred = arps_model(t, q0, Di, b)
        sigma = max(sigma, 1e-6)  # Regularize sigma to avoid divide by zero
        log_likelihood = -0.5 * np.sum(((q_obs - q_pred) / sigma) ** 2 + np.log(2 * np.pi * sigma ** 2))
        return -log_likelihood

    initial_guess = [q_obs[0], 0.01, 0.5, np.std(q_obs)]
    bounds = [(0, None), (0, None), (0, 0.899), (1e-6, None)]  # Constrain sigma, and limit b to 0.899

    result = opt.minimize(negative_log_likelihood, initial_guess, args=(t, q_obs), bounds=bounds, method="L-BFGS-B")

    if result.success:
        return result.x  # Return estimated parameters
    else:
        return [np.nan, np.nan, np.nan, np.nan]

# Fit Arps Model to Monte Carlo Samples
def fit_arps_to_samples(sampled_data, lower_bounds, upper_bounds, fcst_end=5400):
    """Fits Arps model to all Monte Carlo samples and generates forecasts."""
    num_samples = sampled_data.shape[1] - 3  # Exclude 'x', 'y', 'sigma2'
    t_values = sampled_data["x"].values
    param_estimates = []
    wse_values = []
    fcst_end_values = []

    for i in range(1, num_samples + 1):
        q_values = sampled_data[f"sample_{i}"].values

        # Sample initial parameters from uniform distribution
        initial_guess = [
            np.random.uniform(lower_bounds[0], upper_bounds[0]),  # qi
            np.random.uniform(lower_bounds[1], upper_bounds[1]),  # Di
            np.random.uniform(lower_bounds[2], upper_bounds[2]),  # b
        ]
        Df = 0.10 / 365.25  # Convert 10% annual decline to daily

        # Fit Arps Model (MLE)
        params = fit_arps_mle(t_values, q_values)
        param_estimates.append(params)

        # Compute Weighted Squared Error (WSE)
        q_pred = hyp2exp_q(params[0], params[1], params[2], Df, t_values)
        wse_values.append(compute_wse(q_values, q_pred, sampled_data["sigma2"].values))

        # Forecast Future Production (up to fcst_end days)
        t_fcst_end = np.arange(t_values[-1], fcst_end + 1, 1)  # Forecast range
        q_fcst_end = hyp2exp_q(params[0], params[1], params[2], Df, t_fcst_end)

        fcst_end_values.append(q_fcst_end)

    # Convert results to DataFrames
    param_estimates_df = pd.DataFrame(param_estimates, columns=["q0", "Di", "b", "sigma"])
    wse_df = pd.DataFrame(np.array(wse_values).T, columns=[f"sample_{i}" for i in range(1, num_samples + 1)])
    fcst_end_df = pd.DataFrame(np.array(fcst_end_values).T, columns=[f"sample_{i}" for i in range(1, num_samples + 1)])

    return param_estimates_df, wse_df, fcst_end_df

# --- Running the Full Pipeline ---

# Load the provided production data
prod_data_path = "00_data_wrangled/AF-6h_daily_prod.csv"

# Read the CSV file
prod_data = pd.read_csv(prod_data_path)

# Identify the maximum production rate and its corresponding day
max_prod_index = prod_data['oil_month_bpd'].idxmax()
max_prod_day = prod_data.loc[max_prod_index, 'cum_eff_prod_day']

# Select data from max production day to the last recorded day
filtered_data = prod_data[prod_data['cum_eff_prod_day'] >= max_prod_day].reset_index(drop=True)

# Initialize and run the DCA analysis
dca = ProbabilisticDCA(filtered_data, random_state=42)

# Estimate uncertainty
uncertainty_data = dca.estimate_uncertainty()

# Generate Monte Carlo samples
ordered_monte_carlo_samples = dca.monte_carlo_sampling()

# Compute statistics
monte_carlo_stats = dca.compute_monte_carlo_statistics(ordered_monte_carlo_samples)

# Plot Monte Carlo results
dca.plot_monte_carlo_results(monte_carlo_stats, ordered_monte_carlo_samples)

# Fit Arps model to all samples
num_samples = ordered_monte_carlo_samples.shape[1] - 3
param_estimates = []

for i in range(1, num_samples + 1):
    q_values = ordered_monte_carlo_samples[f"sample_{i}"].values
    params = fit_arps_mle(ordered_monte_carlo_samples["x"].values, q_values)
    param_estimates.append(params)

param_estimates_df = pd.DataFrame(param_estimates, columns=["q0", "Di", "b", "sigma"])

# --- Visualization ---
median_q0 = param_estimates_df["q0"].median()
median_Di = param_estimates_df["Di"].median()
median_b = param_estimates_df["b"].median()

t_fit = np.linspace(ordered_monte_carlo_samples["x"].min(), ordered_monte_carlo_samples["x"].max(), 100)
q_fit = arps_model(t_fit, median_q0, median_Di, median_b)

plt.figure(figsize=(10, 6))
plt.scatter(ordered_monte_carlo_samples["x"], ordered_monte_carlo_samples["sample_1"], color='gray', alpha=0.5, label='Sample 1')
plt.plot(t_fit, q_fit, color='red', label='Arps Model Fit (Median)', linewidth=2)
plt.xlabel("Days")
plt.ylabel("Oil Rate (bbl/day)")
plt.title("Arps Model Fit to Monte Carlo Sample")
plt.legend()
plt.grid(True)
plt.show()















