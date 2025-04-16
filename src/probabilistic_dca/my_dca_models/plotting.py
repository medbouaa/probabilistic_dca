
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.nonparametric.smoothers_lowess import lowess
from matplotlib.ticker import FuncFormatter

def generate_lof_plot(data_tbl_full, time_column='x', rate_column='y'):
    plt.figure(figsize=(5, 3), dpi=100)
    fig, ax = plt.subplots(figsize=(5, 3), dpi=100)

    sns.scatterplot(
        data=data_tbl_full,
        x=time_column,
        y=rate_column,
        hue=data_tbl_full['lof_flag'].map({1: 'Inlier', -1: 'Outlier'}),
        palette={'Inlier': 'tab:blue', 'Outlier': 'tab:red'},
        alpha=0.7,
        ax=ax
    )

    ax.set_xlabel('Cumulative Effective Production Days')
    ax.set_ylabel('Oil Rate (bbl/day)')
    ax.set_title('Production Data with LOF Outliers')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()

    return fig

# Data Processing plots

def plot_data_processing(
    models_df, max_prod_day, max_prod, last_day, last_day_cum_oil, frac_value
):
    """
    Creates the figure showing the production data + LOESS smoothing.

    models_df: DataFrame with columns 'x','y'
    max_prod_day, max_prod: for annotation
    last_day, last_day_cum_oil: for figure caption
    frac_value: the LOESS fraction
    """
    # Apply LOWESS smoothing
    lowess_smoothed = lowess(models_df['y'], models_df['x'], frac=frac_value)
    lowess_x_sorted = lowess_smoothed[:, 0]
    lowess_y_sorted = lowess_smoothed[:, 1]

    # Create the figure & style
    fig, ax = plt.subplots(figsize=(5,3), dpi=100)
    sns.set_style("whitegrid")

    # Scatter raw data
    ax.scatter(models_df['x'], models_df['y'], label='Production Data',
               alpha=0.6, color='black')

    # LOESS line
    ax.plot(lowess_x_sorted, lowess_y_sorted, color='blue',
            linewidth=2, label=f'LOWESS (frac={frac_value})')

    # Titles and labels
    ax.set_title("Oil Production Rate", fontsize=14, color='#08306B', weight='bold')
    ax.set_xlabel("Days", fontsize=12, color='#08306B')
    ax.set_ylabel("Oil Rate (bbl/day)", fontsize=12, color='#08306B')
    ax.legend()

    # y-axis formatter
    def comma_formatter(x, pos):
        return f'{x:,.0f}'
    ax.yaxis.set_major_formatter(FuncFormatter(comma_formatter))
    ax.xaxis.set_major_formatter(FuncFormatter(comma_formatter))

    # Add caption
    caption_text = (f"Data for analysis starts at {max_prod_day} days, "
                    f"when the well reaches a maximum production of {max_prod} bbl/d.\n"
                    f"Total days in production: {last_day} days, "
                    f"Total Cum. Production: {last_day_cum_oil} bbl")
    fig.text(0.5, -0.06, caption_text, wrap=True, ha='center',
             fontsize=10, color='#08306B')

    return fig

def plot_crossval_loess(candidate_results):
    """
    candidate_results: list of tuples, e.g. (w, span, mean_mse)
    Returns a Matplotlib figure that shows cross-validation MSE vs. half-window.
    """
    wvals = [cr[0] for cr in candidate_results]
    mses  = [cr[2] for cr in candidate_results]

    # Create figure/axes
    fig, ax = plt.subplots(figsize=(5, 3), dpi=100)

    # Plot the results
    ax.plot(wvals, mses, marker='o')
    ax.set_title("Cross-validation MSE vs. half-window")
    ax.set_xlabel("half-window size (w)")
    ax.set_ylabel("CV MSE")

    # Return the figure object so the caller can display or save it
    return fig


def plot_rolling_std(dataframe):
    """
    Creates a single figure with two subplots:
      - Left: residuals
      - Right: rolling SD
    Returns a single Figure object.
    """
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))

    # Left subplot: Residuals
    ax1.plot(dataframe['x'], dataframe['model_residuals'],
             label='Residuals', color='black', linestyle='-')
    ax1.scatter(dataframe['x'], dataframe['model_residuals'],
                color='blue', s=10, label='Residual Points')
    ax1.set_title('Residuals Plot')
    ax1.set_xlabel('Days')
    ax1.set_ylabel('Residuals')
    ax1.legend()

    # Right subplot: Rolling SD
    ax2.plot(dataframe['x'], dataframe['roll_sd'],
             label='Rolling SD (Python)', color='blue')
    ax2.scatter(dataframe['x'], dataframe['roll_sd'],
                color='red', s=10, label='Adjusted SD Points')
    ax2.set_title('Adjusted Rolling SD to Match R')
    ax2.set_xlabel('Days')
    ax2.set_ylabel('Standard Deviation')
    ax2.legend()

    fig.tight_layout()  # improve spacing
    return fig

def plot_sample_sorted_datasets(df_samples, df_samples_stats):
    """
    Creates a single figure that shows:
      - Observed Data (scatter)
      - Mean Trend, P10, P90
      - Some random sample columns
    Returns a Matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=(5, 3), dpi=100)

    # Main scatter / lines
    ax.scatter(df_samples_stats['x'], df_samples_stats['y'],
               color='red', alpha=0.5, label='Original Data')
    ax.plot(df_samples_stats['x'], df_samples_stats['sample_mean'],
            color='black', label='Mean Trend')
    ax.plot(df_samples_stats['x'], df_samples_stats['sample_p10'],
            color='red', linestyle='dashed', label='P10')
    ax.plot(df_samples_stats['x'], df_samples_stats['sample_p90'],
            color='red', linestyle='dashed', label='P90')

    # Plot individual sampled points
    sample_indices = [1, 10, 200, 300, 400, 500, 600, 700, 800, 990, 1000]
    colors = ['blue'] * 10 + ['black']
    for idx, color in zip(sample_indices, colors):
        col_name = f'sample_{idx}'
        if col_name in df_samples.columns:
            ax.scatter(df_samples['x'], df_samples[col_name],
                       color=color, alpha=0.5, s=10)

    ax.set_title("Sorted Sampled Data")
    ax.set_xlabel("Days")
    ax.set_ylabel("Oil Rate (bbl/day)")
    ax.legend()
    ax.grid(True)

    return fig


# Plot Original Data + P10/Mean/P90
# Finally, we can produce a matplotlib figure that overlays these lines on top of the actual well production data. For the actual (training) data, we’ll use the x (time) and y (rate) columns from train_df.

def plot_model_predictions(train_df, pred_stats):
    """
    Creates a line plot of p10, mean, p50, p90 vs. time,
    plus the original production data as scatter points.
    Returns a Matplotlib Figure object.
    """
    x_data = train_df['x'].values
    y_data = train_df['y'].values

    p10  = pred_stats['p10']
    p50  = pred_stats['p50']
    p90  = pred_stats['p90']
    mean = pred_stats['mean']

    # Create figure, set style
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(5, 3), dpi=100)

    # Plot observed data
    ax.scatter(x_data, y_data, c='red', alpha=0.5, label='Observed data')

    # Overlay lines
    ax.plot(x_data, p10,  color='black',   linestyle='--', label='P10')
    ax.plot(x_data, p50,  color='orange',  linestyle='-',  label='P50')
    ax.plot(x_data, mean, color='green',   linestyle='-',  label='Mean')
    ax.plot(x_data, p90,  color='black',   linestyle='--', label='P90')

    ax.set_title("Model Predictions (Training Fit)", fontsize=14, color='#08306B', weight='bold')
    ax.set_xlabel("Days", fontsize=12)
    ax.set_ylabel("Oil Rate, bbl/day", fontsize=12)
    ax.legend()

    return fig 


# Plotting Test Set prediction

def plot_hindcast(test_df, forecast_stats):
    """
    test_df: must have columns 'x', 'y'
    forecast_stats: dict with 'p10','p50','p90','mean' arrays (#test_times,)
    Returns a Matplotlib Figure.
    """
    x_test = test_df["x"].values
    y_test = test_df["y"].values

    p10  = forecast_stats["p10"]
    p50  = forecast_stats["p50"]
    p90  = forecast_stats["p90"]
    mean = forecast_stats["mean"]

    fig, ax = plt.subplots(figsize=(5, 3), dpi=100)
    ax.scatter(x_test, y_test, color='red', alpha=0.6, label='Observed (Test) Data')

    ax.plot(x_test, p10,  color='black',  linestyle='--', label='P10')
    ax.plot(x_test, mean, color='green',  linestyle='-',  label='Mean')
    ax.plot(x_test, p50,  color='orange', linestyle='-',  label='P50')
    ax.plot(x_test, p90,  color='black',  linestyle='--', label='P90')

    ax.set_title("Hindcast vs. Observed Test Data")
    ax.set_xlabel("Days")
    ax.set_ylabel("Oil Rate (bbl/day)")
    ax.legend()

    return fig
    
def plot_future_forecast(t_future, forecast_stats):
    """
    t_future: array of future time steps
    forecast_stats: dict with 'p10','p50','p90','mean' arrays
    Returns a Matplotlib Figure.
    """
    p10  = forecast_stats["p10"]
    p50  = forecast_stats["p50"]
    p90  = forecast_stats["p90"]
    mean = forecast_stats["mean"]

    fig, ax = plt.subplots(figsize=(5, 3), dpi=100)

    ax.plot(t_future, p10,  label='P10',  color='red',    linestyle='--')
    ax.plot(t_future, mean, label='Mean', color='green',  linestyle='-')
    ax.plot(t_future, p50,  label='P50',  color='orange', linestyle='-')
    ax.plot(t_future, p90,  label='P90',  color='black',  linestyle='--')

    ax.set_title("15-Year Future Forecast")
    ax.set_xlabel("Days")
    ax.set_ylabel("Oil Rate (bbl/day)")
    ax.legend()

    return fig

def plot_post_prob_models(sorted_list):
    # Extract model names and probabilities from the sorted ranked_models list
    sorted_models, sorted_probs = zip(*sorted_list)  # Unpacking tuples into separate lists

    # Remove "Combined" case if present
    if "Combined" in sorted_models:
        idx = sorted_models.index("Combined")
        sorted_models = list(sorted_models[:idx]) + list(sorted_models[idx+1:])
        sorted_probs = list(sorted_probs[:idx]) + list(sorted_probs[idx+1:])

    # Create figure and axes
    fig, ax = plt.subplots(figsize=(5, 3), dpi=100)

    # Plot a bar chart on `ax`
    bar_colors = ["blue", "orange", "green", "red"][:len(sorted_models)]
    ax.bar(sorted_models, sorted_probs, color=bar_colors)

    # Labels and formatting
    ax.set_ylabel("Marginal Posterior Probability")
    ax.set_title("Model Marginal Posterior Probabilities")

    # Add probability values on bars
    for i, v in enumerate(sorted_probs):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=10)

    ax.set_ylim(0, 1)  # Probability between 0 and 1
    ax.grid(axis="y", linestyle="--", alpha=0.6)

    # Return the figure instead of showing
    return fig

# def boxplot_eur(dataframe):
#     box_data = []
#     for i, row in dataframe.iterrows():
#         box_data.append({
#             'label': row['model_name'],
#             'whislo': row['y10'],
#             'q1':     row['y25'],
#             'med':    row['y50'],
#             'q3':     row['y75'],
#             'whishi': row['y90'],
#             'mean':   row['ymean'],
#             'fliers': []
#         })


#     fig, ax = plt.subplots(figsize=(5, 3))
#     bp = ax.bxp(box_data, showmeans=True, meanline=False, vert=False, patch_artist=True)

#     # Optional: color each box differently
#     colors = ["#66c2a5","#fc8d62","#8da0cb","#e78ac3","#a6d854"]
#     for patch, color in zip(bp['boxes'], colors):
#         patch.set_facecolor(color)

#     ax.set_title("Boxplot of Multi-Model Probabilistic EUR")
#     ax.set_xlabel("EUR, bbl")
#     ax.set_yticklabels(dataframe["model_name"])

#     plt.gca().invert_yaxis()  # Flip order if desired
#     plt.grid(True, axis='x', linestyle='--', alpha=0.7)
#     return fig  


def boxplot_eur(dataframe):
    box_data = []
    for i, row in dataframe.iterrows():
        box_data.append({
            'label': row['model_name'],
            'whislo': row['y10'],
            'q1':     row['y25'],
            'med':    row['y50'],
            'q3':     row['y75'],
            'whishi': row['y90'],
            'mean':   row['ymean'],
            'fliers': []
        })

    fig, ax = plt.subplots(figsize=(5, 3))
    bp = ax.bxp(box_data, showmeans=True, meanline=False, vert=False, patch_artist=True)

    # Optional: color each box differently
    colors = ["#66c2a5", "#fc8d62", "#8da0cb", "#e78ac3", "#a6d854"]
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    # ✅ Smaller fonts
    ax.set_title("Boxplot of Multi-Model Probabilistic EUR", fontsize=8)
    ax.set_xlabel("EUR, bbl", fontsize=10)
    ax.set_yticklabels(dataframe["model_name"], fontsize=9)

    ax.tick_params(axis='both', labelsize=9)

    plt.gca().invert_yaxis()  # Flip order if desired
    plt.grid(True, axis='x', linestyle='--', alpha=0.7)

    return fig
  
