
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

######################################################################
# “load_train_results”
######################################################################

def load_train_results(params_csv,
                      sse_csv,
                      preds_csv):
    """
    Reconstruct model_train_results dict from three CSV files that store:
      1) 'params' DataFrame (model_parameters.csv)
      2) 'sse' DataFrame (model_sse.csv)
      3) 'predictions' wide DataFrame (model_predictions.csv)
    """

    # 1) Load 'params' => a DataFrame with index = sample_1, sample_2, ...
    #    e.g. columns: [qi, Di, b, Df]
    df_params = pd.read_csv(params_csv, index_col=0)
    # Now df_params.index => e.g. ['sample_1','sample_2',...]

    # 2) Load 'sse' => a DataFrame with columns like [sample, sse]
    df_sse = pd.read_csv(sse_csv)
    # If the 'sample' column matches the index in df_params, we can set it as the DataFrame index
    df_sse = df_sse.set_index("sample")
    # Reindex in case sample ordering differs from df_params
    df_sse = df_sse.reindex(df_params.index)
    # Now extract the SSE array (same order as params)
    sse_array = df_sse["sse"].values

    # 3) Load 'predictions' => a DataFrame with columns: [x, sample_1, sample_2, ...]
    df_preds = pd.read_csv(preds_csv)
    # Rebuild the dictionary: 'sample_1': array([...]), 'sample_2': array([...]), etc.
    predictions_dict = {}
    for col in df_preds.columns:
        if col.startswith("sample_"):
            predictions_dict[col] = df_preds[col].values
    # Optionally store the time array from df_preds["x"] if you need it
    # x_array = df_preds["x"].values

    # 4) Re-assemble the dictionary
    arps_train_results = {
        "params": df_params,          # DataFrame
        "sse": sse_array,             # NumPy array
        "predictions": predictions_dict  # Dict of {sample_j => array}
    }

    return arps_train_results

######################################################################
# “load_all_model_train_results”
######################################################################

def load_all_model_train_results(model_names, base_dir="00_data_wrangled/model_fit_results"):
    """
    Loads all saved model training results (params, SSE, predictions) from CSV files.

    Parameters:
        model_names (list of str): List of model names (e.g., ["arps", "sem", "crm", "lgm"])
        base_dir (str): Base directory where results are stored.

    Returns:
        model_train_results (dict): Dictionary of model results keyed by model name.
    """
    model_train_results = {}

    for m_name in model_names:
        print(f"Loading saved results for {m_name.upper()}...")

        param_path = f"{base_dir}/{m_name}/{m_name}_parameters.csv"
        sse_path   = f"{base_dir}/{m_name}/{m_name}_sse.csv"
        pred_path  = f"{base_dir}/{m_name}/{m_name}_predictions.csv"

        try:
            results = load_train_results(param_path, sse_path, pred_path)

            # Validate structure
            if not all(key in results for key in ["params", "sse", "predictions"]):
                raise ValueError(f"Incomplete data for model: {m_name}")

            model_train_results[m_name] = results

        except Exception as e:
            print(f"❌ Failed to load {m_name.upper()}: {e}")

    print("✅ All model results loaded.\n")
    return model_train_results

######################################################################
# “calc_model_probabilities”
######################################################################

def calc_model_probabilities(sse_matrix):
    """
    sse_matrix: shape (M, N).
    Returns prob_matrix of same shape (M, N), 
    prob_matrix[i,j] = posterior probability of model i given sample j.
    """
    # exponentiate
    # shape is (M, N)
    exponentiated = np.exp(-0.5 * sse_matrix)
    # sum over models => axis=0
    denom = exponentiated.sum(axis=0)  # shape (N,)
    # Probability
    prob_matrix = exponentiated / denom  # broadcast => shape (M, N)
    return prob_matrix

######################################################################
# “compute_marginal_model_probs”
######################################################################

def compute_marginal_model_probs(prob_matrix, model_names):
    """
    prob_matrix: shape (M, N), M=#models, N=#samples
    model_names: list of length M
    Returns a dict of model_name => marginal probability
    """
    M, N = prob_matrix.shape
    marginal_probs = prob_matrix.sum(axis=1)/N  # shape (M,)
    return dict(zip(model_names, marginal_probs))

######################################################################
# “rank_models_by_probability”
######################################################################

def rank_models_by_probability(marginal_probs):
    """
    marginal_probs: dict of {model_name: prob}
    Returns a list of tuples sorted by descending probability.
    """
    sorted_list = sorted(marginal_probs.items(), key=lambda x: x[1], reverse=True)
    return sorted_list

######################################################################
# “combine_forecasts_across_models”
######################################################################

def combine_forecasts_across_models(forecast_tensor, prob_matrix):
    """
    forecast_tensor: shape (M, N, T) or (M, N) if T=1
      - M = #models, N = #samples, T = #time steps (or 1 if just a single quantity)
    prob_matrix: shape (M, N)
      - prob_matrix[i,j] = P(model i | sample j)

    Returns combined_forecast: shape (N, T) or shape (N,) if T=1
      For each sample j (and time t if T>1),
      combined_forecast[j,t] = sum_i [ forecast_tensor[i,j,t] * prob_matrix[i,j] ]
    """

    # We need to expand prob_matrix to (M, N, T) if T>1, so it broadcasts
    # If forecast_tensor has shape (M, N, T), we do:
    if forecast_tensor.ndim == 3:
        M, N, T = forecast_tensor.shape
        # prob_matrix shape is (M, N). Expand to (M, N, 1)
        expanded_probs = prob_matrix[:, :, None]  # shape (M, N, 1)
        weighted = forecast_tensor * expanded_probs  # shape (M, N, T)
        combined = weighted.sum(axis=0)  # sum over models => shape (N, T)
    elif forecast_tensor.ndim == 2:
        M, N = forecast_tensor.shape
        # prob_matrix shape is (M, N). Weighted sum => shape(N,)
        expanded_probs = prob_matrix  # (M, N)
        weighted = forecast_tensor * expanded_probs
        combined = weighted.sum(axis=0)  # shape (N,)
    else:
        raise ValueError("forecast_tensor must be 2D or 3D (M,N) or (M,N,T).")
    
    return combined