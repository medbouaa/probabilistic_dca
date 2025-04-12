import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
from probabilistic_dca.my_dca_models.plotting import generate_lof_plot

from probabilistic_dca.config import LOF_N_NEIGHBORS, LOF_CONTAMINATION

from probabilistic_dca.logging_setup import setup_logger
logger = setup_logger(__name__)

def load_data(production_csv):
    logger.info("Loading wells and production data")
    #data_wells = pd.read_csv(wells_csv)
    data_production = pd.read_csv(production_csv)
    logger.info("Data loaded successfully")
    return data_production


def remove_outliers(dataframe, time_col, rate_col, cum_col):
    logger.info("Removing outliers using LOF")
    lof = LocalOutlierFactor(n_neighbors=LOF_N_NEIGHBORS, contamination=LOF_CONTAMINATION)
    lof_labels = lof.fit_predict(dataframe[[rate_col]].values)
    dataframe['lof_flag'] = lof_labels
    lof_plot = generate_lof_plot(dataframe, time_col, rate_col)
    clean_df = dataframe[dataframe['lof_flag'] == 1].reset_index(drop=True)
    data_tbl = pd.DataFrame({
        'x': clean_df[time_col],
        'y': clean_df[rate_col],
        'z': clean_df[cum_col],
    })
    logger.info("Outliers removed")
    return data_tbl, lof_plot


def split_train_test(sample_df, x_train_i):
    train_df = sample_df.iloc[:x_train_i+1].copy()
    test_df = sample_df.iloc[x_train_i+1:].copy()
    return train_df, test_df