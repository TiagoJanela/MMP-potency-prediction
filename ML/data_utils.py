from collections import OrderedDict

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import itertools as it


def split_data(df_cpds: pd.DataFrame, df_as: pd.DataFrame, test_size: float = 0.5,
               random_seed: int = 42, split_type: str = None):
    """ Split data into train and test sets
    :param df_cpds: compound data frame
    :param df_as: analogue series data frame
    :param test_size: test size
    :param random_seed: random seed
    :param split_type: type of split
    :return: pd.DataFrame with train and test splits
    """
    df_cpds = df_cpds.copy()
    # select only cpds that are in the mmp df
    df_to_split = (df_cpds[df_cpds['cid'].isin(df_as['cid_1'].values.tolist() + df_as['cid_2'].values.tolist())]
                   .reset_index(drop=True))

    # add mmp_id and analogue_series_id to df_to_split
    df_to_split['mmp_id'] = df_to_split['cid'].apply(
        lambda x: df_as.iloc[[np.where(df_as == x)[0][0]]].mmp_id.values.tolist()[0])
    df_to_split['analog_series_id'] = df_to_split['cid'].apply(
        lambda x: df_as.iloc[[np.where(df_as == x)[0][0]]]['as'].values.tolist()[0])
    df_to_split['similarity'] = df_to_split['cid'].apply(
        lambda x: df_as.iloc[[np.where(df_as == x)[0][0]]].similarity.values.tolist()[0])
    df_to_split['dpot'] = df_to_split['cid'].apply(
        lambda x: df_as.iloc[[np.where(df_as == x)[0][0]]].dpot.values.tolist()[0])

    # split data
    if split_type == 'Stratified':
        mmp_train_idx, mmp_test_idx = train_test_split(df_to_split.index, test_size=test_size,
                                                       random_state=random_seed,
                                                       stratify=df_to_split.mmp_id, shuffle=True)
    elif split_type == 'Random':
        mmp_train_idx, mmp_test_idx = train_test_split(df_to_split.index, test_size=test_size,
                                                       random_state=random_seed, shuffle=True)

    else:
        raise ValueError('split_type must be one of the following: Stratified, Random')

    # assign split to train and test sets
    train_set = df_to_split.iloc[mmp_train_idx]
    test_set = df_to_split.iloc[mmp_test_idx]
    train_set['train_test'] = 'train'
    test_set['train_test'] = 'test'

    # combine train and test sets
    train_test_set = pd.concat([train_set, test_set], axis=0).reset_index(drop=True)

    train_idx = train_test_set[train_test_set['train_test'] == 'train'].index
    test_idx = train_test_set[train_test_set['train_test'] == 'test'].index

    return train_test_set, train_idx, test_idx


def get_df_results(df_pred_perf: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """ Get results data frame with additional columns"""
    df_pred_perf["trial"] = kwargs["trial"]
    df_pred_perf["Approach"] = kwargs["approach"]
    df_pred_perf["Fingerprint"] = kwargs["fingerprint"]
    df_pred_perf["Split"] = kwargs["split"]
    df_pred_perf["mm_trial"] = kwargs["mm_trial"]

    return df_pred_perf


def get_delta_pot_class(dpot, min_pot=0, max_dpot=5, rangev=0.5):
    """ Get delta pot class for a given delta pot value"""

    p_dp = min_pot
    for deltapot_class in [i for i in np.arange(min_pot, max_dpot, rangev)]:
        if dpot == 0:
            return f'({deltapot_class}, {deltapot_class+rangev}]'
        elif p_dp < dpot <= deltapot_class:
            return f'({p_dp}, {deltapot_class}]'
        p_dp = deltapot_class


def dpot_class_metrics(df):

    """ Calculate performance metrics for each MMP delta pot class"""

    db_query = OrderedDict({'Target ID': df['Target ID'].unique()[:],
                            'mm_trial': df.mm_trial.unique()[:],
                            'trial': df.trial.unique()[:],
                            'Split': df.Split.unique()[:],
                            'Algorithm': df.Algorithm.unique()[:],
                            'dpot_class': df.dpot_class.unique()[:],
                            })

    df_params = df.copy()

    dpot_performance = []
    db_search_query = {n: {name: value for name, value in zip(db_query.keys(), comb)} for n, comb in
                       enumerate(it.product(*list(db_query.values())), 1)}

    for i, idx_params in enumerate(tqdm(db_search_query)):

        cur_params = db_search_query[idx_params]
        filters = np.ones(len(df_params), dtype=bool)

        for param_name, param_value in cur_params.items():
            filters &= df_params[param_name] == param_value

        pred_trial = df_params[filters]

        if len(pred_trial) == 0:
            continue
        else:
            result_dict = {"Target ID": cur_params.get('Target ID'),
                           "Trial": cur_params.get('trial'),
                           "mm trial": cur_params.get('mm_trial'),
                           "Split": cur_params.get('Split'),
                           "dPot": cur_params.get('dpot_class'),
                           "Algorithm": cur_params.get('Algorithm'),
                           "Test size": len(pred_trial),
                           "MAE": mean_absolute_error(pred_trial['Experimental'], pred_trial['Predicted']),
                           "MSE": mean_squared_error(pred_trial['Experimental'], pred_trial['Predicted']),
                           "RMSE": mean_squared_error(pred_trial['Experimental'], pred_trial['Predicted'],
                                                      squared=False),
                           "Average similarity": pred_trial['similarity'].mean(),
                           }

            dpot_performance.append(result_dict)
    results_dpot = pd.DataFrame(dpot_performance)
    results_dpot.set_index(["Target ID", "Algorithm", "Test size", "mm trial", "Trial", "Split", "dPot",],
                           inplace=True)
    results_dpot.columns = pd.MultiIndex.from_product([["Value"], ["MAE", "MSE", "RMSE", "Average similarity"]],
                                                      names=["Value", "Metric"])
    results_dpot = results_dpot.stack().reset_index().set_index("Target ID")
    results_dpot.reset_index(inplace=True)

    return results_dpot
