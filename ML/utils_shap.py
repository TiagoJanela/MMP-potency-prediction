import os
import re

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error


def get_df_shap(data, trial, model, split, model_explainer, predictions_test, expected_value, shap_values):
    """ Get the shap df"""
    # shap df
    df_shap_trial = pd.DataFrame({
        "trial": trial,
        "algorithm": model,
        "split": split,
        "explainer": model_explainer,
        "cid": data.cid,
        "smiles": data.smiles,
        "target ID": data.target,
        "experimental": data.labels,
        "prediction": predictions_test,
        "expected_value": expected_value,
        "mae": [mean_absolute_error([t], [p]) for t, p in zip(data.labels, predictions_test)],
        "analog_series_id": data.analog_series_id,
        "mmp_id": data.mmp_id,
        "similarity": data.similarity,
        "dPot": data.dPot,
        "shap_values": shap_values,
        "fingerprint": data.features.tolist(),

    })
    df_shap_trial['conf_expected'] = df_shap_trial.apply(lambda x: sum(x['shap_values']) + x['expected_value'], axis=1)

    return df_shap_trial


def get_mmp_core(tid, as_id, cid, mmp_fpath):
    """ Get the mmp core"""
    if os.path.exists(mmp_fpath + f"df_mmp_{tid}.csv"):
        df_mmp_tid = pd.read_csv(mmp_fpath + f"df_mmp_{tid}.csv")
    else:
        raise ValueError(f'No mmp file found for tid: {tid}')
    mmp_sub = df_mmp_tid.loc[(df_mmp_tid['as'] == as_id) & (df_mmp_tid['cid'] == cid)]['core'].values[0]

    sub_core = re.sub(r"\[\*:\d+\]", "", mmp_sub)
    sub_core_f = re.sub(r"\(\)", "", sub_core)
    return sub_core_f


def get_shap_bit_values(df):
    """ Get the shap bit values"""
    df_temp = df.copy()

    # Get the fingerprint and shap values
    mmp_fps = np.stack(df_temp.fingerprint.values.tolist())
    shap_values = np.stack(df_temp.shap_values.values.tolist())

    # Get the absent features for each compound
    absent_feat0 = np.where((mmp_fps[0] == 0) | (mmp_fps[0] == 1), mmp_fps[0] ^ 1, mmp_fps[0])
    absent_feat1 = np.where((mmp_fps[1] == 0) | (mmp_fps[1] == 1), mmp_fps[1] ^ 1, mmp_fps[1])

    # Get the uncommon bits
    uncommon_bits_mmp = sorted(list(set(np.nonzero(mmp_fps[0])[0]) ^ set(np.nonzero(mmp_fps[1])[0])))

    common_abs = list(np.intersect1d(np.nonzero(absent_feat0), np.nonzero(absent_feat1)))
    common_pres = list(np.intersect1d(np.nonzero(mmp_fps[0]), np.nonzero(mmp_fps[1])))

    results = []
    for i in range(len(df_temp)):
        predicted_value = df_temp.prediction.values[i]
        # Get the common bits
        common_pres_sum = sum(np.take(list(df_temp.shap_values.values[i]), common_pres))
        common_abs_sum = sum(np.take(list(df_temp.shap_values.values[i]), common_abs))
        # Get the uncommon bits
        uncommon_dict = {k: v for k, v in zip(uncommon_bits_mmp, mmp_fps[i][uncommon_bits_mmp])}
        uncommon_bits_sum = sum(shap_values[i][uncommon_bits_mmp])
        uncommon_pres = [k for k, v in uncommon_dict.items() if v == 1]
        uncommon_abs = [k for k, v in uncommon_dict.items() if v == 0]
        uncommon_pres_sum = sum(np.take(list(df_temp.shap_values.values[i]), uncommon_pres))
        uncommon_abs_sum = sum(np.take(list(df_temp.shap_values.values[i]), uncommon_abs))

        expected_value = df_temp.expected_value.values[i]

        dict_results = pd.DataFrame([{'cid': df_temp.cid.values[i],
                                      'Set': df_temp.Set.values[i],
                                      'Experimental': df_temp.experimental.values[i],
                                      'Prediction': predicted_value,
                                      'Total': expected_value + common_pres_sum + common_abs_sum + uncommon_pres_sum + uncommon_abs_sum,
                                      'Expected value': expected_value,
                                      'Commonly present': common_pres_sum,
                                      'Commonly absent': common_abs_sum,
                                      'Distinct present': uncommon_pres_sum,
                                      'Distinct absent': uncommon_abs_sum,
                                      }])

        results.append(dict_results)

    df_results = pd.concat(results)
    df_results['trial'] = df_temp.trial.unique()[0]
    df_results['split'] = df_temp.split.unique()[0]
    df_results['mmp_id'] = df_temp.mmp_id.unique()[0]
    df_results['Target ID'] = df_temp['target ID'].unique()[0]
    df_results['as_id'] = df_temp.analog_series_id.unique()[0]
    df_results['Explainer'] = df_temp.explainer.unique()[0]
    df_results['Algorithm'] = df_temp.algorithm.unique()[0]
    df_results['dpot'] = df_temp.dPot.unique()[0]

    return df_results
