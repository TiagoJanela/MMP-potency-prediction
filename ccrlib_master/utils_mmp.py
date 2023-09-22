import random
from itertools import combinations

import numpy as np
import pandas as pd
from rdkit import DataStructs

from regression_shap_mmp.ML.ml_utils_reg import ECFP4


def DelOnlyRatom(df, return_delidx=False):
    """
    - Delete MMPs including a substructure whose smiles is only Ratom
    - After deleting, New index is assigned
    - if 'return_index' is true, this function returns df and dropped index (for dubuging)
    """

    smiles = df[["sub1", "sub2"]]

    smiles['sub1-sub2'] = smiles.min(axis=1) + " - " + smiles.max(axis=1)

    idx = [i for i in df.index[np.where(smiles["sub1-sub2"].str.startswith("[R1] -"))[0]]]

    df = df.drop(idx)
    df.index = range(df.shape[0])

    if return_delidx:
        return df, idx

    else:
        return df


def count_heavy_atoms(smi: str):
    """
    Count the number of heavy atoms in a Smiles string

    :param smi: Smiles
    :return: number of heavy atoms

    >>> count_heavy_atoms("Oc1cc(N)ccc1C")
    9
    >>> count_heavy_atoms("[CH3]N1C(=NC(C1=O)(c2ccccc2)c3cc[cH]cc3)N")
    20
    >>> count_heavy_atoms("CC[C@H](C)[C@@H]1C(=O)N[C@@H](C(=O)N1[C@H](C2=COC(=N2)C)C(=O)N3CCOCC3)C4CC5=CC=CC=C5C4")
    36
    """
    # assumes only implicit hydrogens!
    in_bracket = 0
    heavy_count = 0
    for c in smi:
        if in_bracket:
            if in_bracket == 1 and c not in "HR*":
                heavy_count += 1
            in_bracket = 0 if c == "]" else in_bracket + 1
        elif c == "[":
            in_bracket = 1
        elif c.upper() in "BCNOSPFI":
            heavy_count += 1
    return heavy_count


def get_delta_potency(df, cpd_idx, cpd_idy, target_id):
    """ Get the delta potency between two compounds
    :param df: dataframe with potency values
    :param cpd_idx: id of the first compound
    :param cpd_idy: id of the second compound
    :param target_id: target id
    :return: delta potency
     """
    # Select target df
    df_tid_ = df.loc[df.tid == target_id]

    # Select compounds
    cpdx = df_tid_.loc[df_tid_.cid == cpd_idx]
    cpdy = df_tid_.loc[df_tid_.cid == cpd_idy]

    # Get potency values
    potx = cpdx.pPot.values[0]
    poty = cpdy.pPot.values[0]

    delta_pot = abs(potx - poty)

    return delta_pot


def get_potency(df, cpd_id, target_id):
    """ Get the potency of a compound
    :param df: dataframe with potency values
    :param cpd_id: id of the compound
    :param target_id: target id
    :return: potency
     """
    # Select target df
    df_tid_ = df.loc[df.tid == target_id]

    # Select compounds
    cpd = df_tid_.loc[df_tid_.cid == cpd_id]

    # Get potency values
    potency = cpd.pPot.values[0]

    return potency


def get_df_mmp_simple(mmp_result):
    matrix = []
    for i, mmp in enumerate(list(mmp_result)):
        core = mmp[0]
        cpd1 = list(mmp[1])
        cpd2 = list(mmp[2])
        sub1 = cpd1[0]
        sub2 = cpd2[0]
        cpd_id1 = cpd1[1]
        cpd_id2 = cpd2[1]
        mmp_row = (core, sub1, sub2, cpd_id1, cpd_id2)
        matrix.append(mmp_row)

    return pd.DataFrame(matrix, columns=['core', 'sub2', 'sub1', 'cpd_id2', 'cpd_id1'])


def get_similarity(df, cpd_idx, cpd_idy, target_id):
    """ Get the similarity between two compounds"""
    # Select target df
    df_tid_ = df.loc[df.tid == target_id]

    # Select compounds
    cpdx = df_tid_.loc[df_tid_.cid == cpd_idx]
    cpdy = df_tid_.loc[df_tid_.cid == cpd_idy]
    cpds = pd.concat([cpdx, cpdy], axis=0)

    fps = ECFP4(cpds.nonstereo_aromatic_smiles.values)
    sim = DataStructs.TanimotoSimilarity(fps[0], fps[1])

    return sim


def get_mms(mmp_result):
    """ Get the matched molecular series from the mmp_result object"""
    df_mms_final = pd.DataFrame()
    for i, mms in enumerate(list(mmp_result.mms.items())[:]):
        core = mms[0]
        for x, cpd in enumerate(mms[1]):
            cid = cpd[1]
            sub = cpd[0]
            mms_row = [(core, sub, cid, i)]
            df_mms_ = pd.DataFrame(mms_row, columns=['core', 'sub', 'cid', 'as']).sort_values(by='cid')
            df_mms_final = pd.concat([df_mms_final, df_mms_])

    df_mms_final.reset_index(drop=True, inplace=True)
    return df_mms_final


def get_unique_mmps(df_mms, random_state=42):
    """ Get the unique mmps from the matched molecular series"""

    # Get duplicated compounds in matched molecular series
    mms_dup = df_mms[df_mms.duplicated('cid', keep=False)].copy()
    # Calculate the number of heavy atoms in the core
    mms_dup['c_ha'] = mms_dup['core'].apply(count_heavy_atoms)
    # Shuffle the list of duplicated compounds (unique)
    mms_dup_cid = mms_dup['cid'].unique()
    random.Random(random_state).shuffle(mms_dup_cid)

    mms_cid_unique = []
    for cid in mms_dup_cid:
        df_cid = mms_dup[mms_dup['cid'] == cid]
        # If the number of core heavy atoms is the same, randomly select one
        if df_cid['c_ha'].nunique() == 1:
            selected_row = df_cid.sample(1, random_state=random_state)
        # If the number of core heavy atoms is different,
        # select the one with the highest number of heavy atoms in the core
        else:
            selected_row = df_cid.sort_values(by='c_ha', ascending=False).head(1)
        mms_cid_unique.append(selected_row)

    mms_cid_unique = pd.concat(mms_cid_unique)

    mms_final = pd.concat([
        mms_cid_unique[['core', 'sub', 'cid', 'as']],
        df_mms[~df_mms['cid'].isin(mms_cid_unique['cid'])][['core', 'sub', 'cid', 'as']]
    ]).sort_values(by='as').reset_index(drop=True)

    return mms_final


def get_df_mmp(mms_final, random_state=42):
    """ Get the matched molecular pairs from the matched molecular series"""
    mmp_final_df = pd.DataFrame()

    # Shuffle the list of analog series
    analog_series = mms_final['as'].unique()
    random.Random(random_state).shuffle(analog_series)

    # iterate over each matched molecular series
    for AS in analog_series[:]:
        # get the matched molecular series
        df_AS = mms_final[mms_final['as'] == AS]

        # Shuffle the list of compounds in the matched molecular series
        cids_as = df_AS['cid'].unique()
        random.Random(random_state).shuffle(cids_as)

        # get the combinations of compounds in the matched molecular series
        cid_mmp = combinations(cids_as, 2)

        # iterate over each combination of compounds
        already_in_df = []
        mmp_rows = []
        for mmp in cid_mmp:

            # get the ids of the compounds in the combination
            mmp_ids = list(mmp)

            # if the one or two of the cpds are already in the dataframe, skip it
            if any(mmp_id in [item for row in already_in_df for item in row] for mmp_id in mmp_ids):
                continue

            # if the combination is not in the dataframe, add it
            else:
                df_mmp_ = df_AS[df_AS['cid'].isin(mmp)]
                df_mmp_1_ = df_mmp_.iloc[:1].rename(columns={'cid': 'cid_1', 'sub': 'sub_1'}).set_index(['core', 'as'])
                df_mmp_2_ = df_mmp_.iloc[1:].rename(columns={'cid': 'cid_2', 'sub': 'sub_2'}).set_index(['core', 'as'])
                df_mmp__ = pd.concat([df_mmp_1_, df_mmp_2_], axis=1, join='inner')
                mmp_rows.append(df_mmp__)
                already_in_df.append(mmp_ids)

        if mmp_rows:
            mmp_final_df = pd.concat([mmp_final_df, *mmp_rows])

    return mmp_final_df
