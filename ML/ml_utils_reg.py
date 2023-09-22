# imports
import os
import random
from itertools import tee
from typing import List

# Plotting
import matplotlib
import numpy as np
import scipy.sparse as sparse
import seaborn as sns
import tensorflow as tf
# Keras/Tensorflow
from matplotlib import pyplot as plt
# Rdkit
from rdkit import DataStructs, Chem
from rdkit.Chem import AllChem


class TanimotoKernel:
    def __init__(self, sparse_features=False):
        self.sparse_features = sparse_features

    @staticmethod
    def similarity_from_sparse(matrix_a: sparse.csr_matrix, matrix_b: sparse.csr_matrix):
        intersection = matrix_a.dot(matrix_b.transpose()).toarray()
        norm_1 = np.array(matrix_a.multiply(matrix_a).sum(axis=1))
        norm_2 = np.array(matrix_b.multiply(matrix_b).sum(axis=1))
        union = norm_1 + norm_2.T - intersection
        return intersection / union

    @staticmethod
    def similarity_from_dense(matrix_a: np.ndarray, matrix_b: np.ndarray):
        intersection = matrix_a.dot(matrix_b.transpose())
        norm_1 = np.multiply(matrix_a, matrix_a).sum(axis=1)
        norm_2 = np.multiply(matrix_b, matrix_b).sum(axis=1)
        union = np.add.outer(norm_1, norm_2.T) - intersection

        return intersection / union

    def __call__(self, matrix_a, matrix_b):
        if self.sparse_features:
            return self.similarity_from_sparse(matrix_a, matrix_b)
        else:
            raise self.similarity_from_dense(matrix_a, matrix_b)


def tanimoto_from_sparse(matrix_a: sparse.csr_matrix, matrix_b: sparse.csr_matrix):
    DeprecationWarning("Please use TanimotoKernel.sparse_similarity")
    return TanimotoKernel.similarity_from_sparse(matrix_a, matrix_b)


def tanimoto_from_dense(matrix_a: np.ndarray, matrix_b: np.ndarray):
    DeprecationWarning("Please use TanimotoKernel.sparse_similarity")
    return TanimotoKernel.similarity_from_dense(matrix_a, matrix_b)


def create_directory(path: str, verbose: bool = True):
    if not os.path.exists(path):

        if len(path.split("/")) <= 2:
            os.mkdir(path)
        else:
            os.makedirs(path)
        if verbose:
            print(f"Created new directory '{path}'")
    return path


def pairwise(iterable):
    """s -> (s0,s1), (s1,s2), (s2, s3), ..."""
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def construct_check_mol_list(smiles_list: List[str]) -> List[Chem.Mol]:
    mol_obj_list = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]
    if None in mol_obj_list:
        invalid_smiles = []
        for smiles, mol_obj in zip(smiles_list, mol_obj_list):
            if not mol_obj:
                invalid_smiles.append(smiles)
        invalid_smiles = "\n".join(invalid_smiles)
        raise ValueError(f"Following smiles are not valid:\n {invalid_smiles}")
    return mol_obj_list


def construct_check_mol(smiles: str) -> Chem.Mol:
    mol_obj = Chem.MolFromSmiles(smiles)
    if not mol_obj:
        raise ValueError(f"Following smiles are not valid: {smiles}")
    return mol_obj


def ECFP4(smiles_list: List[str], n_bits=2048, radius=2) -> List[DataStructs.cDataStructs.ExplicitBitVect]:
    """
    Converts array of SMILES to ECFP bitvectors.
        AllChem.GetMorganFingerprintAsBitVect(mol, radius, length)
        n_bits: number of bits
        radius: ECFP fingerprint radius

    Returns: RDKit mol objects [List]
    """
    mols = construct_check_mol_list(smiles_list)
    return [AllChem.GetMorganFingerprintAsBitVect(m, radius, n_bits) for m in mols]


def set_seeds(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)


def set_global_determinism(seed):
    set_seeds(seed=seed)

    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    # os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)



def plot_regression_models_cat(df, metric, plot_type='boxplot',
                               filename=None, results_path=None,
                               x=None, y=None,
                               col=None,
                               row=None,
                               ymin=None, ymax=None, yticks=None,
                               xticks=None,
                               palette='tab10',
                               x_labels='',
                               y_labels='',
                               hue=None, hue_order=None, title=True,
                               order=None,
                               legend_title="",
                               col_nr=2,
                               font_size=30, height=10, aspect=1.2, width=None, bbox_to_anchor=(-0.01, -0.15),
                               sharey=True,
                               theme='whitegrid',
                               show=False,
                               fig=None,
                               sub_fig_title=None,
                               **kwargs):

    # database
    performance_df_ = df.loc[df.Metric.isin(metric)]

    # plt parameters
    sns.set_theme(style=theme)
    font = {'size': font_size}
    matplotlib.rc('font', **font)

    if plot_type == 'boxplot':
        kind = "box"
    elif plot_type == 'barplot':
        kind = "bar"
    elif plot_type == 'pointplot':
        kind = "point"

    g = sns.catplot(data=performance_df_, x=x, y=y,
                    kind=kind,
                    height=height, aspect=aspect, #width=width,
                    order=order, palette=palette,
                    hue=hue,
                    hue_order=hue_order,
                    col=col, # col_wrap=col_nr,
                    row=row,
                    legend=False, sharey=sharey, **kwargs)

    g.set_ylabels(y_labels, labelpad=15, fontsize=font_size)
    g.set_xlabels(f'{x_labels}', labelpad=15, fontsize=font_size)
    g.set(ylim=(ymin, ymax))
    g.tick_params(labelsize=font_size)

    if title:
        if row and col:
            print(row, col)
            g.set_titles(r"{row_var}: {row_name} - {col_var}: {col_name}", size=font_size,) #fontweight="bold"
            # g.set_titles(r"$\\bf{row_var}$: {row_name} - $\\bf{col_var}$: {col_name}")
        if isinstance(title, str):
            g.set_titles(title)
        else:
            if fig is not None:
                g.set_titles(r"{row_var}: {row_name} - {col_name}", size=font_size)
            else:
                g.set_titles("{col_var}: {col_name}", size=font_size)
    if yticks:
        g.set(ylim=(ymin, ymax), yticks=yticks)
    if xticks:
        g.set_xticklabels(xticks, fontsize=font_size)

    plt.tight_layout()
    g.despine(right=True, top=True)
    legend = plt.legend(loc='lower center', bbox_to_anchor=bbox_to_anchor, ncol=len(hue_order), #, prop={'size': font_size-10}
               frameon=False, title=legend_title, prop={'size': font_size}, labelspacing=1.5)

    plt.setp(legend.get_title(), fontsize='medium')

    if sub_fig_title:
        plt.suptitle(f'{sub_fig_title}', fontsize = 45, x=0, y=1, fontweight='bold')

    if results_path:
        plt.savefig(results_path + f'{filename}.png', dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    return g
