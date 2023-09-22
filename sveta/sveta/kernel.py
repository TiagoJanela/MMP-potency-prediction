from itertools import product
import math
import numpy as np
from scipy import sparse
from scipy.special import binom


def tanimoto_similarity_sparse(matrix_a: sparse.csr_matrix, matrix_b: sparse.csr_matrix):
    """Calculates the Tanimoto similarity between two sparse matrices and returns a similarity matrix.

    Parameters
    ----------
    matrix_a: sparse.csr_matrix
        matrix a
    matrix_b: sparse.csr_matrix
        matrix b
    Returns
    -------
        np.ndarray
    """
    intersection = matrix_a.dot(matrix_b.transpose()).toarray()
    norm_1 = np.array(matrix_a.multiply(matrix_a).sum(axis=1))
    norm_2 = np.array(matrix_b.multiply(matrix_b).sum(axis=1))
    union = norm_1 + norm_2.T - intersection
    return intersection / union


def tanimoto_similarity_dense(matrix_a: np.ndarray, matrix_b: np.ndarray):
    """Calculates the Tanimoto similarity between two dense matrices and returns a similarity matrix.

    Parameters
    ----------
    matrix_a: np.ndarray
        matrix a
    matrix_b: np.ndarray
        matrix b
    Returns
    -------
        np.ndarray
    """
    intersection = matrix_a.dot(matrix_b.transpose())
    norm_1 = np.multiply(matrix_a, matrix_a).sum(axis=1)
    norm_2 = np.multiply(matrix_b, matrix_b).sum(axis=1)
    union = np.add.outer(norm_1, norm_2.T) - intersection

    return intersection / union


def inv_muiltinom_coeff(number_of_players: int, coalition_size: int) -> float:
    """Factor to weight coalitions ins the Shapley formalism.

    Parameters
    ----------
    number_of_players: int
        total number of available players according to the Shapley formalism
    coalition_size
        number of players selected for a coalition
    Returns
    -------
        float
        weight for contribution of coalition
    """
    n_total_permutations = math.factorial(number_of_players)
    n_permutations_coalition = math.factorial(coalition_size)
    n_permutations_remaining_players = math.factorial(number_of_players - 1 - coalition_size)

    return n_permutations_remaining_players * n_permutations_coalition / n_total_permutations


def sveta_f_plus(n_intersecting_features: int, n_difference_features: int, no_player_value: float = 0):
    """

    Parameters
    ----------
    n_intersecting_features
    n_difference_features
    no_player_value: float
        value of an empty coalition. Should be always zero. Likely to be removed later.

    Returns
    -------

    """
    if n_intersecting_features == 0:
        return 0

    shap_sum = np.float64(0)
    total_features = n_intersecting_features + n_difference_features
    # sampling contribution to empty coalition
    # Tanimoto of emtpy would cause an error (0/0) so it is manually set to the value of no_player_value
    shap_sum += (1 - no_player_value) * inv_muiltinom_coeff(total_features, 0)

    coalition_iterator = product(range(n_intersecting_features), range(n_difference_features + 1))

    # skipping empty coaliton as this is done already
    _ = next(coalition_iterator)
    for coal_present, coal_absent in coalition_iterator:
        coal_size = coal_absent + coal_present
        d_tanimoto = coal_absent / (coal_size * coal_size + coal_size)
        n_repr_coal = binom(n_difference_features, coal_absent) * binom(n_intersecting_features - 1, coal_present)
        shap_sum += d_tanimoto * inv_muiltinom_coeff(total_features, coal_size) * n_repr_coal
    return shap_sum


def sveta_f_minus(n_intersecting_features: int, n_difference_features: int, no_player_value: int = 0):
    """

    Parameters
    ----------
    n_intersecting_features: int
    n_difference_features: int
    no_player_value:float
        value of an empty coalition. Should be always zero. Likely to be removed later.

    Returns
    -------
    float

    """
    if n_difference_features == 0:
        return 0
    shap_sum = 0
    total_features = n_intersecting_features + n_difference_features

    # sampling contribution to empty coalition
    # Tanimoto of emtpy would cause an error (0/0) so it is manually set to the value of no_player_value
    shap_sum += (0 - no_player_value) * inv_muiltinom_coeff(total_features, 0)
    coalition_iterator = product(range(n_intersecting_features + 1), range(n_difference_features))

    n_comb = math.factorial(total_features)
    # skipping empty coaliton as this is done already
    _ = next(coalition_iterator)
    for coal_present, coal_absent in coalition_iterator:
        coal_size = coal_absent + coal_present
        d_tanimoto = -coal_present / (coal_size * coal_size + coal_size)
        n_repr_coal = binom(n_difference_features - 1, coal_absent) * binom(n_intersecting_features, coal_present)
        shap_sum += d_tanimoto * inv_muiltinom_coeff(total_features, coal_present + coal_absent) * n_repr_coal
    return shap_sum
