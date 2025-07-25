# funkcje pomocniczne

import os

import pandas as pd
import numpy as np

from dimod import BinaryQuadraticModel
from typing import Union, Callable
from collections import namedtuple

type Path = Union[str, os.PathLike]

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PEGASUS_ROOT = os.path.join(ROOT, "data", "pegasus")
ZEPHYR_ROOT = os.path.join(ROOT, "data", "zephyr")
SQUARE1_ROOT = os.path.join(ROOT, "data", "50x50x1")
SQUARE2_ROOT = os.path.join(ROOT, "data", "50x50x2")
RAW_PEGASUS_ROOT = os.path.join(ROOT, "data", "raw_data", "pegasus")
RAW_SQUARE1_ROOT = os.path.join(ROOT, "data", "raw_data", "square1")
RAW_SQUARE2_ROOT = os.path.join(ROOT, "data", "raw_data", "square2")


def read_instance(path: Path, convention: str = "minus_half") -> tuple:
    """
    :param path:
    :param convention:
    :return: J, h
    """
    df = pd.read_csv(path, sep=" ", header=None, comment="#", names=["i", "j", "value"], index_col=False)

    n = max(df[["i", "j"]].max())
    h = np.zeros(n)
    J = np.zeros((n, n))

    for row in df.itertuples():
        if row.i == row.j:
            h[row.i - 1] = row.value
        elif row.i > row.j:
            J[row.j - 1, row.i - 1] = row.value
        else:
            J[row.i - 1, row.j - 1] = row.value
    if convention == "dwave":
        return J, h
    elif convention == "minus_half":
        return dwave_conv_to_minus_half_convention(J, h)
    elif convention == "minus_half_plus_h":
        return dwave_conv_to_minus_half_convention(J, -h)
    else:
        raise ValueError("Wrong convention")


def read_instance_dict(path: Path, convention: str = "dwave") -> tuple:
    """
    :param path:
    :param convention:
    :return: J, h
    """
    df = pd.read_csv(path, sep=" ", header=None, comment="#", names=["i", "j", "value"])

    h = {}
    J = {}

    for row in df.itertuples():
        if row.i == row.j:
            h[row.i - 1] = row.value
        elif row.i > row.j:
            J[(row.j - 1, row.i - 1)] = row.value  # by zachować górnotrójkątność
        else:
            J[(row.i - 1, row.j - 1)] = row.value
    if convention == "dwave":
        return J, h


def dwave_conv_to_minus_half_convention(J: np.ndarray, h: np.ndarray):
    n = len(h)
    herminian_matrix = np.zeros((n, n))

    # de facto wyciągamy -1/2 przed macierz i zamieniamy ją na hermitowską
    for i in range(n):
        for j in range(i + 1, n):
            J_ij = J[i, j]
            herminian_matrix[i, j] = -J_ij
            herminian_matrix[j, i] = -J_ij

    x = np.random.choice([-1, 1], size=n)
    assert np.allclose(-2 * x @ J @ x.T, x @ herminian_matrix @ x.T)
    assert np.array_equal(herminian_matrix.T, herminian_matrix)  # wszystkie macierze są rzeczywiste

    new_external_fields = -1 * h
    return herminian_matrix, new_external_fields


def ising_to_qubo(J, h):
    bqm = BinaryQuadraticModel(h, J, vartype="SPIN")
    qubo, offset = bqm.to_qubo()
    N = bqm.num_variables
    Q = np.zeros((N, N))
    for (i, j), v in qubo.items():
        if i == j:
            Q[i, i] = v
        elif i > j:
            Q[j, i] = v
        else:
            Q[i, j] = v
    return Q, offset


def calculate_energy(J: np.ndarray, h: np.ndarray, state: np.ndarray, convention: str = "minus_half"):
    if convention == "minus_half":
        return -1 / 2 * state @ J @ state.T - state @ h
    elif convention == "dwave":
        return state @ J @ state.T + state @ h


def calculate_energy_dict(J: dict, h: dict, state: Union[dict, np.ndarray], convention: str = "dwave"):
    if isinstance(state, np.ndarray):
        s = {node: state[idx] for idx, node in enumerate(h.keys())}
    else:
        s = state
    if convention == "dwave":
        E = sum(v * s[i] * s[j] for (i, j), v in J.items())
        E += sum(v * s[i] for i, v in h.items())
    else:
        raise NotImplementedError("Not implemented")
    return E


def calculate_energy_matrix(J: np.ndarray, h: np.ndarray, state: np.ndarray, convention: str = "minus_half"):
    n, _ = J.shape
    if convention == "minus_half":
        A = np.multiply(-1 / 2, J)
        B = np.matmul(A, state) - h.reshape(n, 1)
        C = np.multiply(state, B)
    elif convention == "dwave":
        B = np.matmul(J, state) + h.reshape(n, 1)
        C = np.multiply(state, B)
    return np.sum(C, axis=0)


def all_pegasuses(function: Callable, *args):
    for size in ["P4", "P8", "P16"]:
        for category in ["CBFM-P", "RAU", "RCO"]:
            if size == "P16" and category == "RAU":
                continue
            function(*args, size=size, category=category)


def save_raw_data(df, raw_root, size, category, name):
    save_path = os.path.join(raw_root, size, category)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    df.to_csv(os.path.join(save_path, name), index_label=False)


def save_raw_data_square(df, raw_root,  name):
    save_path = os.path.join(raw_root)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    df.to_csv(os.path.join(save_path, name), index_label=False)