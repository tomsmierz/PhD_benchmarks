import os
import pandas as pd
import numpy as np

from dwave.samplers import SimulatedAnnealingSampler
from src.utils import (PEGASUS_ROOT, ZEPHYR_ROOT, read_instance_dict, all_pegasuses, Path, RAW_PEGASUS_ROOT,
                       save_raw_data, SQUARE1_ROOT, RAW_SQUARE1_ROOT, save_raw_data_square, RAW_SQUARE2_ROOT,
                       SQUARE2_ROOT, calculate_energy, read_instance)
from tqdm import tqdm
from time import time
from math import inf
from typing import Optional

sampler = SimulatedAnnealingSampler()


def get_temp(J, h):

    # W gorącej temperaturze chcemy by każdy spin mógł się zmienić z prawdopodobieństwem co najmniej 50%
    # rozwiązujemy:
    #   0.50 = exp(-hot_beta * max_delta_energy)
    # rozwiązaniem jest hot_beta = log(2)/max_delta_energy, czyli T = max_delta_energy/log(2)
    # gdzie max_delta_energy = 2*max_effective_field, a max_effective_field = max_i (|h_i| + sum_j |J_ij|)

    h_abs = np.abs(h)
    J_abs = np.abs(J)
    values = [h_abs[i] + sum(J_abs[i, :]) for i in range(len(h))]
    max_effective_field = max(values)
    hot = (2*max_effective_field) / np.log(2)

    # w zimnej temp zakładamy że jesteśmy w minimum (lub stanie podstawowym) i chcemy ograniczyć
    # prawdopodobieństwo że jakikolwiek spin się zmieni. Dodatkowo dla uproszczenia, zakładamy że tylko
    # spiny o minimalnej luce mogą wejść w stan wzbudzony. Rozwiązujemy:
    #   0.01 ~ #minimal_gaps exp(- cold_beta min_i min_delta_energy_i)
    # gdzie #minimal_gaps to ilość przypadków o minimalnej luce energetycznej
    # min_delta_energy_i = min_i(min_delta_energy_i).
    # rozwiązaniem jest cold_beta = log(#minimal_gaps/0.01) / min_delta_energy czyli:
    # T =  min_delta_energy / log(#minimal_gaps/0.01)


    h_non_zero = h_abs[np.where(h_abs!=0)]
    def J_non_zero(A):
        return A[np.where(A!=0)]

    values_array = [min(h_non_zero[i], np.min(J_non_zero(J_abs[i, :]))) for i in range(len(h_non_zero))]
    min_effective_field = min(values_array)
    cold = (2*min_effective_field) / np.log(1/0.01)

    return hot, cold



def calculate_delta_e_matrix(J, h, idx, state, M):
    s_k = state[idx, :]  # <- wiersz
    h_k = np.array([h[idx]] * M)
    sum_j = J[idx, :] @ state
    return 2 * s_k * (h_k + sum_j)


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


def acceptance_probability(delta_e, temp, r):
    beta = 1 / temp
    prob = np.ones_like(delta_e)
    mask = prob >= 0
    prob[mask] = np.exp(-beta * delta_e[mask])
    accept = prob > r
    return accept


def simulated_annealing_cpu(J, h, M: int, num_steps: int, temp_range: Optional[tuple] = None,
                               schedule: str = "linear"):
    # inicjalizacja
    n = len(h)
    solution = np.random.choice([-1, 1], size=(n, M))
    energy = calculate_energy_matrix(J, h, solution)

    # Jeżeli temperatura nie była podana to jest dobierana automatycznie
    if not temp_range:
        T_0, T_final = get_temp(J, h)
    else:
        T_0, T_final = temp_range

    # Ustawiamy schedule
    if schedule == "linear":
        schedule = np.linspace(T_0, T_final, num=num_steps, endpoint=True)

    elif schedule == "exponential":
        schedule = np.geomspace(T_0, T_final, num=num_steps, endpoint=True)

    else:
        raise ValueError("Nieprawidłowy schedule")

    for k in range(num_steps):
        temp = schedule[k]
        for idx in range(n):
            delta_e = calculate_delta_e_matrix(J, h, idx, solution, M)
            r = np.random.random()
            mask = acceptance_probability(delta_e, temp, r)

            solution[idx, mask] *= -1
            energy[mask] += delta_e[mask]

    return solution, energy