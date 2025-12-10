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


def acceptance_probability(delta_e: float, temp: float):
    if delta_e < 0:
        return 1
    else:
        beta = 1 / temp
        probability = np.exp(-beta * delta_e)

        return probability


def calculate_delta_e(J, h, idx, state):
    s_k = state[idx]
    sum_j = J[idx, :] @ state.T
    return 2 * s_k * (h[idx] + sum_j)


def calculate_delta_e_parallel():
    ...


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
    hot = (2 * max_effective_field) / np.log(2)

    # w zimnej temp zakładamy że jesteśmy w minimum (lub stanie podstawowym) i chcemy ograniczyć
    # prawdopodobieństwo że jakikolwiek spin się zmieni. Dodatkowo dla uproszczenia, zakładamy że tylko
    # spiny o minimalnej luce mogą wejść w stan wzbudzony. Rozwiązujemy:
    #   0.01 ~ #minimal_gaps exp(- cold_beta min_i min_delta_energy_i)
    # gdzie #minimal_gaps to ilość przypadków o minimalnej luce energetycznej
    # min_delta_energy_i = min_i(min_delta_energy_i).
    # rozwiązaniem jest cold_beta = log(#minimal_gaps/0.01) / min_delta_energy czyli:
    # T =  min_delta_energy / log(#minimal_gaps/0.01)

    h_non_zero = h_abs[np.where(h_abs != 0)]

    def J_non_zero(A):
        return A[np.where(A != 0)]

    values_array = [min(h_non_zero[i], np.min(J_non_zero(J_abs[i, :]))) for i in range(len(h_non_zero))]
    if values_array:
        min_effective_field = min(values_array)
        cold = (2 * min_effective_field) / np.log(1 / 0.01)
    else:
        cold = 1

    return hot, cold


def simulated_annealing(J, h, num_steps: int, temp_range: Optional[tuple] = None, schedule: str = "linear"):
    # inicjalizacja
    n = len(h)
    solution = np.random.choice([-1, 1], size=n)
    energy = calculate_energy(J, h, solution)

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
        for s in range(n):
            delta_e = calculate_delta_e(J, h, s, solution)
            r = np.random.random()

            if acceptance_probability(delta_e, temp) >= r:
                solution[s] *= -1
                energy += delta_e

    return solution, energy


def simulated_annealing_parallel(J, h, num_steps: int, num_reads: int,
                                 temp_range: Optional[tuple] = None, schedule: str = "exponential"):
    n = len(h)
    solutions = np.random.choice([-1, 1], size=(n, num_reads))

    if not temp_range:
        T_0, T_final = get_temp(J, h)
    else:
        T_0, T_final = temp_range

    if schedule == "linear":
        schedule = np.linspace(T_0, T_final, num=num_steps, endpoint=True)

    elif schedule == "exponential":
        schedule = np.geomspace(T_0, T_final, num=num_steps, endpoint=True)

    else:
        raise ValueError("wrong schedule")

    for k in range(num_steps):
        temp = schedule[k]
        for s in range(n):
            s_vect = np.array([s for _ in range(n)])
            calculate_delta_e(J, h, s_vect, solutions)


def perform_sa_dw(sa_sampler, root_path: Path, raw_root: Path, size: str, category: str):
    df_best = pd.DataFrame(columns=["Instance", "Energy", "State"], index=None)
    best_energies = {i: inf for i in range(0, 20)}
    best_states = {}

    for num_reads in [10, 100, 1000, 10000]:
        df = pd.DataFrame(columns=["Instance", "Energy", "Time", "State"], index=None)
        instances = []
        energies = []
        states = []
        times = []

        for i in tqdm(range(0, 20), desc=f"{size} {category} {num_reads}"):
            number = f"{i + 1}".zfill(3)
            path = os.path.join(root_path, size, category, "instances", f"{number}_sg.txt")
            J, h = read_instance_dict(path)

            start = time()
            sampleset = sa_sampler.sample_ising(h, J, num_reads=num_reads)
            end = time()
            elapsed = end - start
            energy = sampleset.first.energy

            temp_df = sampleset.to_pandas_dataframe(sample_column=True)
            save_raw_data(temp_df, raw_root, size, category, f"SA_{size}_{category}_{num_reads}.csv")

            state = ""
            for j in sampleset.first.sample.values():
                state += str(j) + ","

            state = state[:-1]

            instances.append(number)
            energies.append(energy)
            times.append(elapsed)
            states.append(state)

            if best_energies[i] > energy:
                best_energies[i] = energy
                best_states[i] = state

        df["Instance"] = instances
        df["Energy"] = energies
        df["Time"] = times
        df["State"] = states

        df.to_csv(os.path.join(root_path, size, category, f"SA_{num_reads}.csv"))
    df_best["Instance"] = instances
    df_best["Energy"] = list(best_energies.values())
    df_best["State"] = list(best_states.values())


def perform_sa_square(root_path: Path, raw_root: Path):
    df_best = pd.DataFrame(columns=["Instance", "Energy", "State", "Time"], index=None)
    best_energies = {i: inf for i in range(0, 10)}
    best_states = {}

    for num_reads in [10, 100, 1000, 10000]:
        df = pd.DataFrame(columns=["Instance", "Energy", "Time", "State"], index=None)
        instances = []
        energies = []
        states = []
        times = []

        for i in tqdm(range(0, 10), desc=f"Square {num_reads}"):
            number = f"{i + 1}".zfill(3)
            path = os.path.join(root_path, "instances", f"{number}.txt")
            J, h = read_instance_dict(path)

            tmp_energies = []
            tmp_states = []

            start = time()
            for reads in range(num_reads):
                state, energy = simulated_annealing(J, h, 100, schedule="exponential")
            end = time()
            elapsed = end - start


            temp_df = sampleset.to_pandas_dataframe(sample_column=True)
            save_raw_data_square(temp_df, raw_root, f"SA_{num_reads}.csv")

            state = ""
            for j in sampleset.first.sample.values():
                state += str(j) + ","

            state = state[:-1]

            instances.append(number)
            energies.append(energy)
            times.append(elapsed)
            states.append(state)

            if best_energies[i] > energy:
                best_energies[i] = energy
                best_states[i] = state

        df["Instance"] = instances
        df["Energy"] = energies
        df["Time"] = times
        df["State"] = states

        df.to_csv(os.path.join(root_path, f"SA_{num_reads}.csv"))
    df_best["Instance"] = instances
    df_best["Energy"] = list(best_energies.values())
    df_best["State"] = list(best_states.values())
    df_best.to_csv(os.path.join(root_path, "SA_best.csv"), index_label=False)


if __name__ == '__main__':
    path = os.path.join(SQUARE1_ROOT, "instances", "001.txt")
    J, h = read_instance(path)
    simulated_annealing_parallel(J, h, 100, 100)



