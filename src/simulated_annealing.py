import os
import pandas as pd

from dwave.samplers import SimulatedAnnealingSampler
from src.utils import (PEGASUS_ROOT, ZEPHYR_ROOT, read_instance_dict, all_pegasuses, Path, RAW_PEGASUS_ROOT,
                       save_raw_data, SQUARE1_ROOT, RAW_SQUARE1_ROOT, save_raw_data_square, RAW_SQUARE2_ROOT, SQUARE2_ROOT)
from tqdm import tqdm
from time import time
from math import inf

sampler = SimulatedAnnealingSampler()


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


def perform_sa_square(sa_sampler, root_path: Path, raw_root: Path):
    df_best = pd.DataFrame(columns=["Instance", "Energy", "State"], index=None)
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

            start = time()
            sampleset = sa_sampler.sample_ising(h, J, num_reads=num_reads)
            end = time()
            elapsed = end - start
            energy = sampleset.first.energy

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
    perform_sa_square(sampler, SQUARE2_ROOT, RAW_SQUARE2_ROOT)



