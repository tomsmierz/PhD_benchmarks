import os
import pandas as pd

from dwave.samplers import SimulatedAnnealingSampler
from src.utils import (PEGASUS_ROOT, ZEPHYR_ROOT, read_instance_dict, all_pegasuses, Path, RAW_PEGASUS_ROOT,
                       save_raw_data)
from tqdm import tqdm
from time import time

sampler = SimulatedAnnealingSampler()


def perform_sa(sa_sampler, root_path: Path, raw_root: Path, size: str, category: str):
    df_best = pd.DataFrame(columns=["Instance", "Energy", "State"], index=None)
    instances = []
    energies = []
    states = []
    times = []

    for i in tqdm(range(0, 20), desc=f"{size} {category}"):
        number = f"{i + 1}".zfill(3)
        path = os.path.join(root_path, size, category, "instances", f"{number}_sg.txt")
        J, h = read_instance_dict(path)
        instances.append(number)

        start = time()
        sampleset = sa_sampler.sample_ising(h, J, num_reads=1000)
        end = time()
        temp_df = sampleset.to_pandas_dataframe(sample_column=True)
        save_raw_data(temp_df, raw_root, size, category, f"SA_{size}_{category}_{num_reads}.csv")
        elapsed = end - start

        energies.append(sampleset.first.energy)
        times.append(elapsed)

        state = ""
        for j in sampleset.first.sample.values():
            state += str(j) + ","

        state = state[:-1]
        states.append(state)

        df_best["Instance"] = instances
        df_best["Energy"] = energies
        df_best["State"] = states


        df_best.to_csv(os.path.join(root_path, size, category, "SA.csv"), index=False)


if __name__ == '__main__':
    all_pegasuses(perform_sa, sampler, PEGASUS_ROOT)



