import os

import pandas as pd
import numpy as np

from collections import namedtuple
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.join(ROOT, "data", "rich_k_4")
ground_state_sum = namedtuple("ground_state_sum", ["N", "dwave", "sim"])



if __name__ == '__main__':

    data = pd.DataFrame()
    idx = 0
    for file_name in tqdm(os.listdir(DATA_ROOT)):
        data_row = {}
        file_path = os.path.join(DATA_ROOT, file_name)

        with open(file_path, "r") as f:
            df = pd.read_csv(file_path,sep=";")

        tau = df["annealing_time"][0]
        sdev = df["sdev"][0]

        temp = {}
        for row in df.itertuples():
            number = row.instance_number
            sim = sum(eval(row.sim_gr_state_prob))
            dwave = sum(eval(row.dwave_gr_state_prob))
            tvd = row.tvd
            fidelity = row.fidelity
            temp[number] = (dwave, sim, tvd, fidelity)

        arr = np.asarray(list(temp.values()))

        meds = tuple(np.median(arr, axis=0))
        mins = tuple(arr.min(axis=0))
        maxs = tuple(arr.max(axis=0))

        data.at[idx, "annealing_time"] = tau
        data.at[idx, "sdev"] = sdev

        data.at[idx, "median_dwave_gs_prob"] = meds[0]
        data.at[idx, "min_dwave"] = mins[0]
        data.at[idx, "max_dwave"] = maxs[0]

        data.at[idx, "median_sim_gs_prob"] = meds[1]
        data.at[idx, "min_sim"] = mins[1]
        data.at[idx, "max_sim"] = maxs[1]

        data.at[idx, "median_tvd"] = meds[2]
        data.at[idx, "min_tvd"] = mins[2]
        data.at[idx, "max_tvd"] = maxs[2]

        data.at[idx, "median_fidelity"] = meds[3]
        data.at[idx, "min_fidelity"] = mins[3]
        data.at[idx, "max_fidelity"] = maxs[3]
        idx +=1

    df_00 = data[data["sdev"] == 0.00].copy()
    df_01 = data[data["sdev"] == 0.01].copy()
    df_03 = data[data["sdev"] == 0.03].copy()
    df_10 = data[data["sdev"] == 0.10].copy()

    df_00 = df_00.sort_values("annealing_time").reset_index(drop=True)
    df_01 = df_01.sort_values("annealing_time").reset_index(drop=True)
    df_03 = df_03.sort_values("annealing_time").reset_index(drop=True)
    df_10 = df_10.sort_values("annealing_time").reset_index(drop=True)

    df_00.to_csv(os.path.join(ROOT, "results", "sim_0.00.csv"), index=False)
    df_01.to_csv(os.path.join(ROOT, "results", "sim_0.01.csv"), index=False)
    df_03.to_csv(os.path.join(ROOT, "results", "sim_0.03.csv"), index=False)
    df_10.to_csv(os.path.join(ROOT, "results", "sim_0.10.csv"), index=False)
