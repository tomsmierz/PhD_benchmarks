import os

import matplotlib.pyplot as plt
import pandas as pd

from src.utils import PEGASUS_ROOT, ZEPHYR_ROOT, Path, all_pegasuses



def plot_energies():
    ...


def aggregate_data(root_path: Path, size: str, category: str):
    data_path = os.path.join(root_path, size, category)

    main_df = pd.DataFrame()
    main_df["Instance"] = list(range(1, 21))
    for filename in os.listdir(data_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(data_path, filename)
            name, _ = filename.split(".")
            df = pd.read_csv(file_path)
            energies = df["Energy"].tolist()
            main_df[name] = energies

    save_path = os.path.join(root_path, "aggregated", f"{size}_{category}.csv")
    main_df.to_csv(save_path, index=False)


if __name__ == '__main__':

    aggregate_data(PEGASUS_ROOT, size="P16", category="RCO")
