import os
import pandas as pd

from dwave.samplers import SimulatedAnnealingSampler
from src.utils import PEGASUS_ROOT, ZEPHYR_ROOT, read_instance_dict
from tqdm import tqdm

sampler = SimulatedAnnealingSampler()

if __name__ == '__main__':

    for size in ["P8", "P16"]:
        for instance_class in ["CBFM-P", "RCO", "RAU"]:
            df = pd.DataFrame(columns=["Instance", "Energy", "State"], index=None)
            instances = []
            energies = []
            states = []
            for i in tqdm(range(0, 20), desc=f"{size} {instance_class}"):
                number = f"{i+1}".zfill(3)
                path = os.path.join(PEGASUS_ROOT, size, instance_class, "instances", f"{number}_sg.txt")
                J, h = read_instance_dict(path)

                sampleset = sampler.sample_ising(h, J, num_reads=1000)
                instances.append(number)
                energies.append(sampleset.first.energy)

                state = ""
                for j in sampleset.first.sample.values():
                    state += str(j) + ","

                state = state[:-1]
                states.append(state)

            df["Instance"] = instances
            df["Energy"] = energies
            df["State"] = states

            df.to_csv(os.path.join(PEGASUS_ROOT, size, instance_class, "SA.csv"), index=False)

