import unittest
import numpy as np

from src.utils import read_instance, read_instance_dict, calculate_energy, calculate_energy_dict


class InstanceRead(unittest.TestCase):
    def test_read_instance(self):
        J, h = read_instance("small_instance.txt")
        J_dict, h_dict = read_instance_dict("small_instance.txt")

        state = np.random.choice([-1, 1], size=5)
        e1 = calculate_energy_dict(J_dict, h_dict, state)
        e2 = calculate_energy(J, h, state)
        self.assertAlmostEqual(e1, e2)  # add assertion here


if __name__ == '__main__':
    unittest.main()
