import unittest
import time

import numpy as np


"""@NotAlways
class NumpyAssignTestCase(unittest.TestCase):

    def setUp(self):
        self.iter_count = 100

    def test_assign_slower_than_copy_with_small(self):
        x = np.random.uniform(-1.0, 1.0, (256,))
        y = x.copy()

        t = time.perf_counter_ns()
        for _ in range(self.iter_count):
            x += np.random.uniform(-1.0, 1.0, x.shape)
            y[:] = x
        assign_t = time.perf_counter_ns() - t

        t = time.perf_counter_ns()
        for _ in range(self.iter_count):
            x += np.random.uniform(-1.0, 1.0, x.shape)
            y = x.copy()
        copy_t = time.perf_counter_ns() - t

        self.assertLess(assign_t, copy_t)

    def test_assign_faster_than_copy_with_big(self):
        x = np.random.uniform(-1.0, 1.0, (256*256,))
        y = x.copy()

        t = time.perf_counter_ns()
        for _ in range(self.iter_count):
            x += np.random.uniform(-1.0, 1.0, x.shape)
            y[:] = x
        assign_t = time.perf_counter_ns() - t

        t = time.perf_counter_ns()
        for _ in range(self.iter_count):
            x += np.random.uniform(-1.0, 1.0, x.shape)
            y = x.copy()
        copy_t = time.perf_counter_ns() - t

        self.assertLess(assign_t, copy_t)
"""


if __name__ == "__main__":
    unittest.main()
