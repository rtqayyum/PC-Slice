"""Pure scoring tests for exploratory scheduler ranking."""
import unittest
import numpy as np
from src.surgical_scheduler_scoring import progress_weighted_cost


class SchedulerScoreTests(unittest.TestCase):
    def test_progress_is_preferred_with_equal_traffic(self):
        tip=np.array([0.0,0.0,0.0])
        target=np.array([0.03,0.0,0.0])
        poor=np.array([[0.001,0,0],[0.002,0,0]])
        good=np.array([[0.01,0,0],[0.029,0,0]])
        s0,d0=progress_weighted_cost(400,30,poor,target,tip,0.004,2.)
        s1,d1=progress_weighted_cost(400,30,good,target,tip,0.004,2.)
        self.assertLess(s1,s0)
        self.assertLess(d1,d0)

    def test_zero_weight_keeps_original_cost(self):
        a,d=progress_weighted_cost(258,20,np.array([[0.0,0,0]]),
                                   np.array([.02,0,0]),np.zeros(3),.004,0.)
        self.assertAlmostEqual(a,258/20)
        self.assertGreaterEqual(d,0.)

    def test_invalid_inputs_rejected(self):
        with self.assertRaises(ValueError):
            progress_weighted_cost(258,10,np.array([1.,2.,3.]),
                                   np.zeros(3),np.zeros(3),.004,2.)


if __name__ == "__main__":
    unittest.main()
