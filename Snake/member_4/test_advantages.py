import unittest
import numpy as np
from advantages import compute_gae

class AdvantageTests(unittest.TestCase):
    def test_terminal_does_not_bootstrap(self):
        a,r=compute_gae([2],[5],[True],100)
        np.testing.assert_allclose(a,[-3]);np.testing.assert_allclose(r,[2])
    def test_rollout_cut_bootstraps(self):
        a,r=compute_gae([1],[2],[False],10,gamma=.9)
        np.testing.assert_allclose(a,[8]);np.testing.assert_allclose(r,[10])
    def test_reset_does_not_leak(self):
        a,r=compute_gae([2,999],[0,0],[True,True],0)
        np.testing.assert_allclose(a,[2,999])
    def test_lambda_one_monte_carlo(self):
        a,r=compute_gae([1,2],[0,0],[False,True],0,gamma=.9,lam=1)
        np.testing.assert_allclose(r,[2.8,2],rtol=1e-6)
    def test_lambda_zero_td(self):
        a,r=compute_gae([1,2],[3,4],[False,True],0,gamma=.9,lam=0)
        np.testing.assert_allclose(a,[1.6,-2],rtol=1e-6)

if __name__=='__main__':unittest.main()
