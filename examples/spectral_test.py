import numpy as np
import unittest
from scipy.optimize import approx_fprime

# Assuming the kernel functions are defined in `spectral.py`
from spectral import kernelFermiT, kernelFermiT_dω, kernelFermiT_dω2, kernelFermiT_dω3, kernelFermiT_dω4, kernelFermiT_dω5

class TestKernelFermiT(unittest.TestCase):
    def test_dω(self):
        betas = [1.0, 10.0, 100.0]
        w = 1.0
        epsilon = 1e-5
        atol = 1e-3

        for beta in betas:
            taus = np.array([0.0, 0.01, 0.1, 0.5, 0.9, 0.99, 1.0])*beta
            for tau in taus:
                if tau == beta:
                    tau -= 1e-6  # Adjust to keep τ within (-β, β]
                print(f"Test kernelFermiT w-derivative with w={w}, tau={tau}, beta={beta}")

                # Testing the first derivative
                analytic = kernelFermiT_dω(tau, w, beta)
                numeric = approx_fprime(np.array([w]), lambda w: kernelFermiT(tau, w[0], beta), epsilon)[0]
     
                self.assertTrue(np.abs(analytic - numeric) < atol, "First derivative test failed")

                # Testing the second derivative
                analytic = kernelFermiT_dω2(tau, w, beta)
                numeric = approx_fprime(np.array([w]), lambda w: kernelFermiT_dω(tau, w[0], beta), epsilon)[0]
                self.assertTrue(np.abs(analytic - numeric) < atol, "Second derivative test failed")

                # Testing the third derivative
                analytic = kernelFermiT_dω3(tau, w, beta)
                numeric = approx_fprime(np.array([w]), lambda w: kernelFermiT_dω2(tau, w[0], beta), epsilon)[0]
                self.assertTrue(np.abs(analytic - numeric) < atol, "Third derivative test failed")

                # Testing the fourth derivative
                analytic = kernelFermiT_dω4(tau, w, beta)
                numeric = approx_fprime(np.array([w]), lambda w: kernelFermiT_dω3(tau, w[0], beta), epsilon)[0]
                self.assertTrue(np.abs(analytic - numeric) < atol, "Fourth derivative test failed")

                # Testing the fifth derivative
                analytic = kernelFermiT_dω5(tau, w, beta)

                numeric = approx_fprime(np.array([w]), lambda w: kernelFermiT_dω4(tau, w[0], beta), epsilon)[0]
                print(analytic, numeric)
                self.assertTrue(np.abs(analytic - numeric) < atol, "Fifth derivative test failed")

if __name__ == "__main__":
    unittest.main()