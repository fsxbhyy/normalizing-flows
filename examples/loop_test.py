import unittest
import torch
from loop_pool import LoopPool  # Assuming the LoopPool class is in a module named your_module
import numpy as np

class TestLoopPool(unittest.TestCase):
    def test_loop_pool_initialization_and_indexing(self):
        loopbasis = [
            [1.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, -1.0, 0.0]
        ]
        loop_pool = LoopPool('K', 3, 4, loopbasis)
        self.assertEqual(loop_pool.size(), len(loopbasis))
        self.assertTrue(np.allclose(loop_pool[0], np.array([1.0, 1.0, 0.0, 0.0])))
        self.assertTrue(np.allclose(loop_pool[-1], np.array([1.0, 0.0, -1.0, 0.0])))

        loop_pool[1] = np.array([1.0, 0.0, -1.0, 0.0])
        self.assertTrue(np.allclose(loop_pool[1], np.array([1.0, 0.0, -1.0, 0.0])))

    def test_loop_pool_append_and_update(self):
        dim, N = 3, 4
        loop_pool = LoopPool('K', dim, N)

        basis1 = [1.0, 0.0, 0.0, 1.0]
        basis2 = [1.0, 1.0, 0.0, 0.0]
        basis3 = [1.0, 0.0, -1.0, 1.0]

        idx1 = loop_pool.append_basis(basis1)
        idx2 = loop_pool.append_basis(basis2)
        idx3 = loop_pool.append_basis(basis2)
        idx4 = loop_pool.append_basis(basis1)
        idx5 = loop_pool.append_basis(basis3)

        self.assertEqual(loop_pool.size(), 3)
        self.assertEqual(idx1, idx4)
        self.assertEqual(idx2, idx3)

        varK = np.random.rand(dim, N)
        loop_pool.update(varK)

        self.assertTrue(np.allclose(loop_pool.loop(0), np.dot(varK, np.array(basis1))))
        self.assertTrue(np.allclose(loop_pool.loop(1), np.dot(varK, np.array(basis2))))
        self.assertTrue(np.allclose(loop_pool.loop(2), np.dot(varK, np.array(basis3))))

if __name__ == '__main__':
    unittest.main()