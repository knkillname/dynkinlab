import unittest

from dynkinlab import matrices


class TestInfiniteMatrix(unittest.TestCase):
    def test_instantiation(self):
        m = matrices.InfiniteMatrix(dtype=int)
        self.assertEqual(m[0, 0], 0)
        self.assertEqual(m[1, 1], 0)
        self.assertEqual(m[1, 0], 0)
        self.assertEqual(m[0, 1], 0)

        m = matrices.InfiniteMatrix(dtype=float)
        self.assertEqual(m[0, 0], 0.0)
        self.assertEqual(m[1, 1], 0.0)
        self.assertEqual(m[1, 0], 0.0)
        self.assertEqual(m[0, 1], 0.0)

    def test_instantiation_with_data(self):
        m = matrices.InfiniteMatrix(data=[(0, 0, 1), (1, 1, 2), (1, 0, 3)], dtype=int)
        self.assertEqual(m[0, 0], 1)
        self.assertEqual(m[1, 1], 2)
        self.assertEqual(m[1, 0], 3)
        self.assertEqual(m[0, 1], 0)

    def test_infer_dtype(self):
        m = matrices.InfiniteMatrix(data=[(0, 0, 1), (1, 1, 2), (1, 0, 3), (0, 1, 4)])
        self.assertEqual(m.dtype, int)
        m = matrices.InfiniteMatrix(data=[(0, 0, 1.0), (1, 1, 2.0)])
        self.assertEqual(m.dtype, float)

    def test_setitem(self):
        m = matrices.InfiniteMatrix(dtype=int)
        m[0, 0] = 1
        m[1, 1] = 2
        m[1, 0] = 3
        m[0, 1] = 4

        self.assertEqual(m[0, 0], 1)
        self.assertEqual(m[1, 1], 2)
        self.assertEqual(m[1, 0], 3)
        self.assertEqual(m[0, 1], 4)
        self.assertEqual(m[2, 2], 0)

    def test_setitem_with_slicer(self):
        m = matrices.InfiniteMatrix(dtype=int)
        m[0, 0:2] = [1, 2]
        m[0:2, 0] = [3, 4]
        m[1:3, 1:3] = [[5, 6], [7, 8]]

        self.assertEqual(m[0, 0], 3)
        self.assertEqual(m[0, 1], 2)
        self.assertEqual(m[1, 0], 4)
        self.assertEqual(m[1, 1], 5)
        self.assertEqual(m[1, 2], 6)
        self.assertEqual(m[2, 1], 7)
        self.assertEqual(m[2, 2], 8)

    def test_setitem_with_slicer_and_step(self):
        m = matrices.InfiniteMatrix(dtype=int)
        m[0, 0:4:2] = [1, 2]
        m[0:4:2, 0] = [3, 4]
        m[1:5:2, 1:5:2] = [[5, 6], [7, 8]]

        self.assertEqual(m[0, 0], 3)
        self.assertEqual(m[0, 2], 2)
        self.assertEqual(m[2, 0], 4)
        self.assertEqual(m[1, 1], 5)
        self.assertEqual(m[1, 3], 6)
        self.assertEqual(m[3, 1], 7)
        self.assertEqual(m[3, 3], 8)
        self.assertEqual(m[2, 2], 0)

    def test_setitem_with_list(self):
        m = matrices.InfiniteMatrix(dtype=int)
        m[0, [0, 2]] = [1, 3]
        m[[0, 2], 1] = [2, 8]
        m[[1, 2], [0, 2]] = [[4, 6], [7, 9]]

        self.assertEqual(m[0, 0], 1)
        self.assertEqual(m[0, 1], 2)
        self.assertEqual(m[0, 2], 3)
        self.assertEqual(m[1, 0], 4)
        self.assertEqual(m[1, 1], 0)
        self.assertEqual(m[1, 2], 6)
        self.assertEqual(m[2, 0], 7)
        self.assertEqual(m[2, 1], 8)
        self.assertEqual(m[2, 2], 9)

    def test_getitem_with_slicer(self):
        m = matrices.InfiniteMatrix(
            data=[(0, 0, 1), (1, 1, 2), (1, 0, 3), (0, 1, 4)], dtype=int
        )

        self.assertEqual(m[0, 0:2].tolist(), [1, 4])
        self.assertEqual(m[0:2, 0].tolist(), [1, 3])
        self.assertEqual(m[1:3, 1:3].tolist(), [[2, 0], [0, 0]])

    def test_getitem_with_infinite_slicer(self):
        m = matrices.InfiniteMatrix(
            data=[(0, 0, 1), (1, 1, 2), (1, 0, 3), (0, 1, 4)], dtype=int
        )

        with self.assertRaises(IndexError):
            m[0:2, 0:]
        with self.assertRaises(IndexError):
            m[0:, 0:2]

    def test_getitem_with_slicer_and_step(self):
        m = matrices.InfiniteMatrix(
            data=[(0, 0, 1), (1, 1, 2), (1, 0, 3), (0, 1, 4)], dtype=int
        )
        self.assertEqual(m[0, 0:2:2], [1])
        self.assertEqual(m[0:2:2, 0], [1])
        self.assertEqual(m[1:3:2, 1:3:2], [[2]])

    def test_getitem_with_list(self):
        m = matrices.InfiniteMatrix(
            data=[(0, 0, 1), (1, 1, 2), (1, 0, 3), (0, 1, 4)], dtype=int
        )
        self.assertEqual(m[[0, 1, 3], 0], [1, 3, 0])
        self.assertEqual(m[0, [0, 1, 3]], [1, 4, 0])
        self.assertEqual(m[[1, 2, 4], [1, 2, 4]], [[2, 0, 0], [0, 0, 0], [0, 0, 0]])


if __name__ == "__main__":
    unittest.main()
