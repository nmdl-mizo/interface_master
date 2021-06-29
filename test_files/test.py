#!/usr/bin/env python
# -*- coding:utf-8
import unittest
import numpy as np
from interfacemaster.interface_generator import core, convert_vector_index, get_primitive_hkl


class Tests(unittest.TestCase):
    def test_bicrystal_for_known_GB(self):
        my_interface = core('CIS-exp.cif','CIS-exp.cif')
        axis = convert_vector_index(my_interface.conv_lattice_1, my_interface.lattice_1, [2,-2,1])
        my_interface.parse_limit(du=1e-2, S=1e-2, sgm1=100, sgm2=100, dd=1e-2)
        my_interface.search_one_position(axis, 180, 1, 0.01)
        hkl = get_primitive_hkl(np.array([1, -1, 2]), my_interface.conv_lattice_1, my_interface.lattice_1)
        self.assertIsNone(np.testing.assert_array_equal(hkl, np.array([1, -1, 1])))
        my_interface.compute_bicrystal(hkl, orthogonal=True, lim=50, tol=1e-2)
        my_interface.get_bicrystal(xyz_1=[3,1,1], xyz_2=[3,1,1], dydz=my_interface.CNID[:,1])

    def test_get_primitive_hkl(self):
        hkl = np.array([1, -1, 2])
        C_lattice = np.array([
            [5.78100000e+00, 9.29655704e-16, 0.00000000e+00],
            [0.00000000e+00, 5.78100000e+00, 0.00000000e+00],
            [3.53984157e-16, 3.53984157e-16, 1.16422000e+01]
        ])
        P_lattice = np.array([
            [ 5.78100000e+00,  9.29655704e-16, -2.89050000e+00],
            [-0.00000000e+00,  5.78100000e+00, -2.89050000e+00],
            [-0.00000000e+00,  3.53984157e-16,  5.82110000e+00]
        ])
        result = get_primitive_hkl(hkl, C_lattice, P_lattice)
        self.assertIsNone(np.testing.assert_array_equal(result, np.array([1, -1, 1])))


if __name__ == "__main__":
    unittest.main()