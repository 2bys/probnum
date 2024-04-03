from typing import Tuple

import numpy as np
import pytest
import pytest_cases
import scipy.sparse

import probnum as pn
from probnum.problems.zoo.linalg import random_spd_matrix

def case_complex_matrix_vector_multiplication():
    matrix = np.array([[1+2j, 3+4j], [5+6j, 7+8j]], dtype=np.complex128)
    linop = pn.linops.Matrix(matrix)
    vector = np.array([1+1j, 2+2j], dtype=np.complex128)
    expected_result = np.dot(matrix, vector)
    return linop, vector, expected_result