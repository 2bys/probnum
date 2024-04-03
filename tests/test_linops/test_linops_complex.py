import pathlib
from typing import Optional, Tuple, Union

import numpy as np
import pytest
import pytest_cases

import probnum as pn

case_modules = [
    ".test_linops_cases." + path.stem
    for path in (pathlib.Path(__file__).parent / "test_linops_cases").glob("*_cases.py")
]

@pytest.fixture(
    scope="function",
    params=[pytest.param(seed, id=f"seed{seed}") for seed in [1, 42, 256]],
)
def rng(request) -> np.random.Generator:
    return np.random.default_rng(seed=request.param)

# complex tests @2bys
@pytest_cases.parametrize_with_cases("linop,matrix", cases=case_modules)
def test_complex_matvec(
    linop: pn.linops.LinearOperator,
    matrix: np.ndarray,
    rng: np.random.Generator,
):
    vec = rng.normal(size=linop.shape[1]) + 1j * rng.normal(size=linop.shape[1])

    linop_matvec = linop @ vec
    matrix_matvec = matrix @ vec

    np.testing.assert_allclose(linop_matvec, matrix_matvec, atol=1e-12)

@pytest_cases.parametrize_with_cases("linop,matrix", cases=case_modules)
def test_conjugate_transpose(
    linop: pn.linops.LinearOperator,
    matrix: np.ndarray,
):
    matrix_hermitian = np.conj(matrix.T)
    linop_hermitian = linop.H.todense()

    np.testing.assert_allclose(linop_hermitian, matrix_hermitian, atol=1e-12)

@pytest_cases.parametrize_with_cases("linop,matrix", cases=case_modules)
def test_complex_solve_vector(
    linop: pn.linops.LinearOperator,
    matrix: np.ndarray,
    rng: np.random.Generator,
):
    vec = rng.normal(size=linop.shape[0]) + 1j * rng.normal(size=linop.shape[0])

    if linop.is_square:
        try:
            np_linalg_solve = np.linalg.solve(matrix, vec)
            linop_solve = linop.solve(vec)

            np.testing.assert_allclose(linop_solve, np_linalg_solve, atol=1e-12)
        except np.linalg.LinAlgError:
            with pytest.raises(np.linalg.LinAlgError):
                linop.solve(vec)

@pytest_cases.parametrize_with_cases("linop,matrix", cases=case_modules)
def test_complex_eigvals(
    linop: pn.linops.LinearOperator,
    matrix: np.ndarray,
):
    if linop.is_square:
        linop_eigvals = linop.eigvals()
        _, matrix_eigvals = np.linalg.eig(matrix)

        np.testing.assert_allclose(np.sort(linop_eigvals), np.sort(matrix_eigvals), atol=1e-12)
    else:
        with pytest.raises(np.linalg.LinAlgError):
            linop.eigvals()

@pytest_cases.parametrize_with_cases("linop,matrix", cases=case_modules)
def test_complex_det_and_trace(
    linop: pn.linops.LinearOperator,
    matrix: np.ndarray,
):
    if linop.is_square:
        linop_det = linop.det()
        matrix_det = np.linalg.det(matrix)

        linop_trace = linop.trace()
        matrix_trace = np.trace(matrix)

        np.testing.assert_allclose(linop_det, matrix_det, atol=1e-12)
        np.testing.assert_allclose(linop_trace, matrix_trace, atol=1e-12)