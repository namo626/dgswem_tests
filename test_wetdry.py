import numpy as np
import os
import pytest
from util import *

case = "wetdry"
ADCIRC = "../dgswem_serial"
ADCIRC_CUDA = "../gpu"

"""
@pytest.fixture(scope="module")
def adg_solution():
    return make_solution(case, ADCIRC, "serial")
"""
@pytest.fixture(scope="module")
def true_solution():
    return read_solution(case,  "true")

@pytest.fixture(scope="module")
def cuda_solution():
    return make_solution(case, ADCIRC_CUDA, "cuda")


def test_cuda(true_solution, cuda_solution):
    np.testing.assert_allclose(cuda_solution, true_solution, rtol=1e-6)