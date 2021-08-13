# -*- coding: utf-8 -*-
# Copyright 2019-2021 The kikuchipy developers
#
# This file is part of kikuchipy.
#
# kikuchipy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# kikuchipy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with kikuchipy. If not, see <http://www.gnu.org/licenses/>.

"""Solvers for the refinement of crystal orientations and projection
centers by optimizing the similarity between experimental and simulated
patterns.
"""

from typing import Callable, Tuple

import numpy as np

from kikuchipy.indexing._refinement._objective_functions import (
    _refine_orientation_objective_function,
)
from kikuchipy.pattern._pattern import _rescale


def _refine_orientation_solver(
    pattern: np.ndarray,
    rescale: bool,
    rotation: np.ndarray,
    direction_cosines: np.ndarray,
    method: Callable,
    method_kwargs: dict,
    fixed_parameters: tuple,
    trust_region: list,
) -> Tuple[float, float, float, float]:
    """Maximize the similarity between an experimental pattern and a
    re-projected simulated pattern by optimizing the orientation (Euler
    angles) used in the re-projection.

    Parameters
    ----------
    pattern
        Experimental pattern of shape (nrows, ncols).
    rescale
        Whether pattern intensities must be rescaled to [-1, 1] and data
        type 32-bit floats.
    rotation
        Euler angles (phi1, Phi, phi2) in radians.
    direction_cosines
        Vector array of shape (nrows, ncols, 3) and data type 32-bit
        floats.
    method
        A supported :mod:`scipy.optimize` function. See `method`
        parameter in :meth:`kikuchipy.signals.EBSD.refine_orientation`.
    method_kwargs
        Keyword arguments passed to the `method` function. For the list
        of possible arguments, see the SciPy documentation.
    fixed_parameters
        Fixed parameters used in the re-projection.
    trust_region
        List of angular deviation in radians from the initial
        orientation for the three Euler angles. Only used for
        optimization methods that support bounds (excluding
        "Powell").

    Returns
    -------
    score
        Highest normalized cross-correlation score.
    phi1, Phi, phi2
        Optimized orientation (Euler angles) in radians.
    """
    if rescale:
        pattern = pattern.astype(np.float32)
        pattern = _rescale(
            pattern, imin=np.min(pattern), imax=np.max(pattern), omin=-1, omax=1
        )

    direction_cosines = direction_cosines.reshape((-1, 3))

    params = (pattern,) + (direction_cosines,) + fixed_parameters
    method_name = method.__name__

    if method_name == "minimize":
        solution = method(
            fun=_refine_orientation_objective_function,
            x0=rotation,
            args=params,
            **method_kwargs,
        )
    elif method_name == "differential_evolution":
        solution = None
    elif method_name == "dual_annealing":
        solution = None
    elif method_name == "basinhopping":
        solution = None

    score = 1 - solution.fun
    phi1 = solution.x[0]
    Phi = solution.x[1]
    phi2 = solution.x[2]

    return score, phi1, Phi, phi2
