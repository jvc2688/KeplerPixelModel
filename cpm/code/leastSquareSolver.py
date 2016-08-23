# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["linear_least_squares"]

import numpy as np
from scipy import linalg


def linear_least_squares(A, y, yvar=None, l2=None):
    """
        Solve a linear system as fast as possible.
        
        :param A: ``(ndata, nbasis)``
        The basis matrix.
        
        :param y: ``(ndata)``
        The observations.
        
        :param yvar:
        The observational variance of the points ``y``.
        
        :param l2:
        The L2 regularization strength. Can be a scalar or a vector (of length
        ``A.shape[1]``).
        
        """
    # Incorporate the observational uncertainties.
    if yvar is not None:
        CiA = A / yvar[:, None]
        Ciy = y / yvar[:, None]
    else:
        CiA = A
        Ciy = y
    
    # Compute the pre-factor.
    AT = A.T
    ATA = np.dot(AT, CiA)
    
    # Incorporate any L2 regularization.
    if l2 is not None:
        if np.isscalar(l2):
            l2 = l2 + np.zeros(A.shape[1])
        ATA[np.diag_indices_from(ATA)] += l2
    
    # Solve the equations overwriting the temporary arrays for speed.
    factor = linalg.cho_factor(ATA, overwrite_a=True)
    return linalg.cho_solve(factor, np.dot(AT, Ciy), overwrite_b=True)