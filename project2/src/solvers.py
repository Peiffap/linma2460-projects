#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
solvers.py

Author: Gilles Peiffer
Date: 2020-06-07

This file contains the numerical methods
needed for the second exercise of LINMA2460.
"""

import numpy as np
import numba


#@numba.jit(nopython=True)
def solve(n, function, rho, method, eps, maximum_iterations):
    """
    Solve an optimization problem using either the subgradient or the ellipsoid method.

    Either numerical method can be used to solve the problem,
    depending on which one is specified.

    Parameters
    ----------
    n : int
        Dimension of the problem domain.
    function : dict
        Parameters alpha and beta of the objective function.
    rho : float
        Initial distance from the minimum.
    method : {'subgradient', 'ellipsoid'}
        Method to use to solve the problem.
    eps : float
        Required accuracy.
    maximum_iterations : int
        Maximal number of iterations.

    Returns
    -------
    x : ndarray
        Iterates of the optimization process.
    vals : ndarray
        Function values at each iteration.
    """

    rng = np.random.default_rng(seed=69)
    it = 0

    # Start iterating at distance rho from x_opt = 0.
    x = np.zeros((maximum_iterations + 1, n))
    x[0] = rng.random((n, ))
    x[0] *= rho / np.linalg.norm(x[0])

    alpha = function['alpha']
    beta = function['beta']

    def f(x):
        """
        Compute the value of the objective function at x.

        Parameters
        ----------
        x : ndarray
            The point at which to evaluate f.

        Returns
        -------
        fx : float
            The value of the objective function at x, f(x).
        """

        xabs = np.abs(x)
        return alpha * np.sum(xabs[:-1]) + beta * (np.max(xabs) - x[0])

    vals = np.zeros((maximum_iterations + 1, ))
    vals[0] = f(x[0])

    H = None
    constants = None

    # Compute constants for the ellipsoid method.
    if method == 'ellipsoid':
        H = np.identity(n) * rho**2
        constants = [1 / (n + 1), n**2 / (n**2 - 1), 2 / (n + 1)]

    while vals[it] > eps and it < maximum_iterations:
        it += 1

        # Compute subgradient.
        g = np.sign(x[it - 1])
        g[-1] = 0
        g *= alpha
        xabs = np.abs(x[it - 1])
        m = np.max(xabs)
        ind = np.argwhere(xabs == m)
        g[ind] += beta * np.sign(x[it - 1, ind])
        g[0] -= beta

        if method == 'subgradient':
            x[it] = x[it - 1] - rho / np.sqrt(it) * g / np.linalg.norm(g)
        elif method == 'ellipsoid':
            tmp1 = H @ g
            tmp2 = g @ tmp1
            x[it] = x[it - 1] - tmp1 * constants[0] / np.sqrt(tmp2)
            H = constants[1] * (H - constants[2] / tmp2 * np.outer(tmp1, tmp1))

        vals[it] = f(x[it])

    return x[:it + 1], vals[:it + 1]
