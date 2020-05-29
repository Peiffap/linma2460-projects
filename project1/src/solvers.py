#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
solvers.py

Author: Gilles Peiffer
Date: 2020-05-29

This file contains the numerical methods
needed for the first exercise of LINMA2460.
"""

import numpy as np


def solve(n, m, set_type, parameters, function, eps, residue, method,
          maximum_iterations):
    """
    Solve an optimization problem using either the gradient or the optimal method.

    Either numerical method can be used to solve the problem,
    depending on which one is specified.

    Parameters
    ----------
    n : int
        Dimension of the problem domain.
    m : int
        Number of active constraints.
        This parameter is only used if 'set' is 'box' or 'simplex'.
    set_type : {'ball', 'box', 'simplex'}
        Type of set being used.
    parameters : dict
        Parameters of the set being used:
         * if set_type is 'ball', this contains R, the radius of the ball;
         * if set_type is 'box', this contains a and b (a[i] <= b[i]), the bounds of the box;
         * if set_type is 'simplex', this contains the parameter p in the simplex definition.
    function : dict
        Contains the parameters of the objective function.
    eps : float
        Desired accuracy of the solution.
    residue : float
        Distance between initial solution and minimum.
    method : {'gradient', 'optimal'}
        Type of numerical method being used.
    maximum_iterations : int
        Maximal number of iterations.

    Returns
    -------
    x : ndarray
        Iterates of the method.
    distances : ndarrray
        Distances between intermediate solutions and optimal solution.
    values : ndarray
        Values of the objective function at every iteration.
    """

    rng = np.random.default_rng(seed=69)

    x_opt = np.zeros((n, ))
    x_init = np.zeros((n, ))

    # Get parameters based on set type.
    if set_type == 'ball':
        # Radius of the ball.
        radius = parameters['R']

        # Randomly generate optimal solution inside ball.
        r_opt = radius * rng.random()
        x_opt = rng.random((n, ))
        x_opt *= r_opt / np.linalg.norm(x_opt)

        # Generate initial point at a distance residue from x_opt inside the ball.
        # A random point at the correct distance is generated,
        # until one falls inside the ball.
        shift = -1 + 2 * rng.random((n, ))
        x_init = x_opt + residue * shift / np.linalg.norm(shift)
        while np.linalg.norm(x_init) > radius:
            shift = -1 + 2 * rng.random((n, ))
            x_init = x_opt + residue * shift / np.linalg.norm(shift)
    elif set_type == 'box':
        # Parameters of the box.
        a = parameters['a']
        b = parameters['b']

        # Randomly generate optimal solution inside box.
        x_opt = (b - a) * rng.random((n, )) + a
        if m > 0:
            # Generate a random set of indices
            # at which to constrain the problem.
            # Half of them are set to a[i], the other half to b[i].
            permutation = rng.permutation(n)[:m]
            x_opt[permutation[:m // 2]] = a[permutation[:m // 2]]
            x_opt[permutation[m // 2:]] = b[permutation[m // 2:]]

        # Generate initial point at a distance residue from x_opt inside the box.
        # A random point at the correct distance is generated,
        # until one falls inside the box.
        shift = -1 + 2 * rng.random((n, ))
        # Help by always moving away from borders on constrained parts.
        shift[x_opt == a] = np.abs(shift[x_opt == a])
        shift[x_opt == b] = np.abs(shift[x_opt == b])
        x_init = x_opt + residue * shift / np.linalg.norm(shift)
        while not np.all(np.less_equal(a, x_init)) or not np.all(
                np.less_equal(x_init, b)):
            shift = -1 + 2 * rng.random((n, ))
            # Help by always moving away from borders on constrained parts.
            shift[x_opt == a] = np.abs(shift[x_opt == a])
            shift[x_opt == b] = -np.abs(shift[x_opt == b])
            x_init = x_opt + residue * shift / np.linalg.norm(shift)
    elif set_type == 'simplex':
        # Parameter of the simplex.
        p = parameters['p']

        # Randomly generate optimal solution inside simplex.
        permutation = rng.permutation(n)[:n - m]
        x_opt = np.zeros((n, ))
        x_opt[permutation] = rng.random((n - m), )
        x_opt[permutation] = p * x_opt[permutation] / np.sum(
            x_opt[permutation])

        # Generate initial point at a distance residue from x_opt inside the simplex.
        # A random point with zero sum is generated and added to x_opt.
        shift = rng.random((n, ))
        shift -= np.mean(shift)
        shift *= residue / np.linalg.norm(shift)
        indices_opt = np.argsort(x_opt)
        shift[::-1].sort()
        x_init[indices_opt] = x_opt[indices_opt] + shift
        while not np.all(x_init >= 0):
            shift = rng.random((n, ))
            shift -= np.mean(shift)
            shift *= residue / np.linalg.norm(shift)
            shift[::-1].sort()
            x_init[indices_opt] = x_opt[indices_opt] + shift

    # Number of iterations.
    it = 0

    # Iterates, starting from initial position
    x = np.zeros((maximum_iterations + 1, n))
    x[0] = x_init

    # Parameters of the objective function.
    alpha = function['alpha']
    beta = function['beta']
    gamma = function['gamma']

    # Lipschitz-continuity coefficients.
    L = alpha + 4 * beta + gamma
    mu = alpha

    # Set initial parameters for the optimal method.
    y = None
    beta_opt = None
    if method == 'optimal':
        y = np.zeros((maximum_iterations + 1, n))
        y[0] = x[0]
        beta_opt = (np.sqrt(L) - np.sqrt(mu)) / (np.sqrt(L) + np.sqrt(mu))

    def f(x):
        """
        Compute the value of the function at a point x.

        The function is built as a closure using the values of the parameters
        and of the optimal solution.
        It is kept in a numerically stable version, as mentioned in the pdf.

        Parameters
        ----------
        x : ndarray
            Point at which to evaluate f.

        Returns
        -------
        fx: float
            Value of f(x).
        """
        xdif = x - x_opt
        fx = alpha / 2 * np.linalg.norm(xdif)**2  # First term.
        fx += beta / 2 * (xdif[0]**2 + np.sum(
            (xdif[:-1] - xdif[1:])**2) + xdif[-1]**2)  # Second term.

        def f2(x):
            """
            Compute f2 in definition of objective function.

            The computation is done so as to preserve numerical stability.

            Parameters
            ----------
            x : ndarray
                Point at which to compute the function value.

            Returns
            -------
            fx : float
                Value of f2(x).
            """
            delta = np.max(x)
            return delta + np.log(np.sum(np.exp(x - delta)))

        # Compute f2'(x_opt) more efficiently.
        tmp = np.exp(x_opt - np.max(x_opt))

        fx += gamma * (f2(x) - f2(x_opt) - np.dot(tmp / np.sum(tmp), xdif)
                       )  # Third term.

        return fx

    def fp(x):
        """
        Compute value of f' at point x.

        Parameters
        ----------
        x : ndarray
            Point at which we evaluate f'.

        Returns
        -------
        fpx : float
            Value of f'(x).
        """
        # For more efficient computation.
        xdif = x - x_opt
        tmp_x_opt = np.exp(x_opt - np.max(x_opt))
        tmp_x = np.exp(x - np.max(x))

        aux = np.zeros((n, ))
        aux[:-1] += xdif[1:]
        aux[1:] += xdif[:-1]

        # Derivative of f evaluated at x.
        fpx = ((alpha + 2 * beta) * xdif - beta * aux + gamma *
               (tmp_x / np.sum(tmp_x) - tmp_x_opt / np.sum(tmp_x_opt)))

        return fpx

    # Distance between iterate and optimal solution.
    distances = np.zeros((maximum_iterations + 1, ))
    distances[0] = residue

    # Values of the objective function.
    values = np.zeros((maximum_iterations + 1, ))
    values[0] = f(x[0])

    it = 1
    # Iterate while convergence hasn't been reached.
    # Optimal value of the function is 0, hence we can simply compare to eps.
    while np.abs(values[it - 1]) > eps and 0 < it <= maximum_iterations:
        # Decide which method to use.

        # Compute next iterate.
        if method == 'gradient':
            x[it] = x[it - 1] - 1 / L * fp(x[it - 1])
        elif method == 'optimal':
            x[it] = y[it - 1] - 1 / L * fp(y[it - 1])

        # Project iterate if needed.
        if set_type == 'ball':
            radius = parameters['R']

            norm = np.linalg.norm(x[it])
            if norm > radius:
                x[it] *= radius / norm
        elif set_type == 'box':
            a = parameters['a']
            b = parameters['b']

            # Project components outside of the box on the box.
            tmpa = x[it] < a
            tmpb = x[it] > b
            x[it][tmpb] = b[tmpb]
            x[it][tmpa] = a[tmpa]
        elif set_type == 'simplex':
            p = parameters['p']

            # Project onto simplex;
            # https://arxiv.org/abs/1101.6081v2

            # If there is only one point in the domain, it must be p.
            if n == 1:
                x[it] = p
            else:
                x_sorted = np.sort(x[it])
                i = n - 1

                t_opt = None
                while True:
                    ti = (np.sum(x_sorted[i:]) - p) / (n - i)

                    if ti >= x_sorted[i]:
                        t_opt = ti
                        break

                    i = i - 1
                    if i == 0:
                        t_opt = np.mean(x_sorted) - p / n
                        break

                np.maximum(x[it] - t_opt, 0, out=x[it])

        # Update y if necessary.
        if method == 'optimal':
            y[it] = x[it] + beta_opt * (x[it] - x[it - 1])

        # Store values.
        distances[it] = np.linalg.norm(x[it] - x_opt)
        values[it] = f(x[it])

        it += 1

    # Prune empty preallocated space.
    return x[:it], distances[:it], values[:it]
