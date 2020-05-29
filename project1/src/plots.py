#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tikzplotlib

plt.style.use("ggplot")

from solvers import solve


def reduce(l):
    l = list(l)
    if len(l) <= 10:
        return l
    r = l[:10]

    if len(l) <= 100:
        return r + l[10::10]
    r = r + l[10:100:10]

    if len(l) <= 1000:
        return r + l[100::100]
    r = r + l[100:1000:100]

    if len(l) <= 10000:
        return r + l[1000::1000]
    r = r + l[1000:10000:1000]

    if len(l) <= 100000:
        return r + l[10000::10000]


# Ball small
#    accuracy
#    distance

# Ball moderate
#    accuracy
#    distance

# Ball large
#     accuracy
#     distance

# ----------------

# Box small
#     accuracy
#     distance

# Box moderate
#     accuracy
#     distance

# Box large
#     accuracy
#     distance

# ----------------

# Simplex small
#     accuracy
#     distance

# Simplex moderate
#     accuracy
#     distance


# Simplex large
#     accuracy
#     distance
def part1():
    maximum_iterations = 20_000
    set_types = ['ball', 'box', 'simplex']
    sizes = ['small', 'moderate', 'large']
    problems = {
        'small': {
            'n': 10,
            'm': 0,
            'function': {
                'alpha': 2 / (10 - 1),
                'beta': 0.25,
                'gamma': 1
            },
            'eps': 1e-12,
            'residue': 10
        },
        'moderate': {
            'n': 100,
            'm': 0,
            'function': {
                'alpha': 2 / (1_000 - 1),
                'beta': 0.25,
                'gamma': 1
            },
            'eps': 1e-12,
            'residue': 1e2
        },
        'large': {
            'n': 1_000,
            'm': 0,
            'function': {
                'alpha': 2 / (1_000_000 - 1),
                'beta': 0.25,
                'gamma': 1
            },
            'eps': 1e-12,
            'residue': 1e3
        }
    }

    clrs = sns.color_palette('husl', 3)

    for set_type in set_types:
        for size in sizes:
            print("Size: %s, set: %s" % (size, set_type))

            params = None
            if set_type == 'ball':
                params = {'R': 100 * problems[size]['residue']}
            elif set_type == 'box':
                params = {
                    'a':
                    np.zeros((problems[size]['n'], )),
                    'b':
                    100 * problems[size]['residue'] * np.ones(
                        (problems[size]['n'], ))
                }
            elif set_type == 'simplex':
                params = {'p': 100 * problems[size]['residue']}

            x_grad, dist_grad, vals_grad = solve(
                problems[size]['n'], problems[size]['m'], set_type, params,
                problems[size]['function'], problems[size]['eps'],
                problems[size]['residue'], 'gradient', maximum_iterations)

            x_opt, dist_opt, vals_opt = solve(
                problems[size]['n'], problems[size]['m'], set_type, params,
                problems[size]['function'], problems[size]['eps'],
                problems[size]['residue'], 'optimal', maximum_iterations)

            L = problems[size]['function']['alpha'] + 4 * problems[size][
                'function']['beta'] + problems[size]['function']['gamma']
            mu = problems[size]['function']['alpha']
            kappa = L / mu

            residue = problems[size]['residue']

            l_grad = np.array(reduce(range(1, len(x_grad) + 1)))
            l_opt = np.array(reduce(range(1, len(x_opt) + 1)))

            vals_grad = vals_grad[l_grad - 1]
            vals_opt = vals_opt[l_opt - 1]

            dist_grad = dist_grad[l_grad - 1]
            dist_opt = dist_opt[l_opt - 1]

            cor212 = 2 * L * residue**2 / (l_grad + 4)
            eq2223 = (L + mu) / 2 * residue**2 * np.exp(
                -l_opt * np.sqrt(mu / L))
            thm223 = [
                min(np.power(1 - mu / L, k), 4 * L /
                    (2 * np.sqrt(L) + k * np.sqrt(mu))**2) *
                (vals_opt[0] + mu / 2 * residue**2) for k in l_opt
            ]
            thm2113acc = mu / 2 * (
                (np.sqrt(kappa) - 1) /
                (np.sqrt(kappa) + 1))**(2 * l_opt) * residue**2

            fig = plt.figure()
            plt.loglog(l_grad,
                       vals_grad,
                       '-v',
                       markerfacecolor='none',
                       c=clrs[0])
            plt.loglog(l_opt,
                       vals_opt,
                       '-v',
                       markerfacecolor='none',
                       c=clrs[1])
            plt.loglog(l_grad, cor212, '-', c=clrs[0])
            plt.loglog(l_opt, eq2223, '-', c=clrs[1])
            plt.loglog(l_opt, thm223, '--', c=clrs[1])
            plt.loglog(l_opt, thm2113acc, '-', c=clrs[2])
            plt.xlabel("Iteration number, \\(k\\)")
            plt.ylabel("Accuracy, \\(f(\\xk) - f(\\xopt)\\)")
            plt.legend([
                "Gradient method", "Optimal method",
                "Upper bound of Corollary~\\ref{cor:2.1.2}",
                "Upper bound of \\thmref{thm:2.2.23}",
                "Upper bound of \\thmref{thm:2.2.3}",
                "Lower bound of \\thmref{thm:2.1.13}"
            ])

            tikzplotlib.save("../report/plots/p1_%s_%s_acc.tikz" %
                             (set_type, size),
                             figure=fig,
                             axis_width="\\linewidth")

            thm2214 = (1 - mu / L)**l_grad * residue
            thm2113dist = ((np.sqrt(kappa) - 1) /
                           (np.sqrt(kappa) + 1))**l_opt * residue

            fig = plt.figure()
            plt.loglog(l_grad,
                       dist_grad,
                       '-v',
                       markerfacecolor='none',
                       c=clrs[0])
            plt.loglog(l_opt,
                       dist_opt,
                       '-v',
                       markerfacecolor='none',
                       c=clrs[1])
            plt.loglog(l_grad, thm2214, '--', c=clrs[2])
            plt.loglog(l_opt, thm2113dist, '-', c=clrs[2])
            plt.xlabel("Iteration number, \\(k\\)")
            plt.ylabel("Distance to minimum, \\(\\norm{\\xk - \\xopt}\\)")
            plt.legend([
                "Gradient method", "Optimal method",
                "Upper bound of Theorem~\\ref{thm:2.2.14}",
                "Lower bound of \\thmref{thm:2.1.13}"
            ])

            tikzplotlib.save("../report/plots/p1_%s_%s_dist.tikz" %
                             (set_type, size),
                             figure=fig,
                             axis_width="\\linewidth")


# Ball
#     kappa
#     R
#     n

# Box
#     kappa
#     R
#     n
#     m


# Simplex
#     kappa
#     R
#     n
#     m
def part2():
    maximum_iterations = 20_000
    set_types = ['ball', 'box', 'simplex']

    default_kappa = 1000
    default_residue = 100
    default_n = 100
    default_m = 10
    epsilon = 1e-12

    clrs = sns.color_palette('husl', 3)

    variab = {
        'kappa': [10**i for i in range(1, 11)],
        'residue': [10, 20, 50, 100, 200, 500, 1000, 2000],
        'n': [10, 20, 50, 100, 200, 500, 1000, 2000],
        'm': [5 * i for i in range(21)]
    }

    for set_type in set_types:
        print(set_type)

        params = None

        if set_type == 'ball':
            params = {'R': 100 * default_residue}
        elif set_type == 'box':
            params = {
                'a': np.zeros((default_n, )),
                'b': 100 * default_residue * np.ones((default_n, ))
            }
        elif set_type == 'simplex':
            params = {'p': 100 * default_residue}

        n_iter_grad = []
        n_iter_opt = []
        for kappa in variab['kappa']:

            x_grad, dist_grad, vals_grad = solve(default_n, default_m,
                                                 set_type, params, {
                                                     'alpha': 2 / (kappa - 1),
                                                     'beta': 0.25,
                                                     'gamma': 1
                                                 }, epsilon, default_residue,
                                                 'gradient',
                                                 maximum_iterations)

            x_opt, dist_opt, vals_opt = solve(default_n, default_m, set_type,
                                              params, {
                                                  'alpha': 2 / (kappa - 1),
                                                  'beta': 0.25,
                                                  'gamma': 1
                                              }, epsilon, default_residue,
                                              'optimal', maximum_iterations)

            n_iter_grad.append(len(x_grad))
            n_iter_opt.append(len(x_opt))

        plt.figure()
        plt.loglog(variab['kappa'],
                   n_iter_grad,
                   '-v',
                   markerfacecolor='none',
                   c=clrs[0])
        plt.loglog(variab['kappa'],
                   n_iter_opt,
                   '-v',
                   markerfacecolor='none',
                   c=clrs[1])
        plt.xlabel("Condition number, \\(\\kappa\\)")
        plt.ylabel("Number of iterations until convergence")
        plt.legend(["Gradient method", "Optimal method"])

        tikzplotlib.save("../report/plots/p2_%s_kappa.tikz" % (set_type),
                         axis_width="\\linewidth")

        n_iter_grad = []
        n_iter_opt = []

        if set_type == 'box' or set_type == 'simplex':
            for m in variab['m']:
                x_grad, dist_grad, vals_grad = solve(
                    default_n, m, set_type, params, {
                        'alpha': 2 / (default_kappa - 1),
                        'beta': 0.25,
                        'gamma': 1
                    }, epsilon, default_residue, 'gradient',
                    maximum_iterations)

                x_opt, dist_opt, vals_opt = solve(
                    default_n, m, set_type, params, {
                        'alpha': 2 / (default_kappa - 1),
                        'beta': 0.25,
                        'gamma': 1
                    }, epsilon, default_residue, 'optimal', maximum_iterations)

                n_iter_grad.append(len(x_grad))
                n_iter_opt.append(len(x_opt))

            plt.figure()
            plt.loglog(variab['m'],
                       n_iter_grad,
                       '-v',
                       markerfacecolor='none',
                       c=clrs[0])
            plt.loglog(variab['m'],
                       n_iter_opt,
                       '-v',
                       markerfacecolor='none',
                       c=clrs[1])
            plt.xlabel("Number of active constraints, \\(m\\)")
            plt.ylabel("Number of iterations until convergence")
            plt.legend(["Gradient method", "Optimal method"])

            tikzplotlib.save("../report/plots/p2_%s_m.tikz" % (set_type),
                             axis_width="\\linewidth")

            n_iter_grad = []
            n_iter_opt = []

        for n in variab['n']:

            if set_type == 'ball':
                params = {'R': 100 * default_residue}
            elif set_type == 'box':
                params = {
                    'a': np.zeros((n, )),
                    'b': 100 * default_residue * np.ones((n, ))
                }
            elif set_type == 'simplex':
                params = {'p': 100 * default_residue}

            x_grad, dist_grad, vals_grad = solve(
                n, default_m, set_type, params, {
                    'alpha': 2 / (default_kappa - 1),
                    'beta': 0.25,
                    'gamma': 1
                }, epsilon, default_residue, 'gradient', maximum_iterations)

            x_opt, dist_opt, vals_opt = solve(n, default_m, set_type, params, {
                'alpha': 2 / (default_kappa - 1),
                'beta': 0.25,
                'gamma': 1
            }, epsilon, default_residue, 'optimal', maximum_iterations)

            n_iter_grad.append(len(x_grad))
            n_iter_opt.append(len(x_opt))

        plt.figure()
        plt.loglog(variab['n'],
                   n_iter_grad,
                   '-v',
                   markerfacecolor='none',
                   c=clrs[0])
        plt.loglog(variab['n'],
                   n_iter_opt,
                   '-v',
                   markerfacecolor='none',
                   c=clrs[1])
        plt.xlabel("Dimension of the problem, \\(n\\)")
        plt.ylabel("Number of iterations until convergence")
        plt.legend(["Gradient method", "Optimal method"])

        tikzplotlib.save("../report/plots/p2_%s_n.tikz" % (set_type),
                         axis_width="\\linewidth")

        n_iter_grad = []
        n_iter_opt = []

        for residue in variab['residue']:

            if set_type == 'ball':
                params = {'R': 100 * residue}
            elif set_type == 'box':
                params = {
                    'a': np.zeros((default_n, )),
                    'b': 100 * residue * np.ones((default_n, ))
                }
            elif set_type == 'simplex':
                params = {'p': 100 * residue}

            x_grad, dist_grad, vals_grad = solve(
                default_n, default_m, set_type, params, {
                    'alpha': 2 / (default_kappa - 1),
                    'beta': 0.25,
                    'gamma': 1
                }, epsilon, residue, 'gradient', maximum_iterations)

            x_opt, dist_opt, vals_opt = solve(
                default_n, default_m, set_type, params, {
                    'alpha': 2 / (default_kappa - 1),
                    'beta': 0.25,
                    'gamma': 1
                }, epsilon, residue, 'optimal', maximum_iterations)

            n_iter_grad.append(len(x_grad))
            n_iter_opt.append(len(x_opt))

        plt.figure()
        plt.loglog(variab['residue'],
                   n_iter_grad,
                   '-v',
                   markerfacecolor='none',
                   c=clrs[0])
        plt.loglog(variab['residue'],
                   n_iter_opt,
                   '-v',
                   markerfacecolor='none',
                   c=clrs[1])
        plt.xlabel(
            "Initial distance from the minimum, \\(\\norm{\\xk - \\xopt}\\)")
        plt.ylabel("Number of iterations until convergence")
        plt.legend(["Gradient method", "Optimal method"])

        tikzplotlib.save("../report/plots/p2_%s_residue.tikz" % (set_type),
                         axis_width="\\linewidth")


if __name__ == '__main__':
    #part1()
    part2()
