#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tikzplotlib
import pickle

plt.style.use("ggplot")

from solvers import solve

clrs = sns.husl_palette(4)


def reduce(l):
    l = list(l)
    if len(l) <= 10:
        return l
    r = l[:10]

    if len(l) <= 100:
        return r + l[10::10] + [l[-1]]
    r = r + l[10:100:10]

    if len(l) <= 1000:
        return r + l[100::100] + [l[-1]]
    r = r + l[100:1000:100]

    if len(l) <= 10000:
        return r + l[1000::1000] + [l[-1]]
    r = r + l[1000:10000:1000]

    if len(l) <= 100000:
        return r + l[10000::10000] + [l[-1]]
    r = r + l[10000:100000:10000]

    if len(l) <= 1000000:
        return r + l[100000::100000] + [l[-1]]


def plt_no_incr_decr():
    # Plot iterates and function val non-decrease side by side for both methods.
    subgradient = pickle.load(
        open("../report/data/no_incr_decr_subgradient.p", "rb"))
    x_subgradient, f_subgradient = list(zip(*subgradient))

    ellipsoid = pickle.load(
        open("../report/data/no_incr_decr_ellipsoid.p", "rb"))
    x_ellipsoid, f_ellipsoid = list(zip(*ellipsoid))

    l_subgradient = np.array(reduce(range(1, len(f_subgradient) + 1)))
    l_ellipsoid = np.array(reduce(range(1, len(f_ellipsoid) + 1)))

    x_subgradient = np.array(x_subgradient)[l_subgradient - 1]
    f_subgradient = np.array(f_subgradient)[l_subgradient - 1]
    x_ellipsoid = np.array(x_ellipsoid)[l_ellipsoid - 1]
    f_ellipsoid = np.array(f_ellipsoid)[l_ellipsoid - 1]

    plt.figure()
    plt.plot([i[0] for i in x_subgradient], [i[1] for i in x_subgradient],
             "-v",
             markerfacecolor='none',
             c=clrs[0])
    plt.plot([i[0] for i in x_ellipsoid], [i[1] for i in x_ellipsoid],
             "-v",
             markerfacecolor='none',
             c=clrs[3])
    plt.plot(0, 0, 'x', c='black')
    plt.xlabel("\\(x\\)")
    plt.ylabel("\\(y\\)")
    plt.legend(["Subgradient method", "Ellipsoid method", "\\(\\xopt\\)"])

    tikzplotlib.save("../report/plots/no_incr_decr_iterates.tikz",
                     axis_width="0.5\\linewidth")

    plt.figure()
    plt.loglog(l_subgradient,
               f_subgradient,
               "-v",
               markerfacecolor='none',
               c=clrs[0])
    plt.loglog(l_ellipsoid,
               f_ellipsoid,
               "-v",
               markerfacecolor='none',
               c=clrs[3])
    plt.xlabel("Iteration number, \\(k\\)")
    plt.ylabel("Accuracy, \\(f(\\xk) - \\fopt\\)")
    plt.legend(["Subgradient method", "Ellipsoid method"])

    tikzplotlib.save("../report/plots/no_incr_decr_vals.tikz",
                     axis_width="0.5\\linewidth")


def plt_roc():
    """
    Show theoretical guarantees are satisfied.
    """
    subgradient_s = pickle.load(
        open("../report/data/roc_subgradient_small.p", "rb"))
    subgradient_m = pickle.load(
        open("../report/data/roc_subgradient_medium.p", "rb"))
    subgradient_l = pickle.load(
        open("../report/data/roc_subgradient_large.p", "rb"))

    _, f_subgradient_s = list(zip(*subgradient_s))
    _, f_subgradient_m = list(zip(*subgradient_m))
    _, f_subgradient_l = list(zip(*subgradient_l))

    ellipsoid_s = pickle.load(
        open("../report/data/roc_ellipsoid_small.p", "rb"))
    ellipsoid_m = pickle.load(
        open("../report/data/roc_ellipsoid_medium.p", "rb"))
    ellipsoid_l = pickle.load(
        open("../report/data/roc_ellipsoid_large.p", "rb"))

    _, f_ellipsoid_s = list(zip(*ellipsoid_s))
    _, f_ellipsoid_m = list(zip(*ellipsoid_m))
    _, f_ellipsoid_l = list(zip(*ellipsoid_l))

    l_subgradient_s = np.array(reduce(range(1, len(f_subgradient_s))))
    l_subgradient_m = np.array(reduce(range(1, len(f_subgradient_m))))
    l_subgradient_l = np.array(reduce(range(1, len(f_subgradient_l))))

    l_ellipsoid_s = np.array(reduce(range(1, len(f_ellipsoid_s))))
    l_ellipsoid_m = np.array(reduce(range(1, len(f_ellipsoid_m))))
    l_ellipsoid_l = np.array(reduce(range(1, len(f_ellipsoid_l))))

    f_subgradient_s = np.minimum.accumulate(
        np.array(f_subgradient_s)[l_subgradient_s])
    f_subgradient_m = np.minimum.accumulate(
        np.array(f_subgradient_m)[l_subgradient_m])
    f_subgradient_l = np.minimum.accumulate(
        np.array(f_subgradient_l)[l_subgradient_l])

    f_ellipsoid_s = np.minimum.accumulate(
        np.array(f_ellipsoid_s)[l_ellipsoid_s])
    f_ellipsoid_m = np.minimum.accumulate(
        np.array(f_ellipsoid_m)[l_ellipsoid_m])
    f_ellipsoid_l = np.minimum.accumulate(
        np.array(f_ellipsoid_l)[l_ellipsoid_l])

    l_sg = [l_subgradient_s, l_subgradient_m, l_subgradient_l]
    l_el = [l_ellipsoid_s, l_ellipsoid_m, l_ellipsoid_l]
    sg = [f_subgradient_s, f_subgradient_m, f_subgradient_l]
    el = [f_ellipsoid_s, f_ellipsoid_m, f_ellipsoid_l]
    name = ['small', 'medium', 'large']

    L = {'small': 10, 'medium': 100, 'large': 1000}
    rho = {'small': 10, 'medium': 100, 'large': 1000}
    n = {'small': 10, 'medium': 100, 'large': 1000}

    for l_subgradient, l_ellipsoid, f_subgradient, f_ellipsoid, name in zip(
            l_sg, l_el, sg, el, name):
        plt.figure()
        plt.loglog(l_subgradient,
                   f_subgradient,
                   "-v",
                   markerfacecolor='none',
                   c=clrs[0])
        plt.loglog(l_ellipsoid,
                   f_ellipsoid,
                   "-v",
                   markerfacecolor='none',
                   c=clrs[3])

        thm322 = L[name] * rho[name] / 2 * np.array(
            [(1 + np.sum([1 / (i + 1) for i in range(k + 1)])) /
             np.sum([1 / np.sqrt(i + 1) for i in range(k + 1)])
             for k in l_subgradient])

        plt.loglog(l_subgradient, thm322, '-', c=clrs[0])

        thm3211 = L[name] * rho[name] * (1 - 1 /
                                         (n[name] + 1)**2)**(l_ellipsoid / 2)

        plt.loglog(l_ellipsoid, thm3211, '-', c=clrs[3])

        n_range = np.array(reduce(range(1, n[name])))
        thm321 = L[name] * rho[name] / (2 * (2 + np.sqrt(n_range + 1)))

        plt.loglog(n_range, thm321, '-', c=clrs[1])

        plt.xlabel("Iteration number, \\(k\\)")
        plt.ylabel("Best accuracy, \\(\\foptk - \\fopt\\)")
        plt.legend([
            "Subgradient method", "Ellipsoid method",
            "Upper bound of \\thmref{thm:3.2.2}",
            "Upper bound of \\thmref{thm:3.2.11}",
            "Lower bound of \\thmref{thm:3.2.1}"
        ])

        tikzplotlib.save("../report/plots/roc_%s.tikz" % (name),
                         axis_width="0.7\\linewidth")


def plt_time():
    """
    Show execution times.
    """
    sg = pickle.load(open("../report/data/exec_time_subgradient.p", "rb"))
    el = pickle.load(open("../report/data/exec_time_ellipsoid.p", "rb"))

    n = np.array(range(10, 1001, 10))

    plt.figure()
    plt.plot(n, sg, "-v", markerfacecolor='none', c=clrs[0])
    plt.plot(n, el, "-v", markerfacecolor='none', c=clrs[3])

    plt.xlabel("Problem dimension, \\(n\\)")
    plt.ylabel("Execution time per iteration [\\si{\\milli\\second}]")
    plt.legend(["Subgradient method", "Ellipsoid method"])

    tikzplotlib.save("../report/plots/exec_time.tikz",
                     axis_width="0.5\\linewidth")


def plt_params():
    """
    Show the influence of each parameter.
    """
    L = pickle.load(open("../report/data/param_L.p", "rb"))
    rho = pickle.load(open("../report/data/param_rho.p", "rb"))
    n = pickle.load(open("../report/data/param_n.p", "rb"))

    L_sg = L['subgradient']
    L_el = L['ellipsoid']
    rho_sg = rho['subgradient']
    rho_el = rho['ellipsoid']
    n_sg = n['subgradient']
    n_el = n['ellipsoid']

    Ls = range(1, 100, 10)
    rhos = range(1, 100, 10)
    ns = range(2, 250, 20)

    plt.figure()
    plt.semilogy(Ls, L_sg, "-v", markerfacecolor='none', c=clrs[0])
    plt.semilogy(Ls, L_el, "-v", markerfacecolor='none', c=clrs[3])
    plt.xlabel("Lipschitz parameter, \\(L\\)")
    plt.ylabel("Iterations until convergence")
    plt.legend(["Subgradient method", "Ellipsoid method"])

    tikzplotlib.save("../report/plots/param_L.tikz",
                     axis_width="0.6\\linewidth")

    plt.figure()
    plt.semilogy(rhos, rho_sg, "-v", markerfacecolor='none', c=clrs[0])
    plt.semilogy(rhos, rho_el, "-v", markerfacecolor='none', c=clrs[3])
    plt.xlabel("Initial distance from the minimum, \\(\\rho\\)")
    plt.ylabel("Iterations until convergence")
    plt.legend(["Subgradient method", "Ellipsoid method"])

    tikzplotlib.save("../report/plots/param_rho.tikz",
                     axis_width="0.6\\linewidth")

    plt.figure()
    plt.semilogy(ns, n_sg, "-v", markerfacecolor='none', c=clrs[0])
    plt.semilogy(ns, n_el, "-v", markerfacecolor='none', c=clrs[3])
    plt.xlabel("Problem dimension, \\(n\\)")
    plt.ylabel("Iterations until convergence")
    plt.legend(["Subgradient method", "Ellipsoid method"])

    tikzplotlib.save("../report/plots/param_n.tikz",
                     axis_width="0.6\\linewidth")


if __name__ == '__main__':
    #plt_no_incr_decr()
    #plt_roc()
    #plt_time()
    plt_params()
