#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pickle
import time
import numpy as np

from solvers import solve


def bm_no_incr_decr():
    """
    Show there is no guarantee of incremental decrease at every iteration.
    """
    n = 2
    eps = 0.001
    L = 1
    rho = 10
    x_subgradient, f_subgradient = solve(n, {
        'alpha': L / (np.sqrt(n) + 2),
        'beta': L / (np.sqrt(n) + 2)
    }, rho, 'subgradient', eps, 101)

    subgradient = zip(x_subgradient, f_subgradient)
    pickle.dump(subgradient,
                open("../report/data/no_incr_decr_subgradient.p", "wb"))

    x_ellipsoid, f_ellipsoid = solve(n, {
        'alpha': L / (np.sqrt(n) + 2),
        'beta': L / (np.sqrt(n) + 2)
    }, rho, 'ellipsoid', eps, 101)

    ellipsoid = zip(x_ellipsoid, f_ellipsoid)
    pickle.dump(ellipsoid, open("../report/data/no_incr_decr_ellipsoid.p",
                                "wb"))


def bm_roc():
    """
    Show theoretical guarantees are satisfied.
    """
    max_it = 100_000
    epss = [1e-12, 1e-1, 10]
    ns = [10, 100, 1000]
    Ls = [10, 100, 1000]
    rhos = [10, 100, 1000]
    names = ['small', 'medium', 'large']
    for eps, n, L, rho, name in zip(epss, ns, Ls, rhos, names):
        print("Size: %s" % (name))
        print(" - subgradient")
        x_subgradient, f_subgradient = solve(n, {
            'alpha': L / (np.sqrt(n) + 2),
            'beta': L / (np.sqrt(n) + 2)
        }, rho, 'subgradient', eps, max_it)

        subgradient = zip(x_subgradient, f_subgradient)
        pickle.dump(subgradient,
                    open("../report/data/roc_subgradient_%s.p" % (name), "wb"))

        print(" - ellipsoid")

        x_ellipsoid, f_ellipsoid = solve(n, {
            'alpha': L / (np.sqrt(n) + 2),
            'beta': L / (np.sqrt(n) + 2)
        }, rho, 'ellipsoid', eps, max_it)

        ellipsoid = zip(x_ellipsoid, f_ellipsoid)
        pickle.dump(ellipsoid,
                    open("../report/data/roc_ellipsoid_%s.p" % (name), "wb"))


def bm_time():
    """
    Show that ellipsoid is slow.
    """
    L = 10
    eps = 1e-6
    rho = 10
    max_it = 1000
    sg = []
    el = []
    for n in range(10, 1001, 10):
        print("n: %d" % n)
        s = time.perf_counter()
        _ = solve(n, {
            'alpha': L / (np.sqrt(n) + 2),
            'beta': L / (np.sqrt(n) + 2)
        }, rho, 'subgradient', eps, max_it)
        e = time.perf_counter()

        sg.append(e - s)

        s = time.perf_counter()
        _ = solve(n, {
            'alpha': L / (np.sqrt(n) + 2),
            'beta': L / (np.sqrt(n) + 2)
        }, rho, 'ellipsoid', eps, max_it)
        e = time.perf_counter()

        el.append(e - s)

    pickle.dump(sg, open("../report/data/exec_time_subgradient.p", "wb"))
    pickle.dump(el, open("../report/data/exec_time_ellipsoid.p", "wb"))


def bm_params():
    """
    Show the influence of each parameter.
    """
    default_n = 10
    default_L = 10
    default_rho = 10
    eps = 1e-1
    max_it = 1_000_000_000

    Ls = range(1, 100, 10)
    rhos = range(1, 100, 10)
    ns = range(2, 250, 20)

    L_influence = {'subgradient': [], 'ellipsoid': []}
    for L in Ls:
        print("L: %d" % (L))
        print(" - subgradient")
        x_subgradient, _ = solve(
            default_n, {
                'alpha': L / (np.sqrt(default_n) + 2),
                'beta': L / (np.sqrt(default_n) + 2)
            }, default_rho, 'subgradient', eps, max_it)

        L_influence['subgradient'].append(len(x_subgradient))

        print(" - ellipsoid")

        x_ellipsoid, _ = solve(
            default_n, {
                'alpha': L / (np.sqrt(default_n) + 2),
                'beta': L / (np.sqrt(default_n) + 2)
            }, default_rho, 'ellipsoid', eps, max_it)

        L_influence['ellipsoid'].append(len(x_ellipsoid))

    pickle.dump(L_influence, open("../report/data/param_L.p", "wb"))

    rho_influence = {'subgradient': [], 'ellipsoid': []}
    for rho in rhos:
        print("rho: %d" % (rho))
        print(" - subgradient")
        x_subgradient, _ = solve(
            default_n, {
                'alpha': default_L / (np.sqrt(default_n) + 2),
                'beta': default_L / (np.sqrt(default_n) + 2)
            }, rho, 'subgradient', eps, max_it)

        rho_influence['subgradient'].append(len(x_subgradient))

        print(" - ellipsoid")

        x_ellipsoid, _ = solve(
            default_n, {
                'alpha': default_L / (np.sqrt(default_n) + 2),
                'beta': default_L / (np.sqrt(default_n) + 2)
            }, rho, 'ellipsoid', eps, max_it)

        rho_influence['ellipsoid'].append(len(x_ellipsoid))

    pickle.dump(rho_influence, open("../report/data/param_rho.p", "wb"))

    n_influence = {'subgradient': [], 'ellipsoid': []}
    for n in ns:
        print("n: %d" % (n))
        print(" - subgradient")
        x_subgradient, _ = solve(
            n, {
                'alpha': default_L / (np.sqrt(n) + 2),
                'beta': default_L / (np.sqrt(n) + 2)
            }, default_rho, 'subgradient', eps, max_it)

        n_influence['subgradient'].append(len(x_subgradient))

        print(" - ellipsoid")

        x_ellipsoid, _ = solve(
            n, {
                'alpha': default_L / (np.sqrt(n) + 2),
                'beta': default_L / (np.sqrt(n) + 2)
            }, default_rho, 'ellipsoid', eps, max_it)

        n_influence['ellipsoid'].append(len(x_ellipsoid))

    pickle.dump(n_influence, open("../report/data/param_n.p", "wb"))


if __name__ == '__main__':
    #bm_no_incr_decr()
    #bm_roc()
    #bm_time()
    bm_params()
