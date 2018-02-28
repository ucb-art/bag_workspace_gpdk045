import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sig
import scipy.interpolate as interp
import scipy.optimize as opt

from bag.util.search import minimize_cost_golden_float


def get_tset(t, y, tot_err):
    last_idx = np.where(y < 1.0 - tot_err)[0][-1]
    last_max_vec = np.where(y > 1.0 + tot_err)[0]
    if last_max_vec.size > 0 and last_max_vec[-1] > last_idx:
        last_idx = last_max_vec[-1]
        last_val = 1.0 + tot_err
    else:
        last_val = 1.0 - tot_err

    if last_idx == t.size - 1:
        return t[-1]
    f = interp.InterpolatedUnivariateSpline(t, y - last_val)
    t0 = t[last_idx]
    t1 = t[last_idx + 1]
    return opt.brentq(f, t0, t1)


def calc_tset(gain, tot_err, k, tvec):
    w0 = 2 * np.pi * 1e6
    w1 = w0 / gain
    w2 = w0 * k

    num = [gain]
    den = [1/(w1*w2), 1/w1 + 1/w2, gain + 1]
    _, yvec = sig.step((num, den), T=tvec)
    return get_tset(tvec, yvec, tot_err)


def get_opt_poles(gain, tot_err, t_targ):
    kmin = 1.0
    kmax = 100.0
    ktol = 1e-3

    tvec = np.linspace(0, 10e-6, 1000)
    def opt_fun(ktest):
        return -1 * calc_tset(gain, tot_err, ktest, tvec)

    result = minimize_cost_golden_float(opt_fun, 0, kmin, kmax, tol=ktol)
    k_opt = result.xmax
    t_opt = -result.vmax
    
    scale = t_opt / t_targ
    f2 = k_opt * scale * 1e6
    f1 = f2 / k_opt / gain
    return f1, f2


def plot_tset_vs_k(gain, tot_err):
    kvec = np.linspace(1, 10, 51)
    tvec = np.linspace(0, 10e-6, 5000)
    tlist = [calc_tset(gain, tot_err, k, tvec) for k in kvec]

    plt.figure(1)
    plt.plot(kvec, tlist, '-o')
    plt.show()


def plot_y(gain, f1, f2, t_targ, tot_err):
    n = 5000
    tvec = np.linspace(0, 5 * t_targ, n)
    num = [gain]
    den = [1 / (4 * np.pi**2 * f1 * f2), (1 / f1 + 1 / f2) / 2 / np.pi,
           gain + 1]
    _, yvec = sig.step((num, den), T=tvec)

    plt.figure(1)
    plt.plot(tvec, yvec, 'b')
    plt.plot(tvec, [1 - tot_err] * n, 'r')
    plt.plot(tvec, [1 + tot_err] * n, 'r')
    plt.show()


def run_main():
    gain = 40
    tot_err = 0.04
    t_targ = 10e-9

    # plot_tset_vs_k(gain, tot_err)
    f1, f2 = get_opt_poles(gain, tot_err, t_targ)
    print(f1, f2)
    plot_y(gain, f1, f2, t_targ, tot_err)


if __name__ == '__main__':
    run_main()
