import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sig
import scipy.interpolate as interp
import scipy.optimize as opt

from bag.util.search import minimize_cost_golden_float
from bag.data.lti import LTICircuit
from verification_ec.mos.query import MOSDBDiscrete


def get_db(spec_file, intent, interp_method='spline', sim_env='tt'):
    # initialize transistor database from simulation data
    mos_db = MOSDBDiscrete([spec_file], interp_method=interp_method)
    # set process corners
    mos_db.env_list = [sim_env]
    # set layout parameters
    mos_db.set_dsn_params(intent=intent)
    return mos_db


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


def get_opt_poles(specs, gain):
    t_targ = specs['t_settle']
    tot_err = specs['tot_err']

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


def design_load(specs, db):
    sim_env = specs['sim_env']
    vgs_res = specs['vgs_res']
    vdd = specs['vdd']
    voutcm = specs['voutcm']
    vstar_in = specs['vstar_in']
    vstar_load = specs['vstar_load']
    casc_scale = specs['casc_scale']

    vgs_idx = db.get_fun_arg_index('vgs')
    vstar_fun = db.get_function('vstar', env=sim_env)
    ibias_fun = db.get_function('ibias', env=sim_env)
    vgs_min, vgs_max = vstar_fun.get_input_range(vgs_idx)

    def zero_fun1(vm, vgs):
        arg = db.get_fun_arg(vgs=vgs, vds=vm-vdd, vbs=0)
        return (vstar_fun(arg) - vstar_load) * 1e3

    def zero_fun2(vc, vm, itarg):
        arg = db.get_fun_arg(vgs=vc-vm, vds=voutcm-vm, vbs=vdd-vm)
        return (ibias_fun(arg) * casc_scale - itarg) * 1e6

    num_pts = int(np.ceil((vgs_max - vgs_min) / vgs_res)) + 1
    vgs_list, vds_list, vcasc_list, ibias_list = [], [], [], []
    for vgs_val in np.linspace(vgs_min, vgs_max, num_pts, endpoint=True):
        fout1 = zero_fun1(vdd - 5e-3, vgs_val)
        fout2 = zero_fun1(voutcm + 5e-3, vgs_val)
        if fout1 * fout2 > 0:
            continue
        vmid = opt.brentq(zero_fun1, voutcm + 5e-3, vdd - 5e-3, args=(vgs_val,))
        if vmid + vgs_max <= 0:
            continue
        barg = db.get_fun_arg(vgs=vgs_val, vds=vmid-vdd, vbs=0)
        ib = ibias_fun(barg)
        args = (vmid, ib)
        fout1 = zero_fun2(vmid + vgs_max, *args)
        fout2 = zero_fun2(0, *args)
        if fout1 * fout2 > 0:
            continue
        vcasc = opt.brentq(zero_fun2, 0, vmid + vgs_max, args=args)

        vgs_list.append(vgs_val)
        vds_list.append(vmid-vdd)
        vcasc_list.append(vcasc)
        ibias_list.append(ib)

    f, ax_list = plt.subplots(3, sharex=True)
    val_list_list = [vds_list, vcasc_list, ibias_list]
    ylabel_list = ['Vds (V)', 'Vcasc (V)', '$I_{bias}$ (uA)']
    sp_list = [1, 1, 1e6]
    for ax, val_list, ylabel, sp in zip(ax_list, val_list_list,
                                        ylabel_list, sp_list):
        ax.plot(vgs_list, np.asarray(val_list) * sp)
        ax.set_ylabel(ylabel)


def run_main():
    interp_method = 'spline'
    sim_env = 'tt'
    nmos_spec = 'specs_mos_char/nch_w0d5.yaml'
    pmos_spec = 'specs_mos_char/pch_w0d5.yaml'
    intent = 'lvt'

    pch_db = get_db(pmos_spec, intent, interp_method=interp_method,
                    sim_env=sim_env)

    specs = dict(
        sim_env=sim_env,
        vgs_res=5e-3,
        vdd=1.2,
        voutcm=0.6,
        tot_err=0.04,
        t_settle=10e-9,
        vstar_in=150e-3,
        casc_scale=2.0,
        vstar_load=200e-3,
        )

    design_load(specs, pch_db)
    plt.show()

if __name__ == '__main__':
    run_main()
