import pprint

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sig

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


def design_load(specs, db, ax_list=None, label=None):
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
    vgs_list, vds_list, vcasc_list, ibias_list, ro_list = [], [], [], [], [] 
    p1_list, p2_list, gain_list = [], [], []
    for vgs_val in np.linspace(vgs_min, vgs_max, num_pts, endpoint=True):
        fout1 = zero_fun1(vdd - 5e-3, vgs_val)
        fout2 = zero_fun1(voutcm + 5e-3, vgs_val)
        if fout1 * fout2 > 0:
            continue
        vmid = opt.brentq(zero_fun1, voutcm + 5e-3, vdd - 5e-3, args=(vgs_val,))
        if vmid + vgs_max <= 0:
            continue
        barg = db.get_fun_arg(vgs=vgs_val, vds=vmid-vdd, vbs=0)
        bot_op = db.query(vbs=0, vds=vmid-vdd, vgs=vgs_val)
        ib = bot_op['ibias']
        args = (vmid, ib)
        fout1 = zero_fun2(vmid + vgs_max, *args)
        fout2 = zero_fun2(0, *args)
        if fout1 * fout2 > 0:
            continue
        vcasc = opt.brentq(zero_fun2, 0, vmid + vgs_max, args=args)
        top_op = db.query(vbs=vdd-vmid, vds=voutcm-vmid, vgs=vcasc-vmid)

        cir = LTICircuit()
        cir.add_transistor(bot_op, 'mid', 'gnd', 'gnd', 'gnd', fg=1)
        cir.add_transistor(top_op, 'out', 'gnd', 'mid', 'gnd', fg=casc_scale)
        cur_tf = cir.get_transfer_function('out', 'out', in_type='i')
        ro = cur_tf.num[-1] / cur_tf.den[-1]
        p1, p2 = cur_tf.poles
        p1, p2 = min(-p1, -p2), max(-p1, -p2)
        p2 /= 2 * np.pi

        # add approximation from input branch
        cpar = 1 / (ro * p1)
        p1 = 1 / (2 * np.pi * 2 * cpar * (ro / 4))
        cur_gain = 2 * ib / vstar_in * (ro / 4)

        vgs_list.append(vgs_val)
        vds_list.append(vmid-vdd)
        vcasc_list.append(vcasc)
        ibias_list.append(ib)
        ro_list.append(ro * ib)
        p1_list.append(p1)
        p2_list.append(p2)
        gain_list.append(cur_gain)

    if ax_list is not None:
        val_list_list = [vds_list, gain_list, p1_list, p2_list]
        ylabel_list = ['$V_{DS}$ (V)', 'Gain (V/V)',
                       '$p_{1o}$ (GHz)', '$p_{2o}$ (GHz)']
        sp_list = [1, 1, 1e-9, 1e-9]
        for ax, val_list, ylabel, sp in zip(ax_list, val_list_list,
                                            ylabel_list, sp_list):
            ax.plot(vgs_list, np.asarray(val_list) * sp, '-o', label=label)
            ax.set_ylabel(ylabel)

        ax_list[-1].set_xlabel('$V_{GS}$ (V)')
    return vgs_list[0], vds_list[0], vcasc_list[0]

def run_main():
    interp_method = 'spline'
    sim_env = 'tt'
    nmos_spec = 'specs_mos_char/nch_w0d5.yaml'
    pmos_spec = 'specs_mos_char/pch_w0d5.yaml'
    intent = 'lvt'

    nch_db = get_db(nmos_spec, intent, interp_method=interp_method,
                    sim_env=sim_env)

    nch_op = nch_db.query(vbs=0, vds=0.5, vgs=0.5)

    pprint.pprint(nch_op)
    # building circuit
    cir = LTICircuit()
    cir.add_transistor(nch_op, 'out', 'in', 'gnd', 'gnd', fg=2)
    cir.add_res(10e3, 'out', 'gnd')
    cir.add_cap(100e-15, 'out', 'gnd')

    # get gain/poles/zeros/bode plot
    trans_fun = cir.get_transfer_function('in', 'out', in_type='v')
    print('poles: %s' % trans_fun.poles)
    print('zeros: %s' % trans_fun.zeros)
    # note: don't use the gain attribute.  For some reason it's broken
    print('gain: %.4g' % (trans_fun.num[-1] / trans_fun.den[-1]))
    fvec = np.logspace(5, 10, 1000)
    _, mag, phase = sig.bode(trans_fun, w=2 * np.pi * fvec)
    
    # get transient response
    state_space = cir.get_state_space('in', 'out', in_type='v')
    tvec = np.linspace(0, 1e-8, 1000)
    _, yvec = sig.step(state_space, T=tvec)
    
    _, (ax0, ax1) = plt.subplots(2, sharex=True)
    ax0.semilogx(fvec, mag)
    ax0.set_ylabel('Magnitude (dB)')
    ax1.semilogx(fvec, phase)
    ax1.set_ylabel('Phase (degrees)')
    ax1.set_xlabel('Frequency (Hz)')

    plt.figure(2)
    plt.plot(tvec, yvec)
    plt.ylabel('Output (V/V)')
    plt.xlabel('Time (s)')

    plt.show()


if __name__ == '__main__':
    run_main()
