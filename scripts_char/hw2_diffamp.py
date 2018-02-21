# -*- coding: utf-8 -*-

import pprint

import numpy as np
import scipy.optimize as sciopt

from bag.util.search import BinaryIterator
from verification_ec.mos.query import MOSDBDiscrete

def get_db(spec_file, intent, interp_method='spline', sim_env='tt'):
    # initialize transistor database from simulation data
    mos_db = MOSDBDiscrete([spec_file], interp_method=interp_method)
    # set process corners
    mos_db.env_list = [sim_env]
    # set layout parameters
    mos_db.set_dsn_params(intent=intent)
    return mos_db


def design_input(specs):
    """Find operating point that meets the given vstar spec."""
    db = specs['in_db']
    voutcm = specs['voutcm']
    vstar = specs['vimax']
    vdst = specs['vdst_min']
    in_type = specs['in_type']

    if in_type == 'nch':
        vb = 0
        vtail = vdst
    else:
        vb = specs['vdd']
        vtail = vb - vdst

    return db.query(vbs=vb-vtail, vds=voutcm-vtail, vstar=vstar)


def design_load(specs, input_op):
    """Design load.

    Sweep vgs.  For each vgs, compute gain and max bandwidth.  If
    both gain and BW specs are met, pick operating point that minimizes
    gamma_r * gm_r
    """
    db = specs['load_db']
    sim_env = specs['sim_env']
    vout = specs['voutcm']
    vgs_res = specs['vgs_res']
    gain_min = specs['gain_min']
    bw = specs['bw']
    in_type = specs['in_type']

    if in_type == 'nch':
        vs = specs['vdd']
    else:
        vs = 0

    gm_fun = db.get_function('gm', env=sim_env)
    gds_fun = db.get_function('gds', env=sim_env)
    cdd_fun = db.get_function('cdd', env=sim_env)
    gamma_fun = db.get_function('gamma', env=sim_env)
    ib_fun = db.get_function('ibias', env=sim_env)

    vgs_idx = db.get_fun_arg_index('vgs')
    vgs_min, vgs_max = ib_fun.get_input_range(vgs_idx)
    num_points = int(np.ceil((vgs_max - vgs_min) / vgs_res)) + 1

    gm_i = input_op['gm']
    itarg = input_op['ibias']
    gds_i = input_op['gds']
    cdd_i = input_op['cdd']

    vgs_best = None
    metric_best = float('inf')
    gain_max = 0
    bw_max = 0
    vgs_vec = np.linspace(vgs_min, vgs_max, num_points, endpoint=True)
    bw_list, gain_list, gamma_list, gm_list, metric_list = [], [], [], [], []
    for vgs_val in vgs_vec:
        farg = db.get_fun_arg(vgs=vgs_val, vds=vout-vs, vbs=0)
        scale = itarg / ib_fun(farg)
        gm_r = gm_fun(farg) * scale
        gds_r = gds_fun(farg) * scale
        cdd_r = cdd_fun(farg) * scale
        gamma_r = gamma_fun(farg)

        bw_cur = (gds_r + gds_i) / (cdd_i + cdd_r) / 2 / np.pi
        gain_cur = gm_i / (gds_r + gds_i)
        metric_cur = gamma_r * gm_r
        bw_list.append(bw_cur)
        gain_list.append(gain_cur)
        metric_list.append(metric_cur)
        gamma_list.append(gamma_r)
        gm_list.append(gm_r)
        if gain_cur >= gain_min and bw_cur >= bw:
            if metric_cur < metric_best:
                metric_best = metric_cur
                vgs_best = vgs_val
        else:
            gain_max = max(gain_max, gain_cur)
            bw_max = max(bw_max, bw_cur)

    if vgs_best is None:
        raise ValueError('No solution.  max gain = %.4g, '
                         'max bw = %.4g' % (gain_max, bw_max))
    
    import matplotlib.pyplot as plt
    f, ax_list = plt.subplots(5, sharex=True)
    ax_list[0].plot(vgs_vec, np.asarray(bw_list) / 1e9)
    ax_list[0].set_ylabel('max Bw (GHz)')
    ax_list[1].plot(vgs_vec, gain_list)
    ax_list[1].set_ylabel('gain (V/V)')
    ax_list[2].plot(vgs_vec, gamma_list)
    ax_list[2].set_ylabel(r'$\gamma_r$')
    ax_list[3].plot(vgs_vec, np.asarray(gm_list) * 1e3)
    ax_list[3].set_ylabel(r'$g_{mr}$ (mS)')
    ax_list[4].plot(vgs_vec, np.asarray(metric_list) * 1e3)
    ax_list[4].set_ylabel(r'$\gamma_r\cdot g_{mr}$ (mS)')
    ax_list[4].set_xlabel('Vgs (V)')
    plt.show(block=False)

    result = db.query(vbs=0, vds=vout-vs, vgs=vgs_best)
    scale = itarg / result['ibias']
    return scale, result

def design_amp(specs, input_op, load_op, load_scale):
    fstart = specs['fstart']
    fstop = specs['fstop']
    vsig = specs['vsig']
    temp = specs['noise_temp']
    snr_min = specs['snr_min']
    bw = specs['bw']
    cload = specs['cload']
    vdd = specs['vdd']
    vdst = specs['vdst_min']
    in_type = specs['in_type']
    k = 1.38e-23

    gm_i = input_op['gm']
    gds_i = input_op['gds']
    gamma_i = input_op['gamma']
    cdd_i = input_op['cdd']
    gm_l = load_op['gm'] * load_scale
    gds_l = load_op['gds'] * load_scale
    cdd_l = load_op['cdd'] * load_scale
    gamma_l = load_op['gamma']

    snr_linear = 10.0**(snr_min / 10)
    gds_tot = gds_i + gds_l
    cdd_tot = cdd_i + cdd_l
    gain = gm_i / gds_tot
    noise_const = gm_i / (gamma_i * gm_i + gamma_l * gm_l)
    print(gm_i, gm_l, gamma_i, gamma_l, noise_const)
    # get scale factor for BW-limited case
    scale_bw = max(1, 2 * np.pi * bw * cload / (gds_tot - 2 * np.pi * bw * cdd_tot))
    if fstart < 0:
        noise_const *= vsig**2 * gain / (4 * k * temp)
        cload_tot = snr_linear / noise_const
        rout = 1 / (2 * np.pi * bw * cload_tot)
        scale_noise = 1 / (gds_tot * rout)
        if scale_noise < scale_bw:
            print('BW-limited, scale_bw = %.4g, scale_noise = %.4g' % (scale_bw, scale_noise))
            # we are BW-limited, not noise limited
            scale = scale_bw
            cload_add = 0
        else:
            print('noise-limited.')
            scale = scale_noise
            cload_add = cload_tot - scale * (cdd_i + cdd_l) - cload
    else:
        noise_const *= vsig**2 / (16 * k * temp * (fstop - fstart))
        gm_final = snr_linear / noise_const
        scale_noise = gm_final / gm_i
        if scale_noise < scale_bw:
            print('BW-limited, scale_bw = %.4g, scale_noise = %.4g' % (scale_bw, scale_noise))
            # we are BW-limited, not noise limited
            scale = scale_bw
        else:
            print('noise-limited.')
            scale = scale_noise
        
        cload_add = 0
    
    # get number of segments
    seg_in = int(np.ceil(scale))
    print(seg_in, load_scale)
    seg_load = int(np.ceil(seg_in * load_scale))
    # recompute amplifier performance
    gm_i *= seg_in
    gds_i *= seg_in
    cdd_i *= seg_in
    gm_l = load_op['gm'] * seg_load
    gds_l = load_op['gds'] * seg_load
    cdd_l = load_op['cdd'] * seg_load
    
    gds_tot = gds_i + gds_l
    cdd_tot = cdd_i + cdd_l
    if in_type == 'nch':
        vincm = vdst + input_op['vgs']
    else:
        vincm = vdd - vdst + input_op['vgs']
    amp_specs = dict(
        ibias=input_op['ibias'] * seg_in * 2,
        gain=gm_i / gds_tot,
        bw=gds_tot / (2 * np.pi * (cload + cload_add + cdd_tot)),
        vincm=vincm,
        cload=cload + cload_add,
        )

    return seg_in, seg_load, amp_specs


def design_tail(specs, itarg, seg_min):
    """Find smallest tail transistor that biases the differential amplifier."""
    db = specs['in_db']
    sim_env = specs['sim_env']
    vds = specs['vdst_min']
    in_type = specs['in_type']
    
    if in_type == 'pch':
        vds *= -1

    ib_fun = db.get_function('ibias', env=sim_env)
    vgs_idx = db.get_fun_arg_index('vgs')
    vgs_min, vgs_max = ib_fun.get_input_range(vgs_idx)

    # binary search on number of fingers.
    seg_tail_iter = BinaryIterator(seg_min, None, step=2)
    while seg_tail_iter.has_next():
        seg_tail = seg_tail_iter.get_next()

        def fun_zero(vgs):
            farg = db.get_fun_arg(vgs=vgs, vds=vds, vbs=0)
            return ib_fun(farg) * seg_tail - itarg

        val_min = fun_zero(vgs_min)
        val_max = fun_zero(vgs_max)
        if val_min > 0 and val_max > 0:
            # smallest possible current > itarg
            seg_tail_iter.down()
        elif val_min < 0 and val_max < 0:
            # largest possbile current < itarg
            seg_tail_iter.up()
        else:
            vbias = sciopt.brentq(fun_zero, vgs_min, vgs_max)  # type: float
            seg_tail_iter.save_info(vbias)
            seg_tail_iter.down()

    seg_tail = seg_tail_iter.get_last_save()
    if seg_tail is None:
        raise ValueError('No solution for tail.')
    vgs_opt = seg_tail_iter.get_last_save_info()

    tail_op = db.query(vbs=0, vds=vds, vgs=vgs_opt)
    return seg_tail, tail_op


def run_main():
    interp_method = 'spline'
    sim_env = 'tt'
    nmos_spec = 'specs_mos_char/nch_w0d5.yaml'
    pmos_spec = 'specs_mos_char/pch_w0d5.yaml'
    intent = 'lvt'

    nch_db = get_db(nmos_spec, intent, interp_method=interp_method,
                    sim_env=sim_env)
    pch_db = get_db(pmos_spec, intent, interp_method=interp_method,
                    sim_env=sim_env)

    specs = dict(
        in_type='pch',
        sim_env=sim_env,
        in_db=pch_db,
        load_db=nch_db,
        cload=10e-15,
        vgs_res=5e-3,
        vdd=1.2,
        voutcm=0.6,
        vdst_min=0.2,
        vimax=0.25,
        gain_min=4.0,
        bw=10e9,
        snr_min=50,
        vsig=0.05,
        fstart=1.4e9,
        fstop=1.6e9,
        # fstart=-1,
        # fstop=-1,
        noise_temp=300,
        )

    input_op = design_input(specs)
    load_scale, load_op = design_load(specs, input_op)
    seg_in, seg_load, amp_specs = design_amp(specs, input_op, load_op, load_scale)
    seg_tail, tail_op = design_tail(specs, amp_specs['ibias'], seg_in * 2)
    print('amplifier performance:')
    pprint.pprint(amp_specs)
    for name, seg, op in (('input', seg_in, input_op),
                          ('load', seg_load, load_op),
                          ('tail', seg_tail, tail_op)):

        print('%s seg = %d' % (name, seg))
        print('%s op:' % name)
        pprint.pprint(op)
    

if __name__ == '__main__':
    run_main()
