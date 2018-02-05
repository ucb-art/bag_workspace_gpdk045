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
    vgs_res = specs['vgs_res']
    bot_gain_min = specs['bot_gain_min']
    casc_scale_max = specs['casc_scale_max']
    casc_scale_step = specs['casc_scale_step']
    casc_bias_step = specs['casc_bias_step']
    in_type = specs['in_type']

    casc_scale_list = np.arange(1, casc_scale_max + casc_scale_step / 2, casc_scale_step)
    if in_type == 'nch':
        vb = 0
        vtail = vdst
        casc_bias_list = np.arange(voutcm, vdd + casc_bias_step / 2, casc_bias_step)
    else:
        vb = specs['vdd']
        vtail = vb - vdst
        casc_bias_list = np.arange(0, voutcm + casc_bias_step / 2, casc_bias_step)

    ib_fun = db.get_function('ibias')
    gm_fun = db.get_function('gm')

    
    vgs_idx = db.get_fun_arg_index('vgs')
    vgs_min, vgs_max = ib_fun.get_input_range(vgs_idx)
    num_points = int(np.ceil((vgs_max - vgs_min) / vgs_res)) + 1
    vgs_val_list = np.linspace(vgs_min, vgs_max, num_points, endpoint=True)
    def fun_zero1(vmid, vgs_bot):
        barg = db.get_fun_arg(vgs=vgs_bot, vds=vmid-vtail, vbs=vb-vtail)
        return (2 * ib_fun(barg) / gm_fun(barg) - vstar)*1e3

    def fun_zero2(cb, cs, vmid, itarg):
        carg = db.get_fun_arg(vgs=cb-vmid, vds=voutcm-vmid, 
                              vbs=vb-vmid)
        return (ib_fun(carg) * cs - itarg)*1e6

    metric_best = 0
    op_best = None

    if in_type == 'nch':
        vmid_bounds = (vtail + 5e-3, voutcm - 5e-3)
    else:
        vmid_bounds = (voutcm + 5e-3, vtail - 5e-3)

    bot_gain_max = 0
    for vgs_val in vgs_val_list:
        try:
            vmid = sciopt.brentq(fun_zero1, vmid_bounds[0], vmid_bounds[1], args=(vgs_val,))
        except ValueError:
            continue

        barg = db.get_fun_arg(vgs=vgs_val, vds=vmid-vtail, vbs=vb-vtail)
        itarg = ib_fun(barg)
        if in_type == 'nch':
            cb_min, cb_max = vmid + vgs_min, vdd
        else:
            cb_min, cb_max = 0, vmid + vgs_max

        for casc_scale in casc_scale_list:
            args2 = (casc_scale, vmid, itarg)
            try:
                casc_bias = sciopt.brentq(fun_zero2, cb_min, cb_max, args=args2)
            except ValueError:
                continue

            cres = db.query(vgs=casc_bias-vmid, 
                            vds=voutcm-vmid,
                            vbs=vb-vmid)
            bres = db.query(vgs=vgs_val,
                            vds=vmid-vtail,
                            vbs=vb-vtail)
            metric_cur = bres['gm'] / bres['gamma'] / bres['ibias']
            if bres['gm'] / bres['gds'] >= bot_gain_min:
                if metric_cur > metric_best:
                    metric_best = metric_cur
                    op_best = casc_scale, cres, bres
            else:
                bot_gain_max = max(bot_gain_max, bres['gm'] / bres['gds'])

    if op_best is None:
        raise ValueError('No solutions found.  bot_gain_max = %.4g' % bot_gain_max)

    casc_scale, cres, bres = op_best
    gmb = bres['gm']
    gdsb = bres['gds']
    ibias = bres['ibias']
    gamma = bres['gamma']
    gmc = cres['gm'] * casc_scale
    gdsc = cres['gds'] * casc_scale
    cddc = cres['cdd'] * casc_scale
    input_op = dict(
        casc_scale=casc_scale,
        casc_bias=vb - cres['vbs'] + cres['vgs'],
        vincm=vb - bres['vbs'] + bres['vgs'],
        vin_mid=vb - bres['vbs'] + bres['vds'],
        ibias=ibias,
        gm=gmb,
        gmc=gmc,
        gdsb=gdsb,
        gdsc=gdsc,
        gds=gdsb * gdsc / (gdsb + gdsc + gmc),
        cdd=cddc,
        gamma=gamma,
        )

    return input_op


def design_load(specs, input_op):
    """Design load.

    Sweep vgs.  For each vgs, compute gain and max bandwidth.  If
    both gain and BW specs are met, pick operating point that maximizes
    gamma_r * gm_r
    """
    db = specs['load_db']
    sim_env = specs['sim_env']
    vdd = specs['vdd']
    voutcm = specs['voutcm']
    vgs_res = specs['vgs_res']
    gain_min = specs['gain_min']
    bw = specs['bw']
    bot_gain_min = specs['bot_gain_min']
    casc_scale_max = specs['casc_scale_max']
    casc_scale_step = specs['casc_scale_step']
    casc_bias_step = specs['casc_bias_step']
    in_type = specs['in_type']
    
    casc_scale_list = np.arange(1, casc_scale_max + casc_scale_step / 2, casc_scale_step)
    if in_type == 'nch':
        vs = specs['vdd']
        casc_bias_list = np.arange(0, voutcm + casc_bias_step / 2, casc_bias_step)
    else:
        vs = 0
        casc_bias_list = np.arange(voutcm, vdd + casc_bias_step / 2, casc_bias_step)
                                   
    gm_fun = db.get_function('gm', env=sim_env)
    gds_fun = db.get_function('gds', env=sim_env)
    cdd_fun = db.get_function('cdd', env=sim_env)
    gamma_fun = db.get_function('gamma', env=sim_env)
    ib_fun = db.get_function('ibias', env=sim_env)

    vgs_idx = db.get_fun_arg_index('vgs')
    vgs_min, vgs_max = ib_fun.get_input_range(vgs_idx)
    num_points = int(np.ceil((vgs_max - vgs_min) / vgs_res)) + 1

    gm_i = input_op['gm']
    gm_ic = input_op['gmc']
    gds_ib = input_op['gdsb']
    gds_ic = input_op['gdsc']
    itarg = input_op['ibias']
    gds_i = input_op['gds']
    cdd_i = input_op['cdd']

    def fun_zero(vmid, vgs_bot, cs, cb):
        carg = db.get_fun_arg(vgs=cb-vmid, vds=voutcm-vmid, 
                              vbs=vs-vmid)
        barg = db.get_fun_arg(vgs=vgs_bot, vds=vmid-vs, 
                              vbs=0)
        return (ib_fun(carg) * cs - ib_fun(barg))*1e6

    best_ans = None
    metric_best = float('inf')
    gain_max = 0
    bw_max = 0
    for vgs_val in np.linspace(vgs_min, vgs_max, num_points, endpoint=True):
        for casc_scale in casc_scale_list:
            for casc_bias in casc_bias_list:
                # here
                args = (vgs_val, casc_scale, casc_bias)
                fout1 = fun_zero(voutcm, *args)
                fout2 = fun_zero(vs, *args)
                if fout1 * fout2 > 0:
                    # no solution
                    print('no solution for %s' % args)
                    continue
                vmid = sciopt.brentq(fun_zero, voutcm, vs, args=args)
                
                barg = db.get_fun_arg(vgs=vgs_val, vds=vmid-vs, vbs=0)
                carg = db.get_fun_arg(vgs=casc_bias-vmid, vds=voutcm-vmid,
                                      vbs=vs-vmid)
                load_scale = itarg / ib_fun(barg)
                gm_b = gm_fun(barg)
                gds_b = gds_fun(barg)
                gamma_b = gamma_fun(barg)
                gm_c = gm_fun(carg) * casc_scale
                gds_c = gds_fun(carg) * casc_scale
                cdd_c = cdd_fun(carg) * casc_scale

                gm_l = gm_b * load_scale
                gds_l = gds_b * gds_c / (gds_b + gds_c + gm_c) * load_scale
                cdd_l = cdd_c * load_scale
                gamma_l = gamma_b

                bw_cur = (gds_l + gds_i) / (cdd_i + cdd_l) / 2 / np.pi
                gain_cur = gm_i / ((gds_l + gds_ic)/(gm_ic + gds_ic)*(gm_ic + gds_ic + gds_ib) - gds_ic)
                metric_cur = gamma_l * gm_l
                
                if gm_b / gds_b >= bot_gain_min and gain_cur >= gain_min and bw_cur >= bw:
                    if metric_cur < metric_best:
                        metric_best = metric_cur
                        best_ans = dict(
                            casc_scale=casc_scale,
                            casc_bias=casc_bias,
                            vload=vgs_val + vs,
                            vload_mid=vmid,
                            load_scale=load_scale,
                            gm=gm_l,
                            gds=gds_l,
                            cdd=cdd_l,
                            gamma=gamma_l,
                            )
                else:
                    gain_max = max(gain_max, gain_cur)
                    bw_max = max(bw_max, bw_cur)

    if best_ans is None:
        raise ValueError('No solution.  max gain = %.4g, '
                         'max bw = %.4g' % (gain_max, bw_max))
    
    return best_ans

def design_amp(specs, input_op, load_op):
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

    ibias = input_op['ibias']
    gm_i = input_op['gm']
    gds_i = input_op['gds']
    gds_ic = input_op['gdsc']
    gds_ib = input_op['gdsb']
    gm_ic = input_op['gmc']
    gamma_i = input_op['gamma']
    cdd_i = input_op['cdd']
    gm_l = load_op['gm']
    gds_l = load_op['gds']
    cdd_l = load_op['cdd']
    gamma_l = load_op['gamma']
    load_scale = load_op['load_scale']

    snr_linear = 10.0**(snr_min / 10)
    gds_tot = gds_i + gds_l
    cdd_tot = cdd_i + cdd_l
    gain = gm_i / ((gds_l + gds_ic)/(gm_ic + gds_ic)*(gm_ic + gds_ic + gds_ib) - gds_ic)
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
    gds_ic *= seg_in
    gm_ic *= seg_in
    gds_ib *= seg_in
    gm_l = gm_l / load_scale * seg_load
    gds_l = gds_l / load_scale * seg_load
    cdd_l = cdd_l / load_scale * seg_load
    
    gds_tot = gds_i + gds_l
    cdd_tot = cdd_i + cdd_l
    gain = gm_i / ((gds_l + gds_ic)/(gm_ic + gds_ic)*(gm_ic + gds_ic + gds_ib) - gds_ic)
    amp_specs = dict(
        seg_in=seg_in,
        seg_load=seg_load,
        ibias=ibias * seg_in * 2,
        gain=gain,
        bw=gds_tot / (2 * np.pi * (cload + cload_add + cdd_tot)),
        cload=cload + cload_add,
        )

    return amp_specs


def design_tail(specs, amp_specs):
    """Find smallest tail transistor that biases the differential amplifier."""
    db = specs['in_db']
    sim_env = specs['sim_env']
    vds = specs['vdst_min']
    in_type = specs['in_type']

    itarg = amp_specs['ibias']
    seg_min = amp_specs['seg_in'] * 2
    
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
    tail_op['seg_tail'] = seg_tail
    return tail_op


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
        bot_gain_min=4.8,
        in_type='pch',
        sim_env=sim_env,
        in_db=pch_db,
        load_db=nch_db,
        cload=10e-15,
        vgs_res=5e-3,
        vdd=1.2,
        voutcm=0.6,
        vdst_min=0.15,
        vimax=0.15,
        gain_min=30.0,
        bw=0.5e9,
        snr_min=50,
        vsig=0.01,
        fstart=0.1e9,
        fstop=0.3e9,
        # fstart=-1,
        # fstop=-1,
        noise_temp=300,
        casc_scale_max=4,
        casc_scale_step=0.5,
        casc_bias_step=0.1,
        )

    input_op = design_input(specs)
    load_op = design_load(specs, input_op)
    amp_specs = design_amp(specs, input_op, load_op)
    tail_op = design_tail(specs, amp_specs)
    print('amplifier performance:')
    pprint.pprint(amp_specs)
    for name, op in (('input', input_op),
                     ('load', load_op),
                     ('tail', tail_op)):
        print('%s op:' % name)
        pprint.pprint(op)
    

if __name__ == '__main__':
    run_main()
