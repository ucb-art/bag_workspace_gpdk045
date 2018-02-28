import pprint

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


def design_load(specs, db):
    sim_env = specs['sim_env']
    vgs_res = specs['vgs_res']
    vdd = specs['vdd']
    voutcm = specs['voutcm']
    vstar_in = specs['vstar_in']
    vstar_load = specs['vstar_pload']
    casc_scale = specs['casc_pscale']

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

        return bot_op, top_op


def design_input(specs, db, idb):
    sim_env = specs['sim_env']
    vgs_res = specs['vgs_res']
    vdd = specs['vdd']
    vincm = specs['vincm']
    voutcm = specs['voutcm']
    vstar_in = specs['vstar_in']
    vstar_load = specs['vstar_nload']
    casc_scale = specs['casc_nscale']

    vgs_idx = db.get_fun_arg_index('vgs')
    vstar_fun = db.get_function('vstar', env=sim_env)
    ibias_fun = db.get_function('ibias', env=sim_env)
    vgs_min, vgs_max = vstar_fun.get_input_range(vgs_idx)

    def zero_fun1(vm, vgs):
        arg = db.get_fun_arg(vgs=vgs, vds=vm, vbs=0)
        return (vstar_fun(arg) - vstar_load) * 1e3

    def zero_fun2(vc, vm, itarg):
        arg = db.get_fun_arg(vgs=vc-vm, vds=voutcm-vm, vbs=-vm)
        return (ibias_fun(arg) * casc_scale - itarg) * 1e6

    vstari_fun = idb.get_function('vstar', env=sim_env)
    vgs_idx = idb.get_fun_arg_index('vgs')
    vgsi_min, vgsi_max = vstari_fun.get_input_range(vgs_idx)
    def zero_fun3(vs, vm):
        arg = idb.get_fun_arg(vgs=vincm-vs, vds=vm-vs, vbs=vdd-vs)
        return (vstari_fun(arg) - vstar_in) * 1e3

    num_pts = int(np.ceil((vgs_max - vgs_min) / vgs_res)) + 1
    vgs_vec = np.linspace(vgs_min, vgs_max, num_pts, endpoint=True)
    for vgs_val in vgs_vec[::-1]:
        fout1 = zero_fun1(5e-3, vgs_val)
        fout2 = zero_fun1(voutcm - 5e-3, vgs_val)
        if fout1 * fout2 > 0:
            continue
        vmid = opt.brentq(zero_fun1, 5e-3, voutcm - 5e-3, args=(vgs_val,))
        if vmid + vgs_min >= vdd:
            continue
        bot_op = db.query(vbs=0, vds=vmid, vgs=vgs_val)
        ib = bot_op['ibias']
        args = (vmid, ib)
        fout1 = zero_fun2(vmid + vgs_min, *args)
        fout2 = zero_fun2(vdd, *args)
        if fout1 * fout2 > 0:
            continue
        vcasc = opt.brentq(zero_fun2, vmid + vgs_min, vdd, args=args)
        top_op = db.query(vbs=-vmid, vds=voutcm-vmid, vgs=vcasc-vmid)

        args = (vmid, )
        fout1 = zero_fun3(vincm - vgsi_max, *args)
        fout2 = zero_fun3(vdd - 5e-3, *args)
        if fout1 * fout2 > 0:
            continue
        vtail = opt.brentq(zero_fun3, vincm - vgsi_max, vdd - 5e-3, args=args)
        in_op = idb.query(vbs=vdd-vtail, vds=vmid-vtail, vgs=vincm-vtail)

        return in_op, bot_op, top_op


def design_amp(specs, in_op, nbot_op, ntop_op, pbot_op, ptop_op):
    
    casc_nscale = specs['casc_nscale']
    casc_pscale = specs['casc_pscale']

    nbot_fg = 2
    ibias = nbot_op['ibias']
    ntop_fg = casc_nscale
    pbot_fg = ibias / pbot_op['ibias']
    ptop_fg = pbot_fg * casc_pscale
    in_fg = ibias / in_op['ibias']

    scale = 3
    cir = LTICircuit()
    cir.add_transistor(in_op, 'x', 'in', 'gnd', 'gnd', fg=in_fg * scale)
    cir.add_transistor(nbot_op, 'x', 'gnd', 'gnd', 'gnd', fg=nbot_fg * scale)
    cir.add_transistor(ntop_op, 'out', 'gnd', 'x', 'gnd', fg=ntop_fg * scale)
    cir.add_transistor(ptop_op, 'out', 'gnd', 'm', 'gnd', fg=ptop_fg * scale)
    cir.add_transistor(pbot_op, 'm', 'gnd', 'gnd', 'gnd', fg=pbot_fg * scale)
    cir.add_cap(20e-15 * scale, 'out', 'gnd')
    cur_tf = cir.get_transfer_function('in', 'out', in_type='v')
    gain = cur_tf.num[-1] / cur_tf.den[-1]
    p1, p2, p3 = sorted(-cur_tf.poles)
    p1 /= 2 * np.pi
    p2 /= 2 * np.pi
    p3 /= 2 * np.pi

    for name, op, fg in [('in', in_op, in_fg), ('nbot', nbot_op, nbot_fg), ('ntop', ntop_op, ntop_fg),
                         ('ptop', ptop_op, ptop_fg), ('pbot', pbot_op, pbot_fg)]:
        print('%s fg : \t%.3g' % (name, fg * scale))
        for val in ('gm', 'gds', 'gb', 'ibias'):
            if val in op:
                print('%s %s : \t%.6g' % (name, val, op[val] * scale * fg))

    print('vtail: %.6g' % (specs['vincm'] - in_op['vgs']))
    print('vmidn: %.6g' % (nbot_op['vds']))
    print('vmidp: %.6g' % (specs['vdd'] + pbot_op['vds']))
    print('vb1: %.6g' % (specs['vdd'] + pbot_op['vgs']))
    print('vb2: %.6g' % (specs['vdd'] + pbot_op['vds'] + ptop_op['vgs']))
    print('vb3: %.6g' % (nbot_op['vds'] + ntop_op['vgs']))
    print('vb4: %.6g' % (nbot_op['vgs']))

    print(gain, p1/1e9, p2/1e9, p3/1e9, nbot_fg * ibias)

def run_main():
    interp_method = 'linear'
    sim_env = 'tt'
    nmos_spec = 'specs_mos_char/nch_w0d5.yaml'
    pmos_spec = 'specs_mos_char/pch_w0d5.yaml'
    intent = 'lvt'

    pch_db = get_db(pmos_spec, intent, interp_method=interp_method,
                    sim_env=sim_env)
    nch_db = get_db(nmos_spec, intent, interp_method=interp_method,
                    sim_env=sim_env)

    specs = dict(
        sim_env=sim_env,
        vgs_res=5e-3,
        vdd=1.2,
        vincm=0.6,
        voutcm=0.6,
        tot_err=0.04,
        t_settle=10e-9,
        vstar_in=110e-3,
        casc_pscale=2.0,
        casc_nscale=2.0,
        vstar_pload=300e-3,
        vstar_nload=300e-3,
        )
    
    pbot_op, ptop_op = design_load(specs, pch_db)
    in_op, nbot_op, ntop_op = design_input(specs, nch_db, pch_db)
    
    design_amp(specs, in_op, nbot_op, ntop_op,
               pbot_op, ptop_op)
    print(nbot_op['vgs'])
    print(ntop_op['vgs'] + nbot_op['vds'])
    print(pbot_op['vgs'] + specs['vdd'])
    print(ptop_op['vgs'] + pbot_op['vds'] + specs['vdd'])

if __name__ == '__main__':
    run_main()
