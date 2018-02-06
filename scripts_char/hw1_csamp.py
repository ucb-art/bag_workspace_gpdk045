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

def design_cs_amp(db, sim_env, vgs_res, vds_res,
                  vdd, cload, gain_min, fbw):
    """Design resistively-loaded common-source amplifier.
    
    This algorithm sweeps all combination of Vgs and Vds,
    then pick the solution that meets specs.
    """
    ib_fun = db.get_function('ibias', env=sim_env)
    
    # get Vgs/Vds sweep values
    vgs_idx = db.get_fun_arg_index('vgs')
    vds_idx = db.get_fun_arg_index('vds')
    vgs_min, vgs_max = ib_fun.get_input_range(vgs_idx)
    vds_min, vds_max = ib_fun.get_input_range(vds_idx)
    vds_max = min(vds_max, vdd - vds_res)
    num_vgs = int(np.ceil((vgs_max - vgs_min) / vgs_res)) + 1
    num_vds = int(np.ceil((vds_max - vds_min) / vds_res)) + 1
    vgs_vec = np.linspace(vgs_min, vgs_max, num_vgs, endpoint=True)
    vds_vec = np.linspace(vds_min, vds_max, num_vds, endpoint=True)

    # find best operating point
    best_ibias = float('inf')
    best_op = None
    for vgs in vgs_vec:
        for vds in vds_vec:
            # get SS parameters
            farg = db.get_fun_arg(vgs=vgs, vds=vds, vbs=0)
            op_info = db.query(vgs=vgs, vds=vds, vbs=0)
            ibias = op_info['ibias']
            gds = op_info['gds']
            cdd = op_info['cdd']
            gm = op_info['gm']

            rload = (vdd - vds) / ibias
            gds_tot = gds + 1 / rload
            gain_cur = gm / gds_tot
            if gain_cur < gain_min:
                continue
            fbw_max = 1.0 / (2 * np.pi * rload * cdd)
            if fbw_max <= fbw:
                continue
            scale = int(np.ceil((cload / cdd) / (fbw_max / fbw - 1)))
            ibias_cur = ibias * scale
            if ibias_cur < best_ibias:
                best_ibias = ibias_cur
                best_op = dict(
                    vgs=vgs,
                    vds=vds,
                    nf=scale,
                    rload=rload / scale,
                    ibias=ibias_cur,
                    gain=gain_cur,
                    bw=gds_tot * scale / (2 * np.pi * (cload + scale * cdd)),
                    gm=gm * scale,
                    gds=gds * scale,
                    cdd=cdd * scale,
                    )
                pprint.pprint(best_op)

    if best_op is None:
        raise ValueError('No solutions.')

    return best_op


def run_main():
    interp_method = 'spline'
    sim_env = 'tt'
    nmos_spec = 'specs_mos_char/nch_w0d5.yaml'
    intent = 'lvt'

    nch_db = get_db(nmos_spec, intent, interp_method=interp_method,
                    sim_env=sim_env)

    specs = dict(
        sim_env=sim_env,
        db=nch_db,
        vgs_res=5e-3,
        vds_res=10e-3,
        vdd=1.2,
        cload=50e-15,
        gain_min=2.0,
        fbw=4e9,
        )

    amp_specs = design_cs_amp(**specs)
    pprint.pprint(amp_specs)
    print('done')

if __name__ == '__main__':
    run_main()
