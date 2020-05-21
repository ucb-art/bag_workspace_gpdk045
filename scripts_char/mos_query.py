# -*- coding: utf-8 -*-

import pprint

from verification.mos.query import MOSDBDiscrete

def get_db(spec_file, intent, interp_method='spline', sim_env='tt'):
    # initialize transistor database from simulation data
    mos_db = MOSDBDiscrete([spec_file], interp_method=interp_method,
                           is_schematic=True)
    # set process corners
    mos_db.env_list = [sim_env]
    # set layout parameters
    mos_db.set_dsn_params(intent=intent)
    return mos_db

def run_main():
    interp_method = 'spline'
    sim_env = 'tt'
    nmos_spec = 'specs_mos_char/nch_w0d5_90n.yaml'

    intent = 'lvt'
    
    db = get_db(nmos_spec, intent, interp_method=interp_method, sim_env=sim_env)

    pprint.pprint(db.query(vbs=0, vds=0.45, vgs=0.45))
    pprint.pprint(db.query(vbs=0, vds=0.5, vgs=0.5))
    pprint.pprint(db.query(vbs=0.15, vds=0.55, vgs=0.55))

if __name__ == '__main__':
    run_main()
