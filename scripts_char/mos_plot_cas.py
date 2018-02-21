# -*- coding: utf-8 -*-

import pprint

import numpy as np
import matplotlib.pyplot as plt
# noinspection PyUnresolvedReferences
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import ticker

from verification_ec.mos.query import MOSDBDiscrete

def get_db(spec_file, intent, interp_method='spline', sim_env='tt'):
    # initialize transistor database from simulation data
    mos_db = MOSDBDiscrete([spec_file], interp_method=interp_method,
                           is_schematic=True, width_var='wb')
    # set process corners
    mos_db.env_list = [sim_env]
    # set layout parameters
    mos_db.set_dsn_params(intent=intent)
    return mos_db


def plot_data(db, name='ibias', bounds=None, unit_val=None, unit_label=None,
              vbs=0.0, nvds=41, nvgs=81, fig_idx=1):
    """Get interpolation function and plot/query."""
    # get function from database
    f = db.get_function(name)
    # get input range from function
    vds_min, vds_max = f.get_input_range(1)
    vgs_min, vgs_max = f.get_input_range(2)
    if bounds is not None:
        if 'vgs' in bounds:
            v0, v1 = bounds['vgs']
            if v0 is not None:
                vgs_min = max(vgs_min, v0)
            if v1 is not None:
                vgs_max = min(vgs_max, v1)
        if 'vds' in bounds:
            v0, v1 = bounds['vds']
            if v0 is not None:
                vds_min = max(vds_min, v0)
            if v1 is not None:
                vds_max = min(vds_max, v1)

    # evaluate function
    arg = [vbs, (vds_min + vds_max) / 2, (vgs_min + vgs_max) / 2]
    print('Evaluate %s at: %s' % (name, arg))
    print(f(arg))

    # this function also supports Numpy broadcasting,
    # so you can evaluate at multiple points at once.
    # we use this feature to efficiently plot the function
    vbs_vec = [vbs]
    vds_vec = np.linspace(vds_min, vds_max, nvds, endpoint=True)
    vgs_vec = np.linspace(vgs_min, vgs_max, nvgs, endpoint=True)
    vbs_mat, vds_mat, vgs_mat = np.meshgrid(vbs_vec, vds_vec, vgs_vec, indexing='ij', copy=False)
    arg = np.stack((vbs_mat, vds_mat, vgs_mat), axis=-1)
    ans = f(arg)

    # reshape to remove vbs dimension
    vds_mat = vds_mat.reshape((nvds, nvgs))
    vgs_mat = vgs_mat.reshape((nvds, nvgs))
    ans = ans.reshape((nvds, nvgs))

    # generate 3D plot
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-2, 3))
    if unit_label is not None:
        zlabel = '%s (%s)' % (name, unit_label)
    else:
        zlabel = name

    fig = plt.figure(fig_idx)
    ax = fig.add_subplot(111, projection='3d')
    if unit_val is not None:
        ans = ans / unit_val
    ax.plot_surface(vds_mat, vgs_mat, ans, rstride=1, cstride=1, linewidth=0, cmap=cm.cubehelix)
    ax.set_title(name)
    ax.set_xlabel('Vds (V)')
    ax.set_ylabel('Vgs (V)')
    ax.set_zlabel(zlabel)
    ax.w_zaxis.set_major_formatter(formatter)


def run_main():
    interp_method = 'spline'
    sim_env = 'tt'
    nmos_spec = 'specs_mos_char/nch_w0d5_casc.yaml'

    intent = 'lvt'
    
    db = get_db(nmos_spec, intent, interp_method=interp_method, sim_env=sim_env)

    plot_data(db, 'ibias', fig_idx=1)
    plot_data(db, 'gm', fig_idx=2)
    plot_data(db, 'gds', fig_idx=3)
    plot_data(db, 'vstar', fig_idx=4)
    plot_data(db, 'cdd', fig_idx=5)
    plt.show()

if __name__ == '__main__':
    run_main()
