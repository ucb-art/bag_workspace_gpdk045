import os
import pprint

import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

from bag import BagProject
from bag.io import load_sim_results, save_sim_results, load_sim_file

# Global variables
# schematic generator library/cell name
tb_lib = 'ee240b_mos_char'
tb_cell = 'tb_mos_ibias'
# library to create new schematics in
impl_lib = 'AAAFOO_hw1_tb_ibias'
# directory to save simulation data
data_dir = os.path.join('data', 'hw1')
# sweep values
mos_list = ['nch', 'pch']
thres_list = ['svt']
lch_list = [45e-9, ]

# create data directory if it doesn't exist
os.makedirs(data_dir, exist_ok=True)


def get_tb_name(mos_type, thres, lch):
    """Get generated testbench name from parameters."""
    lch_int = int(round(lch*1e9))
    return '%s_%s_%s_%dn' % (tb_cell, mos_type, thres, lch_int)


def characterize(prj):
    # iterate through all parameters combinations
    for mos_type in mos_list:
        for thres in thres_list:
            for lch in lch_list:
                # new generated testbench name
                tb_name = get_tb_name(mos_type, thres, lch)
                print('Creating testbench %s...' % tb_name)
                # create schematic generator
                tb_sch = prj.create_design_module(tb_lib, tb_cell)
                tb_sch.design(mos_type=mos_type, lch=lch, threshold=thres)
                tb_sch.implement_design(impl_lib, top_cell_name=tb_name)
                # copy and load ADEXL state of generated testbench
                tb_obj = prj.configure_testbench(impl_lib, tb_name)
                # make sure vgs/vds has correct sign
                if mos_type == 'nch':
                    tb_obj.set_parameter('vgs_start', 0)
                    tb_obj.set_parameter('vgs_stop', 1.2)
                    tb_obj.set_sweep_parameter('vds', start=0.0, stop=1.2, step=0.05)
                else:
                    tb_obj.set_parameter('vgs_start', -1.2)
                    tb_obj.set_parameter('vgs_stop', 0.0)
                    tb_obj.set_sweep_parameter('vds', start=-1.2, stop=0.0, step=0.05)
                # update testbench changes and run simulation
                tb_obj.update_testbench()
                print('Simulating testbench %s...' % tb_name)
                save_dir = tb_obj.run_simulation()
                # load simulation results into Python
                print('Simulation done, saving results')
                results = load_sim_results(save_dir)
                # save simulation results to data directory
                save_sim_results(results, os.path.join(data_dir, '%s.data' % tb_name))

    print('Characterization done.')


def print_data_info():
    tb_name = get_tb_name(mos_list[0], thres_list[0], lch_list[0])
    results = load_sim_file(os.path.join(data_dir, '%s.data' % tb_name))
    # show that results is a dictionary
    print('results type: %s' % type(results))
    print('results keys: %s' % list(results.keys()))
    # show values in the result dictionary
    vds = results['vds']
    vgs = results['vgs']
    ibias = results['ibias']
    # show how the sweep variables and outputs are structured
    print('vds type: {}, vds shape: {}'.format(type(vds), vds.shape))
    print('vgs type: {}, vgs shape: {}'.format(type(vgs), vgs.shape))
    print('ibias type: {}, ibias shape: {}'.format(type(ibias), ibias.shape))
    # show how to get sweep parameter order
    print('sweep_params:')
    pprint.pprint(results['sweep_params'])
    

    
def plot_data():
    # output data name
    output = 'ibias'

    # plot all ibias data
    fig_idx = 1
    for mos_type in mos_list:
        for thres in thres_list:
            for lch in lch_list:
                # load simulation results
                tb_name = get_tb_name(mos_type, thres, lch)
                results = load_sim_file(os.path.join(data_dir, '%s.data' % tb_name))
                # plot data
                plot_2d_data(mos_type, results, 'ibias', fig_idx)
                fig_idx += 1

    plt.show()


def plot_2d_data(mos_type, results, output, fig_idx):
    # get sweep variables name and order
    xvar_list = results['sweep_params'][output]
    # outer sweep variable 1D array
    xvec = results[xvar_list[0]]
    # inner sweep variable 1D array
    yvec = results[xvar_list[1]]
    # output 2D array
    zmat = results[output]
    # change 1D sweep array into 2D array with same size as results
    xmat, ymat = np.meshgrid(xvec, yvec, indexing='ij', copy=False)

    if mos_type == 'pch':
        # make sure bias current is positive
        zmat *= -1
    
    # plot 2D array output
    fig = plt.figure(fig_idx)
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(xmat, ymat, zmat, rstride=1, cstride=2, linewidth=0, cmap=cm.cubehelix)
    ax.set_xlabel(xvar_list[0])
    ax.set_ylabel(xvar_list[1])
    ax.set_zlabel(output)
    

if __name__ == '__main__':
    local_dict = locals()
    if 'bprj' not in local_dict:
        print('Creating BAG project')
        bprj = BagProject()
    else:
        print('Loading BAG project')
        bprj = local_dict['bprj']

    characterize(bprj)
    # print_data_info()
    # plot_data()
