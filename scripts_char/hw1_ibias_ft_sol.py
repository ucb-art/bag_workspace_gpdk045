import os
import pprint

import numpy as np
from scipy.interpolate import RectBivariateSpline

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

from bag import BagProject
from bag.io import load_sim_results, save_sim_results, load_sim_file

# Global variables
# schematic generator library/cell name
tb_lib = 'ee240b_mos_char'
tb_cell = 'tb_mos_ft'
# library to create new schematics in
impl_lib = 'AAAFOO_hw1_tb_ft'
# directory to save simulation data
data_dir = os.path.join('data', 'hw1')
# sweep values
mos_list = ['nch', 'pch']
thres_list = ['lvt']
lch_list = [45e-9, 90e-9, 180e-9]

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
                    tb_obj.set_sweep_parameter('vgs', start=0.0, stop=1.2, step=0.02)
                    tb_obj.set_sweep_parameter('vds', start=0.0, stop=1.2, step=0.10)
                else:
                    tb_obj.set_sweep_parameter('vgs', start=-1.2, stop=0.0, step=0.02)
                    tb_obj.set_sweep_parameter('vds', start=-1.2, stop=0.0, step=0.10)
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
    
def plot_data():
    # output data name
    output_names = ['ft']

    # plot all ibias data
    fig_idx = 1
    for mos_type in mos_list:
        for thres in thres_list:
            for lch in lch_list[:1]:
                # load simulation results
                tb_name = get_tb_name(mos_type, thres, lch)
                results = load_sim_file(os.path.join(data_dir, '%s.data' % tb_name))
                # plot data
                for output in output_names:
                    plot_2d_data(mos_type, results, output, fig_idx)
                    fig_idx += 1

    plt.show()

def compute_ft():
    for mos_type in mos_list:
        for thres in thres_list:
            for lch in lch_list:
                # load simulation results
                tb_name = get_tb_name(mos_type, thres, lch)
                fname = os.path.join(data_dir, '%s.data' % tb_name)
                results = load_sim_file(fname)
                info = compute_ft_helper(mos_type, results)
                for key, val in info.items():
                    results[key] = val
                    results['sweep_params'][key] = results['sweep_params'][output]
                save_sim_results(results, fname)


def compute_ft_helper(mos_type, results):
    # get sweep variables name and order
    xvar_list = results['sweep_params']['ig']
    xname, yname, fname = xvar_list
    # outer sweep variable 1D array
    xvec = results[xname]
    # inner sweep variable 1D array
    yvec = results[yname]
    # frequency 1D array
    fvec = results[fname]

    # get magnitude 3D array
    igmat = results['ig']
    idsmat = results['ids']
    mag_mat = np.abs(idsmat / igmat)
    # get |mag - 1|
    abs_diff_mat = np.abs(mag_mat - 1)
    # get ft matrix
    arg_mat = np.argmin(abs_diff_mat, axis=2)
    # broadcast frequency 1D array to 3D array
    _, _, fmat = np.meshgrid(xvec, yvec, fvec, indexing='ij', copy=False)
    # get frequencies at which TF magnitude is closest to 1
    fmat = fmat[arg_mat]
    return dict(ft=fmat)

def plot_2d_data(mos_type, results, output, fig_idx):
    if output in results:
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

        print(output, np.max(zmat))
        if output == 'ibias' and mos_type == 'pch':
            # make sure bias current is positive
            zmat = -zmat
    
        # plot 2D array output
        fig = plt.figure(fig_idx)
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(xmat, ymat, zmat, rstride=1, cstride=2, linewidth=0, cmap=cm.cubehelix)
        ax.set_xlabel(xvar_list[0])
        ax.set_ylabel(xvar_list[1])
        ax.set_zlabel(output)
    else:
        print('output %s not found' % output)
    

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
    compute_ft()
    plot_data()
    
