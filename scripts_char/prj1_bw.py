import pprint
import matplotlib.pyplot as plt

import numpy as np
import scipy.interpolate as interp
import scipy.optimize as opt

from verification_ec.mos.query import MOSDBDiscrete


def get_db(spec_file, intent, interp_method='spline', sim_env='tt'):
    # initialize transistor database from simulation data
    mos_db = MOSDBDiscrete([spec_file], interp_method=interp_method)
    # set process corners
    mos_db.env_list = [sim_env]
    # set layout parameters
    mos_db.set_dsn_params(intent=intent)
    return mos_db


def get_wn(A, rf, ro, ci, cl):
    return np.sqrt((1 + A) / (rf * ro * ci * cl))


def get_w3db(wn, z):
    tmp = 2 * z**2 - 1
    return wn * np.sqrt(np.sqrt(tmp**2 + 1) - tmp)


def get_z(A, ro, rf, ci, cl):
    return 0.5 / np.sqrt(1 + A) * (np.sqrt(ro * ci / (rf * cl)) +
                                   np.sqrt(ro * cl / (rf * ci)) +
                                   np.sqrt(rf * ci / (ro * cl)))


def get_pm(z):
    return np.rad2deg(np.arctan(2 * z / np.sqrt(np.sqrt(1 + 4 * z**4) -
                                                2 * z**2)))


def get_opt_rf(ftarg, phase_margin, gm, ro, ci, cl):
    wtarg = 2 * np.pi * ftarg
    wtarg_log = np.log10(wtarg)
    A = gm * ro
    rf_vec = ro * np.logspace(-3, 3, 7001)
    rf_log_vec = np.log10(rf_vec)

    wn_vec = get_wn(A, rf_vec, ro, ci, cl)
    z_vec = get_z(A, ro, rf_vec, ci, cl)
    pm_vec = get_pm(z_vec)
    w3db_vec = get_w3db(wn_vec, z_vec)

    max_idx = np.argmax(w3db_vec)
    w3db_max = w3db_vec[max_idx]
    w3db_dc = w3db_vec[0]
    if w3db_max < wtarg:
        return np.nan, np.nan, np.nan, np.nan, np.nan

    w3db_log_vec = np.log10(w3db_vec)

    fun = interp.interp1d(rf_log_vec, w3db_log_vec - wtarg_log,
                          copy=False, assume_sorted=True,
                          bounds_error=False,
                          fill_value='extrapolate')

    # stability check
    rf_log_opt = None
    min_idx = np.argmin(pm_vec)
    min_pm = pm_vec[min_idx]
    if min_pm < phase_margin:
        fun_pm = interp.interp1d(rf_log_vec, pm_vec - phase_margin,
                                 copy=False, assume_sorted=True,
                                 bounds_error=False,
                                 fill_value='extrapolate')
        rf_log_min = opt.brentq(fun_pm, rf_log_vec[0], rf_log_vec[min_idx])
        rf_log_max = opt.brentq(fun_pm, rf_log_vec[min_idx], rf_log_vec[-1])
        if fun(rf_log_max) < 0 and fun(rf_log_min) > 0:
            # optimal Rf lies in the forbidden zone.  we're forced to use
            # rf_min
            rf_log_opt = rf_log_min
            
    if rf_log_opt is None:
        if w3db_dc < wtarg:
            a = rf_log_vec[0]
            b = rf_log_vec[max_idx]
        else:
            a = rf_log_vec[max_idx]
            b = rf_log_vec[-1]
    
        rf_log_opt = opt.brentq(fun, a, b)
    
    rf_opt = 10.0**rf_log_opt
    rdc_opt = (-A * rf_opt + ro) / (1 + A)
    z_opt = get_z(A, ro, rf_opt, ci, cl)
    pm_opt = get_pm(z_opt)
    wn_opt = get_wn(A, rf_opt, ro, ci, cl)
    f3db_opt = get_w3db(wn_opt, z_opt) / (2 * np.pi)

    return rf_opt, rdc_opt, z_opt, pm_opt, f3db_opt


def design_k(k, specs, tia_op):
    ftarg = specs['ftarg']
    phase_margin = specs['phase_margin']
    ci0 = specs['ci']
    cl0 = specs['cl']

    gm = tia_op['gm']
    ro = tia_op['ro']
    cgg = tia_op['cgg']
    cdd = tia_op['cdd']

    ci = ci0 + k * cgg
    cl = cl0 + k * cdd
    gm = k * gm
    ro = ro / k
    
    return get_opt_rf(ftarg, phase_margin, gm, ro, ci, cl)


def plot_vs_k(specs, tia_op):
    ftarg = specs['ftarg']
    k_list = np.logspace(0, 2, 201)

    rf_list, rdc_list, z_list, pm_list, f3db_list = [], [], [], [], []
    idx0 = 0
    for idx, k in enumerate(k_list):
        rf, rdc, z, pm, f3db = design_k(k, specs, tia_op)
        if np.isnan(rdc) or rdc > 0:
            idx0 = idx + 1
            del rf_list[:]
            del rdc_list[:]
            del z_list[:]
            del pm_list[:]
            del f3db_list[:]
        else:
            rf_list.append(rf)
            rdc_list.append(rdc)
            z_list.append(z)
            pm_list.append(pm)
            f3db_list.append(f3db)
    
    k_list = k_list[idx0:]
    gain_list = np.abs(rdc_list)
    max_idx = np.argmax(gain_list)

    fig, (ax0, ax1, ax3) = plt.subplots(3, sharex=True)
    title_str = 'Amplifier Performance v.s. size'
    ax0.set_title(title_str)
    ax0.loglog(k_list, rf_list, color='b', label='$R_f$')
    ax0.loglog(k_list, gain_list, color='g', label='$|R_{dc}|$')
    ax0.loglog([k_list[max_idx]], [rf_list[max_idx]], marker='o', color='b')
    ax0.loglog([k_list[max_idx]], [gain_list[max_idx]], marker='o', color='g')
    ax0.set_ylabel('Resistance ($\Omega$)')
    ax0.legend()

    ax1.loglog(k_list, f3db_list, '-bo')
    fmin = min(np.min(f3db_list), ftarg) * 0.95
    fmax = max(np.max(f3db_list), ftarg) * 1.05
    ax1.set_ylabel('$f_{3db}$ (Hz)')
    ax1.set_ylim([fmin, fmax])

    ax3.semilogx(k_list, z_list, 'b')
    ax3.semilogx([k_list[max_idx]], [z_list[max_idx]], marker='o', color='b')
    ax3.set_ylabel('$\zeta$', color='b')
    ax3.tick_params('y', colors='b')
    ax3.set_xlabel('Size')
    ax3 = ax3.twinx()
    ax3.semilogx(k_list, pm_list, 'g')
    ax3.semilogx([k_list[max_idx]], [pm_list[max_idx]], marker='o', color='g')
    ax3.semilogx(k_list, [45] * len(k_list), '--g')
    ax3.set_ylabel('$\phi_{PM}$ (deg)', color='g')
    ax3.tick_params('y', colors='g')
    
    fig.tight_layout()
    plt.show()


def get_tia_op(specs, ndb, pdb):
    vdd = specs['vdd']
    vincm = specs['vincm']

    nch_op = ndb.query(vbs=0, vds=vincm, vgs=vincm)
    pch_op = pdb.query(vbs=0, vds=vincm-vdd, vgs=vincm-vdd)

    ibias = nch_op['ibias']
    pscale = ibias / pch_op['ibias']
    gm = nch_op['gm'] + pch_op['gm'] * pscale
    gds = nch_op['gds'] + pch_op['gds'] * pscale
    cgg = nch_op['cgg'] + pch_op['cgg'] * pscale
    cdd = nch_op['cdd'] + pch_op['cdd'] * pscale

    return dict(
        ibias=ibias,
        pscale=pscale,
        gm=gm,
        gds=gds,
        ro=1.0/gds,
        cgg=cgg,
        cdd=cdd,
        gmro=gm/gds,
        )
        

def run_main():
    interp_method = 'spline'
    sim_env = 'tt'
    nmos_spec = 'specs_mos_char/nch_w0d5.yaml'
    pmos_spec = 'specs_mos_char/pch_w0d5.yaml'
    intent = 'lvt'

    specs = dict(
        vdd=1.2,
        vincm=0.6,
        ftarg=5e9,
        phase_margin=45,
        ci=100e-15,
        cl=40e-15,
        )

    pch_db = get_db(pmos_spec, intent, interp_method=interp_method,
                    sim_env=sim_env)
    nch_db = get_db(nmos_spec, intent, interp_method=interp_method,
                    sim_env=sim_env)

    tia_op = get_tia_op(specs, nch_db, pch_db)
    pprint.pprint(tia_op)
    
    plot_vs_k(specs, tia_op)

if __name__ == '__main__':
    np.seterr(all='raise')
    run_main()
