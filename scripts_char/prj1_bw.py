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


def get_gain(A, rf, ro):
    return (-A * rf + ro) / (1 + A)


def get_wn_and_z(A, rf, ro, ci, cl, cf):
    wn = np.sqrt((1 + A) / (rf * ro * (ci * cl + cl * cf + ci * cf)))
    z = wn / 2 * (rf * cf + ro * cl + ro * ci + rf * ci + A * rf * cf) / (1 + A)
    return wn, z


def get_w3db(wn, z):
    tmp = 2 * z**2 - 1
    return wn * np.sqrt(np.sqrt(tmp**2 + 1) - tmp)


def get_pm(z):
    return np.rad2deg(np.arctan(2 * z / np.sqrt(np.sqrt(1 + 4 * z**4) -
                                                2 * z**2)))


def get_vn_std_ber(A, rf, ro, ci, cl, cf, noise_gm):
    k = 1.38e-23
    T = 300

    wn, z = get_wn_and_z(A, rf, ro, ci, cl, cf)
    gm = A / ro

    v1 = 4 * k * T * noise_gm * (ro/(1 + A))**2 * wn / (8*z) * (1 + (wn*rf*ci)**2)
    v2 = 4 * k * T / rf * (A * rf / (1 + A))**2 * wn / (8*z) * (1 + (wn*ci/gm)**2)
    return np.sqrt(v1 + v2) * 7


def get_opt_rf(specs, gm, ro, ci, cl, cf, noise_gm):
    ftarg = specs['ftarg']
    phase_margin = specs['phase_margin']
    
    wtarg = 2 * np.pi * ftarg
    wtarg_log = np.log10(wtarg)
    A = gm * ro
    rf_vec = ro * np.logspace(-3, 3, 7001)
    rf_log_vec = np.log10(rf_vec)

    wn_vec, z_vec = get_wn_and_z(A, rf_vec, ro, ci, cl, cf)
    pm_vec = get_pm(z_vec)
    w3db_vec = get_w3db(wn_vec, z_vec)

    max_idx = np.argmax(w3db_vec)
    w3db_max = w3db_vec[max_idx]
    w3db_dc = w3db_vec[0]
    if w3db_max < wtarg:
        return (np.nan,) * 6

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
    rdc_opt = get_gain(A, rf_opt, ro)
    wn_opt, z_opt = get_wn_and_z(A, rf_opt, ro, ci, cl, cf)
    pm_opt = get_pm(z_opt)
    f3db_opt = get_w3db(wn_opt, z_opt) / (2 * np.pi)
    vn_std = get_vn_std_ber(A, rf_opt, ro, ci, cl, cf, noise_gm)

    return rf_opt, rdc_opt, z_opt, pm_opt, f3db_opt, vn_std


def design_k(k, specs, tia_op):
    ci0 = specs['ci']
    cl0 = specs['cl']

    gmn = tia_op['gmn']
    gmp = tia_op['gmp']
    gamman = tia_op['gamman']
    gammap = tia_op['gammap']
    ro = tia_op['ro']
    cgg = tia_op['cgg']
    cdd = tia_op['cdd']
    cgd = tia_op['cgd']

    ci = ci0 + k * cgg
    cl = cl0 + k * cdd
    cf = k * cgd
    gm = k * (gmn + gmp)
    noise_gm = k * (gamman * gmn + gammap * gmp)
    ro = ro / k
    return get_opt_rf(specs, gm, ro, ci, cl, cf, noise_gm)


def plot_vs_k(specs, tia_op):
    ftarg = specs['ftarg']
    k_list = np.logspace(0, 2, 201)

    idx0 = 0
    rf_list, rdc_list, z_list, pm_list, f3db_list, vstd_list = [], [], [], [], [], []
    for idx, k in enumerate(k_list):
        rf, rdc, z, pm, f3db, vstd = design_k(k, specs, tia_op)
        if np.isnan(rdc) or rdc > 0:
            idx0 = idx + 1
            del rf_list[:]
            del rdc_list[:]
            del z_list[:]
            del pm_list[:]
            del f3db_list[:]
            del vstd_list[:]
        else:
            rf_list.append(rf)
            rdc_list.append(rdc)
            z_list.append(z)
            pm_list.append(pm)
            f3db_list.append(f3db)
            vstd_list.append(vstd)
    
    k_list = k_list[idx0:]
    gain_list = np.abs(rdc_list)
    max_idx = np.argmax(gain_list)

    opt_info = dict(
        size=k_list[max_idx],
        rf=rf_list[max_idx],
        rdc=gain_list[max_idx],
        )
    print('opt info:')
    pprint.pprint(opt_info)

    fig, (ax0, ax1, ax2, ax3) = plt.subplots(4, sharex=True)
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

    ax2.semilogx(k_list, np.array(vstd_list) * 1e3, 'b')
    ax2.set_ylabel('$V_{noise}$ (mV)')

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
    gmp = pch_op['gm'] * pscale
    gmn = nch_op['gm']
    gm = gmn + gmp
    gds = nch_op['gds'] + pch_op['gds'] * pscale
    cgd = nch_op['cgd'] + pch_op['cgd'] * pscale
    cgg = nch_op['cgg'] + pch_op['cgg'] * pscale - cgd
    cdd = nch_op['cdd'] + pch_op['cdd'] * pscale - cgd
    return dict(
        ibias=ibias,
        pscale=pscale,
        gmn=gmn,
        gmp=gmp,
        gamman=nch_op['gamma'],
        gammap=pch_op['gamma'],
        gds=gds,
        ro=1.0/gds,
        cgg=cgg,
        cdd=cdd,
        cgd=cgd,
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
        ftarg=4e9,
        phase_margin=45,
        isw=10e-6,
        ci=20e-15,
        cl=40e-15,
        )

    pch_db = get_db(pmos_spec, intent, interp_method=interp_method,
                    sim_env=sim_env)
    nch_db = get_db(nmos_spec, intent, interp_method=interp_method,
                    sim_env=sim_env)

    tia_op = get_tia_op(specs, nch_db, pch_db)
    pprint.pprint(tia_op)
    
    plot_vs_k(specs, tia_op)
    # print(design_k(45, specs, tia_op))

if __name__ == '__main__':
    np.seterr(all='raise')
    run_main()
