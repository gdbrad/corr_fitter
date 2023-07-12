import gvar as gv 
import h5py as h5 
import numpy as np 
import os 
import corr_fitter.bs_utils as bs 

def group_prior_dict(prior):
    new_prior = {}

    # Define the states list and the mapping
    states = ['proton', 'delta_pp', 'lambda_z', 'sigma_p', 'sigma_star_p', 'xi_star_z', 'xi_z']
    mapping = {
        'proton': 'proton',
        'delta': 'delta_pp',
        'lam': 'lambda_z',
        'sigma_st': 'sigma_star_p',
        'sigma': 'sigma_p',
        'xi_st': 'xi_star_z',
        'xi': 'xi_z'
    }
    # Check if the dictionary is already in the nested format
    if isinstance(list(prior.values())[0], dict):
        return prior

    for key, value in prior.items():
        for old_prefix in sorted(mapping.keys(), key=len, reverse=True):
            new_prefix = mapping[old_prefix]
            if key.startswith(old_prefix):
                new_key = key.replace(old_prefix, new_prefix)
                for state in states:
                    if new_key.startswith(state):
                        if state not in new_prior:
                            new_prior[state] = {} 
                        new_prior[state][new_key] = value 
                        break
                break 

    return new_prior

def get_corrs(data_file,particles,p_dict):
    """forming a gvar averaged dataset for the raw correlator data"""
    output = {}
    for part in particles:
        output.update(get_raw_corr(data_file,p_dict['abbr'],part))
    return gv.dataset.avg_data(output)

def get_raw_corr(file_h5,abbr,particle,normalize=None):
    """fetching raw correlator data from h5 file"""
    data = {}
    data_normalized = {}
    particle_path = '/'+abbr+'/'+particle
    with h5.File(file_h5,"r") as f:
        if f[particle_path].shape[2] == 2:
            data['SS'] = f[particle_path][:, :, 0, 0].real
            data['PS'] = f[particle_path][:, :, 1, 0].real
            if normalize:
                for key in ['SS','PS']:
                    data_normalized[key] = data[key] /np.mean(data[key][:,0])
                return data_normalized
    return data

def pickle_out(fit_out,out_path,species=None):
    """save out fit parameters to a pickled file able to be handled by gvar"""
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    fit_dump = {}
    fit_dump['prior'] = fit_out.prior
    fit_dump['p'] = fit_out.p
    fit_dump['logGBF'] = fit_out.logGBF
    fit_dump['Q'] = fit_out.Q
    if species == 'meson':
        return gv.dump(fit_dump,out_path+'meson_fit_params')
    elif species == 'hyperons':
        return gv.dump(fit_dump,out_path+'hyperons')