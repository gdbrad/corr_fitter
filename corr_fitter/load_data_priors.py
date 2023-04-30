import pandas as pd
import gvar as gv 
import h5py as h5 
import numpy as np 
import os 
import corr_fitter.bs_utils as bs 

def generate_latex_line(hyperon_fit):
    ordered_keys = ['proton_E0', 'xi_E0', 'sigma_E0', 'lam_E0', 'xi_st_E0', 'delta_E0', 'sigma_st_E0']
    latex_line = ""

    for key in ordered_keys:
        if key in hyperon_fit.p:
            p = hyperon_fit.p[key]
            latex_line += f"{p} & "
        else:
            latex_line += "& "

    # Remove the last ampersand and space
    latex_line = latex_line[:-2]

    return latex_line

def pickle_out(fit_out,out_path,species=None):
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
def get_posterior(fit_out,param='all'):
    output = {}
    for observable in fit_keys(fit_out):
        if param is None:
            output[observable] = {param : fit_out[observable].p[param] for param in fit_keys(fit_out)[observable]}
        elif param == 'all':
            output[observable] = fit_out[observable].p
        else:
            output[observable] = fit_out[observable].p[param]

    return output

def get_hyperon_posterior(bs_data):
    post = {}
    posterior = {}
    hyperon_gs = {}

    for ens in bs_data:
        hyperon_gs[ens] = {}

        if ens != 'spec':
            out_path = os.getcwd()+'/fit_results/'+ens+'/all/'
            post[ens]= gv.load(out_path+"fit_params")
            posterior[ens] = post[ens]['p']
            for hyperon in ['lam', 'sigma', 'sigma_st', 'xi_st', 'xi']:
                hyperon_gs[ens]['m_'+hyperon]=posterior[ens][hyperon+'_E0']
    return hyperon_gs


def fit_keys(fit_out):
    output = {}
    for observable in fit_out.keys():
        keys1 = list(fit_out.prior[observable].keys())
        keys2 = list(fit_out[observable].p.keys())
        output[observable] = np.intersect1d(keys1, keys2)
    return output

def get_raw_corr(file_h5,abbr,particle,normalize=None):
    data = {}
    data_normalized = {}
    particle_path = '/'+abbr+'/'+particle
    with h5.File(file_h5,"r") as f:
        if f[particle_path].shape[3] == 1:
            data['SS'] = f[particle_path][:, :, 0, 0].real.astype(np.float64)
            data['PS'] = f[particle_path][:, :, 1, 0].real .astype(np.float64)
            if normalize:
                for key in ['SS','PS']:
                    data_normalized[key] = data[key] /np.mean(data[key][:,0])
                return data_normalized
    return data

def get_raw_corr_normalize(file_h5,abbr,particle):
    data = {}
    data_normalized = {}
    particle_path = '/'+abbr+'/'+particle
    with h5.File(file_h5,"r") as f:
        if f[particle_path].shape[3] == 1:
            data['SS'] = f[particle_path][:, :, 0, 0].real
            data['PS'] = f[particle_path][:, :, 1, 0].real 
            data_normalized['SS'] = data['SS']/data['SS'].mean(axis=0)[0]
            data_normalized['PS'] = data['PS']/data['PS'].mean(axis=0)[0]
    return data_normalized

def get_raw_corr_new(file_h5,abbr,normalize=None):
    data = {}
    data_normalized ={}
    with h5.File(file_h5,"r") as f:
        for baryon in ['lambda_z', 'sigma_p', 'proton', 'xi_z']:
            particle_path = '/'+abbr+'/'+baryon
            data[baryon+'_SS'] = f[particle_path][:, :, 0, 0].real
            data[baryon+'_PS'] = f[particle_path][:, :, 1, 0].real 
        if normalize:
            for baryon in ['lambda_z', 'sigma_p', 'proton', 'xi_z']:

                data_normalized[baryon+'_SS'] = data[baryon+'_SS'] / np.mean(data[baryon+'_SS'][1,:])
                data_normalized[baryon+'_PS'] = data[baryon+'_PS']/ np.mean(data[baryon+'_PS'][1,:])
            return data_normalized

    return data


def resample_correlator(raw_corr,bs_list, n):
    resampled_raw_corr_data = ({key : raw_corr[key][bs_list[n, :], :]
    for key in raw_corr.keys()})
    resampled_corr_gv = resampled_raw_corr_data
    return resampled_corr_gv

def fetch_prior(model_type,p_dict):

    prior_nucl = {}
    prior = {}
    # prior_xi = {}
    states= p_dict[str(model_type)]
    newlist = [x for x in states]
    for x in newlist:
        path = os.path.normpath("./priors/{0}/{1}/prior_nucl.csv".format(p_dict['abbr'],x))
        df = pd.read_csv(path, index_col=0).to_dict()
        for key in list(df.keys()):
            length = int(np.sqrt(len(list(df[key].values()))))
            prior_nucl[key] = list(df[key].values())[:length]
        prior = gv.gvar(prior_nucl)
    return prior

# def get_data_phys_pt()
