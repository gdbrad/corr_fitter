import numpy as np
import gvar as gv
import lsqfit
import tqdm
import h5py
import copy
import sys
import os

import corr_fitter.bs_utils as bs


def run_bootstrap(args, fit, fp, data_cfg, x_fit, svdcut=None):

    # make sure results dir exists
    if args.bs_write:
        if not os.path.exists('bs_results'):
            os.makedirs('bs_results')
    if len(args.bs_results.split('/')) == 1:
        bs_file = 'bs_results/'+args.bs_results
    else:
        bs_file = args.bs_results

    # check if we already wrote this dataset
    if args.bs_write:
        have_bs = False
        if os.path.exists(bs_file):
            with h5py.File(bs_file, 'r') as f5:
                if args.bs_path in f5:
                    if len(f5[args.bs_path]) > 0 and not args.overwrite:
                        have_bs = True
                        print(
                            'you asked to write bs results to an existing dset and overwrite =', args.overwrite)
    else:
        have_bs = False

    if not have_bs:
        print('beginning Nbs=%d bootstrap fits' % args.Nbs)

        # use the fit posterior to set the initial guess for bs loop
        p0_bs = dict()
        for k in fit.p:
            p0_bs[k] = fit.p[k].mean

        # seed bs random number generator
        if not args.bs_seed and 'bs_seed' not in dir(fp):
            tmp = input('you have not passed a BS seed nor is it defined in the input file\nenter a seed or hit return for none')
            if not tmp:
                bs_seed = None
            else:
                bs_seed = tmp
        elif 'bs_seed' in dir(fp):
            bs_seed = fp.bs_seed
        if args.bs_seed:
            if args.verbose:
                print('WARNING: you are overwriting the bs_seed from the input file')
            bs_seed = args.bs_seed

        # create bs_list
        Ncfg = data_cfg[next(iter(data_cfg))].shape[0]
        bs_list = bs.get_bs_list(Ncfg, args.Nbs, Mbs=args.Mbs, seed=bs_seed)

        # make BS list for priors
        p_bs_mean = dict()
        for k in fit.prior:
            if args.bs0_restrict and '_0' in k:
                sdev = args.bs0_width * fit.p[k].sdev
            else:
                sdev = fit.prior[k].sdev
            if 'log' in k:
                dist_type='lognormal'
            else:
                dist_type='normal'
            p_bs_mean[k] = bs.bs_prior(args.Nbs, mean=fit.prior[k].mean,
                                       sdev=sdev, seed=bs_seed+'_'+k, dist=dist_type)

        # set up posterior lists of bs results
        post_bs = dict()
        for k in fit.p:
            post_bs[k] = []

        fit_str = []

        # loop over bootstraps
        for bs in tqdm.tqdm(range(args.Nbs)):
            ''' all gvar's created in this switch are destroyed at restore_gvar
                [they are out of scope] '''
            gv.switch_gvar()

            bs_data = dict()
            for k in fit.y:
                if 'mres' not in k:
                    bs_data[k] = data_cfg[k][bs_list[bs]]
                else:
                    mres = k.split('_')[0]
                    bs_data[mres] = data_cfg[mres+'_MP'] / data_cfg[mres+'_PP']
            bs_gv = gv.dataset.avg_data(bs_data)

            y_bs = {k: v[x_fit[k]['t_range']]
                    for (k, v) in bs_gv.items() if k in fit.y}

            p_bs = dict()
            for k in p_bs_mean:
                p_bs[k] = gv.gvar(p_bs_mean[k][bs], fit.prior[k].sdev)

            if svdcut:
                fit_bs = lsqfit.nonlinear_fit(data=(x_fit, y_bs), prior=p_bs, p0=p0_bs,
                                              fcn=fit_funcs.fit_function, svdcut=svdcut)
            else:
                fit_bs = lsqfit.nonlinear_fit(data=(x_fit, y_bs), prior=p_bs, p0=p0_bs,
                                              fcn=fit_funcs.fit_function)
            fit_str.append(str(fit_bs))
            for r in post_bs:
                post_bs[r].append(fit_bs.p[r].mean)

            ''' end of gvar scope used for bootstrap '''
            gv.restore_gvar()

        for r in post_bs:
            post_bs[r] = np.array(post_bs[r])
        if args.bs_write:
            # write the results
            with h5py.File(bs_file, 'a') as f5:
                try:
                    f5.create_group(args.bs_path)
                except Exception as e:
                    print(e)
                for r in post_bs:
                    if len(post_bs[r]) > 0:
                        if r in f5[args.bs_path]:
                            del f5[args.bs_path+'/'+r]
                        f5.create_dataset(args.bs_path+'/'+r, data=post_bs[r])

        return post_bs, fit_str
    else:
        sys.exit('not running BS - bs_results exist')