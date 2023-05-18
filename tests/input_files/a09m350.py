import gvar as gv 
import numpy as np 
p_dict = {
    'abbr' : 'a09m350',
    'srcs'     :['S'],
    'snks'     :['SS','PS'],

   't_range' : {
        'sigma_p' : [4,19],
        'proton' : [5,19],
        'delta_pp' : [5,19],
        'xi_z' :  [4,19],
        'xi_star_z' : [4,16],
        'sigma_star_p' : [3,17],
        'lambda_z' : [3,20],
        'hyperons':   [3,19],
        'all':   [3,19]
    },

    'tag':{
        'sigma' : 'sigma',
        'sigma_st' : 'sigma_st',
        'xi' :  'xi',
        'xi_st' : 'xi_st',
        'lam' : 'lam',
        'proton': 'proton',
        'delta' : 'delta'
    },
    'n_states' : {
        'sigma_p' : 3,
        'delta_pp': 3,
        'proton': 3,
        'sigma_star_p':3,
        'xi_z' :3,
        'xi_star_z':3,
        'lambda_z':3,
        'hyperons': 2,
        'all':2,
    },
    
    'make_plots' : True,
    'save_prior' : False,
    
    'show_all' : True,
    'show_many_states' : False, # This doesn't quite work as expected: keep false
    'use_prior' : True
}

prior = gv.BufferDict()
prior = {
    'proton_E': np.array(['0.4(30)', '0.7(3.2)', '1.1(3.2)', '1.3(3.2)'], dtype=object),
    'proton_z_PS': np.array(['0.0(3.3)e-03', '0.0(3.3)e-03', '0.0(3.3)e-03', '0.0(3.3)e-03'],dtype=object),
    'proton_z_SS': np.array(['4.4(4.4)e-06', '4.4(4.4)e-06', '4.4(4.4)e-06', '4.4(4.4)e-06'],dtype=object),
    'delta_E': np.array(['0.6(3.3)', '0.7(3.2)', '1.1(3.2)', '1.3(3.2)'], dtype=object),
    'delta_z_PS': np.array(['0.0(3.3)e-03', '0.0(3.3)e-03', '0.0(3.3)e-03', '0.0(3.3)e-03'],dtype=object),
    'delta_z_SS': np.array(['4.4(4.4)e-06', '4.4(4.4)e-06', '4.4(4.4)e-06', '4.4(4.4)e-06'],dtype=object),
    'sigma_E': np.array(['0.6(3.3)', '0.9(32)', '1.0(3.2)', '1.1(3.2)'], dtype=object),
    'sigma_st_E': np.array(['0.5(3)', '0.6(32)', '1.0(3.2)', '1.1(3.2)'], dtype=object),
    'sigma_st_z_PS': np.array(['0.0(3.3)e-03', '0.0(3.3)e-03', '0.0(3.3)e-03', '0.0(3.3)e-03'],dtype=object),
    'sigma_st_z_SS': np.array(['0.0(3.3)e-03', '0.0(3.3)e-03', '0.0(3.3)e-03', '0.0(3.3)e-03'],dtype=object),
    'sigma_z_PS': np.array(['0.0(3.3)e-04', '0.0(3.3)e-04', '0.0(3.3)e-03', '0.0(3.3)e-03'],dtype=object),
    'sigma_z_SS': np.array(['0.0(3.3)e-04', '0.0(3.3)e-04', '0.0(3.3)e-03', '0.0(3.3)e-03'],dtype=object),
    'lam_E': np.array(['0.5(3)', '0.7(3.2)', '1.1(3.2)', '1.3(3.2)'], dtype=object),
    'lam_z_PS': np.array(['0.0(3.3)e-04', '0.0(3.3)e-04', '0.0(3.3)e-03', '0.0(3.3)e-03'],dtype=object),
    'lam_z_SS': np.array(['4.4(4.4)e-05', '4.4(4.4)e-05', '4.4(4.4)e-06', '4.4(4.4)e-06'],dtype=object),
    'xi_E': np.array(['0.6(2)', '0.8(32)', '0.9(32)', '1.55(32)'], dtype=object),
    'xi_st_E': np.array(['0.73(2)', '0.9(32)', '1.1(32)', '1.55(32)'], dtype=object),
    'xi_st_z_PS': np.array(['0.0(3.3)e-03', '0.0(3.3)e-03', '0.0(3.3)e-03', '0.0(3.3)e-03'],dtype=object),
    'xi_st_z_SS': np.array(['0.0(3.3)e-03', '0.0(3.3)e-03', '0.0(3.3)e-03', '0.0(3.3)e-03'],dtype=object),
    'xi_z_PS': np.array(['0.0(3.3)e-03', '0.0(3.3)e-03', '0.0(3.3)e-03', '0.0(3.3)e-03'],dtype=object),
    'xi_z_SS': np.array(['0.0(3.3)e-03', '0.0(3.3)e-03', '0.0(3.3)e-03', '0.0(3.3)e-03'],dtype=object),
    }
'''
$\delta_{GMO}$ xpt extrapolation model and prior information
'''
model_info = {}
model_info['particles'] = ['piplus','kplus','eta']
model_info['order_chiral'] = 'lo'
model_info['tree_level'] = True
model_info['loop_level'] = False
model_info['delta'] = True
model_info['abbr'] = ['a12m180L']
model_info['observable'] = ['delta_gmo'] #'centroid', 'octet'


# TODO put prior routines in here, filename save options 
