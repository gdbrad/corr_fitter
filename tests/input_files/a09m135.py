import gvar as gv 
import numpy as np 
p_dict = {
    'abbr' : 'a09m135',
    'hyperons' : ['proton','xi_z','sigma_p','lambda_z','xi_star_z', 'delta_pp', 'sigma_star_p'], 
    'srcs'     :['S'],
    'snks'     :['SS','PS'],

   't_range' : {
        'sigma_p' : [5,15],
        'proton' : [5,15],
        'delta_pp' : [5,15],
        'xi_z' :  [6,14],
        'xi_star_z' : [4,15],
        'sigma_star_p' : [6,15],
        'lambda_z' : [5,16],
        'hyperons':   [10,23],
        'all':   [10,23],
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
        'proton': 2,
        'sigma_star_p':3,
        'xi_z' :3,
        'xi_star_z':3,
        'lambda_z':3,
        'hyperons': 3,
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
    'proton_E': np.array(['0.43(1)', '0.6(3.2)', '1.1(3.2)', '1.3(3.2)'], dtype=object),
    'proton_z_PS': np.array(['0.0(3.3)e-03', '0.0(3.3)e-03', '0.0(3.3)e-03', '0.0(3.3)e-03'],dtype=object),
    'proton_z_SS': np.array(['4.4(4.4)e-06', '4.4(4.4)e-06', '4.4(4.4)e-06', '4.4(4.4)e-06'],dtype=object),
    'delta_E': np.array(['0.6(1)', '0.8(3.2)', '1.1(3.2)', '1.3(3.2)'], dtype=object),
    'delta_z_PS': np.array(['0.0(3.3)e-03', '0.0(3.3)e-03', '0.0(3.3)e-03', '0.0(3.3)e-03'],dtype=object),
    'delta_z_SS': np.array(['4.4(4.4)e-06', '4.4(4.4)e-06', '4.4(4.4)e-06', '4.4(4.4)e-06'],dtype=object),
    'sigma_E': np.array(['0.52(2)', '0.7(32)', '1.0(3.2)', '1.1(3.2)'], dtype=object),
    'sigma_st_E': np.array(['0.62(2)', '0.8(32)', '1.0(3.2)', '1.1(3.2)'], dtype=object),
    'sigma_st_z_PS': np.array(['0.0(3.3)e-03', '0.0(3.3)e-03', '0.0(3.3)e-03', '0.0(3.3)e-03'],dtype=object),
    'sigma_st_z_SS': np.array(['0.0(3.3)e-03', '0.0(3.3)e-03', '0.0(3.3)e-03', '0.0(3.3)e-03'],dtype=object),
    'sigma_z_PS': np.array(['0.0(3.3)e-03', '0.0(3.3)e-03', '0.0(3.3)e-03', '0.0(3.3)e-03'],dtype=object),
    'sigma_z_SS': np.array(['0.0(3.3)e-03', '0.0(3.3)e-03', '0.0(3.3)e-03', '0.0(3.3)e-03'],dtype=object),
    'lam_E': np.array(['0.5(2)', '0.7(3.2)', '1.1(3.2)', '1.3(3.2)'], dtype=object),
    'lam_z_PS': np.array(['0.0(3.3)e-03', '0.0(3.3)e-03', '0.0(3.3)e-03', '0.0(3.3)e-03'],dtype=object),
    'lam_z_SS': np.array(['4.4(4.4)e-06', '4.4(4.4)e-06', '4.4(4.4)e-06', '4.4(4.4)e-06'],dtype=object),
    'xi_E': np.array(['0.57(2)', '0.9(32)', '1.2(32)', '1.55(32)'], dtype=object),
    'xi_st_E': np.array(['0.7(2)', '0.9(32)', '1.2(32)', '1.55(32)'], dtype=object),
    'xi_st_z_PS': np.array(['0.0(3.3)e-03', '0.0(3.3)e-03', '0.0(3.3)e-03', '0.0(3.3)e-03'],dtype=object),
    'xi_st_z_SS': np.array(['0.0(3.3)e-03', '0.0(3.3)e-03', '0.0(3.3)e-03', '0.0(3.3)e-03'],dtype=object),
    'xi_z_PS': np.array(['0.0(3.3)e-03', '0.0(3.3)e-03', '0.0(3.3)e-08', '0.0(3.3)e-08'],dtype=object),
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
