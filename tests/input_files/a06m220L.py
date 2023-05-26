import gvar as gv 
import numpy as np 
p_dict = {
    'abbr' : 'a06m220L',
    'hyperons' : ['proton','xi_z','sigma_p','lambda_z','xi_star_z', 'delta_pp', 'sigma_star_p'], 
    'srcs'     :['S'],
    'snks'     :['SS','PS'],

   't_range' : {
        'sigma_p' : [8,30],
        'proton' : [12,30],
        'delta_pp' : [10,30],
        'xi_z' :  [8,30],
        'xi_star_z' : [10,30],
        'sigma_star_p' : [10,30],
        'lambda_z' : [8,30],
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
    'proton_E': np.array(['0.3(3)', '0.7(3.2)', '1.1(3.2)', '1.3(3.2)'], dtype=object),
    'proton_z_PS': np.array(['0.0(3.3)e-04', '0.0(3.3)e-04', '0.0(3.3)e-04', '0.0(3.3)e-04'],dtype=object),
    'proton_z_SS': np.array(['4.4(4.4)e-06', '4.4(4.4)e-06', '4.4(4.4)e-06', '4.4(4.4)e-06'],dtype=object),
    'delta_E': np.array(['0.4(1)', '0.65(1)', '0.85(1)', '1.3(3.2)'], dtype=object),
    'delta_z_PS': np.array(['4.0(4.0)e-05', '4.0(4.0)e-05', '4.0(4.0)e-05', '4.0(4.0)e-05'],dtype=object),
    'delta_z_SS': np.array(['2.0(2.0)e-06', '2.0(2.0)e-06', '2.0(2.0)e-06', '4.4(4.4)e-06'],dtype=object),
    'sigma_E': np.array(['0.3(3)', '0.5(32)', '1.0(3.2)', '1.1(3.2)'], dtype=object),
    'sigma_st_E': np.array(['0.45(3)', '0.6(32)', '1.0(3.2)', '1.1(3.2)'], dtype=object),
    'sigma_st_z_PS': np.array(['0.0(3.3)e-04', '0.0(3.3)e-04', '0.0(3.3)e-04', '0.0(3.3)e-04'],dtype=object),
    'sigma_st_z_SS': np.array(['0.0(3.3)e-04', '0.0(3.3)e-04', '0.0(3.3)e-04', '0.0(3.3)e-04'],dtype=object),
    'sigma_z_PS': np.array(['0.0(3.3)e-04', '0.0(3.3)e-04', '0.0(3.3)e-04', '0.0(3.3)e-04'],dtype=object),
    'sigma_z_SS': np.array(['0.0(3.3)e-04', '0.0(3.3)e-04', '0.0(3.3)e-04', '0.0(3.3)e-04'],dtype=object),
    'lam_E': np.array(['0.3(3)', '0.7(3.2)', '1.1(3.2)', '1.3(3.2)'], dtype=object),
    'lam_z_PS': np.array(['0.0(3.3)e-04', '0.0(3.3)e-04', '0.0(3.3)e-04', '0.0(3.3)e-04'],dtype=object),
    'lam_z_SS': np.array(['4.4(4.4)e-06', '4.4(4.4)e-06', '4.4(4.4)e-06', '4.4(4.4)e-06'],dtype=object),
    'xi_E': np.array(['0.4(1)', '0.6(1)', '0.8(32)', '1.55(32)'], dtype=object),
    'xi_z_PS': np.array(['0.0(3.3)e-05', '0.0(3.3)e-05', '0.0(3.3)e-05', '0.0(3.3)e-05'],dtype=object),
    'xi_z_SS': np.array(['0.0(3.3)e-04', '0.0(3.3)e-04', '0.0(3.3)e-04', '0.0(3.3)e-04'],dtype=object),
    'xi_st_E': np.array(['0.5(3)', '0.7(32)', '0.8(32)', '1.55(32)'], dtype=object),
    'xi_st_z_PS': np.array(['0.0(3.3)e-06', '0.0(3.3)e-06', '0.0(3.3)e-06', '0.0(3.3)e-06'],dtype=object),
    'xi_st_z_SS': np.array(['0.0(3.3)e-06', '0.0(3.3)e-06', '0.0(3.3)e-06', '0.0(3.3)e-06'],dtype=object),
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
