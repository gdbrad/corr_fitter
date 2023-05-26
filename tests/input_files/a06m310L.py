import gvar as gv 
import numpy as np 
p_dict = {
    'abbr' : 'a06m310L',
    'hyperons' : ['proton','xi_z','sigma_p','lambda_z','xi_star_z', 'delta_pp', 'sigma_star_p'], 
    'srcs'     :['S'],
    'snks'     :['SS','PS'],

   't_range' : {
        'sigma_p' : [6,27],
        'proton' : [6,27],
        'delta_pp' : [10,27],
        'xi_z' :  [6,27],
        'xi_star_z' : [10,27],
        'sigma_star_p' : [10,27],
        'lambda_z' : [6,28],
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
        'proton': 3,
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
    'proton_E': np.array(['0.3(30)', '0.7(3.2)', '1.1(3.2)', '1.3(3.2)'], dtype=object),
    'proton_z_PS': np.array(['0.0(3.3)e-03', '0.0(3.3)e-03', '0.0(3.3)e-03', '0.0(3.3)e-03'],dtype=object),
    'proton_z_SS': np.array(['4.4(4.4)e-06', '4.4(4.4)e-06', '4.4(4.4)e-06', '4.4(4.4)e-06'],dtype=object),
    'delta_E': np.array(['0.41(1)', '0.55(3.2)', '1.1(3.2)', '1.3(3.2)'], dtype=object),
    'delta_z_PS': np.array(['3(3.3)e-03', '3(3.3)e-03', '3(3.3)e-03', '3(3.3)e-03'],dtype=object),
    'delta_z_SS': np.array(['4.4(4.4)e-06', '4.4(4.4)e-06', '4.4(4.4)e-06', '4.4(4.4)e-06'],dtype=object),
    'sigma_E': np.array(['0.4(30)', '0.5(32)', '1.0(3.2)', '1.1(3.2)'], dtype=object),
    'sigma_st_E': np.array(['0.43(1)', '0.6(32)', '1.0(3.2)', '1.1(3.2)'], dtype=object),
    'sigma_st_z_PS': np.array(['0.0(3.3)e-03', '0.0(3.3)e-03', '0.0(3.3)e-03', '0.0(3.3)e-03'],dtype=object),
    'sigma_st_z_SS': np.array(['0.0(3.3)e-03', '0.0(3.3)e-03', '0.0(3.3)e-03', '0.0(3.3)e-03'],dtype=object),
    'sigma_z_PS': np.array(['0.0(3.3)e-03', '0.0(3.3)e-03', '0.0(3.3)e-03', '0.0(3.3)e-03'],dtype=object),
    'sigma_z_SS': np.array(['0.0(3.3)e-03', '0.0(3.3)e-03', '0.0(3.3)e-03', '0.0(3.3)e-03'],dtype=object),
    'lam_E': np.array(['0.4(3)', '0.7(3.2)', '1.1(3.2)', '1.3(3.2)'], dtype=object),
    'lam_z_PS': np.array(['0.0(3.3)e-03', '0.0(3.3)e-03', '0.0(3.3)e-03', '0.0(3.3)e-03'],dtype=object),
    'lam_z_SS': np.array(['4.4(4.4)e-06', '4.4(4.4)e-06', '4.4(4.4)e-06', '4.4(4.4)e-06'],dtype=object),
    'xi_E': np.array(['0.41(1)', '0.65(32)', '0.8(32)', '1.55(32)'], dtype=object),
    'xi_st_E': np.array(['0.48(1)', '0.7(32)', '0.8(32)', '1.55(32)'], dtype=object),
    'xi_st_z_PS': np.array(['3(3.3)e-03', '3(3.3)e-03', '3(3.3)e-03', '3(3.3)e-03'],dtype=object),
    'xi_st_z_SS': np.array(['0.0(3.3)e-03', '0.0(3.3)e-03', '0.0(3.3)e-03', '0.0(3.3)e-03'],dtype=object),
    'xi_z_PS': np.array(['0.0(3.3)e-03', '0.0(3.3)e-03', '0.0(3.3)e-03', '0.0(3.3)e-03'],dtype=object),
    'xi_z_SS': np.array(['0.0(3.3)e-03', '0.0(3.3)e-03', '0.0(3.3)e-03', '0.0(3.3)e-03'],dtype=object),
    }