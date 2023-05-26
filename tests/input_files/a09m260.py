import gvar as gv 
import numpy as np 
p_dict = {
    'abbr' : 'a09m260',
    'hyperons' : ['lambda_z', 'sigma_p', 'sigma_star_p', 'xi_star_z', 'xi_z'], 
    'srcs'     :['S'],
    'snks'     :['SS','PS'],

   't_range' : {
        'sigma_p' : [5,23],
        'proton' : [4,23],
        'delta_pp' : [4,23],
        'xi_z' :  [5,22],
        'xi_star_z' : [5,23],
        'sigma_star_p' : [4,23],
        'lambda_z' : [4,23],
        'hyperons':   [4,23],
        'all':   [4,23],
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
        'delta_pp': 4,
        'proton': 3,
        'sigma_star_p':4,
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
    'proton_E': np.array(['0.47(10)', '0.7(3.2)', '1.1(3.2)', '1.3(3.2)'], dtype=object),
    'proton_z_PS': np.array(['1.1(1.1)e-04', '1.1(1.1)e-04', '1.1(1.1)e-04', '1.1(1.1)e-04'],dtype=object),
    'proton_z_SS': np.array(['4.4(4.4)e-06', '4.4(4.4)e-06', '4.4(4.4)e-06', '4.4(4.4)e-06'],dtype=object),
    'delta_E': np.array(['0.6(10)', '0.7(3.2)', '1.1(3.2)', '1.3(3.2)'], dtype=object),
    'delta_z_PS': np.array(['0.0(3.3)e-03', '0.0(3.3)e-03', '0.0(3.3)e-03', '0.0(3.3)e-03'],dtype=object),
    'delta_z_SS': np.array(['4.4(4.4)e-06', '4.4(4.4)e-06', '4.4(4.4)e-06', '4.4(4.4)e-06'],dtype=object),
    'sigma_E': np.array(['0.55(10)', '0.7(32)', '1.0(3.2)', '1.1(3.2)'], dtype=object),
    'sigma_z_PS': np.array(['0.0(3.3)e-03', '0.0(3.3)e-03', '0.0(3.3)e-03', '0.0(3.3)e-03'],dtype=object),
    'sigma_z_SS': np.array(['0.0(3.3)e-03', '0.0(3.3)e-03', '0.0(3.3)e-03', '0.0(3.3)e-03'],dtype=object),
    'sigma_st_E': np.array(['0.5(3)', '0.6(32)', '1.0(3.2)', '1.1(3.2)'], dtype=object),
    'sigma_st_z_PS': np.array(['0.0(3.3)e-03', '0.0(3.3)e-03', '0.0(3.3)e-03', '0.0(3.3)e-03'],dtype=object),
    'sigma_st_z_SS': np.array(['0.0(3.3)e-03', '0.0(3.3)e-03', '0.0(3.3)e-03', '0.0(3.3)e-03'],dtype=object),
    'lam_E': np.array(['0.5(3)', '0.7(3.2)', '1.1(3.2)', '1.3(3.2)'], dtype=object),
    'lam_z_PS': np.array(['0.0(3.3)e-04', '0.0(3.3)e-04', '0.0(3.3)e-03', '0.0(3.3)e-03'],dtype=object),
    'lam_z_SS': np.array(['4.4(4.4)e-05', '4.4(4.4)e-05', '4.4(4.4)e-06', '4.4(4.4)e-06'],dtype=object),
    'xi_E': np.array(['0.6(10)', '0.8(32)', '0.9(32)', '1.55(32)'], dtype=object),
    'xi_z_PS': np.array(['0.0(3.3)e-03', '0.0(3.3)e-03', '0.0(3.3)e-03', '0.0(3.3)e-03'],dtype=object),
    'xi_z_SS': np.array(['0.0(3.3)e-03', '0.0(3.3)e-03', '0.0(3.3)e-03', '0.0(3.3)e-03'],dtype=object),
    'xi_st_E': np.array(['0.7(20)', '0.9(32)', '1.0(32)', '1.55(32)'], dtype=object),
    'xi_st_z_PS': np.array(['0.0(3.3)e-03', '0.0(3.3)e-03', '0.0(3.3)e-03', '0.0(3.3)e-03'],dtype=object),
    'xi_st_z_SS': np.array(['0.0(3.3)e-03', '0.0(3.3)e-03', '0.0(3.3)e-03', '0.0(3.3)e-03'],dtype=object),
    }