import gvar as gv 
import numpy as np
p_dict = {
    'abbr' : 'a12m310',
    'srcs'     :['S'],
    'snks'     :['SS','PS'],

   't_range' : {
        'sigma_p' : [3,15],
        'xi_z' :  [4,13],
        'xi_star_z' : [3,14],
        'sigma_star_p' : [3,14],
        'proton' :   [3,14],
        'delta_pp' : [3,14],
        'lambda_z' : [8,15],
        'hyperons':   [5,15],
        'all':   [5,15]
    },
    'n_states' : {
        'sigma_p' : 3,
        'sigma_star_p':3,
        'xi_z' :3,
        'xi_star_z':3,
        'delta_pp':3,
        'proton':3,
        'lambda_z':2,
        'hyperons': 2,
        'all':2

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
    
    'make_plots' : True,
    'save_prior' : False,
    
    'show_all' : True,
    'show_many_states' : False, # This doesn't quite work as expected: keep false
    'use_prior' : True
    }

prior = gv.BufferDict()
prior = {
    'sigma_E': np.array(['0.73(22)', '0.8(3.2)', '0.9(3.2)', '1.0(3.2)'], dtype=object),
    'sigma_z_PS': np.array(['0.0(3.3)e-05', '0.0(3.3)e-05', '0.0(3.3)e-05', '0.0(3.3)e-05'],dtype=object),
    'sigma_z_SS': np.array(['4.4(4.4)e-05', '4.4(4.4)e-05', '4.4(4.4)e-05', '4.4(4.4)e-05'],dtype=object),
    'sigma_st_E': np.array(['0.8(30)', '0.9(32)', '1.0(3.2)', '1.1(3.2)'], dtype=object),
    'sigma_st_z_PS': np.array(['0.0(3.3)e-05', '0.0(3.3)e-05', '0.0(3.3)e-05', '0.0(3.3)e-05'],dtype=object),
    'sigma_st_z_SS': np.array(['0.0(3.3)e-05', '0.0(3.3)e-05', '0.0(3.3)e-05', '0.0(3.3)e-05'],dtype=object),
    'lam_E': np.array(['0.7(2.2)', '0.9(3.2)', '1.1(3.2)', '1.3(3.2)'], dtype=object),
    'lam_z_PS': np.array(['0.0(3.3)e-06', '0.0(3.3)e-06', '0.0(3.3)e-06', '0.0(3.3)e-06'],dtype=object),
    'lam_z_SS': np.array(['4.4(4.4)e-06', '4.4(4.4)e-06', '4.4(4.4)e-06', '4.4(4.4)e-06'],dtype=object),
    'proton_E': np.array(['0.61(22)', '0.9(2.2)', '1.0(2.2)', '1.1(2.2)'], dtype=object),
    'proton_z_PS': np.array(['0.0(3.3)e-05', '0.0(3.3)e-05', '0.0(3.3)e-06', '0.0(3.3)e-06'],dtype=object),
    'proton_z_SS': np.array(['4.4(4.4)e-07', '4.4(4.4)e-07', '4.4(4.4)e-07', '4.4(4.4)e-07'],dtype=object),
    'xi_E': np.array(['0.8(2.2)', '1.28(32)', '1.45(32)', '1.55(32)'], dtype=object),
    'xi_z_PS': np.array(['0.0(3.3)e-05', '0.0(3.3)e-05', '0.0(3.3)e-06', '0.0(3.3)e-06'],dtype=object),
    'xi_z_SS': np.array(['4.4(4.4)e-06', '4.4(4.4)e-07', '4.4(4.4)e-07', '4.4(4.4)e-07'],dtype=object),
    'xi_st_E': np.array(['0.96(2)', '1.1(32)', '1.3(32)', '1.55(32)'], dtype=object),
    'xi_st_z_PS': np.array(['0.0(3.3)e-05', '0.0(3.3)e-05', '0.0(3.3)e-05', '0.0(3.3)e-05'],dtype=object),
    'xi_st_z_SS': np.array(['0.0(3.3)e-05', '0.0(3.3)e-05', '0.0(3.3)e-05', '0.0(3.3)e-05'],dtype=object),
    'delta_E': np.array(['0.91(22)', '0.95(2.2)', '1.0(2.2)', '1.1(2.2)'], dtype=object),
    'delta_z_PS': np.array(['0.0(3.3)e-04', '0.0(3.3)e-04', '0.0(3.3)e-04', '0.0(3.3)e-04'],dtype=object),
    'delta_z_SS': np.array(['0.0(3.3)e-04', '0.0(3.3)e-04', '0.0(3.3)e-04', '0.0(3.3)e-04'],dtype=object)}

