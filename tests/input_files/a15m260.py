import gvar as gv 
import numpy as np
p_dict = {
    'abbr' : 'a15m260',
    'srcs'     :['S'],
    'snks'     :['SS','PS'],

   't_range' : {
        'sigma_p' : [4,11],
        'xi_z' :  [3,12],
        'xi_star_z' : [3,12],
        'sigma_star_p' : [3,12],
        'proton' :   [4,11],
        'delta_pp' : [2,11],
        'lambda_z' : [4,11],
        'hyperons':   [5,15],
        'all':   [5,15]
    },
    'n_states' : {
        'sigma_p' : 2,
        'sigma_star_p':3,
        'xi_z' :3,
        'xi_star_z':3,
        'delta_pp':3,
        'proton':2,
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
    'sigma_E': np.array(['0.88(2)', '1.0(3.2)', '1.2(3.2)', '1.4(3.2)'], dtype=object),
    'sigma_z_PS': np.array(['0.0(3.3)e-05', '0.0(3.3)e-05', '0.0(3.3)e-05', '0.0(3.3)e-05'],dtype=object),
    'sigma_z_SS': np.array(['4.4(4.4)e-05', '4.4(4.4)e-05', '4.4(4.4)e-05', '4.4(4.4)e-05'],dtype=object),
    'sigma_st_E': np.array(['1.12(2)', '1.3(32)', '1.4(3.2)', '1.6(3.2)'], dtype=object),
    'sigma_st_z_PS': np.array(['0.0(3.3)e-04', '0.0(3.3)e-04', '0.0(3.3)e-04', '0.0(3.3)e-05'],dtype=object),
    'sigma_st_z_SS': np.array(['0.0(3.3)e-04', '0.0(3.3)e-04', '0.0(3.3)e-04', '0.0(3.3)e-05'],dtype=object),
    'lam_E': np.array(['0.85(2)', '0.9(3.2)', '1.1(3.2)', '1.3(3.2)'], dtype=object),
    'lam_z_PS': np.array(['0.0(3.3)e-05', '0.0(3.3)e-05', '0.0(3.3)e-05', '0.0(3.3)e-05'],dtype=object),
    'lam_z_SS': np.array(['4.4(4.4)e-06', '4.4(4.4)e-06', '4.4(4.4)e-06', '4.4(4.4)e-06'],dtype=object),
    'proton_E': np.array(['0.75(2)', '0.9(2.2)', '1.0(2.2)', '1.1(2.2)'], dtype=object),
    'proton_z_PS': np.array(['0.0(3.3)e-05', '0.0(3.3)e-05', '0.0(3.3)e-06', '0.0(3.3)e-06'],dtype=object),
    'proton_z_SS': np.array(['4.4(4.4)e-05', '4.4(4.4)e-05', '4.4(4.4)e-05', '4.4(4.4)e-05'],dtype=object),
    'xi_E': np.array(['0.95(2)', '1.28(32)', '1.45(32)', '1.55(32)'], dtype=object),
    'xi_z_PS': np.array(['0.0(3.3)e-05', '0.0(3.3)e-05', '0.0(3.3)e-05', '0.0(3.3)e-05'],dtype=object),
    'xi_z_SS': np.array(['4.4(4.4)e-05', '4.4(4.4)e-05', '4.4(4.4)e-05', '4.4(4.4)e-05'],dtype=object),
    'xi_st_E': np.array(['1.18(2)', '1.3(32)', '1.5(32)', '1.7(32)'], dtype=object),
    'xi_st_z_PS': np.array(['0.0(3.3)e-05', '0.0(3.3)e-05', '0.0(3.3)e-05', '0.0(3.3)e-05'],dtype=object),
    'xi_st_z_SS': np.array(['0.0(3.3)e-05', '0.0(3.3)e-05', '0.0(3.3)e-05', '0.0(3.3)e-05'],dtype=object),
    'delta_E': np.array(['1.12(2)', '1.2(2.2)', '1.4(2.2)', '1.6(2.2)'], dtype=object),
    'delta_z_PS': np.array(['0.0(3.3)e-04', '0.0(3.3)e-04', '0.0(3.3)e-04', '0.0(3.3)e-04'],dtype=object),
    'delta_z_SS': np.array(['0.0(3.3)e-04', '0.0(3.3)e-04', '0.0(3.3)e-04', '0.0(3.3)e-04'],dtype=object)}

