import gvar as gv 
import numpy as np
import gvar as gv 
import numpy as np
p_dict = {
    'abbr' : 'a12m220S',
    'srcs'     :['S'],
    'snks'     :['SS','PS'],

   't_range' : {
        'sigma_p' : [3,10],
        'xi_z' :  [4,11],
        'xi_star_z' : [6,10],
        'sigma_star_p' : [6,10],
        'proton' :   [5,11],
        'delta_pp' : [3,12],
        'lambda_z' : [3,13],
        'hyperons':   [5,15],
        'all':   [5,15]
    },
    'n_states' : {
        'sigma_p' : 2,
        'sigma_star_p':1,
        'xi_z' :2,
        'xi_star_z':1,
        'delta_pp':2,
        'proton':1,
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
    'sigma_z_PS': np.array(['0.0(3.3)e-06', '0.0(3.3)e-06', '0.0(3.3)e-06', '0.0(3.3)e-06'],dtype=object),
    'sigma_z_SS': np.array(['4.4(4.4)e-07', '4.4(4.4)e-07', '4.4(4.4)e-07', '4.4(4.4)e-07'],dtype=object),
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
    'xi_st_E': np.array(['0.8(22)', '0.9(32)', '1.0(32)', '1.55(32)'], dtype=object),
    'xi_st_z_PS': np.array(['0.0(3.3)e-05', '0.0(3.3)e-05', '0.0(3.3)e-08', '0.0(3.3)e-08'],dtype=object),
    'xi_st_z_SS': np.array(['0.0(3.3)e-05', '0.0(3.3)e-05', '0.0(3.3)e-05', '0.0(3.3)e-05'],dtype=object),
    'delta_E': np.array(['0.91(22)', '0.95(2.2)', '1.0(2.2)', '1.1(2.2)'], dtype=object),
    'delta_z_PS': np.array(['0.0(3.3)e-04', '0.0(3.3)e-04', '0.0(3.3)e-04', '0.0(3.3)e-04'],dtype=object),
    'delta_z_SS': np.array(['0.0(3.3)e-04', '0.0(3.3)e-04', '0.0(3.3)e-04', '0.0(3.3)e-04'],dtype=object)}

