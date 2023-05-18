import gvar as gv 
import numpy as np 
p_dict = {
    'abbr' : 'a09m220_o',
    'hyperons' : ['lambda_z', 'sigma_p', 'sigma_star_p', 'xi_star_z', 'xi_z'], 
    'srcs'     :['S'],
    'snks'     :['SS','PS'],
    't_range' : {
        'sigma_p' : [5,18],
        'proton' : [4,17],
        'delta_pp' : [5,18],
        'xi_z' :  [6,14],
        'xi_star_z' : [5,17],
        'sigma_star_p' : [5,17],
        'lambda_z' : [5,17],
        'hyperons':   [10,22],
        'all':   [10,22],
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
        'delta_pp': 2,
        'proton': 3,
        'sigma_star_p':4,
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
    'proton_E': np.array(['0.4(20)', '0.7(2.2)', '1.1(2.2)', '1.2(2.2)'], dtype=object),
    'proton_z_PS': np.array(['0.0(2.2)e-02', '0.0(2.2)e-02', '0.0(2.2)e-02', '0.0(2.2)e-02'],dtype=object),
    'proton_z_SS': np.array(['4.4(4.4)e-06', '4.4(4.4)e-06', '4.4(4.4)e-06', '4.4(4.4)e-06'],dtype=object),
    'delta_E': np.array(['0.6(2.2)', '0.7(2.2)', '1.1(2.2)', '1.2(2.2)'], dtype=object),
    'delta_z_PS': np.array(['0.0(2.2)e-02', '0.0(2.2)e-02', '0.0(2.2)e-02', '0.0(2.2)e-02'],dtype=object),
    'delta_z_SS': np.array(['4.4(4.4)e-06', '4.4(4.4)e-06', '4.4(4.4)e-06', '4.4(4.4)e-06'],dtype=object),
    'sigma_E': np.array(['0.5(20)', '0.7(22)', '1.0(2.2)', '1.1(2.2)'], dtype=object),
    'sigma_st_E': np.array(['0.65(2)', '0.9(22)', '1.0(2.2)', '1.1(2.2)'], dtype=object),
    'sigma_st_z_PS': np.array(['0.0(2.2)e-02', '0.0(2.2)e-02', '0.0(2.2)e-02', '0.0(2.2)e-02'],dtype=object),
    'sigma_st_z_SS': np.array(['0.0(2.2)e-02', '0.0(2.2)e-02', '0.0(2.2)e-02', '0.0(2.2)e-02'],dtype=object),
    'sigma_z_PS': np.array(['0.0(2.2)e-02', '0.0(2.2)e-02', '0.0(2.2)e-02', '0.0(2.2)e-02'],dtype=object),
    'sigma_z_SS': np.array(['0.0(2.2)e-02', '0.0(2.2)e-02', '0.0(2.2)e-02', '0.0(2.2)e-02'],dtype=object),
    'lam_E': np.array(['0.5(2)', '0.7(2.2)', '1.1(2.2)', '1.2(2.2)'], dtype=object),
    'lam_z_PS': np.array(['0.0(2.2)e-04', '0.0(2.2)e-04', '0.0(2.2)e-02', '0.0(2.2)e-02'],dtype=object),
    'lam_z_SS': np.array(['4.4(4.4)e-05', '4.4(4.4)e-05', '4.4(4.4)e-06', '4.4(4.4)e-06'],dtype=object),
    'xi_E': np.array(['0.5(20)', '0.6(22)', '0.8(22)', '1.55(22)'], dtype=object),
    'xi_st_E': np.array(['0.5(20)', '0.7(22)', '0.8(22)', '1.55(22)'], dtype=object),
    'xi_st_z_PS': np.array(['0.0(2.2)e-02', '0.0(2.2)e-02', '0.0(2.2)e-08', '0.0(2.2)e-08'],dtype=object),
    'xi_st_z_SS': np.array(['0.0(2.2)e-02', '0.0(2.2)e-02', '0.0(2.2)e-02', '0.0(2.2)e-02'],dtype=object),
    'xi_z_PS': np.array(['0.0(2.2)e-02', '0.0(2.2)e-02', '0.0(2.2)e-08', '0.0(2.2)e-08'],dtype=object),
    'xi_z_SS': np.array(['0.0(2.2)e-02', '0.0(2.2)e-02', '0.0(2.2)e-02', '0.0(2.2)e-02'],dtype=object),
    }

