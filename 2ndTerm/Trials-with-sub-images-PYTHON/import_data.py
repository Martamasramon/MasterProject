# %%
import scipy.io as scio
import numpy as np 

# %%
def import_data():
    ### IMPORT DATA ###
    tissue3 = scio.loadmat('../../Data/tissue3_1.mat')
    tissue6 = scio.loadmat('../../Data/tissue6.mat')
    tissue30 = scio.loadmat('../../Data/tissue30.mat')
    tissue31 = scio.loadmat('../../Data/tissue31.mat')
    tissue34_1 = scio.loadmat('../../Data/tissue34_1.mat')
    tissue34_2 = scio.loadmat('../../Data/tissue34_2.mat')
    tissue36 = scio.loadmat('../../Data/tissue36.mat')
    tissue37 = scio.loadmat('../../Data/tissue37.mat')
    tissue39_1 = scio.loadmat('../../Data/tissue39_1.mat')

    calibration = scio.loadmat('../../Data/cal_14.mat')
    x = calibration['Cal'][0][0][0]
    x = [item for sublist in x for item in sublist]

    tissue_data = [tissue3,tissue6,tissue30,tissue31,tissue34_1,tissue34_2,tissue36,tissue37,tissue39_1]
    file_names = ['map_t3_nobl.txt','map_t6_nobl.txt','map_t30_nobl.txt','map_t31_nobl.txt','map_t34_1_nobl.txt','map_t34_2_nobl.txt','map_t36_nobl.txt','map_t37_nobl.txt','map_t39_1_nobl.txt']

    tissue = []
    count = 0
    for item in tissue_data: 
        t_target = item['bcc']
        width = t_target.shape[0]
        height = t_target.shape[1]
        t_map = np.loadtxt(file_names[count]).reshape(width*height, 1024)
        tissue.append((t_map,t_target,width,height))
        count += 1
        
    return tissue


