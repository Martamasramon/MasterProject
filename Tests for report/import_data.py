from compress import decompress
import scipy.io as scio
import numpy as np 
import mat73
from os import listdir

def import_data():
    ### IMPORT DATA ###
    tissue3 = scio.loadmat('./Data/tissue3_1.mat')
    tissue6 = scio.loadmat('./Data/tissue6.mat')
    tissue30 = scio.loadmat('./Data/tissue30.mat')
    tissue31 = scio.loadmat('./Data/tissue31.mat')
    tissue34_1 = scio.loadmat('./Data/tissue34_1.mat')
    tissue34_2 = scio.loadmat('./Data/tissue34_2.mat')
    tissue36 = scio.loadmat('./Data/tissue36.mat')
    tissue37 = scio.loadmat('./Data/tissue37.mat')
    tissue39_1 = scio.loadmat('./Data/tissue39_1.mat')

    calibration = scio.loadmat('./Data/cal_14.mat')
    x = calibration['Cal'][0][0][0]
    x = [item for sublist in x for item in sublist]

    tissue_data = [tissue3,tissue6,tissue30,tissue31,tissue34_1,tissue34_2,tissue36,tissue37,tissue39_1]
    file_names = ['./Data/map_t3_nobl.txt','./Data/map_t6_nobl.txt','./Data/map_t30_nobl.txt','./Data/map_t31_nobl.txt','./Data/map_t34_1_nobl.txt','./Data/map_t34_2_nobl.txt','./Data/map_t36_nobl.txt','./Data/map_t37_nobl.txt','./Data/map_t39_1_nobl.txt']

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

## Save useful tissue with bcc data only
def get_tissues_with_bcc():
    '''
        Reads all files in ./Raman_list and finds tissues with bcc maps 
        (bcc and Hyperspectral data must have dimensions 200x200).
        
        Outputs:
        - files: list of all file names in ./Raman_list
        - tissues_with_bcc: list of tuples in the format (hyperspectral data, bcc map)
    '''
    files = listdir('./Raman_list')
    tissues_with_bcc = []
    
    for i in range(len(files)):
        name = files[i]
        try:
            data = scio.loadmat('./Raman_list/'  + name)
        except:
            data = mat73.loadmat('./Raman_list/' + name)
        
        name  = name[7:-14]
        t_map = data['map_' + name]
        
        for key, value in data.items():
            if key == 'bcc':
                t_bcc = data['bcc']
                if (t_map.shape[0] == t_bcc.shape[0]) and (t_map.shape[1] == t_bcc.shape[1]):
                    if (t_map.shape[0] == 200) and (t_map.shape[1] == 200):
                        tissues_with_bcc.append((t_map, t_bcc))
                    
    return files, tissues_with_bcc


def get_tissues_with_name():
    '''
        Reads all files in ./Raman_list and finds tissues with bcc maps 
        (bcc and Hyperspectral data must have dimensions 200x200).
        
        Outputs:
        - files: list of all file names in ./Raman_list
        - tissues_with_name: list of tuples in the format (hyperspectral data, bcc map)
    '''
    files = listdir('./Raman_list')
    tissues_with_name = []
    
    for i in range(len(files)):
        name = files[i]
        try:
            data = scio.loadmat('./Raman_list/'  + name)
        except:
            data = mat73.loadmat('./Raman_list/' + name)
        
        name  = name[7:-14]
        t_map = data['map_' + name]
        
        for key, value in data.items():
            if key == 'bcc':
                t_bcc = data['bcc']
                if (t_map.shape[0] == t_bcc.shape[0]) and (t_map.shape[1] == t_bcc.shape[1]):
                    if (t_map.shape[0] == 200) and (t_map.shape[1] == 200):
                        tissues_with_name.append((t_map, name))
                    
    return tissues_with_name


def get_preprocessed_data():
    '''
        Reads all files in ./Raman_list and finds tissues with bcc maps 
        (bcc and Hyperspectral data must have dimensions 200x200).
        
        Outputs:
        - tissues_with_bcc: list of tuples in the format (hyperspectral data, bcc map)
    '''
    files = listdir('./Preprocessed')
    
    tissues_with_bcc = []
    for i in range(len(files)):
        name = files[i][:-4]
        try:
            raw_data = scio.loadmat('./Raman_list/tissue_'  + name + '_workspace.mat')
        except:
            raw_data = mat73.loadmat('./Raman_list/tissue_' + name + '_workspace.mat')
        
        t_bcc = raw_data['bcc']
        t_map = decompress('./Preprocessed/' + name + '.lz4')
        
        tissues_with_bcc.append((t_map,t_bcc))
                    
    return tissues_with_bcc