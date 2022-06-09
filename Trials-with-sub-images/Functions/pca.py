import numpy as np 
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

def my_pca(tissue, pca_components):
    ## Train PCA only on relevant BCC data
    length = tissue[0][0].shape[1]
    pca_training_data = np.zeros((1000,length))
    count = 0

    for (t_map,t_target,width,height) in tissue:
        for i in range(width):
            for j in range(height):
                if count < 1000:
                    if t_target[i,j] == 1:
                        index = i*height + j
                        pca_training_data[count] = t_map[index,:]
                        count += 1
                        
    ### Perform PCA dimensionality reduction ###
    norm_data = normalize(pca_training_data)
    pca = PCA(n_components=pca_components) 
    pca.fit(norm_data)

    for i in range(len(tissue)): 
        transformed = pca.transform(tissue[i][0])
        tissue[i] = (transformed,tissue[i][1],tissue[i][2],tissue[i][3])
        
    return tissue


