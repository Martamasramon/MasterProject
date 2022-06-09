from import_data    import import_data
from real_data      import set_params, test_params

data = import_data() # list of touples: (t_map,t_target,width,height), t_map with dimensions (width*height, 1024)

for var in [10,20,30,50,100]:
    this_data = data.copy()
    params = set_params(PCA_COMPONENTS=var)
    name = 'test_pca_' + str(var)
    test_params(params,this_data,name)
    
    
