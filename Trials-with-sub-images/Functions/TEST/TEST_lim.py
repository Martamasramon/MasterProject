from import_data    import import_data
from real_data      import set_params, test_params

data = import_data() # list of touples: (t_map,t_target,width,height), t_map with dimensions (width*height, 1024)

for var in range(4,25,4):
    params = set_params(N_BCC_LIM=var)
    name = 'test_lim_' + str(var)
    test_params(params,data,name)
    
    
