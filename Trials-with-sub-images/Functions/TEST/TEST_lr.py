from import_data    import import_data
from real_data      import set_params, test_params

data = import_data() # list of touples: (t_map,t_target,width,height), t_map with dimensions (width*height, 1024)

for var in [0.01,0.05,0.01,0.005,0.001,0.0005,0.0001,0.00005]:
    params = set_params(LEARNING_RATE=var)
    name = 'test_lr_' + str(var)
    test_params(params,data,name)
    
    
