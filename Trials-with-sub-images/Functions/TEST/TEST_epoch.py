from import_data    import import_data
from real_data      import set_params, test_params

data = import_data() # list of touples: (t_map,t_target,width,height), t_map with dimensions (width*height, 1024)

for var in range(5000,13000,2500):
    params = set_params(EPOCHS=var)
    name = 'test_epochs_' + str(var)
    test_params(params,data,name)
    
    
