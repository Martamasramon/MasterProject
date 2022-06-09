from import_data    import import_data
from real_data      import set_params, test_params

data = import_data() # list of touples: (t_map,t_target,width,height), t_map with dimensions (width*height, 1024)

for var in range(6,100,6):
    params = set_params(IMG_WIDTH=var)
    name = 'test_imgwidth_' + str(var)
    test_params(params,data,name)
    
    
