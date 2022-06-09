## Imports and adapted functions
from fake_data import test_fake_data
import numpy as np
import matplotlib.pyplot as plt

## Parameters
parameters = {
    'LEARNING_RATE' : 0.01,
    'EPOCHS'        : 5000,
    'DROPOUT_RATE'  : 0.1,
    'TRAIN_CUTOFF'  : 1200,
    'SUBIMG_SPACING': 5,
    'IMG_WIDTH'     : 10,
    'PCA_COMPONENTS': 10,
    'N_BCC_LIM'     : 20,
    'BL_AMPLITUDE'  : 50,
    'PEAK_HEIGHTS'  : [30,60],
    'PEAK_WIDTHS'   : [20,10],
    'NOISE_STD'     : 3
}

## Create image
map = np.zeros((200,200))

map[10:15,5:10]         = 1
map[20:24,11:15]        = 1
map[30:35,15:17]        = 1
map[40:45,40:46]        = 1
map[77:83,68:75]        = 1
map[100:110,99:100]     = 1
map[120:123,125:140]    = 1
map[135:143,145:147]    = 1
map[144:148,150:155]    = 1
map[167:179,188:199]    = 1
map[20:40,180:190]      = 1
map[30:35,170:175]      = 1
map[40:60,190:195]      = 1
map[40:44,100:110]      = 1
map[80:90,10:50]        = 1
map[100:120,10:30]      = 1
map[120:141,30:40]      = 1
map[135,10:20]          = 1
map[144:158,15:30]      = 1
map[177:189,50:70]      = 1

results = test_fake_data(map, parameters)
plt.imshow(results)
plt.savefig('test_res.png')
