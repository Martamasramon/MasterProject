from real_data          import set_params
from import_data        import get_preprocessed_data     
from pca                import my_pca2                
from subimages          import subimages2, reconstruct_img2
from CNN2D              import CNN2D, train_model, test_model, get_error

import matplotlib.pyplot as plt
import math
import random
import string

# Define all parameters
params = set_params()

# Get all preprocessed files & reduce dimensionality
data = get_preprocessed_data()
pca_reduced = my_pca2(data, params['PCA_COMPONENTS'])

# Divide each tissue into sub-images (input to CNN)
sub_images = subimages2(pca_reduced, params['IMG_WIDTH'], params['SUBIMG_SPACING'], params['PCA_COMPONENTS'], params['N_BCC_LIM'])

# Create CNN model and train it with 80% tissue
model = CNN2D(params['IMG_WIDTH'],params['PCA_COMPONENTS'],params['DROPOUT_RATE'],kernel=5)

num_training = math.floor( len(data)*0.8 )
for i in range(num_training):
    model = train_model(model, sub_images[i][0], sub_images[i][1], params['LEARNING_RATE'], params['EPOCHS'], params['IMG_WIDTH'], params['PCA_COMPONENTS'])

# Analyse the results of the CNN predictions
results = []
for i in range(len(data)):
    approximation = test_model(model, sub_images[i][0], params['IMG_WIDTH'], params['PCA_COMPONENTS'])
    _, result = get_error(approximation, sub_images[i][1])
    reconstructed = reconstruct_img2(result, params['IMG_WIDTH'], params['SUBIMG_SPACING'])
    results.append((data[i][1],reconstructed))

# Plot 
fig, axs = plt.subplots(7,6, figsize=(20,20))
for i in range(len(data)):
    row = math.floor( i/3 )
    col = (i*2)%6
    axs[row, col  ].imshow(results[i][0])
    axs[row, col+1].imshow(results[i][1])
title = ','.join('='.join((key,str(val))) for (key,val) in params.items())
fig.suptitle(title)

# Give random number signature to this test iteration and save
letters = string.digits
name = ''.join(random.choice(letters) for i in range(10))
plt.savefig(name + '.png')