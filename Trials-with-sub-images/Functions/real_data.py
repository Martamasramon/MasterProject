from CNN2D       import CNN2D, train_model, test_model, get_error
from pca         import my_pca
from subimages   import subimages, reconstruct_img
import matplotlib.pyplot as plt

# Parameters
def set_params(PCA_COMPONENTS=20,IMG_WIDTH=10,SUBIMG_SPACING=5,N_BCC_LIM=8,LEARNING_RATE=0.001,EPOCHS=5000,DROPOUT_RATE=0.1):
    parameters = {
        'PCA_COMPONENTS'  : PCA_COMPONENTS,
        'IMG_WIDTH'       : IMG_WIDTH,
        'SUBIMG_SPACING'  : SUBIMG_SPACING,
        'N_BCC_LIM'       : N_BCC_LIM,
        'LEARNING_RATE'   : LEARNING_RATE,
        'EPOCHS'          : EPOCHS,
        'DROPOUT_RATE'    : DROPOUT_RATE
    }
    return parameters

# For now load data without bl, later include baseline removal 
# Also just using data with bcc map 
def test_params(parameters, data,name):
    LEARNING_RATE   = parameters['LEARNING_RATE']
    EPOCHS          = parameters['EPOCHS']
    DROPOUT_RATE    = parameters['DROPOUT_RATE']
    SUBIMG_SPACING  = parameters['SUBIMG_SPACING']
    IMG_WIDTH       = parameters['IMG_WIDTH']
    PCA_COMPONENTS  = parameters['PCA_COMPONENTS']
    N_BCC_LIM       = parameters['N_BCC_LIM']
    
    length = len(data)
    data = my_pca(data, PCA_COMPONENTS)
    sub_images = subimages(data, IMG_WIDTH, SUBIMG_SPACING, PCA_COMPONENTS, N_BCC_LIM)

    # Create model
    model = CNN2D(IMG_WIDTH,PCA_COMPONENTS,DROPOUT_RATE)

    for i in range(5):
        model = train_model(model, sub_images[i][0], sub_images[i][1], LEARNING_RATE, EPOCHS, IMG_WIDTH, PCA_COMPONENTS)

    results = []
    for i in range(length):
        approximation = test_model(model, sub_images[i][0], IMG_WIDTH, PCA_COMPONENTS)
        _, result = get_error(approximation, sub_images[i][1])
        reconstructed = reconstruct_img(result, [200,200], IMG_WIDTH, SUBIMG_SPACING)
        results.append((data[i][1],reconstructed))
    
    # Plot 
    fig, axs = plt.subplots(2,9, figsize=(10,50))
    for i in range(length):
        axs[0,i].imshow(results[i][0])
        axs[1,i].imshow(results[i][1])
    title = ','.join('='.join((key,str(val))) for (key,val) in parameters.items())
    fig.suptitle(title)
    plt.savefig(name + '.png')