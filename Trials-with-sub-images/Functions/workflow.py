from CNN2D       import CNN2D, train_model, test_model, get_error, analyse_results
from PCA         import my_pca
from subimages   import subimages

def train_test_workflow(tissue, PCA_COMPONENTS, IMG_WIDTH, SUBIMG_SPACING, DROPOUT_RATE, LEARNING_RATE, EPOCHS, TRAIN_CUTOFF):
    ## PCA transformation
    new_tissue = my_pca(tissue, PCA_COMPONENTS)
    
    ### Get sub-images of size 40x40 ###
    sub_images, target = subimages(new_tissue, IMG_WIDTH, SUBIMG_SPACING, PCA_COMPONENTS)
    
    ### Create 2D-CNN model ###
    model = CNN2D(IMG_WIDTH,PCA_COMPONENTS,DROPOUT_RATE)

    ### Train model ###
    # Input of CNN needts to be (NumberSamples,Channels,Height,Width) 
    model = train_model(model, sub_images[:TRAIN_CUTOFF], target[:TRAIN_CUTOFF], LEARNING_RATE, EPOCHS, IMG_WIDTH, PCA_COMPONENTS)

    ### Test model ###
    approximation_train = test_model(model, sub_images[:TRAIN_CUTOFF], IMG_WIDTH, PCA_COMPONENTS)
    approximation_test  = test_model(model, sub_images[TRAIN_CUTOFF:], IMG_WIDTH, PCA_COMPONENTS)

    _ = analyse_results(approximation_train, approximation_test, target, TRAIN_CUTOFF, plot=1)
