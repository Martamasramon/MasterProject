## Imports and adapted functions
from CNN2D                import CNN2D, train_model, test_model, analyse_train_test
from gniadecka_functions  import first_derivative_bl_removal
from subimages            import subimages, reconstruct_img
import numpy as np
from random import randint
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

def make_map():
    my_map = np.zeros((200,200))
    for i in range(12):
        x_range = randint(0,20)
        y_range = randint(0,20)
        x_start = randint(20,180)
        y_start = randint(20,180)
        
        my_map[x_start-x_range:x_start+x_range, y_start-y_range:y_start+y_range] = 1
    return my_map

def make_peak(data, min_val, max_val, max_width):
    location  = randint(0,1023)
    amplitude = randint(min_val,max_val)
    width     = randint(0,max_width)

    for i in range (width):
        if location + i < 1024:
            data[location + i] = amplitude - 3*i
        if location - i >= 0:
            data[location - i] = amplitude - 3*i

    return data

def add_noise(bcc_map, healthy, bcc, mean, std):
    img_spectra = np.zeros((200,200,1024))

    for i in range(200):
        for j in range(200):
            if bcc_map[i,j] == 0:
                img_spectra[i,j] = healthy + np.random.normal(mean,std,1024)
            else:
                img_spectra[i,j] = bcc     + np.random.normal(mean,std,1024)

    return img_spectra

def add_baseline(data, max_amplitude):
    x    = np.linspace(0,1023,4)
    x_bl = np.linspace(0,1023,1024)

    coeff = [randint(0,max_amplitude),randint(0,max_amplitude),randint(0,max_amplitude),randint(0,max_amplitude)]
    f = interp1d(x,coeff,kind='cubic')
    baseline = f(x_bl)

    data += baseline
    return data

def my_pca(bcc_map, img_spectra, pca_components):
    ## Train PCA only on relevant BCC data
    pca_training_data = np.zeros((1000,1024))
    transformed       = np.zeros((200,200,pca_components))
    count = 0

    for i in range(200):
        for j in range(200):
            if count < 1000:
                if bcc_map[i,j] == 1:
                    pca_training_data[count,:] = img_spectra[i,j,:]
                    count += 1

    ### Perform PCA dimensionality reduction ###
    norm_data = normalize(pca_training_data)
    pca = PCA(n_components=pca_components)
    pca.fit(norm_data)

    for i in range(200):
        transformed[i,:,:] = pca.transform(img_spectra[i,:,:])

    return transformed


def test_fake_data(bcc_map, parameters):
    LEARNING_RATE   = parameters['LEARNING_RATE']
    EPOCHS          = parameters['EPOCHS']
    DROPOUT_RATE    = parameters['DROPOUT_RATE']
    TRAIN_CUTOFF    = parameters['TRAIN_CUTOFF']
    SUBIMG_SPACING  = parameters['SUBIMG_SPACING']
    IMG_WIDTH       = parameters['IMG_WIDTH']
    PCA_COMPONENTS  = parameters['PCA_COMPONENTS']
    N_BCC_LIM       = parameters['N_BCC_LIM']
    BL_AMPLITUDE    = parameters['BL_AMPLITUDE']
    PEAK_HEIGHTS    = parameters['PEAK_HEIGHTS']
    PEAK_WIDTHS     = parameters['PEAK_WIDTHS']
    NOISE_STD       = parameters['NOISE_STD']

    ###############################################
    ## 1. Create "ideal" spectra for healthy vs. BCC
    healthy = np.zeros((1024))
    bcc     = np.zeros((1024))

    for i in range(5): # 5 big peaks
        healthy = make_peak(healthy, PEAK_HEIGHTS[0], PEAK_HEIGHTS[1], PEAK_WIDTHS[0])
        bcc     = make_peak(bcc    , PEAK_HEIGHTS[0], PEAK_HEIGHTS[1], PEAK_WIDTHS[0])
    for i in range(10): # 10 small peaks
        healthy = make_peak(healthy, 1, PEAK_HEIGHTS[0], PEAK_WIDTHS[1])
        bcc     = make_peak(bcc    , 1, PEAK_HEIGHTS[0], PEAK_WIDTHS[1])

    # Remove negative elements (make_peak can return negative values)
    healthy = [0 if i < 0 else i for i in healthy]
    bcc     = [0 if i < 0 else i for i in bcc]

    ###############################################
    ## 2. Add 0-mean Gaussian noise
    mean = 0
    img_spectra = add_noise(bcc_map, healthy,bcc,mean,NOISE_STD)

    _, axs = plt.subplots(2,2, figsize=(20,10))
    axs[0,0].plot(np.transpose(img_spectra[0:5,0,:]))
    axs[0,0].title.set_text('Initial signals - healthy')
    axs[0,1].plot(np.transpose(img_spectra[30:35,15,:]))
    axs[0,1].title.set_text('Initial signals - bcc')

    ###############################################
    ## 3. Add and remove baseline
    for i in range(200):
        for j in range(200):
            img_spectra[i,j]     = add_baseline(img_spectra[i,j],BL_AMPLITUDE)
            x = np.linspace(0,1023,1024)
            [img_spectra[i,j],_] = first_derivative_bl_removal(x,img_spectra[i,j])

    axs[1,0].plot(np.transpose(img_spectra[0:5,0,:]))
    axs[1,0].title.set_text('Add and remove baseline - healthy \n')
    axs[1,1].plot(np.transpose(img_spectra[30:35,15,:]))
    axs[1,1].title.set_text('Add and remove baseline - bcc \n')
    #plt.savefig('signals.png')

    ###############################################
    ## 4. Perform PCA
    pca_reduced = my_pca(bcc_map,img_spectra,PCA_COMPONENTS)

    ###############################################
    ## 5. Get sub-images
    subimg_input = (pca_reduced.reshape(40000,10),bcc_map,200,200)
    sub_images, target = subimages(subimg_input, IMG_WIDTH, SUBIMG_SPACING, PCA_COMPONENTS, N_BCC_LIM)

    ###############################################
    ## 6. CNN
    model = CNN2D(IMG_WIDTH,PCA_COMPONENTS,DROPOUT_RATE)
    train_model(model, sub_images[:TRAIN_CUTOFF], target[:TRAIN_CUTOFF], LEARNING_RATE, EPOCHS, IMG_WIDTH, PCA_COMPONENTS)

    approximation_train = test_model(model, sub_images[:TRAIN_CUTOFF], IMG_WIDTH, PCA_COMPONENTS)
    approximation_test  = test_model(model, sub_images[TRAIN_CUTOFF:], IMG_WIDTH, PCA_COMPONENTS)

    train_res, test_res, _ = analyse_train_test(approximation_train, approximation_test, target, TRAIN_CUTOFF, parameters, plot=1, text=1)
    total_res = train_res
    for i in test_res:
        total_res.append(i)
        
    reconstructed = reconstruct_img(total_res, [200,200], IMG_WIDTH, SUBIMG_SPACING)
    return reconstructed