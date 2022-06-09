## Imports and adapted functions
from curses.ascii import SUB
from CNN2D                import CNN2D, train_model, test_model, analyse_train_test
from gniadecka_functions  import first_derivative_bl_removal
from accuracy             import evaluate_performance
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

def subimages(bcc_map, spectra, img_width, subimg_spacing, pca_components, n_bcc_lim):
    sub_images = []
    target = []

    for i in range(0,200-img_width+1,subimg_spacing): # Can increase overlap - now 0 (40 spacing)
        for j in range(0,200-img_width+1,subimg_spacing):
            sub_img = spectra[i:i+img_width,j:j+img_width,:]
            final_spectra = []

            # Change data structure from (40,40,PCA_COMPONENTS) to (PCA_COMPONENTS,40,40)
            for k in range(pca_components):
                wavenum = np.zeros((img_width,img_width))
                for m in range(img_width):
                    for n in range (img_width):
                        wavenum[m,n] = sub_img[m,n,k]
                final_spectra.append(wavenum)

            bcc = bcc_map[i:i+img_width,j:j+img_width]
            sub_images.append(final_spectra)
            if np.count_nonzero(bcc) > n_bcc_lim:
                target.append(1)
            else:
                target.append(0)

    return sub_images, target

def reconstruct_img(results, img_width, subimg_spacing):
    reconstructed = np.zeros((200,200))
    
    count = 0
    for i in range(0,200-img_width+1,subimg_spacing): 
        for j in range(0,200-img_width+1,subimg_spacing):  
            if results[count] > 0:
                reconstructed[i:i+img_width,j:j+img_width] = 1
            count += 1

    return reconstructed

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

    ###############################################
    ## 3. Add and remove baseline
    for i in range(200):
        for j in range(200):
            img_spectra[i,j]     = add_baseline(img_spectra[i,j],BL_AMPLITUDE)
            x = np.linspace(0,1023,1024)
            [img_spectra[i,j],_] = first_derivative_bl_removal(x,img_spectra[i,j])

    ###############################################
    ## 4. Perform PCA
    pca_reduced = my_pca(bcc_map,img_spectra,PCA_COMPONENTS)

    ###############################################
    ## 5. Get sub-images
    sub_images, target = subimages(bcc_map, pca_reduced, IMG_WIDTH, SUBIMG_SPACING, PCA_COMPONENTS, N_BCC_LIM)
    
    ###############################################
    ## 6. CNN
    model = CNN2D(IMG_WIDTH,PCA_COMPONENTS,DROPOUT_RATE)
    train_model(model, sub_images[:TRAIN_CUTOFF], target[:TRAIN_CUTOFF], LEARNING_RATE, EPOCHS, IMG_WIDTH, PCA_COMPONENTS)

    approximation_train = test_model(model, sub_images[:TRAIN_CUTOFF], IMG_WIDTH, PCA_COMPONENTS)
    approximation_test  = test_model(model, sub_images[TRAIN_CUTOFF:], IMG_WIDTH, PCA_COMPONENTS)

    res_train, res_test, _ = analyse_train_test(approximation_train, approximation_test, target, TRAIN_CUTOFF, parameters, 0, 0)
    res_total = res_train + res_test
    #reconstructed = reconstruct_img(res_total, IMG_WIDTH, SUBIMG_SPACING)
    
    # Plot results
    #title = 'bl_amplitude: ' + str(BL_AMPLITUDE) + ', noise std: ' + str(NOISE_STD)
    #fig, axs = plt.subplots(1,2, figsize=(20,5))
    #axs[0].imshow(bcc_map)
    #axs[1].imshow(reconstructed)
    #fig.suptitle(title)
    #plt.savefig('fake_data_' + str(BL_AMPLITUDE) + '_' + str(NOISE_STD) + '.png')
    
    # Save results in text file
    performance         = np.zeros((4,1))
    performance[0], performance[1], performance[2], performance[3], _ = evaluate_performance(res_test, target[TRAIN_CUTOFF:])
    
    with open('fake_data.txt', 'a') as f:
        f.write('-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- \n')
        f.write('Baseline amplitude: ' + str(BL_AMPLITUDE) + '\n')
        f.write('Noise std: ' + str(NOISE_STD) + '\n')  
        for item in performance:
            f.write(f"{str(item):^10}")
        f.write('\n-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- \n\n\n\n')
