import numpy as np

def subimages(tissue, img_width, subimg_spacing, pca_components, n_bcc_lim):
    new_tissues = []
    
    for (t_map,t_target,width,_) in tissue:
        sub_images = []
        target = []
        for i in range(0,200-img_width+1,subimg_spacing): 
            for j in range(0,200-img_width+1,subimg_spacing):  
                spectra = np.zeros((img_width,img_width,pca_components))
                
                for line in range(img_width):
                    index = (i+line)*width + j
                    spectra[:,line,:] = t_map[index:index+img_width,:]
                
                final_spectra = []
                # Change data structure from (40*40,PCA_COMPONENTS) to (PCA_COMPONENTS,40,40)
                for k in range(pca_components):
                    final_spectra.append(np.transpose(spectra[:,:,k]))
                
                sub_images.append(final_spectra)


                bcc = t_target[i:i+img_width,j:j+img_width]
                #target.append(bcc)
                if np.count_nonzero(bcc) > n_bcc_lim:
                    target.append(1)
                else:
                    target.append(0)
        new_tissues.append((sub_images, target))
                    
    return new_tissues


def reconstruct_img(results, shape, img_width, subimg_spacing):
    width  = shape[0]
    height = shape[1]
    reconstructed = np.zeros((width,height))
    
    count = 0
    for i in range(0,width-img_width+1,subimg_spacing): 
        for j in range(0,height-img_width+1,subimg_spacing):  
            if results[count] > 0:
                reconstructed[i:i+img_width,j:j+img_width] = 1
            count += 1

    return reconstructed