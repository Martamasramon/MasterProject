# %%
import numpy as np

# %%
def subimages(tissue, img_width, subimg_spacing, pca_components, n_bcc_lim):
    sub_images = []
    target = []

    for (t_map,t_target,width,height) in tissue:
        for i in range(0,width-img_width+1,subimg_spacing): # Can increase overlap - now 0 (40 spacing) 
            for j in range(0,height-img_width+1,subimg_spacing):
                index = i*height + j
                spectra = t_map[index:index+img_width*img_width,:]
                final_spectra = []
                
                # Change data structure from (40,40,PCA_COMPONENTS) to (PCA_COMPONENTS,40,40)
                for k in range(pca_components):
                    wavenum = np.zeros((img_width,img_width))
                    for m in range(img_width):
                        for n in range (img_width):
                            index = m*img_width + n
                            wavenum[m,n] = spectra[index,k]
                    final_spectra.append(wavenum)
                
                bcc = t_target[i:i+img_width,j:j+img_width]
                sub_images.append(final_spectra)
                if np.count_nonzero(bcc) > n_bcc_lim:
                    target.append(1)
                else:
                    target.append(0)
                    
    return sub_images, target


