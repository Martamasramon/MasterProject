# # Functions for Gniadecka et al.'s algorithm
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
import numpy as np 
from numpy import gradient
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

def moving_average(x, w):
    a = np.ones(int(w/2)) * x[0]
    z = np.ones(int(w/2)) * x[-1]
    x = np.concatenate((a,x,z))
    return np.convolve(x, np.ones(w), 'valid') / w

def left_boundary(signal, index):
    index = int(np.floor(index))
    if index == 0:
        return index
        
    while signal[index-1]>signal[index]:
        index -= 1
        if index == 0:
            return index
    
    while signal[index]>0.1 and signal[index-1]<signal[index]:
        index -= 1
        if index == 0:
            return index
        
    return index

def right_boundary(signal, index):
    index = int(np.ceil(index))
    if index == 1023:
        return index
        
    while signal[index-1]<signal[index]:
        index += 1
        if index == 1023:
            return index
    
    while signal[index]<-0.1 and signal[index-1]>signal[index]:
        index += 1
        if index == 1023:
            return index
        
    return index

def cum_sum(x):
    count = 0
    for i in range(len(x)):
        count += x[i]
        x[i] = count
    return x

def first_derivative_bl_removal(x,signal,
                                SG_WIN  = 11,
                                MA_WIN  = 61,
                                MA_WIN2 = 31,
                                MIN_BOUND_LEN = 8):
    """ 
    Input is a raman spectrum:
    - x holds the wavenumbers 
    - signal holds the intensity at each wavenumber
    
    Output:
    - final spectra (without baseline)
    """
    LEN_SIG = len(signal)
    
    # Get baseline contribution to derivative of spectrum
    s_sg = savgol_filter(signal, SG_WIN, 2, mode='nearest')
    s_dif = gradient(s_sg)
    s_sg2 = savgol_filter(s_dif, SG_WIN, 2, mode='nearest')
    s_ma1 = moving_average(s_sg2, MA_WIN)
    s_ma2 = moving_average(s_ma1, MA_WIN)
    s_ma3 = moving_average(s_ma2, MA_WIN)

    # Remove baseline contribution to obtain relevant derivative features (peaks)
    signal_derivative = s_sg2 - s_ma3

    # Find peak locations
    peaks = []
    for i in range(LEN_SIG-1):
        if signal_derivative[i]>0 and signal_derivative[i+1]<0:
            peaks.append(i+0.5)

    # Find peak boundaries
    left_boundaries = []
    right_boundaries = []

    for i in range(len(peaks)):
        lb = left_boundary(signal_derivative,peaks[i])
        rb = right_boundary(signal_derivative,peaks[i])
        left_boundaries.append(lb)
        right_boundaries.append(rb)
        
    # Remove peaks with small length between boundaries (considered not important)
    count = 0
    for i in range(len(peaks)):
        if right_boundaries[i-count]-left_boundaries[i-count] < MIN_BOUND_LEN:
            left_boundaries.pop(i-count)
            right_boundaries.pop(i-count)
            count += 1
        
    # Interpolate around peaks
    interpolated_peaks = signal.copy()

    for i in range(len(left_boundaries)):
        l = left_boundaries[i]
        r = right_boundaries[i]
        
        peak = s_ma3[l:r]
        no_mean = peak - np.mean(peak)
        integrated = cum_sum(no_mean)

        x = [l,r]
        y = [signal[l], signal[r]]
        f = interp1d(x,y)
        
        count = 0
        while l+count < r:
            interpolated_peaks[l+count] = integrated[count] + f(l+count)
            count += 1
     
    # Smooth       
    baseline = moving_average(interpolated_peaks, MA_WIN2)
    baseline = moving_average(baseline, MA_WIN2)
    
    # Add constant (make all values positive) - don't see why
    # minimum = np.min(signal-baseline,0)
    
    final_spectrum = signal-baseline
    
    return final_spectrum, baseline
    

def dim_red_PCA(data,n_components='mle'):
    """[
    Input:
        data: input data, array of size (n_samples,n_features)
        n_components (optional): number of PCs used in PCA

    Output:
        transformed: reduced data
    """
    # n_samples  = data.shape[0]
    # n_features = data.shape[1]
    
    norm_data = normalize(data)
    
    pca = PCA(n_components=n_components) 
    pca.fit(norm_data)
    transformed = pca.transform(norm_data)

    return pca, transformed


