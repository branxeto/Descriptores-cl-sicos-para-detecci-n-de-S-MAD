import numpy as np
from skimage.feature import local_binary_pattern

def uLBP_features(img):
    radio = 1 # Radio de mira
    puntos = 8 * radio # Toma 8 puntos aparte del centro
    
    lbp_uniforme = local_binary_pattern(img, puntos, radio, method='uniform') # Devuelve matriz 2D
    
    n_bins = int(lbp_uniforme.max() + 1)
    vector_ulbp, _ = np.histogram(lbp_uniforme.ravel(), bins=n_bins, range=(0, n_bins))
    
    vector_ulbp = vector_ulbp.astype('float')
    vector_ulbp /= (vector_ulbp.sum() + 1e-7)
    
    return vector_ulbp
            