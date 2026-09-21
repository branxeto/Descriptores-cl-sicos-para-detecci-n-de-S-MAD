import numpy as np
from scipy.signal import correlate2d

def bsif_features(img, texture_filters):
    # Inicializar
    img = img.astype(float)
    num_scl = texture_filters.shape[2]
    
    # En el código original se inicializa en 1, no en 0
    code_img = np.ones(img.shape)
    
    # Radio para el padding
    r = texture_filters.shape[0] // 2
    
    # Wrap image (Padding circular)
    # np.pad con mode='wrap' hace exactamente el mismo ensamblaje matricial que MATLAB
    img_wrap = np.pad(img, pad_width=r, mode='wrap')
    
    # Loop sobre las escalas (filtros)
    for i in range(num_scl):
        # En MATLAB extrae el filtro en orden inverso. 
        # En Python ajustamos por el índice 0.
        tmp = texture_filters[:, :, num_scl - 1 - i]
        
        # filter2 en MATLAB es correlación cruzada 2D, equivalente a correlate2d
        ci = correlate2d(img_wrap, tmp, mode='valid')
        
        # Binarización y suma (2^i porque Python itera desde 0)
        code_img = code_img + (ci > 0) * (2 ** i)
        
    rango_max = 2 ** num_scl
    hist_vals, _ = np.histogram(code_img.ravel(), bins=rango_max, range=(1, rango_max + 1))
    
    # Normalizar histograma
    hist_vals = hist_vals.astype(float) / np.sum(hist_vals)
        
    return hist_vals