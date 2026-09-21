import numpy as np
from sklearn.feature_selection import SelectKBest, mutual_info_classif

# 1. Función para DESCUBRIR las mejores características (se usa una sola vez con todo el dataset)
def obtener_indices_ganadores(X_total, y_total, cantidad_a_retener):
    """
    X_total: Matriz con las features de TODAS las imágenes.
    y_total: Vector con las etiquetas de TODAS las imágenes.
    """
    selector = SelectKBest(score_func=mutual_info_classif, k=cantidad_a_retener)
    selector.fit(X_total, y_total)
    
    # Extraemos solo las posiciones (ej: [5, 12, 128, 255...]) de las features buenas
    indices = selector.get_support(indices=True)
    return indices


# 2. Función para APLICAR la selección a UNA SOLA IMAGEN 
def filtrar_features_imagen(vector_una_imagen, indices_ganadores):
    """
    Recibe el vector original de una sola imagen (ej. 8100 features de HOG) 
    y retorna solo las seleccionadas.
    """
    # Aseguramos que sea un arreglo de numpy
    vector = np.array(vector_una_imagen)
    
    # Si el vector es 1D (ej: forma (8100,))
    if vector.ndim == 1:
        vector_reducido = vector[indices_ganadores]
    # Si viene como matriz de una fila (ej: forma (1, 8100))
    else:
        vector_reducido = vector[:, indices_ganadores]
        
    return vector_reducido