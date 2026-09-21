from skimage.feature import hog

def HOG_features(img):
    vector_hog, imagen_hog = hog(img, 
                             orientations=9, 
                             pixels_per_cell=(8, 8),
                             cells_per_block=(2, 2), 
                             visualize=True, 
                             feature_vector=True)
    return vector_hog