import matplotlib as plt
import numpy as np
from scipy.ndimage import uniform_filter


def extract_feat(imgs , features_fns , verbose = True):
    """
        images are passed in standard num_of * width * height * channels
        features_fns is an array of functions which :
                takes in : width * height * channels 
                splits out : single dim array with the feature extraction function applied (lenght Fi)

        returns : 
        array of (f1 + f2 + f3 , N) # What makes sure that f1 , f2 , f3 are all of the same size ? 
                                   ?? Not added , but concatenated 
        All the features applied to a single image are concatenated into a single column
        Number of rows = number of feature dimensions 
        Number of cols = number of images
    """
    num_imgs = imgs.shape[0]
    if num_imgs == 0:
        return np.array([])
    np.squeeze

    # get feature dimensions
    features_dims = []
    first_img_features = []
    for feature_fn in features:
        feats = feature_fn(imgs[0].squeeze()) 
        assert len(feats.shape) == 1 , "Feature fn result must be 1D"
        feature_dims.append(feats.size)
        first_img_features.append(feats)

    # Allocate large array to store all the features as columns ?? Multiple columns or single columns 
    total_feature_dim = sum(features_dims) # Total length of the column
    imgs_features = np.zeros((total_feature_dim, num_imgs))
    imgs_features[:total_features_dim , 0] = np.hstack(first_img_features)

    # Extract features for the other images
    for i in xrange(1 , num_imgs):
        idx = 0
        for fn , fn_dim in zip(features_fns , features_dims):
            next_idx = idx + fn_dim
            imgs_features[id:next_idx, i]  = fn(imgs[i].squeeze()) # concatenate the output of applying each feature
            idx = next_idx
        if verbose and i %100 == 0:
            print "Done extracting features for %d / %d images" %(i , num_imgs)

    return imgs_features



def rgb2gray(rgb):
    return np.dot(rgb[... , :3] , [0.299 , 0.537 , 0.144])


def hog_feature(im):
    """
        Compute Histogram of Oriented Gradients (HOG)  features of img
    """
    if im.ndim == 3:
        image = rgb2gray(im)
    else:
        image = np.atleast_2d(im)


    sx , sy = image.shape
    orientation = 9 # number of gradient bins
    cx , cy = (8,8) # pixcels per cell


    gx






























