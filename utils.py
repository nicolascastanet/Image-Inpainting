from PIL import Image
import matplotlib.colors as mcolors
import random
import copy
from sklearn import linear_model
import numpy as np
from numpy import linalg as LA
import pandas as pd
import random
import matplotlib.pyplot as plt
from matplotlib import cm

filename = 'mer2_rogn.jpg'
def read_im(fn):
    pixels = plt.imread(fn)
    # Normalize between 0 and 1
    # confirm pixel range
    print('Data Type: %s' % pixels.dtype)
    print('Min: %.3f, Max: %.3f' % (pixels.min(), pixels.max()))

    # convert from integers to floats
    pixels = pixels.astype('float32')
    # normalize to the range 0-1
    pixels /= 255.0
    # confirm the normalization
    print('Min: %.3f, Max: %.3f' % (pixels.min(), pixels.max()))
    
    # Convert to HSV
    pixels = mcolors.rgb_to_hsv(pixels)
    
    return pixels

def plot_img(img):
    plt.imshow(mcolors.hsv_to_rgb(img))

def get_patch(i,j,h,img):
    #import ipdb; ipdb.set_trace()
    return img[int(i-(h-1)/2):int(i+(h-1)/2),int(j-(h-1)/2):int(j+(h-1)/2)]

def get_patch_boundaries(i,j,h):
    """
    retourne les indices de la bordure intérieure du patch de centre (i,j) et de taille h 
    """
    
    b1 = np.array([[u,int(j-(h-1)/2)] for u in range(int(i-(h-1)/2),int(i+(h-1)/2))])
    b2 = np.array([[u,int(j+(h-1)/2)-1] for u in range(int(i-(h-1)/2),int(i+(h-1)/2))])    
    b3 = np.array([[int(i-(h-1)/2),u] for u in range(int(j-(h-1)/2),int(j+(h-1)/2))])
    b4 = np.array([[int(i+(h-1)/2)-1,u] for u in range(int(j-(h-1)/2),int(j+(h-1)/2))])
    
    ind_pix = []
    for u in range(int(i-(h-1)/2),int(i+(h-1)/2)):
        for v in range(int(j-(h-1)/2),int(j+(h-1)/2)):
            ind_pix.append([u,v])
    
    border = np.concatenate((b1,b2,b3,b4)).T
    
    return border, np.array(ind_pix).T
    

def get_all_patchs(h, img, step):
    #return all patchs with length h
    n, m, d = img.shape
    patchs = []
    for i in range(int(h/2), int(n - h/2), step):
        for j in range(int(h/2), int(m - h/2), step):
            patchs.append(conv_patch(get_patch(i,j,h,img)))
    return patchs

def get_dict(h, img, step):
    #return all patchs with and without missing pixels
    patchs = get_all_patchs(h, img, step)
    dict_patchs = []
    noised_patchs = []
    for p in patchs:
        if -100 in set(p):
            noised_patchs.append(p)
        else:
            dict_patchs.append(p)
            
    return dict_patchs, noised_patchs


def aprox_patch(patch, dico, classi):
    
    expr_pixels_index = np.where(patch != -100)[0]
    
    Y = patch[expr_pixels_index]
    X = np.array(dico).T[expr_pixels_index,:]

    #import ipdb; ipdb.set_trace()
    classi.fit(X,Y)
    
    return classi.coef_

def reconstruc_patch(patch, coeffs, dico):
    
    reconstruct_patch = copy.deepcopy(patch)
    noised_pixels_index = np.where(patch == -100)[0]
    X = np.array(dico).T
    reconstruct_pixels = np.dot(X,coeffs)
    
    #import ipdb; ipdb.set_trace()
    for i in noised_pixels_index:
        reconstruct_patch[i] = reconstruct_pixels[i]
    return reconstruct_patch
    
    
    
def conv_patch(patch):
    #convert tensor to vect
    n, m, d = patch.shape
    p = patch.reshape((n**2)*3,1)
    return p.squeeze()

def plot_patch(patch, h):
    p = patch.reshape(h,h,3)
    plt.imshow(mcolors.hsv_to_rgb(p))


def noise(prc, img):
    # Fonction à optimiser !
    
    noised_img = copy.deepcopy(img)
    n, m, d = img.shape
    
    nb_pixel = n*m
    nb_pixel_suppr = int((prc/100)*nb_pixel)
    
    random_rows = [random.randint(0,n-1) for i in range(nb_pixel_suppr)]
    random_columns = [random.randint(0,m-1) for i in range(nb_pixel_suppr)]
    
    noised_img[random_rows, random_columns, :] = -100
   
    return noised_img


def delete_rect(img,i,j,height,width):
    #à optimiser
    n, m, d = img.shape
    n_img = copy.deepcopy(img)
    
    #On initialise les confiances des pixels lorsque la zone à restaurer est sélectionnée
    confidence = np.ones((n,m))
    
    n_img[max(0,int(i-(height-1)/2)):min(n,int(i+(height-1)/2)),max(0,int(j-(width-1)/2)):min(m,int(j+(width-1)/2)),:] = -100
    confidence[max(0,int(i-(height-1)/2)):min(n,int(i+(height-1)/2)),max(0,int(j-(width-1)/2)):min(m,int(j+(width-1)/2))] = 0

    #On initialise les frontières de la zone à restaurer (bordure extérieure)
    b1 = np.array([[u,max(0,int(j-(width-1)/2))-1] for u in range(max(0,int(i-(height-1)/2)),min(n,int(i+(height-1)/2)))])
    b2 = np.array([[u,min(m,int(j+(width-1)/2))] for u in range(max(0,int(i-(height-1)/2)),min(n,int(i+(height-1)/2)))])    
    b3 = np.array([[max(0,int(i-(height-1)/2))-1,u] for u in range(max(0,int(j-(width-1)/2)),min(m,int(j+(width-1)/2)))])
    b4 = np.array([[min(n,int(i+(height-1)/2)),u] for u in range(max(0,int(j-(width-1)/2)),min(m,int(j+(width-1)/2)))])
    
    border = np.concatenate((b1,b2,b3,b4)).T
    
    #test de la bordure
    #n_img[border[0], border[1]] = [0,1,1]
    
    return n_img, border, confidence


def update_bound(old_img, new_img, old_bound,i,j,h):
    """
    old_bound : matrice d'indices des pixels sur la bordure extérieure de l'image avant l'itération actuelle
    patch_bound : matrice d'indices des pixels sur la bordure intérieure du patch
    """
    patch_bound, patch_ind = get_patch_boundaries(i,j,h)
    x = np_intersect2D(old_bound.T, patch_ind) #On récupère les pixels du patch qui étaient sur la bordure
    
    x_hashable = map(tuple, x)
    pix_suppr = set(x_hashable)
    
    bound_hashable = map(tuple,old_bound.T)
    set_patch_bound = set(bound_hashable)
    
    set_new_bound = set_patch_bound.difference(pix_suppr)
    
    new_bound = np.array(list(set_new_bound))
    
    new_p = []
    
    for p in patch_bound.T:
        #import ipdb;ipdb.set_trace()
        if -100.0 in set(old_img[p[0],p[1]]) and -100.0 not in set(new_img[p[0],p[1]]):
            new_p.append(list(p))
            
    #mport ipdb;ipdb.set_trace()
    if len(np.array(new_p)) == 0:
        return new_bound
    else:
        return np.concatenate((new_bound,np.array(new_p)))



def priorities(border, conf_matr, h, img):
    max_c, max_p = -1, -1
    for p in border.T:
        patch = get_patch(p[0],p[1],h,img)
        patch_bound, patch_ind = get_patch_boundaries(p[0],p[1],h)
        #import ipdb; ipdb.set_trace()
        conf_p = np.sum(conf_matr[patch_ind[0], patch_ind[1]])/(patch.shape[0]*patch.shape[1])
        conf_matr[p[0], p[1]] = conf_p
        if conf_p > max_c:
            max_c = conf_p
            max_p = p
    
    return max_p, conf_matr


def np_intersect2D(A,B):
    
    nrows, ncols = A.shape
    dtype={'names':['f{}'.format(i) for i in range(ncols)],
       'formats':ncols * [A.dtype]}

    C = np.intersect1d(A.view(dtype), B.view(dtype))
    C = C.view(A.dtype).reshape(-1, ncols)
    return C


sea_img = read_im(filename)
plot_img(sea_img)
print(sea_img.shape)