from utils import *


# Lecture de l'image
filename = "personne_champs.jpg"
p = read_im(filename)
plt.figure()
plt.imshow(mcolors.hsv_to_rgb(p))
plt.show()

# Selection de la zone à restaurer
#img, border, conf = delete_rect(p,125,170,150,100) #Voilier
#img, border, conf = delete_rect(p,100,70,50,50) #Oiseaux
#img, border, conf = delete_rect(p,100,200,100,90) #Arbre
img, border, conf = delete_rect(p,90,210,80,40) #personne champs

plt.figure()
plt.imshow(mcolors.hsv_to_rgb(img))
plt.show()

# Hyper-parametres

h = 27 #patch length
step = 1 # Pas de construction du dictionnaire
max_iterations = 100000 #iterations du LASSO
a = 0.004

# Creation du dictionnaire de patch

all_patchs = get_all_patchs(h, img, step)
print(f"num of patchs of length {h}: {len(all_patchs)}\n--------------------------------")
dict_patchs, noised_patchs = get_dict(h,img, step)
print(f"num of patchs in the dict: {len(dict_patchs)}\nnum of noised patchs: {len(noised_patchs)}")

#Classifier
clf = linear_model.Lasso(alpha=a, max_iter = max_iterations, normalize = False, positive = False, warm_start = True)

i = 0

# Tant qu'il y a des pixels à restaurer
while len(np.where(img == -100)[0]) != 0:
    print(f"\nitération {i}")
    # On calcul les priorités des pixels sur la bordure et on retourne le pixel prioritaire
    #import ipdb;ipdb.set_trace()
    pixel, conf = priorities(border, conf, h, img)
    
    # On recupère le patch du pixel prioritaire
    patch = get_patch(pixel[0],pixel[1],h,img)
    patch_bound, patch_ind = get_patch_boundaries(pixel[0],pixel[1],h)
    
    # On calcule les coefficients de reconstitution du patch
    coeffs = aprox_patch(conv_patch(patch), dict_patchs,clf)
    new_patch = np.clip(reconstruc_patch(conv_patch(patch), coeffs, dict_patchs),0,1)
    
    
    # On crée une nouvelle image avec le nouveau patch reconstitué
    new_img = copy.deepcopy(img)
    new_img[patch_ind[0], patch_ind[1]] = new_patch.reshape(patch.shape[0]**2,3)
    
    # On met à jour la frontière de la zone à restaurer
    border = update_bound(img, new_img, border, pixel[0], pixel[1], h)
    border = border.T
    
    # On met à jour l'image
    img = copy.deepcopy(new_img)
    
    plt.figure()
    plt.imshow(mcolors.hsv_to_rgb(img))
    plt.show()
    
    i+=1
# Affichage de l'image restaurer
plt.imshow(mcolors.hsv_to_rgb(img))