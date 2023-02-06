import numpy as np
import os
import cv2
def noisy(noise_typ,image):
  if noise_typ == "gauss":
    row,col,ch= image.shape
    mean = 0
    var = 0.1
    sigma = var**0.5
    gauss = np.random.normal(mean,sigma,(row,col,ch))
    gauss = gauss.reshape(row,col,ch)
    noisy = image + gauss
    return noisy
  elif noise_typ == "s&p":
    row,col,ch = image.shape
    s_vs_p = 0.5
    amount = 0.4
    out = np.copy(image)
    # Salt mode
    num_salt = np.ceil(amount * image.size * s_vs_p)
    coords = []
    for i in image.shape:
        temp_val = np.random.randint(0, i - 1, int(num_salt))
        coords.append(temp_val)
#     coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
    out[coords] = 1

      # Pepper mode
    num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
    coords = [np.random.randint(0, i - 1, int(num_pepper))
            for i in image.shape]
    out[coords] = 0
    return out
    
  elif noise_typ == "poisson":
    print("attacking")
    vals = len(np.unique(image))
    vals = 2 ** np.ceil(np.log2(vals))
    noisy = np.random.poisson(image * vals) / float(vals)
    return noisy
  elif noise_typ =="speckle":
    row,col,ch = image.shape
    gauss = np.random.randn(row,col,ch)
    gauss = gauss.reshape(row,col,ch)        
    noisy = image + image * gauss
    return noisy




import easyocr
import json
import os
from PIL import Image



dataset = 'harmeme'
img_path = '../datasets/'+dataset+'/img'
attack = 's&p_0.4'


write_dir = '../datasets/'+dataset+'/'+attack

with open(file_test,'r') as f:
    data = json.load(f)

    
if not os.path.exists(write_dir):
    os.makedirs(write_dir)

for dp in data:
    img_path_temp = os.path.join(img_path,dp['img'])
    inp_img = cv2.imread(img_path_temp)
    att_img = noisy('s&p',inp_img)
    write_path = os.path.join(write_dir,dp['img'])
    print(write_path)
    cv2.imwrite(write_path,att_img)
