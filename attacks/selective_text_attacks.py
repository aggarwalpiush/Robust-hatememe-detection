import easyocr
import json
import os
from PIL import Image

import numpy as np
import os
import cv2


def get_text_locs (ocr_out):
    x_range = []
    y_range = []
    for box in ocr_out:
        box = box[0]
        min_x = min(min(box[0][0],box[1][0]),min(box[2][0],box[3][0]))
        max_x = max(max(box[0][0],box[1][0]),max(box[2][0],box[3][0]))
        min_y = min(min(box[0][1],box[1][1]),min(box[2][1],box[3][1]))
        max_y = max(max(box[0][1],box[1][1]),max(box[2][1],box[3][1]))
        x_range.append([min_x,max_x])
        y_range.append([min_y,max_y])
    return x_range, y_range





def blur_text (image, xs, ys):
    out = np.copy(image)
    temp = np.copy(image)
    for i in range(len(xs)):
        x_min = int(max(0,xs[i][0]))
        x_max = int(min(xs[i][1],image.shape[1]))
        y_min = int(max(0,ys[i][0]))
        y_max = int(min(ys[i][1],image.shape[0]))
        temp_image = image[y_min:y_max,x_min:x_max,:]
        image = temp
#         print(temp_image.shape)
        if len(temp_image)>0:
            blurImg = cv2.blur(temp_image,(5,5)) 
            out[y_min:y_max,x_min:x_max,:] = blurImg
    return out

def salt_pepper_noise(image, xs, ys):
    row,col,ch = image.shape
    out = np.copy(image)
    # Salt mode
    s_vs_p = 0.5
    amount = 0.2
#     num_salt = np.ceil(amount * image.size * s_vs_p)
#     print([i for i in image.shape])
#     print(image.size)
    
    for j in range(len(xs)):
        coords = []
        num_salt = np.ceil(amount * (xs[j][1]-xs[j][0])*(ys[j][1]-ys[j][0])*3 * s_vs_p)
        for i,value in enumerate(image.shape):
            if i==0:
                temp_val = np.random.randint(max(0,ys[j][0]), min(ys[j][1]-1,image.shape[0]-1), int(num_salt))
            elif i==1:
                temp_val = np.random.randint(max(0,xs[j][0]), min(xs[j][1]-1,image.shape[1]-1), int(num_salt))
            else:
                temp_val = np.random.randint(0, value - 1, int(num_salt))
            coords.append(temp_val)
        out[coords] = 1
        coords = []
        num_pepper = np.ceil(amount* (xs[j][1]-xs[j][0])*(ys[j][1]-ys[j][0])*3  * (1. - s_vs_p))
        for i,value in enumerate(image.shape):
            if i==0:
                temp_val = np.random.randint(max(0,ys[j][0]), min(ys[j][1]-1,image.shape[0]-1), int(num_pepper))
            elif i==1:
                temp_val = np.random.randint(max(0,xs[j][0]), min(xs[j][1]-1,image.shape[1]-1), int(num_pepper))
            else:
#                 print(value)
                temp_val = np.random.randint(0, value - 1, int(num_pepper))
            coords.append(temp_val)
        out[coords] = 0
    return out

dataset = 'harmeme'

img_path = '../datasets/'+dataset+'/img'
attack = 'blur_text_5'


write_dir = '../datasets/'+dataset+'/'+attack

with open(file_test,'r') as f:
    data = json.load(f)

    
if not os.path.exists(write_dir):
    os.makedirs(write_dir)
reader = easyocr.Reader(['en'])
for i,dp in enumerate(data):
    img_path_temp = os.path.join(img_path,dp['img'])
    # img_path_temp = os.path.join(img_path,'89432.png')
    result_attack = reader.readtext(img_path_temp,decoder='wordbeamsearch')
    inp_img = cv2.imread(img_path_temp)
    x_range, y_range = get_text_locs(result_attack)
    #     att_img = salt_pepper_noise(inp_img,x_range,y_range)
    att_img = blur_text(inp_img,x_range,y_range)
    write_path = os.path.join(write_dir,dp['img'])
    # write_path = os.path.join(write_dir,'89432.png')

    print(i,write_path)
    cv2.imwrite(write_path,att_img)



# img = blur_text(inp_img,x_range,y_range)
# cv2.imwrite('test.png',img)