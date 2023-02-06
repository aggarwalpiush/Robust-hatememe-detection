import easyocr
import json
import os
from PIL import Image



dataset = 'harmeme'
file_test = '../datasets/'+dataset+'/files_new/test.json'
img_path = '../datasets/'+dataset+'/img'
# attack_img_path = '../datasets/'+dataset+'/blur_text_5'
attack_img_path = '../datasets/'+dataset+'/s&p_text_0.2'

with open(file_test,'r') as f:
    data = json.load(f)
reader = easyocr.Reader(['en']) 
    
list_new = []
for i,dp in enumerate(data):
    print(i)
#     img_path_temp = os.path.join(img_path,dp['img'])
    attack_path_temp = os.path.join(attack_img_path,dp['img'])
#     result = reader.readtext(img_path_temp,decoder='wordbeamsearch')
#     sent = ""
# #     print(result)
#     for i in result:
#         sent += i[1] + " "
    print(attack_img_path)
    result_attack = reader.readtext(attack_path_temp,decoder='wordbeamsearch')
    sent_attack = ""
    for i in result_attack:
        sent_attack += i[1] + " "
#     dp['text_ocr'] = sent
    dp['text_s&p_text_0.2'] = sent_attack
#     dp['text_blur_text_5'] = sent_attack
    print(dp)
    list_new.append(dp)
    
    
# print(list_new)


json_object = json.dumps(list_new, indent = 4)
  
# Writing to sample.json
write_dir = '../datasets/'+dataset+'/files_new'
if not os.path.exists(write_dir):
    os.makedirs(write_dir)
file_name = os.path.join(write_dir,'test.json')
with open(file_name, "w") as outfile:
    outfile.write(json_object)