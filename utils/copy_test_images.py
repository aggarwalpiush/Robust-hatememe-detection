import os 
import shutil
import json


dataset_name = 'harmeme'

base_dir = '../datasets'



data_dir = os.path.join(base_dir,dataset_name,'test_imgs')
file_name = os.path.join(base_dir,dataset_name,'files','test.json')
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

    
with open(file_name,'r') as f:
    data = json.load(f)
    
    
for dp in data:
    img_name = dp['img']
    img_name_source = os.path.join(base_dir,dataset_name,'img',img_name)
#     print(img_name)
    dest_dir = os.path.join(data_dir,img_name)
    print(img_name_source,dest_dir)
    shutil.copy(img_name_source,dest_dir)

