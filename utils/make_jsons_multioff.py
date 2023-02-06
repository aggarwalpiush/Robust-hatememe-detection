import json
import pandas as pd
import os



data = pd.read_csv('datasets/multioff/MultiOFF_Dataset/Split Dataset/Testing_meme_dataset.csv')
print (data)
# with open('datasets/harmeme/test.jsonl','r') as f:
#     data = list(f)
    

list_data = []    
for i in range(len(data)):
    point = data.loc[i]
    dict_temp = {}
    dict_temp['img'] = point['image_name']
#     print (point['labels'])
    if point['label']=='offensive':
        print("going")
        dict_temp['label'] = 1
    else:
        dict_temp['label']  = 0
    dict_temp['text'] = point['sentence']
    list_data.append(dict_temp)

    
print (list_data)
    
json_object = json.dumps(list_data, indent = 4)
  
# Writing to sample.json
with open("datasets/multioff/files/test.json", "w") as outfile:
    outfile.write(json_object)