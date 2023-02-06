import json
import pandas as pd
import os



data = pd.read_csv('datasets/mami/val.csv')
print (data)
# with open('datasets/harmeme/test.jsonl','r') as f:
#     data = list(f)
    

list_data = []    
for i in range(len(data)):
    point = data.loc[i]
    dict_temp = {}
    dict_temp['img'] = point['file_name']
#     print (point['labels'])
    if point['misogynous']==1:
        print("going")
        dict_temp['label'] = 1
    else:
        dict_temp['label']  = 0
#     dict_temp['label'] = int(point['misogynous'])
    dict_temp['text'] = point['Text Transcription']
    list_data.append(dict_temp)

    
print (list_data)
    
json_object = json.dumps(list_data, indent = 4)
  
# Writing to sample.json
with open("datasets/mami/files/val.json", "w") as outfile:
    outfile.write(json_object)