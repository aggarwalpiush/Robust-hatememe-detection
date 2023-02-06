import json
import os



datasets = ['train','val','test']
for dt in datasets:

    with open('../datasets/harmeme/files/'+dt+'.json','r') as f:
        data = json.load(f)


    # list_data = []    

    # for point in data:
    #     point = json.loads(point)
    #     dict_temp = {}
    #     dict_temp['img'] = point['img'].split('/')[-1]
    # #     print (point['labels'])
    # #     if point['labels'][0]=='not harmful':
    # #         print("going")
    # #         dict_temp['label'] = 0
    # #     else:
    # #         dict_temp['label']  = 1
    #     dict_temp['label'] = point['label']
    #     dict_temp['text'] = point['text']
    #     list_data.append(dict_temp)



    # json_object = json.dumps(list_data, indent = 4)

    # # Writing to sample.json
    # with open("datasets/fb/files/test.json", "w") as outfile:
    #     outfile.write(json_object)

    hate = 0
    nonhate = 0

    for point in data:
        if point['label']==1:
            hate+=1
        else:
            nonhate+=1


    print(hate,nonhate,len(data))