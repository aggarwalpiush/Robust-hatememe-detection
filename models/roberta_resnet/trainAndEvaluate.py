import pandas as pd
import os
import numpy as np
import torch
import random
import functools
import operator
import cv2
import collections
import torchvision.models as models
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader, Dataset
from transformers import get_linear_schedule_with_warmup, BertTokenizer, BertModel, VisualBertModel, VisualBertConfig, RobertaTokenizer, RobertaModel
from sklearn.metrics import matthews_corrcoef, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm, trange
from dataloader import meme_dataset
import json
import warnings
import sys
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore")




# TOKENIZER = BertTokenizer.from_pretrained('bert-base-uncased')
TOKENIZER = RobertaTokenizer.from_pretrained('roberta-base')
DROPOUT = 0.2
HIDDEN_SIZE = 128
BATCH_SIZE = 8
NUM_LABELS = 2
NUM_EPOCHS = 30 #50
EARLY_STOPPING = {"patience": 30, "delta": 0.005}
LEARNING_RATES = [0.0001, 0.001, 0.01, 0.1]
WEIGHT_DECAY = 0.1 
WARMUP = 0.06 
INPUT_LEN = 768
VIS_OUT = 2048
# VIS_OUT = 1280
criterion = nn.CrossEntropyLoss().cuda()



class CNN_roberta_Classifier(nn.Module):
    def __init__(self, vis_out, input_len, dropout, hidden_size, num_labels):
        super(CNN_roberta_Classifier,self).__init__()
        self.lm = RobertaModel.from_pretrained('roberta-base')
#         self.lm = BertModel.from_pretrained('bert-base-uncased')
        self.vm = models.resnet50(pretrained=True)
        self.vm.fc = nn.Sequential(nn.Linear(vis_out,input_len))
#         self.vm = models.efficientnet_b5(pretrained=True)
#         self.vmlp = nn.Linear(vis_out,input_len)
#         print(self.vm)
        
        embed_dim = input_len
        self.merge = torch.nn.Sequential(torch.nn.ReLU(),
            torch.nn.Linear(2 * embed_dim, 2 * embed_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(2 * embed_dim, embed_dim))
        self.mlp =  nn.Sequential(nn.Linear(input_len, hidden_size),
                                      nn.ReLU(),
                                      nn.Linear(hidden_size, hidden_size),
                                      nn.ReLU(),
                                      nn.Linear(hidden_size, num_labels))
        self.image_space = nn.Sequential(nn.Linear(input_len, input_len),
                                      nn.ReLU(),
                                      nn.Linear(input_len, input_len),
                                      nn.ReLU(),
                                      nn.Linear(input_len, input_len))
        self.text_space = nn.Sequential(nn.Linear(input_len, input_len),
                                      nn.ReLU(),
                                      nn.Linear(input_len, input_len),
                                      nn.ReLU(),
                                      nn.Linear(input_len, input_len))
        

        
    def forward(self, image, text, label):
#         img_cls, image_prev = self.vm(image)
#         image = self.vmlp(image_prev)
        image = self.vm(image)
        text = self.lm(**text).last_hidden_state[:,0,:]
        image_shifted = image
        text_shifted = text
        img_txt = (image,text)
        img_txt = torch.cat(img_txt, dim=1)
        merged = self.merge(img_txt)
        label_output = self.mlp(merged)
        return label_output, merged, image_shifted, text_shifted
    


def validation(dl,model):
    fin_targets=[]
    fin_outputs=[]
    single_labels = []
    img_names = []
    for i, data in enumerate(dl):
        data['image'] = data['image'].cuda()
        for key in data['text'].keys():
            data['text'][key] = data['text'][key].squeeze().cuda()
        data['slabel'] = data['slabel'].cuda()
        with torch.no_grad():
            predictions, merged, _ , _ = model(data['image'],data['text'], data['slabel'])
            predictions_softmax = nn.Softmax(dim=1)(predictions)
#             print(predictions_softmax,data['slabel'])
            outputs = predictions.argmax(1, keepdim=True).float()
            fin_targets.extend(data['slabel'].squeeze().cpu().detach().numpy().tolist())
            fin_outputs.extend(outputs.cpu().detach().numpy().tolist())
            single_labels.extend(data['slabel'])
            img_names.extend(data['img_info'])
    return fin_targets, fin_outputs, single_labels, img_names



def train(train_dataloader, dev_dataloader, model, optimizer, scheduler, dataset_name, shot, randomNum):
    max_acc = 0
    for epoch in range(NUM_EPOCHS):
        model.train()
        with tqdm(train_dataloader, unit="batch") as tepoch:
            train_total_correct = 0
            train_num_correct = 0
            train_loss_values, val_loss_values, train_acc_values, val_acc_values = [], [], [], []
            train_preds = []
            for data in tepoch:
                tepoch.set_description(f"Epoch {epoch}")
                data['image'] = data['image'].cuda()
                for key in data['text'].keys():
#                     print(data['text'][key].shape)
                    data['text'][key] = data['text'][key].squeeze(dim=1).cuda()
                data['slabel'] = data['slabel'].cuda()
#                 print(data['text'].shape)
                output, merged, image_shifted, text_shifted = model(data['image'],data['text'], data['slabel'])
                pred = output.argmax(1, keepdim=True).float()
                loss = criterion(output, data['slabel'])
                train_loss_values.append(loss)
                loss.backward() 
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                train_total_correct += data['image'].shape[0]
                pred = output.argmax(1, keepdim=True).float()
                tepoch.set_postfix(loss=loss.item())
        print ("loss ",sum(train_loss_values)/len(train_loss_values))
        model.eval()
        targets, outputs, slabels, img_names = validation(dev_dataloader,model)
        accuracy = accuracy_score(targets, outputs)
        f1_score_micro = f1_score(targets, outputs, average='micro')
        f1_score_macro = f1_score(targets, outputs, average='macro')
#         print(f"Accuracy Score = {accuracy}")
#         print(f"F1 Score (Micro) = {f1_score_micro}")
#         print(f"F1 Score (Macro) = {f1_score_macro}")
        
        if f1_score_macro > max_acc:
            max_acc = f1_score_macro
            print ("new best saving, ",max_acc)
            torch.save(model.state_dict(),'saved_m/'+dataset_name+str(shot)+'_'+str(randomNum)+'.pth')
            
        print ("Best so far, ",max_acc)

            
            
def write_test_results(outputs, image_names):
    dict_single = {}
    for i in range (len(image_names)):
        image_name = image_names[i]
        pred = str(int(outputs[i][0]))
        dict_single[image_name] = pred
    dict_single = collections.OrderedDict(sorted(dict_single.items()))
    json_object = json.dumps(dict_single, indent = 4)
    json_file_name = 'preds/' + EXP_NAME  + '.json'
    with open(json_file_name, "w") as outfile:
        outfile.write(json_object)

        
def get_torch_dataloaders(dataset_name,shot, randomNum, global_path):
    train_dataset = meme_dataset(dataset_name, 'train_'+str(shot)+'_'+str(randomNum), TOKENIZER,None,None)
    dev_dataset = meme_dataset(dataset_name, 'val', TOKENIZER,None,None)
    test_dataset = meme_dataset(dataset_name, 'test', TOKENIZER,None,None)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)
    dev_dataloader = DataLoader(dev_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)
    return train_dataloader, dev_dataloader, test_dataloader
        

def main():
    dataset_name = sys.argv[1]
    shot = sys.argv[2]
    randomNum = sys.argv[3]
    global_path = '../datasets'
    train_dataloader, dev_dataloader, test_dataloader = get_torch_dataloaders(dataset_name,shot, randomNum, global_path)
    model = CNN_roberta_Classifier(VIS_OUT, INPUT_LEN, DROPOUT, HIDDEN_SIZE,NUM_LABELS).cuda()
    optimizer = optim.AdamW(model.parameters(), lr = 1e-5, weight_decay = 0.1) 
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 0.06 * (len(train_dataloader)*BATCH_SIZE * NUM_EPOCHS), num_training_steps = (1-0.06) * (len(train_dataloader)*BATCH_SIZE * NUM_EPOCHS))
    print ("Starting training on, ", dataset_name, len(train_dataloader), len(dev_dataloader), len(test_dataloader))
    train(train_dataloader, dev_dataloader, model, optimizer, scheduler, dataset_name, shot, randomNum)
    model.load_state_dict(torch.load('saved_m/'+dataset_name+str(shot)+'_'+str(randomNum)+'.pth'))
    model.eval()
    targets, outputs, slabels, img_names = validation(test_dataloader,model)
    f1_score_macro = f1_score(targets, outputs, average='macro')
    accuracy = accuracy_score(targets, outputs)
    print('==================\n')
    print(dataset_name)
    print(shot)
    print(randomNum)
    print(accuracy,f1_score_macro)
    print('==================\n')
    #print ("Final F1 score on test set: ", dataset_name, f1_score_macro)
    #print("Final Accuracy on test set: ", dataset_name, accuracy)

if __name__ == "__main__":
    main()









