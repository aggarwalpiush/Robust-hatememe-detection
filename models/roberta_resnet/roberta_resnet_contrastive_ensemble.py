import pandas as pd
import os
import numpy as np
import torch
import random
import functools
import operator
import cv2
import collections
import codecs
import torchvision.models as models
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader, Dataset
from transformers import get_linear_schedule_with_warmup, BertTokenizer, BertModel, VisualBertModel, VisualBertConfig, RobertaTokenizer, RobertaModel
from sklearn.metrics import matthews_corrcoef, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm, trange
from dataloader import meme_dataset
import json
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore")
import torch.nn.functional as F



# TOKENIZER = BertTokenizer.from_pretrained('bert-base-uncased')
TOKENIZER = RobertaTokenizer.from_pretrained('roberta-base')
DROPOUT = 0.2
HIDDEN_SIZE = 128
BATCH_SIZE = 4
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




class ContrastiveLoss(nn.Module):
    def __init__(self, batch_size, temperature=0.5):
        super().__init__()
        self.batch_size = batch_size
        self.register_buffer("temperature", torch.tensor(temperature))
        self.register_buffer("negatives_mask", (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool)).float())
            
    def forward(self, emb_i, emb_j):
        """
        emb_i and emb_j are batches of embeddings, where corresponding indices are pairs
        z_i, z_j as per SimCLR paper
        """
        print(emb_i.shape)
        print(emb_j.shape)
        z_i = F.normalize(emb_i, dim=1)
        z_j = F.normalize(emb_j, dim=1)
        bs = z_i.shape[0]
        representations = torch.cat([z_i, z_j], dim=0)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)
        
        sim_ij = torch.diag(similarity_matrix, bs)
        sim_ji = torch.diag(similarity_matrix, -bs)
        positives = torch.cat([sim_ij, sim_ji], dim=0)
        nm = (~torch.eye(bs * 2, bs * 2, dtype=bool)).cuda().float()
        nominator = torch.exp(positives / self.temperature)
        denominator = nm * torch.exp(similarity_matrix / self.temperature)
    
        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))
        loss = torch.sum(loss_partial) / (2 * bs)
        return loss

contrastive_loss = ContrastiveLoss(BATCH_SIZE).cuda()

class CNN_roberta_Classifier(nn.Module):
    def __init__(self, vis_out, input_len, dropout, hidden_size, num_labels):
        super(CNN_roberta_Classifier,self).__init__()
        self.lm = RobertaModel.from_pretrained('roberta-base')
        self.vm = models.resnet50(pretrained=True)
        self.vm.fc = nn.Sequential(nn.Linear(vis_out,input_len))
        
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
        

        
    def forward(self, image, text, a_image, a_text, label, train):
        image = self.vm(image)
#         print(text.shape)
#         print(self.lm(**text).last_hidden_state.shape)'=
        for key in text.keys():
            if len(text[key].shape)==1:
                text[key] = text[key].unsqueeze(dim=0)
        text = self.lm(**text).last_hidden_state[:,0,:]
        if train:
            a_image = self.vm(a_image)
            a_text = self.lm(**a_text).last_hidden_state[:,0,:]
        img_txt = (image,text)
        img_txt = torch.cat(img_txt, dim=1)
        merged = self.merge(img_txt)
        if train:
            a_img_txt = (a_image,a_text)
            a_img_txt = torch.cat(a_img_txt, dim=1)
            a_merged = self.merge(a_img_txt)
        else:
            a_merged = merged
        label_output = self.mlp(merged)
        return label_output, merged, a_merged
    


def validation(dl,model):
    fin_targets=[]
    fin_outputs=[]
    probab_output=[]
    single_labels = []
    img_names = []
    for i, data in enumerate(dl):
        data['image'] = data['image'].cuda()
        for key in data['text'].keys():
            data['text'][key] = data['text'][key].squeeze().cuda()
        data['slabel'] = data['slabel'].cuda()
        with torch.no_grad():
            predictions, merged, _  = model(data['image'],
                                            data['text'], 
                                            data['image'],
                                            data['text'],
                                            data['slabel'],False)
            predictions_softmax = nn.Softmax(dim=1)(predictions)
            outputs = predictions.argmax(1, keepdim=True).float()
#             if data['slabel'].shape[0] == 1:
#                 data['slabel'] = data['slabel'].unsqueeze(0)
#             if 
            probab_output.extend(predictions_softmax.cpu().detach().numpy().tolist())
            fin_targets.extend(data['slabel'].cpu().detach().numpy().tolist())
            fin_outputs.extend(outputs.cpu().detach().numpy().tolist())
            single_labels.extend(data['slabel'])
            img_names.extend(data['img_info'])
    print(probab_output)
    print(fin_targets)
    print(fin_outputs)
    return fin_targets, probab_output, single_labels, img_names






def train(train_dataloader, dev_dataloader, model, optimizer, scheduler, dataset_name):
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
                data['augmented_image'] = data['augmented_image'].cuda()
                for key in data['text'].keys():
                    data['text'][key] = data['text'][key].squeeze(dim=1).cuda()
                for key in data['augmented_text'].keys():
                    data['augmented_text'][key] = data['augmented_text'][key].squeeze(dim=1).cuda()
                data['slabel'] = data['slabel'].cuda()
                output, merged, a_merged = model(data['image'],
                                                 data['text'], 
                                                 data['augmented_image'],
                                                 data['augmented_text'],
                                                 data['slabel'],True)
                pred = output.argmax(1, keepdim=True).float()
                classification_loss = criterion(output, data['slabel'])
                merged = merged.cuda()
                a_merged = a_merged.cuda()
                cont_loss =  contrastive_loss(merged,a_merged)
                loss = classification_loss + cont_loss
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
        if f1_score_macro > max_acc:
            max_acc = f1_score_macro
            print ("new best saving, ",max_acc)
            torch.save(model.state_dict(),'saved/'+dataset_name+'_contrastive_0.5'+'.pth')
            
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

'''        
def get_torch_dataloaders(dataset_name,global_path):
    train_dataset = meme_dataset(dataset_name, 'train', TOKENIZER,None,None)
    dev_dataset = meme_dataset(dataset_name, 'val', TOKENIZER,None,None)
    test_dataset = meme_dataset(dataset_name, 'test', TOKENIZER,None,None)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)
    dev_dataloader = DataLoader(dev_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)
    return train_dataloader, dev_dataloader, test_dataloader
'''     


def get_torch_dataloaders(dataset_name,global_path, imga, texta):
    test_dataset = meme_dataset(dataset_name, 'test', TOKENIZER, imga, texta)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)
    return test_dataloader

def main():
    global_path = '../datasets'
    datasets = ['harmeme', 'fb', 'mami']
    attacks = ['ocr','spread_1','spread_3','newsprint','s&p','s&p0.4','blur_text_5','s&p_text_0.2' , 'with_sp_5px', 'without_sp_5px']

#     datasets = ['mami']
    for dataset_name in datasets:
        for attack in attacks:
            texta = attack
            imga = attack
            test_dataloader = get_torch_dataloaders(dataset_name,global_path,imga,texta)
            #train_dataloader, dev_dataloader, test_dataloader = get_torch_dataloaders(dataset_name,global_path)
            model = CNN_roberta_Classifier(VIS_OUT, INPUT_LEN, DROPOUT, HIDDEN_SIZE,NUM_LABELS).cuda()
#           model.load_state_dict(torch.load('saved/'+dataset_name+'.pth'))
            #optimizer = optim.AdamW(model.parameters(), lr = 1e-5, weight_decay = 0.1) 
            #scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 0.06 * (len(train_dataloader)*BATCH_SIZE * NUM_EPOCHS), num_training_steps = (1-0.06) * (len(train_dataloader)*BATCH_SIZE * NUM_EPOCHS))
        #print ("Starting training on, ", dataset_name, len(train_dataloader), len(dev_dataloader), len(test_dataloader))
       # train(train_dataloader, dev_dataloader, model, optimizer, scheduler, dataset_name)
            model.load_state_dict(torch.load('saved/'+dataset_name+'_contrastive'+'.pth'))
            model.eval()
            targets, outputs, slabels, img_names = validation(test_dataloader,model)
            with codecs.open(str(dataset_name)+'_'+str(attack)+'_contrastive_0.5_errorAnalysis.tsv', 'w', 'utf-8') as ea_obj:
                for i, each_img in enumerate(img_names):
                    ea_obj.write("%s\t%s\t%s\n" %(each_img, str(targets[i]), str(outputs[i])))
            #f1_score_macro = f1_score(targets, outputs, average='macro')
            #accuracy = accuracy_score(targets, outputs)
            #print("Final F1 score on test set: ", dataset_name, f1_score_macro)
            #print("Final Accuracy on test set: ", dataset_name, accuracy)

if __name__ == "__main__":
    main()









