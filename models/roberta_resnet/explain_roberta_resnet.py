#! /usr/bin/env python
# -*- coding : utf-8 -*-


import lime
import sklearn
import sklearn.ensemble
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
import sklearn.metrics
from explain_dataloader import meme_dataset
import sys
from lime import lime_text
from lime.lime_text import LimeTextExplainer
from lime.lime_meme import LimeMemeExplainer
from sklearn.pipeline import make_pipeline
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
import json
import torch
from torchvision import models, transforms
from torch.autograd import Variable
import easyocr
from skimage.segmentation import mark_boundaries

TOKENIZER = RobertaTokenizer.from_pretrained('roberta-base')
DROPOUT = 0.2
HIDDEN_SIZE = 128
BATCH_SIZE = 12
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

    def forward(self, image, text):
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


def get_torch_dataloaders(image_path, text):
    test_dataset = meme_dataset(image_path, text, TOKENIZER)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)
    return test_dataloader

# define softmax function
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

# define prediction function
def predict_probs(meme_pixels, text):
   # meme = meme_pixels.save('../../datasets/harmeme/covid_memes_5425.png', 'PNG')
    model = CNN_roberta_Classifier(VIS_OUT, INPUT_LEN, DROPOUT, HIDDEN_SIZE, NUM_LABELS).cuda()
    model.load_state_dict(torch.load('saved/harmeme.pth'))
    model.eval()
    test_dataloader = get_torch_dataloaders(meme_pixels, text)
    for i, data in enumerate(test_dataloader):
        data['image'] = data['image'].cuda()
        for key in data['text'].keys():
            #print(torch.squeeze(data['text'][key], 1).cuda())
            data['text'][key] = torch.squeeze(data['text'][key], 1).cuda()
        #print(data['text'])
        with torch.no_grad():
            predictions, _, _ , _ = model(data['image'],data['text']) 
    #predictions,_,_,_ = model(texts)
    #print(predictions)
    #x = np.array(list(predictions.cpu()))
    #return np.apply_along_axis(softmax, 1, x)
    #print(x)
    #return x
    return F.softmax(predictions.cpu()).detach().numpy()

def get_pil_transform():
    transf = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.CenterCrop(448)
    ])

    return transf

def get_preprocess_transform():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    transf = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    return transf

def get_image(path):
    with open(os.path.abspath(path), 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def main():
    #class_names = [0, 1]
    #global_path = '../datasets'
    #dataset_name = sys.argv[1]
    #model = CNN_roberta_Classifier(VIS_OUT, INPUT_LEN, DROPOUT, HIDDEN_SIZE, NUM_LABELS).cuda()
    #model.load_state_dict(torch.load('saved/' + dataset_name + '.pth'))
    #model.eval()
    #sub_data = dataset_name
    #texta = 'ocr'
    #imga = 'ocr'
    #test_dataloader = get_torch_dataloaders(sub_data, global_path, imga, texta)
    #c = make_pipeline(test_dataloader, model)
    explainer = LimeMemeExplainer()
    
    #idx = 83
    #for step, data in enumerate(test_dataloader):
    '''
    texts =  {
    "img": "37405.png",
    "label": 1,
    "text": "introducing fidget spinner for women",
    "text_ocr": "introducing fidget spinner for women",
    "text_spread_1": "introducing fidget spinner for women",
    "text_spread_3": "introducing fidget spinner for women",
    "text_newsprint": "introducing fidget spinner for women",
    "text_s&p": "introducing fidget spinnerfor women",
    "text_s&p0.4": "introducing fidget spinnerfor women",
    "text_blur_text_5": "introducing fidget spinner for women.",
    "text_s&p_text_0.2": "introducing fidget spinner for women",
    "text_with_sp_5px": "introducing fidget spinner for women",
    "text_without_sp_5px": "introducing LIPI LIPI fidget spinner for women"
    }
    
    texts =  {
    "img": "covid_memes_5433.png",
    "label": 1,
    "text": "WHEN YOU'RE THE DOCTOR BUT\nHC\nEWS\nTHE VILLAGE IDIOT IS GIVING\nMEDICAL ADVICE\nmakeaneme.og\n",
    "text_ocr": "HC EN WHEN YOU'RE THE DOCTOR BUT... THE VILLAGE IDIOT IS GIVING MEDICAL ADVICE makeaneme.org",
    "text_spread_1": "WHEN YOU'RE THE DOCTOR BUT HC THE VILLAGE IDIOT IS GIVING MEDICAL ADVICE \"ONENTY AND THE",
    "text_spread_3": "WHICONTHE WALL",
    "text_newsprint": "WHEN YOU'RE THE DOCTOR BUT THE VILLAGE IDIOT IS GIVING MEDICAL ADVICE",
    "text_s&p": "WHEN YOU'RE THE DOCTOR BUT THE VILLAGE IDIOT IS GIVING MEDICAL ADVICE m",
    "text_s&p0.4": "WHEN YOU'RE THE DOCTOR BUT THOUILLAGE (HOT IS GIVING 2985 ke",
    "text_blur_text_5": "WHEN YOU'RE THE DOCTOR BU HC EVILLAGE IDIOT IS GIVING MEDICAL ADVICE EN WS makeaneme.org",
    "text_s&p_text_0.2": "\u041d\u0421 WHEN YOU'RE THE DOCTOR BUC A THE VILLAGE IDIOT IS GIVING MEDICAL ADVICE makeaneme.org",
    "text_with_sp_5px": "WHEN YOU'RE THE DOCTOR BUT THE VILLAGE IDIOT IS GIVING MEDICAL ADVICE m",
    "text_without_sp_5px": "HC EN WHEN YOU'RE THE DOCTOR BUT... LOVE THE VILLAGE IDIOT IS GIVING MEDICAL ADVICE makeaneme.org"
    }
    '''
    texts =  {
  "img": "covid_memes_5427.png",
  "label": 1,
  "text": "CRUX.\nCROANINGO\n\"It's irresponsible and it's dangerous': Experts\nrip Trump's idea of injecting disinfectant to\ntreat COVID-19\nMedical experts denounce Trump's\nlatest 'dangerous' suggestion to treat\nCovid-19\n"
,
  "text_ocr": "CRUX GROANING 'It's irresponsible and it's dangerous': Experts rip Trump's idea of injecting disinfectant to treat COVID-19 Medical experts denounce Trump's latest 'dangerous' suggestion to treat Covid-19",
  "text_spread_1": "CRUX GROANING 'It's irresponsible and it's dangerous': Experts rip Trump's idea of injecting disinfectant to treat COVID-19 Medical experts denounce Trump's latest 'dangerous' suggestion to treat Covid-19",
  "text_spread_3": "CRUX GROANING 'It's irresponsible and it's dangerous': Experts rip Trump's idea of injecting disinfectant to treat COVID-19 Medical experts denounce Trump's latest 'dangerous' suggestion to treat Covid-19",
  "text_newsprint": "CRUX. - CROANING 'It's irresponsible and it's dangerous': Experts rip Trump's idea of injecting disinfectant to treat COVID-19 Medical experts denounce Trump's latest dangerous' suggestion to treat Covid-19",
  "text_s&p": "CRUX. CROANING 'It's irresponsible and it's dangerous': Experts rip Trump's idea of injecting disinfectant to treat COVID-19 Medical experts denounce Trump's latest 'dangerous' suggestion to treat Covid-19",
  "text_s&p0.4": "O GROANING 'It's irresponsible and it's dangerous': Experts rip Trump's idea of injecting disinfectant to treat COVID-19 Medical experts denounce Trump's latest dangerous' suggestion to treat Covid-19 CRUX.",
  "text_blur_text_5": "CRUX GROANING 'It's irresponsible and it's dangerous': Experts rip Trump's idea of injecting disinfectant to treat COVID-19 Medical experts denounce Trump's latest 'dangerous' suggestion to treat Covid-19",
  "text_s&p_text_0.2": "CRUX. SKA GROANINGD 'It's irresponsible and it's dangerous': Experts rip Trump's idea of injecting disinfectant to treat COVID-19 Medical experts denounce Trump's latest dangerous' suggestion to treat Covid-19",
  "text_with_sp_5px": "CRUX. CROANING 'It's irresponsible and it's dangerous': Experts rip Trump's idea of injecting disinfectant to treat COVID-19 Medical experts denounce Trump's latest 'dangerous' suggestion to treat Covid-19",
  "text_without_sp_5px": "CRUX GROANING 'It's irresponsible and it's dangerous': Experts rip Trump's idea of injecting disinfectant to treat COVID-19 Medical experts denounce Trump's latest 'dangerous' suggestion to treat Covid-19"
 }
    #images = ['../../datasets/fb/test_imgs/37405.png', '../../datasets/fb/s&p0.4/37405.png', '../../datasets/fb/newsprint/37405.png', '../../datasets/fb/s&p_text_0.2/37405.png', '../../datasets/fb/spread_1/37405.png']
    images = ['../../datasets/harmeme/test_imgs/covid_memes_5427.png', '../../datasets/harmeme/s&p0.4/covid_memes_5427.png', '../../datasets/harmeme/newsprint/covid_memes_5427.png', '../../datasets/harmeme/s&p_text_0.2/covid_memes_5427.png', '../../datasets/harmeme/spread_1/covid_memes_5427.png']
    for image in images:
        image_type = os.path.basename(os.path.dirname(image))
        sample_image = get_image(image)
        pill_transf = get_pil_transform()
        preprocess_transform = get_preprocess_transform()
        ocr = easyocr.Reader(['en'])
        #text_string = ' '.join(ocr.readtext(image, detail = 0))
        if image_type == 'test_imgs':
            text_string = texts['text']
        else:
            text_string = texts['text_' + str(image_type)]
        exp = explainer.explain_instance(np.array(pill_transf(sample_image)),text_string, 
                                         predict_probs, # classification function
                                         top_labels=2, 
                                         hide_color=0, 
                                         num_samples=1000)
        
        temp, mask = exp.get_image_and_mask(exp.top_labels[1], positive_only=False, num_features=10, hide_rest=False)
        img_boundry2 = mark_boundaries(temp/255.0, mask)
        plt.imsave(str(image_type) + '.png', img_boundry2)
        print(str(image_type))
        print('\n====================\n')
        print(exp.as_list())
    #break

if __name__ == '__main__':
    main()





