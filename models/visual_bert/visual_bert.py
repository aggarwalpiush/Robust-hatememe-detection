import collections
import os

from param import args

import numpy as np

from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
import pandas as pd
from dataloader import HMTorchDataset, HMEvaluator, HMDataset
from optimization import AdamW, get_linear_schedule_with_warmup
# from utils.pandas_scripts import clean_data
from dataloader import load_obj_tsv
from entryV import ModelV
import json
# Two different SWA Methods - https://pytorch.org/blog/pytorch-1.6-now-includes-stochastic-weight-averaging/
if args.swa:
    from torch.optim.swa_utils import AveragedModel, SWALR
    from torch.optim.lr_scheduler import CosineAnnealingLR

if args.contrib:
    from torchcontrib.optim import SWA


# Largely sticking to standards set in LXMERT here
DataTuple = collections.namedtuple("DataTuple", 'dataset loader evaluator')

def get_tuple(dataset_name, split, bs:int, shuffle=False, drop_last=False) -> DataTuple:

#     dset =  HMDataset(splits)
#     adv_attack = None
    tset = HMTorchDataset(dataset_name, split, None,None)
    evaluator = HMEvaluator(tset)
    data_loader = DataLoader(
        tset, batch_size=bs,
        shuffle=shuffle, num_workers=args.num_workers,
        drop_last=drop_last, pin_memory=True
    )

    return DataTuple(dataset=tset, loader=data_loader, evaluator=evaluator)

class HM:
    def __init__(self,dataset):
        self.dataset_name = dataset
        self.train_tuple = get_tuple(self.dataset_name,'train', bs=args.batch_size, shuffle=True, drop_last=False)
        valid_bsize = 2048 if args.multiGPU else 50
        self.valid_tuple = get_tuple(self.dataset_name, 'val', bs=valid_bsize,shuffle=False, drop_last=False)
        self.test_tuple = get_tuple(self.dataset_name,'test', bs=valid_bsize,shuffle=False, drop_last=False)
        # Select Model, X is default
        self.model = ModelV(args)
        print ("out of init of model")
        # Load pre-trained weights from paths
        if args.loadpre is not None:
            self.model.load(args.loadpre)
        
        # GPU options
        if args.multiGPU:
            self.model.lxrt_encoder.multi_gpu()

        self.model = self.model.cuda()

        # Losses and optimizer
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.nllloss = nn.NLLLoss()

#         if args.train is not None:
        batch_per_epoch = len(self.train_tuple.loader)
        self.t_total = int(batch_per_epoch * args.epochs // args.acc)
        print("Total Iters: %d" % self.t_total)

        def is_backbone(n):
            if "encoder" in n:
                return True
            elif "embeddings" in n:
                return True
            elif "pooler" in n:
                return True
            print("F: ", n)
            return False

        no_decay = ['bias', 'LayerNorm.weight']

        params = list(self.model.named_parameters())
        if args.reg:
            optimizer_grouped_parameters = [
                {"params": [p for n, p in params if is_backbone(n)], "lr": args.lr},
                {"params": [p for n, p in params if not is_backbone(n)], "lr": args.lr * 500},
            ]

            for n, p in self.model.named_parameters():
                print(n)

            self.optim = AdamW(optimizer_grouped_parameters, lr=args.lr)
        else:
            optimizer_grouped_parameters = [
                {'params': [p for n, p in params if not any(nd in n for nd in no_decay)], 'weight_decay': args.wd},
                {'params': [p for n, p in params if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
                ]

            self.optim = AdamW(optimizer_grouped_parameters, lr=args.lr)

#         if args.train is not None:
        self.scheduler = get_linear_schedule_with_warmup(self.optim, self.t_total * 0.1, self.t_total)
        
        self.output = args.output
        os.makedirs(self.output, exist_ok=True)

        # SWA Method:
        if args.contrib:
            self.optim = SWA(self.optim, swa_start=self.t_total * 0.75, swa_freq=5, swa_lr=args.lr)

        if args.swa: 
            self.swa_model = AveragedModel(self.model)
            self.swa_start = self.t_total * 0.75
            self.swa_scheduler = SWALR(self.optim, swa_lr=args.lr)

    def train(self, train_tuple, eval_tuple):

        dset, loader, evaluator = train_tuple
        iter_wrapper = (lambda x: tqdm(x, total=len(loader))) if args.tqdm else (lambda x: x)

        print("Batches:", len(loader))

        self.optim.zero_grad()

        best_roc = 0.
        ups = 0
        
        total_loss = 0.

        for epoch in range(args.epochs):
            if args.reg:
                if args.model != "X":
                    print(self.model.model.layer_weights)

            id2ans = {}
            id2prob = {}
            print ("training for epoch ",epoch)
            for i, (ids, feats, boxes, sent, target) in iter_wrapper(enumerate(loader)):
#                 print(i,len(loader))
                if ups == args.midsave:
                    self.save("MID")

                self.model.train()

                if args.swa:
                    self.swa_model.train()
                
                feats, boxes, target = feats.cuda(), boxes.cuda(), target.long().cuda()

                # Model expects visual feats as tuple of feats & boxes
                logit = self.model(sent, (feats, boxes))
#                 print ("logit computed")

                # Note: LogSoftmax does not change order, hence there should be nothing wrong with taking it as our prediction 
                # In fact ROC AUC stays the exact same for logsoftmax / normal softmax, but logsoftmax is better for loss calculation
                # due to stronger penalization & decomplexifying properties (log(a/b) = log(a) - log(b))
                logit = self.logsoftmax(logit)
                score = logit[:, 1]

#                 if i < 1:
#                     print(logit[0, :].detach())
               
                # Note: This loss is the same as CrossEntropy (We splitted it up in logsoftmax & neg. log likelihood loss)
                loss = self.nllloss(logit.view(-1, 2), target.view(-1))

                # Scaling loss by batch size, as we have batches with different sizes, since we do not "drop_last" & dividing by acc for accumulation
                # Not scaling the loss will worsen performance by ~2abs%
                loss = loss * logit.size(0) / args.acc
                loss.backward()

                total_loss += loss.detach().item()

                # Acts as argmax - extracting the higher score & the corresponding index (0 or 1)
                _, predict = logit.detach().max(1)
                # Getting labels for accuracy
                for qid, l in zip(ids, predict.cpu().numpy()):
                    id2ans[qid] = l
                # Getting probabilities for Roc auc
                for qid, l in zip(ids, score.detach().cpu().numpy()):
                    id2prob[qid] = l

                if (i+1) % args.acc == 0:

                    nn.utils.clip_grad_norm_(self.model.parameters(), args.clip)

                    self.optim.step()

                    if (args.swa) and (ups > self.swa_start):
                        self.swa_model.update_parameters(self.model)
                        self.swa_scheduler.step()
                    else:
                        self.scheduler.step()
                    self.optim.zero_grad()

                    ups += 1

                    # Do Validation in between
                    if ups % 500 == 0: 
                        print(epoch, ups)
                        
            log_str = "\nEpoch(U) %d(%d): Train AC %0.2f RA %0.4f LOSS %0.4f\n" % (epoch, ups, evaluator.evaluate(id2ans)*100, 
            evaluator.roc_auc(id2ans)*100, total_loss)

            # Set loss back to 0 after printing it
            total_loss = 0.
            if self.valid_tuple is not None:  # Do Validation
                acc, roc_auc = self.evaluate(eval_tuple)
                if roc_auc > best_roc:
                    best_roc = roc_auc
                    best_acc = acc
                    self.save(self.dataset_name)
                    # Only save BEST when no midsave is specified to save space
                    #if args.midsave < 0:
                    #    self.save("BEST")

                log_str += "\nEpoch(U) %d(%d): DEV AC %0.2f RA %0.4f \n" % (epoch, ups, acc*100.,roc_auc*100)
                log_str += "Epoch(U) %d(%d): BEST AC %0.2f RA %0.4f \n" % (epoch, ups, best_acc*100., best_roc*100.)

            print(log_str, end='')

            with open(self.output + "/log.log", 'a') as f:
                f.write(log_str)
                f.flush()

            if (epoch + 1) == args.epochs:
                if args.contrib:
                    self.optim.swap_swa_sgd()


    def predict(self, eval_tuple: DataTuple, dump=None, out_csv=True):

        dset, loader, evaluator = eval_tuple
        id2ans = {}
        id2prob = {}

        for i, datum_tuple in enumerate(loader):

            ids, feats, boxes, sent = datum_tuple[:4]

            self.model.eval()

            if args.swa:
                self.swa_model.eval()

            with torch.no_grad():
                
                feats, boxes = feats.cuda(), boxes.cuda()
                logit = self.model(sent, (feats, boxes))

                # Note: LogSoftmax does not change order, hence there should be nothing wrong with taking it as our prediction
                logit = self.logsoftmax(logit)
                score = logit[:, 1]

                if args.swa:
                    logit = self.swa_model(sent, (feats, boxes))
                    logit = self.logsoftmax(logit)

                _, predict = logit.max(1)

                for qid, l in zip(ids, predict.cpu().numpy()):
                    id2ans[qid] = l

                # Getting probas for Roc Auc
                for qid, l in zip(ids, score.cpu().numpy()):
                    id2prob[qid] = l

        if dump is not None:
            if out_csv == True:
                evaluator.dump_csv(id2ans, id2prob, dump)
            else:
                evaluator.dump_result(id2ans, dump)

        return id2ans, id2prob

    def evaluate(self, eval_tuple: DataTuple, dump=None):
        """Evaluate all data in data_tuple."""
        id2ans, id2prob = self.predict(eval_tuple, dump=dump)

        acc = eval_tuple.evaluator.evaluate(id2ans)
        roc_auc = eval_tuple.evaluator.roc_auc(id2ans)

        return acc, roc_auc

    def save(self, name):
        if args.swa:
            torch.save(self.swa_model.state_dict(),
                    os.path.join(self.output, "%s.pth" % name))
        else:
            torch.save(self.model.state_dict(),
                    os.path.join("saved/", "%s.pth" % name))

    def load(self, path):
        print("Load model from %s" % path)
            
        state_dict = torch.load("%s" % path)
        new_state_dict = {}
        for key, value in state_dict.items():
            # N_averaged is a key in SWA models we cannot load, so we skip it
            if key.startswith("n_averaged"):
                print("n_averaged:", value)
                continue
            # SWA Models will start with module
            if key.startswith("module."):
                new_state_dict[key[len("module."):]] = value
            else:
                new_state_dict[key] = value
        state_dict = new_state_dict
        self.model.load_state_dict(state_dict)

        
        
def write_test_results(img2id, model_name):
    dict_single = {}
    for x in img2id.keys():
        dict_single[str(x.item())+'.jpg'] = str(img2id[x])
    dict_single = collections.OrderedDict(sorted(dict_single.items()))
    json_object = json.dumps(dict_single, indent = 4)
    json_file_name = 'preds/' + model_name + '.json'
    with open(json_file_name, "w") as outfile:
        outfile.write(json_object)        
        

def main():
    # Build Class
    datasets = ['fb','mami','harmeme']
    for dataset in datasets:
        hm = HM(dataset)
        print ("init complete")
        hm.train(hm.train_tuple, hm.valid_tuple)
        hm.load(os.path.join('saved',hm.dataset_name+'.pth'))
        acc, f1 = hm.evaluate(hm.test_tuple)
        print (acc,f1)


if __name__ == "__main__":

    # Create pretrain.jsonl & traindev data
#     clean_data("./data")

    main()

