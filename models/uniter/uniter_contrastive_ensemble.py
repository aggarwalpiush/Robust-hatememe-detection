import collections
import os

from param import args

import numpy as np

from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
import pandas as pd
from dataloader_contrastive import HMTorchDataset, HMEvaluator, HMDataset
from optimization import AdamW, get_linear_schedule_with_warmup
# from utils.pandas_scripts import clean_data
from dataloader import load_obj_tsv
from entryU_contrastive import ModelU
import json
# Two different SWA Methods - https://pytorch.org/blog/pytorch-1.6-now-includes-stochastic-weight-averaging/
if args.swa:
    from torch.optim.swa_utils import AveragedModel, SWALR
    from torch.optim.lr_scheduler import CosineAnnealingLR

if args.contrib:
    from torchcontrib.optim import SWA
import torch.nn.functional as F


# Largely sticking to standards set in LXMERT here
DataTuple = collections.namedtuple("DataTuple", 'dataset loader evaluator')

def get_tuple(dataset_name, split,  img_adv_attack, text_adv_attack, bs:int, shuffle=False, drop_last=False) -> DataTuple:

#     dset =  HMDataset(splits)
#     adv_attack = None
    tset = HMTorchDataset(dataset_name, split,  img_adv_attack, text_adv_attack)
    evaluator = HMEvaluator(tset)
    data_loader = DataLoader(
        tset, batch_size=bs,
        shuffle=shuffle, num_workers=args.num_workers,
        drop_last=drop_last, pin_memory=True
    )

    return DataTuple(dataset=tset, loader=data_loader, evaluator=evaluator)


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
        #print(emb_i.shape)
        #print(emb_j.shape)
        emb_i = emb_i[:, -1, :]
        emb_j = emb_j[:, -1, :]
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


contrastive_loss = ContrastiveLoss(args.batch_size).cuda()



class HM:
    def __init__(self, img_adv_attack, text_adv_attack):
        self.dataset_name = 'harmeme'
        #self.train_tuple = get_tuple(self.dataset_name,'train', bs=args.batch_size, shuffle=True, drop_last=False)
        valid_bsize = 2
        #self.valid_tuple = get_tuple(self.dataset_name, 'val', bs=valid_bsize,shuffle=False, drop_last=False)
        self.test_tuple = get_tuple(self.dataset_name,'test', img_adv_attack, text_adv_attack, bs=valid_bsize, shuffle=False, drop_last=False)
        # Select Model, X is default
        self.model = ModelU(args)
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
        batch_per_epoch = len(self.test_tuple.loader)
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
            for i, (ids, feats, boxes, feat_aug, boxes_aug, sent, sent_aug, target) in iter_wrapper(enumerate(loader)):
#                 print(i,len(loader))
                if ups == args.midsave:
                    self.save("MID")

                self.model.train()

                if args.swa:
                    self.swa_model.train()
                
                feats, boxes, target = feats.cuda(), boxes.cuda(), target.long().cuda()
                feats_aug, boxes_aug = feat_aug.cuda(), boxes_aug.cuda()
                # Model expects visual feats as tuple of feats & boxes
                logit, seq_out, seq_out_aug = self.model(sent, (feats, boxes), sent_aug, (feats_aug, boxes_aug))
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


                merged = seq_out.cuda()
                a_merged = seq_out_aug.cuda()
                cont_loss = contrastive_loss(merged, a_merged)
                loss = loss + cont_loss
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
        id2probnorm = {}


        for i, datum_tuple in enumerate(loader):

            ids, feats, boxes, sent = datum_tuple[:4]
            ids, feats, boxes, feats_aug, boxes_aug, sent, sent_aug = datum_tuple[:7]

            self.model.eval()

            if args.swa:
                self.swa_model.eval()

            with torch.no_grad():
                
                feats, boxes = feats.cuda(), boxes.cuda()
                feats_aug, boxes_aug = feats_aug.cuda(), boxes_aug.cuda()
                logit,_,_ = self.model(sent, (feats, boxes), sent_aug, (feats_aug, boxes_aug))

                # Note: LogSoftmax does not change order, hence there should be nothing wrong with taking it as our prediction
                logit = self.logsoftmax(logit)
                score = logit[:, 1]
                score1 = logit[:, 1]
                score0 = logit[:, 0]
                #prob,_= logit.detach().max(1)
                #print(ids)
                #print(logit)
                if args.swa:
                    logit = self.swa_model(sent, (feats, boxes))
                    logit = self.logsoftmax(logit)

                _, predict = logit.max(1)
                #print(predict.cpu().numpy())

                for qid, l in zip(ids, predict.cpu().numpy()):
                    id2ans[qid] = l

                # Getting probas for Roc Auc
                for qid, l in zip(ids, score.cpu().numpy()):
                    id2prob[qid] = l

                for qid, l, m in zip(ids, score1.cpu().numpy(), score0.cpu().numpy()):
                    id2probnorm[qid] = l/(l+m)
        '''
        if dump is not None:
            if out_csv == True:
                evaluator.dump_csv(id2ans, id2prob, dump)
            else:
                evaluator.dump_result(id2ans, dump)
        '''
        return id2ans, id2prob, id2probnorm

    def evaluate(self, eval_tuple: DataTuple, dump=None, attack=None):
        """Evaluate all data in data_tuple."""
        id2ans, id2probi, id2probnorm  = self.predict(eval_tuple, dump=dump)

        acc = eval_tuple.evaluator.evaluate(id2ans)
        roc_auc = eval_tuple.evaluator.roc_auc(id2ans)
        eval_tuple.evaluator.dump_tsv(id2ans, id2probnorm, attack, self.dataset_name)

        return acc, roc_auc

    def save(self, name):
        if args.swa:
            torch.save(self.swa_model.state_dict(),
                    os.path.join(self.output, "%s_contrastive_0.5.pth" % name))
        else:
            torch.save(self.model.state_dict(),
                    os.path.join("saved/", "%s_contrastive_0.5.pth" % name))

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
    f = open('../../attack_results_uniter_contrastive.tsv', 'a')

    # Build Class
#     text_adv_attacks = [None,txtaugs.InsertPunctuationChars(),txtaugs.InsertZeroWidthChars(),txtaugs.ReplaceFunFonts(),
#                         txtaugs.ReplaceSimilarChars(),txtaugs.InsertWhitespaceChars(),txtaugs.SimulateTypos(),
#                         txtaugs.SwapGenderedWords(),txtaugs.InsertZeroWidthChars()]

#     text_string_attacks = ['No attack','InsertPunctuationChars','InsertZeroWidthChars','ReplaceFunFonts',
#                            'ReplaceSimilarChars','InsertWhitespaceChars', 'SimulateTypos',
#                            'SwapGenderedWords','InsertZeroWidthChars']

#     img_adv_attacks = [None,'overlay_stripes','random_10','random_1000','shuffle_1','shuffle_point1']
#     img_string_attacks = ['No attack','overlay_stripes','random_10','random_1000','shuffle_1','shuffle_point1']

    list_results = []
    adv_attacks1 = [None,'spread_1','spread_3','newsprint','s&p','s&p0.4']
    adv_attacks2 = ['blur_text_5','s&p_text_0.2']
    adv_attacks3 = [ 'with_sp_5px', 'without_sp_5px']
    for i,attack in enumerate(adv_attacks1 + adv_attacks2 + adv_attacks3):
        text_attack = attack
#         if attack == 's&p0.4':
#             text_attack = 's&p_0.4'
        hm = HM(attack,text_attack)
        print ("init complete")
        hm.load(os.path.join('saved',hm.dataset_name+'_contrastive_0.5.pth'))
        acc, f1 = hm.evaluate(hm.test_tuple, dump=None, attack=attack)
        list_results.append([attack,f1])
        
        #f.write(hm.dataset_name + '\t' + str(attack) + '\t' + '\t' + 'uniter_contrastive_0.5'  + '\t' + str(0.0) + '\t' + str(f1) + '\n')
        print(attack,f1)
    print(list_results)


if __name__ == "__main__":

    # Create pretrain.jsonl & traindev data
#     clean_data("./data")

    main()

