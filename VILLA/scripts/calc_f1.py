#! usr/bin/env python
# -*- codding : utf-8 -*-


import sys
import os
import pandas as pd
from sklearn.metrics import f1_score
import glob
import jsonlines


def calc_f1(gold_file, pred_files):
    f1_scores = {}
    for pdfile in pred_files:
        y_pred = []
        y_true = []
        predfile_df = pd.read_csv(pdfile)
        with jsonlines.open(gold_file) as gold_reader:
            for i, inst in enumerate(gold_reader):
                #print(inst)
               # print(predfile_df.loc[(predfile_df['id'] == inst['id'])])
                y_true.append(int(inst['label']))
                y_pred.append(int(predfile_df.loc[(predfile_df['id'] == inst['id']).idxmax(), 'label']))
            #y_pred_new = [1 if x == 0 else 0 for x in y_pred]
            print(y_true)
            print(y_pred)
            f1_scores[os.path.basename(pdfile).split('.')[0]] = f1_score(y_true, y_pred, average='macro')
            del predfile_df
    return f1_scores


def main():
    dataset = sys.argv[1]
    gold_file = os.path.join(dataset, 'gold_val.jsonl')
    pred_files = []
    for fil in glob.glob(os.path.join(dataset, '*.csv')):
        pred_files.append(fil)
    print(calc_f1(gold_file, pred_files))


if __name__ == '__main__':
    main()

