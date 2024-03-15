# BERT-SVM classifier for AMP prediction
# 10-fold cross validation training code modified by Ruihan Dong
# 24-03-14

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '7'
import warnings
warnings.filterwarnings("ignore")

import re
import numpy as np
import pandas as pd
from time import time
from tqdm.auto import tqdm
from thundersvm import SVC

from sklearn.model_selection import KFold
from sklearn.metrics import make_scorer, accuracy_score, precision_score, matthews_corrcoef, roc_auc_score, f1_score
from sklearn.utils import shuffle

import torch
from transformers import AutoTokenizer, AutoModel, pipeline


def show_metrics(y_true, y_pred):
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for i in range(len(y_true)):
        if y_true[i] == 1 and y_pred[i] == 1:
            tp = tp + 1
        if y_true[i] == 1 and y_pred[i] == 0:
            fn = fn + 1
        if y_true[i] == 0 and y_pred[i] == 1:
            fp = fp + 1
        if y_true[i] == 0 and y_pred[i] == 0:
            tn = tn + 1
    
    sensitivity = float(tp) / (float(tp) + float(fn))
    specificity = float(tn) / (float(tn) + float(fp))
    auroc = roc_auc_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    print('sensitivity {:.4f}, specificity {:.4f}, accuracy {:.4f}, precision {:.4f}, mcc {:.4f}, F1 {:.4f}, auroc {:.4f}'.format(sensitivity, specificity, accuracy, precision, mcc, f1, auroc))
    return [sensitivity, specificity, accuracy, precision, mcc, f1, auroc]


tokenizer = AutoTokenizer.from_pretrained("./prot_bert_bfd", do_lower_case=False)
model = AutoModel.from_pretrained("./prot_bert_bfd")
fe = pipeline('feature-extraction', model=model, tokenizer=tokenizer, device=0)


def BERT_Extracting_seq_features(seq):
    sequences_Example = [" ".join(seq)]
    sequences_Example = [re.sub(r"[UZOB]", "X", sequence) for sequence in sequences_Example]

    embedding = fe(sequences_Example)
    embedding = np.array(embedding)
    embedding = embedding.flatten()

    if len(embedding) < 43008:
        pad_0_number = 43008 - len(embedding)
        embedding = np.pad(embedding, (0, pad_0_number))
    return embedding


with open('./dataset/MCPSS_train.txt','r') as f:  # MCPSS_train.txt toy/positive.txt
    a=f.readlines()
    positive_set=[]
    for ittem in a:
        ittem=ittem.replace("\n", "")
        positive_set.append(BERT_Extracting_seq_features(ittem))

with open('./dataset/neg10_MCPSS_train.txt','r') as ff:  # neg10_MCPSS_train.txt toy/negative.txt
    b=ff.readlines()
    negative_set=[]
    for ittem in b:
        ittem=ittem.replace("\n", "")
        negative_set.append(BERT_Extracting_seq_features(ittem))

c_list = [5.736152510448679]
kernel_list = ['rbf']
gamma_list = [0.002212216291070448]

# 10-fold cross validation
kf = KFold(n_splits = 10, shuffle = False)
X = positive_set + negative_set
print('Total seqs: ', len(X))
Y = [1] * len(positive_set) + [0] * len(negative_set)
df = pd.DataFrame({'seq':X, 'label':Y})
df = shuffle(df)

def df2list(df):
    return df.values.tolist()

metrics = []
for fold, (train_idx, test_idx) in enumerate(kf.split(df)):
    print('Fold: ', str(fold+1))
    X_train = df['seq'].iloc[train_idx]
    y_train = df['label'].iloc[train_idx]
    X_test = df['seq'].iloc[test_idx]
    y_test = df['label'].iloc[test_idx]

    train_svc = SVC(gamma = gamma_list[0], C = c_list[0], kernel = kernel_list[0], probability = True)
    train_svc.fit(df2list(X_train), df2list(y_train))
    test_preds = train_svc.predict(df2list(X_test))
    metrics.append(show_metrics(test_preds, df2list(y_test)))

average = np.mean(metrics, axis=0)
print(average)
metrics.append(average)
df_results = pd.DataFrame(metrics, columns=['sensitivity', 'specificity', 'accuracy', 'precision', 'mcc', 'f1', 'auroc'])
df_results.to_csv('train_10fold_results.csv')
