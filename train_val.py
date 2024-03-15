# BERT-SVM classifier for AMP prediction
# indenpendent training code modified by Ruihan Dong
# 24-03-07

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

# from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.metrics import make_scorer, accuracy_score, precision_score, matthews_corrcoef, roc_auc_score, f1_score

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


# dataset preprocessing
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
	else:
		embedding = embedding[:43008]
	return embedding


def read_dataset(filename):
	df = pd.read_csv(filename)
	encoding_seqs = []
	seqs = df['Seq'].values.tolist()
	for seq in tqdm(seqs):
		emb = BERT_Extracting_seq_features(seq)
		encoding_seqs.append(emb)

	labels = df['Label'].values.tolist()
	return encoding_seqs, labels

datapath = './dataset/benchmark/'  # or cls_benchmark_imbalanced
train_reps, train_labels = read_dataset(datapath + 'training_data.csv')
val_reps, val_labels = read_dataset(datapath + 'val_data.csv')

# C gamma grid search
c_list = np.linspace(1, 2, 10)
gamma_list = np.linspace(0.002, 0.02, 10)
print('Total trials: ', len(gamma_list) * len(c_list))

best_mcc = 0
best_params = {'C':c_list[0], 'gamma':gamma_list[0], 'kernel':'rbf'}

for gamma in gamma_list:
	for C in c_list:
		grid_svc = SVC(C = C, gamma = gamma, kernel = 'rbf')
		grid_svc.fit(train_reps, train_labels)
		val_mcc = matthews_corrcoef(val_labels, grid_svc.predict(val_reps))
		print('Grid search running: C {}, gamma {}, val_mcc {:.4f}'.format(C, gamma, val_mcc))
		if val_mcc > best_mcc:
			best_mcc = val_mcc
			best_params = {'C':C, 'gamma':gamma, 'kernel':'rbf'}

print('best params: ', best_params)
t0 = time()
model_svc = SVC(**best_params)
model_svc.fit(train_reps, train_labels)
t1 = time()
print('Training time: {:.3f} min'.format((t1 - t0) / 60))

# calculate metrics
val_pre_label = model_svc.predict(val_reps)
show_metrics(val_labels, val_pre_label)
