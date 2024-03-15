# BERT-SVM classifier for AMP prediction
# prediction code modified by Ruihan Dong
# 24-01-15

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '6'
import warnings
warnings.filterwarnings("ignore")

import re
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from thundersvm import SVC

import torch
from transformers import AutoTokenizer, AutoModel, pipeline


tokenizer = AutoTokenizer.from_pretrained('./prot_bert_bfd', do_lower_case=False)
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

    return [embedding]


clf_choice = SVC()
clf_choice.load_from_file('./model/0.2MCPall_shuffleFalse_neg6_newuniprot_MCP_SS.model')

filename = 'input.txt'
df = pd.read_csv(filename, header = None)
file_pep_list = df.iloc[:,0].values.tolist()

print('\n>>>>>>Filter SVM model start<<<<<<\n')

preds = []
for pep in tqdm(file_pep_list):
    c_choice = clf_choice.predict(BERT_Extracting_seq_features(pep))
    preds.append(int(c_choice[0]))

df_out = pd.DataFrame(preds)
df_out.to_csv('./results/results.txt', index=False, header=False)
