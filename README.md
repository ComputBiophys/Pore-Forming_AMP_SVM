# PAP-SVM
An SVM Classifier for Screening Perforated Antimicrobial Peptides (2024 modified)

## Requirements
* thundersvm
* prot_bert_bfd (from transformers)
* torch
* sklearn

## Usage
### For independent training/validation
```bash
python train_val.py
```

### For 10-fold cross validation
```bash
python train_10fold.py
```

### For predicting (or screening from new seqs)
```bash
python predict.py
```
