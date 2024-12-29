# PAP-SVM
An SVM Classifier for Screening Pore-forming Antimicrobial Peptides (ver.2024)

## Requirements
* thundersvm
* transformers (for prot_bert_bfd)
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
