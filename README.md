# Pore-Forming_AMP_SVM
An SVM Classifier for Screening Pore-Forming Antimicrobial Peptides (2024 ver.)

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

### For predicting/screening
```bash
python predict.py
```
* Note: The length of input peptide should not be longer than 40 amino acids.

## Citation
To be updated
