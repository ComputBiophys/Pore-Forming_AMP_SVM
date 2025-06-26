# Pore-Forming_AMP_SVM
An SVM Classifier for Screening Pore-Forming Antimicrobial Peptides (2024 ver.)

## Requirements
* thundersvm
* transformers (for prot_bert_bfd)
* torch
* sklearn
* numpy
* matplotlib
* pickleshare
* tensorflow==1.7.0

## Usage （SVM predictor）
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

## Usage （MCP predictor and CM predictor）
### Requirements (Python 3.6)
```bash
pip install -r requirements.txt
```

### Feature generation

* Note: Input features from SPIDER3-Single (Heffernan, R, et al. J Comput Chem. 2018)

### MCP prediction
```bash
python MCPpep_predictor.py ./example test
```

### CM prediction
```bash
python CMpep_predictor.py ./example test
```
* Note: The complete parameter files can be downloaded at https://disk.pku.edu.cn/link/AA90CAA6C9568B4E809BE5FFCFD95381E7.

### Inter-chain CM plot
```bash
python plot_dimer.py test
```

## Citation
To be updated
