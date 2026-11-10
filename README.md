# Pore-Forming_AMP_SVM
Codes for *Mechanism-Driven Screening of Membrane-Targeting and Pore-Forming Antimicrobial Peptides*

## Requirements
### SVM
* thundersvm
* transformers (for prot_bert_bfd)
* torch
* sklearn
### MCP and CM predictors
* numpy
* matplotlib
* pickleshare
* tensorflow==1.7.0 (if you are using tensorflow 2, please modify the script with 'import tensorflow.compat.v1 as tf')

## Usage for SVM
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
* The model weight can be downloaded [here](https://disk.pku.edu.cn/link/AA87F234B8B573491CB7648B498438E6B9) (Pickup Code: gNq1).

## Usage for MCP and CM predictors
### Requirements
```bash
pip install -r requirements.txt
```

### Feature generation
* Input features are generated from SPIDER3-Single (Heffernan, R. et al., *J Comput Chem* 2018). [Download](https://zhouyq-lab.szbl.ac.cn/download/)

### MCP prediction
```bash
python MCPpep_predictor.py ./example test
```

### CM prediction
```bash
python CMpep_predictor.py ./example test
```
* Note: The complete parameter files can be downloaded at [here](https://drive.google.com/file/d/1Ekc-NNp6jq2DoJapQLmLtv1CtpxxRTCc/view?usp=sharing).

### Inter-chain CM plot
```bash
python plot_dimer.py test
```

## Citation
```
@article{Li2025AMP,
  author = {Li, Jiaxuan and Yang, Chenguang and Dong, Ruihan and Juarez, Juan Francisco Bada and Wang, Lei and Wettstein, Maximilian Emanuel and Wang, Dali and Cao, Chan and Lu, Ying and Song, Chen},
  title = {Mechanism-Driven Screening of Membrane-Targeting and Pore-Forming Antimicrobial Peptides},
  journal = {Advanced Science},
  pages = {e16470},
  doi = {https://doi.org/10.1002/advs.202516470},
  year = {2025}
}
```
