# **SMOC-Net**  
**Noise-Aware Epileptic Seizure Prediction  Network via Self-Attention Feature Alignment**  

A nosie-aware epileptic seizure prediction network that aligns multi-layer features and maintains the contextual consistency, achieving SOTA performance.

This project provides the preprocessed Kaggle dataset, along with the code for model training and testing, and the trained model checkpoints.

## **🚀 Quick Start**
### **1. Prepare Datasets**

We provide the preprocessed Kaggle dataset for model training, which was preprocessed as detailed in the paper.  The data is available for download via the link below:

[Kaggle Data Download Link](https://pan.baidu.com/s/12aW93-VMInjGH0eU-xmYrg?pwd=pqnw)

### **2. Run Training**

The code for model training is located at `script/train_NSFA.py`. Use the following command to run the training for patient Dog_1:

```bash
python train_NSFA.py --patient_id 1 --device_number 0 --data_dir Kaggle_dataset
```

Please replace `Kaggle_dataset` in the command with the actual path of your data directory.

### **3. Run Evaluation**
The code for model testing is located at `script/test_Kaggle.py`. Use the following command to run the evaluation for patient Dog_1:

```bash
python test_Kaggle.py --patient_id 1 --device_number 0 --batch_size 200 --ckpt_dir NSFA_ckpt --data_dir Kaggle_dataset
```

Please replace `NSFA_ckpt` in the command with the actual path of your model checkpoint. We provide our model checkpoints, which can be downloaded via the following link:

[NSFA Checkpoints Download Link](https://pan.baidu.com/s/16q-m8yD4HyqNG-LVxN3m4g?pwd=j39t)

## 📖 Citation
```bash
@ARTICLE{Dong2025NSFA,
  author={Dong, Qiulei and Wang, Zhixi and Gao, Mengyu},
  journal={IEEE Journal of Biomedical and Health Informatics}, 
  title={Noise-Aware Epileptic Seizure Prediction Network via Self-Attention Feature Alignment}, 
  year={2025},
  pages={1-13},
  keywords={Feature extraction;Electroencephalography;Time-frequency analysis;Data mining;Training;Signal to noise ratio;Transformers;Aggregates;Bioinformatics;Sensitivity;Epileptic seizure prediction;feature alignment;noise-aware regularizer},
  doi={10.1109/JBHI.2025.3579229}}
```
