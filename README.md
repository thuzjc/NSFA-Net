# **NSFA-Net**  
**Noise-Aware Epileptic Seizure Prediction  Network via Self-Attention Feature Alignment**  

A nosie-aware epileptic seizure prediction network that aligns multi-layer features and maintains the contextual consistency, achieving SOTA performance.

This project provides the preprocessed Kaggle dataset, along with the code for model training and testing, and the trained model checkpoints.

## **🚀 Quick Start**
### **1. Prepare Datasets**

We provide the preprocessed Kaggle dataset for model training, which was preprocessed as detailed in the paper.  The data is available for download via the link below:

[Download Link for Preprocessed Kaggle data with the SOP set to 60 minutes](https://pan.baidu.com/s/12aW93-VMInjGH0eU-xmYrg?pwd=pqnw)

[Download Link for Preprocessed Kaggle data with the SOP set to 30 minutes](https://pan.baidu.com/s/1YUrX_WmwsSGxMOxTOs1b2Q?pwd=4f47)

### **2. Run Training**

The code for model training is located at `script/train_NSFA.py`. Use the following command to run the training for patient Dog_1:

```bash
python train_NSFA.py --patient_id 1 --device_number 0 --data_dir Kaggle_dataset_30 --target_preictal_interval 30
```

Please replace `Kaggle_dataset` in the command with the actual path of your data directory.

### **3. Run Evaluation**
The code for model testing is located at `script/test_Kaggle.py`. Use the following command to run the evaluation for patient Dog_1:

```bash
python test_Kaggle.py --patient_id 1 --device_number 0 --batch_size 200 --ckpt_dir NSFA_ckpt_30 --data_dir Kaggle_dataset_30 --target_preictal_interval 30
```

Please replace `NSFA_ckpt` in the command with the actual path of your model checkpoint. We provide our model checkpoints, which can be downloaded via the following link:

[Download Link for NSFA Checkpoints with the SOP set to 60 minutes](https://pan.baidu.com/s/1iqceQj95tpzETs7C4AluFQ?pwd=792j)

[Download Link for NSFA Checkpoints with the SOP set to 30 minutes]( https://pan.baidu.com/s/154zBFvzSdpwyPv0owHJtrQ?pwd=m2pj)

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
