# **NSFA-Net**  
**Noise-Aware Epileptic Seizure Prediction  Network via Self-Attention Feature Alignment**  

A noise-aware epileptic seizure prediction network that aligns multi-layer features and maintains the contextual consistency, achieving SOTA performance.

This project provides the code for model training and testing.

## **🚀 Quick Start**
### **1. Prepare Dataset**

We provide code for preprocessing the raw Kaggle dataset, which includes downsampling, segmentation, and STFT transformation. Use the following command to run the dataset preprocessing:

```bash
python Kaggle_preprocess_30.py --patient_id 1
```

Please download the raw Kaggle dataset via the following link:

[Download Link for the raw Kaggle dataset](https://www.kaggle.com/c/seizure-prediction/data)

### **2. Run Training**

The code for model training is located at `train_NSFA.py`. Use the following command to run the training for patient Dog_1:

```bash
python train_NSFA.py --patient_id 1 --device_number 0 --data_dir ../exp_30/Kaggle_dataset_30 --target_preictal_interval 30
```

Please replace the `data_dir` in the command with the actual path of your preprocessed data.

### **3. Run Evaluation**
The code for model testing is located at `test_Kaggle.py`. Use the following command to run the evaluation for patient Dog_1:

```bash
python test_Kaggle.py --patient_id 1 --device_number 0 --batch_size 200 --ckpt_dir ../exp_30/NSFA_ckpt_30 --data_dir ../exp_30/Kaggle_dataset_30 --target_preictal_interval 30
```

Please replace the `ckpt_dir` and the `data_dir` in the command with the actual path of your model checkpoint and preprocessed data. 
