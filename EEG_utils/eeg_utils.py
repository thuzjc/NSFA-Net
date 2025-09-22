# -*- coding: utf-8 -*-
"""
Created on Sun Sep  5 21:32:05 2021

@author: chongxian
"""

def GetPatientList(dataset):
    """
    Get Patient Name List
    """
    patient_list=[]
    if dataset=="XUANWU":
        patient_list={'1': 'lxz',
                      '2': 'hu jing',
                      '3': 'zy'}
    elif dataset=="CHB":
        patient_list={'1' : 'chb01', 
                      '2' : 'chb02',
                      '3' : 'chb03',
                      '5' : 'chb05',
                      '6' : 'chb06',
                      '7' : 'chb07',
                      '8' : 'chb08',
                      '9' : 'chb09',
                      '10': 'chb10',
                      '11': 'chb11',
                      '13': 'chb13',
                      '14': 'chb14',
                      '16': 'chb16',
                      '17': 'chb17',
                      '18': 'chb18',
                      '20': 'chb20',
                      '21': 'chb21',
                      '22': 'chb22',
                      '23': 'chb23'}
    elif dataset=="KAGGLE":
        patient_list={'1' : 'Dog_1', 
                      '2' : 'Dog_2',
                      '3' : 'Dog_3',
                      '4' : 'Dog_4',
                      '5' : 'Dog_5'}
    else:
        print("111111111---------------------------")
        print("\nplease input correct dataset name\n")
        exit()
    return patient_list

def GetSeizureList(dataset):
    """
    Get Seizure List
    """
    seizure_list=[]
    if dataset=="XUANWU":
        seizure_list={'1': [0,1,2,3,4,5,6,7], 
                      '2': [0,1,2],
                      '3': [0,1,2,3,4]}
    elif dataset=="CHB":
        seizure_list={'1' : [0,1,2,3,4,5,6], 
    		          '2' : [0,1,2],
    		          '3' : [0,1,2,3,4,5],
    		          '5' : [0,1,2,3,4],
    		          '6' : [0,1,2,3,4,5,6],
    		          '7' : [0,1,2],
    		          '8' : [0,1,2,3,4],
    		          '9' : [0,1,2,3],
                      '10': [0,1,2,3,4,5],
    		          '11': [0,1,2],
    		          '13': [0,1,2,3,4],
    				  '14': [0,1,2,3,4,5],
                      '16': [0,1,2,3,4,5,6,7],
    				  '17': [0,1,2],
    				  '18': [0,1,2,3,4,5],  # ,5
    				  '20': [0,1,2,3,4,5,6,7],
    		          '21': [0,1,2,3],
    		          '22': [0,1,2],
    				  '23': [0,1,2,3,4,5,6],}
    elif dataset=="KAGGLE":
        seizure_list={'1' : [0,1,2,3],  # 4
    		          '2' : [0,1,2,3,4,5,6],  # 7
    		          '3' : [0,1,2,3,4,5,6,7,8,9,10,11],  # 12
    		          '4' : [0,1,2,3,4,5,6,7,8,9,10,11,12,13],  # 14
    		          '5' : [0,1,2,3,4]}  # 5
    else:
        print("222222222222222---------------------------")
        print("\nplease input correct dataset name\n")
        exit()
    return seizure_list

def GetDataPath(dataset):
    """
    Get Data Path
    """
    data_path=""
    if dataset=="XUANWU":
        data_path="/share/home/wangzhixi/local_tools/TA-STS-ConvNet/data/XUANWU"
    elif dataset=="CHB":
        data_path="/share/home/wangzhixi/local_tools/TA-STS-ConvNet/data/CHB-MIT"
    elif dataset=="KAGGLE":
        data_path="/share/home/wangzhixi/local_tools/TA-STS-ConvNet/data/Kaggle"
    else:
        print("3333333333---------------------------")
        print("\nplease input correct dataset name\n")
        exit()
    return data_path

def GetDataType(dataset):
    """
    Get Data Type
    """
    data_type=""
    if dataset=="XUANWU":
        data_type="SEEG"
    elif dataset=="CHB":
        data_type="EEG"
    elif dataset=="KAGGLE":
        data_type="iEEG"
    else:
        print("44444444444---------------------------")
        print("\nplease input correct dataset name\n")
        exit()
    return data_type

def GetInputChannel(dataset, patient_id, ch_num):
    '''
    Get Model Input Channel Number for each patient
    '''
    if dataset=="XUANWU":#ANT
        if ch_num==0:
            ch_num_list={'1' : 133,
                         '2' : 115,
                         '3' : 189}
            return ch_num_list[str(patient_id)]
    elif dataset=="CHB":
        if ch_num!=18:
            print("\nplease input correct channel number for CHB name\n")
            return
        ch_num=18 
    elif dataset=="KAGGLE":
        if patient_id==5:
            ch_num=15
        else:
            ch_num=16 
    else:
        print("555555555---------------------------")
        print("\nplease input correct dataset name\n")
        return
    return ch_num

def GetBatchsize(dataset, patient_id, model_name):
    '''
    Get Batchsize for each patient
    '''
    batchsize=256
    if dataset=="XUANWU":#ANT
        batchsize = 32  # 256
    elif dataset=="CHB":
        if model_name == "swin_AlignFA_Third_layer12":
            batchsize = 50 if patient_id == 9 or patient_id ==10 else 54  # 200 256 swin_AlignFA_Contrastive->40
        elif model_name == "swin_AlignFA_Third_layer123":
            batchsize = 40 if patient_id == 9 or patient_id ==10 else 44
        elif model_name == "swin_Third":
            batchsize = 120 if patient_id == 9 or patient_id ==10 else 128
        elif model_name == "swin_AlignFA_Contrastive_256A2":
            batchsize = 56 if patient_id == 9 or patient_id ==10 else 60
        elif model_name == "swin_AlignFA_Contrastive_256A3":
            batchsize = 116 if patient_id == 9 or patient_id ==10 else 120
        elif model_name.startswith("swin_AlignFA_Third_noRRB"):
            batchsize = 86 if patient_id == 9 or patient_id ==10 else 90
        elif model_name.startswith("CNN") or model_name.startswith("VIT") or model_name.startswith("ResNet"):
            batchsize = 128  # swin_AlignFA_Contrastive->40
        else:
            batchsize = 86 if patient_id == 9 or patient_id ==10 else 90
    elif dataset=="KAGGLE":
        if model_name == "swin_AlignFA_Third_layer12":
            batchsize = 54  # swin_AlignFA_Contrastive->40
        elif model_name == "swin_AlignFA_Third_layer123":
            batchsize = 48  # swin_AlignFA_Contrastive->40
        elif model_name.startswith("CNN") or model_name.startswith("VIT") or model_name.startswith("ResNet"):
            batchsize = 128  # swin_AlignFA_Contrastive->40
        else:
            batchsize = 86 if patient_id == 9 or patient_id ==10 else 90
    else:
        print("6666666666---------------------------")
        print("\nplease input correct dataset name\n")
        exit()
    return batchsize

import scipy.io as io
from EEG_model.swin_AlignFA_noRRBtest import swin_tiny_patch4_window7_224 as swin_AlignFA_noRRBtest

# from EEG_model.EEG_Transformer import eegt 
# from EEG_model.rest_v2 import restv2_tiny

def GetModel(input_channel, device_number, model_name, dataset_name, position_embedding):
    '''
    Get Model TA_STS_ConvNet
    '''
    
    if model_name == 'NSFA':
        model = swin_AlignFA_noRRBtest(num_classes = 2, input_channel = input_channel)
    # elif model_name == "EEG_Transformer":
    #     model = eegt()
    # elif model_name == "restv2_tiny":
    #     model = restv2_tiny()
    else:
        print("mode name incorrect : {}".format(model_name))
        exit()
    return model

from EEG_model.seizureNetLoss import CE_Loss, FocalLoss, SSCLoss, SSC_CELoss, ME_Loss
import torch.nn as nn
def GetLoss(loss):
    if loss == "CE":
        Loss = CE_Loss()
    elif loss == "FL":
        Loss = FocalLoss()
    elif loss == "ME":
        Loss = ME_Loss()
    elif loss == "SSC":
        Loss = SSCLoss()
    elif loss == "SSC_CE":
        Loss = SSC_CELoss()
    elif loss == "Cosine":
        Loss = nn.CosineEmbeddingLoss(margin=0.5)
    else:
        print("Loss {} does not exist".format(loss))
        exit()
    return Loss

def mkdir(path):
    import os
    path=path.strip()
    path=path.rstrip("\\")
    isExists=os.path.exists(path)
    if not isExists:
        os.makedirs(path) 
        print(path+' create successfully')
        return True
    else:
        print(path+' path already exist')
        return False
 