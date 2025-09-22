# -*- coding: utf-8 -*-

"""

Created on Tue Jul  6 16:49:27 2021

  

@author: phantom

  

#CHB
"""

import warnings

warnings.filterwarnings("ignore")

import numpy as np

import torch

import torch.nn as nn

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt

from EEG_dataset.dataset_test import testDataset

from EEG_utils.eeg_utils import GetPatientList, GetSeizureList, GetInputChannel, GetModel

#from EEG_utils.write_to_excel import WriteToExcel, WriteSeizureToExcel, CalculateAverageToExcel

from torch.nn import functional as F

import torch.utils.data as Data

import argparse

# import scipy.io as io

import os

from sklearn.metrics import roc_auc_score,roc_curve, auc

import time

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import pandas as pd
def save_lists_to_excel(headers, data_lists, filename):
    if len(headers) != len(data_lists):
        raise ValueError("Headers and data lists must have the same length.")
    data_dict = {headers[i]: data_lists[i] for i in range(len(headers))}
    df = pd.DataFrame(data_dict)
    df.to_excel(filename, index=False)
    print(f'save data to {filename}')

def add_data_to_excel(filename, new_data):
    try:
        df = pd.read_excel(filename)
    except:
        df = pd.DataFrame(columns = new_data.keys())
    new_row = pd.DataFrame([new_data])
    df = pd.concat([df, new_row], ignore_index=True)
    data_columns = ['AUROC', 'SN', 'FPR']
    if new_data.get('ID') == 'Dog_5':
        avg_values = df[data_columns].mean().to_dict()
        avg_row = {'ID': 'Average', **avg_values}
        for col in df.columns:
            if col not in avg_row:
                avg_row[col] = None
        avg_row_df = pd.DataFrame([avg_row])
        df = pd.concat([df, avg_row_df], ignore_index = True)
    df = df.round(4)
    df.to_excel(filename, index=False)
    print(f'add data to {filename}')
    

  

def fact(x):
    if x == 1 or x == 0:

        return 1

    return x * fact(x - 1)

  
  

def Comb(n, m):
    ma = fact(m) * fact(n - m)

    son = fact(n)

    result = son / ma

    return result

  
  

def P_value_FPR(fpr, tp, nfold, sop=30):   # 30min=0.5h
    p_ = 0

    sop = sop / 60

    P = 1 - np.exp(-fpr * sop)

    for i in range(tp, nfold + 1):

        Q1 = (1 - P) ** (nfold - i)

        Q2 = (P ** i)

        p_ = p_ + Comb(nfold, i) * Q1 * Q2

    return p_

  

def setup_seed(seed):

    '''

    set up seed for cuda and numpy

    '''

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True

  
  

def new_calculate_sn_fpr(scores, threshold, interictal_length, preictal_length, count_persistence, interictal_step, preictal_step):

    count = 0

    interval = 0

    false_alarm_list = []

    true_alarm_list = []

    for index in range(len(scores)):

        if scores[index] > threshold:

            count += 1

        else:

            count = 0

        if count >= count_persistence:

            interval = interictal_length + preictal_length - index

            if index >= interictal_length and index < interictal_length + preictal_length:

                true_alarm_list.append(interval)

            elif index < interictal_length:

                false_alarm_list.append(interval)

            count = 0

    FPR = len(false_alarm_list) / (((interictal_length * interictal_step) + preictal_length * preictal_step + 30-5)/3600)

    SN = 1 if true_alarm_list else 0

    return SN, FPR

  

def smooth(a,WSZ):

    '''

    smoothing function, which is used to smooth the seizure predicting results

    a:original data. NumPy 1-D array containing the data to be smoothed

    a need to be 1-D. If not, use np.ravel() or np.squeeze() to make a transpose

    WSZ: moving_average_length, smoothing window size needs, which must be odd number,

    as in the original MATLAB implementation

    '''

    if(WSZ%2==0):

        WSZ-=1

    out0 = np.convolve(a,np.ones(WSZ,dtype=int),'valid')/WSZ

    r = np.arange(1,WSZ-1,2)

    start = np.cumsum(a[:WSZ-1])[::2]/r

    stop = (np.cumsum(a[:-WSZ:-1])[::2]/r)[::-1]

    return np.concatenate((start, out0, stop))

  
  

def plot_roc(labels, predict_prob):

    '''

    plot ROC curve

    labels : true labels

    predict_prob : predicted probabilities

    '''

    false_positive_rate,true_positive_rate,thresholds=roc_curve(labels, predict_prob)

    roc_auc=auc(false_positive_rate, true_positive_rate)

    plt.title('ROC')

    plt.plot(false_positive_rate, true_positive_rate,'b',label='AUC = %0.4f'% roc_auc)

    plt.legend(loc='lower right')

    plt.plot([0,1],[0,1],'r--')

    plt.ylabel('TPR')

    plt.xlabel('FPR')

    plt.show()

  

def auto_threshold_probability(auto_threshold, output_probability, labels):

    auto_threshold = False

    if auto_threshold:

        predictions = output_probability[:,1]

        # 初始化最佳阈值和最大AUC

        best_threshold = 0.5

        best_auc = 0.0

        best_performance = 0.0

  

        # 寻找最佳AUC和阈值

        label = labels.flatten()

        for threshold_p in np.arange(0.1, 1.0, 0.05):

            # 将预测概率转化为二进制预测结果

            adjusted_prediction = (predictions >= threshold_p).astype(int)  # 预测概率大于阈值的为1.小于阈值的为0

  

            # 计算当前阈值的AUC

            current_auc = roc_auc_score(labels, adjusted_prediction)

            # print("current_auc", current_auc)

  

            # 检查是否获得更高的AUC

            if current_auc > best_auc:

                best_threshold = threshold_p

                best_auc = current_auc

                # print("------------------------")

                # print("best_auc", best_auc, "best_threshold", best_threshold)

        adjusted_prediction = (predictions >= best_threshold).astype(int)

        print("Best Threshold:", best_threshold)

        print("Best AUC:", best_auc)

        predicting_probability = (predictions >= best_threshold).astype(int)

    else:

        predicting_probability=output_probability[:,1]

        best_threshold = None

        return predicting_probability, best_threshold

  

if __name__ == "__main__":    

    parser = argparse.ArgumentParser(description = 'Seizure predicting on Xuanwu Dataset')

    parser.add_argument('--patient_id', type = int, default = 3, metavar = 'patient_id')

    parser.add_argument('--ch_num', type = int, default = 18, metavar = 'number of channel')

    parser.add_argument('--model_name', type = str, default = "swin_AlignFA_Third_noRRB", metavar = 'N')  # TA_STS_ConvNet  VIT8_16_32patch_12depth

    parser.add_argument('--used_model', type = str, default = "swin_AlignFA_noRRBtest", metavar = 'N')  # TA_STS_ConvNet  VIT8_16_32patch_12depth

    parser.add_argument('--dataset_name', type = str, default = "KAGGLE", metavar = 'XUANWU / CHB / Kaggle')

    parser.add_argument('--description', type = str, default = "normal", metavar = 'normal / TL / TL-lock / DA') #normal/transfer learning/transfer learning lock/domain adaptation

    parser.add_argument('--target_preictal_interval', type = int, default = 60, metavar = 'how long we decide as preictal. Default set to 15 min')    

    parser.add_argument('--interictal_step', type = int, default = 30, metavar = 'step of sliding window (second) for interictal data') #if stft : 30  

    parser.add_argument('--step_preictal', type = int, default = 5, metavar = 'step of sliding window (second)')

    parser.add_argument('--device_number', type = int, default = 0, metavar = 'CUDA device number')

    parser.add_argument('--position_embedding', type = bool, default = False, metavar = "whether train use position embedding")  

    parser.add_argument('--seed', type = int, default = 20221110, metavar = 'N')

    parser.add_argument('--batch_size', type = int, default = 200, metavar = 'batchsize')

    parser.add_argument('--using_cuda', type = bool, default = True, metavar = 'whether using cuda')

    parser.add_argument('--threshold', type = float, default = 0.6, metavar = 'alarm threshold')

    parser.add_argument('--moving_average_length', type = int, default = 25, metavar = 'length of smooth window')

    parser.add_argument('--persistence_second', type = int, default = 5, metavar = 'N')  # 5

    parser.add_argument('--tw0', type = float, default = 1/120, metavar = '1/120 hour, which is 30 seconds')

    parser.add_argument('--is_test', type = bool, default = True, metavar = "whether test") # suffix

    parser.add_argument('--suffix', type = str, default = "smooth", metavar = 'bestauc / bestp / bestFPR')

    parser.add_argument("--part", type = int, default = 0, metavar = 'which part to test')
    parser.add_argument('--ckpt_dir', type = str, default = '/mnt/share/zjc/model/KAGGLE_new/NSFA_resave')
    parser.add_argument("--auto_threshold", type = bool, default = False, metavar = 'whether use auto threshold')
    parser.add_argument('--data_dir', type=str, default = '../Kaggle_new_split', metavar='data directory')

    args = parser.parse_args()
    part = args.part

    model_name=args.model_name

    used_model=args.used_model

    dataset_name=args.dataset_name

    description=args.description

    patient_id=args.patient_id

    seed=args.seed

    moving_average_length=args.moving_average_length

    # moving_average_length = 6

    # moving_average_length = 25

    # moving_average_length = 5

    target_preictal_interval=args.target_preictal_interval

    interictal_step=args.interictal_step

    step_preictal=args.step_preictal

    device_number=args.device_number

    using_cuda=args.using_cuda

    ch_num=args.ch_num

    position_embedding = args.position_embedding

    suffix = args.suffix

    batch_size = args.batch_size

    threshold=args.threshold

    persistence_second=args.persistence_second

    tw0=args.tw0

    tw=target_preictal_interval/60

    patient_list=GetPatientList(dataset_name)

    seizure_list=GetSeizureList(dataset_name)

    seizure_count=len(seizure_list[str(patient_id)])

    patient_name=patient_list[str(patient_id)]

    print("patient : {}".format(patient_id))

    print("dataset : {} | seizure : {} filter : {} | threshold : {} | persistence : {} | tw0 : {} | tw : {}".format(args.dataset_name,

                    seizure_count, moving_average_length*step_preictal, threshold, persistence_second, tw0*3600, tw*3600))

  

    eval_param={"descri": description,

                "filter": moving_average_length*step_preictal,

                "threshold": threshold,

                "persistence": persistence_second}

    excel_list={'1' : 0,

                '2' : 6,

                '3' : 15,

                '4' : 29,

                '5' : 45,}

    TP_list=[]

    FN_list=[]

    TN_list=[]

    FP_list=[]

    FPR_list=[]

    SEN_list=[]

    AUC_list=[]

    AUC3_list=[]

    InferTime_list1 = []

    InferTime_list2 = []

    PW_count=0

    line_count=0

    input_channel=GetInputChannel(dataset_name, patient_id, ch_num)

    #LOOCV for predicting
    ckpt_dir = args.ckpt_dir
    # for ckpt in ckpt_part.keys():  
    for i in range(seizure_count):

        #load test data
        print(f'testing for patient_{patient_id} part_{i}')
        ckpt_path = os.path.join(ckpt_dir, str(patient_id), f'part{i}.pth')
        test_set = testDataset(dataset_name, i, using_ictal=1, patient_id=patient_id, patient_name=patient_name,

                                     ch_num=input_channel, target_preictal_interval=target_preictal_interval, step_preictal=step_preictal, data_dir = args.data_dir)

        test_loader = Data.DataLoader(dataset = test_set,batch_size = batch_size, shuffle = False)

        labels=test_set.y_data.numpy()

        preictal_length=test_set.preictal_length

        interictal_length=test_set.interictal_length

        print("interrictal : {} | preictal : {}: ".format(interictal_length, preictal_length))

  

        #get model

        # input_channel=GetInputChannel(dataset_name, patient_id, ch_num)

        model = GetModel(input_channel, device_number, used_model, dataset_name, position_embedding)  

        # if torch.cuda.device_count() > 1:

            #   print("Use", torch.cuda.device_count(), 'gpus')

        #       model = nn.DataParallel(model)  

        # model.load_state_dict(torch.load('{}/model/{}/{}/{}/patient{}_{}.pth'.format(model_path, dataset_name, model_name,  patient_id, patient_id, i)))

        device = torch.device(f"cuda:{device_number}" )
        
        print(f'Loading ckpt from path:{ckpt_path}')
        model.load_state_dict(torch.load(ckpt_path, map_location=device)['state_dict'])

        #set cuda

        # if using_cuda:

        #     torch.backends.cudnn.enabled = True

        #     torch.backends.cudnn.benchmark = True

        #     torch.cuda.set_device(device_number)

        #     model = model.cuda()            

        device = torch.device(f"cuda:{device_number}" ) # 指定gpu2为主GPU

        model.to(device)

        torch.cuda.set_device(device_number)

        #start predicting

        #start_time1 = time.clock()

        start_time2 = time.time()

        model.eval()

        output_probablity=[]

        output_list=[]

        feature_list=[]

        with torch.no_grad():

            for k, (data, target) in enumerate(test_loader):

                if using_cuda:

                    data = data.to(device)

                    target = target.to(device)

                # return_value, embedding = model(data)

                # output=return_value[0]

                # output=return_value

                output, embedding = model(data)

                output_nosoftmax=output.cpu().detach().numpy()

                output=F.softmax(output, dim=1)

                #output=torch.clamp(output, min=1e-9, max=1-1e-9)

                output=output.cpu().detach().numpy()

                if len(output_probablity)==0:

                    output_probablity.append(output)  

                    output_probablity=np.array(output_probablity).squeeze()

                    output_list.append(output_nosoftmax)  

                    output_list=np.array(output_list).squeeze()

                else:

                    output_probablity=np.vstack((output_probablity, output))

                    output_list=np.vstack((output_list, output_nosoftmax))

        #infer_time1 = (time.clock()-start_time1)/(preictal_length+interictal_length)

        infer_time2 = (time.time()-start_time2)/(preictal_length+interictal_length)

        #InferTime_list1.append(infer_time1)

        InferTime_list2.append(infer_time2)

        #save output probabilities

        # np.save('{}/model/{}/{}/{}/probablity{}_{}.npy'.format(model_path, dataset_name, model_name, patient_id, patient_id, i), output_probablity)

        # np.save('{}/model/{}/{}/{}/output{}_{}.npy'.format(model_path, dataset_name, model_name, patient_id, patient_id, i), output_list)

        # np.save('{}/model/{}/{}/{}/label{}_{}.npy'.format(model_path, dataset_name, model_name, patient_id, patient_id, i), labels)

        # predicting_probablity=output_probablity[:,1]

        # 修改阈值

        predicting_probability, best_threshold = auto_threshold_probability(args.auto_threshold, output_probablity, labels) # args.auto_threshold = False


        #calculate AUC, draw ROC curve

        y_true=labels

        y_probablity=output_probablity

        y_score1=y_probablity[:,1]

        auc_value1=roc_auc_score(y_true, y_score1)

        print("AUC1", auc_value1)

        auc_value2=roc_auc_score(y_true, predicting_probability)

        print("AUC2", auc_value2)

        best_auc =max(auc_value1,auc_value2)

        AUC_list.append(best_auc)

        predicting_probablity_smooth=smooth(predicting_probability, moving_average_length)

        print(f'moving_average_length:{moving_average_length}')

        auc_value3=roc_auc_score(y_true, predicting_probablity_smooth)

        # print("AUC3", auc_value3)

        AUC3_list.append(auc_value3)

        predicting_result=output_probablity.argmax(axis = 1)

        # np.save('{}/model/{}/{}/{}/pre_label{}_{}.npy'.format(model_path, dataset_name, model_name, patient_id, patient_id, i), predicting_result)

        # np.save('{}/model/{}/{}/{}/smooth_probablity{}_{}.npy'.format(model_path, dataset_name, model_name, patient_id, patient_id, i), predicting_probablity_smooth)

        #calculate confusion matrix

        TP,FP,TN,FN=0,0,0,0

        for j in range(len(labels)):

            if predicting_result[j] ==1 and labels[j]==1:

                TP+=1

            elif predicting_result[j] ==0 and labels[j]==1:

                FN+=1

            elif predicting_result[j] ==0 and labels[j]==0:

                TN+=1

            else:

                FP+=1

        TP_list.append(TP)

        FN_list.append(FN)

        TN_list.append(TN)

        FP_list.append(FP)

        #calculate currect alarm and false alarm. calculate Sensitivity and FPR/h

        count=0

        interval=0  # 距离发作点的时间

        false_alarm_list=[]  # 误报时间点列表

        true_alarm_list=[]  # 正报时间点列表

        for index in range(len(predicting_probablity_smooth)):

            # probability is over threshold, start counting

            if predicting_probablity_smooth[index]>threshold:

                PW_count += 1

                count+=1

            else:

                count=0

            #if count is over persistence second，decide as one alarm

            if count>=persistence_second//step_preictal:

                interval=interictal_length+preictal_length-index

                #if the alarm is within 15min，True alarm

                if index >= interictal_length and index < interictal_length + preictal_length:

                    true_alarm_list.append(interval)

                #if the alarm is not within 15min，False alarm

                elif index < interictal_length:

                    false_alarm_list.append(interval)

                count=0

        FPR=len(false_alarm_list)/((interictal_length*interictal_step+preictal_length*step_preictal + target_preictal_interval - step_preictal)/3600)

  

        FPR_list.append(FPR)

        if len(true_alarm_list) > 0:

            SEN_list.append(1)

        else:

            SEN_list.append(0)

            true_alarm_list.append(-1)

        print("TP {} FN {} TN {} FP {} "

        "sen {:.2%} spe {:.2%} acc {:.2%}; "

        "TA {} FA {} SN:{} FPR {:.4f} AUC {:.4f} "

        "PT {} IT1 {:.4f} IT2 {:.4f}".format(

            TP, FN, TN, FP,

            TP/(TP+FN) if (TP+FN) > 0 else 0,  # 避免除零错误

            TN/(TN+FP) if (TN+FP) > 0 else 0,  # 避免除零错误

            (TP+TN)/(TP+FN+TN+FP) if (TP+FN+TN+FP) > 0 else 0,  # 避免除零错误

            len(true_alarm_list), len(false_alarm_list), SEN_list[-1],

            FPR, best_auc,

            true_alarm_list[0] if true_alarm_list else 0,  # 避免空列表错误

            infer_time2, infer_time2

        ))


    #calculate p-value
    save_data = {'ID' :patient_name, 'AUROC': np.mean(AUC_list), 'SN': np.mean(SEN_list), 'FPR': np.mean(FPR_list)}
    filename = f'{ckpt_dir}/mine_result.xlsx'
    add_data_to_excel(filename, save_data)