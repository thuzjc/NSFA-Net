# -*- coding: utf-8 -*-
"""
Created on Sat May 15 22:42:15 2021

@author: phantom

#example of preprocess one patient : 
python chb_preprocess.py --patient_id=1
"""
import os
import glob
import numpy as np
import random
from EEG_preprocess.Kaggle_mat_file import KaggleMatFile
from EEG_preprocess.chb_stft import getSpectral_STFT
import matplotlib.pyplot as plt
import argparse
import sys
sys.path.append('')
from EEG_utils.eeg_utils import GetInputChannel, mkdir, GetPatientList, GetDataPath, GetSeizureList


def wgn(x, snr):
    ## x: 输入振动信号，形状为 (信号长度, 通道数)
    ## snr: 噪声强度，例如 -8, 0, 4, 8 等
    ### snr=0 表示噪声功率与信号功率相等
    ### snr>0 表示信号强于噪声，值越大噪声越小
    ### snr<0 表示噪声强于信号，值越小噪声越大
    
    # 计算每个通道的信号功率（沿信号长度维度求和后取平均）
    # axis=0 对应信号长度维度，len(x) 是信号长度
    Ps = np.sum(np.abs(x)**2, axis=0) / len(x)  # 结果形状: (通道数,)
    
    # 根据SNR计算每个通道的噪声功率
    Pn = Ps / (10**(snr / 10))  # 结果形状: (通道数,)
    
    # 将噪声功率扩展为与输入信号同形状 (信号长度, 通道数)
    # 对每个通道的噪声功率，沿信号长度维度复制
    Pn = np.repeat(Pn.reshape(1, -1), len(x), axis=0)  # 结果形状: (信号长度, 通道数)
    
    # 生成高斯白噪声并按噪声功率加权
    # 噪声形状与输入信号一致 (信号长度, 通道数)
    noise = np.random.randn(*x.shape) * np.sqrt(Pn)
    
    # 信号加噪声
    signal_add_noise = x + noise
    return signal_add_noise
# def wgn(x, snr):
#         ## x: input vibration signal shape (a,b); a:samples number; b samples length
#         ## snr: noise intensity,like -8,4,2,0,2,4,8
#         ### snr=0 means noise equal to vibration signal 
#         ### snr>0 means vibration signal stronger than noise, →∞ means no noise
#         ### snr<0 means noise stronger than vibration signal  →-∞ means no signal
#         Ps = np.sum(abs(x)**2,axis=1)/len(x)
#         Pn = Ps/(10**((snr/10)))
#         row,columns=x.shape
#         Pn = np.repeat(Pn.reshape(-1,1),columns, axis=1)

#         noise = np.random.randn(row,columns) * np.sqrt(Pn)
#         signal_add_noise = x + noise
#         return signal_add_noise

def setup_seed(seed):
    '''
    set up random seed for numpy
    '''
    np.random.seed(seed)
    random.seed(seed)
    
class KagglePatient:
    def __init__(self, patient_id, dataset_name, data_path, ch_num, doing_lowpass_filter, preictal_interval):
        
        self.interictal_interval=90 # 90min or longer before a seizure, decide as interictal data
        self.preictal_interval=preictal_interval #how long we decide as preictal. Default set to 15 min
        self.postictal_interval=120 # within 120min after a seizure, decide as postictal data
        self.patient_id = patient_id
        self.dataset_name = dataset_name
        self.data_path=data_path
        self.ch_num=ch_num
        self.doing_lowpass_filter=doing_lowpass_filter
        self.patient_name=self.get_patient_name()
        
        #load edf files with seizure
        # self._edf_files_seizure = list(map(
        #     lambda filename: KaggleMatFile(filename, self.patient_id, self.ch_num, self.doing_lowpass_filter),
        #     sorted(glob.glob("/share/home/wangzhixi/dataset/Kaggle/Dog_55/%s/%s/Dog_%d_preictal_segment_*.mat" % (self.patient_name, self.patient_name, self.patient_id)))
        # ))
        self._edf_files_seizure = list(map(
            lambda filename: KaggleMatFile(filename, self.patient_id, self.ch_num, self.doing_lowpass_filter),
            sorted(glob.glob("/mnt/share/zjc/Kaggle_origin/%s/Dog_%d_preictal_segment_*.mat" % ( self.patient_name, self.patient_id)))
        ))
        #load edf files without seizures
        # self._edf_files_unseizure = list(map(
        #     lambda filename: KaggleMatFile(filename, self.patient_id, self.ch_num, self.doing_lowpass_filter),
        #     sorted(glob.glob("/share/home/wangzhixi/dataset/Kaggle/Dog_55/%s/%s/Dog_%d_interictal_segment_*.mat" % (self.patient_name, self.patient_name, self.patient_id)))
        # ))
        self._edf_files_unseizure = list(map(
            lambda filename: KaggleMatFile(filename, self.patient_id, self.ch_num, self.doing_lowpass_filter),
            sorted(glob.glob("/mnt/share/zjc/Kaggle_origin/%s/Dog_%d_interictal_segment_*.mat" % ( self.patient_name, self.patient_id)))
        ))
    def get_patient_name(self):
        """
        Get patient name
        """
        patient_list=GetPatientList(self.dataset_name)
        return patient_list[str(self.patient_id)]
    
    def get_seizure_time_list(self):
        """
        Get seizure time (second) in each BDF file
        for each patient, seizure times are stored in a list. [(start, end), (start, end),...]
        
        seizure_time_list={'1' : [4, 10, 16, 22],
                           '2' : [4, 10, 16, 22, 28, 34, 40],
                           '3' : [4, 10, 16, 22, 28, 34, 40, 46, 52, 58, 64, 70],
                           '4' : [4, 10, 16, 21, 30, 36, 41, 47, 53, 59, 65, 71, 77, 83],
                           '5' : [4, 10, 16, 22, 25],}
        """
        # seizure_time_list={'1' : [1, 7, 13, 19],
        #                    '2' : [1, 7, 13, 19, 25, 31, 37],
        #                    '3' : [1, 7, 13, 19, 25, 31, 37, 43, 49, 55, 61, 67],
        #                    '4' : [1, 7, 13, 21, 30, 36, 41, 47, 53, 59, 65, 71, 77, 83],
        #                    '5' : [1, 7, 13, 19, 25],}
        seizure_time_list={'1':[1,7,13,19],
                           '2':[1, 7, 13, 19, 25, 31, 37],
                           '3' : [1, 7, 13, 19, 25, 31, 37, 43, 49, 55, 61, 67],
                           '4' : [1,7,13,27,33,44,50,56,62,68,74,80,86,92],
                           '5' : [1, 7, 13, 19, 25],}
        for key in seizure_time_list.keys():
            seizure_time_list[key] = [time + 3 for time in seizure_time_list[key]]
        return seizure_time_list[str(self.patient_id)]
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'EDF preprocessing on CHB Dataset')
    #parser.add_argument('--data_path', type = str, default = "/data/GLH/CHB-MIT/CHB-MIT/", metavar = 'data path')
    parser.add_argument('--patient_id', type = int, default = 3, metavar = 'patient id')
    parser.add_argument('--target_preictal_interval', type = int, default = 60, metavar = 'how long we decide as preictal. Default set to 15 min') #in minute
    parser.add_argument('--dataset_name', type=str, default="Kaggle", metavar='dataset name : XUANWU / CHB / Kaggle')
    parser.add_argument('--seed', type = int, default = 19980702, metavar = 'random seed')
    parser.add_argument('--ch_num', type = int, default = 16, metavar = 'number of channel')
    parser.add_argument('--sfreq', type = int, default = 256, metavar = 'sample frequency')
    parser.add_argument('--window_length', type = int, default = 30, metavar = 'sliding window length')  #if stft : 30              else 5
    parser.add_argument('--preictal_step', type = int, default = 5, metavar = 'step of sliding window (second) for preictal data')  #if stft : 5               else 5
    parser.add_argument('--interictal_step', type = int, default = 30, metavar = 'step of sliding window (second) for interictal data') #if stft : 30             else 5
    parser.add_argument('--doing_STFT', type = bool, default = True, metavar = 'whether to do STFT') #if stft : True           else False
    parser.add_argument('--doing_lowpass_filter', type = bool, default = False, metavar = 'whether to do low pass filter') #if stft : False else True
    parser.add_argument('--title', type=str, default="60minpre", metavar='dataset name : XUANWU / CHB / Kaggle')
    args = parser.parse_args()
    setup_seed(args.seed)
    
    dataset_name=args.dataset_name
    title = args.title
    patient_list=GetPatientList(dataset_name)
    patient_id=args.patient_id
    patient_name=patient_list[str(patient_id)]
    sfreq=args.sfreq
    window_length=args.window_length
    preictal_step=args.preictal_step
    interictal_step=args.interictal_step
    doing_STFT=args.doing_STFT
    doing_lowpass_filter=args.doing_lowpass_filter
    target_preictal_interval=args.target_preictal_interval
    preictal_interval=args.target_preictal_interval*60 #seconds
    data_path=GetDataPath(dataset_name)
    #### change
    data_path = '/mnt/share/zjc/Kaggle_origin'
    save_dir = '/mnt/share/zjc/Kaggle_new_split30'
    ch_num = GetInputChannel(dataset_name, patient_id, args.ch_num)  # 18
    patient=KagglePatient(patient_id, dataset_name, data_path, ch_num, doing_lowpass_filter, target_preictal_interval)
    seizure_list = GetSeizureList(args.dataset_name)
    seizure_time_list=patient.get_seizure_time_list()
    print("\nprocessing patient : id {} {}\n ".format(patient_id, patient_name))
    
    #create dir to save results
    mkdir("%s/%s/%dmin_%dstep_%dch" % (save_dir, patient_name, target_preictal_interval, preictal_step, ch_num))
    mkdir("%s/%s/%dmin_%dstep_%dch_gauss" % (save_dir, patient_name, target_preictal_interval, preictal_step, ch_num))
    snr_list = [1,2,3,4,5,6,7,8]
    # preprocessing ictal and preictal data
    # for each edf file with seizure, a sliding window is used to transpose data into clips
    # i is the i-th seizure of each patient
    print("clipping preictal data")
    preictal_list_all = []
    NoSeizure = 0
    for i, edf_file in enumerate(patient._edf_files_seizure):
        
        num_segment = int((edf_file.get_filepath())[-8:-4])
        # print((edf_file.get_filepath())[-8:-4])
        print("num_segment=", num_segment)
        if num_segment in seizure_time_list:
            num_segment1 = num_segment + 1
            num_segment2 = num_segment + 2


            #preictal_interval is set to 900s(15min)
            preictal_interval=target_preictal_interval*60  # 15*60

            #load data from edf file
            print(edf_file.get_filepath())
            ant_data=edf_file.get_preprocessed_data()
            print("seizure {} \n".format(NoSeizure))

            supplement_filepath1="%s/%s/Dog_%d_preictal_segment_%s.mat" % (data_path, patient_name, patient_id, f"{num_segment1:04d}")
            supplement_filepath2="%s/%s/Dog_%d_preictal_segment_%s.mat" % (data_path, patient_name, patient_id, f"{num_segment2:04d}")
            supplement_file1=KaggleMatFile(supplement_filepath1, patient_id, ch_num, doing_lowpass_filter)
            supplement_file2=KaggleMatFile(supplement_filepath2, patient_id, ch_num, doing_lowpass_filter)
            print("load supplement file1 : {}".format(supplement_filepath1))
            print("load supplement file2 : {}".format(supplement_filepath2))
            ant_data1=supplement_file1.get_preprocessed_data()
            ant_data2=supplement_file2.get_preprocessed_data()
            
            ant_data=np.concatenate((ant_data, ant_data1))
            ant_data=np.concatenate((ant_data, ant_data2))
            print("new data {}".format(ant_data.shape))
        
            #process preictal data
            preictal_list=[]
            preictal_snr_list = []
            preictal_count=0
            while preictal_step * preictal_count + window_length <= 60*10*3:  # *3
                # print("preictal_step * preictal_count-----", preictal_step * preictal_count)
                preictal_start = preictal_step*preictal_count
                preictal_end = preictal_step*preictal_count + window_length
                preictal_data = ant_data[preictal_start * sfreq : preictal_end * sfreq]
                preictal_snr = wgn(preictal_data, np.random.choice(snr_list))

                #whether doing stft
                if doing_STFT:
                    preictal_data=getSpectral_STFT(preictal_data) #(22, 59, 114)
                    preictal_snr = getSpectral_STFT(preictal_snr)

    #                print("doing STFT {}".format(preictal_data.shape))
                preictal_list.append(preictal_data)
                preictal_snr_list.append(preictal_snr)
                preictal_count += 1
            preictal_list=np.array(preictal_list)
            preictal_snr_list = np.array(preictal_snr_list)
            
            #save preictal data to npy file
            if doing_STFT:
                np.save("%s/%s/%dmin_%dstep_%dch/preictal%d.npy" % (save_dir, patient_name, target_preictal_interval, preictal_step, ch_num, NoSeizure), preictal_list)
                np.save("%s/%s/%dmin_%dstep_%dch_gauss/preictal%d.npy" % (save_dir, patient_name, target_preictal_interval, preictal_step, ch_num, NoSeizure), preictal_snr_list)
            else:
                save_path="%s/%s/%dmin_%dstep_%dch/preictal%d.npy" % (save_dir, patient_name, target_preictal_interval, preictal_step, ch_num, NoSeizure)
                print("save to {}".format(save_path))
                np.save(save_path, preictal_list)
            NoSeizure += 1
            print("preictal shape {}".format(preictal_list.shape))

    
    
    # # concatenate preictal data
    # if len(preictal_list_all) == 0:
    #     preictal_list_all = preictal_list
    # else:
    #     preictal_list_all = np.vstack((preictal_list_all, preictal_list))
    # print("all prerictal shape: {}".format(preictal_list_all.shape))

    # np.random.shuffle(preictal_list_all)
    # count = 0
    # preictal_length = len(preictal_list_all) // len(seizure_list[str(patient_id)])
    # # print("preictal_length-----", preictal_length)
    # while (count + 1) * preictal_length <= len(preictal_list_all):
    #     preictal_data = preictal_list_all[count * preictal_length: (count + 1) * preictal_length]
    #     #save preictal data to npy file
    #     if doing_STFT:
    #         np.save("%s/%s/%dmin_%dstep_%dch/preictal%d.npy" % (data_path, patient_name, target_preictal_interval, preictal_step, ch_num, count), preictal_data)
    #     else:
    #         save_path="%s/%s/%dmin_%dstep_%dch/preictal%d.npy" % (data_path, patient_name, target_preictal_interval, preictal_step, ch_num, count)
    #         print("save to {}".format(save_path))
    #         np.save(save_path, preictal_data)
    #     print("preictal count {} : {}\n".format(count, preictal_data.shape))
    #     count += 1
    
    # preprocessing interictal data
    # for each edf file without seizure, a sliding window is used to transpose data into clips
    
    
    
    
    print("clipping interictal data")
    interictal_list_all=[]
    interictal_list_snr_all = []
    for i, edf_file in enumerate(patient._edf_files_unseizure):
        #load data from edf file
        print(edf_file.get_filepath())
        ant_data=edf_file.get_preprocessed_data()
        print("unseizure {} \n shape {}".format(i+1, ant_data.shape))
        
        #process interictal数据
        interictal_list=[]
        interictal_snr_list = []
        interictal_count=0
        while interictal_step*interictal_count + window_length <= 60*10:
            interictal_start = interictal_step * interictal_count
            interictal_end = interictal_step*interictal_count + window_length
            interictal_data = ant_data[interictal_start * sfreq:interictal_end * sfreq]
            interictal_snr = wgn(interictal_data, np.random.choice(snr_list))
            #whether doing stft
            if doing_STFT:
                interictal_data=getSpectral_STFT(interictal_data) #(22, 59, 114)
                interictal_snr = getSpectral_STFT(interictal_snr)
            interictal_list.append(interictal_data)
            interictal_snr_list.append(interictal_snr)
            interictal_count += 1
        interictal_list=np.array(interictal_list)
        interictal_snr_list = np.array(interictal_snr_list)
        
        print("interictal shape {}".format(interictal_list.shape))
        
        #concatenate interictal data
        if len(interictal_list_all)==0:
            interictal_list_all = interictal_list
            interictal_list_snr_all = interictal_snr_list
        else:
            interictal_list_all = np.vstack((interictal_list_all, interictal_list))
            interictal_list_snr_all = np.vstack((interictal_list_snr_all, interictal_snr_list))
        print("all interictal shape: {}".format(interictal_list_all.shape))
        print("all interictal snr shape: {}".format(interictal_list_snr_all.shape))
        
    #shuffle interictal data and divide into n gourps. n is the number of seizures of each patient
    np.random.shuffle(interictal_list_all)
    indices = np.random.permutation(interictal_list_all.shape[0])
    interictal_list_all = interictal_list_all[indices]
    interictal_list_snr_all = interictal_list_snr_all[indices]
    count=0
    interictal_length=len(interictal_list_all)//len(seizure_list[str(patient_id)])
    while (count+1) * interictal_length <= len(interictal_list_all):
        interictal_data=interictal_list_all[count * interictal_length : (count+1) * interictal_length]
        interictal_snr_data = interictal_list_snr_all[count * interictal_length : (count+1) * interictal_length]
        #save interictal data to npy file
        if doing_STFT:
            np.save("%s/%s/%dmin_%dstep_%dch/interictal%d.npy" % (save_dir, patient_name, target_preictal_interval, preictal_step, ch_num, count), interictal_data)
            np.save("%s/%s/%dmin_%dstep_%dch_gauss/interictal%d.npy" % (save_dir, patient_name, target_preictal_interval, preictal_step, ch_num, count), interictal_snr_data)
        else:
            save_path="%s/%s/%dmin_%dstep_%dch/interictal%d.npy" % (save_dir, patient_name, target_preictal_interval, preictal_step, ch_num, count)
            print("save to {}".format(save_path))
            np.save(save_path, interictal_data)
        print("interictal count {} : {}".format(count, interictal_data.shape))
        print("interictal snr count {}: {}".format(count, interictal_snr_data.shape))
        count+=1
    
