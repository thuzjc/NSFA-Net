# using GLH's data
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 11:11:39 2021

@author: phantom
"""
import torch
from torch.utils.data import Dataset
import numpy as np
from EEG_utils.eeg_utils import GetDataPath, GetDataType


class trainDataset(Dataset):
    def __init__(self, model_name="CNN", dataset_name="CHB", n=[], ite=1, augmentation=1, using_ictal=1,
                 scaler=0, patient_id=None, patient_name=None, ch_num=1, target_preictal_interval=15, step_preictal=5, simple_mode = False, data_dir = None):
        data, label = [], []
        self.scaler = scaler  # whether to balance preictal data and interictal data
        self.augmentation = augmentation  # whether using data augmentation
        self.using_ictal = using_ictal  # whether using ictal data
        self.data_path = GetDataPath(dataset_name)  # data path
        if dataset_name == 'KAGGLE' and data_dir == None:
            self.data_path = '/mnt/share/zjc/Kaggle_new_split'
        elif dataset_name == 'CHB' and data_dir == None:
            self.data_path = '/mnt/share/zjc/CHBMIT_new_split'
        else:
            self.data_path = data_dir
        self.data_type = GetDataType(dataset_name)  # EEG/IEEG
        self.patient_id = patient_id  # patient id
        self.patient_name = patient_name  # patient name
        self.ch_num = ch_num  # number of channels
        self.target_preictal_interval = target_preictal_interval  # how long we decide as preictal. Default set to 15 min
        self.step_preictal = step_preictal  # step of sliding window
        self.index = []
        self.simple_mode = simple_mode
        # LOOCV for n times for each patient. n is the number of seizures
        for i in n:
            # print(self.data_path, self.patient_name, self.target_preictal_interval, self.step_preictal, self.ch_num, i)
            preIctal = np.load(("%s/%s/%dmin_%dstep_%dch/preictal%d.npy" % (
                                    self.data_path, self.patient_name, self.target_preictal_interval, self.step_preictal, self.ch_num, i)))
                
            interIctal = np.load(("%s/%s/%dmin_%dstep_%dch/interictal%d.npy" % (
                                    self.data_path, self.patient_name, self.target_preictal_interval, self.step_preictal, self.ch_num, i)))
            
            if len(preIctal.shape) == 3:
                preIctal = preIctal.transpose(0, 2, 1)  # (180, 19, 1280)
                # Ictal = Ictal.transpose(0, 2, 1)  # (38, 19, 1280)
                interIctal = interIctal.transpose(0, 2, 1)

            if self.augmentation == 1:
                print("doing augmentation")
                temp = []
                indT = np.arange(len(preIctal) * 2)  # (428,)
                for _ in range(ite):
                    tmp = np.concatenate(np.split(preIctal, 2, -1), 0)  # (428, 19, 640)
                    np.random.shuffle(indT)
                    tmp = tmp[indT]
                    tmp = np.concatenate(np.split(tmp, 2, 0), -1)  # (214, 19, 1280)
                    temp.append(tmp)
                temp.append(preIctal)
                temp = np.concatenate(temp, 0)
                preIctal = temp  # (428, 19, 1280)

            ind = np.arange(0, len(interIctal))
            np.random.shuffle(ind)

            if self.scaler == 1 and len(preIctal) <len(interIctal):
                print("-----------pre------------")
                data.append(preIctal)
                data.append(interIctal[ind[:int(self.scaler * len(preIctal))], :, :])
                label.append(np.ones((preIctal.shape[0], 1)))
                label.append(np.zeros((int(self.scaler * len(preIctal)), 1)))
                self.index.append(ind[:int(self.scaler*len(preIctal))])
            elif self.scaler ==1 and len(preIctal) >len(interIctal) and (model_name.startswith("CNN") or model_name.startswith("VIT") or model_name.startswith("ResNet")):
                print("-----------inter------------")
                data.append(preIctal[ind[:int(self.scaler * len(interIctal))], :, :])
                data.append(interIctal)
                label.append(np.ones((int(self.scaler * len(interIctal)), 1)))
                label.append(np.zeros((interIctal.shape[0], 1)))
            else:
                data.append(preIctal)
                data.append(interIctal)
                label.append(np.ones((preIctal.shape[0], 1)))
                label.append(np.zeros((interIctal.shape[0], 1)))
            print('seizure {} : preictal {} | interIctal {}'.format(i, preIctal.shape, interIctal.shape))
            if self.simple_mode:
                break
        # numpy to torch
        # data, label = np.array(data), np.array(label)
        data, label = np.concatenate(data, 0), np.concatenate(label, 0)
        if len(preIctal.shape) == 3:
            data = data[:, np.newaxis, :, :].astype('float32')
        # elif len(preIctal.shape) == 4:  # spectralCNN
        #     data = data[:, np.newaxis, :, :, :].astype('float32')
        label = label.astype('int64')
        self.x_data = torch.from_numpy(data)  # ([2592, 1, 19, 1280])
        self.y_data = torch.from_numpy(label)  # ([2592, 1])
        self.len = data.shape[0]
        print("Target aug: {} Dataset : {} {}".format(str(augmentation), self.x_data.shape, self.y_data.shape))
        print("preIctal {} | interIctal {}\n".format(sum(np.array(label) == 1), sum(np.array(label) == 0)))

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len
    
    def get_index(self):
        return self.index


class trainSNRDataset(Dataset):
    def __init__(self, model_name="CNN", dataset_name="CHB", n=[], ite=1, augmentation=1, using_ictal=1,
                 scaler=0, patient_id=None, patient_name=None, ch_num=1, target_preictal_interval=15, step_preictal=5, index = None, simple_mode = False, data_dir = None):
        data, label = [], []
        self.scaler = scaler  # whether to balance preictal data and interictal data
        self.augmentation = augmentation  # whether using data augmentation
        self.using_ictal = using_ictal  # whether using ictal data
        self.data_path = GetDataPath(dataset_name)  # data path
        if dataset_name == 'KAGGLE' and data_dir == None:
            self.data_path = '/mnt/share/zjc/Kaggle_new_split'
        elif dataset_name == 'CHB' and data_dir == None:
            self.data_path = '/mnt/share/zjc/CHBMIT_new_split'
        else:
            self.data_path = data_dir
        self.data_type = GetDataType(dataset_name)  # EEG/IEEG
        self.patient_id = patient_id  # patient id
        self.patient_name = patient_name  # patient name
        self.ch_num = ch_num  # number of channels
        self.target_preictal_interval = target_preictal_interval  # how long we decide as preictal. Default set to 15 min
        self.step_preictal = step_preictal  # step of sliding window
        self.simple_mode = simple_mode
        # LOOCV for n times for each patient. n is the number of seizures
        for iind, i in enumerate(n):
            preIctal = np.load(("%s/%s/%dmin_%dstep_%dch_gauss/preictal%d.npy" % (
                                    self.data_path, self.patient_name, self.target_preictal_interval, self.step_preictal, self.ch_num, i)))
                
            interIctal = np.load(("%s/%s/%dmin_%dstep_%dch_gauss/interictal%d.npy" % (
                                    self.data_path, self.patient_name, self.target_preictal_interval, self.step_preictal, self.ch_num, i)))
            
            
           

            # print("preIctal=", preIctal.shape)
            # print("interIctal=", interIctal.shape)
            # print("Ictal=", Ictal.shape)
            # shape transpose
            if len(preIctal.shape) == 3:
                preIctal = preIctal.transpose(0, 2, 1)  # (180, 19, 1280)
                # Ictal = Ictal.transpose(0, 2, 1)  # (38, 19, 1280)
                interIctal = interIctal.transpose(0, 2, 1)

            # whether to use ictal data for training
            # if self.using_ictal == 1:
            #     print("using ictal data")
            #     preIctal = np.concatenate((preIctal, Ictal), 0)
                # print("concatenated preictal {}".format(preIctal.shape))#(38, 19, 1280)

            # data augmentation
            if self.augmentation == 1:
                print("doing augmentation")
                temp = []
                indT = np.arange(len(preIctal) * 2)  # (428,)
                for _ in range(ite):
                    tmp = np.concatenate(np.split(preIctal, 2, -1), 0)  # (428, 19, 640)
                    np.random.shuffle(indT)
                    tmp = tmp[indT]
                    tmp = np.concatenate(np.split(tmp, 2, 0), -1)  # (214, 19, 1280)
                    temp.append(tmp)
                temp.append(preIctal)
                temp = np.concatenate(temp, 0)
                preIctal = temp  # (428, 19, 1280)
            # data.append(preIctal)

            # whether to balance interictal and preictal data
            ind = np.arange(0, len(interIctal))
            np.random.shuffle(ind)
            if self.scaler == 1 and len(preIctal) <len(interIctal):
                print("-----------pre------------")
                if index != None:
                    ind = index[iind]
                    # print(ind)
                data.append(preIctal)
                data.append(interIctal[ind, :, :])
                label.append(np.ones((preIctal.shape[0], 1)))
                label.append(np.zeros((len(ind), 1)))
                print(data[0].shape, data[1].shape)
            elif self.scaler ==1 and len(preIctal) >len(interIctal) and (model_name.startswith("CNN") or model_name.startswith("VIT") or model_name.startswith("ResNet")):
                print("-----------inter------------")
                data.append(preIctal[ind[:int(self.scaler * len(interIctal))], :, :])
                data.append(interIctal)
                label.append(np.ones((int(self.scaler * len(interIctal)), 1)))
                label.append(np.zeros((interIctal.shape[0], 1)))

            # elif self.scaler ==1 and len(preIctal) >len(interIctal):
            #     print("interIctal is not enough!")
            #     # print(interIctal.shape)
            #     # print("ind",ind.shape)
            #     if self.patient_id in (8, 14, 16):
            #         interIctal=np.tile(interIctal,(3,1,1,1))    
            #         ind=np.tile(ind,(3)) 
            #     else:
            #         interIctal=np.tile(interIctal,(2,1,1,1))    
            #         ind=np.tile(ind,(2))    
            #     # print(interIctal.shape)
            #     # print(len(data))
            #     data.append(interIctal[ind[:int(self.scaler * len(preIctal))], :, :])
            #     # print(len(data))
            #     label.append(np.ones((preIctal.shape[0], 1)))
            #     label.append(np.zeros((int(self.scaler * len(preIctal)), 1)))
            else:
                data.append(preIctal)
                data.append(interIctal)
                label.append(np.ones((preIctal.shape[0], 1)))
                label.append(np.zeros((interIctal.shape[0], 1)))
            print('gauss seizure {} : preictal {} | interIctal {}'.format(i, preIctal.shape, interIctal.shape))
            if self.simple_mode:
                break
        # numpy to torch
        # data, label = np.array(data), np.array(label)
        data, label = np.concatenate(data, 0), np.concatenate(label, 0)
        if len(preIctal.shape) == 3:
            data = data[:, np.newaxis, :, :].astype('float32')
        # elif len(preIctal.shape) == 4:  # spectralCNN
        #     data = data[:, np.newaxis, :, :, :].astype('float32')
        label = label.astype('int64')
        self.x_data = torch.from_numpy(data)  # ([2592, 1, 19, 1280])
        self.y_data = torch.from_numpy(label)  # ([2592, 1])
        self.len = data.shape[0]
        print("Gauss Target aug : {} Dataset : {} {}".format(str(augmentation), self.x_data.shape, self.y_data.shape))
        print("Gauss preIctal {} | interIctal {}\n".format(sum(np.array(label) == 1), sum(np.array(label) == 0)))

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


class trainDataset2(Dataset):
    def __init__(self, dataset_name="CHB", n=[], ite=1, augmentation=1, using_ictal=1,
                 scaler=0, patient_id=None, patient_name=None, ch_num=1, target_preictal_interval=15, step_preictal=5):
        data, label = [], []
        self.scaler = scaler  # whether to balance preictal data and interictal data
        self.augmentation = augmentation  # whether using data augmentation
        self.using_ictal = using_ictal  # whether using ictal data
        self.data_path = GetDataPath(dataset_name)  # data path
        self.data_type = GetDataType(dataset_name)  # EEG/IEEG
        self.patient_id = patient_id  # patient id
        self.patient_name = patient_name  # patient name
        self.ch_num = ch_num  # number of channels
        self.target_preictal_interval = target_preictal_interval  # how long we decide as preictal. Default set to 15 min
        self.step_preictal = step_preictal  # step of sliding window

        # LOOCV for n times for each patient. n is the number of seizures
        for i in n:
            # data loading from npy files
            preIctal = np.load(("%s/%s/%dmin_%dstep_%dch_cwt/preictal%d.npy" % (
            self.data_path, self.patient_name, self.target_preictal_interval, self.step_preictal, self.ch_num, i)))
            interIctal = np.load(("%s/%s/%dmin_%dstep_%dch_cwt/interictal%d.npy" % (
            self.data_path, self.patient_name, self.target_preictal_interval, self.step_preictal, self.ch_num, i)))
            Ictal = np.load(("%s/%s/%dmin_%dstep_%dch_cwt/ictal%d.npy" % (
            self.data_path, self.patient_name, self.target_preictal_interval, self.step_preictal, self.ch_num, i)))
            # preIctal = np.load(("%s/%s/6_fold/preictal%d.npy" % (self.data_path, self.patient_name, i)))
            # interIctal = np.load(("%s/%s/6_fold/interictal%d.npy" % (self.data_path, self.patient_name, i)))
            # Ictal = np.load(("%s/%s/6_fold/ictal%d.npy" % (self.data_path, self.patient_name, i)))

            # print("preIctal=", preIctal.shape)
            # print("interIctal=", interIctal.shape)
            # print("Ictal=", Ictal.shape)
            # shape transpose
            if len(preIctal.shape) == 3:
                preIctal = preIctal.transpose(0, 2, 1)  # (180, 19, 1280)
                Ictal = Ictal.transpose(0, 2, 1)  # (38, 19, 1280)
                interIctal = interIctal.transpose(0, 2, 1)

            # whether to use ictal data for training
            if self.using_ictal == 1:
                print("using ictal data")
                preIctal = np.concatenate((preIctal, Ictal), 0)
                # print("concatenated preictal {}".format(preIctal.shape))#(38, 19, 1280)

            # data augmentation
            if self.augmentation == 1:
                print("doing augmentation")
                temp = []
                indT = np.arange(len(preIctal) * 2)  # (428,)
                for _ in range(ite):
                    tmp = np.concatenate(np.split(preIctal, 2, -1), 0)  # (428, 19, 640)
                    np.random.shuffle(indT)
                    tmp = tmp[indT]
                    tmp = np.concatenate(np.split(tmp, 2, 0), -1)  # (214, 19, 1280)
                    temp.append(tmp)
                temp.append(preIctal)
                temp = np.concatenate(temp, 0)
                preIctal = temp  # (428, 19, 1280)
            data.append(preIctal)

            # whether to balance interictal and preictal data
            ind = np.arange(0, len(interIctal))
            np.random.shuffle(ind)

            if self.scaler == 1 and len(preIctal) <len(interIctal):
                data.append(interIctal[ind[:int(self.scaler * len(preIctal))], :, :])
                label.append(np.ones((preIctal.shape[0], 1)))
                label.append(np.zeros((int(self.scaler * len(preIctal)), 1)))
            elif self.scaler ==1 and len(preIctal) >len(interIctal):
                # print(interIctal.shape)
                # print("ind",ind.shape)
                if self.patient_id in (8, 14, 16):
                    interIctal=np.tile(interIctal,(3,1,1,1))    
                    ind=np.tile(ind,(3)) 
                else:
                    interIctal=np.tile(interIctal,(2,1,1,1))    
                    ind=np.tile(ind,(2))    
                # print(interIctal.shape)
                # print(len(data))
                data.append(interIctal[ind[:int(self.scaler * len(preIctal))], :, :])
                # print(len(data))
                label.append(np.ones((preIctal.shape[0], 1)))
                label.append(np.zeros((int(self.scaler * len(preIctal)), 1)))
            else:
                data.append(interIctal)
                label.append(np.ones((preIctal.shape[0], 1)))
                label.append(np.zeros((interIctal.shape[0], 1)))
            print('seizure {} : preictal {} | Ictal {} | interIctal {}'.format(i, preIctal.shape, Ictal.shape,
                                                                               interIctal.shape))

        # numpy to torch
        data, label = np.array(data), np.array(label)
        data, label = np.concatenate(data, 0), np.concatenate(label, 0)
        if len(preIctal.shape) == 3:
            data = data[:, np.newaxis, :, :].astype('float32')
        # elif len(preIctal.shape) == 4:  # spectralCNN
        #     data = data[:, np.newaxis, :, :, :].astype('float32')
        label = label.astype('int64')
        self.x_data = torch.from_numpy(data)  # ([2592, 1, 19, 1280])
        self.y_data = torch.from_numpy(label)  # ([2592, 1])
        self.len = data.shape[0]
        print("Target {} Dataset : {} {}".format(str(augmentation), self.x_data.shape, self.y_data.shape))
        print("preIctal {} | interIctal {}\n".format(sum(np.array(label) == 1), sum(np.array(label) == 0)))

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len

'''
class ThirdDataset(Dataset):
    def __init__(self, dataset_name="CHB", n=[], ite=1, augmentation=1, using_ictal=1,
                 scaler=0, patient_id=None, patient_name=None, ch_num=1, target_preictal_interval=15, step_preictal=5):
        data, label, third = [], [], []
        self.scaler = scaler  # whether to balance preictal data and interictal data
        self.augmentation = augmentation  # whether using data augmentation
        self.using_ictal = using_ictal  # whether using ictal data
        self.data_path = GetDataPath(dataset_name)  # data path
        self.data_type = GetDataType(dataset_name)  # EEG/IEEG
        self.patient_id = patient_id  # patient id
        self.patient_name = patient_name  # patient name
        self.ch_num = ch_num  # number of channels
        self.target_preictal_interval = target_preictal_interval  # how long we decide as preictal. Default set to 15 min
        self.step_preictal = step_preictal  # step of sliding window

        # LOOCV for n times for each patient. n is the number of seizures
        for i in n:
            # data loading from npy files
            thirdClass = np.load(("%s/%s/%dmin_%dstep_%dch_3rdClass/interictal%d.npy" % (
            self.data_path, self.patient_name, self.target_preictal_interval, self.step_preictal, self.ch_num, i)))
            if self.patient_id in (8, 14, 16, 23):
                preIctal = np.load(("%s/%s/%dmin_%dstep_%dch/preictal%d.npy" % (
                self.data_path, self.patient_name, self.target_preictal_interval, self.step_preictal, self.ch_num, i)))
            else:
                preIctal = np.load(("%s/%s/%dmin_%dstep_%dch_gauss8/preictal%d.npy" % (
                self.data_path, self.patient_name, self.target_preictal_interval, self.step_preictal, self.ch_num, i)))
            if self.patient_id in (2, 7, 9, 11, 21, 22):  # (1, 8, 14, 16, 20, 23)
                interIctal = np.load(("%s/%s/%dmin_%dstep_%dch/interictal%d.npy" % (
                self.data_path, self.patient_name, self.target_preictal_interval, self.step_preictal, self.ch_num, i)))
            else:
                interIctal = np.load(("%s/%s/%dmin_%dstep_%dch_gauss8/interictal%d.npy" % (
                self.data_path, self.patient_name, self.target_preictal_interval, self.step_preictal, self.ch_num, i)))

            # print("preIctal=", preIctal.shape)
            # print("interIctal=", interIctal.shape)
            # print("Ictal=", Ictal.shape)
            # shape transpose
            if len(preIctal.shape) == 3:
                preIctal = preIctal.transpose(0, 2, 1)  # (180, 19, 1280)
                interIctal = interIctal.transpose(0, 2, 1)
                thirdClass = thirdClass.transpose(0, 2, 1)

            data.append(preIctal)
            # whether to balance interictal and preictal data
            ind = np.arange(0, len(interIctal))
            ind1 = np.arange(0, len(thirdClass))
            np.random.shuffle(ind)
            np.random.shuffle(ind1)

            if self.scaler == 1 and len(preIctal) <len(thirdClass):
                third.append(thirdClass[ind1[:int(self.scaler * len(preIctal))], :, :])
            else:
                third.append(thirdClass)

            if self.scaler == 1 and len(preIctal) <len(interIctal):
                data.append(interIctal[ind[:int(self.scaler * len(preIctal))], :, :])
                label.append(np.ones((preIctal.shape[0], 1)))
                label.append(np.zeros((int(self.scaler * len(preIctal)), 1)))
            else:
                data.append(interIctal)
                label.append(np.ones((preIctal.shape[0], 1)))
                label.append(np.zeros((interIctal.shape[0], 1)))
            print('seizure {} : preictal {} | interIctal {} | thirdClass {}'.format(i, preIctal.shape, interIctal.shape,
                                                                               thirdClass.shape))

        # numpy to torch
        data, label, third = np.array(data), np.array(label), np.array(third)
        data, label, third = np.concatenate(data, 0), np.concatenate(label, 0), np.concatenate(third, 0)
        if len(preIctal.shape) == 3:
            data = data[:, np.newaxis, :, :].astype('float32')
            third = third[:, np.newaxis, :, :].astype('float32')
        # elif len(preIctal.shape) == 4:  # spectralCNN
        #     data = data[:, np.newaxis, :, :, :].astype('float32')
        label = label.astype('int64')
        self.x_data = torch.from_numpy(data)  # ([2592, 1, 19, 1280])
        self.y_data = torch.from_numpy(label)  # ([2592, 1])
        self.z_data = torch.from_numpy(third)
        self.len = data.shape[0]
        print("Target {} Dataset : {} {}".format(str(augmentation), self.x_data.shape, self.y_data.shape))
        print("preIctal {} | interIctal{} | thirdClass[{}] | len[{}]\n".format(sum(np.array(label) == 1), sum(np.array(label) == 0), third.shape[0], data.shape[0]))

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index], self.z_data[index]

    def __len__(self):
        return self.len
'''

class ThirdDataset(Dataset):
    def __init__(self, dataset_name="CHB", n=[], ite=1, augmentation=1, using_ictal=1,
                 scaler=0, patient_id=None, patient_name=None, ch_num=1, target_preictal_interval=15, step_preictal=5):
        data, label = [], []
        self.scaler = scaler  # whether to balance preictal data and interictal data
        self.augmentation = augmentation  # whether using data augmentation
        self.using_ictal = using_ictal  # whether using ictal data
        self.data_path = GetDataPath(dataset_name)  # data path
        self.data_type = GetDataType(dataset_name)  # EEG/IEEG
        self.patient_id = patient_id  # patient id
        self.patient_name = patient_name  # patient name
        self.ch_num = ch_num  # number of channels
        self.target_preictal_interval = target_preictal_interval  # how long we decide as preictal. Default set to 15 min
        self.step_preictal = step_preictal  # step of sliding window

        # LOOCV for n times for each patient. n is the number of seizures
        for i in n:
            # data loading from npy files
            thirdClass = np.load(("%s/%s/%dmin_%dstep_%dch_3rdClass/interictal%d.npy" % (
            self.data_path, self.patient_name, self.target_preictal_interval, self.step_preictal, self.ch_num, i)))
            # if self.patient_id in (8, 14, 16, 23):
            #     preIctal = np.load(("%s/%s/%dmin_%dstep_%dch/preictal%d.npy" % (
            #     self.data_path, self.patient_name, self.target_preictal_interval, self.step_preictal, self.ch_num, i)))
            # else:
            #     preIctal = np.load(("%s/%s/%dmin_%dstep_%dch_gauss8/preictal%d.npy" % (
            #     self.data_path, self.patient_name, self.target_preictal_interval, self.step_preictal, self.ch_num, i)))
            
            if self.patient_id in (3, 4):
                preIctal = np.load(("%s/%s/%dmin_%dstep_%dch/preictal%d.npy" % (
                self.data_path, self.patient_name, self.target_preictal_interval, self.step_preictal, self.ch_num, i)))
            else:
                preIctal = np.load(("%s/%s/%dmin_%dstep_%dch_gauss8/preictal%d.npy" % (
                self.data_path, self.patient_name, self.target_preictal_interval, self.step_preictal, self.ch_num, i)))

            # print("preIctal=", preIctal.shape)
            # print("interIctal=", interIctal.shape)
            # print("Ictal=", Ictal.shape)
            # shape transpose
            if len(preIctal.shape) == 3:
                preIctal = preIctal.transpose(0, 2, 1)  # (180, 19, 1280)
                thirdClass = thirdClass.transpose(0, 2, 1)

            # whether to balance interictal and preictal data
            ind1 = np.arange(0, len(thirdClass))
            np.random.shuffle(ind1)

            if self.scaler == 1 and len(preIctal) <len(thirdClass):
                data.append(thirdClass[ind1[:int(self.scaler * len(preIctal))], :, :])
                label.append(np.zeros((int(self.scaler * len(preIctal)), 1)))
            else:
                data.append(thirdClass)
                label.append(np.zeros((thirdClass.shape[0], 1)))
            print('seizure {} : preictal {} | thirdClass {}'.format(i, preIctal.shape, thirdClass.shape))

        # numpy to torch
        data, label = np.array(data), np.array(label)
        data, label = np.concatenate(data, 0), np.concatenate(label, 0)
        if len(preIctal.shape) == 3:
            data = data[:, np.newaxis, :, :].astype('float32')
        # elif len(preIctal.shape) == 4:  # spectralCNN
        #     data = data[:, np.newaxis, :, :, :].astype('float32')
        label = label.astype('int64')
        self.x_data = torch.from_numpy(data)  # ([2592, 1, 19, 1280])
        self.len = data.shape[0]
        print("thirdClass[{}] | len[{}]\n".format(data.shape[0], data.shape[0]))

    def __getitem__(self, index):
        return self.x_data[index]

    def __len__(self):
        return self.len
