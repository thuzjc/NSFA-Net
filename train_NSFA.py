# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 14:37:03 2021

@author: phantom

# one example of training on CHB : 
python train.py --dataset_name=CHB --model_name=TA_STS_ConvNet --device_number=1 --patient_id=1 --step_preictal=1 --ch_num=18
"""

from __future__ import print_function
import os

import warnings
warnings.filterwarnings("ignore", message="Default grid_sample and affine_grid behavior has changed to align_corners=False since 1.3.0.")
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'
#os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3'

import copy
from datetime import datetime

import torch.utils.data
import argparse
import numpy as np
import torch.utils.data as Data
from torch.utils.data import RandomSampler
from matplotlib import pyplot as plt
from torch import nn

from EEG_dataset.dataset_train import trainDataset, trainSNRDataset
from EEG_dataset.dataset_test import testDataset
from EEG_trainer.montrainer_DP import Trainer
from EEG_utils.eeg_utils import GetPatientList, GetSeizureList, GetInputChannel,  GetModel, GetLoss, mkdir
import torch
import random

# 设置CUDA_VISIBLE_DEVICES环境变量
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
# os.environ['CUDA_VISIBLE_DEVICES'] = '5,6,7'

import sys

sys.path.append('.')
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP


def setup_seed(seed):
    '''
    set up seed for cuda and numpy
    '''
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False


def load_latest_file(dir_path, file_ext):
    """
    加载指定目录下匹配特定后缀的最新文件
    :param dir_path: 目录路径
    :param file_ext: 文件后缀
    :return: 最新文件的内容，如果没有匹配的文件则返回 None
    """

    print("dir_path:", dir_path, "file_ext:", file_ext)
    # 获取目录中所有匹配后缀的文件路径
    matching_files = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.endswith(file_ext)]
    # matching_files = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.endswith(file_ext) and f.startswith('aspp12_')]

    # 如果没有匹配的文件，则返回 None
    if not matching_files:
        return None

    # 按文件修改时间排序
    matching_files.sort(key=lambda x: os.path.getmtime(x))
    # 获取最新的文件路径
    latest_file = matching_files[-1]

    # 加载最新的文件
    with open(latest_file, 'r') as f:
        file_content = f

    return file_content, latest_file


def main(train, test, LOO, patient_name, args):
    '''
    training process
    '''
    patient_id = args.patient_id
    cuda = args.cuda
    device_number = args.device_number
    seed = args.seed
    ch_num = args.ch_num
    batch_size = args.batch_size
    model_name = args.model_name
    dataset_name = args.dataset_name
    checkpoint_dir = args.checkpoint_dir
    target_preictal_interval = args.target_preictal_interval
    transfer_learning = args.transfer_learning
    first = args.first
    domain_adaptation = args.domain_adaptation
    augmentation = False
    step_preictal = args.step_preictal
    balance = args.balance
    using_ictal = args.using_ictal
    data_dir = args.data_dir
    using_ictal = False
    # init_distributed_mode(args)

    # cuda and random seed
    cuda = cuda and torch.cuda.is_available()
    torch.cuda.set_device(device_number)
    print("set cuda device : ", device_number)
    setup_seed(seed)

    # dataset loader. training set and test set
    input_channel = GetInputChannel(dataset_name, patient_id, ch_num)
    train_set = trainDataset(dataset_name=dataset_name,n= train, ite=1, augmentation=augmentation, using_ictal=False,
                             scaler=balance,
                             patient_id=patient_id, patient_name=patient_name, ch_num=input_channel,
                             target_preictal_interval=target_preictal_interval, step_preictal=step_preictal, model_name=model_name, simple_mode = args.simple_mode, data_dir = args.data_dir)
    
    train_snr_set = trainSNRDataset(dataset_name=dataset_name, n=train, augmentation = augmentation, using_ictal = False,
                                    scaler = balance, patient_id = patient_id, patient_name = patient_name, ch_num=input_channel, 
                                    target_preictal_interval = target_preictal_interval, step_preictal = step_preictal, model_name = model_name,index = train_set.index, simple_mode = args.simple_mode, data_dir = args.data_dir)
    test_set = testDataset(dataset_name, test, using_ictal=using_ictal,
                           patient_id=patient_id, patient_name=patient_name, ch_num=input_channel,
                           target_preictal_interval=target_preictal_interval, step_preictal=step_preictal, data_dir = args.data_dir)
    # TODO PARA
    g = torch.Generator()
    g.manual_seed(seed)
    sampler = RandomSampler(train_set, generator = g)
    trainloader = Data.DataLoader(dataset=train_set, batch_size=batch_size, sampler = sampler, generator = g, shuffle=False, num_workers=16, pin_memory=True)
    testloader = Data.DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)
    trainsnrloader = Data.DataLoader(dataset=train_snr_set, batch_size = batch_size, sampler = sampler, generator = g, shuffle=False,num_workers = 16, pin_memory=True)
    # model
    model = GetModel(input_channel, device_number, model_name, dataset_name, patient_id)
    # model = TLDM768(num_classes=2, channel=GetInputChannel(dataset_name, patient_id, 0))

    device = torch.device(device_number)

    if first:
        print("1st")
    else:
        if model_name.startswith("MONSTB"):
            dir_path = '{}/model/{}/{}/{}/stft/'.format(checkpoint_dir, dataset_name, model_name, patient_id)
        elif model_name.startswith("MONSTL"):
            dir_path = '{}/model/{}/{}/{}/stftC/'.format(checkpoint_dir, dataset_name, model_name, patient_id)

        file_ext = 'patient{}_{}_step_preictal{}.pth'.format(patient_id, LOO, step_preictal)
        current_dict = model.state_dict()

        dir_path = "/share/home/zhanghan/code/SeizureDP/TA-STS-206/model/PreTbackbone/"
        file_ext = ".pth"

        _, file_content = load_latest_file(dir_path, file_ext)

        if file_content is None:
            print("No file found")
        else:
            state_dict = torch.load(file_content, map_location=device)
            ############### FIRST ###############
            # state_dict = {('swinB.' + k): v for k, v in state_dict.items() if (('swinB.' + k) in current_dict)}
            if dataset_name.startswith("XUANWU"):
                for k, v in copy.deepcopy(state_dict).items():
                    if "conv1x1" in k:
                        del state_dict[k]
            ############### FIRST ###############
            model = nn.DataParallel(model)
            print(model.load_state_dict(state_dict, strict=False))
            print(file_content + " Load model successfully!")

    # model = nn.DataParallel(model)


    # whether using transfer learning. load pre-trained model weights
    if transfer_learning:
        print("loading pre-trained model")
        if domain_adaptation:
            print("load Domain adaptation transfer weights")
            model.load_state_dict(torch.load(
                '{}/model/{}/{}/{}/stftsrc/patient{}_{}_step_preictal{}_source.pth'.format(checkpoint_dir, dataset_name,
                                                                                           model_name, patient_id,
                                                                                           patient_id, LOO,
                                                                                           step_preictal)))
        else:
            print("load transfer weights")
            model.load_state_dict(torch.load(
                '{}/model/{}/{}/{}/stftsrc/patient{}_step_preictal{}_source.pth'.format(checkpoint_dir, dataset_name,
                                                                                        model_name, patient_id,
                                                                                        patient_id, step_preictal)))

    # parameter number
    print('parameters:', sum(param.numel() for param in model.parameters() if param.requires_grad))

    # cuda
    if cuda:
        model = model.cuda(device)
        # TODO PARA
    print("Using cuda : {}".format(cuda))
    save_dir = f'{checkpoint_dir}/model/{dataset_name}/{model_name}/{patient_id}/stftC' 
    loss1 = "FL"
    loss2 = "FL"

    print("Loss : {}".format(loss1))
    print("Loss : {}".format(loss2))

    Loss1 = GetLoss(loss1)
    Loss2 = GetLoss(loss2)
    print('define loss function :', Loss1, Loss2)
    trainer = Trainer(model, Loss1, Loss2, trainloader,trainsnrloader, testloader, args, save_dir)
    print('define trainer ')

    # training
    print("Training...")
    train_state, train_acc_list, train_loss_list = trainer.train()

    fig, axs = plt.subplots(1, 2, figsize=(16, 8))
    axs[0].plot(train_loss_list, label='Train')
    axs[0].set_xlabel('Epochs')
    axs[0].set_ylabel('Loss')
    axs[0].legend()

    axs[1].plot(train_acc_list, label='Train')
    axs[1].set_xlabel('Epochs')
    axs[1].set_ylabel('Accuracy')
    axs[1].legend()
    plt.tight_layout()
    if model_name.startswith("MONSTB"):
        mkdir("{}/model/{}/{}/{}/stft/".format(checkpoint_dir, dataset_name, model_name, patient_id))
        fig.savefig("{}/model/{}/{}/{}/stft/{}_{}_patient{}_{}_loss_acc.png".format(checkpoint_dir, dataset_name, model_name, patient_id, first,
                                                                                    str(datetime.now().strftime("%Y%m%d%H%M%S")), patient_id, LOO))
    plt.close()


    if model_name.startswith("MONSTB"):


        ###
        # torch.save(model.state_dict(),
        #            "/share/home/zhanghan/code/SeizureDP/TA-STS-206/model/PreTbackbone/k_{}_{}_patient{}_{}_step_preictal{}.pth".format(first,
        #                                                                                                                                str(datetime.now().strftime(
        #                                                                                                                                    "%Y%m%d%H%M%S")),
        #                                                                                                                                patient_id, LOO,
        #                                                                                                                                step_preictal))
        mkdir("{}/model/{}/{}/{}/stftC/".format(checkpoint_dir, dataset_name, model_name, patient_id))
        time = str(datetime.now().strftime("%Y%m%d%H%M%S"))
        torch.save(train_state,
                   "{}/model/{}/{}/{}/stftC/{}_{}_patient{}_{}_step_preictal{}.pth".format(checkpoint_dir, dataset_name,
                                                                                           model_name, patient_id,
                                                                                           first,
                                                                                           time,
                                                                                           patient_id, LOO,
                                                                                           step_preictal))
        print("save model to {}/model/{}/{}/{}/stftC/{}_{}_patient{}_{}_step_preictal{}.pth".format(checkpoint_dir, dataset_name,
                                                                                           model_name, patient_id,
                                                                                           first,
                                                                                           time,
                                                                                           patient_id, LOO,
                                                                                           step_preictal))
    elif model_name.startswith("NSFA"):
        mkdir("{}/model/{}/{}/{}/stftC/".format(checkpoint_dir, dataset_name, model_name, patient_id))
        time = str(datetime.now().strftime("%Y%m%d%H%M%S"))
        print("save model to {}/model/{}/{}/{}/stftC/{}_{}_patient{}_{}_step_preictal{}.pth".format(checkpoint_dir, dataset_name,
                                                                                           model_name, patient_id,
                                                                                           first,
                                                                                           time,
                                                                                           patient_id, LOO,
                                                                                           step_preictal))
        torch.save(train_state,
                   "{}/model/{}/{}/{}/stftC/{}_{}_patient{}_{}_step_preictal{}.pth".format(checkpoint_dir, dataset_name,
                                                                                           model_name, patient_id,
                                                                                           first,
                                                                                           time,
                                                                                           patient_id, LOO,
                                                                                           step_preictal))
    elif model_name.startswith("MONSTL"):
        mkdir("{}/model/{}/{}/{}/stftC/".format(checkpoint_dir, dataset_name, model_name, patient_id))
        torch.save(train_state,
                   "{}/model/{}/{}/{}/stftC/{}_{}_patient{}_{}_step_preictal{}.pth".format(checkpoint_dir, dataset_name,
                                                                                           model_name, patient_id,
                                                                                           first,
                                                                                           str(datetime.now().strftime(
                                                                                               "%Y%m%d%H%M%S")),
                                                                                           patient_id, LOO,
                                                                                           step_preictal)) #LOO: 测试id

    elif model_name.startswith("CNN"):
        mkdir("{}/model/{}/{}/{}/stft/".format(checkpoint_dir, dataset_name, model_name, patient_id))
        torch.save(train_state,
                   "{}/model/{}/{}/{}/stft/cnn_{}_{}_patient{}_{}_step_preictal{}.pth".format(checkpoint_dir, dataset_name,
                                                                                                model_name, patient_id, first,
                                                                                                str(datetime.now().strftime(
                                                                                                    "%Y%m%d%H%M%S")),
                                                                                                patient_id, LOO,
                                                                                                step_preictal))

    return 0


if __name__ == "__main__":
    # Parse the JSON arguments
    parser = argparse.ArgumentParser(description='Seizure predicting on Xuanwu/CHB Dataset')
    parser.add_argument('--patient_id', type=int, default=1, metavar='patient id')
    parser.add_argument('--device_number', type=int, default=6, metavar='CUDA device number')
    parser.add_argument('--ch_num', type=int, default=18, metavar='number of channel')
    parser.add_argument('--step_preictal', type=int, default=5)  # 窗口滑动步长
    parser.add_argument('--seed', type=int, default=42, metavar='random seed')
    parser.add_argument('--cuda', type=bool, default=True, metavar="whether to use cuda")
    parser.add_argument("--using_ictal", type=int, default=0,
                        metavar='whether to use ictal data , default=1 use ictal data')
    parser.add_argument('--batch_size', type=int, default=32, metavar='batchsize')
    parser.add_argument('--learning_rate', type=float, default=0.0001, metavar='learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, metavar='N')
    parser.add_argument('--num_epochs', type=int, default=100, metavar='number of epochs')
    parser.add_argument('--early_stop_patience', type=int, default=5, metavar='N')
    parser.add_argument('--model_name', type=str, default='NSFA', metavar='model name')
    parser.add_argument('--dataset_name', type=str, default='KAGGLE', metavar='dataset name')
    parser.add_argument('--first', type=bool, default=True, metavar='whether it is the first training')
    parser.add_argument('--transfer_learning', type=bool, default=False, metavar='whether to use transfer learning')
    parser.add_argument('--domain_adaptation', type=bool, default=False, metavar='whether to use domain adaptation')
    parser.add_argument('--target_preictal_interval', type=int, default=60,)
    parser.add_argument("--checkpoint_dir", type=str, default = './checkpoint', metavar='checkpoint directory')
    parser.add_argument('--TestWhenTraining', type=int, default = 1)
    parser.add_argument('--part', type=int, nargs = '+', required =False, metavar='part of seizure to test')
    parser.add_argument('--balance', type = bool, default= False, metavar='whether to balance the dataset')
    parser.add_argument('--simple_mode', type = bool, default = False, metavar = 'whether to use simple mode')
    parser.add_argument('--auto_threshold', type=bool, default = True, metavar = 'whether to use auto_threshold when test')
    parser.add_argument('--data_dir', type=str, default = '../Kaggle_new_split', metavar='data directory')
    args = parser.parse_args()
   
    # get patient list, patient id, seizure list, etc
    patient_list = GetPatientList(args.dataset_name)
    patient_id = args.patient_id
    patient_name = patient_list[str(patient_id)]
    seizure_list = GetSeizureList(args.dataset_name)
    seizure_count = len(seizure_list[str(patient_id)])

    args.checkpoint_dir = os.getcwd()
    print("dataset : {} \npatient {} \nseizure count : {}\n".format(args.dataset_name, patient_id, seizure_count))
    print(vars(args))
    # LOOCV for each patient.
    for LOO in seizure_list[str(patient_id)]:
        print(f'LOO/test: {LOO}, patient: {patient_name}, dataset: {args.dataset_name}, model: {args.model_name}')
        # if LOO == 1:
        #     continue
        # if LOO == 0:
        #     continue
        # if LOO == 2:
        #     continue
        # if LOO == 3:
        #     continue
        # if LOO == 4:
        #     continue
        test = LOO
        if  args.part!=None and test not in args.part:
            continue
        args.now_part = test
        train = list(set(seizure_list[str(patient_id)]) - set([test]))
        main(train, test, LOO, patient_name, args)
