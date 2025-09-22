# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 21:36:15 2021

@author: phantom
"""

from torch.autograd import Variable
from torch import optim
from itertools import cycle
import torch
import shutil
import copy
import time
import math
import numpy as np
from tqdm import tqdm
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

class Trainer:
    def __init__(self, model, loss, CE_Loss,  ME_loss, source_train_loader, source_third_loader, target_test_loader, args):
        self.model = model
        self.args = args
        self.args.start_epoch = 0
        self.source_train_loader = source_train_loader
        self.source_third_loader = source_third_loader
        # self.source_train_loader2 = source_train_loader2
        self.target_test_loader = target_test_loader
        
        # Loss function and Optimizer
        self.loss = loss # MMD loss
        self.CE_Loss = CE_Loss # CrossEntropyLoss
        self.ME_loss = ME_loss # Max Entropy Loss
        self.optimizer = self.get_optimizer()#Adam
        self.schedular = torch.optim.lr_scheduler.StepLR(self.optimizer, self.args.steplr, gamma = 0.1, last_epoch = -1)
        self.best_model_params = copy.deepcopy(model.state_dict())
        self.best_acc = 0.0
        self.best_loss = 1000000
        self.best_optimizer_params = copy.deepcopy(self.optimizer.state_dict())
        
        #early stop
        self.max_train_acc=0
        self.early_stop_timer=0

    def train(self):
        '''
        function of training
        '''
        train_acc_list = []
        train_loss_list = []
        start = time.time()
        tqdm_iterator = tqdm(
            self.source_train_loader,
            desc="Training Epoch",  # 进度条前显示文字描述
            leave=False              # 循环结束后保留进度条，如果需要
        )
        for epoch in range(self.args.start_epoch, self.args.start_epoch+self.args.num_epochs):
            train_loss = 0.0
            train_acc = 0.0
            CE = 0.0
            SSC = 0.0
            ME = 0.0
            tqdm_iterator = tqdm(
                self.source_train_loader,
                desc="Training Epoch",  # 进度条前显示文字描述
                leave=False              # 循环结束后保留进度条，如果需要
            )
            for batch_idx, ((data1, target), data2) in enumerate(zip(tqdm_iterator, cycle(self.source_third_loader))):
               
                self.model.train()
                # data0 = data[0]
                # data1, target = data0
                # data2 = data[1]
                # print("data1   ", data1.shape)
                # print("target  ", target.shape)
                # print("data2   ", data2.shape)
                # data1 = data[0]
                # target = data[1]
                # data2 = data[1]
                if self.args.cuda:
                    data1 = data1.cuda()
                    target = target.cuda()
                    data2 = data2.cuda()
                data1, target, data2 = Variable(data1), Variable(target), Variable(data2)

            # print("---------", len(list(enumerate(self.source_third_loader))))
            # for batch_idx, (data, target, data1) in enumerate(self.source_third_loader):
            #     self.model.train()
            #     print(batch_idx)
            #     print(data)
            #     if self.args.cuda:
            #         data = data.cuda()
            #         target = target.cuda()
            #         data1 = data1.cuda()
            #     data, target, data1 = Variable(data), Variable(target), Variable(data1)
            
            # for batch_idx, (data, target) in enumerate(self.source_train_loader):
            #     self.model.train()
            #     print(batch_idx)
            #     if self.args.cuda:
            #         data = data.cuda()
            #         target = target.cuda()
            #     data, target = Variable(data), Variable(target)

                self.optimizer.zero_grad()
                
                #predict
                # output, feature = self.model(data)
                # output, embedding, output2 = self.model(data1, data2)
                output, embedding = self.model(data1)
                output2, _ = self.model(data2)

                loss1 = self.CE_Loss(output, target)
                # loss2 = self.loss(embedding, target)
                loss3 = self.ME_loss(output2)
                # loss = loss1 + loss2/(loss2/loss1).detach() + loss3/(loss3/loss1).detach()
                # loss = loss1/loss1.detach() + loss2/loss2.detach() + loss3/loss3.detach()
                # loss = loss1 + loss2/10 - loss3/10
                loss = loss1 - loss3/self.args.ratios
                # loss = self.loss(output, embedding, target)

                loss.backward()
                self.optimizer.step()
                train_loss += loss.data.item()
                CE += loss1.data.item()
                # SSC += loss2.data.item()
                ME += loss3.data.item()
                index = output.cpu().data.numpy().argmax(axis = 1)
                label =target.cpu().data.numpy()[:, 0]
                train_acc += sum(index == label)
                
            train_acc /= len(self.source_train_loader.dataset)
            train_loss /= len(self.source_train_loader.dataset)
            CE /= len(self.source_train_loader.dataset)
            SSC /= len(self.source_train_loader.dataset)
            ME /= len(self.source_third_loader.dataset)
            train_acc_list.append(train_acc)
            train_loss_list.append(train_loss)
            
            #early stop
            if CE < self.best_loss:
                self.best_loss =CE

            if train_acc > self.max_train_acc:
                self.max_train_acc = train_acc
                self.early_stop_timer = 0
            elif(train_acc > 0.996)&(CE < 0.01):
                self.early_stop_timer += 1
            # if self.early_stop_timer >= self.args.early_stop_patience or train_acc == 1.:
            #     print("\nearly stop\n")
            #     break;
            if self.args.dataset_name=="CHB":
                if math.isnan(loss.item()):
                    print("epoch : {}   train : acc {:.4} | loss {:.4} | CE {:.4} | SSC {:.4} | ME {:.4} | early-stop count {}".format(epoch,train_acc,train_loss,CE,SSC,ME,self.early_stop_timer))                
                    print("\nloss = Nan\n")
                    print("\nearly stop\n")
                    break;
                if train_acc == 1.0 and CE < 0.001:
                    print("epoch : {}   train : acc {:.4} | loss {:.4} | CE {:.4} | SSC {:.4} | ME {:.4} | early-stop count {}".format(epoch,train_acc,train_loss,CE,SSC,ME,self.early_stop_timer))                
                    print("\nacc == 1.0 and loss < 0.001\n")
                    print("\nearly stop\n")
                    break;
                if epoch >=50 and CE < 0.001:  # 0.001
                    print("epoch : {}   train : acc {:.4} | loss {:.4} | CE {:.4} | SSC {:.4} | ME {:.4} | early-stop count {}".format(epoch,train_acc,train_loss,CE,SSC,ME,self.early_stop_timer))               
                    print("\nepoch >=50 and loss < 0.001\n")
                    print("\nearly stop\n")
                    break;
                if epoch >=60 and train_acc >= 0.9995:  # 0.9995
                    print("epoch : {}   train : acc {:.4} | loss {:.4} | CE {:.4} | SSC {:.4} | ME {:.4} | early-stop count {}".format(epoch,train_acc,train_loss,CE,SSC,ME,self.early_stop_timer))                
                    print("\nepoch >=60 and acc >= 0.9995\n")
                    print("\nearly stop\n")
                    break;
                if epoch >=90 and train_acc >= 0.999 and CE <= 0.005:
                    print("epoch : {}   train : acc {:.4} | loss {:.4} | CE {:.4} | SSC {:.4} | ME {:.4} | early-stop count {}".format(epoch,train_acc,train_loss,CE,SSC,ME,self.early_stop_timer))                
                    print("\nepoch >=90 and acc >= 0.999 and loss <= 0.002\n")
                    print("\nearly stop\n")
                    break;
                if (train_loss- self.best_loss)<0.001 and (self.early_stop_timer >= self.args.early_stop_patience):
                    print("epoch : {}   train : acc {:.4} | loss {:.4} | CE {:.4} | SSC {:.4} | ME {:.4} | early-stop count {}".format(epoch,train_acc,train_loss,CE,SSC,ME,self.early_stop_timer))                
                    print("\nearly_stop_timer >= patience\n")
                    print("\nearly stop\n")
                    break;
            if self.args.dataset_name=="Kaggle":
                if math.isnan(loss.item()):
                    print("epoch : {}   train : acc {:.4} | loss {:.4} | CE {:.4} | SSC {:.4} | ME {:.4} | early-stop count {}".format(epoch,train_acc,train_loss,CE,SSC,ME,self.early_stop_timer))                
                    print("\nloss = Nan\n")
                    print("\nearly stop\n")
                    break;
                if train_acc == 1.0 and CE < 0.001:
                    print("epoch : {}   train : acc {:.4} | loss {:.4} | CE {:.4} | SSC {:.4} | ME {:.4} | early-stop count {}".format(epoch,train_acc,train_loss,CE,SSC,ME,self.early_stop_timer))                
                    print("\nacc == 1.0 and loss < 0.001\n")
                    print("\nearly stop\n")
                    break;
                if epoch >=50 and CE < 0.01:  # 0.001
                    print("epoch : {}   train : acc {:.4} | loss {:.4} | CE {:.4} | SSC {:.4} | ME {:.4} | early-stop count {}".format(epoch,train_acc,train_loss,CE,SSC,ME,self.early_stop_timer))               
                    print("\nepoch >=50 and loss < 0.001\n")
                    print("\nearly stop\n")
                    break;
                if epoch >=60 and train_acc >= 0.999 and ME > -0.007:  # 0.9995
                    print("epoch : {}   train : acc {:.4} | loss {:.4} | CE {:.4} | SSC {:.4} | ME {:.4} | early-stop count {}".format(epoch,train_acc,train_loss,CE,SSC,ME,self.early_stop_timer))                
                    print("\nepoch >=60 and acc >= 0.9995\n")
                    print("\nearly stop\n")
                    break;
                if epoch >=90 and train_acc >= 0.99 and CE <= 0.005:
                    print("epoch : {}   train : acc {:.4} | loss {:.4} | CE {:.4} | SSC {:.4} | ME {:.4} | early-stop count {}".format(epoch,train_acc,train_loss,CE,SSC,ME,self.early_stop_timer))                
                    print("\nepoch >=90 and acc >= 0.999 and loss <= 0.002\n")
                    print("\nearly stop\n")
                    break;
                if (train_loss- self.best_loss)<0.001 and (self.early_stop_timer >= self.args.early_stop_patience):
                    print("epoch : {}   train : acc {:.4} | loss {:.4} | CE {:.4} | SSC {:.4} | ME {:.4} | early-stop count {}".format(epoch,train_acc,train_loss,CE,SSC,ME,self.early_stop_timer))                
                    print("\nearly_stop_timer >= patience\n")
                    print("\nearly stop\n")
                    break;
                
            #early stop
            # if train_acc > self.max_train_acc:
            #     self.max_train_acc = train_acc
            #     self.early_stop_timer = 0
            # else:
            #     self.early_stop_timer += 1
            # if self.early_stop_timer >= self.args.early_stop_patience:
            #     print("\nearly stop\n")
            #     break;
            #print results
            print("epoch : {}   train : acc {:.4} | loss {:.4} | CE {:.4} | SSC {:.4} | ME {:.4} | early-stop count {}".format(epoch,train_acc,train_loss,CE,SSC,ME,self.early_stop_timer))
            # mkdir("{}/model/{}/{}/{}".format(self.args.checkpoint_dir, self.args.dataset_name, self.args.model_name, self.args.patient_id))
            torch.save(self.model.state_dict(),
               "{}/model/{}/{}/{}/patient{}_{}_last.pth".format(self.args.checkpoint_dir, self.args.dataset_name, self.args.model_name, self.args.patient_id,
                                                           self.args.patient_id, self.args.LOO))
            #test model
            if (epoch + 1) % 10 == 0:
                torch.save(self.model.state_dict(),
                   "{}/model/{}/{}/{}/patient{}_{}_epoch{}.pth".format(
                       self.args.checkpoint_dir, 
                       self.args.dataset_name, 
                       self.args.model_name, 
                       self.args.patient_id,
                       self.args.patient_id, 
                       self.args.LOO, 
                       epoch + 1  # 这里将当前 epoch 数加到文件名中
                   ))
            if self.args.TestWhenTraining == 1:
                test_acc, test_loss, index_list, target_list=self.test()
                TP,FP,TN,FN=0,0,0,0
                for i in range(len(index_list)):
                    for j in range(len(index_list[i])):
                        if index_list[i][j]==1 and target_list[i][j]==1:
                            TP+=1
                        elif index_list[i][j]==0 and target_list[i][j]==1:
                            FN+=1
                        elif index_list[i][j]==0 and target_list[i][j]==0:
                            TN+=1
                        else:
                            FP+=1
                print("test : TP {} | FN {} | TN {} | FP {} | sen {:.4%} | spe {:.4%} | acc {:.4%}\n".format(TP,FN,TN,FP,TP/(TP+FN),TN/(TN+FP),(TP+TN)/(TP+FN+TN+FP)))

        end = time.time()
        print(f"Training time: {(end-start)/60:.3f} minutes")
        return self.model,train_acc_list,train_loss_list
        
    def test(self):
        '''
        function of testing
        '''
        from sklearn.metrics import roc_auc_score,roc_curve, auc
        target_list=[]
        index_list=[]
        self.model.eval()
        test_loss = 0.0
        test_acc = 0.0
        output_probablity=[]
        output_list=[]
        with torch.no_grad():
            for i, (data, target) in enumerate(self.target_test_loader):
                if self.args.cuda:
                    data = data.cuda()
                    target = target.cuda()
                    
                #model predict
                # output, feature = self.model(data)
                output, embedding = self.model.forward_once(data)
                output_nosoftmax=output.cpu().detach().numpy()
                loss = self.loss(output, target)
                output=torch.nn.functional.softmax(output, dim=1)
                output=torch.clamp(output, min=1e-9, max=1-1e-9)
                output=output.cpu().detach().numpy()
                test_loss += loss.data.item()
                index = output.argmax(axis = 1)
                label =target.cpu().data.numpy()[:, 0]
                test_acc += sum(index == label)
                
                target_list.append(target.cpu())
                index_list.append(index)
                if len(output_probablity)==0:
                    output_probablity.append(output)  
                    output_probablity=np.array(output_probablity).squeeze()
                    output_list.append(output_nosoftmax)  
                    output_list=np.array(output_list).squeeze()
                else:
                    output_probablity=np.vstack((output_probablity, output))
                    output_list=np.vstack((output_list, output_nosoftmax))
        if torch.is_tensor(target_list[0]):
            all_targets = torch.cat(target_list).cpu().numpy().flatten()
        else:
            # 如果已经是 numpy list
            all_targets = np.concatenate(target_list).flatten()
        auc_value1 = roc_auc_score(all_targets, output_probablity[:,1])
        predicting_probablity=output_probablity[:,1]
        moving_average_length = 25 if self.args.dataset_name == "Kaggle" else 6
        predicting_probablity_smooth=smooth(predicting_probablity,moving_average_length)
        auc_value3=roc_auc_score(all_targets, predicting_probablity_smooth)
        
        test_acc /= len(self.target_test_loader.dataset)
        test_loss /= len(self.target_test_loader.dataset)
        self.model.train()
        count=0
        interval=0  # 距离发作点的时间
        false_alarm_list=[]  # 误报时间点列表
        true_alarm_list=[]  # 正报时间点列表
        interictal_length = self.args.test_set_interictal
        preictal_length = self.args.test_set_preictal
        for index in range(len(predicting_probablity_smooth)):
			# probability is over threshold, start counting
            if predicting_probablity_smooth[index]>0.6:

                count+=1
            else:
                count=0
            #if count is over persistence second，decide as one alarm
            if count>=1:
                interval=interictal_length+preictal_length-index
                #if the alarm is within 15min，True alarm
                if index >= interictal_length and index < interictal_length + preictal_length:
                    true_alarm_list.append(interval)
                #if the alarm is not within 15min，False alarm
                elif index < interictal_length:
                    false_alarm_list.append(interval)
                count=0
                
        # if model_name == "spectralCNN":
        #     FPR=len(false_alarm_list)/((interictal_length*30+(preictal_length+ictal_length)*step_preictal)/3600)#spectralCNN
        # else:
        #     FPR=len(false_alarm_list)/((interictal_length+preictal_length+ictal_length)*step_preictal/3600)
        # FPR=len(false_alarm_list)/((interictal_length*target_preictal_interval+preictal_length*step_preictal)/3600)
        FPR=len(false_alarm_list)/((interictal_length*30+preictal_length*5 + 30 - 5)/3600)

        
        if len(true_alarm_list) > 0:
            SN = 1
        else:
            SN = 0
        print('====> Test set loss: {:.4f}, AUC_soft {:.4f}, AUC_smooth {:.4f}, FPR {:.4f} per hour, SN {:.4f}'.format(
            test_loss, auc_value1, auc_value3, FPR, SN))
        return test_acc, test_loss, index_list, target_list#index.reshape(5, 20)

    def test_on_trainings_set(self):
        print('testing...')
        self.model.eval()
        test_loss = 0
        for i, (data, _) in enumerate(self.source_train_loader):
            if self.args.cuda:
                data = data.cuda()
            data = Variable(data, volatile=True)
            recon_batch, mu, logvar, z = self.model(data)
            test_loss += self.loss(recon_batch, data, mu, logvar).data[0]
            '''
            if i % 50 == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n],
                                        recon_batch.view(-1, 3, 32, 32)[:n]])
                self.summary_writer.add_image('training_set/image', comparison, i)
            '''
        test_loss /= len(self.target_test_loader.dataset)
        print('====> Test on training set loss: {:.4f}'.format(test_loss))
        self.model.train()

    def get_optimizer(self):
        return optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.args.learning_rate,
                          weight_decay=self.args.weight_decay, betas = (0.5, 0.999))
        #return optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr = self.args.learning_rate)

    def adjust_learning_rate(self, epoch):
        """Sets the learning rate to the initial LR multiplied by 0.98 every epoch"""
        learning_rate = self.args.learning_rate * (self.args.learning_rate_decay ** epoch)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = learning_rate
            #param_group['lr'] = param_group['lr']*0.2
    
    def adjust_learning_rate_step(self):
        """Sets the learning rate to the initial LR multiplied by 0.98 every epoch"""
        #learning_rate = self.args.learning_rate * (self.args.learning_rate_decay ** epoch)
        for param_group in self.optimizer.param_groups:
            #param_group['lr'] = learning_rate
            param_group['lr'] = param_group['lr']*0.99

    def save_checkpoint(self, epoch, state, is_best=False, filename='checkpoint{}.pth'):
        '''
        a function to save checkpoint of the training
        :param state: {'epoch': cur_epoch + 1, 'state_dict': self.model.state_dict(),
                            'optimizer': self.optimizer.state_dict()}
        :param is_best: boolean to save the checkpoint aside if it has the best score so far
        :param filename: the name of the saved file
        '''
        torch.save(state, self.args.checkpoint_dir + filename.format(epoch))
        if is_best:
            shutil.copyfile(self.args.checkpoint_dir + filename,
                            self.args.checkpoint_dir + 'model_best.pth.tar')

    def load_checkpoint(self, filename):
        filename = self.args.checkpoint_dir + filename
        try:
            print("Loading checkpoint '{}'".format(filename))
            checkpoint = torch.load(filename)
            self.args.start_epoch = checkpoint['epoch']
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("Checkpoint loaded successfully from '{}' at (epoch {})\n"
                  .format(self.args.checkpoint_dir, checkpoint['epoch']))
        except:
            print("No checkpoint exists from '{}'. Skipping...\n".format(self.args.checkpoint_dir))
