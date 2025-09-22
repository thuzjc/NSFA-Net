
import time

from matplotlib import pyplot as plt
from torch.autograd import Variable
from torch import optim
import torch
import shutil
import copy
from pytorch_metric_learning import distances, losses, miners, reducers, testers
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import numpy as np
import os
from collections import Counter


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
    def __init__(self, model, loss1, loss2, source_train_loader, source_snr_train_loader, target_test_loader, args, save_dir):
        self.model = model
        self.args = args
        self.args.start_epoch = 0
        self.source_train_loader = source_train_loader
        self.source_snr_train_loader = source_snr_train_loader
        self.target_test_loader = target_test_loader
        # Loss function and Optimizer
        self.loss1 = loss1  # CE loss
        self.loss2 = loss2  # NA loss
        self.optimizer = self.get_optimizer(args)  # Adam
        # self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 50, gamma=0.5, last_epoch=-1)
        # self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=10, verbose=True) #当监控的指标在若干个epoch内没有改善时，自动将学习率按照给定的factor进行缩小
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 30, gamma = 0.4, last_epoch = -1)
        # self.best_model_params = copy.deepcopy(model.state_dict())
        self.best_acc = 0.0
        self.best_loss = 1000000
        # self.best_optimizer_params = copy.deepcopy(self.optimizer.state_dict())

        # Define triplet loss utility functions
        self.distance = distances.CosineSimilarity()
        # self.distance = distances.LpDistance
        self.reducer = reducers.ThresholdReducer(low=0)
        self.mining_func = miners.TripletMarginMiner(margin=0.45, distance=self.distance, type_of_triplets="all")
        self.tripletloss = losses.TripletMarginLoss(margin=0.45, distance=self.distance, reducer=self.reducer)
        self.threshold = 0.6
        # early stop
        self.max_train_acc = 0
        self.min_train_loss = 1000000
        self.early_stop_timer = 0
        self.save_dir = save_dir
        self.save_every_train_epoch = 10
        if os.path.exists(self.save_dir) is False:
            os.makedirs(self.save_dir)
    def train(self):
        '''
        function of training
        '''
        train_acc_list = []
        train_loss_list = []
        for epoch in range(self.args.start_epoch, self.args.start_epoch + self.args.num_epochs):
            start_time = time.time()
            train_loss = 0.0
            train_acc = 0.0
            train_triplets_loss = 0.0
            num_triplets = 0
            save_train = 0

            for batch_idx, ((data1, target1), (data2, target2)) in tqdm(enumerate(zip(self.source_train_loader, self.source_snr_train_loader)), total=len(self.source_train_loader),desc="Training"):
                # TODO
                if data1.shape[0] == 1:
                    print("batch size = 1")
                    print(data1.shape)
                    continue
                self.model.train()
                if self.args.cuda:
                    data1 = data1.cuda()
                    target1 = target1.cuda()
                    data2 = data2.cuda()
                    target2=target2.cuda()
                data1, target1 = Variable(data1), Variable(target1)
                data2, target2 = Variable(data2), Variable(target2)
                self.optimizer.zero_grad()

                outputs1, embeddings = self.model(data1)
                outputs2, embeddings = self.model(data2)

                # print(ori.shape)
                # print(target.shape)
                labels1 = target1.squeeze()
                labels2 = target2.squeeze()

                indices_tuple = self.mining_func(embeddings, labels1) #挖掘当前batch中合法的三元组索引
                tlloss = self.tripletloss(embeddings, labels1, indices_tuple) # 计算三元组损失
                num_triplets += self.mining_func.num_triplets #累加的三元组数量

                loss1 = self.loss1(outputs1, target1)
                loss2 = self.loss2(outputs2, target2)
                # loss2 = self.loss2(output, target)
                # loss = loss1 + 3 * loss2 + 9 * tlloss
                loss = loss1 + 0.1*loss2 + 0.5* tlloss
                # loss = loss1

                loss.backward()

                # 打印梯度异常的参数
                for name, param in self.model.named_parameters():
                    if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                        # if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                        print('梯度异常:', name)
                        print('梯度值:', torch.abs(param.grad).max())
                        print('梯度:', param.grad)
                        print('参数:', param)
                        print('-----------------------------------')

                self.optimizer.step()
                train_loss += loss.data.item()
                train_triplets_loss += tlloss.data.item()
                # train_triplets_loss = 0

                # index = output.cpu().data.numpy().argmax(axis=1)
                index = outputs1.cpu().data.numpy().argmax(axis=1)
                label = target1.cpu().data.numpy()[:, 0]
                train_acc += sum(index == label)

            train_acc /= len(self.source_train_loader.dataset)
            train_loss /= len(self.source_train_loader.dataset)
            train_triplets_loss /= len(self.source_train_loader.dataset)
            train_acc_list.append(train_acc)
            train_loss_list.append(train_loss)
            if epoch > 5 and epoch % self.save_every_train_epoch == 0:
                print(f"valuable acc to save ckpt:{train_acc:.4f}, best:{self.best_acc:.4f}")
                torch.save(self.model.state_dict(), f"{self.save_dir}/model_train_{self.args.now_part}_{epoch}_acc={train_acc:.4f}.pth")
            # early stop
            if train_loss < self.best_loss:
                self.best_loss = train_loss
            if train_acc > self.best_acc:
                self.best_acc = train_acc
            if (train_acc > self.max_train_acc) & (train_acc < 0.98):
                self.max_train_acc = train_acc
                self.early_stop_timer = 0
            elif (train_acc > 0.98) & (train_loss < 0.0035) & ((train_loss - self.best_loss) < 0.0015) & ((train_acc - self.best_acc) < 0.01) & (
                    train_triplets_loss < 0.00025):
                self.early_stop_timer += 1
                print("early-stop count : {}".format(self.early_stop_timer))
                print("best : acc {:.4} | loss {:.4}".format(self.best_acc, self.best_loss))
            elif train_loss > 0.2:
                self.early_stop_timer = 0
            if self.early_stop_timer >= self.args.early_stop_patience:
                print("\nearly stop\n")
                print("epoch : {}\ntrain : acc {:.4} | loss {:.4} | train_triplets {} | train_triplets_loss {:.4} | early-stop count {}".format(epoch,
                                                                                                                                                train_acc,
                                                                                                                                                train_loss,
                                                                                                                                                num_triplets,
                                                                                                                                                train_triplets_loss,
                                                                                                                                                self.early_stop_timer))
                break
            if (train_acc > 0.9975) & (train_loss < 0.0025) & (self.early_stop_timer >= 5) & (train_triplets_loss < 0.0001) & (
                    (train_loss - self.best_loss) < 0.0015) & ((train_acc - self.best_acc) < 0.01):
                print("\nearly stop 2\n")
                print("epoch : {}\ntrain : acc {:.4} | loss {:.4} | train_triplets {} | train_triplets_loss {:.4} | early-stop count {}".format(epoch,
                                                                                                                                                train_acc,
                                                                                                                                                train_loss,
                                                                                                                                                num_triplets,
                                                                                                                                                train_triplets_loss,
                                                                                                                                                self.early_stop_timer))
                break
            if (train_acc > 0.999) & (train_loss < 0.0035) & (self.early_stop_timer >= 2) & (train_triplets_loss < 0.0001) & (
                    (train_loss - self.best_loss) < 0.0001) & ((train_acc - self.best_acc) < 0.01) & (train_triplets_loss < 0.0001) & (num_triplets < 50000):
                print("\nearly stop X\n")
                print("epoch : {}\ntrain : acc {:.4} | loss {:.4} | train_triplets {} | train_triplets_loss {:.4} | early-stop count {}".format(epoch,
                                                                                                                                                train_acc,
                                                                                                                                                train_loss,
                                                                                                                                                num_triplets,
                                                                                                                                                train_triplets_loss,
                                                                                                                                                self.early_stop_timer))
                break

            #################### noise ####################
            if (train_acc > 0.98) & (train_loss < 0.03) & ((train_loss - self.best_loss) < 0.002) & ((train_acc - self.best_acc) < 0.01) & (
                    train_triplets_loss < 0.0015):
                self.early_stop_timer += 1
                print("early-stop count : {}".format(self.early_stop_timer))
                print("best : acc {:.4} | loss {:.4}".format(self.best_acc, self.best_loss))

            if (train_acc > 0.99) & (train_loss < 0.02) & (self.early_stop_timer >= 8) & (train_triplets_loss < 0.001) & (
                    (train_loss - self.best_loss) < 0.001) & ((train_acc - self.best_acc) < 0.01) & (num_triplets < 4500000):
                print("\nearly stop 2\n")
                print("epoch : {}\ntrain : acc {:.4} | loss {:.4} | train_triplets {} | train_triplets_loss {:.4} | early-stop count {}".format(epoch,
                                                                                                                                                train_acc,
                                                                                                                                                train_loss,
                                                                                                                                                num_triplets,
                                                                                                                                                train_triplets_loss,
                                                                                                                                                self.early_stop_timer))
                break
            if (train_acc > 0.99) & (train_loss < 0.01) & (self.early_stop_timer >= 5) & (train_triplets_loss < 0.001) & (
                    (train_loss - self.best_loss) < 0.0005) & ((train_acc - self.best_acc) < 0.01) & (num_triplets < 2500000):
                print("\nearly stop X\n")
                print("epoch : {}\ntrain : acc {:.4} | loss {:.4} | train_triplets {} | train_triplets_loss {:.4} | early-stop count {}".format(epoch,
                                                                                                                                                train_acc,
                                                                                                                                                train_loss,
                                                                                                                                                num_triplets,
                                                                                                                                                train_triplets_loss,
                                                                                                                                                self.early_stop_timer))
                break
            #################### noise ####################

            #################### noise ####################
            if (train_acc > 0.98) & (train_loss < 0.01) & ((train_loss - self.best_loss) < 0.002) & ((train_acc - self.best_acc) < 0.01):
                self.early_stop_timer += 1
                print("early-stop count : {}".format(self.early_stop_timer))
                print("best : acc {:.4} | loss {:.4}".format(self.best_acc, self.best_loss))

            if (train_acc > 0.99) & (train_loss < 0.005) & (self.early_stop_timer >= 5) & ((train_loss - self.best_loss) < 0.0005) & (
                    (train_acc - self.best_acc) < 0.01):
                print("\nearly stop X\n")
                print("epoch : {}\ntrain : acc {:.4} | loss {:.4} | train_triplets {} | train_triplets_loss {:.4} | early-stop count {}".format(epoch,
                                                                                                                                                train_acc,
                                                                                                                                                train_loss,
                                                                                                                                                num_triplets,
                                                                                                                                                train_triplets_loss,
                                                                                                                                                self.early_stop_timer))
                break
            #################### noise ####################

            self.scheduler.step()

            # print the current learning rate
            print(f"Current learning rate: {self.optimizer.param_groups[0]['lr']}")

            # print results
            print("epoch : {}\ntrain : acc {:.4} | loss {:.4} | train_triplets {} | train_triplets_loss {:.4} | early-stop count {}".format(epoch, train_acc,
                                                                                                                                            train_loss,
                                                                                                                                            num_triplets,
                                                                                                                                            train_triplets_loss,
                                                                                                                                            self.early_stop_timer))
            # 记录每个 epoch 的结束时间
            end_time = time.time()
            # 计算每个 epoch 的时间差
            epoch_time = end_time - start_time
            # 打印每个 epoch 的时间信息
            print("Epoch {}: {:.2f} seconds".format(epoch, epoch_time))
            # test model
            if self.args.TestWhenTraining == 1:
                test_acc, index_list, target_list, probs = self.test()
                TP, FP, TN, FN = 0, 0, 0, 0
                for i in range(len(index_list)):
                    if index_list[i] == 1 and target_list[i] == 1:
                        TP += 1
                    elif index_list[i] == 0 and target_list[i] == 1:
                        FN += 1
                    elif index_list[i] == 0 and target_list[i] == 0:
                        TN += 1
                    else:
                        FP += 1
                
                print("test : TP {} | FN {} | TN {} | FP {} | sen {:.4%} | spe {:.4%} | acc {:.4%} ".format(TP, FN, TN, FP, TP / (TP + FN), TN / (TN + FP),
                                                                                                             (TP + TN) / (TP + FN + TN + FP)))

        return {'epoch':epoch, 'state_dict':self.model.state_dict()}, train_acc_list, train_loss_list

    def test(self,):
        '''
        function of testing
        '''
        print("self.test()")
        target_list = []
        index_list = []
        prob_list = []
        self.model.eval()
        test_acc = 0.0
        with torch.no_grad():
            for i, (data, target) in enumerate(self.target_test_loader):
                if self.args.cuda:
                    data = data.cuda()
                    target = target.cuda()

                output, embeddings = self.model(data)
                    
                index = output.cpu().data.numpy().argmax(axis=1)
                probs = torch.nn.functional.softmax(output, dim=1).cpu().numpy()
                label = target.cpu().data.numpy()[:, 0]
                test_acc += sum(index == label)

                target_list.append(label)
                index_list.append(index) #根据比较两个类别的logits值取出大的logits值
                prob_list.append(probs)
        # concat
        target_list = np.concatenate(target_list, 0)
        index_list = np.concatenate(index_list, 0)
        probs = np.concatenate(prob_list, 0)
        test_acc /= len(self.target_test_loader.dataset)
        self.model.train()
        return test_acc, index_list, target_list, probs  # index.reshape(5, 20)

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

    def get_optimizer(self, args):
        if self.args.model_name.startswith("TA_STS_ConvNet"):
            return optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.args.learning_rate,
                              weight_decay=self.args.weight_decay)
        elif self.args.model_name.startswith("STAN"):
            return optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.args.learning_rate, weight_decay=5E-2)
        elif self.args.model_name.startswith("MONST"):
            return optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.args.learning_rate, weight_decay=5E-2)
        else:
            return optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.args.learning_rate, weight_decay=5E-2)

        # return optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr = self.args.learning_rate)

    def adjust_learning_rate(self, epoch):
        """Sets the learning rate to the initial LR multiplied by 0.98 every epoch"""
        learning_rate = self.args.learning_rate * (self.args.learning_rate_decay ** epoch)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = learning_rate
            # param_group['lr'] = param_group['lr']*0.2

    def adjust_learning_rate_step(self):
        """Sets the learning rate to the initial LR multiplied by 0.98 every epoch"""
        # learning_rate = self.args.learning_rate * (self.args.learning_rate_decay ** epoch)
        for param_group in self.optimizer.param_groups:
            # param_group['lr'] = learning_rate
            param_group['lr'] = param_group['lr'] * 0.99

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
