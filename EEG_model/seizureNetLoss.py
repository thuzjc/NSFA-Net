# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 09:51:34 2021

@author: phantom
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class CE_Loss(nn.Module):
    def __init__(self):
        super(CE_Loss, self).__init__()
        self.crossEntropy = nn.CrossEntropyLoss(reduction='sum')
    
    def forward(self, output, label):
        CE_loss = self.crossEntropy(output, label.squeeze())
        return CE_loss

# class FocalLoss(nn.Module):
#     """
#     This criterion is a implemenation of Focal Loss, which is proposed in 
#     Focal Loss for Dense Object Detection.

#         Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])

#     The losses are averaged across observations for each minibatch.

#     Args:
#         alpha(1D Tensor, Variable) : the scalar factor for this criterion
#         gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5), 
#                                 putting more focus on hard, misclassiﬁed examples
#         size_average(bool): By default, the losses are averaged over observations for each minibatch.
#                             However, if the field size_average is set to False, the losses are
#                             instead summed for each minibatch.

#     """
#     def __init__(self, class_num=2, alpha=None, gamma=2, size_average=False):
#         super(FocalLoss, self).__init__()
#         if alpha is None:
#             # self.alpha = Variable(torch.ones(class_num, 1))
#             self.register_buffer('alpha', torch.ones(class_num,1))
#         else:
#             if isinstance(alpha, Variable):
#                 self.register_buffer('alpha', alpha)
#             else:
#                 self.register_buffer('alpha', torch.tensor(alpha))
#         self.gamma = gamma
#         self.class_num = class_num
#         self.size_average = size_average

#     def forward(self, inputs, targets):
#         device = inputs.device
#         N = inputs.size(0)
#         C = inputs.size(1)
#         P = F.softmax(inputs, dim=1)

#         class_mask = torch.zeros(N, C, device=device)
#         ids = targets.view(-1, 1)
#         class_mask.scatter_(1, ids.data, 1.)
#         ids_flat = ids.view(-1).to(device)
#         alpha = self.alpha[ids_flat].to(device)

#         probs = (P * class_mask).sum(1).view(-1, 1)
#         log_p = probs.log()
#         batch_loss = -alpha * (torch.pow((1 - probs), self.gamma)) * log_p
#         if self.size_average:
#             loss = batch_loss.mean()
#         else:
#             loss = batch_loss.sum()
#         return loss


class FocalLoss(nn.Module):
    """
    This criterion is a implemenation of Focal Loss, which is proposed in 
    Focal Loss for Dense Object Detection.

        Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])

    The losses are averaged across observations for each minibatch.

    Args:
        alpha(1D Tensor, Variable) : the scalar factor for this criterion
        gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5), 
                                putting more focus on hard, misclassiﬁed examples
        size_average(bool): By default, the losses are averaged over observations for each minibatch.
                            However, if the field size_average is set to False, the losses are
                            instead summed for each minibatch.

    """
    def __init__(self, class_num=2, alpha=None, gamma=2, size_average=False):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs, dim=1)

        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)
        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]

        probs = (P*class_mask).sum(1).view(-1,1)

        log_p = probs.log()
        batch_loss = -alpha*(torch.pow((1-probs), self.gamma))*log_p 
        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss
    

class SSCLoss(nn.Module):
    def __init__(self, class_num=2, temperature = 0.5):
        super(SSCLoss, self).__init__()
        self.class_num = class_num
        self.temperature = temperature

    def forward(self, inputs, targets):
        n = targets.shape[0]  # batch

        #这步得到它的相似度矩阵
        similarity_matrix = F.cosine_similarity(inputs.unsqueeze(1), inputs.unsqueeze(0), dim=2).cuda()
        #这步得到它的label矩阵，相同label的位置为1
        mask = torch.ones_like(similarity_matrix) * (targets.expand(n, n).eq(targets.expand(n, n).t())).cuda()

        #这步得到它的不同类的矩阵，不同类的位置为1
        mask_no_sim = torch.ones_like(mask).cuda() - mask

        #这步产生一个对角线全为0的，其他位置为1的矩阵
        mask_dui_jiao_0 = torch.ones(n ,n).cuda() - torch.eye(n, n ).cuda()

        #这步给相似度矩阵求exp,并且除以温度参数T
        similarity_matrix = torch.exp(similarity_matrix/self.temperature).cuda()

        #这步将相似度矩阵的对角线上的值全置0，因为对比损失不需要自己与自己的相似度
        similarity_matrix = similarity_matrix*mask_dui_jiao_0


        #这步产生了相同类别的相似度矩阵，标签相同的位置保存它们的相似度，其他位置都是0，对角线上也为0
        sim = mask*similarity_matrix


        #用原先的对角线为0的相似度矩阵减去相同类别的相似度矩阵就是不同类别的相似度矩阵
        no_sim = similarity_matrix - sim


        #把不同类别的相似度矩阵按行求和，得到的是对比损失的分母(还差一个与分子相同的那个相似度，后面会加上)
        no_sim_sum = torch.sum(no_sim , dim=1).cuda()

        '''
        将上面的矩阵扩展一下，再转置，加到sim（也就是相同标签的矩阵上），然后再把sim矩阵与sim_num矩阵做除法。
        至于为什么这么做，就是因为对比损失的分母存在一个同类别的相似度，就是分子的数据。做了除法之后，就能得到
        每个标签相同的相似度与它不同标签的相似度的值，它们在一个矩阵（loss矩阵）中。
        '''
        no_sim_sum_expend = no_sim_sum.repeat(n, 1).T
        sim_sum  = sim + no_sim_sum_expend
        loss = torch.div(sim , sim_sum).cuda()


        '''
        由于loss矩阵中，存在0数值，那么在求-log的时候会出错。这时候，我们就将loss矩阵里面为0的地方
        全部加上1，然后再去求loss矩阵的值，那么-log1 = 0 ，就是我们想要的。
        '''
        loss = mask_no_sim + loss + torch.eye(n, n ).cuda()


        #接下来就是算一个批次中的loss了
        loss = -torch.log(loss).cuda() #求-log
        # loss = torch.sum(torch.sum(loss, dim=1) )/(2*n)  #将所有数据都加起来除以2n

        # print(loss)  #0.9821
        #最后一步也可以写为---建议用这个， (len(torch.nonzero(loss)))表示一个批次中样本对个数的一半
        loss = torch.sum(torch.sum(loss, dim=1).cuda()).cuda() / (len(torch.nonzero(loss).cuda()))
        return loss

class SSC_CELoss(nn.Module):
    def __init__(self, class_num=2, temperature = 0.5):
        super(SSC_CELoss, self).__init__()
        self.class_num = class_num
        self.temperature = temperature
        self.crossEntropy = nn.CrossEntropyLoss(reduction='sum')

    def forward(self, inputs, embedding, targets):
        n = embedding.shape[0]  # batch

        #这步得到它的相似度矩阵
        similarity_matrix = F.cosine_similarity(embedding.unsqueeze(1), embedding.unsqueeze(0), dim=2).cuda()
        #这步得到它的label矩阵，相同label的位置为1
        mask = torch.ones_like(similarity_matrix) * (targets.expand(n, n).eq(targets.expand(n, n).t())).cuda()

        #这步得到它的不同类的矩阵，不同类的位置为1
        mask_no_sim = torch.ones_like(mask).cuda() - mask

        #这步产生一个对角线全为0的，其他位置为1的矩阵
        mask_dui_jiao_0 = torch.ones(n ,n).cuda() - torch.eye(n, n ).cuda()

        #这步给相似度矩阵求exp,并且除以温度参数T
        similarity_matrix = torch.exp(similarity_matrix/self.temperature).cuda()

        #这步将相似度矩阵的对角线上的值全置0，因为对比损失不需要自己与自己的相似度
        similarity_matrix = similarity_matrix*mask_dui_jiao_0


        #这步产生了相同类别的相似度矩阵，标签相同的位置保存它们的相似度，其他位置都是0，对角线上也为0
        sim = mask*similarity_matrix


        #用原先的对角线为0的相似度矩阵减去相同类别的相似度矩阵就是不同类别的相似度矩阵
        no_sim = similarity_matrix - sim


        #把不同类别的相似度矩阵按行求和，得到的是对比损失的分母(还差一个与分子相同的那个相似度，后面会加上)
        no_sim_sum = torch.sum(no_sim , dim=1).cuda()

        '''
        将上面的矩阵扩展一下，再转置，加到sim（也就是相同标签的矩阵上），然后再把sim矩阵与sim_num矩阵做除法。
        至于为什么这么做，就是因为对比损失的分母存在一个同类别的相似度，就是分子的数据。做了除法之后，就能得到
        每个标签相同的相似度与它不同标签的相似度的值，它们在一个矩阵（loss矩阵）中。
        '''
        no_sim_sum_expend = no_sim_sum.repeat(n, 1).T
        sim_sum  = sim + no_sim_sum_expend
        loss = torch.div(sim , sim_sum).cuda()


        '''
        由于loss矩阵中，存在0数值，那么在求-log的时候会出错。这时候，我们就将loss矩阵里面为0的地方
        全部加上1，然后再去求loss矩阵的值，那么-log1 = 0 ，就是我们想要的。
        '''
        loss = mask_no_sim + loss + torch.eye(n, n ).cuda()


        #接下来就是算一个批次中的loss了
        loss = -torch.log(loss).cuda() #求-log
        # loss = torch.sum(torch.sum(loss, dim=1) )/(2*n)  #将所有数据都加起来除以2n

        # print(loss)  #0.9821
        #最后一步也可以写为---建议用这个， (len(torch.nonzero(loss)))表示一个批次中样本对个数的一半
        loss = torch.sum(torch.sum(loss, dim=1).cuda()).cuda() / (len(torch.nonzero(loss).cuda()))
        CE = self.crossEntropy(inputs, targets.squeeze())
        loss = loss/10 + CE
        return loss

class ME_Loss(nn.Module):
    """
    Max Entropy Loss
    """
    def __init__(self, class_num=2):
        super(ME_Loss, self).__init__()
        self.class_num = class_num

    def forward(self, inputs):
        # inputs = torch.stack((inputs))
        ME_loss = torch.mean(torch.sum(torch.softmax(inputs, 1).cuda() * torch.log(torch.softmax(inputs, 1).cuda()), 1), 0)
        return ME_loss