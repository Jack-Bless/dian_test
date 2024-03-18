# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 12:57:50 2024

@author: 21624
"""

import torch
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
device = "cuda" if torch.cuda.is_available() else "cpu"
x_size=784
hide_size=640
out_size=10
o1=526
o2=128
batch_size=640
"""关于batch数值的选择我之前一直以为越小越好，
但事实好像不是，这里选640反而比64效果好"""
lr=0.001



"""———————————————————————————————我 是 分 割 线—————————————————————————————————"""


#下面两个函数分别是评测指标的记录函数(record)和评测指标计算输出函数(cal)
#输出的内容为总的正确率和每个分类单独的各项指标
def evaluation_record(outputs,labels,total,TP,FP,FN,TN,correct):
    _,pred=torch.max(outputs,dim=1)
    total+=labels.size(0)#样本总数
    correct+=(pred==labels).sum().item()#记录总体正确个数
    for i in range(10):
        TP[i]+=((labels==i)*(pred==i)).sum().item()
        FP[i]+=((pred==i)*(labels!=i)).sum().item()
        FN[i]+=((pred!=i)*(labels==i)).sum().item()
        TN[i]+=((pred!=i)*(labels!=i)).sum().item()#分别记录十个不同分类的指标数据
    return TP,FP,FN,TN,total,correct

def evaluation_cal(TP,FP,FN,TN,total,correct):
    accuracy=100*correct/total
    print('accuracy on test set:%d %%'%accuracy)#计算并输出总正确率
    print('\n')
    for i in range(10):
        precision=100*TP[i]/(TP[i]+FP[i])
        recall=100*TP[i]/(TP[i]+FN[i])
        F1_score=2*(precision*recall/(precision+recall))
        print('precision:%d on test set:%d %%'%(i,precision))
        print('recall:%d on test set:%d %%'%(i,recall))
        print('F1-score:%d on test set:%d %%'%(i,F1_score))#计算并输出各项指标
        print('\n')
  
#绘制混淆矩阵，可视化训练过程
def confu_matrix(outputs,labels):
    _,pred=torch.max(outputs,dim=1)
    x=torch.zeros(10,10)
    for i in range(10):
        for j in range(10):
            x[i][j]=((pred==i)*(labels==j)).sum().item()#记录矩阵数据
    plt.imshow(x, cmap='viridis', interpolation='nearest', aspect='auto')#画矩阵
    plt.colorbar()#显示颜色条
    plt.title('pred_data and real_data')#设置标题
    plt.show()




"""———————————————————————————————我 是 分 割 线—————————————————————————————————"""


class Net(torch.nn.Module):
    def __init__(self,x_size,hide_size,out_size,o1,o2):
        super(Net,self).__init__()
        self.linear1=torch.nn.Linear(x_size,hide_size)#将输入层的大小转化为隐藏层大小，便于和上一次的隐藏层输出值融合
        
        self.linear2=torch.nn.Linear(hide_size,o1)#隐藏层输出
        self.linear3=torch.nn.Linear(o1,o2)
        """别问这里为啥又有一层，问就是提高学习效果ᕙ(`▿´)ᕗ"""
       
        self.linear6=torch.nn.Linear(o2,out_size)#到达输出层
        self.tanh=torch.nn.Tanh()#激活，据说这个激活函数效果更好
    def forward(self,x_data,hide_data):
        """————————————————————————下面这两步其实就是整个RNN网络的关键——————————————————————————————"""
        combine_data=self.linear1(x_data)+hide_data#融合上一次的隐藏层数据和这一次的输入数据
        hide_data=self.tanh(combine_data)#计算这一次的隐藏层数据
        """——————————————————————————————————————————————————————————————————"""
        x=self.linear2(hide_data)
        x=self.linear3(x)
        x=self.linear6(x)#输出了，但是没必要激活，后面有CrossEntropy
        return x,hide_data

net=Net(x_size,hide_size,out_size,o1,o2)#初始化各项参数
criterion=torch.nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(net.parameters(),lr=lr)

#下面是准备数据和训练测试的过程，和前面的题目一样，就不多写注释了
transform=transforms.Compose([transforms.ToTensor(),
                             transforms.Normalize((0.1307,),(0.3081))])
train_dataset=datasets.FashionMNIST(root='mnist',
                             train=True,
                             download=True,
                             transform=transform)

train_loader=DataLoader(train_dataset,
                        shuffle=False,
                        batch_size=batch_size,
                        drop_last=True)

test_dataset=datasets.FashionMNIST(root='mnist',
                             train=False,
                             download=True,
                             transform=transform)

test_loader=DataLoader(test_dataset,
                        shuffle=False,
                        batch_size=batch_size,
                        drop_last=True)

def train(epoch):
    x=0
    loss_list=[]
    hide_data=torch.zeros(batch_size,hide_size)
    for data in train_loader:
        inputs,target=data
        inputs=inputs.view(batch_size,x_size)
        optimizer.zero_grad()
        x=x+1
        outputs,hide_data=net(inputs,hide_data.data)
        loss=criterion(outputs,target)
        loss_list.append(loss.item())
        loss.backward()    
        optimizer.step()
    plt.plot(range(x),loss_list)
    plt.show()
    
def test():
    correct=0
    total=0
    TP=[0 for _ in range(11)]
    TN=[0 for _ in range(11)]
    FP=[0 for _ in range(11)]
    FN=[0 for _ in range(11)]
    hide_data=torch.zeros(batch_size,hide_size)
    with torch.no_grad():
        for data in test_loader:
            images,labels=data
            images=images.view(batch_size,x_size)
            outputs,hide_data=net(images,hide_data)
            TP,FP,FN,TN,total,correct=evaluation_record(outputs,labels,total,TP,FP,FN,TN,correct)
    evaluation_cal(TP,FP,FN,TN,total,correct)
    confu_matrix(outputs,labels)
for epoch in range(3):
    train(epoch)
    test()
    print('####################################################\n')




"""———————————————————————————————我 是 分 割 线—————————————————————————————————"""
"""
学习记录：
    对RNN网络的理解：
        RNN网络与基础的全连接层的巨鳖就在于RNN网络引入了上一个样本与这一个样本之间的关系，
        这样就可以体现出数据之间的顺序关系。具体的操作就是把上一次隐藏层中的数据留存下来，
        并且作为这一次输入数据的一部分，和输入数据融合，一起计算这一次的隐藏层数据，实现
        两次运算的关联，从而推理出数据之间的相关性。
    RNN的缺陷：
        RNN网络可以对数据之间的先后顺序进行学习，但是也有局限性，比如随着样本不断计算，
        第一个样本和最后一个样本之间的关联会逐渐不被注意。Self-attention机制实际上就解决了
        这个问题，它可以无视这种距离的远近对关联度计算的影响。下一个题目里有我对attention
        的理解。
    对本题的理解：
        因为fashion-mnist对图片顺序并没有什么要求，所以用RNN网络进行训练优势不大，
        我用基础的全连接层也试了一下，两者的正确率区别不大，而且RNN更耗时间

"""















