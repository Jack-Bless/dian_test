# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 18:36:27 2024

@author: 21624
"""




import torch
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
import matplotlib.pyplot as plt
device = "cuda" if torch.cuda.is_available() else "cpu"

batch_size=64
#第二题的数据
input_size=128
output_size=256
#第三题的数据
inputs_size=784
h1_size=512
h2_size=256
h3_size=128
h4_size=64
outputs_size=10



"""———————————————————————————————我 是 分 割 线—————————————————————————————————"""




#第一题：自制指标测评函数
#下面两个函数分别是评测指标的记录函数(record)和评测指标计算输出函数(cal)
#对评测指标的理解写在本文件最下方
#输出的内容为总体正确率和每个分类单独的各项指标
def evaluation_record(outputs,labels,total,TP,FP,FN,TN,correct):
    _,pred=torch.max(outputs,dim=1)#记录预测值
    total+=labels.size(0)#样本总数
    correct+=(pred==labels).sum().item()#记录总体正确个数
    for i in range(10):
        TP[i]+=((labels==i)*(pred==i)).sum().item()
        FP[i]+=((pred==i)*(labels!=i)).sum().item()
        FN[i]+=((pred!=i)*(labels==i)).sum().item()
        TN[i]+=((pred!=i)*(labels!=i)).sum().item()#分别记录十个不同分类的各项指标数据
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
    x=torch.zeros(10,10)#初始一个空矩阵，便于后续储存测试数据
    for i in range(10):
        for j in range(10):
            x[i][j]=((pred==i)*(labels==j)).sum().item()#以矩阵的形式记录测试数据，后续可以直接画图
    plt.imshow(x, cmap='viridis', interpolation='nearest', aspect='auto')#开画
    plt.colorbar()#显示颜色条
    plt.title('pred_data and real_data')#设置标题
    plt.show()
        
        
        
        
        
"""———————————————————————————————我 是 分 割 线—————————————————————————————————"""
        
        
        
        
        
    
#第二题目标：实现最基本的二层结构，第一层神经元128，第二层为256
#关于对全连接神经网络的理解也在文件最下方
x_data=torch.rand(1,input_size)#初始化输入神经元
weight=torch.rand(output_size,input_size)#初始化权重
bias=torch.rand(1,output_size)#初始化偏移量
"""后来我才知道不需要自己手动初始化权重和偏移，nn.Linear会自动初始化，
但代码我还是保留在这了"""
class model(torch.nn.Module):#初始化网络模型
    def __init__(self):
        super(model,self).__init__()#继承父类，标准格式
        self.linear=torch.nn.Linear(input_size,output_size)#输入128神经元，输出256
    def forward(self,x):
        return self.linear(x)
    
model=model()#类实例化

model.linear.weight=Parameter(weight)#
model.linear.bias=Parameter(bias)#把初始化的权重和偏移量放入模型中。这里必须要把变量形式转化才能成功赋值

y_data=model.forward(x_data)#开始运算

print(x_data.shape)#
print(y_data.shape)#输出x_data和y_data的形状
print('\n')
#完成要求，输入128神经元，输出256神经元





"""———————————————————————————————我 是 分 割 线—————————————————————————————————"""





#第三题目标：mnist分类问题，完成分类并计算指标
"""这里有个小问题，我想把mnist文件转化为jpg看一眼长啥样，
但那个opencv就是死活用不了，最后还是放弃了( ͡° ʖ̯ ͡°)"""
#第一步准备好数据集
transform=transforms.Compose([transforms.ToTensor(),
                             transforms.Normalize((0.5,),(0.5))])
#transform将数据集中的数据转化为可处理的tensor并且正则化

train_dataset=datasets.MNIST(root='mnist_num',#准备一下训练用的mnist
                             train=True,
                             download=True,
                             transform=transform)

train_loader=DataLoader(train_dataset,
                        shuffle=True,
                        batch_size=batch_size,#处理一下数据，64个样本为一个batch并且随机取
                        drop_last=True)#关于这种处理方法的好处也写在文件后面了

test_dataset=datasets.MNIST(root='mnist_num',#准备一下测试用的mnist
                             train=False,
                             download=True,
                             transform=transform)

test_loader=DataLoader(test_dataset,
                        shuffle=True,
                        batch_size=batch_size,
                        drop_last=True)

class Net(torch.nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.l1=torch.nn.Linear(inputs_size,h1_size)
        self.l2=torch.nn.Linear(h1_size,h2_size)
        self.l3=torch.nn.Linear(h2_size,h3_size)
        self.l4=torch.nn.Linear(h3_size,h4_size)#增加隐藏层的数量是为了增加模型复杂度以达到增强学习效果的目的
        self.l5=torch.nn.Linear(h4_size,outputs_size)#线形层模型将数据特征值转换为10，与label相同
        self.relu=torch.nn.ReLU()#激活函数
    def forward(self,x):
        x=x.view(-1,784)#将原本的图片数据大小(N,28,28)转换为矩阵(N,784)，便于模型处理
        """这里把图形矩阵(28*28)压缩成了一条行向量，原本(batch_size,28,28)的数据被
        压缩成(batch_size,784)的数据，也就是一个矩阵，便于线性层处理，
        但在我学习CNN的时候发现了重大华点，也写在文件后面了，学长可以看看我写的对不对(为什么我只说学长.doge)"""
        x=self.relu(self.l1(x))
        x=self.relu(self.l2(x))
        x=self.relu(self.l3(x))
        x=self.relu(self.l4(x))#每经过一个线性层，进行一次非线性变换
        
        return self.l5(x)#返回一个线性数据，后面的损失模型Cross会自动softmax非线性化
net=Net()#实例化
criterion=torch.nn.CrossEntropyLoss()#损失模型
optimizer=torch.optim.SGD(net.parameters(),lr=0.01,momentum=0.5)#带冲量的优化器，效果更好

def train(epoch):#定义一个训练模型
    x=0#可视化过程中作为横坐标
    loss_list=[]#用来记录损失，便于可视化
    for data in train_loader:#每次从数据集中随机拿一个64的batch进行损失计算
        inputs,target=data#拿取一个batch的数据和标签
        optimizer.zero_grad()#梯度归零，防止每次梯度累计
        outputs=net(inputs)#前馈过程
        loss=criterion(outputs,target)#损失计算
        loss_list.append(loss.item())
        x=x+1#记录训练次数
        loss.backward()#开始反向传播
        optimizer.step()
    plt.plot(range(x),loss_list)#x记录的是训练次数,对应的是当次的损失
    plt.show()#可视化训练过程，反映的是训练次数和损失之间的关系
        
def test():
    total=0
    correct=0
    TP=[0 for _ in range(11)]
    TN=[0 for _ in range(11)]
    FP=[0 for _ in range(11)]
    FN=[0 for _ in range(11)]#每次测试指标清零
    with torch.no_grad():
        for data in test_loader:
            images,labels=data
            outputs=net(images)
            TP,FP,FN,TN,total,correct=evaluation_record(outputs,labels,total,TP,FP,FN,TN,correct)
    evaluation_cal(TP,FP,FN,TN,total,correct)
    confu_matrix(outputs,labels)
for epoch in range(3):#反复练个3次
    train(epoch)
    test()
    print('####################################################\n')




"""———————————————————————————————我 是 分 割 线—————————————————————————————————"""

"""
学习记录：
    对于测评指标的基本理解：(多分类问题)
        accuracy反映的是总体预测值和标签值匹配的概率，但是反映的信息不够全面，我们如果想
        了解某个分类是不是更容易被预测出来，或者更容易被误预测等等，就需要其他指标来测评
        precision反映的是某一个分类在所有的预测值中，预测对了的概率
        recall反映的是标签中的某一分类，被正确预测出的概率，体现这个分类是否容易被预测
        F1-score是上面两个的调和平均数。
    对于全连接神经网络的理解：
        根据我个人的理解，神经网络的作用就是找出输入数据和输出数据之间的映射关系，方法是通过
        反向计算梯度，也就是损失(预测值与标签值的差距，这个差距也有多种计算方式)对权重的求导
        w=w-grad*lr,通过反向传播来逐渐减小损失的大小，是预测值向标签值逼近。
    对模型层数的理解：
        增大隐藏层的层数可以提高学习能力。因为增加层数就是增加了权重数量，模型更加复杂，
        对输入数据的变化也会更加敏感，但同时也可能由于过于敏感，对噪声的反映也很强烈，
        这就不是我们想要的结果。我们想要的是模型对输入数据敏感，但对噪声忽视，
        所以模型层数不能太多，也不能太少。
    对全连接层处理图片问题的理解：
        mnist分类问题里，为了使线性层可以对图片数据进行处理，把一个二维矩阵(28,28),
        转化成了一个行向量(784)，这种方法虽然利于模型处理数据，但却破坏了图片的空间结构，
        比如人的头和脖子连接在一起，这种空间上的关系在二维矩阵上可以看出，但在一维向量上
        就没办法表示这种空间上的关系。而这种空间关系往往还是图片数据处理的关键，所以
        全连接层处理mnist问题的accuracy有个极限值，很难再提高了。
        


"""




























