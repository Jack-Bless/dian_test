# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 20:28:31 2024

@author: 21624
"""

import torch
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
#多头的attention，输入的是4*3的矩阵
x_data=torch.rand(4,3)

"""———————————————————————————————我 是 分 割 线—————————————————————————————————"""
"""
multi-head生成了8个头，但是后面的MQA和GQA就只用了两个头，
为了更详细的展示我对运算原理的理解，MQA和GQA的代码用的是普通的nn.linear，
没有使用多维度的张量。
"""
class Multi_head(torch.nn.Module):
    def __init__(self):
        super(Multi_head,self).__init__()
        self.x_q=torch.nn.Linear(3,24)
        self.x_k=torch.nn.Linear(3,24)
        self.x_v=torch.nn.Linear(3,24)
        self.softmax=torch.nn.Softmax()#注意力权重矩阵归一化
    def forward(self,x_data):
        q=self.x_q(x_data)
        k=self.x_k(x_data)
        v=self.x_v(x_data)
        q=q.view(8,4,3)#8个头，每个头都是(4*3)的矩阵
        k=k.view(8,4,3).permute(0,2,1)
        v=v.view(8,4,3)
        a=torch.matmul(q,k)
        b=torch.matmul(a,v)
        return a[0].data,b.data#返回一个注意力权重矩阵
        
multi_head=Multi_head()
t,_=multi_head(x_data)
plt.imshow(t, cmap='hot', interpolation='nearest', aspect='auto')#画矩阵
plt.colorbar()#显示颜色条
plt.title('attention')#设置标题
plt.show()

"""———————————————————————————————我 是 分 割 线—————————————————————————————————"""


class Grouped_query(torch.nn.Module):
    def __init__(self):
        super(Grouped_query,self).__init__()
        
        self.x_q1=torch.nn.Linear(3,3)
        self.x_q2=torch.nn.Linear(3,3)
        self.x_q3=torch.nn.Linear(3,3)
        self.x_q4=torch.nn.Linear(3,3)
        
        self.x_k1=torch.nn.Linear(3,3)
        self.x_k2=torch.nn.Linear(3,3)
        
        self.x_v1=torch.nn.Linear(3,3)
        self.x_v2=torch.nn.Linear(3,3)
        
        self.b1b2=torch.nn.Linear(3,3)
        
        self.softmax=torch.nn.Softmax()
    def forward(self,x_data):
        
        k1=self.x_k1(x_data)
        k2=self.x_k2(x_data)
      
       
        group1=self.x_q1(x_data)+self.x_q2(x_data)
        group2=self.x_q3(x_data)+self.x_q4(x_data)
        
        a1=torch.mm(group1,k1.t())
        a1=self.softmax(a1)
        a2=torch.mm(group2,k2.t())
        a2=self.softmax(a2)
        
        v1=self.x_v1(x_data)
        v2=self.x_v2(x_data)
        
        b1=torch.mm(a1,v1)
        b2=torch.mm(a2,v2)
        
        b=self.b1b2(b1+b2)
        
        return a1.data,b.data
    
grouped_query=Grouped_query()
t,_=grouped_query(x_data)
plt.imshow(t, cmap='hot', interpolation='nearest', aspect='auto')#画矩阵
plt.colorbar()#显示颜色条
plt.title('attention')#设置标题
plt.show()




"""———————————————————————————————我 是 分 割 线—————————————————————————————————"""

class Multi_query(torch.nn.Module):
    def __init__(self):
        super(Multi_query,self).__init__()
        self.x_q1=torch.nn.Linear(3,3)
        self.x_q2=torch.nn.Linear(3,3)
        self.x_q3=torch.nn.Linear(3,3)
        self.x_q4=torch.nn.Linear(3,3)
        
        self.x_k=torch.nn.Linear(3,3)
        
        self.x_v=torch.nn.Linear(3,3)
        
        self.softmax=torch.nn.Softmax()
    def forward(self,x_data):
        
        k=self.x_k(x_data)
        
        group=self.x_q1(x_data)+self.x_q2(x_data)+self.x_q3(x_data)+self.x_q4(x_data)
       
        a=torch.mm(group,k.t())
        a=self.softmax(a)
        
        v=self.x_v(x_data)
        
        b=torch.mm(a,v)
        
        return a.data,b.data
    
multi_query=Multi_query()
t,_=multi_query(x_data)
plt.imshow(t, cmap='hot', interpolation='nearest', aspect='auto')#画矩阵
plt.colorbar()#显示颜色条
plt.title('attention')#设置标题
plt.show()


"""———————————————————————————————我 是 分 割 线—————————————————————————————————"""
"""
学习记录：
    对于self-attention的理解：
        self-attention和RNN有相似之处，他们都是为了探求样本之间的顺序关系，不同于普通
        全连接层只关注单个样本和标签的关系，self-attention还关注了样本之间的关系
        并且他相对于RNN改进的地方在于，self不会随着样本间的距离变远而导致关联性下降，
        也就是说self可以无视距离，第一样本也可以和最后一个样本进行联系。
    对于multi-head的理解：
        普通的self-attention只有一个头，也就是一个输入数据对应一个query，一个key,
        一个value.而对于multi-head，一个输入数据对应了多个queries(复数严谨)，每个
        queries对应了不同的keye和value。这样增多了模型的权重个数，就像我之前写的
        全连接层隐藏层个数的作用一样，可以增强学习效果，但代价是速度会减慢，且还会有
        过拟合的风险。
    对于multi-query的理解：
        与multi-head类似，一个输入数据有多个queries，但是对应的key和value只有一个，
        增大了运算速度，但学习效果不如multi-head,也有可能有欠拟合的风险
    对于grouped-query的理解：
        算是上面两种的一个综合，既兼顾了速度，也有准确性。他将输入数据的多个queries
        进行分组，每组对应一个key和value。


"""






































