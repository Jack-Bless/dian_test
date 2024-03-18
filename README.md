# dian_test
dian团队算法方向招生题

本来顺手就写在代码最后面，但是怕不方便看，就又写了一遍，同时完善了一下

# content
-[第一部分（全连接层）](#学习记录：(第一部分题目：全连接层))

-[第二部分（RNN网络）](#学习记录：(第二部分题目：RNN网络))

-[第三部分（Attention）](#学习记录：(第三部分题目：Attention))

# 学习记录：(第一部分题目：全连接层)

   **对于测评指标的基本理解：(多分类问题)**
    
  accuracy反映的是总体预测值和标签值匹配的概率，但是反映的信息不够全面，我们如果想
        
  了解某个分类是不是更容易被预测出来，或者更容易被误预测等等，就需要其他指标来测评
        
  precision反映的是某一个分类在所有的预测值中，预测对了的概率
        
  recall反映的是标签中的某一分类，被正确预测出的概率，体现这个分类是否容易被预测
        
  F1-score是上面两个的调和平均数，算是一个综合考量。

  *这边贴一张我的测试截图，可以看到每个分类单独计算指标*

  ![image](https://github.com/szddzzy/dian_test/blob/main/images/test_pic1.png)

        
  **对于全连接神经网络的理解:**
    
  根据我个人的理解，神经网络的作用就是找出输入数据和输出数据之间的映射关系，方法是通过
        
  反向计算梯度，也就是损失(预测值与标签值的差距，这个差距也有多种计算方式)对权重的求导
        
  w=w-grad*lr,通过反向传播来逐渐减小损失的大小，是预测值向标签值逼近。

  *下面是一张简单的 loss-weight 函数*

  ![image](https://github.com/szddzzy/dian_test/blob/main/images/test_pic2.png)

        由这张图可以很直观地了解loss和weight之间的关系，每一次梯度更新实际上就是使weight向能够让

        loss减小的方向移动，最后找出让loss最小的weight，也即找出极小值点。

        但在实际中，往往极小值点并不是最小值点。而各种优化器就是为我们解决之一难题的。
        
  **对模型层数的理解：**
    
  增大隐藏层的层数可以提高学习能力。因为增加层数就是增加了权重数量，模型更加复杂，
        
  对输入数据的变化也会更加敏感，但同时也可能由于过于敏感，对噪声的反映也很强烈，
        
  这就不是我们想要的结果。我们想要的是模型对输入数据敏感，但对噪声忽视，
        
  所以模型层数不能太多，也不能太少。

  *下面两张图片是我分别采用较多隐藏层和没有隐藏层的可视化矩阵*

  ![image](https://github.com/szddzzy/dian_test/blob/main/images/test_pic3.png)

  ![image](https://github.com/szddzzy/dian_test/blob/main/images/test_pic4.png)

        可以看出，两者还是有显著差别
        
  **对全连接层处理图片问题的理解：**
    
  mnist分类问题里，为了使线性层可以对图片数据进行处理，把一个二维矩阵(28,28),
        
  转化成了一个行向量(784)，这种方法虽然利于模型处理数据，但却破坏了图片的空间结构，
        
  比如人的头和脖子连接在一起，这种空间上的关系在二维矩阵上可以看出，但在一维向量上
        
  就没办法表示这种空间上的关系。而这种空间关系往往还是图片数据处理的关键，所以
        
  全连接层处理mnist问题的accuracy有个极限值，很难再提高了。

  *实现图片数据维度的转换也很简单，下面一行代码就搞定了*

        x=x.view(-1,784)
        
        #将原本的图片数据大小(N,28,28)转换为矩阵(N,784)，便于模型处理

  *下面是转换为向量的图片数据*

  ![image](https://github.com/szddzzy/dian_test/blob/main/images/test_pic9.png)

        可以看到，矩阵数据被转换为向量，失去了孔吉纳关系，

        对训练效果产生影响。

# 学习记录：(第二部分题目：RNN网络)

  **对RNN网络的理解：**
    
  RNN网络与基础的全连接层的巨鳖就在于RNN网络引入了上一个样本与这一个样本之间的关系，
        
  这样就可以体现出数据之间的顺序关系。具体的操作就是把上一次隐藏层中的数据留存下来，
        
  并且作为这一次输入数据的一部分，和输入数据融合，一起计算这一次的隐藏层数据，实现
        
  两次运算的关联，从而推理出数据之间的相关性。

  *公式贴在下面*

  ![image](https://github.com/szddzzy/dian_test/blob/main/images/v2-9524a28210c98ed130644eb3c3002087_r.png)

        每一次的数据都与上一次有所关联

  **下面两行代码实际上就是整个RNN模型最关键的地方**

        combine_data=self.linear1(x_data)+hide_data#融合上一次的隐藏层数据和这一次的输入数据
        hide_data=self.tanh(combine_data)#计算这一次的隐藏层数据
        
  **RNN的缺陷：**
    
  RNN网络可以对数据之间的先后顺序进行学习，但是也有局限性，比如随着样本不断计算，
        
  第一个样本和最后一个样本之间的关联会逐渐不被注意。并且由于不能并行运算，RNN的速度
        
  对比其他模型也要满上很多。而Self-attention机制实际上就很好的解决这两个问题，
        
  它可以无视这种距离的远近对关联度计算的影响，并且可以并行计算。下一个题目里有
        
  我对attention的理解。
        
  **对本题的理解：**
    
  因为fashion-mnist对图片顺序并没有什么要求，所以用RNN网络进行训练优势不大，
        
  我用基础的全连接层也试了一下，两者的正确率区别不大，而且RNN更耗时间。

  因为上述的各种原因，对fashion-mnist的训练和测试准确率只能维持在85%左右。

  另外，对我来说话有个新发现：

  batch的选择并不是越小越好(我之前一直以为batch选的小，训练次数多，效果就好)

  *图1的batch取640，图二取64(均取第3次训练结果)*

![image](https://github.com/szddzzy/dian_test/blob/main/images/test_pic5.png)

![image](https://github.com/szddzzy/dian_test/blob/main/images/test_pic6.png)

        可以看到，图一的损失要略微的低于图二，训练速度更是快得多，

        可见batch的选取也不是越小越好

# 学习记录：(第三部分题目：attention)

  **对于self-attention的理解：**
  
  self-attention和RNN有相似之处，他们都是为了探求样本之间的顺序关系，不同于普通
        
  全连接层只关注单个样本和标签的关系，self-attention还关注了样本之间的关系
        
  并且他相对于RNN改进的地方在于，self不会随着样本间的距离变远而导致关联性下降，
        
  也就是说self可以无视距离，第一样本也可以和最后一个样本进行联系。

  *对于self-attention来说，最关键的就是下面这行代码(k已经过转置)*

         a=torch.matmul(q,k)

         这计算出来的就是所谓的注意力权重，在后面会详写

  *先贴一个公式*

  ![image](https://github.com/szddzzy/dian_test/blob/main/images/16d50ead3d0a4db0aee8e72bcff3098b.jpg)

  **关于Q,K,V的理解：**

  ![image](https://github.com/szddzzy/dian_test/blob/main/images/q.png)

  ![image](https://github.com/szddzzy/dian_test/blob/main/images/k.png)

  ![image](https://github.com/szddzzy/dian_test/blob/main/images/v.png)

  q,k,v均由输入数据线性变化得到，那么问题是，在注意力权重的公式中，Q*K.T又是什么意思？

  矩阵的相乘实际上就是*第一个矩阵的行向量与第二个矩阵的列向量求内积*，而向量内积的大小

  实际上可以反映两个向量的角度大小，极限情况，两个向量垂直内积就是0。所以向量的内积可以

  用来表示两者的相关度。这就好理解q*k.t实际上就是每个样本的query(矩阵Q的每一行可以看作
  
  一个样本，即单词，的query)与所有样本的key的相关度,而query和key的相关度就能转换为单词

  之间的相关度。所以Q*K.T就是注意力权重，防止他过大，会除以个dk。
        
  **对于multi-head的理解：**
  
  普通的self-attention只有一个头，也就是一个输入数据对应一个query，一个key,
        
  一个value.而对于multi-head，一个输入数据对应了多个queries(复数严谨)，每个
        
  queries对应了不同的keye和value。这样增多了模型的权重个数，就像我之前写的
        
  全连接层隐藏层个数的作用一样，可以增强学习效果，但代价是速度会减慢，且还会有
        
  过拟合的风险。

  *下面是multi-head的原理图*

  ![image](https://github.com/szddzzy/dian_test/blob/main/images/Screenshot_2024-03-15-22-50-55-96_149003a2d400f6a.jpg)

        这张图解释了具体的运算细节，每个头都是由一个query，key，value分出来的，且对应的query与对应的

        key进行运算。

  *下面这张图是这MHA模型的其中一个头的注意力权重矩阵*

![image](https://github.com/szddzzy/dian_test/blob/main/images/test_pic7.png)

  **代码细节：**

  在*multi_head*的代码里，我写了8个头，实现方法如下：

      self.x_q=torch.nn.Linear(3,24)

      #这一步先将q变大，方便后续分出多个头

      q=self.x_q(x_data)

      q=q.view(8,4,3)#8个头，每个头都是(4*3)的矩阵

      #然后这里把原本一个很大的q分成了8个(4，3)矩阵，

      大小与key相同。q的维度也就变成了(8,4,3)。

  *这种方法可以有效的节省代码量，而在后面MQA和GQA的代码中，没有使用这种分头方法，*
  
  *而是按照李宏毅老师的讲法，直接对同一个样本进行多个线性变换，得到多个头，这样更能展现对模型的理解*
      
  **对于multi-query的理解：**
  
  与multi-head类似，一个输入数据有多个queries，但是对应的key和value只有一个，
        
  增大了运算速度，但学习效果不如multi-head,也有可能有欠拟合的风险

  **代码细节：**

  所有的头分成一个组，也只有一个key和value

        group=self.x_q1(x_data)+self.x_q2(x_data)+self.x_q3(x_data)+self.x_q4(x_data)

        #前面线性变换得到的头融合为一个组

        a=torch.mm(group,k.t())

        #将这个组和对应的key计算注意力
        
  **对于grouped-query的理解：**
  
  算是上面两种的一个综合，既兼顾了速度，也有准确性。他将输入数据的多个queries
        
  进行分组，每组对应一个key和value。

  **代码细节：**

  与MQA不一样的是所有的头分成了多个组，每个组都有对应的key和value

        group1=self.x_q1(x_data)+self.x_q2(x_data)
        
        group2=self.x_q3(x_data)+self.x_q4(x_data)
        
        #将前面通过线性变换得到的4个头分成两组
        
        a1=torch.mm(group1,k1.t())
        
        a1=self.softmax(a1)
        
        a2=torch.mm(group2,k2.t())
        
        a2=self.softmax(a2)

        #每组query在与对应的k计算注意力





