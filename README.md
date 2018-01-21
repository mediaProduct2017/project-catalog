# project-catalog

Idea is cheap, show me the code.

## 1. 中文部分 (See the bottom for English part)

## 项目

### (1) 句子与文本分类

#### [文本分类](https://github.com/mediaProduct2017/topic-classifier)
用的是基于KNN思想和EMD测度的一个机器学习算法，论文发表于2015年，训练所需的时间少，在长文本分类方面相对其他算法有优势。Reddit data的5个类别的分类，正确率在60-70%

#### [图像识别](https://github.com/mediaProduct2017/image-classification)
使用基本的CNN进行图片分类与识别（10个类别），正确率接近70%，数据来源是CIFAR-10 dataset

#### MLP_model_deco: 以词为特征的文本分类（语义分析）神经网络模型：使用jieba的限定词性的关键词提取方法

#### syntac_model_deco: 以词为特征的文本分类（语义分析）神经网络模型：句法分析加词性限定挑选特征词

### (2) 问答系统框架

### (3) 自然语言问句转换为sparql等查询语言、自然语言问句转换为entity和attribute意图

#### nl2spar_deco: 自然语言转换为sparql语言（句法分析）：句法树规则，对应图结构表达式

#### parse_tree: 可用于自我学习“自然语言转换为sparql语言”的句法树向量，可作为机器学习模型的特征之一

#### 基于语义的特定模板之句法规则，用于把自然语言转化为sql查询语言

### (4) 短文本中的关系抽取

### (5) 实体抽取

### (6) 自然语言生成、风格变换等

#### [文本生成器](https://github.com/mediaProduct2017/tv-script-decoder)
LSTM decoder

### (7) 词向量、句向量的生成与评测

### (8) 机器学习模型用于连续变量的预测

#### [共享单车使用预测](https://github.com/mediaProduct2017/bike-rentals)
用的是一个简单的neural network, a fully connected network，output layer使用$f(x)=x$做activation

### (9) 数据预处理

#### [训练狗的网站用户付费概率预测](https://github.com/mediaProduct2017/dataAnalysis)
首先需要对数据做预处理，然后是数据仓库的构建，最后才是数学建模

### (10) 数据库的使用

#### redis_mysql_deco: Redis缓存，以及从Redis到Mysql的数据转移（工业级别）


## 文章与笔记

### (1) 句子与文本分类

#### [关于神经网络模型](https://github.com/mediaProduct2017/learn_NeuralNet)
The repository中有例子：

第一个例子

hdr.py

使用简单的neural net来做模型，MNIST dataset，手写数字识别

第二个例子

[共享单车使用预测](https://github.com/mediaProduct2017/bike-rentals)

#### [关于情感分析](https://github.com/mediaProduct2017/learn-SentimentAnalysis)
The repository中有例子：

第一个例子

Sentiment_Classification_Solutions.ipynb

使用简单的neural net来做模型，进行电影评论的情感分析，输入主要用的是one-hot encoded sentence.

第二个例子

Sentiment_RNN_Solution.ipynb

使用的数据与第一个例子相同，使用的模型是RNN with LSTMs，输入用的是word embeddings.

#### [关于卷积神经网络](https://github.com/mediaProduct2017/learn-CNN)

第一个例子

main.py

使用convolutional neural net来做模型，MNIST dataset，手写数字识别

#### [关于循环神经网络](https://github.com/mediaProduct2017/learn-RecurrentNN)

第一个例子

Anna_KaRNNa.ipynb

Character-wise RNN，使用的模型是RNN with LSTMs，两层LSTMs，然后连softmax classifier，用于学习英语，根据学到的经验组词造句。

第二个例子

Sentiment_RNN_Solution.ipynb

电影评论的情感分析，使用的模型是RNN with LSTMs，一层LSTM，然后连logistic classifier，输入用的是word embeddings.

#### [模型的evaluate与validate](https://github.com/mediaProduct2017/learn_evaluate_validate)

### (2) 问答系统框架

[KB-InfoBot: A dialogue bot for information access](https://github.com/MiuLab/KB-InfoBot)

### (7) 词向量、句向量的生成与评测

#### [关于word2vec](https://github.com/mediaProduct2017/learn-word2vec)

### (5) 实体抽取

#### [word window的分类](https://github.com/mediaProduct2017/learn-WordWindow)
有很多实际问题都是word window的分类问题，比如词性判断、实体识别等


## 2. English part

## Projects


## Articles and notes
