# project-catalog

Idea is cheap, show me the code.

## 1. 中文部分 (See the bottom for English part)

## 项目

### (1) **句子与文本分类**

#### syntac_model_deco: 以词为特征的文本分类（语义分析）神经网络模型：句法分析加词性限定挑选特征词

#### [文本分类](https://github.com/mediaProduct2017/topic-classifier)
用的是基于KNN思想和EMD测度的一个机器学习算法，论文发表于2015年，训练所需的时间少，在长文本分类方面相对其他算法有优势。Reddit data的5个类别的分类，正确率在60-70%

#### [图像识别](https://github.com/mediaProduct2017/image-classification)
使用基本的CNN进行图片分类与识别（10个类别），正确率接近70%，数据来源是CIFAR-10 dataset

#### MLP_model_deco, mlp_model_new: 以词为特征的文本分类（语义分析）神经网络模型：使用jieba的限定词性的关键词提取方法


### (2) 问答系统框架

chatbot (dialogue manager), context (agent), intent processor (belief tracker)

通过追问精确识别意图，比如听一首歌，去一个地方等。

### (3) **自然语言问句转换为sparql等查询语言、自然语言问句转换为entity和attribute意图**

#### nl2spar_deco, develop_deco, realse_knowledge_graph_temp: 自然语言转换为sparql语言（句法分析）：句法树规则，对应图结构表达式
问题类与句法模板类的交互（复杂的地方）

从句法模板到图结构表达式（创新点）

从图结构表达式到sparql语句（quepy的创新点）

#### parse_tree: 可用于自我学习“自然语言转换为sparql语言”的句法树向量，可作为机器学习模型的特征之一

#### 基于语义的特定模板之句法规则，用于把自然语言转化为sql查询语言
句法模板，把一句话转变为另一句话，即同义自然语言句子之间的转换

### (4) 短文本中的关系抽取、推理式关系抽取，知识图谱中的推理

### (5) 实体抽取

### (6) 自然语言生成、风格变换等

#### [文本生成器](https://github.com/mediaProduct2017/tv-script-decoder)
LSTM decoder

#### [机器翻译](https://github.com/mediaProduct2017/language-translation-seq2seq)
Seq2seq model

#### [图片生成](https://github.com/mediaProduct2017/face-generation-GAN)
GAN model

### (7) 词向量、句向量的生成与评测

### (8) 机器学习模型用于连续变量的预测

#### [共享单车使用预测](https://github.com/mediaProduct2017/bike-rentals)
用的是一个简单的neural network, a fully connected network，output layer使用$f(x)=x$做activation

### (9) 数据预处理

#### [训练狗的网站用户付费概率预测](https://github.com/mediaProduct2017/dataAnalysis)
首先需要对数据做预处理，然后是数据仓库的构建，最后才是数学建模

#### [中文文本预处理及词云分析](https://github.com/mediaProduct2017/text-reco)

### (10) 数据库的使用

#### [logs-analysis](https://github.com/mediaProduct2017/logs-analysis)

#### redis_mysql_deco: Redis缓存，以及从Redis到Mysql的数据转移（工业级别）
主要是注意降低网络操作的频率，包括建立网络连接，查询数据库等crud操作(create, read, update, delete)

### (11) 数据爬虫

#### [Download news data by a web crawler for reading](https://github.com/mediaProduct2017/reading)

### (17) web开发，网络通信

#### [前端开发](https://github.com/mediaProduct2017/portfolio_site)

#### [后端开发](https://github.com/mediaProduct2017/item-catalog)


## 文章与笔记

### (1) 句子与文本分类

#### [text-classification](https://github.com/mediaProduct2017/text-classification)

#### [关于神经网络模型](https://github.com/mediaProduct2017/learn_NeuralNet)
The repository中有例子：

第一个例子

hdr.py

使用简单的neural net来做模型，MNIST dataset，手写数字识别

第二个例子

神经网络模型用于连续变量预测

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

#### 学习资料

[ZhitingHu/logicnn](https://github.com/ZhitingHu/logicnn)

把规则编码到神经网络参数的选取中

### (2) 问答系统框架

#### 学习资料

[KB-InfoBot: A dialogue bot for information access](https://github.com/MiuLab/KB-InfoBot)

在追问的选择方面有突破，在问答框架方面中规中矩

[RasaHQ/rasa_nlu](https://github.com/RasaHQ/rasa_nlu)

### (5) 实体抽取

#### [word window的分类](https://github.com/mediaProduct2017/learn-WordWindow)
有很多实际问题都是word window的分类问题，比如词性判断、实体识别等

### (7) 词向量、句向量的生成与评测

#### [关于word2vec](https://github.com/mediaProduct2017/learn-word2vec)

### (10) 数据库的使用

#### SQL课程与笔记

### (12) 算法课程与leetcode

### (13) 机器学习课程与笔记

#### [learn-clustering](https://github.com/mediaProduct2017/learn-clustering)

### (14) 文本处理的python技能

#### [learn-python](https://github.com/mediaProduct2017/learn-python)

### (15) Linux与系统工具

#### [learn-conda](https://github.com/mediaProduct2017/learn-conda)
conda, git, jupyter notebook, pycharm

### (16) 并行编程、多线程、多进程

#### [learn-ParallelProgram](https://github.com/mediaProduct2017/learn-ParallelProgram)

### (18) 分布式系统

### (19) 统计学

### (20) 数据可视化

### (21) 计算机体系架构，操作系统

### (22) 代码风格与OOP设计

### (23) Java, C/C++


## 2. English part

## Projects


## Articles and notes
