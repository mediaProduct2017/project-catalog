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

#### MLP_model_deco, mlp_model_new: 以词为特征的文本分类（语义分析）神经网络模型：使用jieba的限定词性的关键词提取方法；excel读取及数据统计


### (2) 问答系统框架

#### nlu2: 利用rasa_nlu做意图识别和实体抽取，然后可以对应到某一类查询语句上并传递参数

chatbot (dialogue manager), context (context agent), intent processor (belief tracker)

通过追问精确识别意图，比如听一首歌，去一个地方等。

检索式问答系统（搜索引擎、数据库查询），任务式问答系统、闲聊系统

检索式问答系统：Sentence alignment (ElasticSearch)和sentence similarity classification用于问句分类；问答对可以从数据库中自动产生（自然语言问句转化为查询语言），问答对中的问句可以先清除掉stop words，再放入ES，用户的问句也可以先清除掉停用词，再进入ES做匹配

belief tracker也是一个pipeline，包括意图识别、实体抽取等多个模块，每个模块都可以用机器学习方法训练

### (3) **自然语言问句转换为sparql等查询语言、自然语言问句转换为entity和attribute意图**

#### release_kg, nl2spar_deco, develop_deco, realse_knowledge_graph_temp: 自然语言转换为sparql语言（句法分析）：句法树规则，对应图结构表达式
问题类与句法模板类的交互（复杂的地方）

release_kg: 从句法模板到图结构表达式（创新点1）

release_kg:从分词后列表模板到图结构表达式（创新点2）

kg_clean: 先清洗不必要的词再进入句法模板（创新点3，减少不必要的虚词对句法分析的干扰）

先做实体抽取后再进入句法模板（减少实体名字对句法分析的干扰）

从图结构表达式到sparql语句（quepy的创新点）

[quepy](https://github.com/machinalis/quepy)

#### parse_tree: 可用于自我学习“自然语言转换为sparql语言”的句法树向量，可作为机器学习模型的特征之一

#### 基于语义的特定模板之句法规则，用于把自然语言转化为sql查询语言
句法模板，把一句话转变为另一句话，即同义自然语言句子之间的转换

### (4) 短文本中的关系抽取、推理式关系抽取，知识图谱中的推理

机器阅读理解的数据集与比赛

### (5) 实体抽取

### (6) 自然语言生成、风格变换等

#### [文本生成器](https://github.com/mediaProduct2017/tv-script-decoder)
LSTM decoder

Seq2seq decoder正在取代传统的语言模型（预测一个句子出现的概率）。在传统的n-gram model中，一个句子的概率等于各个词出现的条件概率，如果是2-gram model，那么一个词出现的概率就只与前一个词相关。n-gram model又叫做n-1阶马尔科夫模型，当前词的出现概率只与前面n-1个词相关。

An n-gram is a continuous sequence of n items from a given sample of text or speech. The items can be letters, words or base pairs according to the application. Facebook的fastText分类算法就是对n-gram words的很好的应用（在中文中，也可以用在n-gram characters上面）。

通常所说的语言模型其实是语言生成模型，即n-gram模型，即可以给出一个生成的句子中各个词出现的概率（句子的概率也就可以相乘得到），每个词的概率都是仅由前面几个词决定的，但Seq2seq decoder出现后，语言生成模型更多的使用深度学习模型，每个词的概率不仅由前一个词决定，还和前面所有的词都相关，是一种更好的生成模型。

语言生成模型在语音识别、机器翻译、分词、词性标注、句法分析等自然语言处理的基础领域都是有用的。

#### [机器翻译](https://github.com/mediaProduct2017/language-translation-seq2seq)
Seq2seq model

#### [图片生成](https://github.com/mediaProduct2017/face-generation-GAN)
GAN model

### (7) 词向量、句向量的生成与评测

#### word_vec_deco2: 从fasttext vector中查询某个词的对应向量，并用PCA做可视化评估

### (8) 机器学习模型用于连续变量的预测

#### [共享单车使用预测](https://github.com/mediaProduct2017/bike-rentals)
用的是一个简单的neural network, a fully connected network，output layer使用$f(x)=x$做activation

### (9) 数据预处理

#### [训练狗的网站用户付费概率预测](https://github.com/mediaProduct2017/dataAnalysis)
首先需要对数据做预处理，然后是数据仓库的构建，最后才是数学建模

#### [中文文本预处理及词云分析](https://github.com/mediaProduct2017/text-reco)

#### mlp_model_new: 0. Excel文件的读入；1. 模板及其分类的统计；2. 球员名字、国家队名字、俱乐部名字的统计 

### (10) 数据库的使用

#### [logs-analysis](https://github.com/mediaProduct2017/logs-analysis)

#### redis_mysql_deco: Redis缓存，以及从Redis到Mysql的数据转移（工业级别）
主要是注意降低网络操作的频率，包括建立网络连接，查询数据库等crud操作(create, read, update, delete)

### (11) 数据爬虫

#### [Download news data by a web crawler for reading](https://github.com/mediaProduct2017/reading)

### (17) web开发，网络通信

#### [前端开发](https://github.com/mediaProduct2017/portfolio_site)

#### [后端开发](https://github.com/mediaProduct2017/item-catalog)


## 文章与笔记（基础知识）

### (1) 句子与文本分类

#### 问答系统中belief tracker的意图识别（entity类的选择）、实体抽取问题（entity已知或需知属性的选择）和agent的状态判定问题

#### 问答系统中belief tracker的entity类所问属性的选择

#### 问答系统中dialogue manager追问还是回答的选择，以及具体追问的选择

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

把规则编码到神经网络参数的选取中：可微分编程，把if, else和其他逻辑语句用便于快速迭代优化的方式表达出来。最典型的例子是logistic regression的objective function，本来是if, else形式的，标签为1或0时，要优化的函数是不同的，通过巧妙的设计，可以把if, else合成一个表达式。这个项目做的是同样的事情，只是要表达的逻辑更加广泛，不只是上面的if, else形式，还包括多种复杂的逻辑结构，最终生成统一的objective function用于优化，用back propagation来优化。

[fastText原理及实践](http://www.52nlp.cn/fasttext)

[fastText python](https://github.com/facebookresearch/fastText/tree/master/python)

[scikit-learn](https://github.com/scikit-learn/scikit-learn)

[pyltp](https://github.com/HIT-SCIR/pyltp)

[jieba](https://github.com/fxsjy/jieba)

### (2) 问答系统框架

百度：

中国的gdp，中国gdp：返回知识图谱中的图表，预计问答对中的问句原文是-中国gdp，“的”作为停用词

中的国gdp：同中国gdp，句子相似度较大

中的国的gdp：不同于中国gdp，句子相似度在阈值之下

#### 学习资料

[自然语言处理SpaCy](https://github.com/explosion/spaCy)

[自然语言处理gensim](https://radimrehurek.com/gensim/)

[爬虫scrapy](https://github.com/scrapy/scrapy)

[RasaHQ/rasa_nlu](https://github.com/RasaHQ/rasa_nlu)

belief tracker的意图分析及实体识别，之后可以用简单的sql或者sparql语句从数据库中获取相关信息（也可以组合多个查询语句构建意图处理函数），构建检索式对话系统

[RasaHQ/rasa_core](https://github.com/RasaHQ/rasa_core)

dialogue manager以及context agent，用于构建多轮对话管理系统，构建任务式对话系统

[检索系统ElasticSearch](https://github.com/elastic/elasticsearch)

[自然语言处理NLTK](http://www.nltk.org/)

[KB-InfoBot: A dialogue bot for information access](https://github.com/MiuLab/KB-InfoBot)

在追问的选择方面（也是分类问题）有突破，在问答框架方面中规中矩

[flask](https://github.com/pallets/flask)

[sqlalchemy](https://github.com/zzzeek/sqlalchemy)

### (5) 实体抽取

#### [word window的分类](https://github.com/mediaProduct2017/learn-WordWindow)
有很多实际问题都是word window的分类问题，比如词性判断、实体识别等

#### 学习资料

自然语言处理中的分词及实体识别：[HMM, MEMM, CRF](http://tripleday.cn/2016/07/14/hmm-memm-crf/)

### (7) 词向量、句向量的生成与评测

#### [关于word2vec](https://github.com/mediaProduct2017/learn-word2vec)

### (10) 数据库的使用

#### SQL课程与笔记

### (12) 算法课程与leetcode

coursera算法课

### (13) 机器学习、自然语言处理课程与笔记

#### [learn-clustering](https://github.com/mediaProduct2017/learn-clustering)

#### 李航：统计学习方法

#### 学习资料

[自然语言处理怎么最快入门](https://www.zhihu.com/question/19895141)

[我爱自然语言处理](http://www.52nlp.cn/)

[LDA](http://www.52nlp.cn/tag/lda)

[概率语言模型及其变形系列-LDA及Gibbs Sampling](http://www.52nlp.cn/%E6%A6%82%E7%8E%87%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B%E5%8F%8A%E5%85%B6%E5%8F%98%E5%BD%A2%E7%B3%BB%E5%88%97-lda%E5%8F%8Agibbs-sampling)

### (14) 文本处理的python技能

#### [learn-python](https://github.com/mediaProduct2017/learn-python)
pandas, numpy

#### [Talk Python To Me](https://talkpython.fm/)

### (15) Linux与系统工具

#### [learn-conda](https://github.com/mediaProduct2017/learn-conda)
conda, git, jupyter notebook, pycharm

### (16) 并行编程、多线程、多进程、协程

#### [learn-ParallelProgram](https://github.com/mediaProduct2017/learn-ParallelProgram)

### (18) 分布式系统

### (19) 统计学

### (20) 数据可视化
matplotlib

### (21) 计算机体系架构，操作系统

### (22) 代码风格与OOP设计

### (23) Java, C/C++


## 2. English part

## Projects


## Articles and notes


## 3. References

Fluent python及中文版, 2015

Dive into python3及中文版, 2014

The hitchhiker's guide to python, 2016

A Survey on Dialogue Systems: Recent Advances and New Frontiers, 2017

Automated Template Generation for Question Answering over Knowledge Graphs, 2017

Harnessing Deep Neural Networks with Logic Rules, 2016

Quepy Documentation, 2017

Rasa: Open Source Language Understanding and Dialogue Management, 2017

强化学习在阿里的技术演进与业务创新, 2017

A Multi-Layer System for Semantic Textual Similarity, 2016

Learning scrapy, 2016

[quepy](https://github.com/machinalis/quepy)
