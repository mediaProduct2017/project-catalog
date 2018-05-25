# project-catalog

Idea is cheap, show me the code.

## 1. 中文部分 (See the bottom for English part)

## 项目

### 自然语言处理任务

### (1) **句子与文本分类**

#### syntac_model_deco: 以词为特征的文本分类（语义分析）神经网络模型：句法分析加词性限定挑选特征词

#### [文本分类](https://github.com/mediaProduct2017/topic-classifier)
用的是基于KNN思想和EMD测度的一个机器学习算法，论文发表于2015年，训练所需的时间少，在长文本分类方面相对其他算法有优势。Reddit data的5个类别的分类，正确率在60-70%

#### [图像识别](https://github.com/mediaProduct2017/image-classification)
使用基本的CNN进行图片分类与识别（10个类别），正确率接近70%，数据来源是CIFAR-10 dataset

#### MLP_model_deco, mlp_model_new: 以词为特征的文本分类（语义分析）神经网络模型：使用jieba的限定词性的关键词提取方法；excel读取及数据统计


### (2) 问答系统框架

#### release_kg: 先对原话进行正则替换预处理（分词之前的处理），再做分词停用词预处理（中文独有特点）；或者先做分词停用词预处理，再做正则替换预处理（英文或中文的处理办法）。

#### nlu2: 利用rasa_nlu做意图识别和实体抽取，然后可以对应到某一类查询语句上并传递参数

chatbot (dialogue manager), context (context agent), intent processor (belief tracker)

通过追问精确识别意图，比如听一首歌，去一个地方等。

检索式问答系统（搜索引擎、数据库查询），任务式问答系统、闲聊系统

检索式问答系统：Sentence alignment (ElasticSearch)和sentence similarity classification用于问句分类；问答对可以从数据库中自动产生（自然语言问句转化为查询语言），问答对中的问句可以先清除掉stop words，再放入ES，用户的问句也可以先清除掉停用词，再进入ES做匹配

belief tracker也是一个pipeline，包括意图识别、实体抽取等多个模块，每个模块都可以用机器学习方法训练

### (3) **自然语言问句转换为sparql等查询语言、自然语言问句转换为entity和attribute意图**

#### 分词与句法分析：用同一套系统的好处是，词性分析也许能认出分出的词，词性分析更准，同理句法分析也更准；用不同的分词系统的好处是，可能拥有更好更灵活的分词系统（除了给出用户自定义词汇外，jieba能针对自定义词汇给出不同权重）。

#### 意图处理和实体抽取：字符序列，词汇集合，词汇列表，句法，正则，深度学习特征

#### release_kg, nl2spar_deco, develop_deco, realse_knowledge_graph_temp: 自然语言转换为sparql语言（句法分析）：句法树规则，对应图结构表达式
问题类与句法模板类的交互（复杂的地方）

release_kg: 从句法模板到图结构表达式（创新点1）

release_kg:从分词后列表模板到图结构表达式（创新点2）

kg_clean, release_kg: 先清洗不必要的词再进入句法模板（创新点3，减少不必要的虚词对句法分析的干扰）

先做实体抽取后再进入句法模板（减少实体名字对句法分析的干扰）

从图结构表达式到sparql语句（quepy的创新点）

[quepy](https://github.com/machinalis/quepy)

#### parse_tree: 可用于自我学习“自然语言转换为sparql语言”的句法树向量，可作为机器学习模型的特征之一

#### 基于语义的特定模板之句法规则，用于把自然语言转化为sql查询语言
句法模板，把一句话转变为另一句话，即同义自然语言句子之间的转换

### (4) 短文本中的关系抽取、推理式关系抽取，机器阅读理解，信息抽取，知识图谱中的推理

机器阅读理解的数据集与比赛

[arfu2016/DuReader](https://github.com/arfu2016/DuReader)

### (5) 实体抽取

/Users/arfu/Documents/Python_projects/PycharmProjects/CubeGirl:

git branch: worldcup_kg

CubeGirl/SourceCode/Daka/chatbot/logic/knowledge_graph/entity_lookup

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

### 自然语言处理手段

### (7) 自然语言的向量化，词向量、句向量的生成与评测，语义分析，vector encoder

#### word_vec_deco2: 从fasttext vector或其他word2vec结果中查询所关心的一系列词汇的对应向量，并用PCA做可视化评估，PCA图像可以作为评估word2vec效果的重要参考

[Module google/universal-sentence-encoder/1](https://www.tensorflow.org/hub/modules/google/universal-sentence-encoder/1)

#### mediaProduct2017/nlu-vector

### (8) tagging，词法分析，one hot encoding

### (9) parsing，句法分析

梅西的身高，梅西的体重

阿森纳的队长，梅西的女友

### (10) classifier

### (11) decoder: a special classifier

### 深度学习、机器学习与数据科学

### (12) 机器学习模型用于连续变量的预测

#### [共享单车使用预测](https://github.com/mediaProduct2017/bike-rentals)
用的是一个简单的neural network, a fully connected network，output layer使用$f(x)=x$做activation

### (13) 数据预处理

#### [训练狗的网站用户付费概率预测](https://github.com/mediaProduct2017/dataAnalysis)
首先需要对数据做预处理，然后是数据仓库的构建，最后才是数学建模

#### [中文文本预处理及词云分析](https://github.com/mediaProduct2017/text-reco)

#### mlp_model_new: 0. Excel文件的读入；1. 模板及其分类的统计；2. 球员名字、国家队名字、俱乐部名字的统计 

### (14) 数据库的使用

#### Postgresql:

[logs-analysis](https://github.com/mediaProduct2017/logs-analysis)

[topic-classifier/db_connect.py](https://github.com/mediaProduct2017/topic-classifier/blob/master/db_connect.py)

sqlalchemy

[item-catalog/database_setup.py](https://github.com/mediaProduct2017/item-catalog/blob/master/database_setup.py)

#### Mysql:

sqlalchemy

[dataAnalysis/1p4ETL_SQL.ipynb](https://github.com/mediaProduct2017/dataAnalysis/blob/master/1p4ETL_SQL.ipynb)

/Users/arfu/Documents/Python_projects/PycharmProjects/CubeGirl:

git branch: redis_mysql_deco: Redis缓存，以及从Redis到Mysql的数据转移（工业级别）

CubeGirl/SourceCode/Daka/chatbot/logic/text_table

主要是注意降低网络操作的频率，包括建立网络连接，查询数据库等crud操作(create, read, update, delete)

#### Redis

#### MongoDB

### (15) 数据爬虫

#### [Download news data by a web crawler for reading](https://github.com/mediaProduct2017/reading)

### Computer Science and Math

### (27) web开发，网络通信

#### [前端开发](https://github.com/mediaProduct2017/portfolio_site)

#### [后端开发](https://github.com/mediaProduct2017/item-catalog)

### (28) 测试

测速，timeit

/Users/arfu/Documents/Python_projects/PycharmProjects/CubeGirl:

git branch: worldcup_kg

CubeGirl/SourceCode/Daka/chatbot/logic/knowledge_graph/one_hot_encoding/test_MLP.py

单元测试

/Users/arfu/Documents/Python_projects/PycharmProjects/CubeGirl:

git branch: develop_deco

CubeGirl/SourceCode/Daka/chatbot/logic/text_table/syntactic_tree/test_syntactic_template.py

句法模板分类测试

/Users/arfu/Documents/Python_projects/PycharmProjects/CubeGirl:

git branch: worldcup_kg

CubeGirl/SourceCode/Daka/chatbot/logic/knowledge_graph/one_hot_encoding/test_questions.py

## 文章与笔记（基础知识）

### 自然语言处理任务

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

### 自然语言处理手段

### (7) 词向量、句向量的生成与评测

#### [关于word2vec](https://github.com/mediaProduct2017/learn-word2vec)

### 深度学习、机器学习与数据科学

### (14) 数据库的使用

#### SQL课程与笔记

[Python DB-API](https://classroom.udacity.com/courses/ud032)

#### Redis笔记

#### [mongoDB课程与笔记](https://github.com/mediaProduct2017/mongoDB_examples)

[使用 MongoDB](https://classroom.udacity.com/courses/ud032)

### (16) 机器学习、自然语言处理课程与笔记

#### [learn-clustering](https://github.com/mediaProduct2017/learn-clustering)

#### 李航：统计学习方法

#### 学习资料

[自然语言处理怎么最快入门](https://www.zhihu.com/question/19895141)

[我爱自然语言处理](http://www.52nlp.cn/)

[LDA](http://www.52nlp.cn/tag/lda)

[概率语言模型及其变形系列-LDA及Gibbs Sampling](http://www.52nlp.cn/%E6%A6%82%E7%8E%87%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B%E5%8F%8A%E5%85%B6%E5%8F%98%E5%BD%A2%E7%B3%BB%E5%88%97-lda%E5%8F%8Agibbs-sampling)

### (17) 深度学习

在深度学习中，模型的参数数目略大于独立方程数（主要取决于带标签的样本数）是有必要的，因为并不是所有的信息都在训练样本中包含了，所以，给待拟合的参数多一些自由度（也就是说，并不是唯一的拟合方式），反而可能拟合出更加正确的模型。至于到底是选择哪个模型（比如hyper parameter的选择），是在validation过程中决定的。

[Deep Learning for Cancer Detection with Sebastian Thrun](https://classroom.udacity.com/nanodegrees/nd101/parts/ee36fc70-6f3e-4a9f-a5b1-fb6d03d40de4/modules/ff4ffa88-13e9-4a64-a280-c793a6cb0064/lessons/54e18898-2666-445d-ba5c-ecab62a61d00/concepts/c225888e-8e82-4020-a641-acdda4008fa5)

### (18) 统计学

协方差看的是两个或者多个向量（随机变量）两两之间的相关关系（或者独立程度）。numpy.cov()可以接受两个向量，给出一个二维矩阵作为结果；也可以接受一个矩阵，矩阵的各个列向量就是要求相关关系的各个向量，假设矩阵有n个列向量，求协方差的结果就是一个n*n的矩阵。

方差求的是一个向量（随机变量）的离散程度，求平方根后得到标准差。

两个向量的相关系数就是用它们的协方差除以二者标准差的乘积，也就是person correlation coefficient, 1表示完全正相关，-1表示完全负相关，0表示二者独立. 对应的p-value的值就是看认定相关系数不等于0而犯错的概率，这个概率越小，越能认为相关系数不等于0.

### (19) 数据可视化
matplotlib

### Computer Science and Math

### (20) 算法课程与leetcode

coursera算法课

日常用到的传统算法的总结

#### Search algorithm

应用：

1. 在查阅一个词在word2vec中是否存在，或者查找vector的时候，就要用到search algorithm。当然，不在乎内存的话，一般是把word2vec的vocabulary用dict或者set来表示，查找的时候就是哈希查找，最省时间。

2. 在字符串中查找正则表达式或者某个子串，所用的Search algorithm是string search algorithm (or called substring search algorithm)，是列表的Search algorithm的变体。

[bisect](https://docs.python.org/3/library/bisect.html): binary search的模块，查找一个词，O(log(n))时间复杂度，n是list的长度

list查找一个词的话，线性时间复杂度

set或者dict查找一个词的话，O(1)时间复杂度，用的是hash table，但是耗内存，以空间换时间

list查找多个词的话，可以把多个词放在set里边，扫描list；也可以扫描list，使用collections中的Counter()，去挨个计数list中的词；这样的话，扫描list一次，就可以把想要的结果全部拿到，时间复杂度是O(n)，与要查找的词的个数无关


#### Sort algorithm

应用：

在search algorithm中，bianry search是如此有用，但前提是，the sequence is sorted. 在这种情况下，排序算法变得非常有用。数据库软件中的indexing过程(搜索引擎中的indexing过程，因为预处理的是文本，不是列表，所以有所不同，一般是建立某种suffix tree)，本质上就是排序的过程（比如，建立binary tree和排序本质上并没有大的不同），排序之后，查找就快的多了。

*shellsort*:

希尔排序使小规模数据排序的最优选择。

Why are we interested in shellsort?

Useful in practice. Fast unless array size is huge (used for small subarrays). --bzip2, /linux/kernel/groups.c

Tiny, fixed footprint for code (used in some embedded systems). --uClibc

Shellsort overview

Idea. Move entries more than one position at a time by h-sorting the array.

How to h-sort an array? Insertion sort, with stride length h.

Why insertion sort?

在局部排序时，h-sorting是前面排好之后才去排后边，也就是说array已经部分有序了，在部分有序的情况下，insertion sort的表现是最好的，超过merge sort.

Proposition. A g-sorted array remains g-sorted after h-sorting it. (h < g)

which increment sequence to use?

N/2, N/4, N/8, ..., 1

3x + 1. 1, 4, 13, 40, 121, 364, ... (OK. Easy to compute.)

Shellsort: analysis

Proposition. The worst-case number of compares used by shellsort with the 3x+1 increments is O(N^3/2).

Property. Number of compares used by shellsort with the 3x+1 increments is at most by a small multiple of N times the # of increments used.

所以，对于小规模数据排序，3x+1 increments的个数是很小的，Number of compares也就不多。

代码解析：

第一步，选择初始的increment，比如N//2

第二步，第一重循环，要求increment>0, 选择初始的排序关注点i（比如increment），increment的变化：increment //= 2

第三步，第二重循环，要求排序关注点i小于列表长度N，选择初始的排序比较点j（比如i-increment），记录排序关注点的值，移动排序关注点的值（移动到哪个位置由insertion sort决定），增加排序关注点i的值，每次加1.

第四步，第三重循环，insertion sort，要求排序比较点j>=0，并且排序比较点的值大于记录好的排序关注点的值（否则没有insertion sort的必要），在循环内，把排序比较点的值右移，然后把排序比较点的index（j）左移。

[Timsort](https://en.wikipedia.org/wiki/Timsort): a hybrid stable sorting algorithm, derived from merge sort and insertion sort

The sorting algorithm used in sorted and list.sort is Timsort, and adaptive algorithm that switches from insertion sort to merge sort strategies, depending on how ordered the data is. This is efficient because real-world data tends to have runs of sorted items.

[fluent python timsort](https://books.google.com.hk/books?id=kYZHCgAAQBAJ&pg=PA66&lpg=PA66&dq=fluent+python+timsort&source=bl&ots=iswgRxyRQf&sig=aTSU_VLmoX8meNk3Hom26vMvX4c&hl=en&sa=X&redir_esc=y&hl=zh-CN&sourceid=cndr#v=onepage&q=fluent%20python%20timsort&f=false)

### string searching algorithm

对于正则表达式的查找，如果string的长度为L的话，较好情况下的时间复杂度是O(L)，最坏情况下的时间复杂度是指数级别的。

如果是多个正则表达式的查找的话，比如n个，按一般方法，时间复杂度是O(nL)，但是google开发出的[re2](https://github.com/google/re2)，时间复杂度可以达到O(L)，与n无关。

[Of running multiple regexp at once](https://fulmicoton.com/posts/multiregexp/)

[facebook/pyre2](https://github.com/facebook/pyre2)

[re3 0.2.23](https://pypi.python.org/pypi/re3/0.2.23)

[wikipedia string searching algorithm](https://en.wikipedia.org/wiki/String_searching_algorithm)

对于子串查询，brute force的解法就是searchable text逐个字符放到列表里，共n个字符，要查的子串有m个字符的话，逐个字符到列表里查询，这样时间复杂度就是O(mn)。或者看n个字符的m-gram，这样所需要的比较就是O(n)，只是有额外的overhead.

一般来说，不需要正则表达式的话，就不要用正则表达式查询，就用子串查询即可。正则表达式查询的速度相对较慢，因为正则不只要做匹配，还要做group-capturing和back-referencing。This implementation has exponential time complexity in worst case.

[substring match faster with regular expression?](https://stackoverflow.com/questions/3303355/substring-match-faster-with-regular-expression)

对于普通的substring searching algorithm，最常用的是Boyer–Moore string search algorithm，时间复杂度best Ω(n/m),
worst O(mn)，其中m be the length of the pattern, n be the length of the searchable text。

在python中，string的in operator用的就是Boyer–Moore string search algorithm的一个变种，严格的说，it's a combination of Boyer-Moore and Horspool.

[Python string 'in' operator implementation algorithm and time complexity](https://stackoverflow.com/questions/18139660/python-string-in-operator-implementation-algorithm-and-time-complexity)


### (21) 文本处理的python技能

#### [learn-python](https://github.com/mediaProduct2017/learn-python)
pandas, numpy

#### [Talk Python To Me](https://talkpython.fm/)

### (22) Linux与系统工具

#### [learn-conda](https://github.com/mediaProduct2017/learn-conda)
conda, git, jupyter notebook, pycharm

### (23) 并行编程、多线程、多进程、协程

#### [learn-ParallelProgram](https://github.com/mediaProduct2017/learn-ParallelProgram)

### (24) 分布式系统

### (25) 计算机体系架构，操作系统

### (26) 代码风格与OOP设计

### (29) Java, C/C++

### (30) 线性代数

矩阵的本质是线性变换，也就是旋转和伸缩。一个矩阵乘以一个列向量，就是对这个列向量进行线性变换。一个矩阵A乘以一个矩阵B，就是把矩阵B分拆成n个列向量，然后对这n个列向量进行线性变换，线性变换的方式是由矩阵A提供的。变换的旋转方向是由矩阵A的特征向量决定的，变换的伸缩方式是由各个特征向量方向上的特征值决定的。

[learn-linear-algebra](https://github.com/mediaProduct2017/learn-linear-algebra)


## 2. English part

## Projects


## Articles and notes


## 3. References

Dive into python3及中文版, 2009 (python3.0, python3.1)

Fluent python及中文版, 2015 (python3.4, python3.5)

Introduction to algorithms (CLRS)及课程, 2017

修改、标注、总结syntac_model_deco

修改、标注、总结release_kg

修改、标注、总结word_vec_deco2

The hitchhiker's guide to python, 2016

A Multi-Layer System for Semantic Textual Similarity, 2016

A Survey on Dialogue Systems: Recent Advances and New Frontiers, 2017

Automated Template Generation for Question Answering over Knowledge Graphs, 2017

Harnessing Deep Neural Networks with Logic Rules, 2016

Quepy Documentation, 2017

[quepy](https://github.com/machinalis/quepy)

Rasa: Open Source Language Understanding and Dialogue Management, 2017

强化学习在阿里的技术演进与业务创新, 2017

Learning scrapy, 2016

访谈：关于技术管理与技术领导，2018-04-30，余晟以为微信号


## 4. Core projects

### /Users/arfu/Documents/Python_projects/PycharmProjects/CubeGirl:

git branch: worldcup_kg

CubeGirl/SourceCode/Daka/chatbot/logic/knowledge_graph/one_hot_encoding

CubeGirl/SourceCode/Daka/chatbot/logic/knowledge_graph/q2answer

CubeGirl/SourceCode/Daka/chatbot/logic/knowledge_graph/entity_lookup

### Others:

[arfu2016/DuReader](https://github.com/arfu2016/DuReader)

arfu2016/DuReader/tf-hub2

[codelucas/newspaper](https://github.com/codelucas/newspaper)

[scrapy/scrapy](https://github.com/scrapy/scrapy)

[Rasa-nlu](https://github.com/RasaHQ/rasa_nlu)

[Rasa-core](https://github.com/RasaHQ/rasa_core)

英文词向量：

[nlp/nlp_models/google_retrain/](https://github.com/arfu2016/nlp/tree/master/nlp_models/google_retrain)

[nlp/nlp_models/retrain_new_vocabulary/](https://github.com/arfu2016/nlp/tree/master/nlp_models/retrain_new_vocabulary)

### future:

1.中文词向量的拟合与评估

中文文本只取前面一部分字节，类似text8

文本先分词，然后只保留中文字符

用于编码文本的二进制文件转换为utf-8编码文本的二进制文件（同一编码的二进制文件就是文本文件）

gensim中几个sentence类的输出格式的总结

利用gensim把中文文本用于word2vec拟合

可用于句向量baseline的计算，可用于实体抽取模型的拟合

2.中文句向量的计算

3.中文句子或句群相似度的计算，用非参方法进行分类

4.中文句向量用于聚类和分析

## 5. Products

### 海外医疗、国内医疗：非结构化数据，可能有成交

[搜狗明医](http://mingyi.sogou.com/)

[疾病药品搜索](http://mingyi.sogou.com/mingyi?query=%E7%B3%96%E5%B0%BF%E7%97%85%E6%96%B0%E8%8D%AF&ie=utf8&fr=common_nav)

[医疗信息英文翻译](http://mingyi.sogou.com/fuwu/pc/vr/landresult/30010139?query=%E7%B3%96%E5%B0%BF%E7%97%85%E6%96%B0%E8%8D%AF)

[海外权威医疗网站](http://mingyi.sogou.com/fuwu/pc/authoritysite)


### 海外看球、国内看球：结构化数据

### Google: talk to books -- 非结构化数据

### 本轮联赛战报，下轮联赛前瞻 -- 非结构化数据

### 舆情跟踪与分析（词向量、句向量用于聚类，句向量构建舆情树），用于金融投资，用于选举和政治反馈，用于企业公关部门

### 用户画像的一部分：根据一个人的阅读的文字把一个人的思想观念向量化，人向量用于聚类，根据聚类结果推荐阅读：补短板以及加深度；一本书可以转换为一个向量树，多本书进行聚类后，也可以组成一个树；一篇文章可以转换为一个向量树（一段可以转换为一个句向量），一个人阅读过的多篇文章进行聚类后，也可以组成一个树

### 机屏分离，对于纽扣大小的主机，语音交互必不可少；屏幕生产厂商也很重要

### 自动驾驶，会飞的汽车，太空旅行，太空移民，肿瘤早筛，肿瘤治愈
