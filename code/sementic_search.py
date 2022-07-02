import os
import jieba
import re
from sentence_transformers import SentenceTransformer, util
import torch
import time
import pickle

#设置程序运行时间
start=time.clock()

#使用模型：
model='distilbert-base-nli-stsb-mean-tokens'
embedder = SentenceTransformer(model)
#定义语料库列表和去停用词语料库列表
corpus = []
corpus_stopwords=[]

#把1000条数据{现在有什么动画片好看呢？	现在有什么好看的动画片吗？	1}的第二句存入corpus[]中
data_dir = r'F:\PycharmProject\sbert\Data_sematicSearch_fromLCQMC'
file="corpus1000.txt"
with open(os.path.join(data_dir, file), mode='r', encoding='utf-8') as file_corpus:
    while True:
        currentline = file_corpus.readline()
        if currentline == '':
            break
        try:
            _sentence_pairs = currentline.strip().split('\t')
            _sentence_one = _sentence_pairs[0].strip()
            _sentence_two = _sentence_pairs[1].strip()
            _label = float(_sentence_pairs[2].strip())
            #把每一列的第二句加入corpus[]集合里
            corpus.append(_sentence_two)
        except ValueError:
            print(currentline)
        except IndexError:
            print(currentline)

file_stopwords = r'F:\PycharmProject\sbert\Data_sematicSearch_fromLCQMC\cn_stopwords.txt'
# 定义函数创建停用词列表
def stopwordslist(file_stopwords):
    stopword = [line.strip() for line in open(file_stopwords,'r', encoding='UTF-8').readlines()]  # 以行的形式读取停用词表，同时转换为列表
    return stopword
stopwords= stopwordslist(file_stopwords)
corpus_stopwords=[]

#=========================  此时  corpus[]  和 stopwords[]   以及 corpus_stopwords[]都准备好了=======================
#下面开始定义去停用词的函数，输入为以上两集合，输出为去停用词之后的  corpus_stopwords[]  集合
def remove_stopwords(a,b,c):   # a:原句子集合 b:停用词集合 c:去停用词集合，初始为空
    for sentence in a:
        sentence1 = sentence.replace(' ', '')
        pattern = re.compile("[^\u4e00-\u9fa5^a-z^A-Z^0-9]")  # 只保留中英文、数字，去掉符号
        sentence2 = re.sub(pattern, '', sentence1)  # 把文本中匹配到的字符替换成空字符
        cutwords = jieba.lcut(sentence2, cut_all=False)  # 精确模式分词

        words = ''
        for word in cutwords:  # for循环遍历分词后的每个词语
            if word not in b:  # 判断分词后的词语是否在停用词表内
                if word != '\t':
                    words += word
        c.append(words)

#调用函数，将corpus去停用词
remove_stopwords(corpus,stopwords,corpus_stopwords)

#Query sentences:
queries= ['现在有什么动画片好看呢？',
           '送自己做的闺蜜什么生日礼物好',
           '近期上映的电影',
           '求英雄联盟大神带？',
           '杭州哪里好玩',
           '最好玩的手机网游',
           '什么东西越热爬得越高',
           '如何入侵他人手机',
           '微信号怎么二次修改',
           '最近有没有什么好看的韩剧'
           ]
# queries= ['喜欢打篮球的男生喜欢什么样的女生',
#            '求秋色之空漫画全集',
#            '学日语软件手机上的',
#            '什么花一年四季都开',
#            '看图猜一电影名',
#            '如何快速忘记一个人',
#            '周杰伦女友是谁',
#            '支付宝手机支付密码是什么',
#            '中国四大古城是哪些？',
#            '石家庄天气如何？'
#            ]
queries_stopwords=[]
#再次调用函数，将queries去停用词
remove_stopwords(queries,stopwords,queries_stopwords)

# #先计算出corpu集合的embedding
# corpus_embeddings = embedder.encode(corpus_stopwords, convert_to_tensor=True)
# #将计算好的embedding存放到corpus_embeddings.pkl文件中
# with open('corpus_embeddings1000.pkl', "wb") as fOut:
#      pickle.dump({'corpus_stopwords': corpus_stopwords, 'corpus_embeddings': corpus_embeddings}, fOut, protocol=pickle.HIGHEST_PROTOCOL)

#从对应文件中加载embedding
with open('corpus_embeddings1000.pkl', "rb") as fIn:
    stored_data = pickle.load(fIn)
    stored_sentences = stored_data['corpus_stopwords']
    stored_embeddings = stored_data['corpus_embeddings']

#找出corpus集合中每个句子最相似的10个句子作为结果输出
top_k = 10
i=1
for query in queries_stopwords:
    query_embedding = embedder.encode(query, convert_to_tensor=True)
    cos_scores = util.pytorch_cos_sim(query_embedding, stored_embeddings)[0]
    cos_scores = cos_scores.cpu()

    #We use torch.topk to find the highest 10 scores
    top_results = torch.topk(cos_scores, k=top_k)

    print("\n\n======================\n\n")
    print("Query{}:{}".format(i,query))
    i+=1
    print("\nTop 10 most similar sentences in corpus:")
    for score, idx in zip(top_results[0], top_results[1]):
        print(corpus_stopwords[idx], "(Score: %.4f)" % (score))

end=time.clock()
print("\n\n======================\n\n")
print("search costs %.4f s"%(end-start))