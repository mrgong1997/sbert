import jieba
import re
import os
import json
read_data_dir= r'F:\PycharmProject\kbqa\CBLUE-main\MagicChangeData\stopwords\KUAKE-QIC'
output_data_dir= r'F:\PycharmProject\kbqa\CBLUE-main\MagicChangeData\stopwords\KUAKE-QIC'

corpus=[]
corpus_stopwords=[]
def gen_corpus(readfile):
    if not os.path.exists(os.path.join(read_data_dir, readfile)):
        raise Exception("Not Found file:{} Error".format(os.path.join(read_data_dir, readfile)))
    with open(os.path.join(read_data_dir, readfile), 'rt', encoding="utf-8") as readfile:
        #读取所有数据到data
        data = json.load(readfile)
        for item in data:
            corpus.append(item['query'])
    readfile.close()
    return data
gen_corpus('KUAKE-QIC_train.json')
file_stopwords = r'F:\PycharmProject\sbert\Data_sematicSearch_fromLCQMC\cn_stopwords.txt'
# 定义函数创建停用词列表
def stopwordslist(file_stopwords):
    stopword = [line.strip() for line in open(file_stopwords,'r', encoding='UTF-8').readlines()]  # 以行的形式读取停用词表，同时转换为列表
    return stopword
stopwords= stopwordslist(file_stopwords)
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
def gen_cbluedata(readfile,writefile):
    if not os.path.exists(os.path.join(read_data_dir, readfile)):
        raise Exception("Not Found file:{} Error".format(os.path.join(read_data_dir, readfile)))

    with open(os.path.join(read_data_dir, readfile), 'rt', encoding="utf-8") as readfile:
        with open(os.path.join(output_data_dir, writefile), mode='w', encoding='utf-8', newline='') as out_file:
            #读取所有数据到data
            n = 0
            data = json.load(readfile)
            for item in data:
                item['query'] = corpus_stopwords[n]
                n = n + 1
            json.dump(data, out_file, ensure_ascii=False, indent=2)
        out_file.close()
    readfile.close()
    return data

gen_cbluedata('KUAKE-QIC_train.json','KUAKE-QIC_train_stop.json')
