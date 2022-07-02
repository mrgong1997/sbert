import jieba
import re
import jieba.posseg as pseg

filename='白雪公主片段'
filepath1 = r'C:\Users\Mr.Gong\Desktop\项目\寒假\停用词表\stopwords-master\白雪公主片段.txt'
filepath2 = r'C:\Users\Mr.Gong\Desktop\项目\寒假\停用词表\stopwords-master\cn_stopwords.txt'


def stopwordslist(filepath2):  # 定义函数创建停用词列表
    stopword = [line.strip() for line in open(filepath2,'r', encoding='UTF-8').readlines()]  # 以行的形式读取停用词表，同时转换为列表
    print(stopword)
    return stopword


def pretext(filename, filepath1):  # 定义函数
    try:
        with open(filepath1, encoding='UTF-8') as file:
            contents = file.read()  # 读取文本文件
            print('【读取的文本为：】' + '\n' + contents)

            content1 = contents.replace(' ', '')  # 去掉文本中的空格
            print('\n【去除空格后的文本：】' + '\n' + content1)

            pattern = re.compile("[^\u4e00-\u9fa5^a-z^A-Z^0-9]")  # 只保留中英文、数字，去掉符号
            content2 = re.sub(pattern, '', content1)  # 把文本中匹配到的字符替换成空字符
            print('\n【去除符号后的文本：】' + '\n' + content2)

    except FileNotFoundError:
        message = "Sorry, the file " + filename + " does not exist."
        print(message)

    else:
        cutwords = jieba.lcut(content2, cut_all=False)  # 精确模式分词
        print('\n【精确模式分词后:】' + '\n' + "/".join(cutwords))

        stopwords = stopwordslist(filepath2)  # 这里加载停用词的路径
        print("stopwors:",stopwords)
        print("cutwords:",cutwords)
        words = ''
        for word in cutwords:  # for循环遍历分词后的每个词语
            if word not in stopwords:  # 判断分词后的词语是否在停用词表内
                if word != '\t':
                    words += word
                    words += "/"
        print('\n【去除停用词后的分词：】' + '\n' + words)
        content3 = words.replace('/', '')  # 去掉文本中的斜线
        print('\n【去除停用词后去除/的分词：】' + '\n' + content3)

        lastword = pseg.lcut(content3)  # 使用for循环逐一获取划分后的词语进行词性标注
        print('\n【对去除停用词后的分词进行词性标注：】' + '\n')
        print([(words.word, words.flag) for words in lastword])  # 转换为列表


stopwordslist(filepath2)  # 调用函数
pretext(filename, filepath1)  # 调用函数