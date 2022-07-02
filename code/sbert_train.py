from sentence_transformers import SentenceTransformer, models, InputExample,SentencesDataset, losses, evaluation, LoggingHandler, util
from torch.utils.data import DataLoader
from torch import nn
import os
import logging
import math
from datetime import datetime
import torch
import json
import csv

class Medical_TrainClass():
    """
    中文医疗意图识别任务模型训练类
    """
    def __init__(self,
                 bert_name_path = r'F:\PycharmProject\kbqa\CBLUE-main\Huggingface\mengzi-bert-base',
                 data_dir = r'F:\PycharmProject\sbert\Data_cblue',
                 output_path = r'F:\PycharmProject\sbert\model_CBLUE_legion',
                 max_seq_length =20,
                 is_train = True,
                 is_dev = True,
                 is_test = False,
                 # train_batch_size=16,
                 train_batch_size =16,
                 # evaluation_step=2000,
                 evaluation_step = 2000,
                 # shuffle=True,
                 shuffle=True,
                 epochs =1):

        logging.basicConfig(format='%(asctime)s - %(message)s',datefmt='%Y-%m-%d %H:%M:%S',level=logging.INFO,handlers=[LoggingHandler()])
        self.bert_name = bert_name_path
        self.max_seq_length = max_seq_length
        self.data_dir = data_dir
        self.is_trains = is_train
        self.is_dev = is_dev
        self.is_test = is_test
        self.output_path = output_path
        self.train_batch_size = train_batch_size
        self.shuffle = shuffle
        self.epochs = epochs
        self.evaluation_step = evaluation_step

        self.__init_graph__()

    def __init_graph__(self):
        # Transformer
        word_embedding_model = models.Transformer(self.bert_name, self.max_seq_length)
        # pooling
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                                       pooling_mode_mean_tokens=True,
                                       pooling_mode_cls_token=False,
                                       pooling_mode_max_tokens=False)
        # DNN
        dense_model = models.Dense(in_features=pooling_model.get_sentence_embedding_dimension(),
                                   out_features=256,
                                   activation_function=nn.Tanh())

        self.model = SentenceTransformer(modules=[word_embedding_model, pooling_model,dense_model])
        self.train_loss  = losses.ContrastiveLoss(self.model)
        # self.train_loss = losses.TripletLoss(self.model)
        # self.train_loss = losses.SoftmaxLoss(self.model, sentence_embedding_dimension=self.model.get_sentence_embedding_dimension(), num_labels=11)


    def train(self):
        logging.info("Read train train dataset")
        # get train examples
        self.train_examples = self.get_train_examples()
        # get train datas
        train_dataset = SentencesDataset(self.train_examples, self.model)
        # load datas
        train_dataloader = DataLoader(train_dataset, shuffle = self.shuffle, batch_size = self.train_batch_size)

        # self.warmup_steps = warmup_steps
        self.warmup_steps = math.ceil(
            len(train_dataset) * self.epochs / self.train_batch_size * 0.1)  # 10% of train data for warm-up
        logging.info("Warmup-steps: {}".format(self.warmup_steps))

        if self.is_dev:
            # get dev examples
            logging.info("Read train train dataset")
        self.dev_examples = self.get_dev_exapmles()
        evaluator = evaluation.EmbeddingSimilarityEvaluator([sentence_1.texts[0] for sentence_1 in self.dev_examples],
                                                                 [sentence_1.texts[1] for sentence_1 in self.dev_examples],
                                                                 [sentence_1.label for sentence_1 in self.dev_examples])
        # evaluator = evaluation.TripletEvaluator([sentence_1.texts[0] for sentence_1 in self.dev_examples],
        #                                         [sentence_1.texts[1] for sentence_1 in self.dev_examples],
        #                                         [sentence_1.texts[2] for sentence_1 in self.dev_examples])

        self.evaluator = evaluation.EmbeddingSimilarityEvaluator.from_input_examples(self.dev_examples, name='CBLUE')
        # self.evaluator = evaluation.TripletEvaluator.from_input_examples(self.dev_examples, name='CBLUE')

        self.model.fit(train_objectives=[(train_dataloader, self.train_loss)],
                       warmup_steps=self.warmup_steps,
                       evaluator=self.evaluator,
                       evaluation_steps=self.evaluation_step,
                       epochs=self.epochs,
                       output_path=self.output_path)


    def added_train(self, output_dir = r'F:\PycharmProject\sbert\modeladd', epochs = 10, evaluation_steps=1000):
        """
        增量训练
        """

        added_examples = self.get_added_train("added.txt")
        train_data_sets = SentencesDataset(added_examples, self.model)
        train_dataloader = DataLoader(train_data_sets, shuffle=True, batch_size=self.train_batch_size)
        train_loss = losses.CosineSimilarityLoss(model=self.model)
        evaluator = evaluation.EmbeddingSimilarityEvaluator.from_input_examples(added_examples)

        # update warmup steps
        self.warmup_steps = math.ceil(
            len(train_data_sets) * self.epochs / self.train_batch_size * 0.1)  # 10% of train data for warm-up

        self.model.fit(train_objectives=[(train_dataloader, train_loss)],
                  evaluator=evaluator,
                  epochs=epochs,
                  evaluation_steps=evaluation_steps,
                  warmup_steps=self.warmup_steps,
                  output_path=output_dir)


    def test(self, model_path = r'F:\PycharmProject\sbert\model_CBLUE_legion',
             output_path= r'F:\PycharmProject\sbert\model_CBLUE_legion',
             tst_name='govern_test'):

        model = SentenceTransformer(model_path)
        test_samples = self.get_test_examples()
        test_evaluator = evaluation.EmbeddingSimilarityEvaluator.from_input_examples(test_samples, name=tst_name)
        test_evaluator(model, output_path)

    def __read__data(self, file):
        examples = []
        if not os.path.exists(os.path.join(self.data_dir, file)):
            raise Exception("Not Found file:{} Error".format(os.path.join(self.data_dir, file)))

        with open(os.path.join(self.data_dir, file), mode='r', encoding='utf-8') as in_file:
            while True:
                _tmp_str = in_file.readline()
                if _tmp_str == '':
                    break
                try:
                    _sentence_pairs = _tmp_str.strip().split(',')
                    _sentence_one = _sentence_pairs[0].strip()
                    _sentence_two = _sentence_pairs[1].strip()
                    _label = float(_sentence_pairs[2].strip())
                    # _sentence_three = _sentence_pairs[2].strip()
                    _example = InputExample(texts=[_sentence_one, _sentence_two], label=_label)
                    # _example = InputExample(texts=[_sentence_one, _sentence_two,_sentence_three])
                    examples.append(_example)
                except ValueError:
                    print(_tmp_str)
                    return
                except IndexError:
                    print(_tmp_str)
                    return

        return examples


    def get_train_examples(self, file="cblue_train.csv"):
    # def get_train_examples(self, file="cblue4triplet_train.csv"):
        return self.__read__data(file)

    def get_dev_exapmles(self, file="cblue_dev.csv"):
    # def get_dev_exapmles(self, file="cblue4triplet_dev.csv"):
        return self.__read__data(file)

    def get_test_examples(self, file="cblue_test.csv"):
        return self.__read__data(file)

    def get_added_train(self, file="added.txt"):
        return self.__read__data(file)


def calc_similarity(sentences,label):

    model = SentenceTransformer( r'F:\PycharmProject\sbert\model_CBLUE_legion')
    embeddings1 = model.encode(sentences)
    embeddings2 = model.encode(label)
    simlarities = util.pytorch_cos_sim(embeddings1, embeddings2)
    return simlarities


read_data_dir= r'F:\PycharmProject\kbqa\CBLUE-main\MagicChangeData\genData\KUAKE-QIC'
output_data_dir= r'F:\PycharmProject\sbert\Data_cblue'

label_list = [line.strip() for line in open('cblue_label','r',encoding='utf8')]
# print(label_list)
label2id = {label:idx for idx,label in enumerate(label_list)}
def dataTest(readfile,writefile):
    if not os.path.exists(os.path.join(read_data_dir, readfile)):
        raise Exception("Not Found file:{} Error".format(os.path.join(read_data_dir, readfile)))
    labellabel=[]
    scorescore=[]
    with open(os.path.join(read_data_dir, readfile), 'rt', encoding="utf-8") as readfile:
        with open(os.path.join(output_data_dir, writefile), mode='w', encoding='utf-8',newline='') as out_file:
            #读取所有数据到data
            data = json.load(readfile)
            for item in data:
                #把json数据中list列表转换成想要的格式
                sentence = item['query']
                label = ['病情诊断', '病因分析', '治疗方案', '就医建议', '指标解读', '疾病表述', '后果表述', '注意事项', '功效作用', '医疗费用', '其他']
                cos_scores = calc_similarity(sentence, label)[0].cpu()
                top = torch.topk(cos_scores, k=1)
                score = "(Score: %.4f)" % (top[0])
                scorescore.append(score)
                labellabel.append(label[top[1]])
                # print(label[top[1]], score)
            n=0
            for item in data:
                item['label']=labellabel[n]
                # item['score']=scorescore[n]
                n = n + 1
            json.dump(data, out_file, ensure_ascii=False, indent=2)
        out_file.close()
    readfile.close()
    return 0
# def genTestData(readfile,writefile):
#     if not os.path.exists(os.path.join(read_data_dir, readfile)):
#         raise Exception("Not Found file:{} Error".format(os.path.join(read_data_dir, readfile)))
#     labellabel=[]
#     scorescore=[]
#     with open(os.path.join(read_data_dir, readfile), 'rt', encoding="utf-8") as readfile:
#         with open(os.path.join(output_data_dir, writefile), mode='w', encoding='utf-8',newline='') as out_file:
#             #读取所有数据到data
#             data = json.load(readfile)
#             for item in data:
#                 del item['score']
#             json.dump(data, out_file, ensure_ascii=False, indent=2)
#         out_file.close()
#     readfile.close()
#     return 0

if __name__ == '__main__':
    medical = Medical_TrainClass()
    #
    # # batch train and test
    medical.train()
    # gagt.test()
    #
    # # added train
    # # gagt.added_train()

    # sentence = "啊哈哈哈哈哈哈"
    # label= ['病情诊断','病因分析',"治疗方案",'就医建议','指标解读','疾病表述','后果表述','注意事项','功效作用','医疗费用','其他']
    # cos_scores = calc_similarity(sentence,label)[0].cpu()
    # top = torch.topk(cos_scores, k=1)
    # print(label[top[1]])


    dataTest('KUAKE-QIC_dev2test.json','KUAKE-QIC_test_sbert.json')
    # genTestData('KUAKE-QIC_test1.json','KUAKE-QIC_test.json')





