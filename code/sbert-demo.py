from sentence_transformers import SentenceTransformer, util
import jieba

model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')

# 句子
q1 = ["The new movie is awesome"]
q2 = ["The new movie is so great"]
q3 = ["The movie is terrible"]
q4 = ["慢性肝炎早期症状有哪些表现","病情诊断","病因分析","治疗方案","就医建议","指标解读","疾病表述","后果表述","注意事项","功效作用","医疗费用","其他"]
# cutwords = jieba.lcut(q4, cut_all=False)
# print(cutwords)

# 编码
embeddings1 = model.encode(q1)
embeddings2 = model.encode(q2)
embeddings4 = model.encode(q4)

# # 打印句子q1编码
# print("Sentence:", q1)
# print("Embedding:", embeddings1)
#
# 计算句子相似度并打印
cos_sim4 = util.pytorch_cos_sim(embeddings4, embeddings4)
print(q4,"&&",q4,"\n","Cosine-Similarity:\n", cos_sim4)
#
# cos_sim13 = util.pytorch_cos_sim(embeddings1, embeddings3)
# print(q1,"&&",q3,"\n","Cosine-Similarity:\n", cos_sim13)

