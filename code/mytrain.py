from sentence_transformers import SentenceTransformer, models,InputExample, losses,SentencesDataset,evaluation
from torch import nn
from torch.utils.data import DataLoader

word_embedding_model = models.Transformer('bert-base-cased', max_seq_length=256)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
dense_model = models.Dense(in_features=pooling_model.get_sentence_embedding_dimension(), out_features=256, activation_function=nn.Tanh())

model = SentenceTransformer(modules=[word_embedding_model, pooling_model, dense_model])
train_examples = [InputExample(texts=['cat is beautiful', 'cat looks beautiful'], label=0.8),
    InputExample(texts=['I like eating', 'cat is white'], label=0.3)]

#Define your train dataset, the dataloader and the train loss
train_dataset = SentencesDataset(train_examples, model)
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=16)
train_loss = losses.CosineSimilarityLoss(model)


sentences1 = ['cat is white', 'cat is beautiful', 'I like running']
sentences2 = ['I like dog', 'cat looks cute', 'cat is eating']
scores = [0.3, 0.6, 0.2]

evaluator = evaluation.EmbeddingSimilarityEvaluator(sentences1, sentences2, scores)
# ... Your other code to load training data

model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=1, warmup_steps=100, evaluator=evaluator, evaluation_steps=500)