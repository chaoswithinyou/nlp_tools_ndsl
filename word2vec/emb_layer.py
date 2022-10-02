import torch
import torch.nn as nn
from tqdm import tqdm
from nltk import tokenize


def word2ids(sen_list, word2idx, maxlen, vector_size):
    x = torch.zeros((len(sen_list),1,maxlen))
    for i in tqdm(range(len(sen_list))):
      doc = tokenize.word_tokenize(sen_list[i])
      fil_doc_w2index = []
      for word in doc:
        try:
          fil_doc_w2index.append(word2idx[word])
        except Exception:
          pass
      if len(fil_doc_w2index)<=maxlen:
        x[i] = torch.cat((torch.LongTensor(fil_doc_w2index),torch.zeros(maxlen-len(fil_doc_w2index)))).unsqueeze(0)
      else:
        x[i] = torch.LongTensor(fil_doc_w2index[-maxlen-1:-1]).unsqueeze(0)
    return x

def create_emb_layer_from_vector_mat(weights_matrix, non_trainable=False):
    weights_matrix = torch.Tensor(weights_matrix)
    num_embeddings, embedding_dim = weights_matrix.size()
    emb_layer = nn.Embedding(num_embeddings, embedding_dim)
    emb_layer.load_state_dict({'weight': weights_matrix})
    if non_trainable:
        emb_layer.weight.requires_grad = False

    return emb_layer, num_embeddings, embedding_dim