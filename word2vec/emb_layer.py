import torch
import torch.nn as nn

def create_emb_layer_from_vector_mat(weights_matrix, non_trainable=False):
    weights_matrix = torch.Tensor(weights_matrix)
    num_embeddings, embedding_dim = weights_matrix.size()
    emb_layer = nn.Embedding(num_embeddings, embedding_dim)
    emb_layer.load_state_dict({'weight': weights_matrix})
    if non_trainable:
        emb_layer.weight.requires_grad = False

    return emb_layer, num_embeddings, embedding_dim