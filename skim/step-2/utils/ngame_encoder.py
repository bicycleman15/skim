"""
NGAME encoder according to the arxiv paper: https://arxiv.org/abs/2207.04452
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import sentence_transformers


class STransformerInputLayer(nn.Module):
    """
    Sentence transformer
    """
    def __init__(self, transformer='roberta-base'):
        super(STransformerInputLayer, self).__init__()
        if isinstance(transformer, str):
            self.transformer = sentence_transformers.SentenceTransformer(transformer)
        else:
            self.transformer = transformer

    def forward(self, data):
        sentence_embedding = self.transformer(data)['sentence_embedding']
        return sentence_embedding


class CustomEncoder(torch.nn.Module):
    """
    Encoder layer with Sentence transformer and an optional projection layer

    * projection layer is applied after reduction and normalization
    """
    def __init__(self, encoder_name, transform_dim):
        super(CustomEncoder, self).__init__()
        self.encoder = STransformerInputLayer(
            sentence_transformers.SentenceTransformer(encoder_name))
        self.transform_dim = transform_dim
        if(self.transform_dim != -1):
            self.transform = nn.Linear(
                self.encoder.transformer[1].word_embedding_dimension, self.transform_dim)
    
    def forward(self, input_ids, attention_mask):
        if(self.transform_dim != -1):
            return self.transform(self.encoder({'input_ids': input_ids, 'attention_mask': attention_mask}))
        else:
            return self.encoder({'input_ids': input_ids, 'attention_mask': attention_mask})

    @property
    def repr_dims(self):
        return self.encoder.transformer[1].word_embedding_dimension \
        if self.transform_dim == -1 else self.transform_dim


class SiameseNetwork(torch.nn.Module):
    """
    A network class to support Siamese style training
    * specialized for sentence-bert or hugging face
    * hard-coded to use a joint encoder

    """
    def __init__(self, encoder_name, transform_dim, device):
        super(SiameseNetwork, self).__init__()
        self.padding_idx = 0
        self.encoder = CustomEncoder(encoder_name, transform_dim)
        self.device = device

    def encode(self, doc_input_ids, doc_attention_mask):
        return F.normalize(
            self.encoder(doc_input_ids.to(self.device), doc_attention_mask.to(self.device)))

    def encode_document(self, doc_input_ids, doc_attention_mask, *args):
        return F.normalize(
            self.encoder(doc_input_ids.to(self.device), doc_attention_mask.to(self.device)))

    def encode_label(self, lbl_input_ids, lbl_attention_mask):        
        return F.normalize(
            self.encoder(lbl_input_ids.to(self.device), lbl_attention_mask.to(self.device)))
    
    def forward(self, doc_input_ids, doc_attention_mask, lbl_input_ids, lbl_attention_mask):
        if(doc_input_ids is None):
            return self.encode_label(lbl_input_ids, lbl_attention_mask)
        elif(lbl_input_ids is None):
            return self.encode_document(doc_input_ids, doc_attention_mask)
        doc_embeddings = self.encode_document(doc_input_ids, doc_attention_mask)
        label_embeddings = self.encode_label(lbl_input_ids, lbl_attention_mask)
        return doc_embeddings, label_embeddings

    @property
    def repr_dims(self):
        return self.encoder.repr_dims