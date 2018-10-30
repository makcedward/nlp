import datetime, os, zipfile
import numpy as np
import torch
import subprocess

from .glove import GloVeEmbeddings
from .sentence_embeddings import SentenceEmbeddings

# InferSent (as of Sep 2018) is not a a library (https://github.com/facebookresearch/InferSent/issues/76), Cloned from https://github.com/facebookresearch/InferSent
from .infersent_lib.models import InferSent


class InferSentEmbeddings(SentenceEmbeddings):
    INFERSENT_GLOVE_MODEL_URL = 'https://s3.amazonaws.com/senteval/infersent/infersent1.pkl'
    INFERSENT_FASTTEXT_MODEL_URL = 'https://s3.amazonaws.com/senteval/infersent/infersent2.pkl'
    
    def __init__(self, 
                 word_embeddings_dir,
                 batch_size=64, word_dimension=300, encoder_lstm_dimension=2048, 
                 pooling_type='max', model_version=1, dropout=0.0, 
                 verbose=0):
        super().__init__(verbose=verbose)
        
        self.word_embeddings_dir = word_embeddings_dir
        self.batch_size = batch_size
        self.word_dimension = word_dimension
        self.encoder_lstm_dimension = encoder_lstm_dimension
        self.pooling_type = pooling_type
        self.dropout = dropout
        self.model_version = model_version
        
    def get_params(self):
        return {
            'bsize': self.batch_size, 
            'word_emb_dim': self.word_dimension, 
            'enc_lstm_dim': self.encoder_lstm_dimension,
            'pool_type': self.pooling_type, 
            'dpout_model': self.dropout, 
            'version': self.model_version
        }
        
    def load_model(self, dest_dir, src=None, trainable=True, verbose=0):
        # TODO: Support V2 model
        if src is None:
            src = InferSentEmbeddings.INFERSENT_GLOVE_MODEL_URL
            
        dest_file = os.path.basename(src)
        file_path = self.download(
            src=src, dest_dir=dest_dir, dest_file=dest_file, 
            uncompress=False, housekeep=False, verbose=verbose)
            
        self.model = InferSent(self.get_params())
        self.model.load_state_dict(torch.load(dest_dir + dest_file))
        
        # TODO: support different glove model and fasttext model
        word_embs = GloVeEmbeddings()
        word_embs.load_model(dest_dir=self.word_embeddings_dir, process=False, verbose=verbose)
        
        self.model.set_w2v_path(word_embs.model_path)
        
    def build_vocab(self, sentences, tokenize=True):
        return self.model.build_vocab(sentences, tokenize=tokenize)
    
    def encode(self, sentences, tokenize=True):
        return self.model.encode(sentences, tokenize=tokenize)
    
    def visualize(self, sentence, tokenize=True):
        self.model.visualize(sentence, tokenize=tokenize)    