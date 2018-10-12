import datetime
import numpy as np

from .embeddings import Embeddings


class WordEmbeddings(Embeddings):
    
    def __init__(self, 
                 handle_oov=True, oov_vector=None, oov_vector_type='zero',
                 padding=True, pad_vector=None, pad_vector_type='zero',
                 max_sequence_length=10, dimension=300,
                 verbose=0):
        super().__init__(verbose=verbose)
        self.handle_oov = handle_oov
        self.oov_vector_type = oov_vector_type
        if handle_oov and oov_vector is None:
            if oov_vector_type == 'zero':
                self.oov_vector = np.zeros(dimension)
            elif oov_vector_type == 'random':
                self.oov_vector = np.random.rand(dimension)
        else:
            self.oov_vector = oov_vector
            
        self.padding = padding
        self.pad_vector_type = pad_vector_type
        if padding and pad_vector is None:
            if pad_vector_type == 'zero':
                self.pad_vector = np.zeros(dimension)
            elif pad_vector_type == 'random':
                self.pad_vector = np.random.rand(dimension)
        else:
            self.pad_vector = pad_vector
        
        self.max_sequence_length = max_sequence_length
        self.dimension = dimension
        
    def get_oov_vector(self):
        return self.oov_vector
        
    def set_oov_vector(self, oov_vector):
        self.oov_vector = oov_vector
        
    def get_pad_vector(self):
        return self.pad_vector
        
    def set_pad_vector(self, pad_vector):
        self.pad_vector = pad_vector
        
    def is_vector_exist(self, word):
        return word in self.model