import keras

from .word_embeddings import WordEmbeddings
from .glove import GloVeEmbeddings

'''
    Source: https://github.com/rgsachin/CoVe
'''


class CoVeEmbeddings(WordEmbeddings):
    COVE_MODEL_KERAS_URL = 'https://github.com/rgsachin/CoVe/raw/master/Keras_CoVe.h5'
    
    def __init__(self, 
                 word_embeddings_dir, 
                 handle_oov=True, oov_vector_type='random', 
                 padding=True, pad_vector_type='random', 
                 max_sequence_length=50, tokenizer=None,
                 verbose=0):
        super().__init__(verbose=verbose)
        
        if tokenizer is None:
            self.tokenizer = self._tokenizer_space
        
        self.word_embeddings_dir = word_embeddings_dir
        self.handle_oov = handle_oov
        self.oov_vector_type = oov_vector_type
        self.padding = padding
        self.pad_vector_type = pad_vector_type
        self.max_sequence_length = max_sequence_length
        
    def load_model(self, dest_dir, src=None, trainable=True, verbose=0):
        if src is None:
            src = self.COVE_MODEL_KERAS_URL
        
        file_path = self.download(
            src=src, dest_dir=dest_dir, dest_file=None, uncompress=False)
    
        self.model = keras.models.load_model(file_path)
        
        self.word_embs_model = GloVeEmbeddings(
            handle_oov=self.handle_oov, oov_vector_type=self.oov_vector_type,
            padding=self.padding, pad_vector_type=self.pad_vector_type, 
            max_sequence_length=self.max_sequence_length)
        self.word_embs_model.load_model(dest_dir=self.word_embeddings_dir, process=False, verbose=verbose)
        
    def encode(self, x, tokenize=True):
        if tokenize:
            tokens = [self.tokenizer(sentence) for sentence in x]
        else:
            tokens = x
        
        x_embs = self.word_embs_model.encode(tokens)
        
        return self.model.predict(x_embs)
    
        