import datetime, os, zipfile
import numpy as np

from .word_embeddings import WordEmbeddings


class GloVeEmbeddings(WordEmbeddings):
    GLOVE_COMMON_CRAWL_MODEL_URL = 'http://nlp.stanford.edu/data/glove.42B.300d.zip'
    
    def __init__(self, 
                 handle_oov=True, oov_vector=None, oov_vector_type='zero',
                 padding=True, pad_vector=None, pad_vector_type='zero',
                 max_sequence_length=10, dimension=300,
                 verbose=0):
        super().__init__(
            handle_oov=handle_oov, oov_vector=oov_vector, oov_vector_type=oov_vector_type,
            padding=padding, pad_vector=pad_vector, pad_vector_type=pad_vector_type,
            max_sequence_length=max_sequence_length, dimension=dimension,
            verbose=verbose)
        
    def load_model(self, dest_dir, src=None, trainable=True, process=True, verbose=0):
        if src is None:
            src = self.GLOVE_COMMON_CRAWL_MODEL_URL
            
        dest_file = os.path.basename(src)
            
        file_path = self.download(
            src=src, dest_dir=dest_dir, dest_file=None, 
            uncompress=True, housekeep=False, verbose=verbose)
            
        self.model_path = dest_dir + dest_file
        
        dest_file = dest_file.replace('.zip', '.txt')
            
        if process and not self.is_file_exist(dest_dir + dest_file):
            with open(dest_dir + dest_file, encoding="utf8" ) as f:
                lines = f.readlines()

            for line in lines:
                line_contents = line.split()
                word = line_contents[0]
                self.model[word] = np.array([float(val) for val in line_contents[1:]])
            
        return self.model
        
    def uncompress(self, file_path):
        self.unzip(file_path)
        
    def encode(self, sentences):
        preds = np.empty([len(sentences), self.max_sequence_length, self.dimension])
        
        for i, words in enumerate(sentences):
            pred = np.empty([self.max_sequence_length, self.dimension])
            cnt = 0
            
            for word in words:
                if self.is_vector_exist(word):
                    pred[cnt] = self.model[word]
                    cnt += 1
                elif self.handle_oov:
                    pred[cnt] = self.oov_vector
                    cnt += 1
                    
                if cnt + 1 >= self.max_sequence_length:
                    break
                    
            if self.padding and (cnt + 1 < self.max_sequence_length):
                for i in range(0, self.max_sequence_length - cnt):
                    pred[cnt] = self.pad_vector
                    cnt += 1
                    
            preds[i] = pred
            
        
        return preds