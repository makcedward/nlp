import datetime
import os
import tensorflow as tf
import tensorflow_hub as tf_hub

from .word_embeddings import WordEmbeddings


class ELMoEmbeddings(WordEmbeddings):
    ELMO_MODEL_V2_URL = "https://tfhub.dev/google/elmo/2"

    def __init__(self, layer, verbose=0):
        super().__init__(verbose=verbose)
        self.layer = layer
        
    def _set_tf_log_level(self, verbose):
        if verbose >= 30:
            tf.logging.set_verbosity(tf.logging.INFO)
        elif verbose >= 20:
            tf.logging.set_verbosity(tf.logging.WARN)
        elif verbose >= 10:
            tf.logging.set_verbosity(tf.logging.DEBUG)
        else:
            tf.logging.set_verbosity(tf.logging.ERROR)
        
    def load(self, src=None, dest_dir=None, trainable=True, verbose=0):
        self._log_time(status='LOADING', msg='file', verbose=verbose)
        self._set_tf_log_level(verbose)
        
        if src == None:
            src = self.ELMO_MODEL_V2_URL
        
        if dest_dir is not None:
            os.environ["TFHUB_CACHE_DIR"] = dest_dir
        
        self.model = tf_hub.Module(src, trainable=trainable)
        
        self._log_time(status='LOADED', msg='', verbose=verbose)
        
        return self.model    
    
    def to_keras_layer(self, x):
        # Source: https://github.com/strongio/keras-elmo/blob/master/Elmo%20Keras.ipynb
        '''
            For signature and layer parameters, you can visit https://alpha.tfhub.dev/google/elmo/2
        '''        
        return self.model(
            tf.squeeze(tf.cast(x, tf.string)), 
            signature="default", as_dict=True)[self.layer]
    
    
    # import operator
# import datetime
# import re

# from bilm.data import Vocabulary

# class ELMoEmbeddings:
#     def __init__(self, tokenizer=None, verbose=0):
#         self.verbose = verbose
        
#         self.tokenizer = self.get_tokenizer(tokenizer)

#     def _space_tokenizer(self, sentence):
#         # There is some unicode from source data
# #         return [t.encode('ascii', 'ignore').decode('ascii') for t in sentence.encode('ascii', 'ignore').decode('ascii').split(' ') if t != '']
# #         return [t.encode('ascii', 'ignore').decode('ascii') for t in sentence.split(' ') if t != '']
#         return [t for t in sentence.split(' ') if t != '']

#     def _spacy_tokenizer(self, sentence, model=None):
#         if model is None:
#             import spacy
#             model = spacy.load('en')

#         return [t.text.encode('ascii', 'ignore') for t in model(str(sentence)) if t.text != '']

#     def get_tokenizer(self, tokenizer):
#         if tokenizer is None or tokenizer == 'space':
#             tokenizer = self._space_tokenizer
#         elif tokenizer == 'spacy':
#             tokenizer = self._spacy_tokenizer

#         return tokenizer
    
#     def preprocess(self, sentence):
#         normalized_space = sentence.replace('\n', ' ').replace('\t', ' ').replace('\r', ' ')
#         normalized_unicode = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\xff]', ' ', normalized_space)

#         normalized_text = re.sub(' +',' ', normalized_unicode)
        
#         return normalized_text
    
#     def get_basic_elements(self, mode):
#         if mode == 'build':
#             return ['<S>', '</S>', '<UNK>']
#         elif mode == 'train':
#             return ['<S>', '</S>']
#         return []

#     def build_vocab(self, sentences, mode, vocab_file_path):
#         word_dict = {}
        
#         basic_elements = self.get_basic_elements(mode)

#         for sentence in sentences:
#             sentence = self.preprocess(sentence)
#             for w in self.tokenizer(sentence):
                
#                 if w not in word_dict:
#                     word_dict[w] = 0
#                 word_dict[w] += 1

#         word_dict = sorted(word_dict.items(), key=operator.itemgetter(1), reverse=True)
#         print('Total Word: %d' % (len(word_dict)))
        
#         with open(vocab_file_path, 'w') as f:
#             for item in basic_elements:
#                 f.write("%s\n" % item)
            
#             for word, count in word_dict:
#                 # Ximenez, characters <-- finding these word to check unicode issue
# #                 print([word])
#                 if word != '':
#                     f.write("%s\n" % word)
                
#     def build_data(self, sentences, data_file_path):
#         with open(data_file_path, 'w') as f:
#             for sentence in sentences:
#                 sentence = self.preprocess(sentence)
#                 tokens = self.tokenizer(sentence)
#                 if len(tokens) > 0:
#                     f.write("%s\n" % ' '.join(str(tokens)))