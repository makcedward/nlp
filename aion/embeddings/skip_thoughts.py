# Copyright 2018 Edward Ma. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import os, datetime

class SkipThoughtsEmbeddingsTorch:
    DICTIONARY_URL = "http://www.cs.toronto.edu/~rkiros/models/dictionary.txt"
    UNISKIP_URL = "http://www.cs.toronto.edu/~rkiros/models/utable.npy"
    BISKIP_URL = "http://www.cs.toronto.edu/~rkiros/models/btable.npy"
    UNISKIPS_URL = "http://www.cs.toronto.edu/~rkiros/models/uni_skip.npz"
    BISKIPS_URL = "http://www.cs.toronto.edu/~rkiros/models/bi_skip.npz"
    UNISKIPS_PKL_URL = "http://www.cs.toronto.edu/~rkiros/models/uni_skip.npz.pkl"
    BISKIPS_PKL_URL = "http://www.cs.toronto.edu/~rkiros/models/bi_skip.npz.pkl"
    
    def __init__(self, model_dir, algorithm='uniskip', tokenizer=None, verbose=0):
        super().__init__(verbose=verbose)
        
        from torch import LongTensor
        from torch.autograd import Variable
        from skipthoughts import UniSkip, BiSkip
        
        self.model_dir = model_dir
        self.algorithm = algorithm
        self.vocab = {}
        self.vocabs = []
        if tokenizer is None:
            self.tokenizer = self._tokenizer_space
        else:
            self.tokenizer = tokenizer
        self.max_sentence_len = -1
        
    def downloads(self, dest_dir, sources=None):
        if sources is None:
            sources = [self.DICTIONARY_URL, self.UNISKIP_URL, self.BISKIP_URL, 
                       self.UNISKIPS_URL, self.BISKIPS_URL, self.UNISKIPS_PKL_URL, 
                       self.BISKIPS_PKL_URL]
        
        for src in sources:
            self.download(src=src, dest_dir=dest_dir, dest_file=None, unzip=False)
        
    def build_vocab(self, sentences, clear_vocab=True, max_sentence_len=-1):
        if clear_vocab:
            self.vocab = {}
            
        self.max_sentence_len = max_sentence_len
        
        for sentence in sentences:
            words = self.tokenizer(sentence)
            if max_sentence_len == -1:
                self.max_sentence_len = max(self.max_sentence_len, len(words))

            for word in words:
                if word not in self.vocab:
                    self.vocabs.append(word)
                    # Reserve the first one for padding
                    self.vocab[word] = len(self.vocab) + 1

    def process(self, sentences):
        word_id_sentences = []
        for sentence in sentences:
            word_ids = [self.vocab[w] for w in self.tokenizer(sentence) if w in self.vocab]
            
            if self.max_sentence_len > len(word_ids):
                for i in range(0, self.max_sentence_len-len(word_ids)):
                    word_ids.append(0)
            elif self.max_sentence_len < len(word_ids):
                word_ids = word_ids[:self.max_sentence_len]
                    
            word_id_sentences.append(word_ids)
            
        return word_id_sentences
    
    def get_algorithm(self, words, model_dir=None):
        if model_dir is None:
            model_dir = self.model_dir
            
        if self.algorithm == 'uniskip':
            return UniSkip(model_dir, words)
        else:
            return BiSkip(model_dir, words)
        
    def to_numpy_layer(self, layer):
        return layer.detach().numpy()        

    def encode(self, sentences, output_format='torch'):
        transformed_sentences = self.process(sentences)
        
        algo = self.get_algorithm(self.vocabs)
        inputs = Variable(LongTensor(transformed_sentences))
        outpus = algo(inputs, lengths=[len(words) for words in transformed_sentences])
        
        if output_format == 'np':
            return self.to_numpy_layer(outpus)
        elif output_format == 'torch':
            return outpus
        
    def predict_batch(self, sentences, output_format='torch', batch_size=1000):
        batches = [sentences[i * batch_size:(i + 1) * batch_size] for i in range((len(sentences) + batch_size-1) // batch_size)]

        results = []
        for batch in batches:
            results.append(skip_thoughts_emb.predict(sentences=batch, output_format=output_format))

        if output_format == 'np':
            return np.concatenate(results, axis=0)
        elif output_format == 'torch':
            return torch.cat(results, 0)