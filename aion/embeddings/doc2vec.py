from gensim.models.doc2vec import Doc2Vec, TaggedDocument

from .document_embeddings import DocumentEmbeddings


class Doc2VecEmbeddings(DocumentEmbeddings):
    def __init__(self, 
                 merge_mode="concat", algorithms="dm", 
                 word_dimension=300, min_word_count=1, 
                 word_window=10, n_job=4, 
                 train_epoch=10, infer_epoch=5,
                 infer_aplha=0.1, infer_min_alpha=0.0001,
                 verbose=0):
        super().__init__(verbose=verbose)
        
        self.merge_mode = merge_mode
        if merge_mode == 'concat':
            self.dm_concat = 1
            self.dm_mean = None
        elif merge_mode == 'mean':
            self.dm_concat = None
            self.dm_mean = 1
        else:
            raise Exception('merge_mode only allows either concat or mean')
        
        self.algorithms = algorithms
        if algorithms == 'dm':
            self.dm = 1
        elif algorithms == 'dbow':
            self.dm = 0
            
        self.word_dimension = word_dimension
        self.min_word_count = min_word_count
        self.word_window = word_window
        self.n_job = n_job
        self.train_epoch = train_epoch
        self.infer_epoch = infer_epoch
        self.infer_alpha = infer_aplha
        self.infer_min_alpha = infer_min_alpha

        self.vocab_size = 0
        self.word2idx = {}
        
    def build_vocab(self, documents, training=True, tokenize=True):
        if tokenize:
            docs = [self._tokenizer_space(document) for document in documents]
        else:
            docs = documents

        vocab = {}
        for words in docs:
            for word in words:
                if word not in vocab:
                    vocab[word] = 1

        if training:
            self.vocab_size = len(vocab)

        
        docs = [TaggedDocument(doc, [i]) for i, doc in enumerate(docs)]
        return docs
        
    def train(self, documents):
        self.model = Doc2Vec(
            documents, dm_concat=self.dm_concat, dm_mean=self.dm_mean, 
            dm=self.dm, vector_size=self.word_dimension, 
            window=self.word_window, min_count=self.min_word_count, 
            workers=self.n_job)
        
        self.model.train(
            documents, total_words=self.vocab_size, 
            epochs=self.train_epoch)
        
    def encode(self, documents, tokenize=True):
        if tokenize:
            docs = [self._tokenizer_space(document) for document in documents]
        else:
            docs = documents
            
        docs = [
            self.model.infer_vector(
                document, alpha=self.infer_alpha, 
                min_alpha=self.infer_min_alpha, 
                steps=self.infer_epoch)
            for document in docs
        ]
            
        return docs
        
        
        
        