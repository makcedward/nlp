import datetime, os, urllib, zipfile


class Embeddings:
    def __init__(self, verbose=0):
        self.verbose = verbose
        self.model = {}
        self.model_path = ''
        
    def _log_time(self, status, msg, verbose):
        if self.verbose >= 10 or verbose >= 10:
            print('%s. [%s] %s' % (datetime.datetime.now(), status, msg))
            
    def download(self, src, dest_dir, dest_file, uncompress, housekeep=False, verbose=0):
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)
    
        if dest_file is None:
            dest_file = os.path.basename(src)
            
        if not self.is_file_exist(dest_dir + dest_file):
            self._log_time(status='DOWNLOAD', msg='From '+src+' to '+dest_dir+dest_file, verbose=verbose)
            file = urllib.request.urlopen(src)
            with open(dest_dir + dest_file,'wb') as output:
                output.write(file.read())
        else:
            self._log_time(status='FOUND', msg=dest_file+' in '+dest_dir, verbose=verbose)
            
        if uncompress:
            self.uncompress(dest_dir + dest_file)
            
        if uncompress and housekeep:
            self.housekeep(dest_dir + dest_file)
            
            
        return dest_dir + dest_file
    
    """
        File related
    """
    
    def uncompress(self):
        raise NotImplemented()
    
    def unzip(self, file_path):
        dest_dir = os.path.dirname(file_path)
        with zipfile.ZipFile(file_path, "r") as zip_ref:
            zip_ref.extractall(dest_dir)
            
    def housekeep(self, file_path):
        os.remove(file_path)
        
    def is_file_exist(self, file_path):
        return os.path.exists(file_path)
    
    def save(self):
        raise NotImplemented()
        
    def load(self):
        raise NotImplemented()
    
    """
        Model related
    """
        
    def get_model(self):
        return self.model
    
    def set_model(self, model):
        self.model = model
    
    def load(self, src=None, dest_dir=None, trainable=True, verbose=0):
        raise NotImplemented()
        
    """
        Vocabulary realted
    """
    
    def load_vocab(self, **kwargs):
        raise NotImplemented()
        
    def build_vocab(self):
        raise NotImplemented()
    
    def get_vocab(self):
        raise NotImplemented()
        
    def _tokenizer_space(self, sentence):
        return sentence.split(' ')
        
    """
        Vector related
    """
    
    def train(self):
        raise NotImplemented()
        
    def encode(self, sentences):
        raise NotImplemented()
        
    def visualize(self):
        raise NotImplemented()
        
    """
        Netowrk realted
    """
    
    def to_numpy_layer(self):
        raise NotImplemented()
        
    def to_keras_layer(self):
        raise NotImplemented()
        
    def to_tensorflow_layer(self):
        raise NotImplemented()
    
    def to_pytorch_layer(self):
        raise NotImplemented()