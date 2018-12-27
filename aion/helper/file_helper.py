import datetime, os, urllib.request, zipfile


class FileHelper:
    def __init__(self, verbose=0):
        self.verbose = verbose
        
    def _log_time(self, status, msg, verbose):
        if self.verbose >= 0 or verbose >= 0:
            print('%s. [%s] %s' % (datetime.datetime.now(), status, msg))
            
    def is_file_exist(self, file_path):
        return os.path.exists(file_path)
            
    def download(self, src, dest_dir, dest_file, uncompress=False, housekeep=False, force_download=False, verbose=0):
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)
            
#         print('dest_dir:', dest_dir)
    
        if dest_file is None:
            dest_file = os.path.basename(src)
            
#         print('dest_file:', dest_file)
            
        if not self.is_file_exist(dest_dir + dest_file) or force_download:
            self._log_time(status='DOWNLOAD', msg='From '+src+' to '+dest_dir+dest_file, verbose=verbose)
            file = urllib.request.urlopen(src)
            with open(dest_dir + dest_file,'wb') as output:
                output.write(file.read())
        else:
            self._log_time(status='FOUND', msg=dest_file+' in '+dest_dir, verbose=verbose)
            
#         if uncompress:
#             self.uncompress(dest_dir + dest_file)
            
#         if uncompress and housekeep:
#             self.housekeep(dest_dir + dest_file)
            
        return dest_dir + dest_file