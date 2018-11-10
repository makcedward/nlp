import re, os
from collections import Counter
from symspellpy.symspellpy import SymSpell as SymSpellPy, Verbosity

class SpellCheck:
    def __init__(self, dictionary=None, verbose=0):
        self.verbose = verbose
        self.dictionary = dictionary
        
    def correction(self, text):
        return ''


'''
    Source: https://norvig.com/spell-correct.html
'''
class SpellCorrector(SpellCheck):
    def __init__(self, dictionary, verbose=0):
        super().__init__(dictionary=dictionary, verbose=verbose)

    def words(text):
        return re.findall(r'\w+', text.lower())

    def P(self, word): 
        "Probability of `word`."
        N = sum(self.dictionary.values())
        return self.dictionary[word] / N

    def correction(self, word): 
        "Most probable spelling correction for word."
        return max(self.candidates(word), key=self.P)

    def candidates(self, word, verbose=0): 
        "Generate possible spelling corrections for word."
        
        known_result = self.known([word])
        edit1_result = self.known(self.edits1(word))
        edit2_result = self.known(self.edits2(word))
        
        if self.verbose > 0 or verbose > 0:
            print('Known Result: ', known_result)
            print('Edit1 Result: ', edit1_result)
            print('Edit2 Result: ', edit2_result)
        
        return (known_result or edit1_result or edit2_result or [word])

    def known(self, words):
        "The subset of `words` that appear in the dictionary of WORDS."
        return set(w for w in words if w in self.dictionary)

    def edits1(self, word):
        "All edits that are one edit away from `word`."
        letters    = 'abcdefghijklmnopqrstuvwxyz'
        splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
        deletes    = [L + R[1:]               for L, R in splits if R]
        transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
        replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
        inserts    = [L + c + R               for L, R in splits for c in letters]
        return set(deletes + transposes + replaces + inserts)

    def edits2(self, word): 
        "All edits that are two edits away from `word`."
        return (e2 for e1 in self.edits1(word) for e2 in self.edits1(e1))
    
    
class SymSpell(SpellCheck):
    def __init__(self, dictionary_file_path='', dictionary=None, verbose=0):
        super().__init__(dictionary=dictionary, verbose=verbose)
        
        self.dictionary_file_path = dictionary_file_path
        self.model = None
        
    def load_vocab(self, corpus_file_path, max_edit_distance_dictionary=2, prefix_length=5):
        #initial_capacity = len(corpus)
        
        #sym_spell = SymSpellPy(
        #    initial_capacity, max_edit_distance_dictionary, 
        #    prefix_length)
        self.model = SymSpellPy(
            max_dictionary_edit_distance=max_edit_distance_dictionary, 
            prefix_length=prefix_length)

        term_index = 0  # column of the term in the dictionary text file
        count_index = 1  # column of the term frequency in the dictionary text file
        if not self.model.load_dictionary(corpus_file_path, term_index, count_index):
            print("Dictionary file not found")
        
    def build_vocab(self, dictionary, file_dir, file_name, verbose=0):
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)

        """
            Data format:
                token, frequency
            Example:
                edward 154
                edwards 50
                ...
        """ 
        if self.verbose > 3 or verbose > 3:
            print('Size of dictionary: %d' % len(dictionary))

        with open(file_dir + file_name, "w") as text_file:
            for token, count in dictionary.items():
                text_file.write(token + ' ' + str(count))
                text_file.write('\n')
        
    def correction(self, word, max_edit_distance_lookup=2, mode='cloest'): 
        if mode == 'cloest':
            suggestion_verbosity = Verbosity.CLOSEST
        elif mode == 'top':
            suggestion_verbosity = Verbosity.TOP
        elif mode == 'all':
            suggestion_verbosity = Verbosity.ALL
              
        results = self.model.lookup(
            word, suggestion_verbosity, max_edit_distance_lookup)
        
        results = [{'word': suggestion.term, 'count': suggestion.count, 'distance': suggestion.distance} for suggestion in results]
        return results
    
    def corrections(self, sentence, max_edit_distance_lookup=2):
        normalized_sentence = (sentence.lower())
        results = self.model.lookup_compound(
            normalized_sentence, max_edit_distance_lookup)
        
        results = [{'word': suggestion.term, 'distance': suggestion.distance} for suggestion in results]
        return results
