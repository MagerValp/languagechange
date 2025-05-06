from typing import List, Set

def expand_dictionary(words: List[str]):
    raise NotImplementedError

class SearchTerm():

    VALID_WORD_FEATURES = ['lemma', 'token', 'pos']

    def __init__(self, term : str, regex : bool = False, word_feature : str | Set = 'token'):
        self.term = term 
        self.regex = regex
        self.word_feature = word_feature if isinstance(word_feature, Set) else {word_feature}
        if not self.word_feature.issubset(self.VALID_WORD_FEATURES):
            raise ValueError("'word_feature' must be set to one of the following values: ", self.VALID_WORD_FEATURES)