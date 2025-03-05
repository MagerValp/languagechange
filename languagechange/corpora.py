import os
import gzip
import random
from languagechange.resource_manager import LanguageChange
from languagechange.usages import Target, TargetUsage, TargetUsageList
import re
from languagechange.utils import LiteralTime, NumericalTime, TimeInterval
from sortedcontainers import SortedKeyList
import logging
import lxml.etree as ET
import trankit
from typing import List, Union, Self
import datetime

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


class Line:

    def __init__(self, raw_text=None, tokens=None, lemmas=None, pos_tags=None, fname=None):
        self._raw_text = raw_text
        self._tokens = tokens
        self._lemmas = lemmas
        self._pos_tags = pos_tags
        self._fname = fname

    def tokens(self):
        if not self._tokens == None:
            return self._tokens
        else:
            return self._lemmas

    def lemmas(self):
        return self._lemmas

    def pos_tags(self):
        return self._pos_tags

    def raw_text(self):
        if not self._raw_text == None:
            return self._raw_text
        else:
            if not self._tokens == None:
                return ' '.join(self._tokens)
            elif not self._lemmas == None:
                return ' '.join(self._lemmas)
            else:
                raise Exception('No valid data in Line')

    def __str__(self):
        return self._raw_text


class Corpus:

    def __init__(self, name, language=None, time=LiteralTime('no time specification'), skip_lines=0, **args):
        self.name = name
        self.language = language
        self.time = time
        self.skip_lines = skip_lines


    def set_sentences_iterator(self, sentences):
        self.sentences_iterator = sentences


    def search(self, words, strategy='REGEX', search_func=None):

        for j,w in enumerate(words):
            if type(w) == str:
                words[j] = Target(w)

        if search_func == None:
            def search_func(word,line):
                offsets = []
                rex = re.compile(f'( |^)+{word}( |$)+',re.MULTILINE)
                for fi in re.finditer(rex, line):
                    s = line[fi.start():fi.end()].find(word)
                    offsets.append([fi.start()+s, fi.start()+s+len(word)])
                return offsets

        usage_dictionary = {} # need to be saved in cache

        if strategy == 'REGEX':

            for word in words:
                usage_dictionary[word.target] = TargetUsageList()

            logging.info("Scanning the corpus..")
            n_usages = 0

            for line in self.line_iterator():
                for word in words:
                    for offsets in search_func(word.target, line.raw_text()):
                        usage_dictionary[word.target].append(TargetUsage(line.raw_text(), offsets, self.time))
                        n_usages = n_usages + 1

            logging.info(f"{n_usages} usages found.")
        else:

            if type(strategy) == str:
                strategy = set([s.strip().upper() for s in strategy.split('+')])
            elif type(strategy) == list:
                strategy = set([s.upper() for s in strategy])

            for word in words:
                word_form = word.target if 'INFLECTED' in strategy else word.lemma
                usage_dictionary[word_form] = TargetUsageList()

            logging.info("Scanning the corpus..")
            n_usages = 0

            for line in self.line_iterator():
                line_tokens = line.tokens() if 'INFLECTED' in strategy else line.lemmas()
                if line_tokens == None:
                    raise Exception(f"Some of the required features {strategy} are not available for Corpus {self.name}")
                for j,token in enumerate(line_tokens):
                    for word in words:
                        word_form = word.target if 'INFLECTED' in strategy else word.lemma
                        if word_form == token:
                            if (not 'POS' in strategy) or ('POS' in strategy and word_form.pos == line.pos[j]):
                                offsets = [0,0]
                                if not j == 0:
                                    offsets[0] = len(' '.join(line.tokens()[:j])) + 1
                                offsets[1] = offsets[0] + len(line.tokens()[j])
                                usage_dictionary[word_form].append(TargetUsage(' '.join(line.tokens()), offsets, self.time))
                                n_usages = n_usages + 1

            logging.info(f"{n_usages} usages found.")
        return usage_dictionary
    

    def tokenize(self, tokenizer = "trankit", split_sentences=False, batch_size=128):
        if tokenizer == "trankit":
            p = trankit.Pipeline(self.language)

            if split_sentences:

                def process_lines(texts):
                    tokenized = p.tokenize(' '.join(texts))
                    for sentence in tokenized['sentences']:
                        yield Line(raw_text=sentence['text'], tokens=[token['text'] for token in sentence['tokens']])

                texts = []
                for line in self.line_iterator():
                    text = line.raw_text()
                    if type(text) == str and len(text.strip()) > 0:
                        texts.append(text)
                    if len(texts) == batch_size:
                        for line in process_lines(texts):
                            yield line
                        texts = []
                if texts != []:
                    for line in process_lines(texts):
                        yield line  
                        
            else:
                for line in self.line_iterator():
                    text = line.raw_text()
                    if type(text) == str and len(text.strip()) > 0:
                        tokenized_sentence = p.tokenize(text, is_sent=True)
                        line._tokens = [token['text'] for token in tokenized_sentence['tokens']]
                        yield line
        
        elif hasattr(tokenizer, "tokenize") and callable(getattr(tokenizer,"tokenize")):
            try:
                for line in self.line_iterator():
                    text = line.raw_text()
                    if type(text) == str and len(text.strip()) > 0:
                        line._tokens = [str(token) for token in tokenizer.tokenize(text)]
                        yield line
            except:
                logging.info(f"ERROR: Could not use method 'tokenize' within {tokenizer} directly as a function to tokenize.")

        elif callable(tokenizer):
            try:
                for line in self.line_iterator():
                    text = line.raw_text()
                    if type(text) == str and len(text.strip()) > 0:
                        line._tokens = [str(token) for token in tokenizer(text)]
                        yield line
            except:
                logging.info(f"ERROR: Could not use tokenizer {tokenizer} directly as a function to tokenize.")


    def lemmatize(self, lemmatizer = "trankit", pretokenized = False, tokenize = False, split_sentences = False, batch_size=128):
        if lemmatizer == "trankit":
            p = trankit.Pipeline(self.language)

            # input which is not sentence split
            if split_sentences:
                
                def process_texts(texts):
                    lemmatized = p.lemmatize(' '.join(texts))
                    lines = []
                    for sentence in lemmatized['sentences']:
                        lines.append(Line(raw_text=sentence['text'], lemmas=[token['lemma'] for token in sentence['tokens']], tokens=[token['text'] for token in sentence['tokens']] if tokenize else None))
                    return lines

                texts = []
                for line in self.line_iterator():
                    text = line.raw_text()
                    if type(text) == str and len(text.strip()) > 0:
                        texts.append(text)
                    if len(texts) == batch_size:
                        for line in process_texts(texts):
                            yield line
                        texts = []

                if texts != []:
                    for line in process_texts(texts):
                        yield line

            # input which is not pretokenized, but each line is its own sentence
            elif not pretokenized:
                for line in self.line_iterator():
                    text = line.raw_text()
                    if type(text) == str and len(text.strip()) > 0:
                        lemmatized_sentence = p.lemmatize(text, is_sent = True)
                        line._lemmas = [token['lemma'] for token in lemmatized_sentence['tokens']]
                        yield line

            # pretokenized input, one or more sentences at a time
            else:

                def modify_lines(lines):
                    lemmatized = p.lemmatize([line.tokens() for line in lines])
                    lemmatized_sentences = lemmatized['sentences']
                    for i, line in enumerate(lines):
                        line._lemmas = [token['lemma'] for token in lemmatized_sentences[i]['tokens']]
                        yield line

                lines = []
                for line in self.line_iterator():
                    tokens = line.tokens()
                    if type(tokens) == list and len(tokens) > 0:
                        lines.append(line)
                    if len(lines) == batch_size:
                        for line in modify_lines(lines):
                            yield line
                        lines = []
                if lines != []:
                    for line in modify_lines(lines):
                        yield line
                        

        # todo: add other lemmatizers if needed
        
        if hasattr(lemmatizer, "lemmatize") and callable(getattr(lemmatizer,"lemmatize")):
            try:
                for line in self.line_iterator():
                    text = line.raw_text()
                    if type(text) == str and len(text.strip()) > 0:
                        line._lemmas = [str(lemma) for lemma in lemmatizer.lemmatize(text)]
                        yield line
            except:
                logging.info(f"ERROR: Could not use method 'lemmatize' within {lemmatizer} directly as a function to lemmatize.")

        elif callable(lemmatizer):
            try:
                for line in self.line_iterator():
                    text = line.raw_text()
                    if type(text) == str and len(text.strip()) > 0:
                        line._lemmas = [str(lemma) for lemma in lemmatizer(text)]
                        yield line
            except:
                logging.info(f"ERROR: Could not use method {lemmatizer} directly as a function to lemmatize.")
    

    def pos_tagging(self, pos_tagger = "trankit", pretokenized = False, tokenize=False, split_sentences = False, batch_size=128):
        if pos_tagger == "trankit":
            p = trankit.Pipeline(self.language)

            # input which is not sentence split
            if split_sentences:

                def process_texts(texts):
                    pos_tagged = p.posdep(' '.join(texts))
                    for sentence in pos_tagged['sentences']:
                        yield Line(raw_text=sentence['text'], pos_tags=[token['upos'] for token in sentence['tokens']], tokens=[token['text'] for token in sentence['tokens']] if tokenize else None)

                texts = []
                for line in self.line_iterator():
                    text = line.raw_text()
                    if type(text) == str and len(text.strip()) > 0:
                        texts.append(text)
                    if len(texts) == batch_size:
                        for line in process_texts(texts):
                            yield line
                        texts = []

                if texts != []:
                    for line in process_texts(texts):
                        yield line

            # input which is not pretokenized, but each line is its own sentence
            elif not pretokenized:
                for line in self.line_iterator():
                    text = line.raw_text()
                    if type(text) == str and len(text.strip()) > 0:
                        pos_tagged_sentence = p.posdep(text, is_sent = True)
                        line._pos_tags = [token['upos'] for token in pos_tagged_sentence['tokens']]
                        if tokenize:
                            line._tokens = [token['text'] for token in pos_tagged_sentence['tokens']]
                        yield line

            # pretokenized input, one or more sentences at a time
            else:

                def modify_lines(lines):
                    pos_tagged = p.posdep([line.tokens() for line in lines])
                    pos_tagged_sentences = pos_tagged['sentences']
                    for i, line in enumerate(lines):
                        line._pos_tags = [token['upos'] for token in pos_tagged_sentences[i]['tokens']]
                        if tokenize:
                            line._tokens = [token['text'] for token in pos_tagged_sentences[i]['tokens']]
                        yield line

                lines = []
                for line in self.line_iterator():
                    tokens = line.tokens()
                    if type(tokens) == list and len(tokens) > 0:
                        lines.append(line)
                    if len(lines) == batch_size:
                        for line in modify_lines(lines):
                            yield line
                        lines = []
                        
                if lines != []:
                    for line in modify_lines(lines):
                        yield line


    def tokens_lemmas_pos_tags(self, nlp_model="trankit", split_sentences = False, batch_size=128):
        if nlp_model == "trankit":
            p = trankit.Pipeline(self.language)

            if not split_sentences:
                for line in self.line_iterator():
                    text = line.raw_text()
                    if type(text) == str and len(text.strip()) > 0:
                        lemmatized_sentence = p.lemmatize(text, is_sent = True)
                        line._lemmas = [token['lemma'] for token in lemmatized_sentence['tokens']]
                        line._tokens = [token['text'] for token in lemmatized_sentence['tokens']]
                        pos_tagged = p.posdep(line.tokens(), is_sent=True)
                        line._pos_tags = [token['upos'] for token in pos_tagged['tokens']]
                        yield line

            else:

                def process_texts(texts):
                    lemmatized_sentences = p.lemmatize(' '.join(texts))
                    tokens = []
                    for sentence in lemmatized_sentences['sentences']:
                        tokens.append([token['text'] for token in sentence['tokens']])
                    pos_tagged_sentences = p.posdep(tokens)
                    for i, sentence in enumerate(lemmatized_sentences['sentences']):
                        yield Line(raw_text=sentence['text'], tokens=[token['text'] for token in sentence['tokens']], lemmas=[token['lemma'] for token in sentence['tokens']],pos_tags=[token['upos'] for token in pos_tagged_sentences['sentences'][i]['tokens']])

                texts = []
                for line in self.line_iterator():
                    text = line.raw_text()
                    if type(text) == str and len(text.strip()) > 0:
                        texts.append(text)
                    if len(texts) == batch_size:
                        for line in process_texts(texts):
                            yield line
                        texts = []
                if len(texts) != 0:
                    for line in process_texts(texts):
                        yield line


    # preliminary function
    def segment_sentences(self, segmentizer = "trankit", batch_size=128):
        if segmentizer == "trankit":
            p = trankit.Pipeline(self.language)

            lines = []
            for line in self.line_iterator():
                lines.append(line.raw_text())
                if len(lines) == batch_size:
                    sentences = p.ssplit(' '.join(lines))
                    for sent in sentences['sentences']:
                        yield Line(sent['text'])
                    lines = []
            if len(lines) != 0:
                sentences = p.ssplit(' '.join(lines))
                for sent in sentences['sentences']:
                    yield Line(sent['text'])

        elif callable(segmentizer):
            try:
                lines = []
                for line in self.line_iterator():
                    lines.append(line.raw_text())
                    if len(lines) == batch_size:
                        sentences = segmentizer(' '.join(lines))
                        for sent in sentences:
                            yield Line(sent)
                        lines = []
                if len(lines) != 0:
                    sentences = segmentizer(' '.join(lines))
                    for sent in sentences:
                        yield Line(sent)
            except:
                logging.info(f"ERROR: Could not use method {segmentizer} directly as a function to split sentences.")


    def folder_iterator(self, path):

        fnames = []

        for fname in os.listdir(path):

            if os.path.isdir(os.path.join(path,fname)):
                fnames = fnames + self.folder_iterator(os.path.join(path,fname))
            else:
                fnames.append(os.path.join(path,fname))

        return fnames


    def cast_to_vertical(corpora, vertical_corpus):

        line_iterators = [corpus.line_iterator() for corpus in corpora]
        iterate = True

        with open(vertical_corpus.path,'w+') as f:

            while iterate:
                lines = []
                for iterator in line_iterators:
                    next_line = next(iterator)
                if not next_line == None:
                    vertical_lines = []
                    for j in range(len(lines[0])):
                        vertical_lines.append('{vertical_corpus.field_separator}'.join([lines[i][j] for i in range(len(lines))]))
                    for line in vertical_lines:
                        f.write(line+'\n')
                    f.write(vertical_corpus.sentence_separator)
                else:
                    iterate = False


    def save(self):
        lc = LanguageChange()
        path = lc.save_resource('corpus',f'{self.language} corpora',self.name)


    def save_tokenized_corpora(corpora : Union[Self, List[Self]], tokens = True, lemmas = False, pos_tags = False, save_format = 'linebyline', file_specification = None, file_ending = ".txt", tokenizer="trankit", lemmatizer="trankit", pos_tagger="trankit"):
        if not type(corpora) is list:
            corpora = [corpora]
        if file_specification == None:
            file_specification = ""
            file_specification += "-tokens" if tokens else '' 
            file_specification += '-lemmas' if lemmas else '' 
            file_specification += '-pos_tags' if pos_tags else ''
        for corpus in corpora:
            tokenized_name = os.path.splitext(corpus.path)[0]+file_specification+file_ending
            with open(tokenized_name, 'w+') as f:
                if save_format == 'linebyline':
                    if tokens:
                        for line in corpus.tokenize(tokenizer):
                            f.write(' '.join(line.tokens())+'\n') # cache needed here
                    elif lemmas:
                        for line in corpus.lemmatize(lemmatizer):
                            f.write(' '.join(line.lemmas())+'\n') # cache needed here
                    elif pos_tags:
                        for line in corpus.pos_tagging(pos_tagger):
                            f.write(' '.join(line.pos_tags())+'\n')
                elif save_format == 'vertical':
                    if tokens:
                        if lemmas:
                            for line in corpus.tokenize_lemmatize():
                                for pair in zip(*(line.tokens(), line.lemmas())):
                                    f.write('\t'.join(pair)+'\n') # cache needed here
                                f.write('\n') # cache needed here
                    elif tokens:
                        for line in corpus.tokenize(tokenizer):
                            f.write('\n'.join(line.tokens())+'\n') # cache needed here
                    elif lemmas:
                        for line in corpus.lemmatize(lemmatizer):
                            f.write('\n'.join(line.lemmas())+'\n') # cache needed here


class LinebyLineCorpus(Corpus):

    def __init__(self, path, **args):
        if not 'name' in args:
            name = path
        super().__init__(name,**args)
        self.path = path

        if 'is_sentence_tokenized' in args:
            self.is_sentence_tokenized = args['is_sentence_tokenized']
        else:
            self.is_sentence_tokenized = False

        if self.is_sentence_tokenized:
            if 'is_tokenized' in args:
                self.is_tokenized = args['is_tokenized']
        else:
            if 'is_tokenized' in args and args['is_tokenized']:
                self.is_sentence_tokenized = True
                self.is_tokenized = True
            else:
                self.is_sentence_tokenized = False
                self.is_tokenized = False

        if 'is_tokenized' in args and args['is_tokenized']:
            if 'is_lemmatized' in args:
                self.is_lemmatized = args['is_lemmatized']
            if 'tokens_splitter' in args:
                self.tokens_splitter = args.tokens_splitter
            else:
                self.tokens_splitter = ' '
        else:
            if 'is_lemmatized' in args and args['is_lemmatized']:
                self.is_sentence_tokenized = True
                self.is_tokenized = True
                self.is_lemmatized = True
                if 'tokens_splitter' in args:
                    self.tokens_splitter = args.tokens_splitter
                else:
                    self.tokens_splitter = ' '
            else:
                self.is_lemmatized = False


    def line_iterator(self):
        
        if os.path.isdir(self.path):
            fnames = self.folder_iterator(self.path)
        else:
            fnames = [self.path]

        def get_data(line):
            line = line.replace('\n','')
            data = {}
            data['raw_text'] = line
            if self.is_lemmatized:
                data['lemmas'] = line.split(self.tokens_splitter)
            elif self.is_tokenized:
                data['tokens'] = line.split(self.tokens_splitter)
            return data

        for fname in fnames:

            if fname.endswith('.txt'):
                with open(fname,'r') as f:
                    for i, line in enumerate(f):
                        if i >= self.skip_lines:
                            data = get_data(line)
                            yield Line(fname=fname, **data)

            elif fname.endswith('.gz'):
                with gzip.open(fname, mode="rt") as f:
                    for i, line in enumerate(f):
                        if i >= self.skip_lines:
                            data = get_data(line)
                            yield Line(fname=fname, **data)

            else:
                raise Exception('Format not recognized')


class VerticalCorpus(Corpus):

    def __init__(self, path, sentence_separator='\n', field_separator='\t', field_map={'token':0, 'lemma':1, 'pos_tag':2}, **args):
        super().__init__(name=path,**args)
        self.path = path
        self.sentence_separator = sentence_separator
        self.field_separator = field_separator
        self.field_map = field_map


    def line_iterator(self):
        
        if os.path.isdir(self.path):
            fnames = self.folder_iterator(self.path)
        else:
            fnames = [self.path]

        def get_data(line):
            data = {}
            splitted_line = [vertical_line.split(self.field_separator) for vertical_line in line]
            raw_text = [vertical_line[self.field_map['token']] for vertical_line in splitted_line]
            data['raw_text'] = ' '.join(raw_text)
            data['tokens'] = raw_text
            if 'lemma' in self.field_map:
                lemma_text = [vertical_line[self.field_map['lemma']] for vertical_line in splitted_line]
                data['lemmas'] = lemma_text
            if 'pos_tag' in self.field_map:
                pos_text = [vertical_line[self.field_map['pos_tag']] for vertical_line in splitted_line]     
            return data

        for fname in fnames:

            if fname.endswith('.txt'):
                with open(fname,'r') as f:
                    line = []
                    for i, vertical_line in enumerate(f):
                        if i >= self.skip_lines:
                            if vertical_line == self.sentence_separator:
                                data = get_data(line)
                                yield Line(fname=fname, **data)
                                line = []
                            else:
                                line.append(vertical_line)

            elif fname.endswith('.gz'):
                with gzip.open(fname, mode="rt") as f:
                    for i, vertical_line in enumerate(f):
                        if i >= self.skip_lines:
                            if vertical_line == self.sentence_separator:
                                data = get_data(line)
                                yield Line(fname=fname, **data)
                                line = []
                            else:
                                line.append(vertical_line)

            else:
                raise Exception('Format not recognized')
            

# Should be able to load and parse a corpus in XML format.
# Supports only tokenized corpora so far.
class XMLCorpus(Corpus):

    def __init__(self, path, sentence_tag='sentence',token_tag='token', is_lemmatized=False, lemma_tag=None, **args):
        if not 'name' in args:
            name = path
        super().__init__(name, **args)
        self.path = path

        if lemma_tag:
            self.lemma_tag = lemma_tag
        else:
            self.lemma_tag = ''

        if is_lemmatized:
            self.is_lemmatized = True
            if lemma_tag != '':
                self.lemma_tag = lemma_tag
            else:
                self.lemma_tag = 'lemma'
        else:
            self.is_lemmatized = False
            self.lemma_tag = ''

        self.sentence_tag = sentence_tag
        self.token_tag = token_tag

    
    def get_attribute(self, tag, attribute):
        return tag.attrib[attribute]


    def line_iterator(self):
        if os.path.isdir(self.path):
            fnames = self.folder_iterator(self.path)
        else:
            fnames = [self.path]

        def get_data(tokens, lemmas = []):
            data = {}
            data['raw_text'] = ' '.join(tokens)
            if self.is_lemmatized and lemmas != []:
                data['lemmas'] = lemmas
            data['tokens'] = tokens
            return data

        for fname in fnames:
            if fname.endswith('.xml'):
                tokens = []
                lemmas = []
                parser = ET.iterparse(fname, events=('start','end'))
                parser = iter(parser)
                event, root = next(parser)
                for event, elem in parser:
                    if elem.sourceline >= self.skip_lines:
                        if elem.tag == self.sentence_tag:
                            if event == 'start':
                                tokens = []
                                lemmas = []
                            # If the sentence has ended, create a new Line object with its content
                            elif event == 'end':
                                if tokens != []:
                                    data = get_data(tokens, lemmas)
                                    yield Line(fname=fname, **data)
                                    elem.clear()
                        elif elem.tag == self.token_tag:
                            if event == 'end':
                                if self.is_lemmatized:
                                    lemma = self.get_attribute(elem, self.lemma_tag)
                                    lemmas.append(lemma)
                                token = elem.text
                                tokens.append(token)
                                elem.clear()
                     

            else:
                raise Exception('Format not recognized')

    
    # Cast to a LineByLine corpus and save the result in the path specified in there
    def cast_to_linebyline(self, linebyline_corpus : LinebyLineCorpus):
        savepath = linebyline_corpus.path
        if hasattr(linebyline_corpus, 'tokens_splitter'):
            tokens_splitter = linebyline_corpus.tokens_splitter
        else:
            tokens_splitter = ' '
        tokenized = linebyline_corpus.is_tokenized
        lemmatized = linebyline_corpus.is_lemmatized
        if lemmatized and not self.is_lemmatized:
            logging.info('ERROR: cannot cast to lemmatized LinebyLineCorpus because this XMLCorpus is not lemmatized.')
            return None
        with open(savepath, 'w+') as f:
            if lemmatized:
                for line in self.line_iterator():
                    f.write(tokens_splitter.join(line.lemmas())+'\n')  # cache needed here
            elif tokenized:
                for line in self.line_iterator():
                    f.write(tokens_splitter.join(line.tokens())+'\n')  # cache needed here
            else:
                for line in self.line_iterator():
                    f.write(line.raw_text()+'\n')  # cache needed here


    def cast_to_vertical(self, vertical_corpus : VerticalCorpus):
        savepath = vertical_corpus.path
        field_separator = vertical_corpus.field_separator
        sentence_separator = vertical_corpus.sentence_separator
        # We need to make sure that the line features (token, lemma, pos, etc.) come in the same order as in the field_map in the vertical_corpus
        sorted_field_names = [key for (key, value) in sorted(vertical_corpus.field_map.items(), key = lambda x : x[1])]
        
        def get_line_feature(line, key):
            field_name_to_line_feature = {'token': line.tokens, 'lemma': line.lemmas, 'pos_tag': line.pos_tags}
            return field_name_to_line_feature[key]()
        
        with open(savepath,'w+') as f:
            for line in self.line_iterator():
                for t in zip(*(get_line_feature(line, key) for key in sorted_field_names)):
                    f.write(field_separator.join(list(t))+'\n') # cache needed here
                f.write(sentence_separator) # cache needed here


# A class for handling XML corpora specifically from spraakbanken.gu.se
class SprakBankenCorpus(XMLCorpus):

    def __init__(self, path, sentence_tag='sentence',token_tag='token', is_lemmatized=True, lemma_tag='lemma', **args):
        super().__init__(path, sentence_tag, token_tag, is_lemmatized, lemma_tag, **args)

    
    def get_attribute(self, tag, attribute):
        content = tag.attrib[attribute]
        if content != None:
            content = content.strip("|").split("|")
            if content != ['']:
                return content[0]
        return tag.text


# todo: add ways to extract the year from corpora
class HistoricalCorpus(SortedKeyList):

    def __init__(self, corpora:list[Corpus]):
        super().__init__(corpora, key= lambda x: x.time)

