from languagechange.resource_manager import LanguageChange
from languagechange.corpora import LinebyLineCorpus
from languagechange.usages import Target, TargetUsageList, DWUGUsage
from languagechange.utils import NumericalTime, LiteralTime
import webbrowser
import os
import pickle
import lxml.etree as ET
import logging
from itertools import islice
import random
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, f1_score
from typing import List, Dict



class Benchmark():

    def __init__(self):
        pass


class SemEval2020Task1(Benchmark):

    def __init__(self, language):
        lc = LanguageChange()
        self.language = language
        home_path = lc.get_resource('benchmarks', 'SemEval 2020 Task 1', self.language, 'no-version')
        semeval_folder = os.listdir(home_path)[0]
        self.home_path = os.path.join(home_path,semeval_folder)
        self.load()


    def load(self):
        self.corpus1_lemma = LinebyLineCorpus(os.path.join(self.home_path, 'corpus1', 'lemma'), name='corpus1_lemma', language=self.language, time=NumericalTime(1), is_lemmatized=True)
        self.corpus2_lemma = LinebyLineCorpus(os.path.join(self.home_path, 'corpus2', 'lemma'), name='corpus2_lemma', language=self.language, time=NumericalTime(2), is_lemmatized=True)
        self.corpus1_token = LinebyLineCorpus(os.path.join(self.home_path, 'corpus1', 'token'), name='corpus1_token', language=self.language, time=NumericalTime(1), is_tokenized=True)
        self.corpus2_token = LinebyLineCorpus(os.path.join(self.home_path, 'corpus2', 'token'), name='corpus2_token', language=self.language, time=NumericalTime(2), is_tokenized=True)
        self.binary_task = {}
        self.graded_task = {}

        with open(os.path.join(self.home_path, 'truth', 'binary.txt')) as f:
            for line in f:
                word, label = line.split()
                word = Target(word)
                self.binary_task[word] = int(label)

        with open(os.path.join(self.home_path, 'truth', 'graded.txt')) as f:
            for line in f:
                word, score = line.split()
                word = Target(word)
                self.graded_task[word] = float(score)


class DWUG(Benchmark):

    def __init__(self, path=None, language=None, version=None):
        lc = LanguageChange()
        if not language == None and not version == None:
            self.language = language
            home_path = lc.get_resource('benchmarks', 'DWUG', self.language, version)
            dwug_folder = os.listdir(home_path)[0]
            self.home_path = os.path.join(home_path,dwug_folder)
        else:
            if not path == None and os.path.exists(path):
                self.home_path = path
            else:
                raise Exception('The path is None or does not exists.')       
        self.load()

    def load(self, config=None):
        self.target_words = os.listdir(os.path.join(self.home_path,'data'))
        self.stats_groupings = {}
        self.stats = {}

        stats_path = None
        if not config == None:
            stats_path = os.path.join(self.home_path,'stats',config)
        elif os.path.exists(os.path.join(self.home_path,'stats','opt')):
            stats_path = os.path.join(self.home_path,'stats','opt')
        else:
            stats_path = os.path.join(self.home_path,'stats')

        with open(os.path.join(stats_path,'stats_groupings.csv')) as f:
            keys = []
            for j,line in enumerate(f):
                line = line.replace('\n','').split('\t')
                if j > 0:
                    values = line
                    D = {keys[j]:values[j] for j in range(1,len(values))}
                    self.stats_groupings[values[0]] = D
                else:
                    keys = line

        with open(os.path.join(stats_path,'stats.csv')) as f:
            keys = []
            for j,line in enumerate(f):
                line = line.replace('\n','').split('\t')
                if j > 0:
                    values = line
                    D = {keys[j]:values[j] for j in range(1,len(values))}
                    self.stats[values[0]] = D
                else:
                    keys = line

        self.binary_task = {}
        self.graded_task = {}
        self.binary_gain_task = {}
        self.bianry_loss_task = {}

        for lemma in self.stats_groupings:

            word = Target(lemma)
            word.set_lemma(lemma)
            self.binary_task[word] = int(self.stats_groupings[lemma]['change_binary'])
            self.graded_task[word] = float(self.stats_groupings[lemma]['change_graded'])
            self.binary_gain_task[word] = int(self.stats_groupings[lemma]['change_binary_gain'])
            self.bianry_loss_task[word] = int(self.stats_groupings[lemma]['change_binary_loss'])


    def get_usage_graph(self, word):
        with open(os.path.join(self.home_path,'graphs','opt',word),'rb') as f:
            return pickle.load(f)

    def show_usage_graph(self, word, config=None):
        def run_from_ipython():
            try:
                __IPYTHON__
                return True
            except NameError:
                return False

        def search_plot_path(path):
            if 'weight' in os.listdir(path):
                return path
            else:
                return search_plot_path(os.path.join(path,os.listdir(path)[0]))

        plot_path = None

        if config == None:
            path = search_plot_path(os.path.join(self.home_path,'plots'))
            plot_path = os.path.join(path,'weight','full')   
        else:
            plot_path = os.path.join(self.home_path,'plots',config,'weight','full') 


        if not run_from_ipython():
            webbrowser.open(os.path.join(plot_path,f'{word}.html'))
        else:
            from IPython.display import display, HTML
            with open(os.path.join(plot_path,f'{word}.html')) as f:
                html = f.read()
                display(HTML(html))

    def get_word_usages(self, word, group='all'):
        group = str(group)
        usages = TargetUsageList()
        with open(os.path.join(self.home_path,'data',word,'uses.csv')) as f:
            keys = []
            for j,line in enumerate(f):
                line = line.replace('\n','').split('\t')
                if j > 0:
                    values = line
                    D = {keys[j]:values[j] for j in range(len(values))}
                    if group == 'all' or D['grouping'] == group:
                        D['text'] = D['context']
                        D['target'] = Target(D['lemma'])
                        D['target'].set_lemma(D['lemma'])
                        D['target'].set_pos(D['pos'])
                        D['offsets'] = [int(i) for i in D['indexes_target_token'].split(':')]
                        D['time'] = LiteralTime(D['date'])
                        usages.append(DWUGUsage(**D))
                else:
                    keys = line
        return usages

    def get_word_annotations(self, word):
        usages = TargetUsageList()
        with open(os.path.join(self.home_path,'data',word,'uses.csv')) as f:
            keys = []
            for j,line in enumerate(f):
                line = line.replace('\n','').split('\t')
                if j > 0:
                    values = line
                    D = {keys[j]:values[j] for j in range(len(values))}
                    D['text'] = D['context']
                    D['target'] = Target(D['lemma'])
                    D['target'].set_lemma(D['lemma'])
                    D['target'].set_pos(D['pos'])
                    D['offsets'] = [int(i) for i in D['indexes_target_token'].split(':')]
                    D['time'] = LiteralTime(D['date'])
                    usages.append(DWUGUsage(**D))
                else:
                    keys = line
        return usages

    def get_stats(self):
        return self.stats

    def get_stats_groupings(self):
        return self.get_stats_groupings
    

    
    """
What we want for cast to WSD:

In WSD, we have an example consisting of a target word in the sentence, and we want to predict the sense of the target word given a predefined sense inventory. In the case of DWUGs, we have usages of different words being assigned to different clusters. Each of these clusters we assume correspond to a sense of the word.



    """
    def cast_to_WSD(self, remove_outliers = True):
        data = []
        for word in self.stats_groupings:
            usages_by_id = {}
            with open(os.path.join(self.home_path,'clusters/opt',f'{word}.csv')) as f:
                for line in islice(f, 1, None):
                    line = line.replace('\n','').split('\t')
                    id, label = line
                    if not remove_outliers or int(label) != -1:
                        usages_by_id[id] = {'id': id, 'label': label}

            
            with open(os.path.join(self.home_path,'data',word,'uses.csv')) as f:
                for line in islice(f, 1, None):
                    line = line.replace('\n','').split('\t')
                    lemma = line[0]
                    id = line[4]
                    if id in usages_by_id:
                        context_tokenized = line[9]
                        word_index = int(line[10])
                        start, end = get_start_end(context_tokenized, word_index)
                        usages_by_id[id].update({'word': lemma, 'text':context_tokenized, 'start':start, 'end':end})
            data.extend(list(usages_by_id.values()))
        
        wsd = WSD()
        wsd.load_from_data(data)
        return wsd

        

# Dataset handling for the Word Sense Disambiguation (WSD) task
class WSD(Benchmark):
    def __init__(self, path = None, dataset = None, language = None):
        # What we want:
        #lc = LanguageChange()
        #home_path = lc.get_resource('benchmarks', 'WSD', dataset, version)

        if dataset == 'XL-WSD':
            print("Create home path")
            self.home_path = path
        else:
            self.home_path = None

        self.words = set()

        if self.home_path is not None:
            self.load(dataset, language)

    # Loads already formatted data, with each example as in self.load()
    def load_from_data(self, data):
        if type(data) == list:
            self.data = {'all': data}
        elif type(data) == dict:
            self.data = data

    # Finds the file paths of the data and labels for possible train, dev and test sets.
    def find_data_paths(self, dataset, language):
        
        train_paths = {'data':None, 'labels':None}
        dev_paths= {'data':None, 'labels':None}
        test_paths = {'data':None, 'labels':None}
        data_paths = {'train':train_paths, 'dev':dev_paths, 'test':test_paths}

        if dataset == 'XL-WSD':

            if os.path.exists(os.path.join(self.home_path, f'training_datasets/semcor_{language.lower()}')):
                data_paths['train']['data'] = f'training_datasets/semcor_{language.lower()}/semcor_{language.lower()}.data.xml'
                data_paths['train']['labels'] = f'training_datasets/semcor_{language.lower()}/semcor_{language.lower()}.gold.key.txt'
            else:
                logging.info(f'No train set found for {language}.')
            
            if os.path.exists(os.path.join(self.home_path, f'evaluation_datasets/dev-{language.lower()}')):
                data_paths['dev']['data'] = f'evaluation_datasets/dev-{language.lower()}/dev-{language.lower()}.data.xml'
                data_paths['dev']['labels'] = f'evaluation_datasets/dev-{language.lower()}/dev-{language.lower()}.gold.key.txt'
            else:
                logging.info(f'No dev set found for {language}. Did you enter the right language code?')

            if os.path.exists(os.path.join(self.home_path, f'evaluation_datasets/test-{language.lower()}')):
                data_paths['test']['data'] = f'evaluation_datasets/test-{language.lower()}/test-{language.lower()}.data.xml'
                data_paths['test']['labels'] = f'evaluation_datasets/test-{language.lower()}/test-{language.lower()}.gold.key.txt'
            else:
                logging.info(f'No test set found for {language}. Did you enter the right language code?')

        return data_paths
    
    # Reads an XML containing WSD data excl. labels
    def read_xml(self, path):
        parser = ET.iterparse(path, events=('start', 'end'))

        data = []

        sentence_tag = 'sentence'
        word_tag = 'wf'
        target_tag = 'instance'

        for event, elem in parser:

            if elem.tag == sentence_tag and event == 'start':
                sent = {'text': [], 'target_words': {}}

            elif elem.tag == word_tag and event == 'end':
                sent['text'].append(elem.text)

            elif elem.tag == target_tag and event == 'end':
                sent['text'].append(elem.text)
                sent['target_words'][elem.attrib['id']] = {'lemma': elem.attrib['lemma'], 'index': len(sent['text']) - 1}

            elif elem.tag == sentence_tag and event == 'end':
                data.append(sent)

            #TODO: elem.clear in the appropriate places

        return data

    def load_from_files(self, data_paths, dataset):

        def get_start_end(text, word_index):
            start = sum(len(s) + 1 for s in text[:word_index])
            end = start + len(text[word_index])
            return start, end
        
        data = {'train':[], 'dev':[], 'test':[]}
        
        if dataset == 'XL-WSD':
            print("Dataset is XL-WSD")
            for key in data_paths.keys():

                data_by_id = {}

                if data_paths[key]['data'] is not None:

                    raw_data = self.read_xml(os.path.join(self.home_path, data_paths[key]['data']))
                    for d in raw_data:
                        for id, target in d['target_words'].items():
                            start, end = get_start_end(d['text'], target['index'])
                            data_by_id[id] = {'text': " ".join(d['text']), 'word': target['lemma'], 'start': start, 'end': end}

                if data_paths[key]['labels'] is not None:
                    with open(os.path.join(self.home_path, data_paths[key]['labels'])) as f:
                        for line in f:
                            line_data = line.strip("\n").split(" ")
                            id = line_data[0]
                            labels = line_data[1:]
                            data_by_id[id]['label'] = labels
                            data_by_id[id]['id'] = id

                data[key] = list(data_by_id.values())

            self.data = data


    def load(self, dataset, language):
        data_paths = self.find_data_paths(dataset, language)
        self.load_from_files(data_paths, dataset)

    def get_dataset(self, key):
        if key in self.data.keys():
            return self.data[key]
        else:
            raise KeyError

    def evaluate(self, predictions : List[Dict] | Dict, dataset, metric, word = None):
        """
            Evaluates predictions by comparing them to the true labels of the dataset.
            Args:
                predictions (list(dict) | dict) : the predictions. If a dict, id:s are expected in both this dict and the
                dataset to compare against.
                dataset (str) : one of ['train','dev','test','dev_larger',...]
                metric (function) : a metric such as scipy.stats.spearmanr, that can be used to compare the predictions
        """
        dataset = self.get_dataset(dataset)

        if word is not None:
            if not word in self.words:
                logging.error(f'Word {word} was not found.')
                raise ValueError
            dataset = filter(lambda d : d['word'] == word, dataset)

        if type(predictions) == dict and 'id' in predictions.keys():
            for d in dataset:
                if not 'id' in d.keys():
                    logging.error('Could not find id:s for all examples in the dataset.')
                    raise KeyError
            pred = [predictions[ex['id']] for ex in dataset]
        else:
            pred = predictions
        truth = [ex['label'] for ex in dataset]
        try:
            stats = metric(truth, pred)
            return stats
        except:
            logging.error(f'Could not use {metric} to compare the true and predicted labels.')

    def evaluate_spearman(self, predictions : List[Dict] | Dict, dataset = 'test', word = None):
        return self.evaluate(predictions, dataset, spearmanr, word)

    def evaluate_accuracy(self, predictions : List[Dict] | Dict, dataset = 'test', word = None):
        return self.evaluate(predictions, dataset, accuracy_score, word)
    
    def evaluate_f1(self, predictions : List[Dict] | Dict, dataset = 'test', word = None, average='macro'):
        return self.evaluate(predictions, dataset, lambda truth, pred : f1_score(truth, pred, average=average), word)
    

            

            

# Format for each data example:
#{'text':text, 'start1': 'end1': label:'synset/babelsysnet'}
