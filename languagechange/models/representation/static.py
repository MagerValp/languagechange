import subprocess
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Union
from languagechange.usages import TargetUsage
from languagechange.corpora import LinebyLineCorpus
from LSCDetection.modules.utils_ import Space
import os
from collections import defaultdict
import logging
import time
from scipy.sparse import dok_matrix
from gensim.models.word2vec import PathLineSentences
from sklearn.utils.extmath import randomized_svd
from sklearn.random_projection import sparse_random_matrix
from scipy.sparse import csr_matrix
env = os.environ.copy()

class RepresentationModel(ABC):

    @abstractmethod
    def encode(self, *args, **kwargs):
        pass

# todo
class StaticModel(RepresentationModel, dict):

    def __init__(self, matrix_path=None, format='w2v'):
        self.space = None
        self.matrix_path = matrix_path
        self.format = format

    @abstractmethod
    def encode(self):
        pass

    @abstractmethod
    def load(self):
        self.space = Space(self.matrix_path, format=self.format)


    def __getitem__(self, k):
        if self.space == None:
            raise Exception('Space is not loaded')
        return self.space.matrix[self.space.row2id[k]]

    def matrix(self):
        if self.space == None:
            raise Exception('Space is not loaded')
        return self.space.matrix

    def row2word(self):
        if self.space == None:
            raise Exception('Space is not loaded')
        return self.space.id2row

class CountModel(StaticModel):

    def __init__(self, corpus:LinebyLineCorpus, window_size:int, savepath:str):
        super(CountModel,self).__init__()
        self.corpus = corpus
        self.window_size = window_size
        self.savepath = savepath
        self.format = 'npz'
        self.matrix_path = os.path.join(self.savepath)

    def encode(self):
        # Previously
        #subprocess.run(["python3", "-m", "LSCDetection.representations.count", self.corpus.path, self.savepath, str(self.window_size)])

        # Code below from LSCDetection:
        """
        Make count-based vector space from corpus.
        """
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
        logging.info(__file__.upper())
        start_time = time.time()

        # Build vocabulary
        logging.info("Building vocabulary")
        sentences = PathLineSentences(self.corpus.path)
        vocabulary = sorted(list(set([word for sentence in sentences for word in sentence if len(sentence)>1]))) # Skip one-word sentences to avoid zero-vectors
        w2i = {w: i for i, w in enumerate(vocabulary)}
        
        # Initialize co-occurrence matrix as dictionary
        cooc_mat = defaultdict(lambda: 0)

        # Get counts from corpus
        sentences = PathLineSentences(self.corpus.path)
        logging.info("Counting context words")
        for sentence in sentences:
            for i, word in enumerate(sentence):
                lowerWindowSize = max(i-self.window_size, 0)
                upperWindowSize = min(i+self.window_size, len(sentence))
                window = sentence[lowerWindowSize:i] + sentence[i+1:upperWindowSize+1]
                if len(window)==0: # Skip one-word sentences
                    continue
                windex = w2i[word]
                for contextWord in window:
                    cooc_mat[(windex,w2i[contextWord])] += 1

        
        # Convert dictionary to sparse matrix
        logging.info("Converting dictionary to matrix")
        cooc_mat_sparse = dok_matrix((len(vocabulary),len(vocabulary)), dtype=float)
        try:
            cooc_mat_sparse.update(cooc_mat)
        except NotImplementedError:
            cooc_mat_sparse._update(cooc_mat)

        outSpace = Space(matrix=cooc_mat_sparse, rows=vocabulary, columns=vocabulary)

        #if is_len:
        #    # L2-normalize vectors
        #    outSpace.l2_normalize()
            
        # Save the matrix
        outSpace.save(self.savepath)

        logging.info("--- %s seconds ---" % (time.time() - start_time))


class PPMI(CountModel):

    def __init__(self, count_model:CountModel, shifting_parameter:int, smoothing_parameter:int, savepath:str):
        super(PPMI,self).__init__(self,count_model.window_size, count_model.savepath)
        self.count_model = count_model
        self.shifting_parameter = shifting_parameter
        self.smoothing_parameter = smoothing_parameter
        self.savepath = savepath
        self.matrix_path = os.path.join(self.savepath)
        self.align_strategies = {'OP', 'SRV', 'WI'}

    def encode(self):
        # Previously
        #subprocess.run(["python3", "-m", "LSCDetection.representations.ppmi", self.count_model.matrix_path, self.savepath, str(self.shifting_parameter), str(self.smoothing_parameter)])

        # Code below from LSCDetection
        """
        Compute the smoothed and shifted PPMI matrix from a co-occurrence matrix. Smoothing is performed as described in

        Omer Levy, Yoav Goldberg, and Ido Dagan. 2015. Improving distributional similarity with lessons learned from word embeddings. Trans. ACL, 3.

        """

        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
        logging.info(__file__.upper())
        start_time = time.time()    

        # Load input matrix
        space = Space(self.count_model.matrix_path)   

        # Apply EPMI weighting
        space.epmi_weighting(self.smoothing_parameter)
        
        # Apply log weighting
        space.log_weighting()

        # Shift values
        space.shifting(self.shifting_parameter)

        # Eliminate negative counts
        space.eliminate_negative()

        # Eliminate zero counts
        space.eliminate_zeros()
            
        outSpace = Space(matrix=space.matrix, rows=space.rows, columns=space.columns)

        #if is_len:
            # L2-normalize vectors
        #    outSpace.l2_normalize()
            
        # Save the matrix
        outSpace.save(self.savepath)

        logging.info("--- %s seconds ---" % (time.time() - start_time))     

class SVD(StaticModel):

    def __init__(self, count_model:CountModel, dimensionality:int, gamma:float, savepath:str):
        super(SVD,self).__init__()
        self.count_model = count_model
        self.dimensionality = dimensionality
        self.gamma = gamma
        self.savepath = savepath
        self.matrix_path = os.path.join(self.savepath)
        self.format = 'w2v'
        self.align_strategies = {'OP', 'SRV', 'WI'}

    def encode(self):
        # Previously
        #subprocess.run(["python3", "-m", "LSCDetection.representations.svd", self.count_model.matrix_path, self.savepath, str(self.dimensionality), str(self.gamma)])

        # Code below from LSCDetection
        """
        Perform dimensionality reduction on a (normally PPMI) matrix by applying truncated SVD as described in

        Omer Levy, Yoav Goldberg, and Ido Dagan. 2015. Improving distributional similarity with lessons learned from word embeddings. Trans. ACL, 3.

        """

        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
        logging.info(__file__.upper())
        start_time = time.time()    

        # Load input matrix
        space = Space(self.count_model.matrix_path)   
        matrix = space.matrix
        
        # Get mappings between rows/columns and words
        rows = space.rows
        id2row = space.id2row
        id2column = space.id2column

        # Apply SVD
        u, s, v = randomized_svd(matrix, n_components=self.dimensionality, n_iter=5, transpose=False)

        # Weight matrix
        if self.gamma == 0.0:
            matrix_reduced = u
        elif self.gamma == 1.0:
            #matrix_reduced = np.dot(u, np.diag(s)) # This is equivalent to the below formula (because s is a flattened diagonal matrix)
            matrix_reduced = s * u
        else:
            #matrix_ = np.dot(u, np.power(np.diag(s), gamma)) # This is equivalent to the below formula
            matrix_reduced = np.power(s, self.gamma) * u
        
        outSpace = Space(matrix=matrix_reduced, rows=rows, columns=[])

        #if is_len:
            # L2-normalize vectors
        #    outSpace.l2_normalize()
            
        # Save the matrix
        outSpace.save(self.savepath, format='w2v')

        logging.info("--- %s seconds ---" % (time.time() - start_time))


# todo: add corpus
class RandomIndexing(StaticModel):

    def __init__(self):
        super(RandomIndexing,self).__init__()
        self.align_strategies = {'OP', 'SRV', 'WI'}
        pass

    def encode(self):
        # Previously
        #subprocess.run(["python3", "-m", "LSCDetection.representations.ri", corpus.path, self.savepath, self.window_size])

        # Code below from LSCDetection
        """
        Create low-dimensional vector space by sparse random indexing from co-occurrence matrix.
        """        
        
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
        logging.info(__file__.upper())
        start_time = time.time()    
        
        # Load input matrix
        countSpace = Space(corpus.path)   # todo: corpus needs to reference something here
        countMatrix = countSpace.matrix
        rows = countSpace.rows
        columns = countSpace.columns
        
        # Generate random vectors
        randomMatrix = csr_matrix(sparse_random_matrix(self.window_size,len(columns)).toarray().T)

        logging.info("Multiplying matrices")
        reducedMatrix = np.dot(countMatrix,randomMatrix)    
        outSpace = Space(matrix=reducedMatrix, rows=rows, columns=[])
        
        #if is_len:
            # L2-normalize vectors
        #    outSpace.l2_normalize()

        # Save the matrix
        outSpace.save(self.savepath, format='w2v')

        logging.info("--- %s seconds ---" % (time.time() - start_time))  