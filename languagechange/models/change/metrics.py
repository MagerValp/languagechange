from scipy.spatial.distance import cdist, cosine
from languagechange.models.meaning.clustering import Clustering, APosterioriaffinityPropagation
from languagechange.models.change.timeseries import TimeSeries
import numpy as np
from collections import Counter
from scipy.spatial.distance import jensenshannon
from typing import List, Union

class ChangeModel():

    def __init__(self):
        pass


class BinaryChange(ChangeModel):

    def __init__(self):
        pass

    def predict(self):
        pass


class GradedChange(ChangeModel):

    def __init__(self):
        pass

    def compute_scores(vectors_list):
        pass


class Threshold(BinaryChange):

    def __init__(self):
        pass

    def set_threshold(self, threshold):
        self.threshold = threshold


class AutomaticThrehold(Threshold):

    def __init__(self):
        pass

    def compute_threshold(self, scores, func = lambda x: np.mean(x)):
        self.threshold = func(scores)


class OptimalThrehold(Threshold):

    def __init__(self):
        pass

    def compute_threshold(self, scores, vrange=np.arange(0.,1.), evaluator=None):
        best_score = None
        best_threshold = None

        for v in vrange:
            labels = np.array(scores < v, dtype=int)
            score = evaluator(labels)
            if score > best_score or best_score == None:
                best_score = score
                best_threshold = v

        self.threshold = best_threshold


class APD(GradedChange):

    def __init__(self):
        pass

    def compute_scores(self, embeddings1, embeddings2, metric='cosine'):

        return np.mean(cdist(embeddings1, embeddings2, metric=metric))


class PRT(GradedChange):

    def __init__(self):
        pass

    def compute_scores(self, embeddings1, embeddings2, metric='cosine'):

        return cosine(embeddings1.mean(axis=0), embeddings2.mean(axis=0))


class PJSD(GradedChange):

    def __init__(self):
        pass

    def compute_scores(self, embeddings1, embeddings2, clustering_algorithm, metric='cosine'):
        clustering = Clustering(clustering_algorithm)
        clustering.get_cluster_results(np.concatenate((embeddings1,embeddings2),axis=0))
        labels1 = clustering.labels[:len(embeddings1)]
        labels2 = clustering.labels[len(embeddings1):]
        labels = set(clustering.labels)
        count1 = Counter(labels1)
        count2 = Counter(labels2)
        p,q = [], []
        for l in labels:
            if l in count1:
                p.append(count1[l]/len(embeddings1))
            else:
                p.append(0.)
            if l in count2:
                q.append(count2[l]/len(embeddings2))
            else:
                q.append(0.)

        return jensenshannon(p, q)


class WiDiD:
    """
        A class that implements WiDiD (https://github.com/FrancescoPeriti/WiDiD).
    """
    def __init__(self, affinity: str = 'cosine',
                 damping: float = 0.9,
                 max_iter: int = 200,
                 convergence_iter: int = 15,
                 copy: bool = True,
                 preference: bool = None,
                 verbose: bool = False,
                 random_state: int = 42,
                 th_gamma: int = 0,
                 pack: str = 'mean',
                 singleton: str = 'one',
                 metric: str = 'cosine'):
        self.app = Clustering(APosterioriaffinityPropagation(affinity=affinity, damping=damping, max_iter=max_iter, convergence_iter=convergence_iter, copy=copy, preference=preference, verbose=verbose, random_state=random_state, th_gamma=th_gamma, pack=pack, singleton=singleton))
        self.metric = metric

    
    def compute_scores(self, embs_list : List[np.array], timeseries_type='consecutive', k=1, change_metric='apd', time_labels: Union[np.array, List] = None):
        """
            Performs a-posteriori affinity propagation (APP) clustering and computes the semantic change as the APD (or another metric) between the prototype embeddings in clusters of different time periods.
            
            Args: 
                embs_list ([np.array]): a list of embeddings for a target word, where each element is embeddings of one time period.
                timeseries_type (str): the type of timeseries (see usage in languagechange.models.change.timeseries).
                k (int): the window size, if moving average (see usage in languagechange.models.change.timeseries).
                change_metric (str): the change metric (e.g. 'apd') to use (see usage in languagechange.models.change.timeseries).
                change_metric (str): the change metric (e.g. 'apd') to use (see usage in languagechange.models.change.timeseries).
                time_labels (np.array|list): labels for the x axis of the timeseries (see usage in languagechange.models.change.timeseries).

            Returns:
                labels ([np.array]): the labels for each embedding in each time period.
                prot_embs ([np.array]): a list of matrices encoding the prototype (average) embedding of each cluster in each time period.
                change_scores (TimeSeries): a timeseries (languagechange.models.change.timeseries.TimeSeries) containing the degree of change between the embeddings in different time periods.
        """
        self.app.get_cluster_results(embs_list)
        all_labels = self.app.labels
        labels = []

        i = 0
        for embs in embs_list:
            labels.append(all_labels[i:i+embs.shape[0]])
            i += embs.shape[0]

        # Compute the centroids of each cluster (the prototype embeddings)
        prot_embs = []
        for i, embs in enumerate(embs_list):
            prot_embs.append(np.array([embs[labels[i] == label].mean(axis=0) for label in np.unique(labels[i])]))

        # Get the change scores between prototype embeddings
        change_scores = TimeSeries(prot_embs, change_metric=change_metric, timeseries_type=timeseries_type, k=k, time_labels=time_labels)

        return labels, prot_embs, change_scores
