import subprocess
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Union
from languagechange.usages import TargetUsage
from languagechange.corpora import LinebyLineCorpus
from LSCDetection.modules.utils_ import Space
from languagechange.models.representation.static import StaticModel
import os


class OrthogonalProcrustes():
    """
    A class to align word embeddings using the Orthogonal Procrustes method.

    This method aligns two embedding spaces by finding an optimal orthogonal transformation.
    """
    
    def __init__(self, savepath1:str, savepath2:str):
        """
        Initialize the class with paths to save the aligned embeddings.

        Args:
            savepath1 (str): Path to save the aligned version of the first model.
            savepath2 (str): Path to save the aligned version of the second model.
        """
        self.savepath1 = savepath1
        self.savepath2 = savepath2


    def align(self, model1:StaticModel, model2:StaticModel):
        """
        Perform orthogonal alignment between two embedding models using a subprocess.

        Args:
            model1 (StaticModel): The first static word embedding model to align.
            model2 (StaticModel): The second static word embedding model to align.
        """
        subprocess.run(["python3", "-m", "LSCDetection.alignment.map_embeddings", 
            "--normalize", "unit",
            "--init_identical",
            "--orthogonal",
            model1.matrix_path,
            model2.matrix_path,
            self.savepath1,
            self.savepath2])