from ..util.logging import init_logging
from .SubtraceClustering import SubtraceClustering

from sklearn.cluster import AgglomerativeClustering
import pandas as pd
import numpy as np


class HierarchicalSC(AgglomerativeClustering, SubtraceClustering):

    def __init__(self, distance_matrix, **kwargs):
        super().__init__(**kwargs)
        self.logger = init_logging(__name__, out_file="HierarchicalSC.log")

        self.distance_matrix = distance_matrix
        self.sample_subtrace_distances = None

    def fit(self, **kwargs):
        super().fit(**kwargs)

    def generate_sample_subtraces_distances(self, subtraces: list[list[str]]):
        # TODO
        # samples should be all activities and subtraces pulled from the dataframe
        # check if there is even a difference between the unique activity subtrace clustering and the full sample clustering
        # there should be, but is the shape/dendrogramm different?
        self.sample_subtrace_distances = np.zeros((len(subtraces), len(subtraces)))

        # samples should be the subtraces

        for i, subtrace in enumerate(subtraces):
            distances = [
                self.distance_matrix[subtrace][other_subtrace]
                for other_subtrace in subtraces
            ]
            self.sample_subtrace_distances[i] = distances

    def transform(self):
        print("execute HSC")
        raise NotImplementedError
