import torch

from dataset import AuxTables
from .featurizer import Featurizer


class DiscFreqFeaturizer(Featurizer):
    def specific_setup(self):
        self.name = 'DiscreteFreqFeaturizer'
        self.tobler_attr = self.env["tobler_attr"]
        self.distances = self.env["tobler_discrete_distances"]

    def create_tensor(self):

        pass

    def feature_names(self):
        return self.name
