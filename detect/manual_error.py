import logging

import pandas as pd

from .detector import Detector


class ManualError(Detector):
    """
    Manually load error
    """

    def __init__(self, fpath, name='ManualError'):
        super(ManualError, self).__init__(name)
        self.error_path = fpath

    def setup(self, dataset, env):
        pass

    def detect_noisy_cells(self):
        """
        detect_noisy_cells returns a pandas.DataFrame read from csv

        :return: pandas.DataFrame with columns:
            _tid_: entity ID
            attribute: attribute
        """
        logging.debug(f"ManualError: load error from {self.error_path}")
        errors_df = pd.read_csv(self.error_path)
        logging.debug(f"ManualError: load {len(errors_df)} errors")
        return errors_df
