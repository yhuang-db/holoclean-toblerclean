import pandas as pd

from dataset import AuxTables
from .detector import Detector
from utils import NULL_REPR


class ToblerDCDetector(Detector):

    def __init__(self, name='ToblerDCDetector'):
        super(ToblerDCDetector, self).__init__(name)

    def setup(self, dataset, env):
        self.ds = dataset
        self.tobler_attr = env['tobler_attr']

    def detect_noisy_cells(self):
        """
        Returns a pandas.DataFrame containing all cells that
        shows confliction to Tobler's Law in self.dataset.

        :return: pandas.DataFrame with columns:
            _tid_: entity ID
            attribute: tobler_attr

        Pair of cells conflicting Tobler's Law:
        Two cells within ToblerDistance has different value
        """
        sql = f'''
        SELECT DISTINCT tid
        FROM (
          SELECT tid_1 AS tid
          FROM {AuxTables.distance_matrix.name}
          WHERE val_1 <> {NULL_REPR}
            AND val_2 <> {NULL_REPR}
            AND val_1 <> val_2
            AND tid_1 <> tid_2
          UNION
          SELECT tid_2 AS tid
          FROM {AuxTables.distance_matrix.name}
          WHERE val_1 <> {NULL_REPR}
            AND val_2 <> {NULL_REPR}
            AND val_1 <> val_2
            AND tid_1 <> tid_2) t
        ORDER BY tid
        '''
        result = self.ds.engine.execute_query(sql)

        error_dict = {'_tid_': [i[0] for i in result]}
        df_error = pd.DataFrame(error_dict)
        df_error['attribute'] = self.tobler_attr
        return df_error
