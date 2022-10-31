import duckdb
import pandas as pd

from dataset import AuxTables
from .detector import Detector


class ToblerDetector(Detector):
    def __init__(self, name='ToblerDetector'):
        super(ToblerDetector, self).__init__(name)

    def setup(self, dataset, env):
        self.ds = dataset
        self.tobler_attr = env['tobler_attr']
        self.tobler_distance = env['tobler_distance']
        self.tobler_threshold = env['tobler_threshold']

    def detect_noisy_cells(self):
        """
        Returns a pandas.DataFrame containing all cells that
        violate Tobler's Law contained in self.dataset.

        :return: pandas.DataFrame with columns:
            _tid_: entity ID
            attribute: tobler_attr

        Cell violating Tobler's Law: if its significant Tobler value
        is not its origin value
        """
        sql = f'''
        SELECT tid_1, val_1, val_2, weight
        FROM {AuxTables.distance_matrix.name}
        WHERE val_1 <> '_nan_'
        '''
        result = self.ds.engine.execute_query(sql)
        df = pd.DataFrame.from_records(data=result, columns=['tid_1', 'val_1', 'val_2', 'weight'])

        tid_val_weight = duckdb.query(
            f'''
            SELECT 
                t1.tid_1,
                t1.val_1,
                t1.val_2, 
                SUM(t1.weight / t2.total_weight) AS val_2_weight
            FROM 
                df t1,
                (SELECT tid_1, SUM(weight) AS total_weight FROM df GROUP BY tid_1) t2
            WHERE t1.tid_1 = t2.tid_1
              AND t1.val_1 <> t1.val_2
            GROUP BY t1.tid_1, t1.val_1, t1.val_2
            HAVING SUM(t1.weight / t2.total_weight) > {self.tobler_threshold}
            '''
        ).to_df()

        error_dict = {'_tid_': tid_val_weight['tid_1']}
        df_error = pd.DataFrame(error_dict)
        df_error['attribute'] = self.tobler_attr
        return df_error
