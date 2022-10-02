from string import Template

import numpy as np
import torch
import torch.nn.functional as F

from dataset import AuxTables
from .featurizer import Featurizer

template_range_search = Template(
    '''
    SELECT _vid_, val_id, array_agg(t4.distance) AS distances
      FROM $init_table AS t1, $init_table as t2, $pos_values AS t3, $distance_matrix as t4
     WHERE t1._tid_ <> t2._tid_
       AND t1._tid_ = t3._tid_
       AND t3.attribute = \'$tobler_attr\'
       AND t3.rv_val::TEXT <> t2.$tobler_attr
       AND t1._tid_ = t4.tid_1
       AND t2._tid_ = t4.tid_2
       AND t4.distance < $radius
    GROUP BY _vid_, val_id;
    '''
)


def cal_weighted_violation(distance_list):
    """
    weight(d) = exp(-d)
    """
    na = np.array(distance_list)
    exp_list = np.exp(np.negative(na))
    return exp_list.sum()


def gen_feat_tensor(violations, total_vars, classes):
    tensor = torch.zeros(total_vars, classes, 1)
    if violations:
        for entry in violations:
            vid = int(entry[0])
            val_id = int(entry[1]) - 1
            feat_val = float(entry[2])
            tensor[vid][val_id][0] = feat_val
    return tensor


class ContinuousFeaturizer(Featurizer):

    def specific_setup(self):
        self.name = "continuous distance feature"
        self.tobler_max_distance = self.env["tobler_continuous_distance"]
        self.tobler_attr = self.env["tobler_attr"]
        self.compute_distance_matrix()

    def create_tensor(self):
        """
        This method creates a tensor which has shape
        (# of cells/rvs, max size of domain, # of features for this featurizer)

        :return: PyTorch Tensor
        """
        query = template_range_search.substitute(init_table=self.ds.raw_data.name,
                                                 pos_values=AuxTables.pos_values.name,
                                                 distance_matrix=AuxTables.distance_matrix.name,
                                                 tobler_attr=self.tobler_attr,
                                                 radius=self.tobler_max_distance)
        result = self.ds.engine.execute_query(query)
        weighted_violations = [[i[0], i[1], cal_weighted_violation(i[2])] for i in result]
        tensor = gen_feat_tensor(weighted_violations, self.total_vars, self.classes)
        tensor = F.normalize(tensor, p=2, dim=1)
        return tensor

    def feature_names(self):
        return f"continuous distance {self.tobler_max_distance}"

    def compute_distance_matrix(self):
        if len(self.env["tobler_location_attr"]) == 2:
            x, y = self.env["tobler_location_attr"]
            sql_create_geom_table = f"""
            SELECT _tid_, ST_MakePoint({x}::real, {y}::real) AS _geom_
            FROM {self.ds.raw_data.name}
            """
            self.ds.engine.create_db_table_from_query(name=AuxTables.geom.name, query=sql_create_geom_table)
            spatial_index_name = f"{AuxTables.geom.name}_idx"
            self.ds.engine.create_spatial_db_index(name=spatial_index_name, table=AuxTables.geom.name, spatial_attr="_geom_")
            self.ds.engine.cluster_db_using_index(table_name=AuxTables.geom.name, index_name=spatial_index_name)

            sql_create_distance_matrix = f"""
            SELECT t1._tid_ AS tid_1, t2._tid_ AS tid_2, ST_Distance(t1._geom_, t2._geom_) AS distance
            FROM {AuxTables.geom.name} t1, {AuxTables.geom.name} t2
            WHERE t1._tid_ <> t2._tid_
              AND ST_DWithin(t1._geom_, t2._geom_, {self.tobler_max_distance})
            """
            self.ds.engine.create_db_table_from_query(name=AuxTables.distance_matrix.name, query=sql_create_distance_matrix)
            distance_matrix_index_name = f"{AuxTables.distance_matrix.name}_idx"
            self.ds.engine.create_db_index(name=distance_matrix_index_name, table=AuxTables.distance_matrix.name, attr_list=["tid_1", "distance"])
            self.ds.engine.cluster_db_using_index(table_name=AuxTables.distance_matrix.name, index_name=distance_matrix_index_name)
        else:
            raise Exception("tobler_location_attr unsupported. ")
