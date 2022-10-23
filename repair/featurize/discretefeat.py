from functools import partial
from math import sqrt
from string import Template

import torch
import torch.nn.functional as F

from dataset import AuxTables
from .featurizer import Featurizer

template_ring_range_search = Template(
    '''
    SELECT _vid_, val_id, COUNT(1) AS distances
      FROM $init_table AS t1, $init_table as t2, $pos_values AS t3, $distance_matrix as t4
     WHERE t1._tid_ <> t2._tid_
       AND t1._tid_ = t3._tid_
       AND t3.attribute = \'$tobler_attr\'
       AND t3.rv_val::TEXT <> t2.$tobler_attr
       AND t1._tid_ = t4.tid_1
       AND t2._tid_ = t4.tid_2
       AND t4.distance >= $inner
       AND t4.distance < $outer
    GROUP BY _vid_, val_id;
    '''
)


def gen_feat_tensor(violations, total_vars, classes):
    tensor = torch.zeros(total_vars, classes, 1)
    if violations:
        for entry in violations:
            vid = int(entry[0])
            val_id = int(entry[1]) - 1
            feat_val = float(entry[2])
            tensor[vid][val_id][0] = feat_val
    return tensor


def get_equal_area_ring(r, n):
    r = float(r)
    return [sqrt((i + 1) * r * r / n) for i in range(n)]


class DiscreteFeaturizer(Featurizer):

    def specific_setup(self):
        self.name = "1000 constraints featurizer"
        self.tobler_attr = self.env["tobler_attr"]
        # self.distances = self.env["tobler_discrete_distances"]  disable explicit ring
        self.max_ring = self.env["tobler_max_ring"]
        self.ring_count = self.env["tobler_ring_count"]
        self.distances = get_equal_area_ring(self.max_ring, self.ring_count)

    def fill_query_template(self, inner, outer):
        return template_ring_range_search.substitute(init_table=self.ds.raw_data.name,
                                                     pos_values=AuxTables.pos_values.name,
                                                     distance_matrix=AuxTables.distance_matrix.name,
                                                     tobler_attr=self.tobler_attr,
                                                     inner=inner,
                                                     outer=outer)

    def generate_sql(self):
        queries = []
        inner_distance = 0
        for outer_distance in self.distances:
            query = self.fill_query_template(inner_distance, outer_distance)
            inner_distance = outer_distance
            queries.append(query)

        return queries

    def create_tensor(self):
        """
         This method creates a tensor which has shape
         (# of cells/rvs, max size of domain, # of features for this featurizer)

        :return: PyTorch Tensor
        """
        queries = self.generate_sql()
        results = self.ds.engine.execute_queries(queries)
        tensors = self._apply_func(partial(gen_feat_tensor, total_vars=self.total_vars, classes=self.classes), results)
        if not len(tensors):
            return None
        combined = torch.cat(tensors, 2)
        combined = F.normalize(combined, p=2, dim=1)
        return combined

    def feature_names(self):
        """
        Returns list of human-readable description/names for each feature
        this featurizer produces.
        """
        names = []
        inner_distance = 0
        for outer_distance in self.distances:
            name = f"discrete distance [{inner_distance}, {outer_distance})"
            names.append(name)
            inner_distance = outer_distance
        return names
