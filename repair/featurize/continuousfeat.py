from string import Template

import torch
import torch.nn.functional as F

from dataset import AuxTables
from .featurizer import Featurizer

template_from_distance_matrix = Template(
    '''
    SELECT _vid_, val_id, array_agg(t4.weight) AS weights
      FROM $init_table AS t1, $init_table as t2, $pos_values AS t3, $distance_matrix as t4
     WHERE t1._tid_ <> t2._tid_
       AND t1._tid_ = t3._tid_
       AND t3.attribute = \'$tobler_attr\'
       AND t3.rv_val::TEXT <> t2.$tobler_attr
       AND t1._tid_ = t4.tid_1
       AND t2._tid_ = t4.tid_2
    GROUP BY _vid_, val_id;
    '''
)


# def cal_weighted_violation(distance_list):
#     """
#     weight(d) = exp(-d) is WRONG !!!
#     weight(d) = exp(-d/1000)
#
#     Alternative:
#     weight(d) = 2ND / (d + ND)
#     ND: normalized distance
#     weight(0) = 2
#     weight(ND) = 1
#     weight(+inf) = 0
#
#     distance_weighted_count = np.multiply(np.reciprocal(na + norm_dist), 2 * norm_dist)
#     return distance_weighted_count.sum()
#     """
#     na = np.array(distance_list)
#     na = na / 1000
#     exp_list = np.exp(np.negative(na))
#     return exp_list.sum()


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
        self.name = "ContinuousViolationFeaturizer"
        self.tobler_attr = self.env["tobler_attr"]
        if 'tobler_distance' in self.env:
            self.tobler_distance = self.env['tobler_distance']
        elif 'tobler_knn' in self.env:
            self.tobler_knn = self.env['tobler_knn']

    def create_tensor(self):
        """
        This method creates a tensor which has shape
        (# of cells/rvs, max size of domain, # of features for this featurizer)

        :return: PyTorch Tensor
        """

        query = template_from_distance_matrix.substitute(
            init_table=self.ds.raw_data.name,
            pos_values=AuxTables.pos_values.name,
            distance_matrix=AuxTables.distance_matrix.name,
            tobler_attr=self.tobler_attr
        )

        result = self.ds.engine.execute_query(query)
        weighted_violations = [[i[0], i[1], sum(i[2])] for i in result]
        # weighted_violations = [[i[0], i[1], cal_weighted_violation(i[2])] for i in result]
        # weighted_violations = [[i[0], i[1], cal_weighted_violation(i[2], self.tobler_normalized_distance)] for i in result]
        tensor = gen_feat_tensor(weighted_violations, self.total_vars, self.classes)
        tensor = F.normalize(tensor, p=2, dim=1)
        return tensor

    def feature_names(self):
        if 'tobler_distance' in self.env:
            return [f'Continuous Feature by range: {self.tobler_distance}']
        elif 'tobler_knn' in self.env:
            return [f'Continuous Feature by knn: {self.tobler_knn}']
        # return [f'Continuous Feature: {self.tobler_max_distance}:{self.tobler_normalized_distance}']
