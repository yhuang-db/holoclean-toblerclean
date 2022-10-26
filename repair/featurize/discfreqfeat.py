import logging
from string import Template

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from dataset import AuxTables
from .featurizer import Featurizer

template_ring_search = Template(
    '''
    SELECT 
        t1._vid_,
        t2.distance,
        t3.$tobler_attr
      FROM 
        $domain_table AS t1, 
        $distance_table AS t2, 
        $raw_table AS t3 
     WHERE t1.attribute = \'$tobler_attr\'
       AND t2.tid_1 = t1._tid_
       AND t2.tid_2 = t3._tid_
       AND t2.distance >= $inner
       AND t2.distance <  $outer
    '''
)


class DiscFreqFeaturizer(Featurizer):
    def specific_setup(self):
        self.name = 'DiscreteFreqFeaturizer'
        self.attrs_number = len(self.ds.attr_to_idx)
        self.tobler_attr = self.env["tobler_attr"]
        self.distances = self.env["tobler_discrete_distances"]

        logging.debug("Start computing local frequency...")
        self.precomputation_dict = {}
        inner_distance = 0
        for outer_distance in tqdm(self.distances):
            df_precomputation = self.freq_precompute(inner_distance, outer_distance)
            self.precomputation_dict[outer_distance] = df_precomputation
            inner_distance = outer_distance
        logging.debug("Done computing local frequency.")

    def create_tensor(self):
        query = f'SELECT _vid_, attribute, domain, init_index FROM {AuxTables.cell_domain.name} ORDER BY _vid_'
        results = self.ds.engine.execute_query(query)
        ring_tensors = []
        for outer_distance in self.distances:
            df_precomputation = self.precomputation_dict[outer_distance]
            vid_set = set(df_precomputation['vid'])
            df_precomputation = df_precomputation.set_index(['vid', 'domain_value'], drop=False)
            tensors = [self.gen_feat_tensor(res, self.classes, df_precomputation, vid_set) for res in tqdm(results)]
            ring_tensor = torch.cat(tensors)
            ring_tensors.append(ring_tensor)
        combined = torch.cat(ring_tensors, 2)
        return combined

    def feature_names(self):
        names = []
        inner_distance = 0
        for outer_distance in self.distances:
            name = f'DiscreteFreq [{inner_distance}, {outer_distance})'
            names.append(name)
            inner_distance = outer_distance
        return names

    def freq_precompute(self, inner_distance, outer_distance):
        """
        for each cell of tobler_attr in domain, find its neighbors within tobler_continuous_distance
        """
        query = template_ring_search.substitute(
            tobler_attr=self.tobler_attr,
            domain_table=AuxTables.cell_domain.name,
            distance_table=AuxTables.distance_matrix.name,
            raw_table=self.ds.raw_data.name,
            inner=inner_distance,
            outer=outer_distance
        )
        results = self.ds.engine.execute_query(query)
        df = pd.DataFrame.from_records(data=results, columns=['vid', 'distance', 'domain_value'])
        df['weight'] = np.exp(-df['distance'] / 1000)  # weight(d) = exp(-d/1000)

        vid_total_weight = df.groupby('vid').apply(lambda d: d['weight'].sum())  # group by vid, sum weight
        vid_total_weight = vid_total_weight.rename('total_weight')
        vid_total_weight = vid_total_weight.reset_index()

        df = df.merge(vid_total_weight, on='vid')
        df['normal_weight'] = df['weight'] / df['total_weight']

        vid_weight = df.groupby(['vid', 'domain_value']).apply(lambda d: d['normal_weight'].sum())
        vid_weight = vid_weight.rename('domain_value_weight')
        vid_weight = vid_weight.reset_index()
        return vid_weight

    def gen_feat_tensor(self, input, classes, df_precomputation, vid_set):
        tensor = torch.zeros(1, classes, self.attrs_number)
        vid = int(input[0])
        attribute = input[1]
        domain = input[2].split('|||')
        init_index = input[3]
        if attribute != self.tobler_attr:
            return tensor
        else:
            attr_idx = self.ds.attr_to_idx[attribute]
            for idx, val in enumerate(domain):
                if vid in vid_set:
                    domain_value_set = set(df_precomputation.loc[(vid,), 'domain_value'])
                    if val in domain_value_set:
                        freq = df_precomputation.at[(vid, val), 'domain_value_weight']
                    else:
                        freq = 0

                    tensor[0][idx][attr_idx] = freq
                else:
                    tensor[0][init_index][attr_idx] = 1
        return tensor
