import logging
from string import Template

import numpy as np
import pandas as pd
import torch

from dataset import AuxTables
from .featurizer import Featurizer

template_range_search = Template(
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
       AND t2.distance < $radius
    '''
)


class ContFreqFeaturizer(Featurizer):
    def specific_setup(self):
        self.name = 'ContinuousFreqFeaturizer'
        self.attrs_number = len(self.ds.attr_to_idx)
        self.tobler_attr = self.env["tobler_attr"]
        self.distance = self.env['tobler_continuous_distance']
        logging.debug("Start computing local frequency...")
        self.freq_precomputation = self.freq_precompute()
        self.vid_set = set(self.freq_precomputation['vid'])
        self.ds.generate_aux_table(aux_table=AuxTables.weight_precomputation, df=self.freq_precomputation, store=True)
        logging.debug("Done computing local frequency.")

    def create_tensor(self):
        query = f'SELECT _vid_, attribute, domain, init_index FROM {AuxTables.cell_domain.name} ORDER BY _vid_'
        results = self.ds.engine.execute_query(query)
        tensors = [self.gen_feat_tensor(res, self.classes) for res in results]
        combined = torch.cat(tensors)
        return combined

    def feature_names(self):
        return [f'Continuous Freq {i}' for i in self.ds.get_attributes()]

    def freq_precompute(self):
        """
        for each cell of tobler_attr in domain, find its neighbors within tobler_continuous_distance
        """
        query = template_range_search.substitute(
            tobler_attr=self.tobler_attr,
            domain_table=AuxTables.cell_domain.name,
            distance_table=AuxTables.distance_matrix.name,
            raw_table=self.ds.raw_data.name,
            radius=self.distance
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

    def gen_feat_tensor(self, input, classes):
        tensor = torch.zeros(1, classes, self.attrs_number)
        vid = int(input[0])
        print(f'vid: {vid}')
        attribute = input[1]
        domain = input[2].split('|||')
        init_index = input[3]
        if attribute != self.tobler_attr:
            return tensor
        else:
            attr_idx = self.ds.attr_to_idx[attribute]
            domain_value_set = set(self.freq_precomputation.loc[self.freq_precomputation['vid'] == vid, 'domain_value'])
            for idx, val in enumerate(domain):
                # if vid is not in precomputation,
                # this means that vid does not have neighbor within search range,
                # therefore, set init_value to 1 and leave rest to 0
                if vid in self.vid_set:
                    # if val is not in precomputation,
                    # this means that val is a random generated one
                    # therefore, set its freq to 0
                    if val in domain_value_set:
                        freq = self.freq_precomputation.loc[
                            (self.freq_precomputation['vid'] == vid) &
                            (self.freq_precomputation['domain_value'] == val),
                            'domain_value_weight'].values[0]
                    else:
                        freq = 0

                    tensor[0][idx][attr_idx] = freq
                else:
                    tensor[0][init_index][attr_idx] = 1

        return tensor
