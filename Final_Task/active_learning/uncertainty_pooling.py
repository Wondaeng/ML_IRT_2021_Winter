import os
from active_learning.uncertainty import uncertainty_score


def uncertainty_pooling(conf, pool, cfg):

    base_path = conf['train_dir'] + '/'
    scores_names = uncertainty_score(base_path, pool, cfg)[:conf['pool_size']]
    names = [i[1] for i in scores_names]

    return names



