import os
import pickle
import sys
import gc

import numpy as np

wiki2vec_file = r"/media/wentian/sdb1/caption_datasets/wikipedia2vec/enwiki_20180420_win10_500d.txt"

index = {}

n_dim = 500

with open(wiki2vec_file, 'r', encoding='utf-8') as f:
    _ = f.readline()
    for line in f:
        strs = line.strip().split()
        word = ' '.join(strs[:-n_dim])
        vector = np.array([float(i) for i in strs[-n_dim:]])
        index[word] = vector
        del strs

        if len(index) % 10000 == 0:
            print(len(index))
            gc.collect()

with open(r'/media/wentian/sdb1/caption_datasets/wikipedia2vec/enwiki_20180420_win10_500d.pkl', 'wb') as f:
    pickle.dump(index, f)

