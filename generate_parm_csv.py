from itertools import product
import pandas as pd

def expand_grid(dictionary):
   return pd.DataFrame([row for row in product(*dictionary.values())],
                       columns=dictionary.keys())

dictionary = {'testID': [x for x in range(7)],
              'weight_decay': [0, 0.0001, 0.001],
              'num_epoches': [300],
              'lr': [1e-4],
              'batch_size': [300]}

expand_grid(dictionary)