from itertools import product
import pandas as pd

def expand_grid(dictionary):
   return pd.DataFrame([row for row in product(*dictionary.values())],
                       columns=dictionary.keys())

dictionary = {'testID': [x for x in range(1, 7)],
              'weight_decay': [0],
              'num_epoches': [1000],
              'lr': [1e-4],
              'batch_size': [300],
              'modelName': ['resnet', 'resnet2',
                            '1dcnnMaxpool3conv100x3channelK11_10_9Lin500_256_124'],
              'tp': ['systemic', 'detailed'],
              }

df = expand_grid(dictionary)

for i in range(df.shape[0]):
    saveto = './params/pars%d.csv' % (i)
    pd.DataFrame(df.iloc[i,:]).to_csv(saveto, index=False)