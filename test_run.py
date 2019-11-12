import train_test as tt
import test_model as tm
import torch
import os
import pandas as pd
import sys

# possible useful inputs specified
# 1dcnnMaxpool3conv100x3channelK11_10_9Lin500_256_124   sys.argv[1]


readFrom = './params/pars' + sys.argv[1] + '.csv'
df = pd.read_csv('./params/pars0.csv')
md_Nm = df.iloc[5].values[0]

setupdic = {
    'filepath': './data/processed/cv2/%s/datamat.csv' % (df.iloc[6].values[0]),
    'modelName': md_Nm,
    'outname' : 'pred.csv',
    'outdir' : './data/savedata/cv2/%s/model' % (df.iloc[6].values[0]),
    'lr' : float(df.iloc[3].values[0]),
    'batch_size' : int(df.iloc[4].values[0]),
    'weight_decay' : float(df.iloc[1].values[0]),
    'num_epoches' : int(df.iloc[2].values[0]),
    # 'testID' : 2,
    'testID' : int(df.iloc[1].values[0]),
}

pathname = '{}/{}/lr{}bs{}decay{}epoch{}/{}'\
    .format(setupdic['outdir'],
            setupdic['modelName'],
            setupdic['lr'],
            setupdic['batch_size'],
            setupdic['weight_decay'],
            setupdic['num_epoches'],
            setupdic['testID'])
## setup the dict path to save the file.
if not os.path.exists(pathname):
    os.makedirs(pathname)
pred_path = os.path.join(pathname, setupdic['outname'])
acc_path = os.path.join(pathname, 'acc.csv')
opim_epo_path = os.path.join(pathname, 'opt_n.csv')
table_path = os.path.join(pathname, 'table.csv')

# define model param
if torch.cuda.is_available():
    if md_Nm == 'resnet':
        model = tm.resnet(in_channel=1, num_classes=1, **setupdic).cuda()
    elif md_Nm == 'resnet2':
        model = tm.resnet2(in_channel=1, num_classes=1, **setupdic).cuda()
    elif md_Nm == '1dcnnMaxpool3conv100x3channelK11_10_9Lin500_256_124':
        model = tm.simple1dcnn(**setupdic).cuda()
    print('GPU yeah!!!!')
else:
    if md_Nm == 'resnet':
        model = tm.resnet(in_channel=1, num_classes=1, **setupdic)
    elif md_Nm == 'resnet2':
        model = tm.resnet2(in_channel=1, num_classes=1, **setupdic)
    elif md_Nm == '1dcnnMaxpool3conv100x3channelK11_10_9Lin500_256_124':
        model = tm.simple1dcnn(**setupdic)
    # model = tm.resnet2(in_channel=1, num_classes=1, **setupdic)
    #model = tm.simple1dcnn(**setupdic)

# start do all the training
# the return is the
t_loss, t_acc, pred, pred_acc, optim_epo, table = tt.start_train(model, **setupdic)
pred.to_csv(pred_path, na_rep="NULL")
pred_acc.to_csv(acc_path, na_rep="NULL")
optim_epo.to_csv(opim_epo_path, na_rep="NULL")
table.to_csv(table_path, na_rep="NULL")
