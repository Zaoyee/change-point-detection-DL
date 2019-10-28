import train_test as tt
import test_model as tm
import torch
import os
import sys

# possible useful inputs specified
setupdic = {
    'filepath': './data/processed/systemic/datamat.csv',
    'modelName': '1dcnnMaxpool3conv100x3channelK11_10_9Lin500_256_124',
    'outname' : 'pred.csv',
    'outdir' : './data/savedata/systemic/model',
    'lr' : 1e-4,
    'batch_size' : 300,
    'weight_decay' : 0.0001,
    'num_epoches' : 300,
    # 'testID' : 2,
    'testID' : int(sys.argv[1]),
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
# define model param
if torch.cuda.is_available():
    # model = tm.resnet2(in_channel=1, num_classes=1, **setupdic).cuda()
    model = tm.simple1dcnn(**setupdic).cuda()
    print('GPU yeah!!!!')
else:
    # model = tm.resnet2(in_channel=1, num_classes=1, **setupdic)
    model = tm.simple1dcnn(**setupdic)

# start do all the training
# the return is the
t_loss, t_acc, pred, pred_acc = tt.start_train(model, **setupdic)
pred.to_csv(pred_path, na_rep="NULL")
pred_acc.to_csv(acc_path, na_rep="NULL")
