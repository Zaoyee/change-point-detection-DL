import train_test as tt
import test_model as tm
import torch
import os
import sys

# possible useful inputs specified
setupdic = {
    'loadpath': './data/processed/detailed/datamat2.csv',
    'modelName': 'resnet',
    'outname' : 'pred.csv',
    'outdir' : './savedata/sysmetic/model/',
    'lr' : 1e-4,
    'batch_size' : 300,
    'weight_decay' : 0,
    'num_epoches' : 1,
    'foldID' : int(sys.argv[1]),
}

## setup the dict path to save the file.
lr = setupdic.get('lr')
batch_size = setupdic.get('batch_size')
weight_decay = setupdic.get('weight_decay')
num_epoches = setupdic.get('num_epoches')
loadpath = setupdic.get('loadpath')
foldID = setupdic.get('foldID')
outdir = setupdic.get('outdir')
modelName = setupdic.get('modelName')
outname = setupdic.get('outname')

pathname = os.path.join(outdir, modelName)
pathname = os.path.join(pathname, 'lr'+str(lr)+'batchsize'+ \
    str(batch_size)+'decay'+str(weight_decay)\
     + 'numepoches'+str(num_epoches))
pathname = os.path.join(pathname, str(foldID)) 
if not os.path.exists(pathname):
    os.makedirs(pathname)
pathname = os.path.join(pathname, outname) 

# define model param
if torch.cuda.is_available():
    model = tm.resnet(in_channel=1, num_classes=1, lr=lr, batch_size=batch_size,
                      weight_decay=weight_decay, num_epoches=num_epoches).cuda()
    print('GPU yeah!!!!')
else:
    model = tm.resnet(in_channel=1, num_classes=1, lr=lr, batch_size=batch_size,
                      weight_decay=weight_decay, num_epoches=num_epoches)

# start do all the training
# the return is the
t_loss, t_acc, pred, pred_acc = tt.start_train(model, loadpath, foldID)
pred.to_csv(pathname, na_rep="NULL")
