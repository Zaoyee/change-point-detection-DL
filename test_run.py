import train_test as tt 
import test_model as tm
import torch

loadpath = './data/processed/detailed/datamat2.csv'
lr = 1e-4
batch_size = 300
weight_decay = 0
num_epoches = 1000
foldID = 1

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
t_loss, t_acc, pred = tt.start_train(model, loadpath, foldID)
