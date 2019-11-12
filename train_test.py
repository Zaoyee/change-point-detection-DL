# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'# %%

import numpy as np
import torch
from torch import nn
from torch.autograd import Variable

# from torch.utils.tensorboard import SummaryWriter
import torch.utils.data as Data
from torch.utils.data import DataLoader
from torch import optim
import pandas as pd


def divideData(data_in, testFlg, test_id, valid_id):
    if testFlg:
        data_temp = data_in[data_in['foldID'] == test_id]
        test_data = data_temp.iloc[:,1:-6]
        test_label = data_temp.iloc[:,-4]
        test_out = data_temp.iloc[:,-3:-1]

        train_data_w = data_in[data_in['foldID'] != test_id]
        train_data = train_data_w.iloc[:, 1:-6]
        train_label = train_data_w.iloc[:, -4]
        train_out = train_data_w.iloc[:, -3:-1]

        return(test_data, train_data, test_label, train_label, test_out, train_out)
    else:
        data_temp = data_in[data_in['foldID'] != test_id]
        valid_data_w = data_temp[data_temp['foldID'] == valid_id]
        valid_data = valid_data_w.iloc[:,1:-6]
        valid_label = valid_data_w.iloc[:,-4]
        valid_out = valid_data_w.iloc[:,-3:-1]

        train_data_w = data_temp[data_temp['foldID'] != valid_id]
        train_data = train_data_w.iloc[:,1:-6]
        train_label = train_data_w.iloc[:,-4]
        train_out = train_data_w.iloc[:,-3:-1]
        return(train_data, valid_data, train_label, valid_label, train_out, valid_out)


def _loadData(inputs, target, batch_size):
    tensor_set = Data.TensorDataset(torch.from_numpy(inputs.values).unsqueeze(1).double(),
     torch.from_numpy(target.values).unsqueeze(1).double())
    tensor_loader = Data.DataLoader(tensor_set, batch_size=batch_size, shuffle=True)
    return(tensor_set, tensor_loader)


def modelsetup(model, lr, weight_decay):
    criterion = nn.MSELoss()
    optimer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    return(criterion, optimer)


def get_acc(pred, target_out):
    correct_1 = np.sum((pred[target_out['max.log.lambda'] == np.inf]) >=
                       (target_out[target_out['max.log.lambda'] == np.inf]['min.log.lambda'].values[:, np.newaxis]))

    correct_2 = np.sum((pred[target_out['min.log.lambda'] == -np.inf]) <=
                       (target_out[target_out['min.log.lambda'] == -np.inf]['max.log.lambda'].values[:, np.newaxis]))

    idx = (target_out['min.log.lambda'] != -np.inf) & (target_out['max.log.lambda'] != np.inf)
    correct_3 = np.sum((pred[idx] >= target_out[idx]['min.log.lambda'].values[:, np.newaxis]) &
                       (pred[idx] <= target_out[idx]['max.log.lambda'].values[:, np.newaxis]))
    acc = (correct_1 + correct_2 + correct_3) / pred.size
    return (acc)

def acc_computer(model, tensor_set, output):
    check = tensor_set.tensors[0]
    if torch.cuda.is_available():
        pred = model(Variable(check).cuda()).cpu().data.numpy()
    else:
        pred = model(Variable(check)).data.numpy()
    correct_1 = np.sum((pred[output['max.log.lambda'] == np.inf]) >=
     (output[output['max.log.lambda'] == np.inf]['min.log.lambda'].values[:,np.newaxis]))

    correct_2 = np.sum((pred[output['min.log.lambda'] == -np.inf]) <=
     (output[output['min.log.lambda'] == -np.inf]['max.log.lambda'].values[:,np.newaxis]))

    idx = (output['min.log.lambda'] != -np.inf) & (output['max.log.lambda'] != np.inf)
    correct_3 = np.sum((pred[idx] >= output[idx]['min.log.lambda'].values[:,np.newaxis]) &
                       (pred[idx] <= output[idx]['max.log.lambda'].values[:,np.newaxis]))
    acc = (correct_1 + correct_2 + correct_3) / pred.size
    return(acc)


def train(model, train_in, train_tar, other_in, other_tar, batch_size, criterion,
          optimer, num_epoches, train_out, other_out, finalflg):
    e = 0
    train_set, train_loader = _loadData(train_in, train_tar,batch_size)
    train_loss_recorder = np.zeros(num_epoches)
    train_acc_recorder = np.zeros(num_epoches)

    other_set, other_loader = _loadData(other_in, other_tar,batch_size)

    if finalflg is False:
        other_loss_recorder = np.zeros(num_epoches)
        other_acc_recorder = np.zeros(num_epoches)
    
    for epoch in range(num_epoches):
        iter_num, print_loss = 0, 0
        for data in train_loader:
            e += 1
            iter_num += 1
            train_inputs, train_targets = data
            if torch.cuda.is_available():
                train_inputs = Variable(train_inputs).cuda()
                train_targets = Variable(train_targets).cuda()
                train_inputs = train_inputs.type(torch.DoubleTensor).cuda()
            else:
                train_inputs = Variable(train_inputs)
                train_targets = Variable(train_targets)
                train_inputs = train_inputs.type(torch.DoubleTensor)
            out = model(train_inputs)
            loss = criterion(out, train_targets.float())
            optimer.zero_grad()
            loss.backward()
            optimer.step()
            if torch.cuda.is_available():
                print_loss += loss.cpu().data.numpy()
            else:
                print_loss += loss.data.numpy()
            train_acc = acc_computer(model, train_set, train_out)

        if finalflg is False:
            v_loss ,v_iter = 0, 0
            for v_data in other_loader:
                v_iter += 1
                v_in, v_tar = v_data
                if torch.cuda.is_available():
                    v_in = Variable(v_in).cuda()
                    v_tar = Variable(v_tar).cuda()
                    v_in = v_in.type(torch.DoubleTensor).cuda()
                else:
                    v_in = Variable(v_in)
                    v_tar = Variable(v_tar)
                    v_in = v_in.type(torch.DoubleTensor)

                v_out = model(v_in)
                temp_loss = criterion(v_out, v_tar.float())
                if torch.cuda.is_available():
                    v_loss += temp_loss.cpu().data.numpy()
                else:
                    v_loss += temp_loss.data.numpy()
                test_acc = acc_computer(model, other_set, other_out)

            print('-'*103)
            print('Epoch [{:-03d}/{}]  |  Train Loss: {:.3f}  |  Train Acc: {:.3f}  | Valid Loss: {:.3f}  |  Valid Acc: {:.3f}  |'.
            format(epoch+1, num_epoches, print_loss/iter_num, train_acc, v_loss/v_iter, test_acc))
            other_loss_recorder[epoch] = v_loss/v_iter
            other_acc_recorder[epoch] = test_acc
        else:
            print('-'*80)
            print('Epoch [{:-03d}/{}]  |  Train Loss:  {:.3f}  |  Train Accuarcy:  {:.3f}  |'
            .format(epoch + 1, num_epoches, print_loss/iter_num, train_acc))
        train_loss_recorder[epoch] = print_loss/iter_num
        train_acc_recorder[epoch] = train_acc

    if finalflg:
        fin_check = other_set.tensors[0]
        if torch.cuda.is_available():
            pred = model(Variable(fin_check).cuda()).cpu().data.numpy()
        else:
            pred = model(Variable(fin_check)).data.numpy()
        return(train_loss_recorder, train_acc_recorder, pred)
    else:
        return(train_loss_recorder, train_acc_recorder,
        other_loss_recorder, other_acc_recorder)

def weight_init(m):
    if isinstance(m,(nn.Conv1d, nn.Linear)):
        #nn.init.orthogonal_(m.weight)
        nn.init.kaiming_uniform_(m.weight)
    elif isinstance(m,nn.BatchNorm1d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()

def writetable(table, testID, valid_id, train_loss,
               train_acc, valid_loss, valid_acc):
    append_size = len(train_acc)
    table['test_ID'] = np.repeat(testID, append_size)
    table['valid_id'] = np.repeat(valid_id, append_size)
    table['train_loss'] = train_loss
    table['train_acc'] = train_acc
    table['valid_loss'] = valid_loss
    table['valid_acc'] = valid_acc
    return(table)

def start_train(model, filepath, testID, **kwargs):
    overallData = pd.read_csv(filepath)    
    valid_id = np.setdiff1d(np.unique(overallData['foldID']), \
        testID)
    t_loss = 0
    t_acc = 0
    v_loss = 0
    v_acc = 0
    batch_size = model.batch_size
    lr = model.lr
    weight_decay = model.weight_decay
    num_epoches = model.num_epoches
    torch.manual_seed(13)

    table_all = pd.DataFrame(columns=['test_ID', 'valid_id', 'train_loss',
                          'train_acc', 'valid_loss', 'valid_acc'])
    for valid_index in valid_id:
        model.apply(weight_init)
        train_data, valid_data, train_label, valid_label, train_out, valid_out = \
            divideData(overallData, testFlg=False, test_id=testID,\
            valid_id=valid_index)
        
        criterion, optimer = modelsetup(model, lr, weight_decay)
        table = pd.DataFrame(columns=['test_ID', 'valid_id', 'train_loss',
                                      'train_acc', 'valid_loss', 'valid_acc'])
        train_loss_recorder, train_acc_recorder,\
        other_loss_recorder, other_acc_recorder = train(model, train_data,\
             train_label, valid_data, valid_label, batch_size, criterion, optimer,
                                                        num_epoches, train_out, valid_out, False)
        table = writetable(table, testID, valid_index, train_loss_recorder,
                           train_acc_recorder, other_loss_recorder, other_acc_recorder)
        table_all = pd.concat([table_all, table], 0, sort=False)

        t_loss += train_loss_recorder
        t_acc += train_acc_recorder
        v_loss += other_loss_recorder
        v_acc += other_acc_recorder

    t_loss /= len(valid_id)
    t_acc /= len(valid_id)
    v_loss /= len(valid_id)
    v_acc /= len(valid_id)

    optim_num_epoches = np.argmax(v_acc)+1

    model.apply(weight_init)

    test_data, train_data, test_label, train_label, \
    test_out, train_out = divideData(overallData, True, testID, None)

    criterion, optimer = modelsetup(model, lr, weight_decay)

    # shows an error here, should not be, need to check out later
    tst_loss, tst_acc, pred = train(model, train_data,
                                    train_label, test_data,
                                    test_label, batch_size, criterion,\
        optimer, optim_num_epoches, train_out, test_out, True)

    pred_acc = get_acc(pred, test_out)
    pred_acc = pd.DataFrame(np.array(pred_acc).reshape(-1,1))
    pred = pd.DataFrame(pred)
    optim_num_epoches = pd.DataFrame(np.array(optim_num_epoches).reshape(-1,1))
    return(tst_loss, tst_acc, pred, pred_acc,
           optim_num_epoches, table_all.reset_index())

