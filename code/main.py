import os
import re
import csv
import configparser
import random
import joblib

from torch import optim, nn
from torch.optim import lr_scheduler
from tqdm import tqdm
import numpy as np
import torch

from model import SiameseNetwork, MLPNetwork

config = configparser.ConfigParser()
config.read('./config.ini', encoding='UTF-8')

device = config['cuda']['CUDA_DEVICE']

List_remove = ['{', '}', '(', ')', '[', ']', '+', '-', '*', '/',
               '&', ':', '.', ',', ';', '!', '<', '>', '\'', '\"']
Dict_cnt = {}


def load_data(path, pre=False, dict_replace=None):
    data = []

    if dict_replace is None:
        dict_replace = {}

    classes = os.listdir(path)
    class_num = 0
    for name in tqdm(classes, desc="Loading data"):
        classdata = []
        classpath = os.path.join(path, name)
        codes = os.listdir(classpath)
        for cname in codes:
            with open(os.path.join(classpath, cname), 'r') as f:
                code = preprocess(f.read(), dict_replace, pre)
            if not pre:
                classdata.append([code, class_num])
        if not pre:
            data.append(classdata)
            class_num += 1

    return data, dict_replace


def load_testdata(path, dict_replace):
    data = []
    dict_name = {}

    codes = os.listdir(path)
    for fullcname in tqdm(codes, desc="Loading data"):
        with open(os.path.join(path, fullcname), 'r') as f:
            code = preprocess(f.read(), dict_replace, pre=False)
        cname, _ = os.path.splitext(fullcname)
        dict_name[cname] = len(data)
        data.append(code)

    return data, dict_name


def preprocess(code, dict_replace, pre=True):
    code = code.replace('\n', ' ')
    for remove in List_remove:
        code = code.replace(remove, f' {remove} ')
    code = re.split(' +', code)

    num_feature = len(dict_replace) + 1
    list_cnt = np.zeros(num_feature)

    for i, word in enumerate(code):
        if pre:
            if not word in Dict_cnt:
                Dict_cnt[word] = 0
            Dict_cnt[word] += 1
        elif word in dict_replace:
            list_cnt[dict_replace[word]] += 1

    # return code
    if not pre:
        if np.sum(list_cnt) > 0:
            list_cnt = list_cnt / np.sum(list_cnt)
        list_cnt[-1] = min(len(code) / 1000, 1.0)

    return list_cnt


def load_dict():
    returndict = {}
    with open(config['path']['OTHER_PATH'], 'r')as f:
        f_csv = csv.reader(f)
        for i, row in enumerate(f_csv):
            returndict[row[0]] = i
    return returndict


def dictprocess(dict_cnt, limit=0):
    dictlist = []
    for ele in list(dict_cnt):
        if dict_cnt[ele] > limit:
            dictlist.append([ele, dict_cnt[ele]])
    dictlist.sort(key=lambda x: x[1], reverse=True)

    returndict = {}
    for i, ele in enumerate(dictlist):
        returndict[ele[0]] = i
    return returndict


def dict2code(data, dict):
    code = ''
    dict_rev = {}
    for ele in list(dict):
        dict_rev[dict[ele]] = ele
    for d in data:
        if d == -1:
            code += 'VAR'
        else:
            code += dict_rev[d]
        code += ' '
    return code


def write_data(other):
    with open(config['path']['OTHER_PATH'], 'w', newline='')as f:
        f_csv = csv.writer(f)
        for word in list(other):
            f_csv.writerow([word, other[word]])


def sample_data(data, num_class=80, num_in_one_class=2):
    sample_batch = []
    sample_class = random.sample(data, k=num_class)
    for oneclass in sample_class:
        samples = random.sample(oneclass, k=num_in_one_class)
        sample_batch.extend(samples)
    # random.shuffle(sample_batch)
    # fit network
    batch_diff = []
    batch_same = []
    batch_self = []
    sample_batchsize = len(sample_batch)
    for i in range(sample_batchsize):
        batch_self.append([sample_batch[i][0], sample_batch[i][0], [0, 1]])
        for j in range(sample_batchsize):
            if i == j:
                continue
            if sample_batch[i][1] == sample_batch[j][1]:
                label = [0, 1]
                batch_same.append([sample_batch[i][0], sample_batch[j][0], label])
            else:
                label = [1, 0]
                batch_diff.append([sample_batch[i][0], sample_batch[j][0], label])
    batch_diff = random.sample(batch_diff, k=len(batch_same))
    if 2 * len(batch_self) > len(batch_same):
        batch_self = random.sample(batch_self, k=int(len(batch_same)/2))
    batch = batch_diff
    batch.extend(batch_same)
    batch.extend(batch_self)
    random.shuffle(batch)
    batch_X = []
    batch_Y = []
    for sample in batch:
        batch_X.append(sample[:2])
        batch_Y.append(sample[2])
    return batch_X, batch_Y


def loss_func(predict, Y, critic):
    _, a = torch.max(predict, 1)
    b = Y[:, 1]
    TP = len(b[a == 1][b[a == 1] == 1])
    FN = len(b[a == 0][b[a == 0] == 1])
    FP = len(b[a == 1][b[a == 1] == 0])
    p = TP / (TP + FP) if TP + FP > 0 else TP
    r = TP / (TP + FN) if TP + FN > 0 else TP
    F1 = 2 * r * p / (r + p) if r + p > 0 else 2 * r * p

    loss = critic(predict, Y)
    # loss = torch.mean((1 - predict) * Y)

    print(f'F1 score: {F1}, precision: {p}, recall: {r}.')
    return loss


def foryou():
    model = torch.load(config['path']['MODEL_PATH'])
    mydict = joblib.load(config['path']['DICT_PATH'])
    test(model, mydict)

def test(model, mydict):
    model.eval()
    path = config['path']['TEST_PATH']
    testdata, dict_name = load_testdata(path, dict_replace=mydict)

    with open(config['path']['SAMPLE_PATH'], 'r')as f_sample:
        f_s = csv.reader(f_sample)
        with open(config['path']['PREDICT_PATH'], 'w', newline='')as f_output:
            f_out = csv.writer(f_output)

            buffer_out = []
            for i, row in enumerate(tqdm(f_s, desc="Predicting")):
                if i == 0:
                    f_out.writerow(row)
                    continue
                fnames = row[0].split('_')
                fname1 = fnames[0]
                fname2 = fnames[1]

                sample1 = testdata[dict_name[fname1]]
                sample2 = testdata[dict_name[fname2]]
                predict = model.forward_single(sample1, sample2)
                predict = predict[0, :]
                result = 0 if predict[0] > predict[1] else 1

                buffer_out.append([row[0], result])
                if len(buffer_out) > 10000:
                    f_out.writerows(buffer_out)
                    buffer_out = []
            f_out.writerows(buffer_out)


def main():
    path = config['path']['TRAIN_PATH']
    data, _ = load_data(path, pre=True)

    write_data(Dict_cnt)
    mydict = dictprocess(Dict_cnt, limit=50)
    # mydict = load_dict()
    joblib.dump(mydict, config['path']['DICT_PATH'])

    data, _ = load_data(path, pre=False, dict_replace=mydict)

    len_feature = len(mydict) + 1
    print(f'num of feature:{len_feature}')

    # model = SiameseNetwork(len_feature, 32, 2, device=device).to(device)
    model = MLPNetwork(len_feature, 2, device=device).to(device)
    opt = optim.Adam(model.parameters(), lr=0.0006)
    scheduler = lr_scheduler.CosineAnnealingLR(opt, 50000, eta_min=0, last_epoch=-1)
    critic = nn.BCELoss().to(device)

    for iter in range(50000):
        opt.zero_grad()
        # [code1, code2, label]
        X, Y = sample_data(data, num_in_one_class=3)
        result = model.forward(X)
        loss = loss_func(result, torch.FloatTensor(Y).to(device), critic)
        loss.backward()
        opt.step()
        scheduler.step()
        print(f'iter:{iter}, loss:{loss}.\n')

    torch.save(model, config['path']['MODEL_PATH'])
    test(model, mydict)

if __name__ == "__main__":
    # main()
    foryou()

