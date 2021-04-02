from lvis.lvis import LVIS
import numpy as np
import pickle
import pdb
import os
import json
import torch
from pycocotools.coco import COCO

def get_cate_gs():

    train_ann_file = './data/lvis/lvis_v0.5_train.json'
    lvis_train = LVIS(train_ann_file)
    train_catsinfo = lvis_train.cats

    binlabel_count = [1, 1, 1, 1, 1]
    label2binlabel = np.zeros((5, 1231), dtype=np.int)

    label2binlabel[0, 1:] = binlabel_count[0]
    binlabel_count[0] += 1

    for cid, cate in train_catsinfo.items():
        ins_count = cate['instance_count']
        if ins_count < 10:
            label2binlabel[1, cid] = binlabel_count[1]
            binlabel_count[1] += 1
        elif ins_count < 100:
            label2binlabel[2, cid] = binlabel_count[2]
            binlabel_count[2] += 1
        elif ins_count < 1000:
            label2binlabel[3, cid] = binlabel_count[3]
            binlabel_count[3] += 1
        else:
            label2binlabel[4, cid] = binlabel_count[4]
            binlabel_count[4] += 1


    savebin = torch.from_numpy(label2binlabel)

    save_path = './data/lvis/label2binlabel.pt'
    torch.save(savebin, save_path)

    # start and length
    pred_slice = np.zeros((5, 2), dtype=np.int)
    start_idx = 0
    for i, bincount in enumerate(binlabel_count):
        pred_slice[i, 0] = start_idx
        pred_slice[i, 1] = bincount
        start_idx += bincount

    savebin = torch.from_numpy(pred_slice)
    save_path = './data/lvis/pred_slice_with0.pt'
    torch.save(savebin, save_path)

    # pdb.set_trace()

    return pred_slice

def get_split():

    train_ann_file = './data/lvis/lvis_v0.5_train.json'
    val_ann_file = './data/lvis/lvis_v0.5_val.json'

    # For training set
    lvis_train = LVIS(train_ann_file)
    # lvis_val = LVIS(val_ann_file)
    train_catsinfo = lvis_train.cats
    # val_catsinfo = lvis_val.cats

    bin10 = []
    bin100 = []
    bin1000 = []
    binover = []

    for cid, cate in train_catsinfo.items():
        ins_count = cate['instance_count']
        if ins_count < 10:
            bin10.append(cid)
        elif ins_count < 100:
            bin100.append(cid)
        elif ins_count < 1000:
            bin1000.append(cid)
        else:
            binover.append(cid)

    splits = {}
    splits['(0, 10)'] = np.array(bin10, dtype=np.int)
    splits['[10, 100)'] = np.array(bin100, dtype=np.int)
    splits['[100, 1000)'] = np.array(bin1000, dtype=np.int)
    splits['[1000, ~)'] = np.array(binover, dtype=np.int)
    splits['normal'] = np.arange(1, 1231)
    splits['background'] = np.zeros((1,), dtype=np.int)
    splits['all'] = np.arange(1231)

    split_file_name = './data/lvis/valsplit.pkl'
    with open(split_file_name, 'wb') as f:
        pickle.dump(splits, f)

def get_cate_gs2():

    train_ann_file = './data/lvis/lvis_v0.5_train.json'
    lvis_train = LVIS(train_ann_file)
    train_catsinfo = lvis_train.cats

    binlabel_count = [1, 1, 1]
    label2binlabel = np.zeros((3, 1231), dtype=np.int)

    label2binlabel[0, 1:] = binlabel_count[0]
    binlabel_count[0] += 1

    for cid, cate in train_catsinfo.items():
        ins_count = cate['instance_count']
        if ins_count < 100:
            label2binlabel[1, cid] = binlabel_count[1]
            binlabel_count[1] += 1
        else:
            label2binlabel[2, cid] = binlabel_count[2]
            binlabel_count[2] += 1


    savebin = torch.from_numpy(label2binlabel)

    save_path = './data/lvis/2bins/label2binlabel.pt'
    torch.save(savebin, save_path)

    # start and length
    pred_slice = np.zeros((3, 2), dtype=np.int)
    start_idx = 0
    for i, bincount in enumerate(binlabel_count):
        pred_slice[i, 0] = start_idx
        pred_slice[i, 1] = bincount
        start_idx += bincount

    savebin = torch.from_numpy(pred_slice)
    save_path = './data/lvis/2bins/pred_slice_with0.pt'
    torch.save(savebin, save_path)

    #pdb.set_trace()

def get_split2():

    train_ann_file = './data/lvis/lvis_v0.5_train.json'
    val_ann_file = './data/lvis/lvis_v0.5_val.json'

    # For training set
    lvis_train = LVIS(train_ann_file)
    # lvis_val = LVIS(val_ann_file)
    train_catsinfo = lvis_train.cats
    # val_catsinfo = lvis_val.cats

    bin100 = []
    binover = []

    for cid, cate in train_catsinfo.items():
        ins_count = cate['instance_count']

        if ins_count < 100:
            bin100.append(cid)
        else:
            binover.append(cid)

    splits = {}
    # splits['(0, 10)'] = np.array(bin10, dtype=np.int)
    splits['[10, 100)'] = np.array(bin100, dtype=np.int)
    # splits['[100, 1000)'] = np.array(bin1000, dtype=np.int)
    splits['[1000, ~)'] = np.array(binover, dtype=np.int)
    splits['normal'] = np.arange(1, 1231)
    splits['background'] = np.zeros((1,), dtype=np.int)
    splits['all'] = np.arange(1231)

    split_file_name = './data/lvis/2bins/valsplit.pkl'
    with open(split_file_name, 'wb') as f:
        pickle.dump(splits, f)

def get_cate_gs3():

    train_ann_file = './data/lvis/lvis_v0.5_train.json'
    lvis_train = LVIS(train_ann_file)
    train_catsinfo = lvis_train.cats

    binlabel_count = [1, 1, 1, 1]
    label2binlabel = np.zeros((4, 1231), dtype=np.int)

    label2binlabel[0, 1:] = binlabel_count[0]
    binlabel_count[0] += 1

    for cid, cate in train_catsinfo.items():
        ins_count = cate['instance_count']
        if ins_count < 100:
            label2binlabel[1, cid] = binlabel_count[1]
            binlabel_count[1] += 1
        elif ins_count < 1000:
            label2binlabel[2, cid] = binlabel_count[2]
            binlabel_count[2] += 1
        else:
            label2binlabel[3, cid] = binlabel_count[3]
            binlabel_count[3] += 1


    savebin = torch.from_numpy(label2binlabel)

    save_path = './data/lvis/3bins/label2binlabel.pt'
    torch.save(savebin, save_path)

    # start and length
    pred_slice = np.zeros((4, 2), dtype=np.int)
    start_idx = 0
    for i, bincount in enumerate(binlabel_count):
        pred_slice[i, 0] = start_idx
        pred_slice[i, 1] = bincount
        start_idx += bincount

    savebin = torch.from_numpy(pred_slice)
    save_path = './data/lvis/3bins/pred_slice_with0.pt'
    torch.save(savebin, save_path)

    #pdb.set_trace()

def get_split3():

    train_ann_file = './data/lvis/lvis_v0.5_train.json'
    val_ann_file = './data/lvis/lvis_v0.5_val.json'

    # For training set
    lvis_train = LVIS(train_ann_file)
    # lvis_val = LVIS(val_ann_file)
    train_catsinfo = lvis_train.cats
    # val_catsinfo = lvis_val.cats

    bin100 = []
    bin1000 = []
    binover = []

    for cid, cate in train_catsinfo.items():
        ins_count = cate['instance_count']

        if ins_count < 100:
            bin100.append(cid)
        elif ins_count < 1000:
            bin1000.append(cid)
        else:
            binover.append(cid)

    splits = {}
    # splits['(0, 10)'] = np.array(bin10, dtype=np.int)
    splits['[0, 100)'] = np.array(bin100, dtype=np.int)
    splits['[100, 1000)'] = np.array(bin1000, dtype=np.int)
    splits['[1000, ~)'] = np.array(binover, dtype=np.int)
    splits['normal'] = np.arange(1, 1231)
    splits['background'] = np.zeros((1,), dtype=np.int)
    splits['all'] = np.arange(1231)

    split_file_name = './data/lvis/3bins/valsplit.pkl'
    with open(split_file_name, 'wb') as f:
        pickle.dump(splits, f)

def get_cate_gs5():

    train_ann_file = './data/lvis/lvis_v0.5_train.json'
    lvis_train = LVIS(train_ann_file)
    train_catsinfo = lvis_train.cats

    binlabel_count = [1, 1, 1, 1, 1, 1]
    label2binlabel = np.zeros((6, 1231), dtype=np.int)

    label2binlabel[0, 1:] = binlabel_count[0]
    binlabel_count[0] += 1

    for cid, cate in train_catsinfo.items():
        ins_count = cate['instance_count']
        if ins_count < 100:
            label2binlabel[1, cid] = binlabel_count[1]
            binlabel_count[1] += 1
        elif ins_count < 500:
            label2binlabel[2, cid] = binlabel_count[2]
            binlabel_count[2] += 1
        elif ins_count < 1000:
            label2binlabel[3, cid] = binlabel_count[3]
            binlabel_count[3] += 1
        elif ins_count < 5000:
            label2binlabel[4, cid] = binlabel_count[4]
            binlabel_count[4] += 1
        else:
            label2binlabel[5, cid] = binlabel_count[5]
            binlabel_count[5] += 1


    savebin = torch.from_numpy(label2binlabel)

    save_path = './data/lvis/5bins/label2binlabel.pt'
    torch.save(savebin, save_path)

    # start and length
    pred_slice = np.zeros((6, 2), dtype=np.int)
    start_idx = 0
    for i, bincount in enumerate(binlabel_count):
        pred_slice[i, 0] = start_idx
        pred_slice[i, 1] = bincount
        start_idx += bincount

    savebin = torch.from_numpy(pred_slice)
    save_path = './data/lvis/5bins/pred_slice_with0.pt'
    torch.save(savebin, save_path)

    #pdb.set_trace()

def get_split5():

    train_ann_file = './data/lvis/lvis_v0.5_train.json'
    val_ann_file = './data/lvis/lvis_v0.5_val.json'

    # For training set
    lvis_train = LVIS(train_ann_file)
    # lvis_val = LVIS(val_ann_file)
    train_catsinfo = lvis_train.cats
    # val_catsinfo = lvis_val.cats

    bin100 = []
    bin500 = []
    bin1000 = []
    bin5000 = []
    binover = []

    for cid, cate in train_catsinfo.items():
        ins_count = cate['instance_count']

        if ins_count < 100:
            bin100.append(cid)
        elif ins_count < 500:
            bin500.append(cid)
        elif ins_count < 1000:
            bin1000.append(cid)
        elif ins_count < 5000:
            bin5000.append(cid)
        else:
            binover.append(cid)

    splits = {}
    # splits['(0, 10)'] = np.array(bin10, dtype=np.int)
    splits['(0, 100)'] = np.array(bin100, dtype=np.int)
    splits['[100, 500)'] = np.array(bin500, dtype=np.int)
    splits['[500, 1000)'] = np.array(bin1000, dtype=np.int)
    splits['[1000, 5000)'] = np.array(bin5000, dtype=np.int)
    splits['[5000, ~)'] = np.array(binover, dtype=np.int)
    splits['normal'] = np.arange(1, 1231)
    splits['background'] = np.zeros((1,), dtype=np.int)
    splits['all'] = np.arange(1231)

    split_file_name = './data/lvis/5bins/valsplit.pkl'
    with open(split_file_name, 'wb') as f:
        pickle.dump(splits, f)

def get_cate_gs6():

    train_ann_file = './data/lvis/lvis_v0.5_train.json'
    lvis_train = LVIS(train_ann_file)
    train_catsinfo = lvis_train.cats

    binlabel_count = [1, 1, 1, 1, 1, 1, 1]
    label2binlabel = np.zeros((7, 1231), dtype=np.int)

    label2binlabel[0, 1:] = binlabel_count[0]
    binlabel_count[0] += 1

    for cid, cate in train_catsinfo.items():
        ins_count = cate['instance_count']
        if ins_count < 50:
            label2binlabel[1, cid] = binlabel_count[1]
            binlabel_count[1] += 1
        elif ins_count < 100:
            label2binlabel[2, cid] = binlabel_count[2]
            binlabel_count[2] += 1
        elif ins_count < 500:
            label2binlabel[3, cid] = binlabel_count[3]
            binlabel_count[3] += 1
        elif ins_count < 1000:
            label2binlabel[4, cid] = binlabel_count[4]
            binlabel_count[4] += 1
        elif ins_count < 5000:
            label2binlabel[5, cid] = binlabel_count[5]
            binlabel_count[5] += 1
        else:
            label2binlabel[6, cid] = binlabel_count[6]
            binlabel_count[6] += 1


    savebin = torch.from_numpy(label2binlabel)

    save_path = './data/lvis/6bins/label2binlabel.pt'
    torch.save(savebin, save_path)

    # start and length
    pred_slice = np.zeros((7, 2), dtype=np.int)
    start_idx = 0
    for i, bincount in enumerate(binlabel_count):
        pred_slice[i, 0] = start_idx
        pred_slice[i, 1] = bincount
        start_idx += bincount

    savebin = torch.from_numpy(pred_slice)
    save_path = './data/lvis/6bins/pred_slice_with0.pt'
    torch.save(savebin, save_path)

    #pdb.set_trace()

def get_split6():

    train_ann_file = './data/lvis/lvis_v0.5_train.json'
    val_ann_file = './data/lvis/lvis_v0.5_val.json'

    # For training set
    lvis_train = LVIS(train_ann_file)
    # lvis_val = LVIS(val_ann_file)
    train_catsinfo = lvis_train.cats
    # val_catsinfo = lvis_val.cats

    bin50 = []
    bin100 = []
    bin500 = []
    bin1000 = []
    bin5000 = []
    binover = []

    for cid, cate in train_catsinfo.items():
        ins_count = cate['instance_count']

        if ins_count < 50:
            bin50.append(cid)
        elif ins_count < 100:
            bin100.append(cid)
        elif ins_count < 500:
            bin500.append(cid)
        elif ins_count < 1000:
            bin1000.append(cid)
        elif ins_count < 5000:
            bin5000.append(cid)
        else:
            binover.append(cid)

    splits = {}
    # splits['(0, 10)'] = np.array(bin10, dtype=np.int)
    splits['(0, 50)'] = np.array(bin50, dtype=np.int)
    splits['[50, 100)'] = np.array(bin100, dtype=np.int)
    splits['[100, 500)'] = np.array(bin500, dtype=np.int)
    splits['[500, 1000)'] = np.array(bin1000, dtype=np.int)
    splits['[1000, 5000)'] = np.array(bin5000, dtype=np.int)
    splits['[5000, ~)'] = np.array(binover, dtype=np.int)
    splits['normal'] = np.arange(1, 1231)
    splits['background'] = np.zeros((1,), dtype=np.int)
    splits['all'] = np.arange(1231)

    split_file_name = './data/lvis/6bins/valsplit.pkl'
    with open(split_file_name, 'wb') as f:
        pickle.dump(splits, f)

def get_cate_gs7():

    train_ann_file = './data/lvis/lvis_v0.5_train.json'
    lvis_train = LVIS(train_ann_file)
    train_catsinfo = lvis_train.cats

    binlabel_count = [1, 1, 1, 1, 1, 1, 1, 1]
    label2binlabel = np.zeros((8, 1231), dtype=np.int)

    label2binlabel[0, 1:] = binlabel_count[0]
    binlabel_count[0] += 1

    for cid, cate in train_catsinfo.items():
        ins_count = cate['instance_count']
        if ins_count < 10:
            label2binlabel[1, cid] = binlabel_count[1]
            binlabel_count[1] += 1
        elif ins_count < 50:
            label2binlabel[2, cid] = binlabel_count[2]
            binlabel_count[2] += 1
        elif ins_count < 100:
            label2binlabel[3, cid] = binlabel_count[3]
            binlabel_count[3] += 1
        elif ins_count < 500:
            label2binlabel[4, cid] = binlabel_count[4]
            binlabel_count[4] += 1
        elif ins_count < 1000:
            label2binlabel[5, cid] = binlabel_count[5]
            binlabel_count[5] += 1
        elif ins_count < 5000:
            label2binlabel[6, cid] = binlabel_count[6]
            binlabel_count[6] += 1
        else:
            label2binlabel[7, cid] = binlabel_count[7]
            binlabel_count[7] += 1


    savebin = torch.from_numpy(label2binlabel)

    save_path = './data/lvis/7bins/label2binlabel.pt'
    torch.save(savebin, save_path)

    # start and length
    pred_slice = np.zeros((8, 2), dtype=np.int)
    start_idx = 0
    for i, bincount in enumerate(binlabel_count):
        pred_slice[i, 0] = start_idx
        pred_slice[i, 1] = bincount
        start_idx += bincount

    savebin = torch.from_numpy(pred_slice)
    save_path = './data/lvis/7bins/pred_slice_with0.pt'
    torch.save(savebin, save_path)

    #pdb.set_trace()

def get_split7():

    train_ann_file = './data/lvis/lvis_v0.5_train.json'
    val_ann_file = './data/lvis/lvis_v0.5_val.json'

    # For training set
    lvis_train = LVIS(train_ann_file)
    # lvis_val = LVIS(val_ann_file)
    train_catsinfo = lvis_train.cats
    # val_catsinfo = lvis_val.cats

    bin10 = []
    bin50 = []
    bin100 = []
    bin500 = []
    bin1000 = []
    bin5000 = []
    binover = []

    for cid, cate in train_catsinfo.items():
        ins_count = cate['instance_count']

        if ins_count < 10:
            bin10.append(cid)
        elif ins_count < 50:
            bin50.append(cid)
        elif ins_count < 100:
            bin100.append(cid)
        elif ins_count < 500:
            bin500.append(cid)
        elif ins_count < 1000:
            bin1000.append(cid)
        elif ins_count < 5000:
            bin5000.append(cid)
        else:
            binover.append(cid)

    splits = {}
    # splits['(0, 10)'] = np.array(bin10, dtype=np.int)
    splits['(0, 10)'] = np.array(bin10, dtype=np.int)
    splits['[10, 50)'] = np.array(bin50, dtype=np.int)
    splits['[50, 100)'] = np.array(bin100, dtype=np.int)
    splits['[100, 500)'] = np.array(bin500, dtype=np.int)
    splits['[500, 1000)'] = np.array(bin1000, dtype=np.int)
    splits['[1000, 5000)'] = np.array(bin5000, dtype=np.int)
    splits['[5000, ~)'] = np.array(binover, dtype=np.int)
    splits['normal'] = np.arange(1, 1231)
    splits['background'] = np.zeros((1,), dtype=np.int)
    splits['all'] = np.arange(1231)

    split_file_name = './data/lvis/7bins/valsplit.pkl'
    with open(split_file_name, 'wb') as f:
        pickle.dump(splits, f)

def get_cate_gs8():

    train_ann_file = './data/lvis/lvis_v0.5_train.json'
    lvis_train = LVIS(train_ann_file)
    train_catsinfo = lvis_train.cats

    binlabel_count = [1, 1, 1, 1, 1, 1, 1, 1, 1]
    label2binlabel = np.zeros((9, 1231), dtype=np.int)

    label2binlabel[0, 1:] = binlabel_count[0]
    binlabel_count[0] += 1

    for cid, cate in train_catsinfo.items():
        ins_count = cate['instance_count']
        if ins_count < 5:
            label2binlabel[1, cid] = binlabel_count[1]
            binlabel_count[1] += 1
        elif ins_count < 10:
            label2binlabel[2, cid] = binlabel_count[2]
            binlabel_count[2] += 1
        elif ins_count < 50:
            label2binlabel[3, cid] = binlabel_count[3]
            binlabel_count[3] += 1
        elif ins_count < 100:
            label2binlabel[4, cid] = binlabel_count[4]
            binlabel_count[4] += 1
        elif ins_count < 500:
            label2binlabel[5, cid] = binlabel_count[5]
            binlabel_count[5] += 1
        elif ins_count < 1000:
            label2binlabel[6, cid] = binlabel_count[6]
            binlabel_count[6] += 1
        elif ins_count < 5000:
            label2binlabel[7, cid] = binlabel_count[7]
            binlabel_count[7] += 1
        else:
            label2binlabel[8, cid] = binlabel_count[8]
            binlabel_count[8] += 1


    savebin = torch.from_numpy(label2binlabel)

    save_path = './data/lvis/8bins/label2binlabel.pt'
    torch.save(savebin, save_path)

    # start and length
    pred_slice = np.zeros((9, 2), dtype=np.int)
    start_idx = 0
    for i, bincount in enumerate(binlabel_count):
        pred_slice[i, 0] = start_idx
        pred_slice[i, 1] = bincount
        start_idx += bincount

    savebin = torch.from_numpy(pred_slice)
    save_path = './data/lvis/8bins/pred_slice_with0.pt'
    torch.save(savebin, save_path)

    #pdb.set_trace()

def get_split8():

    train_ann_file = './data/lvis/lvis_v0.5_train.json'
    val_ann_file = './data/lvis/lvis_v0.5_val.json'

    # For training set
    lvis_train = LVIS(train_ann_file)
    # lvis_val = LVIS(val_ann_file)
    train_catsinfo = lvis_train.cats
    # val_catsinfo = lvis_val.cats

    bin10 = []
    bin100 = []
    bin1000 = []
    binover = []
    bin5 = []
    bin50 = []
    bin500 = []
    bin5000 = []


    for cid, cate in train_catsinfo.items():
        ins_count = cate['instance_count']

        if ins_count < 5:
            bin5.append(cid)
        elif ins_count < 10:
            bin10.append(cid)
        elif ins_count < 50:
            bin50.append(cid)
        elif ins_count < 100:
            bin100.append(cid)
        elif ins_count < 500:
            bin500.append(cid)
        elif ins_count < 1000:
            bin1000.append(cid)
        elif ins_count < 5000:
            bin5000.append(cid)
        else:
            binover.append(cid)

    splits = {}
    splits['(5, 10)'] = np.array(bin10, dtype=np.int)
    splits['[50, 100)'] = np.array(bin100, dtype=np.int)
    splits['[500, 1000)'] = np.array(bin1000, dtype=np.int)
    splits['[5000, ~)'] = np.array(binover, dtype=np.int)
    splits['(0, 5)'] = np.array(bin5, dtype=np.int)
    splits['[10, 50)'] = np.array(bin50, dtype=np.int)
    splits['[100, 500)'] = np.array(bin500, dtype=np.int)
    splits['[1000, 5000)'] = np.array(bin5000, dtype=np.int)
    splits['normal'] = np.arange(1, 1231)
    splits['background'] = np.zeros((1,), dtype=np.int)
    splits['all'] = np.arange(1231)

    split_file_name = './data/lvis/8bins/valsplit.pkl'
    with open(split_file_name, 'wb') as f:
        pickle.dump(splits, f)

def get_cate_gs2_wt():

    train_ann_file = './data/lvis/lvis_v0.5_train.json'
    lvis_train = LVIS(train_ann_file)
    train_catsinfo = lvis_train.cats

    binlabel_count = [1, 1, 1]
    label2binlabel = np.zeros((3, 1231), dtype=np.int)

    label2binlabel[0, 1:] = binlabel_count[0]
    binlabel_count[0] += 1

    for cid, cate in train_catsinfo.items():
        ins_count = cate['instance_count']
        if ins_count < 300:
            label2binlabel[1, cid] = binlabel_count[1]
            binlabel_count[1] += 1
        else:
            label2binlabel[2, cid] = binlabel_count[2]
            binlabel_count[2] += 1


    savebin = torch.from_numpy(label2binlabel)

    save_path = './data/lvis/2bins300/label2binlabel.pt'
    torch.save(savebin, save_path)

    # start and length
    pred_slice = np.zeros((3, 2), dtype=np.int)
    start_idx = 0
    for i, bincount in enumerate(binlabel_count):
        pred_slice[i, 0] = start_idx
        pred_slice[i, 1] = bincount
        start_idx += bincount

    savebin = torch.from_numpy(pred_slice)
    save_path = './data/lvis/2bins300/pred_slice_with0.pt'
    torch.save(savebin, save_path)

    #pdb.set_trace()

def get_split2_wt():

    train_ann_file = './data/lvis/lvis_v0.5_train.json'
    val_ann_file = './data/lvis/lvis_v0.5_val.json'

    # For training set
    lvis_train = LVIS(train_ann_file)
    # lvis_val = LVIS(val_ann_file)
    train_catsinfo = lvis_train.cats
    # val_catsinfo = lvis_val.cats

    bin300 = []
    binover = []

    for cid, cate in train_catsinfo.items():
        ins_count = cate['instance_count']

        if ins_count < 300:
            bin300.append(cid)
        else:
            binover.append(cid)

    splits = {}
    splits['(0, 300)'] = np.array(bin300, dtype=np.int)
    splits['[300, ~)'] = np.array(binover, dtype=np.int)
    splits['normal'] = np.arange(1, 1231)
    splits['background'] = np.zeros((1,), dtype=np.int)
    splits['all'] = np.arange(1231)

    split_file_name = './data/lvis/2bins300/valsplit.pkl'
    with open(split_file_name, 'wb') as f:
        pickle.dump(splits, f)

if __name__ == '__main__':
    get_cate_gs2_wt()
    get_split2_wt()
