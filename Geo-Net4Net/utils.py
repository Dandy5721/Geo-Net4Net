import numpy as np
import os.path
import re

import scipy.io as scio

from sklearn import metrics
from sklearn.metrics import accuracy_score, normalized_mutual_info_score
from tqdm import tqdm

def sorted_aphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(data, key=alphanum_key)

def _read_file(path, dtype=np.float32):
    if os.path.exists(path):
        if path.endswith('.mat'):
            data = scio.loadmat(path)
            return data['rawdata']
        elif path.endswith('.csv'):
            data = np.loadtxt(path, dtype=dtype, delimiter=',')
            return data
        elif path.endswith('.txt'):
            data = np.loadtxt(path, dtype=dtype)
            return data
        elif path.endswith('.npy'):
            data = np.load(path)
            return data
    else:
        raise Exception("The file doesn't exist")

def read_data(path, dtype=np.float32, startswith=None, endswith=None):
    if isinstance(path, list):
        datas = []
        for p in path:
            if (startswith is None or os.path.basename(p).startswith(startswith)) and (
                    endswith is None or p.endswith(endswith)):
                data = _read_file(p, dtype)
                print(p)
                datas.append(data)
        return datas
    elif os.path.isdir(path):
        datas = []
        files_name = [os.path.join(path, name) for name in sorted_aphanumeric(os.listdir(path)) if
                      (startswith is None or name.startswith(startswith))
                      and (endswith is None or name.endswith(endswith))]

        bar = tqdm(files_name)
        for file_name in bar:
            # file_path = os.path.join(path, file_name)
            # file_path = file_name
            data = _read_file(file_name, dtype)
            datas.append(data)
            bar.set_description('reading %s' % (file_name))
        return datas
    else:
        data = _read_file(path, dtype)
        return data

def purity_score(y_true, y_pred):
    # compute contingency matrix
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)

def cluster_score(labels_true, labels_pred):
    purity = purity_score(labels_true, labels_pred)
    acc = accuracy_score(labels_true, labels_pred)
    nmi = normalized_mutual_info_score(labels_true, labels_pred)
    return purity, acc, nmi