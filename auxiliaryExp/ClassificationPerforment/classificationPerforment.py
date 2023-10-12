
import sys
sys.path.insert(0, sys.path[0]+"/../../")
from utils import metrics, evaluate
import pickle
import numpy as np
import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load("./auxiliaryExp/ONEIL.pth", map_location=torch.device('cpu'))
model.to(device)


def clssification(y_true, y_pred, threshold):

    y_true = np.array(list(map(lambda x : 1 if x >= threshold[1] else x, y_true)))
    y_true = np.array(list(map(lambda x : 0 if x < threshold[0] else x, y_true)))

    y_pred = np.array(list(map(lambda x : 1 if x >= threshold[1] else x, y_pred)))
    y_pred = np.array(list(map(lambda x : 0 if x < threshold[0] else x, y_pred)))

    _yt = []
    _yp = []
    for a, b in zip(y_true, y_pred):
        if (a==0 or a==1) and (b==0 or b==1):
            _yt.append(a)
            _yp.append(b)
    y_true = np.array(_yt)
    y_pred = np.array(_yp)

    acc = metrics.acc(y_true, y_pred)
    kappa = metrics.kappa(y_true, y_pred)
    bacc = metrics.bacc(y_true, y_pred)
    roc_auc = metrics.roc_auc(y_true, y_pred)
    prec = metrics.prec(y_true, y_pred)

    print(f"| roc_auc: {roc_auc} |")
    print(f"| acc: {acc} |")
    print(f"| bacc: {bacc} |")
    print(f"| prec: {prec} |")
    print(f"| kappa: {kappa} |")


if __name__ == '__main__':

    with open(f"data/ONEIL_test.pkl", 'rb') as fp:
        _e = pickle.load(fp)

    (trues, preds), _ = evaluate(model, _e, device)
    threshold = [-10, 10]

    clssification(trues, preds, threshold)
