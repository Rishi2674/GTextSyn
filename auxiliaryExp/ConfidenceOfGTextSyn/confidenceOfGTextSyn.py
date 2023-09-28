import sys
sys.path.insert(0, sys.path[0]+"/../../")
import os
import torch
import pickle
import numpy as np
from utils import evaluate

folder = "./auxiliaryExp/ConfidenceOfGTextSyn/models/"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def modelConfidence(study="ONEIL"):
    sa = []
    for li in os.listdir(folder):
        if li == "README":
            continue
        model = torch.load(folder + f'{li}/model.pth', map_location=device)
        with open(f"data/{study}_test.pkl", 'rb') as fp:
            ds = pickle.load(fp)
        _, res = evaluate(model, ds, device)
        sa.append((li, res))
    
    mse = np.array([(i[1][0], i[0]) for i in sa])
    pearson = np.array([(i[1][4][0], i[0]) for i in sa])
    spearman = np.array([(i[1][5][0], i[0]) for i in sa])
    # pickle.dump(sa, open(f'./auxiliaryExp/ConfidenceOfGTextSyn/\
    #                      cofidenceExp_min{mse.min()}_max{mse.max()}_mean{mse.mean()}\
    #                         _var{mse.var()}_std{mse.std()}__14_32_33.pkl', 'wb'))


if __name__ == "__main__":
    modelConfidence()