import sys
sys.path.insert(0, sys.path[0]+"/../../")
from utils import metrics, dataset, DataLoader, collate_merg
import numpy as np
import torch
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load("./auxiliaryExp/ONEIL.pth", map_location=torch.device('cpu'))
model.to(device)



def inputPermutation():
    model.eval()
    trues = []
    preds = []
    predab = []
    predba = []

    data = pickle.load(open("./data/ONEIL_test.pkl", 'rb'))

    data = dataset([i[0] for i in data], [i[1] for i in data], device)
    dataL = DataLoader(data, batch_size=256, shuffle=False, collate_fn=lambda x: collate_merg(x, device))


    with torch.no_grad():
        with tqdm(total = len(dataL)) as tqq:
            for i, (dAB, dBA, c, y) in enumerate(dataL):
                pred1 = model((dAB, c))
                pred2 = model((dBA, c))
                trues.append(y)
                predab.append(pred1)
                predba.append(pred2)
                preds.append((pred1 + pred2) / 2)
                tqq.set_description(f"Item:{i}")
                tqq.update(1)
                
        offset = trues[-2].shape[0] - trues[-1].shape[0]
        trues[-1].resize_(trues[-2].shape)
        preds[-1].resize_(preds[-2].shape)
        predab[-1].resize_(predab[-2].shape)
        predba[-1].resize_(predba[-2].shape)
        trues = torch.stack(trues, 0).view(-1).cpu()
        preds = torch.stack(preds, 0).view(-1).cpu()
        predab = torch.stack(predab, 0).view(-1).cpu()
        predba = torch.stack(predba, 0).view(-1).cpu()
        if offset > 0 :
            trues = trues[:-offset]
            preds = preds[:-offset]
            predab = predab[:-offset]
            predba = predba[:-offset]

    trues = np.array(trues)
    preds = np.array(preds)
    predab = np.array(predab)
    predba = np.array(predba)
    t = (trues - trues.min()) / (trues.max() - trues.min())
    p = (preds - preds.min()) / (preds.max() - preds.min())
    a = (predab - predab.min()) / (predab.max() - predab.min())
    b = (predba - predba.min()) / (predba.max() - predba.min())

    # input permutation
    plt.figure(num=0)
    plt.plot([-80, 20], [-80, 20], color='r')
    plt.scatter(predab, predba, s=2)
    plt.xlabel('DrugA-DrugB-cell line')
    plt.ylabel('DrugB-DrugA-cell line')
    plt.title('Input Permutation')
    plt.savefig('./auxiliaryExp/InputPermutation/'\
                 + f'ontest_inputPermutation_pearson{metrics.pearson(predab, predba)[0]}.png')
    plt.close()


if __name__ == "__main__":
    inputPermutation()