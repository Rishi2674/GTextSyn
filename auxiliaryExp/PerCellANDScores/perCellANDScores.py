
import sys
sys.path.insert(0, sys.path[0]+"/../../")
from utils import metrics, dataset, DataLoader, collate_merg, evaluate
import numpy as np
import torch
import pickle
import dgl
from dgllife.utils import smiles_to_bigraph, CanonicalAtomFeaturizer
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt

SEED=5
node_featurizer = CanonicalAtomFeaturizer(atom_data_field='h')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load("./auxiliaryExp/ONEIL.pth", map_location=torch.device('cpu'))
model.to(device)


def preData():
    # 读取药物组合数据
    comb = pd.read_csv(f'./data/ONEIL.csv')
    comb = comb[['drug_row', 'drug_col', 'cell_line_name', 'tissue_name', 'synergy_loewe']]
    
    # 离群值清理
    comb = comb[comb['synergy_loewe']<400]

    comb = comb.sample(frac=1, random_state=SEED)

    # 读取Drug数据
    smiles = pd.read_csv('./data/drugs.csv')
    smiles = smiles[['dname', 'smiles']]
    _d = list(set(list(comb['drug_row'])+list(comb['drug_col'])))
    smiles = smiles[smiles['dname'].isin(_d)]
    smiles.set_index('dname', inplace=True)

    # 读取Cell数据
    cells = pd.read_csv('./data/cells.csv')
    _c = list(set(comb['cell_line_name']))
    cells = cells[cells['name'].isin(_c)]
    cells.set_index('name', inplace=True)
    mark = []
    # 读取landmark Gene
    landmark = pd.read_csv('./data/raw/landmarkGene.txt', sep='\t')
    landmark = list(landmark['Symbol'])

    '''
    这些基因信息在CCLE_expression_full中不存在, 删除
    'PAPD7', 'HDGFRP3', 'AARS', 'TMEM2', 'TMEM5', 'SQRDL', 'H2AFV', 'KIAA0907', 'HIST2H2BE', 'KIAA0355', 'IKBKAP', 'TSTA3', 'TMEM110', 'WRB', 'FAM69A', 'FAM57A', 'ATP5S', 'NARFL', 'KIF1BP', 'HN1L', 'EPRS', 'HIST1H2BK'
    '''
    exclude = ['PAPD7', 'HDGFRP3', 'AARS', 'TMEM2', 'TMEM5', 'SQRDL', 'H2AFV', 'KIAA0907', 'HIST2H2BE', 'KIAA0355', 'IKBKAP', 'TSTA3', 'TMEM110', 'WRB', 'FAM69A', 'FAM57A', 'ATP5S', 'NARFL', 'KIF1BP', 'HN1L', 'EPRS', 'HIST1H2BK']
    mark = list(set(landmark)-set(exclude))

    mark.sort()
    cells = cells[mark]

    d_graph = {}
    cell = {}
    score = {}

    with tqdm(total=len(comb)) as tqq:
        for item in comb.itertuples():
            smileA = smiles.loc[item.drug_row]['smiles']
            smileB = smiles.loc[item.drug_col]['smiles']
            cellGene = cells.loc[item.cell_line_name].values

            if smileA not in d_graph.keys():
                gA = smiles_to_bigraph(smileA, node_featurizer=node_featurizer)
                gA = dgl.add_self_loop(gA)
                d_graph[smileA] = gA
            else:
                gA = d_graph[smileA]

            if smileB not in d_graph.keys():
                gB = smiles_to_bigraph(smileB, node_featurizer=node_featurizer)
                gB = dgl.add_self_loop(gB)
                d_graph[smileB] = gB
            else:
                gB = d_graph[smileB]

            if not cell.get(item.cell_line_name):
                cell[item.cell_line_name] = []
            cell[item.cell_line_name].append(((gA, gB, cellGene), item.synergy_loewe))
            
            if not score.get(item.synergy_loewe//5):
                score[item.synergy_loewe//5] = []
            score[item.synergy_loewe//5].append(((gA, gB, cellGene), item.synergy_loewe))

            tqq.update(1)

    pickle.dump(cell, open('./auxiliaryExp/PerCellANDScores/percell.pkl', 'wb'))
    pickle.dump(score, open('./auxiliaryExp/PerCellANDScores/perscore.pkl', 'wb'))



def per_type(types):
    model.eval()
    data = pickle.load(open('./auxiliaryExp/PerCellANDScores/' + f"per{types}.pkl", 'rb'))
    res = {}
    with torch.no_grad():
        for k, (typ, da) in enumerate(data.items()):
            trues = []
            preds = []

            _d = dataset([i[0] for i in da], [i[1] for i in da], device)
            dataL = DataLoader(_d, batch_size=256, shuffle=False, collate_fn=lambda x: collate_merg(x, device))

            with tqdm(total = len(dataL)) as tqq:
                for dAB, dBA, c, y in dataL:
                    pred1 = model((dAB, c))
                    pred2 = model((dBA, c))
                    trues.append(y)
                    preds.append((pred1 + pred2) / 2)
                    tqq.set_description(f"{k+1}/{len(data)}")
                    tqq.update(1)
            if len(trues) > 1:
                offset = trues[-2].shape[0] - trues[-1].shape[0]
                trues[-1].resize_(trues[-2].shape)
                preds[-1].resize_(preds[-2].shape)
                trues = torch.stack(trues, 0).view(-1).cpu()
                preds = torch.stack(preds, 0).view(-1).cpu()
                if offset > 0 :
                    trues = trues[:-offset]
                    preds = preds[:-offset]
            else:
                trues = trues[0].cpu()
                preds = preds[0].cpu()

            mse = metrics.mse(trues, preds)
            pearson = metrics.pearson(trues, preds)
            res[typ] = (mse, pearson, len(trues))
    
    t = [(i[1][0], i[0], i[1][2]) for i in res.items()]
    if types == "cell":
        t.sort(reverse=True)
    elif types == "score":
        t.sort(key=lambda x : x[1])

    label = [i[1] for i in t]
    syn = [i[0] for i in t]
    nums = [i[2] for i in t]
    pear = [res[i][1][0] for i in label]

    fig = plt.figure(dpi=800)
    plt.rcParams['xtick.direction'] = 'in'
 
    bar_width = 0.75  # 条形宽度
    index_y1 = np.array(list(range(0, 2*len(label), 2)))  # y1条形图的横坐标
    index_y2 = index_y1 + bar_width  # y2条形图的横坐标

    ax = fig.add_subplot(111)

    # 使用两次 bar 函数画出两组条形图
    ax.bar(index_y1, height=syn, width=bar_width, color='#1E90FF', label='MSE')
    ax.set_ylabel('MSE')

    if types == "cell":
        ax2 = ax.twinx()
        ax2.bar(index_y2, height=pear, width=bar_width, color='#FF7F24', label='Pearson')
        ax2.set_ylim(0, 1.2)
        ax2.set_ylabel('Pearson')
        ax.set_xticks(index_y1 + bar_width/2, label, rotation=45, ha='right', fontsize=6)
    elif types == "score":
        ax2 = ax.twinx()
        ax2.bar(index_y2, height=nums, width=bar_width, color='#FF7F24', label='Nums')
        ax2.set_ylim(0, 5000)
        ax2.set_ylabel('Nums')
        labels = [f"[{int(5*i)},{int(5*(i+1))})" for i in label]
        ax.set_xticks(index_y1 + bar_width/2, labels, rotation=45, ha='right', fontsize=6)

    fig.legend(loc=1, bbox_to_anchor=(1,1), bbox_transform=ax.transAxes)

    if types == "cell":
        plt.title('Predictive performance of model on specific cell line')
        plt.savefig('./auxiliaryExp/PerCellANDScores/percell_mse_pearson.png')
    elif types == "score":
        plt.title('Predictive performance of model on specific scores interval')
        plt.savefig('./auxiliaryExp/PerCellANDScores/perscore_mse_nums.png')

    plt.close()




if __name__ == "__main__":
    preData()
    per_type("cell")
    per_type("score")