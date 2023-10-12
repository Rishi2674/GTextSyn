
import sys
sys.path.insert(0, sys.path[0]+"/../../")
from utils import evaluate
import pandas as pd
import torch
from dgllife.utils import smiles_to_bigraph, CanonicalAtomFeaturizer
import dgl


SEED=5
node_featurizer = CanonicalAtomFeaturizer(atom_data_field='h')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load("./auxiliaryExp/ONEIL.pth", map_location=torch.device('cpu'))
model.to(device)


def novelComb(data, name):
    (label, pred), _ = evaluate(model, data, device)
    pred = pred.sort(descending=True)

    print("drugA\tdrugB\tcell-line\tsynergist")
    for i in list(range(20)):
        _i = pred.indices[i].item()
        print(f"{name[_i][0]}\t{name[_i][1]}\t{name[_i][2]}\t{pred.values[i].item()}")


def get_novelComb_dataset():
    comb = pd.read_csv(f'./data/ONEIL.csv')
    comb = comb[['drug_row', 'drug_col', 'cell_line_name', 'synergy_loewe']]
    
    # Outlier cleaning
    comb = comb[comb['synergy_loewe']<400]

    comb = comb.sample(frac=1, random_state=SEED)

    smiles = pd.read_csv('./data/drugs.csv')
    smiles = smiles[['dname', 'smiles']]
    _d = list(set(list(comb['drug_row'])+list(comb['drug_col'])))
    smiles = smiles[smiles['dname'].isin(_d)]
    smiles.set_index('dname', inplace=True)

    cells = pd.read_csv('./data/cells.csv')
    _c = list(set(comb['cell_line_name']))
    cells = cells[cells['name'].isin(_c)]
    cells.set_index('name', inplace=True)

    mark = []
    landmark = pd.read_csv('./data/raw/landmarkGene.txt', sep='\t')
    landmark = list(landmark['Symbol'])
    exclude = ['PAPD7', 'HDGFRP3', 'AARS', 'TMEM2', 'TMEM5', 'SQRDL', 'H2AFV', 'KIAA0907', 'HIST2H2BE', 'KIAA0355', 'IKBKAP', 'TSTA3', 'TMEM110', 'WRB', 'FAM69A', 'FAM57A', 'ATP5S', 'NARFL', 'KIF1BP', 'HN1L', 'EPRS', 'HIST1H2BK']
    '''
    The data for the following genes does not exist in CCLE_expression_full, so it is deleted.
    'PAPD7', 'HDGFRP3', 'AARS', 'TMEM2', 'TMEM5', 'SQRDL', 'H2AFV', 'KIAA0907', 'HIST2H2BE', 'KIAA0355', 'IKBKAP', 'TSTA3', 'TMEM110', 'WRB', 'FAM69A', 'FAM57A', 'ATP5S', 'NARFL', 'KIF1BP', 'HN1L', 'EPRS', 'HIST1H2BK'
    '''
    mark = list(set(landmark)-set(exclude))
    mark.sort()
    cells = cells[mark]

    cs = list(set(comb['cell_line_name']))

    olddcs = [(i[1]['drug_row'], i[1]['drug_col']) for i in comb.iterrows()]
    olddcs = list(set(olddcs))

    ds = list(set(list(comb['drug_row']) + list(comb['drug_col'])))
    ds.sort()
    newdcs = []

    for i in list(range(len(ds))):
        for j in list(range(i+1, len(ds))):
            if (ds[i], ds[j]) not in olddcs:
                newdcs.append((ds[i], ds[j]))

    d_graph = {}
    save_set = []
    name_set = []

    _i = 1
    for _dc in newdcs:
        for _c in cs:
            smileA = smiles.loc[_dc[0]]['smiles']
            smileB = smiles.loc[_dc[1]]['smiles']
            cellGene = cells.loc[_c].values

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

            save_set.append(((gA, gB, cellGene), _i))
            name_set.append((_dc[0], _dc[1], _c, _i))
            _i += 1
    
    return save_set, name_set



if __name__ == "__main__":
    save_set, name_set = get_novelComb_dataset()
    novelComb(save_set, name_set)