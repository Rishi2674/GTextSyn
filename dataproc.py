import pickle
import pandas as pd
from dgllife.utils import smiles_to_bigraph, CanonicalAtomFeaturizer
import dgl

SEED=5
node_featurizer = CanonicalAtomFeaturizer(atom_data_field='h')


def getDrugs():
    with open("./data/raw/drugs_drugcomb.txt", "r") as f:
        text = f.read()

    text = text.replace('null', 'None')
    tlist = eval(text)
    data = pd.DataFrame(tlist)
    df_drug = data[~data['smiles'].isin(['NULL'])]

    # Keep the drug that appears for the first time and delete the others.
    drugs = df_drug.groupby(['dname']).first().reset_index()

    allDrug = list(drugs['dname'])
    dropDrug = []
    for dname in allDrug:
        smiles = drugs[drugs['dname']==dname]['smiles'].values[0]
        g = smiles_to_bigraph(smiles, node_featurizer=node_featurizer)
        if g is None:
            dropDrug.append(dname)
    drugs = drugs[~drugs['dname'].isin(dropDrug)]

    drugs.set_index('id', inplace=True)
    drugs.sort_index(inplace=True)
    drugs.to_csv("./data/drugs.csv")



def getCells():
    with open("./data/raw/cells_drugcomb.txt", "r") as f:
        text = f.read()

    text = text.replace('null', 'None')
    tlist = eval(text)
    data = pd.DataFrame(tlist)
    data.set_index('id', inplace=True)
    data = data[~data['depmap_id'].isin(['NA'])]
    # Special value processing: The original depmap_id value was "ACH-000833; ACH-001189", which was modified to "ACH-000833"
    data.loc[1288, 'depmap_id'] = 'ACH-000833'
    df_cell = data[['name', 'depmap_id']]

    df_cellexpression = pd.read_csv('./data/raw/CCLE_expression_full.csv')
    df_cellexpression.rename(columns={'Unnamed: 0':'cellName'}, inplace=True)

    cellExpression = pd.merge(df_cell, df_cellexpression, how='inner', left_on='depmap_id', right_on='cellName')
    # Drop non-data columns
    cellExpression.drop(['depmap_id', 'cellName'], axis=1, inplace=True)
    # Drop columns with all zeros.
    cellExpression = cellExpression.loc[:, (cellExpression!=0).any(axis=0)]

    cellExpression.set_index('name', inplace=True)
    cellExpression.columns = cellExpression.columns.str.split(" \(").str[0]
    cellExpression.to_csv("./data/cells.csv")


def getDrugCombs(study):
    df_synergy = pd.read_csv('./data/raw/summary_v_1_5.csv')

    synergy = df_synergy[['drug_row', 'drug_col', 'cell_line_name', 'study_name',
        'tissue_name', 'synergy_zip', 'synergy_loewe', 'synergy_hsa', 'synergy_bliss']]

    # Drop samples for non-cancer research
    synergy = synergy[~synergy['study_name'].isin(['TOURET','GORDON','ELLINGER','MOTT','NCATS_SARS-COV-2DPI','BOBROWSKI','DYALL'])]
    # Drop single drug samples
    synergy = synergy[~synergy['drug_col'].isnull()]

    synergy = synergy[synergy['study_name'].isin(study.split('_'))]
    assert len(synergy)>0, "Study name was entered incorrectly."

    # Drop non-numeric rows in synergy_loewe
    synergy = synergy[~pd.to_numeric(synergy['synergy_loewe'] ,errors='coerce').isnull()]
    synergy['synergy_loewe'] = synergy['synergy_loewe'].astype('float64')
    # Drop rows without cell data
    _cell = pd.read_csv('./data/cells.csv')
    cells = list(_cell['name'])
    del _cell
    synergy = synergy[synergy['cell_line_name'].isin(cells)]
    # Drop rows without drug data
    _drug = pd.read_csv('./data/drugs.csv')
    drugs = list(_drug['dname'])
    del _drug
    synergy = synergy[synergy['drug_row'].isin(drugs)]
    synergy = synergy[synergy['drug_col'].isin(drugs)]

    mask = list(map(lambda x, y: x>y, synergy['drug_row'].astype(str), synergy['drug_col'].astype(str)))
    synergy.loc[mask, 'drug_row'], synergy.loc[mask, 'drug_col'] = synergy.loc[mask, 'drug_col'], synergy.loc[mask, 'drug_row']

    merge_data = synergy[['drug_row', 'drug_col', 'cell_line_name', 'study_name', 'tissue_name']]
    merge_data = merge_data.drop_duplicates(subset = ['drug_row', 'drug_col', 'cell_line_name'])
    merge_data.set_index(['drug_row', 'drug_col', 'cell_line_name'], inplace=True)

    comb = synergy.groupby(['drug_row', 'drug_col', 'cell_line_name']).agg('mean')

    data = pd.merge(merge_data, comb, left_index=True, right_index=True)
    data.reset_index(inplace=True)

    data.to_csv(f'./data/{study}.csv', index=False)



def get_train_test(study, ratio):
    comb = pd.read_csv(f'./data/{study}.csv')
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

    datas = []
    datas.append(comb[0: int(len(comb)*ratio)])
    datas.append(comb[int(len(comb)*ratio):])

    d_graph = {}

    for l, da in zip(['train','test'], datas):
        save_set = []

        for item in da.itertuples():
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

            save_set.append(((gA, gB, cellGene), item.synergy_loewe))

        with open(f'./data/l.pkl', 'wb') as f:
            pickle.dump(save_set, f)



if __name__ == "__main__":
    getDrugs()
    getCells()
    getDrugCombs("ONEIL")
    get_train_test("ONEIL", 0.9)