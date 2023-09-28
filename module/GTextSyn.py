import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import MultiHeadAttention, DrugGraphConv


class GTextSyn(nn.Module):
    def __init__(self, params):
        super(GTextSyn, self).__init__()

        self.emb_dim = 37
        self.dropout = params["dropout"]

        self.drug_graph_conv = DrugGraphConv(74, self.emb_dim)

        self.cell_MLP = nn.Sequential(
            nn.Linear(956, 3072),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(3072, 1024),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(1024, self.emb_dim),
            nn.ReLU()
        )

        self.bilstm = nn.LSTM(37, 37, num_layers=2, batch_first=True, bidirectional=True, dropout=self.dropout)
        
        self.attention = MultiHeadAttention(74, 74, 2)
        self.fc_in = nn.Linear(222, 256)
        self.fc_out = nn.Linear(256, 1)


    def forward(self, data):

        drug = data[0]
        cell = data[1]
        
        drug_emb = self.drug_graph_conv(drug)
        cell = F.normalize(cell, dim=1)
        cell_emb = self.cell_MLP(cell).view(-1, 1, self.emb_dim)
        
        sequence = torch.cat((drug_emb, cell_emb), dim=1)

        output, _ = self.bilstm(sequence)

        out, outw = self.attention(output, return_attention=True)

        out = F.relu(self.fc_in(out.view(out.shape[0], -1)))
        out = self.fc_out(out)

        return out
