import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class MCEC(nn.Module):
    
    def __init__(self, configs):
        super(MCEC, self).__init__()
        
        self.hidden_dim = configs.MC_hidden_dim
        self.norm_methood = configs.MC_norm_methood
        self.dropout = configs.MC_dropout_rate
        self.act = configs.MC_activation
        
        
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        
        if self.norm_methood == 'BN':
            self.enforce_norm_layer = nn.BatchNorm1d(self.seq_len)
            self.correct_norm_layer = nn.BatchNorm1d(self.pred_len)
        if self.norm_methood == 'LN':
            self.enforce_norm_layer = nn.LayerNorm(self.enc_in)
            self.correct_norm_layer = nn.LayerNorm(self.enc_in)
        
        if self.act == 'relu':
            self.activation = nn.ReLU()
        if self.act == 'gelu':
            self.activation = nn.GELU()
        if self.act == 'sigmoid':
            self.activation = nn.Sigmoid()
        if self.act == 'tanh':
            self.activation = nn.Tanh()
            
        # Initialize linear layers
        self.enforce_linear1 = nn.Linear(self.enc_in, self.hidden_dim)
        # self.enforce_linear2 = nn.Linear(self.hidden_dim, self.enc_in)
        self.correct_linear1 = nn.Linear(self.enc_in, self.hidden_dim)
        self.correct_linear2 = nn.Linear(self.hidden_dim, self.enc_in)
            
    def enforce(self,x):
        
        if self.norm_methood:
            x = self.enforce_norm_layer(x)
        
        x = nn.ReLU(self.enforce_linear1(x))
        
        x = nn.Dropout(p=self.dropout)(x)
        
        #x = self.enforce_linear2(x)
        
        return x
        
    def correct(self,x):
        
        if self.norm_methood:
            x = self.correct_norm_layer(x)
        
        x = self.activation(self.correct_linear1(x))
        
        x = nn.Dropout(p=self.dropout)(x)
        
        x = self.correct_linear2(x)
        
        return x
        
            
            
