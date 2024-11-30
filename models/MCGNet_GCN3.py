import torch
import torch.nn as nn
from models.miniGCN import GCN
import torch.nn.functional as F
import numpy as np


class Fusion(nn.Module):
    def __init__(self, input_channel=1, hidden=48, adj=None):
        super(Fusion, self).__init__()
        
        with open('/media/cygzz/data/rtao/projects/MCG-NC/data/data001/ADJ_matrix.csv') as f:
            adj_matrix = torch.tensor(np.loadtxt(f, delimiter=",").astype(np.float32)).unsqueeze(0).unsqueeze(0)
        f.close()
        
        self.trans_layers = GCN(n_embd=hidden*6, n_layer=2, n_head=4, adj=adj)
        
        # self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden*3, 6, 6)) #  b, seq_len, c, w, h 
        # self.convlet = nn.Conv1d(9, 1, 1, 1, 0, bias=False)
        self.is_ischemia = MLP(hidden*6, hidden, 1, 3) # bce
        
        # SPECT
        self.trans_xin_jian = GCN(n_embd=hidden*6, n_layer=2, n_head=4, adj=adj)
        self.xin_jian_spect = MLP(hidden*6, hidden, 1, 3) # bce
        self.trans_qian_bi = GCN(n_embd=hidden*6, n_layer=2, n_head=4, adj=adj)
        self.qian_bi_spect = MLP(hidden*6, hidden, 1, 3) # bce
        self.trans_jiange_bi = GCN(n_embd=hidden*6, n_layer=2, n_head=4, adj=adj)
        self.jiange_bi_spect = MLP(hidden*6, hidden, 1, 3) # bce
        self.trans_xia_bi = GCN(n_embd=hidden*6, n_layer=2, n_head=4, adj=adj)
        self.xia_bi_spect = MLP(hidden*6, hidden, 1, 3) # bce
        self.trans_ce_bi = GCN(n_embd=hidden*6, n_layer=2, n_head=4, adj=adj)
        self.ce_bi_spect = MLP(hidden*6, hidden, 1, 3) # bce
        
        # CTA
        self.trans_LAD = GCN(n_embd=hidden*6, n_layer=2, n_head=4, adj=adj)
        self.LAD_cta = MLP(hidden*6, hidden, 1, 3) # bce
        self.trans_LCX = GCN(n_embd=hidden*6, n_layer=2, n_head=4, adj=adj)
        self.LCX_cta = MLP(hidden*6, hidden, 1, 3) # bce
        self.trans_RCA = GCN(n_embd=hidden*6, n_layer=2, n_head=4, adj=adj)
        self.RCA_cta = MLP(hidden*6, hidden, 1, 3) # bce
                      

    def forward(self, feat_spect, feat_cta, feat_ischemia):   
        # # xin jian,	qian bi,	jiange bi,	xia bi,	ce bi,	Lad,	rca,	lcx 
        feat_cta = F.interpolate(feat_cta, size=feat_spect.shape[-1])  
        feat_ischemia = F.interpolate(feat_ischemia, size=feat_spect.shape[-1])  
        feat_fusion = torch.cat((feat_ischemia, feat_spect , feat_cta), dim=1)
        feat_fusion = self.trans_layers(feat_fusion) # 2, 26, 166, 6, 6
        # main task
        fusion_ischemia = feat_fusion[:,0,:].unsqueeze(1)
        is_ischemia = self.is_ischemia(fusion_ischemia).squeeze(-1)
        
        
         # subs tasks
        feat_lad = self.trans_LAD(torch.cat((feat_ischemia, feat_spect , feat_cta), dim=1))
        feat_lad = feat_lad[:,6,:].unsqueeze(1)
        
        feat_rca = self.trans_RCA(torch.cat((feat_ischemia, feat_spect , feat_cta), dim=1))
        feat_rca = feat_rca[:,7,:].unsqueeze(1)
        
        feat_lcx = self.trans_LCX(torch.cat((feat_ischemia, feat_spect , feat_cta), dim=1))
        feat_lcx = feat_lcx[:,8,:].unsqueeze(1)
        
         # subs tasks
        feat_xin_jian = self.trans_xin_jian(torch.cat((feat_ischemia, feat_spect , feat_cta), dim=1))
        feat_xin_jian = feat_xin_jian[:,1,:].unsqueeze(1)
        
        feat_qian_bi = self.trans_qian_bi(torch.cat((feat_ischemia, feat_spect , feat_cta), dim=1))
        feat_qian_bi = feat_qian_bi[:,2,:].unsqueeze(1)
        
        feat_jiange_bi = self.trans_jiange_bi(torch.cat((feat_ischemia, feat_spect , feat_cta), dim=1))
        feat_jiange_bi = feat_jiange_bi[:,3,:].unsqueeze(1)
        
        feat_xia_bi = self.trans_xia_bi(torch.cat((feat_ischemia, feat_spect , feat_cta), dim=1))
        feat_xia_bi = feat_xia_bi[:,4,:].unsqueeze(1)
        
        feat_ce_bi = self.trans_ce_bi(torch.cat((feat_ischemia, feat_spect , feat_cta), dim=1))
        feat_ce_bi = feat_ce_bi[:,5,:].unsqueeze(1)
        
        # 
        xin_jian = self.xin_jian_spect(feat_xin_jian).squeeze(-1)
        qian_bi = self.qian_bi_spect(feat_qian_bi).squeeze(-1)
        jiange_bi = self.jiange_bi_spect(feat_jiange_bi).squeeze(-1)
        xia_bi = self.xia_bi_spect(feat_xia_bi).squeeze(-1)
        ce_bi = self.ce_bi_spect(feat_ce_bi).squeeze(-1)
        
        # 
        RCA = self.RCA_cta(feat_rca).squeeze(-1)
        LAD = self.LAD_cta(feat_lad).squeeze(-1)
        LCX = self.LCX_cta(feat_lcx).squeeze(-1)  
        
 
        output_cta = {'LAD':LAD, 'LCX':LCX, 'RCA':RCA}
        output_spect = {'xin_jian':xin_jian, 'qian_bi':qian_bi,'jiange_bi': jiange_bi, 'xia_bi':xia_bi, 'ce_bi':ce_bi} 
        ouput_dict = {'is_ischemia':is_ischemia, 'output_spect':output_spect, 'output_cta':output_cta}
        return ouput_dict


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def test():
    import numpy as np
    with open('/media/cygzz/data/rtao/projects/MCG-NC/data/data001/ADJ_matrix2.csv') as f:
        adj_matrix = torch.tensor(np.loadtxt(f, delimiter=",").astype(np.float32)).unsqueeze(0).unsqueeze(0)
    f.close()
    net = Fusion(adj=adj_matrix)
    x = torch.randn([4, 1, 6, 6, 100])
    n_parameters = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    net(torch.randn([4, 5, 288]), torch.randn([4, 3, 98]), torch.randn([4, 1, 98]))  # 1, 1152, 150



# torch.Size([2, 1, 6, 6, 600]) torch.Size([2]) torch.Size([2, 1, 5, 600]) torch.Size([2, 4])
# 30,000,000   5,144,974
# test()

















