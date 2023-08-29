import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def dice_metric(input, label):
    #:param input: (N, C, H, W, D)  # bg in channel 0
    eps = torch.Tensor([1e-3]).cuda()
    dices = []
    # foreground_dice
    for i in range(input.shape[1]):
        input_f = input[:,i,:,:,:]
        label_f = label[:,i,:,:,:]
        input_f = input_f.flatten()
        label_f = label_f.flatten()
        intersect = torch.dot(input_f, label_f)
        input_sum = torch.sum(input_f)
        label_sum = torch.sum(label_f)
        union = input_sum + label_sum
        dice = ((2 * intersect + eps) / (union + eps)).item()
        dices.append(dice)
    return dices

softmax_helper = lambda x: F.softmax(x, 1)

class Loss(nn.Module):
    def __init__(self, old_classes=None, bg_ratio=0.1):
        super(Loss, self).__init__()
        self.bg_ratio = bg_ratio
        self.old_classes = old_classes
        self.apply_nonlin=softmax_helper
        self.alpha=0.5
        self.gamma=2
        self.smooth=1e-5
        self.balance_index=0
        self.size_average=True


    def get_l2(self, input, label, index):
        #:param input: (N, C, H, W, D)  # bg in channel 0
        mse = nn.MSELoss()
        # foreground_dice
        input = input[:,index:,:,:,:]
        # label = label[:,1:,:,:,:]
        loss = mse(input, label)
        # background_dice
        # input_b = input[:,0,:,:,:]
        # label_b = label[:,0,:,:,:]
        # loss_b = mse(input_b, label_b)
        # # assemble
        # loss = loss_f*(1-self.bg_ratio) + self.bg_ratio*loss_b
        return loss
        # loss(pred_hmaps, hmaps, cls, box)

        
    def get_l1(self, input, label, index=None):
        #:param input: (N, C, H, W, D)  # bg in channel 0
        l1 = nn.L1Loss()
        # input = input[:,index:,:,:,:]
        # label = label[:,1:,:,:,:]
        loss = l1(input, label)
        return loss


    def get_ce(self, input, label):
        loss = F.cross_entropy(input, label)
        return loss


    def get_bce(self, input, label): # no sigmoid
        #:param input: (N, C, H, W, D)  # bg in channel 0
        bce = nn.BCEWithLogitsLoss()
        loss = bce(input, label.float())
        return loss
    
    
    def get_KD(self, inputs, targets, mask=None): 
        inputs = inputs.narrow(1, 0, targets.shape[1])

        outputs = torch.log_softmax(inputs, dim=1)
        labels = torch.softmax(targets, dim=1)

        loss = (outputs * labels).mean(dim=1)

        if mask is not None:
            loss = loss * mask.float()

        outputs = -torch.mean(loss)
        return outputs
 
 
    def forward_basic(self, output, gt):
        # pred: logits
        is_ischemia_lss = self.get_bce(output['is_ischemia'].squeeze(1), gt['is_ischemia'])        
        loss_dict = {'is_ischemia':is_ischemia_lss}
        return loss_dict

    def forward_spect(self, output, gt):
        # pred: logits
        is_ischemia_lss = self.get_bce(output['is_ischemia'].squeeze(1), gt['is_ischemia']) 
        # SPECT
        spect_idx = torch.where(gt['SPECT_idx']==1)[0]
        output_spect = output['output_spect']      
        xin_jian_lss = self.get_bce(output_spect['xin_jian'].squeeze(1)[spect_idx], gt['xin_jian'][spect_idx])
        qian_bi_lss = self.get_bce(output_spect['qian_bi'].squeeze(1)[spect_idx], gt['qian_bi'][spect_idx])
        jiange_bi_lss = self.get_bce(output_spect['jiange_bi'].squeeze(1)[spect_idx], gt['jiange_bi'][spect_idx])
        xia_bi_lss = self.get_bce(output_spect['xia_bi'].squeeze(1)[spect_idx], gt['xia_bi'][spect_idx])
        ce_bi_lss = self.get_bce(output_spect['ce_bi'].squeeze(1)[spect_idx], gt['ce_bi'][spect_idx])                
        
        loss_dict = {'is_ischemia':is_ischemia_lss, 
                     'xin_jian':xin_jian_lss, 'qian_bi':qian_bi_lss, 'jiange_bi':jiange_bi_lss,
                      'xia_bi':xia_bi_lss, 'ce_bi':ce_bi_lss,
                     }
        
        return loss_dict


    def forward_joint(self, output, gt):
        # pred: logits
        is_ischemia_lss = self.get_bce(output['is_ischemia'].squeeze(1), gt['is_ischemia'])       
        # SPECT
        spect_idx = torch.where(gt['SPECT_idx']==1)[0]
        output_spect = output['output_spect']      
        xin_jian_lss = self.get_bce(output_spect['xin_jian'].squeeze(1)[spect_idx], gt['xin_jian'][spect_idx])
        qian_bi_lss = self.get_bce(output_spect['qian_bi'].squeeze(1)[spect_idx], gt['qian_bi'][spect_idx])
        jiange_bi_lss = self.get_bce(output_spect['jiange_bi'].squeeze(1)[spect_idx], gt['jiange_bi'][spect_idx])
        xia_bi_lss = self.get_bce(output_spect['xia_bi'].squeeze(1)[spect_idx], gt['xia_bi'][spect_idx])
        ce_bi_lss = self.get_bce(output_spect['ce_bi'].squeeze(1)[spect_idx], gt['ce_bi'][spect_idx])           
        # CTA
        cta_idx = torch.where(gt['CTA_idx']==1)[0] 
        output_cta = output['output_cta']
        LAD_lss = self.get_bce(output_cta['LAD'].squeeze(1)[cta_idx], gt['LAD'][cta_idx])
        LCX_lss = self.get_bce(output_cta['LCX'].squeeze(1)[cta_idx], gt['LCX'][cta_idx])
        RCA_lss = self.get_bce(output_cta['RCA'].squeeze(1)[cta_idx], gt['RCA'][cta_idx])
        
        loss_dict = {'is_ischemia':is_ischemia_lss, 
                      'xin_jian':xin_jian_lss, 'qian_bi':qian_bi_lss, 'jiange_bi':jiange_bi_lss,
                      'xia_bi':xia_bi_lss, 'ce_bi':ce_bi_lss,
                      'LAD':LAD_lss, 'LCX':LCX_lss, 'RCA':RCA_lss 
                     }
        return loss_dict
    
    def forward_cta(self, output, gt):
        # pred: logits
        is_ischemia_lss = self.get_bce(output['is_ischemia'].squeeze(1), gt['is_ischemia'])       
        
        # CTA
        cta_idx = torch.where(gt['CTA_idx']==1)[0] 
        output_cta = output['output_cta']
        LAD_lss = self.get_bce(output_cta['LAD'].squeeze(1)[cta_idx], gt['LAD'][cta_idx])
        LCX_lss = self.get_bce(output_cta['LCX'].squeeze(1)[cta_idx], gt['LCX'][cta_idx])
        RCA_lss = self.get_bce(output_cta['RCA'].squeeze(1)[cta_idx], gt['RCA'][cta_idx])
        
        loss_dict = {'is_ischemia':is_ischemia_lss, 
                      'LAD':LAD_lss, 'LCX':LCX_lss, 'RCA':RCA_lss 
                     }
        return loss_dict

 