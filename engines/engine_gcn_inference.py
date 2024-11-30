import math
import os
import sys
from typing import Iterable
import numpy as np
import torch, pickle
from sklearn.metrics import f1_score
import util.misc as utils
import torch.nn.functional as F
from pathlib import Path
from apex import amp
from losses.loss0930 import Loss
from skimage.metrics import structural_similarity as ssim
from pathlib import Path
import time
import sklearn as sk
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc, accuracy_score
from engines.metrics_with_CI import metrics_CI

def cal_metrics(pred, gt, optimal_threshold=None):
    if optimal_threshold != None:  # customized threshold
        fpr, tpr, threshold = roc_curve(gt, pred)
        auc_value = auc(fpr, tpr)
        pred[pred>=optimal_threshold] = 1
        pred[pred<optimal_threshold] = 0
        TN, FP, FN, TP = confusion_matrix(gt, pred, labels=[0, 1]).ravel() #tn, fp, fn, tp
    else:    
        fpr, tpr, threshold = roc_curve(gt, pred)
        auc_value = auc(fpr, tpr)
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = threshold[optimal_idx]
        
        pred[pred>=optimal_threshold] = 1
        pred[pred<optimal_threshold] = 0        
        TN, FP, FN, TP = confusion_matrix(gt, pred, labels=[0, 1]).ravel() #tn, fp, fn, tp
    
    sens, spec, sens_CI, spec_CI, NPV, NPV_CI, PPV, PPV_CI, acc, acc_CI, F1, F1_CI\
        = metrics_CI(TP, FP, FN, TN, alpha=0.95) #TP, FP, FN, TN,
    results = {'acc':acc, 'sens':sens, 'spec':spec, 'auc':auc_value, 'f1':F1, 'NPV':NPV, 'PPV':PPV,
               'sens_CI':sens_CI, 'spec_CI':spec_CI, 'NPV_CI':NPV_CI, 'PPV_CI':PPV_CI, 'acc_CI':acc_CI,'F1_CI':F1_CI, 
               'opt_point':optimal_threshold}
    return results


def evaluate_loop(model_spect, model_cta, model_ischemia, model_fusion, data_loader_val, device):
    ids, modality = [], []
    pred_is_ischemia = []
    pred_xin_jian, pred_qian_bi, pred_jiange_bi, pred_xia_bi, pred_ce_bi = [], [], [], [], []
    pred_LCX, pred_RCA, pred_LAD = [], [], []
    
    gt_is_ischemia = []
    gt_xin_jian, gt_qian_bi, gt_jiange_bi, gt_xia_bi, gt_ce_bi = [], [], [], [], []
    gt_LCX, gt_RCA, gt_LAD = [], [], []
    model_spect.eval()
    model_cta.eval()
    model_ischemia.eval()
    model_fusion.eval()
    with torch.no_grad():
        for index, input in enumerate(data_loader_val):
            # assemble
            for k in input.keys():
                if (k != 'id') and  (k != 'modality') :
                    input[k] = input[k].to(device)
        
            # forward
            feat_spect = model_spect(input['QRS'], input['T'], return_feat=True)
            feat_cta = model_cta(input['QRS'], input['T'], return_feat=True)
            feat_ischemia = model_ischemia(input['QRS'], input['T'], return_feat=True)
            output = model_fusion(feat_spect, feat_cta, feat_ischemia)
        
            ids += list(input['id'])
            modality += list(input['modality'])
            
            # loss
            pred_is_ischemia += list(torch.sigmoid(output['is_ischemia']).squeeze().detach().cpu().numpy())
            gt_is_ischemia += list(input['is_ischemia'].squeeze().detach().cpu().numpy())
            
            if data_loader_val.dataset.type == 'SPECT':
                pred_xin_jian += list(torch.sigmoid(output['output_spect']['xin_jian']).squeeze().detach().cpu().numpy())
                pred_qian_bi += list(torch.sigmoid(output['output_spect']['qian_bi']).squeeze().detach().cpu().numpy())
                pred_jiange_bi += list(torch.sigmoid(output['output_spect']['jiange_bi']).squeeze().detach().cpu().numpy())
                pred_xia_bi += list(torch.sigmoid(output['output_spect']['xia_bi']).squeeze().detach().cpu().numpy())
                pred_ce_bi += list(torch.sigmoid(output['output_spect']['ce_bi']).squeeze().detach().cpu().numpy())
                
                gt_xin_jian += list(input['xin_jian'].squeeze().detach().cpu().numpy())
                gt_qian_bi += list(input['qian_bi'].squeeze().detach().cpu().numpy())
                gt_jiange_bi += list(input['jiange_bi'].squeeze().detach().cpu().numpy())
                gt_xia_bi += list(input['xia_bi'].squeeze().detach().cpu().numpy())
                gt_ce_bi += list(input['ce_bi'].squeeze().detach().cpu().numpy())  
                
            if data_loader_val.dataset.type == 'CAG':      
                pred_LAD += list(torch.sigmoid(output['output_cta']['LAD']).squeeze().detach().cpu().numpy())    
                pred_RCA += list(torch.sigmoid(output['output_cta']['RCA']).squeeze().detach().cpu().numpy())  
                pred_LCX += list(torch.sigmoid(output['output_cta']['LCX']).squeeze().detach().cpu().numpy())  
                
                gt_LAD += list(input['LAD'].squeeze().detach().cpu().numpy())    
                gt_RCA += list(input['RCA'].squeeze().detach().cpu().numpy())  
                gt_LCX += list(input['LCX'].squeeze().detach().cpu().numpy())
                

        if data_loader_val.dataset.type == 'NI':  
            val_results = {'pred_is_ischemia':pred_is_ischemia, 'gt_is_ischemia':gt_is_ischemia, 'ids':ids, 'modality':modality}
        
        
        if data_loader_val.dataset.type == 'SPECT':  
            val_results = {'pred_is_ischemia':pred_is_ischemia, 'gt_is_ischemia':gt_is_ischemia,
                        'pred_xin_jian':pred_xin_jian, 'gt_xin_jian':gt_xin_jian,
                        'pred_qian_bi':pred_qian_bi, 'gt_qian_bi':gt_qian_bi,
                        'pred_jiange_bi':pred_jiange_bi, 'gt_jiange_bi':gt_jiange_bi,
                        'pred_xia_bi':pred_xia_bi, 'gt_xia_bi':gt_xia_bi,
                        'pred_ce_bi':pred_ce_bi, 'gt_ce_bi':gt_ce_bi, 'ids':ids,'modality':modality
                        }
        if data_loader_val.dataset.type == 'CAG':  
            val_results = {'pred_is_ischemia':pred_is_ischemia, 'gt_is_ischemia':gt_is_ischemia,
                        'pred_LAD':pred_LAD, 'gt_LAD':gt_LAD,
                        'pred_RCA':pred_RCA, 'gt_RCA':gt_RCA,
                        'pred_LCX':pred_LCX, 'gt_LCX':gt_LCX, 'ids':ids,'modality':modality
                        }

            
    return val_results
            
            
def inference(model_spect, model_cta, model_ischemia, model_fusion, data_loader_val_ni, data_loader_val_spect, data_loader_val_cta, device, fold):

    print('start validation NI------', len(data_loader_val_ni))
    val_results_NI = evaluate_loop(model_spect, model_cta, model_ischemia, model_fusion, data_loader_val_ni, device)
    print('start validation SPECT------', len(data_loader_val_spect))
    val_results_SPECT = evaluate_loop(model_spect, model_cta, model_ischemia, model_fusion, data_loader_val_spect, device)
    print('start validation CTA------', len(data_loader_val_cta))
    val_results_CTA = evaluate_loop(model_spect, model_cta, model_ischemia, model_fusion, data_loader_val_cta, device)
    
    pred_ischemia = np.array(val_results_NI['pred_is_ischemia'] + val_results_SPECT['pred_is_ischemia'] + val_results_CTA['pred_is_ischemia'])
    gt_ischemia = np.array(val_results_NI['gt_is_ischemia'] + val_results_SPECT['gt_is_ischemia'] + val_results_CTA['gt_is_ischemia'])
    ids = val_results_NI['ids'] + val_results_SPECT['ids'] + val_results_CTA['ids']
    modality = val_results_NI['modality'] + val_results_SPECT['modality'] + val_results_CTA['modality']
    is_ischemia_results = cal_metrics(pred_ischemia.copy(), gt_ischemia.copy())
    
    for i in range(len(pred_ischemia)):
        # if (gt_ischemia[i] == 0) and (pred_ischemia[i]  > is_ischemia_results['opt_point']):
        #     if modality[i] == '自然人':
        #         pass
        #         # print(ids[i], 'pred:', pred_ischemia[i], 'gt:', gt_ischemia[i])
        if (gt_ischemia[i] == 1) and (pred_ischemia[i] <= is_ischemia_results['opt_point']):
            if modality[i] == 'SPECT':
                print(ids[i], modality[i], 'pred:', pred_ischemia[i], 'gt:', gt_ischemia[i])
        if (gt_ischemia[i] == 0) and (pred_ischemia[i] > is_ischemia_results['opt_point']):
            if modality[i] == 'SPECT':
                print(ids[i], modality[i], 'pred:', pred_ischemia[i], 'gt:', gt_ischemia[i])
        # if (gt_ischemia[i] == 1) and (pred_ischemia[i] <= is_ischemia_results['opt_point']):
        #     if (modality[i] == 'CTA') or (modality[i] == '造影'):
        #         # if ids[i] in case:
        #             print(ids[i], modality[i], 'pred:', pred_ischemia[i], 'gt:', gt_ischemia[i])
        # if (gt_ischemia[i] == 0) and (pred_ischemia[i] > is_ischemia_results['opt_point']):
        #     if (modality[i] == 'CTA') or (modality[i] == '造影'):
                # if ids[i] in case:
                    # print(ids[i], modality[i], 'pred:', pred_ischemia[i], 'gt:', gt_ischemia[i])
        
    # for i in range(len(pred_ischemia)):
    #     if (modality[i] == 'CTA'):
    #             print(ids[i], modality[i], 'pred:', pred_ischemia[i], 'gt:', gt_ischemia[i])

    
    xin_jian_results = cal_metrics(np.array(val_results_SPECT['pred_xin_jian']).copy(), np.array(val_results_SPECT['gt_xin_jian']).copy())
    qian_bi_results = cal_metrics(np.array(val_results_SPECT['pred_qian_bi']).copy(), np.array(val_results_SPECT['gt_qian_bi']).copy())
    jiange_bi_results = cal_metrics(np.array(val_results_SPECT['pred_jiange_bi']).copy(), np.array(val_results_SPECT['gt_jiange_bi']).copy())
    xia_bi_results = cal_metrics(np.array(val_results_SPECT['pred_xia_bi']).copy(), np.array(val_results_SPECT['gt_xia_bi']).copy())
    ce_bi_results = cal_metrics(np.array(val_results_SPECT['pred_ce_bi']).copy(), np.array(val_results_SPECT['gt_ce_bi']).copy())
    
    LAD_results = cal_metrics(np.array(val_results_CTA['pred_LAD']).copy(), np.array(val_results_CTA['gt_LAD']).copy())
    RCA_results = cal_metrics(np.array(val_results_CTA['pred_RCA']).copy(), np.array(val_results_CTA['gt_RCA']).copy())
    LCX_results = cal_metrics(np.array(val_results_CTA['pred_LCX']).copy(), np.array(val_results_CTA['gt_LCX']).copy())
    
    stats_overall = {'is_ischemia_results':is_ischemia_results, 'xin_jian_results':xin_jian_results,
               'qian_bi_results':qian_bi_results, 'jiange_bi_results':jiange_bi_results,
               'xia_bi_results':xia_bi_results, 'xia_bi_results' :xia_bi_results,'ce_bi_results':ce_bi_results,
               'LAD_results':LAD_results, 'RCA_results':RCA_results, 'LCX_results':LCX_results
               }  
    results_overall = {'stats':stats_overall, 'pred_ischemia':pred_ischemia, 'gt_ischemia':gt_ischemia,
                    'pred_xin_jian':np.array(val_results_SPECT['pred_xin_jian']), 'gt_xin_jian':np.array(val_results_SPECT['gt_xin_jian']),
                    'pred_qian_bi':np.array(val_results_SPECT['pred_qian_bi']), 'gt_qian_bi':np.array(val_results_SPECT['gt_qian_bi']),
                    'pred_jiange_bi':np.array(val_results_SPECT['pred_jiange_bi']), 'gt_jiange_bi':np.array(val_results_SPECT['gt_jiange_bi']),
                    'pred_xia_bi':np.array(val_results_SPECT['pred_xia_bi']), 'gt_xia_bi':np.array(val_results_SPECT['gt_xia_bi']),
                    'pred_ce_bi':np.array(val_results_SPECT['pred_ce_bi']), 'gt_ce_bi':np.array(val_results_SPECT['gt_ce_bi']),
                    'pred_LAD':np.array(val_results_CTA['pred_LAD']), 'gt_LAD':np.array(val_results_CTA['gt_LAD']),
                    'pred_RCA':np.array(val_results_CTA['pred_RCA']), 'gt_RCA':np.array(val_results_CTA['gt_RCA']),
                    'pred_LCX':np.array(val_results_CTA['pred_LCX']), 'gt_LCX':np.array(val_results_CTA['gt_LCX']),
                    'ids':ids}
    print('------------Overall------------------')
    print(stats_overall)
    print('------------Group------------------')
    tags = ['造影', '自然人', 'SPECT']
    results_group = dict()
    for t in tags:
        pred, gt = [], []
        for ind, mod in enumerate(modality):
            if mod == t:
                pred.append(pred_ischemia[ind])
                gt.append(gt_ischemia[ind])
        group_result = cal_metrics(np.array(pred).copy(), np.array(gt).copy(), is_ischemia_results['opt_point'])
        print(t, group_result)
        results_group[t] = {'pred':pred, 'gt':gt, 'stats':group_result}
 
    output = {'results_group':results_group, 'results_overall':results_overall}
    with open(os.path.join('Results/predictions', 'results'+str(fold)+'.pickle'), 'wb') as file:
        pickle.dump(output, file)
    file.close()
    
    return
    
    
 
