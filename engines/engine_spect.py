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
from losses.loss import Loss
from skimage.metrics import structural_similarity as ssim
from pathlib import Path
import time
import sklearn as sk
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score

def cal_metrics(pred, gt):
    pred[pred>=0.5] = 1
    pred[pred<0.5] = 0    
    cm = confusion_matrix(gt, pred, labels=[0, 1])
    total=sum(sum(cm))
    acc=(cm[0,0]+cm[1,1])/total
    sens = cm[0,0]/(cm[0,0]+cm[0,1])
    spec = cm[1,1]/(cm[1,0]+cm[1,1])
    try:
        auc = roc_auc_score(gt, pred, labels=[0, 1])
    except:
        auc = -1
    results = {'acc':acc, 'sens':sens, 'spec':spec, 'auc':auc}
    return results


def print_results(output, input):
    #----------------------------------------------------------------------------------------------------------
    # is_ischemia
    is_ischemia_results = cal_metrics(torch.sigmoid(output['is_ischemia']).squeeze().detach().cpu().numpy(), 
                                        input['is_ischemia'].squeeze().detach().cpu().numpy())
    # spect
    idx = torch.where(input['SPECT_idx']==1)[0]
    xin_jian_results = cal_metrics(torch.sigmoid(output['output_spect']['xin_jian'])[idx].squeeze().detach().cpu().numpy(), 
                                    input['xin_jian'][idx].squeeze().detach().cpu().numpy())
    qian_bi_results = cal_metrics(torch.sigmoid(output['output_spect']['qian_bi'])[idx].squeeze().detach().cpu().numpy(), 
                                    input['qian_bi'][idx].squeeze().detach().cpu().numpy())
    jiange_bi_results = cal_metrics(torch.sigmoid(output['output_spect']['jiange_bi'])[idx].squeeze().detach().cpu().numpy(), 
                                    input['jiange_bi'][idx].squeeze().detach().cpu().numpy())
    xia_bi_results = cal_metrics(torch.sigmoid(output['output_spect']['xia_bi'])[idx].squeeze().detach().cpu().numpy(), 
                                    input['xia_bi'][idx].squeeze().detach().cpu().numpy())
    ce_bi_results = cal_metrics(torch.sigmoid(output['output_spect']['ce_bi'])[idx].squeeze().detach().cpu().numpy(), 
                                    input['ce_bi'][idx].squeeze().detach().cpu().numpy())
    #cta
    idx = torch.where(input['CTA_idx']==1)[0]
    LAD_results = cal_metrics(torch.sigmoid(output['output_cta']['LAD'])[idx].squeeze().detach().cpu().numpy(), 
                                    input['LAD'][idx].squeeze().detach().cpu().numpy())
    LCX_results = cal_metrics(torch.sigmoid(output['output_cta']['LCX'])[idx].squeeze().detach().cpu().numpy(), 
                                    input['LCX'][idx].squeeze().detach().cpu().numpy())
    RCA_results = cal_metrics(torch.sigmoid(output['output_cta']['RCA'])[idx].squeeze().detach().cpu().numpy(), 
                                    input['RCA'][idx].squeeze().detach().cpu().numpy())

    # print('is_ischemia_results:', is_ischemia_results)
    # print('spect-xin_jian_results:', xin_jian_results)
    # print('spect-qian_bi_results:', qian_bi_results)
    # print('spect-jiange_bi_results:', jiange_bi_results)
    # print('spect-xia_bi_results:', xia_bi_results)
    # print('spect-ce_bi_results:', ce_bi_results)
    # print('cta-LAD_results:', LAD_results)
    # print('cta-LCX_results:', LCX_results)
    # print('cta-RCA_results:', RCA_results)
    #----------------------------------------------------------------------------------------------------------
    results = {'is_ischemia': is_ischemia_results, 'xin_jian': xin_jian_results,
               'qian_bi':qian_bi_results, 'jiange_bi':jiange_bi_results,
               'xia_bi_results': xia_bi_results, 'ce_bi':ce_bi_results,
               'LAD':LAD_results, 'LCX':LCX_results,'RCA': RCA_results  
               }
    return results
    

def train_one_epoch(model, data_loader, optimizer, device, epoch, log_path, output_dir, lr_scheduler, steps, max_norm,
                    data_loader_val_ni, data_loader_val_spect, data_loader_val_cta, val_log_dir, benchmark_metric, iteration):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('acc', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('sens', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 1
    lsses = []
    train_loss = Loss()
    for input in metric_logger.log_every(data_loader, print_freq, header):              
        for k in input.keys():
            if k != 'id':
                input[k] = input[k].squeeze(0).to(device)
        # forward
        output = model(input['QRS'], input['T'])

        # loss
        loss_dict = train_loss.forward_spect(output, input)

        losses = sum(loss_dict[k] for k in loss_dict.keys())
        loss_value = losses.item()

        if not math.isfinite(loss_value):
            
            output = model(input['QRS'], input['T'])
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict)
            checkpoint_path = output_dir / 'error.pth'
            utils.save_on_master({
                'model': model.module.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'iteration': iteration,
                'input':input,
            }, checkpoint_path)

            sys.exit(1)

        optimizer.zero_grad()
        with amp.scale_loss(losses, optimizer) as scaled_loss:
            scaled_loss.backward()#------------------------

        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)


        # print training metrics
        results = print_results(output, input)
        

        optimizer.step()
        lr_scheduler.step_update(iteration)

        metric_logger.update(loss=loss_value, **loss_dict)
        metric_logger.update(acc=results['is_ischemia']['acc'])
        metric_logger.update(sens=results['is_ischemia']['sens'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        # log
        lss = loss_dict.copy()
        for i, k in enumerate(lss):
            lss[k] = lss[k].detach().cpu().numpy().tolist()
        lss['iteration'] = iteration
        lss['epoch'] = epoch
        lss['results'] = [results]
        lsses.append(lss)
        with open(os.path.join(log_path, str(iteration)+'.pickle'), 'wb') as file:
            pickle.dump(lsses, file)
        file.close()

        
        # ex
        if iteration%20 == 0:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            pred_best = evaluate(model, data_loader_val_ni, data_loader_val_spect, data_loader_val_cta, device, iteration, val_log_dir)
            if pred_best >= benchmark_metric:
                checkpoint_paths = [output_dir / f'checkpoint_{pred_best:04}.pth']
                print('saving best@', pred_best, 'iteration:', iteration)
                benchmark_metric = pred_best

            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model.module.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'iteration': iteration,
                }, checkpoint_path)

        iteration+=1
        steps += 1

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    print('benchmark_metric:',benchmark_metric)
    return benchmark_metric, iteration

def evaluate(model, data_loader_val_ni, data_loader_val_spect, data_loader_val_cta, device, iteration, output_dir):

    print('start validation NI------', len(data_loader_val_ni))
    val_results_NI = evaluate_loop(model, data_loader_val_ni, device)
    print('start validation SPECT------', len(data_loader_val_spect))
    val_results_SPECT = evaluate_loop(model, data_loader_val_spect, device)
    print('start validation CTA------', len(data_loader_val_cta))
    val_results_CTA = evaluate_loop(model, data_loader_val_cta, device)
    
    pred_ischemia = np.array(val_results_NI['pred_is_ischemia'] + val_results_SPECT['pred_is_ischemia'] + val_results_CTA['pred_is_ischemia'])
    gt_ischemia = np.array(val_results_NI['gt_is_ischemia'] + val_results_SPECT['gt_is_ischemia'] + val_results_CTA['gt_is_ischemia'])
    
    try:
        is_ischemia_results = cal_metrics(pred_ischemia, gt_ischemia)
    except:
        print()
    xin_jian_results = cal_metrics(np.array(val_results_SPECT['pred_xin_jian']), np.array(val_results_SPECT['gt_xin_jian']))
    qian_bi_results = cal_metrics(np.array(val_results_SPECT['pred_qian_bi']), np.array(val_results_SPECT['gt_qian_bi']))
    jiange_bi_results = cal_metrics(np.array(val_results_SPECT['pred_jiange_bi']), np.array(val_results_SPECT['gt_jiange_bi']))
    xia_bi_results = cal_metrics(np.array(val_results_SPECT['pred_xia_bi']), np.array(val_results_SPECT['gt_xia_bi']))
    ce_bi_results = cal_metrics(np.array(val_results_SPECT['pred_ce_bi']), np.array(val_results_SPECT['gt_ce_bi']))
    
    LAD_results = cal_metrics(np.array(val_results_CTA['pred_LAD']), np.array(val_results_CTA['gt_LAD']))
    RCA_results = cal_metrics(np.array(val_results_CTA['pred_RCA']), np.array(val_results_CTA['gt_RCA']))
    LCX_results = cal_metrics(np.array(val_results_CTA['pred_LCX']), np.array(val_results_CTA['gt_LCX']))
    
    results = {'is_ischemia_results':is_ischemia_results, 'xin_jian_results':xin_jian_results,
               'qian_bi_results':qian_bi_results, 'jiange_bi_results':jiange_bi_results,
               'xia_bi_results':xia_bi_results, 'xia_bi_results' :xia_bi_results,'ce_bi_results':ce_bi_results,
               'LAD_results':LAD_results, 'RCA_results':RCA_results, 'LCX_results':LCX_results
               }              

    print('------------Valid------------------')
    print(is_ischemia_results)
    print('------------Valid------------------')

    with open(os.path.join(output_dir, str(iteration) + '_valid.pickle'), 'wb') as file:
        pickle.dump(results, file)
    file.close()
    
    # eval_metric = is_ischemia_results['acc']
    eval_metric = is_ischemia_results['jiange_bi_results']
    return eval_metric

def evaluate_loop(model, data_loader_val, device):
    train_loss = Loss()
    ids = []
    pred_is_ischemia = []
    pred_xin_jian, pred_qian_bi, pred_jiange_bi, pred_xia_bi, pred_ce_bi = [], [], [], [], []
    pred_LCX, pred_RCA, pred_LAD = [], [], []
    
    gt_is_ischemia = []
    gt_xin_jian, gt_qian_bi, gt_jiange_bi, gt_xia_bi, gt_ce_bi = [], [], [], [], []
    gt_LCX, gt_RCA, gt_LAD = [], [], []
    model.eval()
    with torch.no_grad():
        for index, input in enumerate(data_loader_val):
            # assemble
            for k in input.keys():
                if k != 'id':
                    input[k] = input[k].to(device)
        
            # forward
            output = model(input['QRS'].to(device), input['T'].to(device))
            ids += list(input['id'])
            


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
                
            if data_loader_val.dataset.type == 'CTA':      
                pred_LAD += list(torch.sigmoid(output['output_cta']['LAD']).squeeze().detach().cpu().numpy())    
                pred_RCA += list(torch.sigmoid(output['output_cta']['RCA']).squeeze().detach().cpu().numpy())  
                pred_LCX += list(torch.sigmoid(output['output_cta']['LCX']).squeeze().detach().cpu().numpy())  
                
                gt_LAD += list(input['LAD'].squeeze().detach().cpu().numpy())    
                gt_RCA += list(input['RCA'].squeeze().detach().cpu().numpy())  
                gt_LCX += list(input['LCX'].squeeze().detach().cpu().numpy())
                
            if (data_loader_val.dataset.type == 'SPECT_NI') or (data_loader_val.dataset.type == 'CTA_NI'): 
                pred_xin_jian += list(torch.sigmoid(output['output_spect']['xin_jian']).squeeze().detach().cpu().numpy())
                pred_qian_bi += list(torch.sigmoid(output['output_spect']['qian_bi']).squeeze().detach().cpu().numpy())
                pred_jiange_bi += list(torch.sigmoid(output['output_spect']['jiange_bi']).squeeze().detach().cpu().numpy())
                pred_xia_bi += list(torch.sigmoid(output['output_spect']['xia_bi']).squeeze().detach().cpu().numpy())
                pred_ce_bi += list(torch.sigmoid(output['output_spect']['ce_bi']).squeeze().detach().cpu().numpy())
                
                pred_LAD += list(torch.sigmoid(output['output_cta']['LAD']).squeeze().detach().cpu().numpy())    
                pred_RCA += list(torch.sigmoid(output['output_cta']['RCA']).squeeze().detach().cpu().numpy())  
                pred_LCX += list(torch.sigmoid(output['output_cta']['LCX']).squeeze().detach().cpu().numpy()) 
                
                gt_xin_jian += list(input['xin_jian'].squeeze().detach().cpu().numpy())
                gt_qian_bi += list(input['qian_bi'].squeeze().detach().cpu().numpy())
                gt_jiange_bi += list(input['jiange_bi'].squeeze().detach().cpu().numpy())
                gt_xia_bi += list(input['xia_bi'].squeeze().detach().cpu().numpy())
                gt_ce_bi += list(input['ce_bi'].squeeze().detach().cpu().numpy())  
                
                gt_LAD += list(input['LAD'].squeeze().detach().cpu().numpy())    
                gt_RCA += list(input['RCA'].squeeze().detach().cpu().numpy())  
                gt_LCX += list(input['LCX'].squeeze().detach().cpu().numpy())
                
                
        if data_loader_val.dataset.type == 'NI':  
            val_results = {'pred_is_ischemia':pred_is_ischemia, 'gt_is_ischemia':gt_is_ischemia, 'ids':ids}
        
        
        
        if data_loader_val.dataset.type == 'SPECT':  
            val_results = {'pred_is_ischemia':pred_is_ischemia, 'gt_is_ischemia':gt_is_ischemia,
                        'pred_xin_jian':pred_xin_jian, 'gt_xin_jian':gt_xin_jian,
                        'pred_qian_bi':pred_qian_bi, 'gt_qian_bi':gt_qian_bi,
                        'pred_jiange_bi':pred_jiange_bi, 'gt_jiange_bi':gt_jiange_bi,
                        'pred_xia_bi':pred_xia_bi, 'gt_xia_bi':gt_xia_bi,
                        'pred_ce_bi':pred_ce_bi, 'gt_ce_bi':gt_ce_bi, 'ids':ids
                        }
        if data_loader_val.dataset.type == 'CTA':  
            val_results = {'pred_is_ischemia':pred_is_ischemia, 'gt_is_ischemia':gt_is_ischemia,
                        'pred_LAD':pred_LAD, 'gt_LAD':gt_LAD,
                        'pred_RCA':pred_RCA, 'gt_RCA':gt_RCA,
                        'pred_LCX':pred_LCX, 'gt_LCX':gt_LCX, 'ids':ids
                        }
            
        if (data_loader_val.dataset.type == 'SPECT_NI') or (data_loader_val.dataset.type == 'CTA_NI'):  
                        val_results = {'pred_is_ischemia':pred_is_ischemia, 'gt_is_ischemia':gt_is_ischemia,
                        'pred_xin_jian':pred_xin_jian, 'gt_xin_jian':gt_xin_jian,
                        'pred_qian_bi':pred_qian_bi, 'gt_qian_bi':gt_qian_bi,
                        'pred_jiange_bi':pred_jiange_bi, 'gt_jiange_bi':gt_jiange_bi,
                        'pred_xia_bi':pred_xia_bi, 'gt_xia_bi':gt_xia_bi,
                        'pred_ce_bi':pred_ce_bi, 'gt_ce_bi':gt_ce_bi, 
                        'pred_LAD':pred_LAD, 'gt_LAD':gt_LAD,
                        'pred_RCA':pred_RCA, 'gt_RCA':gt_RCA,
                        'pred_LCX':pred_LCX, 'gt_LCX':gt_LCX,
                        'ids':ids
                        }
            
    return val_results
            
            
def inference(model, data_loader_val_ni, data_loader_val_spect, data_loader_val_cta, device):

    print('start validation NI------', len(data_loader_val_ni))
    val_results_NI = evaluate_loop(model, data_loader_val_ni, device)
    print('start validation SPECT------', len(data_loader_val_spect))
    val_results_SPECT = evaluate_loop(model, data_loader_val_spect, device)
    print('start validation CTA------', len(data_loader_val_cta))
    val_results_CTA = evaluate_loop(model, data_loader_val_cta, device)
    
    pred_ischemia = np.array(val_results_NI['pred_is_ischemia'] + val_results_SPECT['pred_is_ischemia'] + val_results_CTA['pred_is_ischemia'])
    gt_ischemia = np.array(val_results_NI['gt_is_ischemia'] + val_results_SPECT['gt_is_ischemia'] + val_results_CTA['gt_is_ischemia'])
    ids = val_results_NI['ids'] + val_results_SPECT['ids'] + val_results_CTA['ids']
    ischemia_predictions = val_results_NI['pred_is_ischemia'] + val_results_SPECT['pred_is_ischemia'] + val_results_CTA['pred_is_ischemia']
    
       
    is_ischemia_results = cal_metrics(pred_ischemia, gt_ischemia)
    xin_jian_results = cal_metrics(np.array(val_results_SPECT['pred_xin_jian']), np.array(val_results_SPECT['gt_xin_jian']))
    qian_bi_results = cal_metrics(np.array(val_results_SPECT['pred_qian_bi']), np.array(val_results_SPECT['gt_qian_bi']))
    jiange_bi_results = cal_metrics(np.array(val_results_SPECT['pred_jiange_bi']), np.array(val_results_SPECT['gt_jiange_bi']))
    xia_bi_results = cal_metrics(np.array(val_results_SPECT['pred_xia_bi']), np.array(val_results_SPECT['gt_xia_bi']))
    ce_bi_results = cal_metrics(np.array(val_results_SPECT['pred_ce_bi']), np.array(val_results_SPECT['gt_ce_bi']))
    
    results = {'xin_jian_results':xin_jian_results,'qian_bi_results':qian_bi_results, 'jiange_bi_results':jiange_bi_results,
                'xia_bi_results' :xia_bi_results,'ce_bi_results':ce_bi_results, 
               }   
               
    print('------------Valid------------------')
    print(results)

    print('------------Saving------------------')    
    with open(os.path.join('/media/cygzz/data/rtao/projects/MCG-NC/pred/f0_ischemia_prediction.pickle'), 'wb') as file:
        pickle.dump(results, file)
    file.close()
    return results            


def load_ischemia_results(ids, pred_ischemia='/media/cygzz/data/rtao/projects/MCG-NC/pred/f0_ischemia_prediction.pickle'):
    with open(pred_ischemia, 'rb') as f:  # run this to load from pickle
        is_ischemia = pickle.load(f)
    f.close()
    
    all_ids = is_ischemia['ids']
    all_pred = is_ischemia['ischemia_predictions']
    all_dict = dict()
    for i, id in enumerate(all_ids):
        all_dict[id] = all_pred[i]
        
    pred_ischemia = []
    for id in ids:
        pred_ischemia.append(all_dict[id])
        

    return np.array(pred_ischemia)            
            
def inference_group(model, data_loader_val_spect, dataset_val_spect_ni, data_loader_val_cta, dataset_val_cta_ni, device):

    print('start validation NI from SPECT ------', len(dataset_val_spect_ni))
    val_results_NI_SPECT = evaluate_loop(model, dataset_val_spect_ni, device)
    print('start validation NI from CTA ------', len(dataset_val_cta_ni))
    val_results_NI_CTA = evaluate_loop(model, dataset_val_cta_ni, device)
    print('start validation SPECT------', len(data_loader_val_spect))
    val_results_SPECT = evaluate_loop(model, data_loader_val_spect, device)
    print('start validation CTA------', len(data_loader_val_cta))
    val_results_CTA = evaluate_loop(model, data_loader_val_cta, device)         
    
    
    pred_ischemia = load_ischemia_results(ids = val_results_NI_SPECT['ids'] + val_results_SPECT['ids'])
    # spect group
    # pred_ischemia = np.array(val_results_NI_SPECT['pred_is_ischemia'] + val_results_SPECT['pred_is_ischemia'])
    gt_ischemia = np.array(val_results_NI_SPECT['gt_is_ischemia'] + val_results_SPECT['gt_is_ischemia'])        
    is_ischemia_results_spect = cal_metrics(pred_ischemia, gt_ischemia)
    
    pred_xin_jian_spect = np.array(val_results_NI_SPECT['pred_xin_jian'] + val_results_SPECT['pred_xin_jian'])
    gt_xin_jian_spect = np.array(val_results_NI_SPECT['gt_xin_jian'] + val_results_SPECT['gt_xin_jian'])   
    pred_xin_jian_spect[pred_ischemia<0.5] = 0
    xin_jian_results = cal_metrics(pred_xin_jian_spect, gt_xin_jian_spect)
 
    pred_qian_bi_spect = np.array(val_results_NI_SPECT['pred_qian_bi'] + val_results_SPECT['pred_qian_bi'])
    gt_qian_bi_spect = np.array(val_results_NI_SPECT['gt_qian_bi'] + val_results_SPECT['gt_qian_bi'])   
    pred_qian_bi_spect[pred_ischemia<0.5] = 0
    qian_bi_results = cal_metrics(pred_qian_bi_spect, gt_qian_bi_spect)

    pred_jiange_bi_spect = np.array(val_results_NI_SPECT['pred_jiange_bi'] + val_results_SPECT['pred_jiange_bi'])
    gt_jiange_bi_spect = np.array(val_results_NI_SPECT['gt_jiange_bi'] + val_results_SPECT['gt_jiange_bi'])   
    pred_jiange_bi_spect[pred_ischemia<0.5] = 0
    jiange_bi_results = cal_metrics(pred_jiange_bi_spect, gt_jiange_bi_spect)
    
    pred_xia_bi_spect = np.array(val_results_NI_SPECT['pred_xia_bi'] + val_results_SPECT['pred_xia_bi'])
    gt_xia_bi_spect = np.array(val_results_NI_SPECT['gt_xia_bi'] + val_results_SPECT['gt_xia_bi'])   
    pred_xia_bi_spect[pred_ischemia<0.5] = 0
    xia_bi_results = cal_metrics(pred_xia_bi_spect, gt_xia_bi_spect)
    
    pred_ce_bi_spect = np.array(val_results_NI_SPECT['pred_ce_bi'] + val_results_SPECT['pred_ce_bi'])
    gt_ce_bi_spect = np.array(val_results_NI_SPECT['gt_ce_bi'] + val_results_SPECT['gt_ce_bi'])   
    pred_ce_bi_spect[pred_ischemia<0.5] = 0
    ce_bi_results = cal_metrics(pred_ce_bi_spect, gt_ce_bi_spect)

    results_spect = {'is_ischemia_results':is_ischemia_results_spect, 'xin_jian_results':xin_jian_results,
               'qian_bi_results':qian_bi_results, 'jiange_bi_results':jiange_bi_results,
               'xia_bi_results':xia_bi_results, 'xia_bi_results' :xia_bi_results,'ce_bi_results':ce_bi_results}  
    
    print('------------Valid------------------')
    print(results_spect)
    print('------------Valid------------------')    
    
    pred_ischemia = load_ischemia_results(ids = val_results_NI_CTA['ids'] + val_results_CTA['ids'])
    # CTA group
    # pred_ischemia = np.array(val_results_NI_CTA['pred_is_ischemia'] + val_results_CTA['pred_is_ischemia'])
    gt_ischemia = np.array(val_results_NI_CTA['gt_is_ischemia'] + val_results_CTA['gt_is_ischemia'])        
    is_ischemia_results_cta = cal_metrics(pred_ischemia, gt_ischemia)
    
    pred_lad_cta = np.array(val_results_NI_CTA['pred_LAD'] + val_results_CTA['pred_LAD'])
    gt_lad_cta = np.array(val_results_NI_CTA['gt_LAD'] + val_results_CTA['gt_LAD'])   
    pred_lad_cta[pred_ischemia<0.5] = 0
    LAD_results = cal_metrics(pred_lad_cta, gt_lad_cta)
    
    pred_rca_cta = np.array(val_results_NI_CTA['pred_RCA'] + val_results_CTA['pred_RCA'])
    gt_rca_cta = np.array(val_results_NI_CTA['gt_RCA'] + val_results_CTA['gt_RCA'])   
    pred_rca_cta[pred_ischemia<0.5] = 0
    RCA_results = cal_metrics(pred_rca_cta, gt_rca_cta)

    pred_lcx_cta = np.array(val_results_NI_CTA['pred_LCX'] + val_results_CTA['pred_LCX'])
    gt_lcx_cta = np.array(val_results_NI_CTA['gt_LCX'] + val_results_CTA['gt_LCX'])   
    pred_lcx_cta[pred_ischemia<0.5] = 0
    LCX_results = cal_metrics(pred_lcx_cta, gt_lcx_cta)

    results_cta = {'is_ischemia_results':is_ischemia_results_cta, 'LAD_results':LAD_results,
               'RCA_results':RCA_results, 'LCX_results':LCX_results} 

    print('------------Valid------------------')
    print(results_cta)
    print('------------Valid------------------')      
        

    
    return results_spect, results_cta            
            
            
